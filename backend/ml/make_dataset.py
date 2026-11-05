"""Build the training corpus for the school parser model *from the database itself*.

Two sources of truth feed the generator, so nothing about the school is typed in by hand:
  * the semantic layer  - views, columns, column kinds, how people name them;
  * the live data       - every real student / teacher / class / subject / room / title / date,
                          read through the same ValueIndex the runtime uses.

Example types: single-turn questions, follow-ups that edit the previous query (entity swap, added
filter, pronouns, "how many?"), clarification replies, vague questions that need an `ask`, compound
questions, and refusals (out-of-scope, write requests, manual/how-to, unsupported data, small talk).
Every gold query is compiled against the live schema, so the corpus can never contain a query the
runtime would reject.

Usage:  python backend/ml/make_dataset.py [--n 24000] [--out backend/ml/data] [--seed 7]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "ml"))

import lang  # noqa: E402
from gold_bank import BANK, MANUAL_Q, OOS_Q, HELP_Q, GREET_Q, THANKS_Q, WRITE_Q, UNSUPPORTED_CONCEPTS  # noqa: E402
from app.agent.dsl import (  # noqa: E402
    Clarify, Cond, Control, MULTI, Query, SNAP_KINDS, ValueIndex, compile_query, format_items, load_value_index, parse,
)
from app.agent.normalize import normalize_question  # noqa: E402
from app.agent.semantic_layer import VIEW_BY_NAME, VIEWS, View  # noqa: E402

Items = List[object]
Example = Tuple[str, Optional[str], str]  # (question, ctx or None, target)

LIST_FRAMES = [
    "show {adj} {N} {F}", "list all {adj} {N} {F}", "give me {adj} {N} {F}", "what are the {adj} {N} {F}", "display {adj} {N} {F}",
    "can you show me {adj} {N} {F}", "i would like to see {adj} {N} {F}", "{adj} {N} {F}", "tell me about {adj} {N} {F}", "get {adj} {N} {F}",
    "please list {adj} {N} {F}", "find {adj} {N} {F}", "what {adj} {N} are there {F}", "show me the {adj} {N} {F}", "i need the {adj} {N} {F}",
]
WHO_FRAMES = ["who are the {adj} {N} {F}", "which {adj} {N} {F}", "who are all the {adj} {N} {F}", "name the {adj} {N} {F}"]
COUNT_FRAMES = ["how many {adj} {N} {F}", "how many {adj} {N} are there {F}", "count the {adj} {N} {F}", "number of {adj} {N} {F}",
                "what is the total number of {adj} {N} {F}", "how many {adj} {N} do we have {F}"]
ATTR_FRAMES = ["what is the {P} of {E}", "{E}'s {P}", "tell me the {P} for {E}", "show {P} of {E}", "{P} of {E}", "get me {E}'s {P}",
               "what's {E}'s {P}", "can you tell me the {P} of {E}", "what is {E}'s {P}"]
ENTITY_KINDS = ("student", "faculty", "person", "section", "subject", "room", "title")
PEOPLE_KINDS = ("student", "faculty", "person")
PRONOUNS = ["his", "her", "their"]


def clean(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def Q(view: str, sel: Sequence[str] = (), w: Sequence[Tuple[str, str, str]] = (), g: Optional[str] = None,
      a: Optional[Tuple[str, str]] = None, o: Optional[Tuple[str, str]] = None, n: Optional[int] = None) -> Query:
    return Query(view, list(sel), [Cond(*c) for c in w], g, a, o, n)


class Gen:
    def __init__(self, vi: ValueIndex, ay: Tuple[dt.date, dt.date], seed: int):
        self.vi, self.ay = vi, ay
        self.rng = random.Random(seed)
        self.dates = self._dates()

    # ---- data helpers -------------------------------------------------------
    def dom(self, view: str, col: str) -> List[str]:
        return self.vi.domain(view, col)

    def _dates(self) -> List[dt.date]:
        found = set()
        for (vn, cn), vals in self.vi.values.items():
            if VIEW_BY_NAME[vn].col(cn).kind == "date":
                for v in vals:
                    try:
                        found.add(dt.date.fromisoformat(v[:10]))
                    except ValueError:
                        pass
        extra = [self.ay[0] + dt.timedelta(days=i * 17) for i in range(1, 12)]  # unseen days -> "no data" training
        return sorted(found) + extra

    def fcols(self, view: View) -> List[str]:
        out = []
        for c in view.cols:
            if c.kind == "date" and self.dates:
                out.append(c.name)
            elif c.kind in SNAP_KINDS and (self.dom(view.name, c.name) or (view.name, c.name) in MULTI):
                out.append(c.name)
        return out

    def noun(self, view: View) -> str:
        return self.rng.choice(view.nouns)

    # ---- value verbalisation ---------------------------------------------------
    def pick_cond(self, view: View, exclude: Sequence[str] = (), cols: Optional[Sequence[str]] = None):
        cands = [c for c in (cols or self.fcols(view)) if c not in exclude]
        if not cands:
            return None
        col = self.rng.choice(cands)
        spec = view.col(col)
        rng = self.rng
        if spec.kind == "date":
            d = rng.choice(self.dates)
            surf, iso = lang.surface_date(d, rng)
            return Cond(col, "eq", iso), {"surface": surf if rng.random() < 0.12 else iso, "date": True}
        if (view.name, col) in MULTI:
            pool = self.vi.kind_domain(spec.kind)
            if not pool:
                return None
            val = rng.choice(pool)
            return Cond(col, "has", val), {"surface": val}
        dom = self.dom(view.name, col)
        val = rng.choice(dom)
        return Cond(col, "eq", val), {"surface": val}

    def verb(self, view: View, cond: Cond, info: Dict) -> Dict[str, Optional[str]]:
        """-> {adj, suffix, entity}: how this filter reads inside a sentence."""
        rng = self.rng
        spec = view.col(cond.col)
        kind, val = spec.kind, cond.value
        res: Dict[str, Optional[str]] = {"adj": None, "suffix": None, "entity": None}
        if info.get("date"):
            d = info["surface"]
            res["suffix"] = rng.choice([f"on {d}", f"for {d}", f"dated {d}", f"of {d}"])
            return res
        if cond.op == "has":
            s = lang.surface_subject(val, rng) if kind == "subject" else lang.surface_section(val, rng)
            res["suffix"] = rng.choice([f"for {s}", f"of {s}"]) if (view.name, cond.col) != ("sv_faculty", "subjects_taught") else rng.choice([f"who teach {s}", f"teaching {s}"])
            if (view.name, cond.col) == ("sv_faculty", "class_teacher_of"):
                res["suffix"] = f"who is class teacher of {s}"
            if (view.name, cond.col) == ("sv_subjects", "teachers"):
                res["suffix"] = f"taught by {lang.surface_person(val, rng)}"
            return res
        if kind in PEOPLE_KINDS:
            s = lang.surface_person(val, rng)
            res["entity"] = s
            if kind == "faculty":
                res["suffix"] = rng.choice([f"by {s}", f"taught by {s}", f"of {s}", f"for {s}", f"assigned by {s}"])
            elif kind == "person":
                res["suffix"] = rng.choice([f"of {s}", f"from {s}", f"for {s}"])
            else:
                res["suffix"] = rng.choice([f"of {s}", f"for {s}", f"belonging to {s}", f"of student {s}"])
            if rng.random() < 0.3:
                res["adj"], res["suffix"] = f"{s}'s", None
        elif kind in ("section", "class"):
            s = lang.surface_section(val, rng)
            res["entity"] = s
            res["suffix"] = rng.choice([f"in {s}", f"for {s}", f"of {s}", f"of class {s}" if "class" not in s.lower() else f"of {s}"])
        elif kind == "subject":
            s = lang.surface_subject(val, rng)
            res["entity"] = s
            if rng.random() < 0.4:
                res["adj"] = s
            else:
                res["suffix"] = rng.choice([f"in {s}", f"for {s}", f"of {s}"])
        elif kind == "weekday":
            s = lang.surface_weekday(val, rng)
            if rng.random() < 0.35:
                res["adj"] = s
            else:
                res["suffix"] = rng.choice([f"on {s}", f"for {s}", f"on {s}s"])
        elif kind == "room":
            s = lang.surface_room(val, rng)
            res["entity"] = s
            res["suffix"] = rng.choice([f"in {s}", f"for {s}", f"of {s}"])
        elif kind == "title":
            s = lang.surface_title(val, rng)
            res["entity"] = s
            res["suffix"] = rng.choice([f"for {s}", f"of {s}", f"called {s}", f"in {s}"])
        elif kind == "enum":
            phrases = lang.ENUM_PHRASES.get((view.name, cond.col, val)) or [val.lower()]
            p = rng.choice(phrases)
            if rng.random() < 0.65:
                res["adj"] = p
            else:
                res["suffix"] = rng.choice([f"with {spec.label} {p}", f"that are {p}", f"whose {spec.label} is {p}"])
        else:
            res["suffix"] = f"with {spec.label} {val}"
        return res

    # ---- single-turn generic frames -------------------------------------------
    def generic(self, view: View) -> Optional[Tuple[str, Items, List[str]]]:
        rng = self.rng
        n_conds = rng.choices([0, 1, 2], [0.18, 0.55, 0.27])[0]
        conds, verbs, protect = [], [], []
        used: List[str] = []
        for _ in range(n_conds):
            picked = self.pick_cond(view, used)
            if not picked:
                break
            c, info = picked
            used.append(c.col)
            conds.append(c)
            v = self.verb(view, c, info)
            verbs.append(v)
            protect.append(str(info["surface"]))
        adjs = [v["adj"] for v in verbs if v["adj"]]
        # one possessive/adjective is natural; several get awkward -> push extras to suffix form
        adj = adjs[0] if adjs else ""
        suffixes = [v["suffix"] for v in verbs if v["suffix"]]
        for v in verbs[1:]:
            if v["adj"] and v["adj"] != adj and not v["suffix"]:
                suffixes.append(f"{v['adj']}")
        F = " ".join(suffixes)
        N = self.noun(view)
        kind = rng.choices(["list", "count", "attr", "super", "agg", "thresh", "who"], [0.42, 0.12, 0.2, 0.08, 0.06, 0.06, 0.06])[0]
        key = next((c for c in view.cols if c.kind in ENTITY_KINDS), None)
        nums = [c for c in view.cols if c.kind in ("number", "pct") and c.name not in ("roll_no", "period_no")]
        if kind == "attr" and key and not conds:
            ents = self.dom(view.name, key.name)
            if not ents:
                kind = "list"
        if kind == "count":
            q = clean(rng.choice(COUNT_FRAMES).format(adj=adj, N=N, F=F))
            return q, [Q(view.name, a=("count", "*"), w=[(c.col, c.op, c.value) for c in conds])], protect
        if kind == "who" and any(c.kind in PEOPLE_KINDS for c in view.cols[:2]):
            q = clean(rng.choice(WHO_FRAMES).format(adj=adj, N=N, F=F))
            return q, [Q(view.name, w=[(c.col, c.op, c.value) for c in conds])], protect
        if kind == "attr":
            cols = [c for c in view.cols if c.phrases and c.name != (key.name if key else "") and c.kind not in ("date",) or (c.kind == "date" and c.phrases)]
            if key and cols:
                col = rng.choice(cols)
                dom = self.dom(view.name, key.name)
                if dom:
                    ent = rng.choice(dom)
                    other = [c for c in conds if c.col != key.name]
                    surf = {"student": lang.surface_person, "faculty": lang.surface_person, "person": lang.surface_person}.get(key.kind)
                    if surf:
                        s = surf(ent, rng)
                    elif key.kind == "section":
                        s = lang.surface_section(ent, rng)
                    elif key.kind == "subject":
                        s = lang.surface_subject(ent, rng)
                    elif key.kind == "title":
                        s = lang.surface_title(ent, rng)
                    else:
                        s = lang.surface_room(ent, rng)
                    P = rng.choice(col.phrases)
                    q = clean(rng.choice(ATTR_FRAMES).format(P=P, E=s))
                    wh = [(key.name, "eq", ent)] + [(c.col, c.op, c.value) for c in other[:1]]
                    if other[:1]:
                        d = self.verb(view, other[0], {"surface": other[0].value, "date": view.col(other[0].col).kind == "date"})
                        extra = d["suffix"] or (d["adj"] or "")
                        q = clean(f"{q} {extra}")
                    return q, [Q(view.name, sel=[key.name, col.name], w=wh)], protect + [s]
        if kind == "super" and nums:
            col = rng.choice(nums)
            hi = rng.random() < 0.5
            w_ = rng.choice(["highest", "maximum", "best", "top"] if hi else ["lowest", "minimum", "least", "smallest"])
            q = rng.choice([f"which {N} has the {w_} {col.phrases[0] if col.phrases else col.label} {F}",
                            f"{w_} {col.phrases[0] if col.phrases else col.label} {F}",
                            f"show the {N} with the {w_} {col.phrases[0] if col.phrases else col.label} {F}"])
            return clean(q), [Q(view.name, w=[(c.col, c.op, c.value) for c in conds], o=(col.name, "desc" if hi else "asc"), n=1)], protect
        if kind == "agg" and nums:
            col = rng.choice(nums)
            fn = rng.choice(["avg", "sum", "max", "min"])
            word = {"avg": "average", "sum": "total", "max": "maximum", "min": "minimum"}[fn]
            P = col.phrases[0] if col.phrases else col.label
            q = rng.choice([f"what is the {word} {P} {F}", f"{word} {P} of {N} {F}", f"find the {word} {P} {F}"])
            grp = None
            if not conds and rng.random() < 0.4:
                gcol = next((c for c in view.cols if c.kind in ("section", "subject", "class")), None)
                if gcol:
                    grp = gcol.name
                    q = rng.choice([f"{word} {P} by {gcol.label}", f"what is the {word} {P} per {gcol.label}", f"{word} {P} for each {gcol.label}"])
            return clean(q), [Q(view.name, w=[(c.col, c.op, c.value) for c in conds], g=grp, a=(fn, col.name))], protect
        if kind == "thresh" and nums:
            col = rng.choice(nums)
            x = rng.choice([40, 50, 60, 70, 75, 80, 85, 90]) if col.kind == "pct" else rng.choice([5, 10, 20, 35, 50, 75])
            lt = rng.random() < 0.5
            P = col.phrases[0] if col.phrases else col.label
            word = rng.choice(["below", "less than", "under"] if lt else ["above", "more than", "over"])
            q = rng.choice([f"{N} with {P} {word} {x}", f"which {N} have {P} {word} {x}", f"{N} whose {P} is {word} {x}", f"show {N} having {P} {word} {x} {F}"])
            return clean(q), [Q(view.name, w=[(c.col, c.op, c.value) for c in conds] + [(col.name, "lt" if lt else "gt", str(x))])], protect
        q = clean(rng.choice(LIST_FRAMES).format(adj=adj, N=N, F=F))
        return q, [Q(view.name, w=[(c.col, c.op, c.value) for c in conds])], protect

    # ---- curated semantic frames ---------------------------------------------------
    def person(self, view: str, col: str) -> str:
        return self.rng.choice(self.dom(view, col))

    def specials(self) -> List[Callable[[], Optional[Tuple[str, Items, List[str]]]]]:
        rng, vi, g = self.rng, self.vi, self
        pp = lambda v: lang.surface_person(v, rng)  # noqa: E731
        sec = lambda v: lang.surface_section(v, rng)  # noqa: E731
        sub = lambda v: lang.surface_subject(v, rng)  # noqa: E731

        def dom(v, c):
            return g.dom(v, c)

        def date_pair():
            d = rng.choice(g.dates)
            surf, iso = lang.surface_date(d, rng)
            return (surf if rng.random() < 0.12 else iso), iso

        def stu():
            return rng.choice(dom("sv_students", "student_name"))

        def fac():
            return rng.choice(dom("sv_faculty", "faculty_name"))

        def sct():
            return rng.choice(dom("sv_sections", "section"))

        def cls():
            return rng.choice(dom("sv_exams", "class_name") or dom("sv_marks", "section"))

        def sbj():
            return rng.choice(dom("sv_subjects", "subject_name"))

        S: List[Callable] = []

        def add(fn):
            S.append(fn)
            return fn

        @add
        def att_bare():
            q = rng.choice(["show attendance", "attendance", "attendance report", "what is the attendance", "show me attendance details",
                            "check attendance", "attendance status", "i want to see attendance", "view attendance", "attendance records"])
            return q, [Control("ask", ["sv_attendance_summary", "sv_student_attendance", "sv_faculty_attendance"])], []

        @add
        def att_date_ask():
            d, iso = date_pair()
            q = rng.choice([f"attendance on {d}", f"who was absent on {d}", f"who was absent", f"who is absent today", f"absentees on {d}",
                            f"attendance for {d}", f"who was late on {d}", f"who was present on {d}"])
            if "today" in q:
                q = q.replace("today", iso)
            return q, [Control("ask", ["sv_student_attendance", "sv_faculty_attendance"])], []

        @add
        def att_student_sum():
            s = stu()
            q = rng.choice([f"attendance of {pp(s)}", f"{pp(s)}'s attendance", f"how is {pp(s)}'s attendance", f"attendance percentage of {pp(s)}",
                            f"what is {pp(s)}'s attendance percentage", f"how many periods did {pp(s)} attend", f"how regular is {pp(s)}"])
            return q, [Q("sv_attendance_summary", w=[("student_name", "eq", s)])], [s]

        @add
        def att_section_sum():
            c = rng.choice(dom("sv_attendance_summary", "section"))
            kind = rng.random() < 0.5
            tgt = c if kind else c.rsplit("-", 1)[0]
            q = rng.choice([f"attendance report for {sec(tgt)}", f"student attendance percentage in {sec(tgt)}", f"attendance of students in {sec(tgt)}"])
            col = ("section", "eq", c) if kind else ("class_name", "eq", c.rsplit("-", 1)[0])
            return q, [Q("sv_attendance_summary", w=[col])], [tgt]

        @add
        def att_faculty():
            f = fac()
            if rng.random() < 0.5:
                d, iso = date_pair()
                q = rng.choice([f"was {pp(f)} present on {d}", f"what time did {pp(f)} check in on {d}", f"attendance of teacher {pp(f)} on {d}", f"did {pp(f)} come on {d}"])
                return q, [Q("sv_faculty_attendance", w=[("faculty_name", "eq", f), ("attendance_date", "eq", iso)])], [f]
            q = rng.choice([f"attendance of teacher {pp(f)}", f"when did {pp(f)} check in", f"{pp(f)}'s check in and check out times", f"faculty attendance of {pp(f)}"])
            return q, [Q("sv_faculty_attendance", w=[("faculty_name", "eq", f)])], [f]

        @add
        def att_faculty_all():
            d, iso = date_pair()
            q = rng.choice([f"faculty attendance on {d}", f"teacher attendance on {d}", f"what was the faculty attendance status on {d}",
                            f"staff attendance for {d}", f"which teachers were present on {d}", f"show faculty attendance {d}"])
            return q, [Q("sv_faculty_attendance", w=[("attendance_date", "eq", iso)])], []

        @add
        def att_faculty_status():
            st = rng.choice(dom("sv_faculty_attendance", "status"))
            ph = rng.choice(lang.ENUM_PHRASES.get(("sv_faculty_attendance", "status", st), [st.lower()]))
            noun = rng.choice(["teachers", "faculty members", "staff"])
            q = rng.choice([f"which {noun} were {ph}", f"which {noun} are {ph}", f"{noun} who were {ph}", f"who among the {noun} was {ph}"])
            return q, [Q("sv_faculty_attendance", w=[("status", "eq", st)])], []

        @add
        def att_student_status():
            st = rng.choice(dom("sv_student_attendance", "status"))
            ph = rng.choice(lang.ENUM_PHRASES.get(("sv_student_attendance", "status", st), [st.lower()]))
            w = [("status", "eq", st)]
            extra = ""
            if rng.random() < 0.6:
                d, iso = date_pair()
                w.append(("attendance_date", "eq", iso))
                extra = rng.choice([f" on {d}", f" for {d}"])
            q = rng.choice([f"which students were {ph}{extra}", f"students who were {ph}{extra}", f"list {ph} students{extra}", f"which pupils were {ph}{extra}"])
            return q, [Q("sv_student_attendance", w=w)], []

        @add
        def att_student_on_date():
            s = stu()
            d, iso = date_pair()
            q = rng.choice([f"was {pp(s)} present on {d}", f"did {pp(s)} attend class on {d}", f"attendance of {pp(s)} on {d}", f"was {pp(s)} absent on {d}"])
            return q, [Q("sv_student_attendance", w=[("student_name", "eq", s), ("attendance_date", "eq", iso)])], [s]

        @add
        def att_why_absent():
            s = stu()
            q = rng.choice([f"why was {pp(s)} absent", f"reason for {pp(s)}'s absence", f"why did {pp(s)} miss class", f"remarks for {pp(s)}'s absence"])
            return q, [Q("sv_student_attendance", sel=["attendance_date", "period_no", "subject", "status", "remarks"], w=[("student_name", "eq", s), ("status", "eq", "Absent")])], [s]

        @add
        def att_low():
            x = rng.choice([50, 60, 70, 75, 80, 85, 90])
            q = rng.choice(["who has low attendance", "students with poor attendance", "which students have low attendance"])
            if rng.random() < 0.6:
                q = rng.choice([f"which students have attendance below {x}%", f"students with attendance less than {x} percent", f"who is below {x}% attendance", f"attendance under {x}%"])
                return q, [Q("sv_attendance_summary", w=[("attendance_pct", "lt", str(x))])], []
            return q, [Q("sv_attendance_summary", w=[("attendance_pct", "lt", "75")])], []

        @add
        def att_perfect():
            q = rng.choice(["students with 100% attendance", "who has perfect attendance", "who attended all classes", "which students have full attendance"])
            return q, [Q("sv_attendance_summary", w=[("attendance_pct", "ge", "100")])], []

        @add
        def att_lowest():
            hi = rng.random() < 0.4
            q = rng.choice(["which student has the highest attendance", "best attendance"] if hi else ["which student has the lowest attendance", "who has the worst attendance"])
            return q, [Q("sv_attendance_summary", o=("attendance_pct", "desc" if hi else "asc"), n=1)], []

        @add
        def att_all():
            q = rng.choice(["show the attendance percentage for all students", "student attendance percentages", "attendance percentage of all students",
                            "list students and their attendance rates", "overall attendance of every student", "show all student attendance percentages"])
            return q, [Q("sv_attendance_summary")], []

        @add
        def leave_on_date():
            d, iso = date_pair()
            q = rng.choice([f"who is on leave on {d}", f"who applied for leave on {d}", f"leave requests covering {d}", f"who has leave on {d}"])
            return q, [Q("sv_leaves", w=[("from_date", "le", iso), ("to_date", "ge", iso)])], []

        @add
        def leave_person():
            p = rng.choice(dom("sv_leaves", "applicant"))
            q = rng.choice([f"has {pp(p)} applied for leave", f"leave history of {pp(p)}", f"was {pp(p)}'s leave approved", f"what is the status of {pp(p)}'s leave",
                            f"reason for {pp(p)}'s leave", f"leave requests of {pp(p)}"])
            return q, [Q("sv_leaves", w=[("applicant", "eq", p)])], [p]

        @add
        def leave_pending():
            q = rng.choice(["which leave requests are pending approval", "show pending leave requests", "pending leaves", "which leave requests are currently pending approval",
                            "leaves awaiting admin approval", "what leave requests need review", "are there any leaves waiting for approval"])
            return q, [Q("sv_leaves", w=[("status", "eq", "Pending")])], []

        @add
        def leave_why_rejected():
            q = rng.choice(["why was the leave rejected", "who rejected the leave and why", "admin remarks on rejected leaves", "reason for rejection of leave"])
            return q, [Q("sv_leaves", sel=["applicant", "leave_type", "from_date", "to_date", "status", "reviewed_by", "admin_remarks"], w=[("status", "eq", "Rejected")])], []

        @add
        def leave_limit():
            lt = rng.choice(dom("sv_leave_types", "leave_type"))
            ph = rng.choice(lang.ENUM_PHRASES.get(("sv_leave_types", "leave_type", lt), [lt.lower()]))
            q = rng.choice([f"how many {ph} days are allowed in a year", f"maximum days of {ph}", f"how many {ph}s per year", f"what is the limit for {ph}"])
            return q, [Q("sv_leave_types", w=[("leave_type", "eq", lt)])], []

        @add
        def class_teacher():
            c = sct()
            q = rng.choice([f"who is the class teacher of {sec(c)}", f"class teacher for {sec(c)}", f"who is in charge of {sec(c)}", f"who is {sec(c)}'s class teacher",
                            f"who looks after {sec(c)}"])
            return q, [Q("sv_sections", sel=["section", "class_teacher"], w=[("section", "eq", c)])], [c]

        @add
        def class_teacher_of():
            f = fac()
            q = rng.choice([f"which class is {pp(f)} the class teacher of", f"is {pp(f)} a class teacher", f"which section does {pp(f)} handle as class teacher"])
            return q, [Q("sv_sections", sel=["section", "class_teacher"], w=[("class_teacher", "eq", f)])], [f]

        @add
        def class_teachers():
            q = rng.choice(["list the class teachers", "who are the class teachers", "class teachers of all sections", "sections with their class teachers"])
            return q, [Q("sv_sections", sel=["section", "class_teacher"])], []

        @add
        def who_teaches():
            s = sbj()
            if rng.random() < 0.5:
                c = sct()
                q = rng.choice([f"who teaches {sub(s)} in {sec(c)}", f"which teacher teaches {sub(s)} to {sec(c)}", f"who handles {sub(s)} for {sec(c)}"])
                return q, [Q("sv_course_allocation", sel=["section", "subject", "teacher"], w=[("subject", "eq", s), ("section", "eq", c)])], [s, c]
            q = rng.choice([f"who teaches {sub(s)}", f"which teacher handles {sub(s)}", f"who is the {sub(s)} teacher", f"teachers of {sub(s)}"])
            return q, [Q("sv_course_allocation", sel=["subject", "section", "teacher"], w=[("subject", "eq", s)])], [s]

        @add
        def teaches_what():
            f = fac()
            q = rng.choice([f"which subjects does {pp(f)} teach", f"what does {pp(f)} teach", f"subjects taught by {pp(f)}", f"what subject does {pp(f)} handle"])
            return q, [Q("sv_faculty", sel=["faculty_name", "subjects_taught"], w=[("faculty_name", "eq", f)])], [f]

        @add
        def periods_per_week():
            s, c = sbj(), sct()
            q = rng.choice([f"how many periods per week for {sub(s)} in {sec(c)}", f"weekly periods of {sub(s)} for {sec(c)}", f"how many {sub(s)} periods does {sec(c)} have per week"])
            return q, [Q("sv_course_allocation", sel=["subject", "section", "periods_per_week"], w=[("subject", "eq", s), ("section", "eq", c)])], [s, c]

        @add
        def students_in_class():
            c = sct()
            q = rng.choice([f"who are the students in {sec(c)}", f"students of {sec(c)}", f"list students in {sec(c)}", f"which students study in {sec(c)}"])
            return q, [Q("sv_students", sel=["student_name", "section", "roll_no", "admission_no"], w=[("section", "eq", c)])], [c]

        @add
        def how_many_students():
            if rng.random() < 0.5:
                c = sct()
                q = rng.choice([f"how many students are in {sec(c)}", f"how many students does {sec(c)} have", f"strength of {sec(c)}", f"number of students in {sec(c)}"])
                return q, [Q("sv_students", a=("count", "*"), w=[("section", "eq", c)])], [c]
            q = rng.choice(["how many students are there", "total number of students", "how many students does the school have", "count of students"])
            return q, [Q("sv_students", a=("count", "*"))], []

        @add
        def gender_filter():
            gd = rng.choice(dom("sv_students", "gender") or ["Male", "Female"])
            word = {"Male": rng.choice(["boys", "male students"]), "Female": rng.choice(["girls", "female students"])}.get(gd, "students")
            if rng.random() < 0.5:
                c = sct()
                q = f"show the {word} in {sec(c)}"
                return q, [Q("sv_students", sel=["student_name", "section", "gender"], w=[("section", "eq", c), ("gender", "eq", gd)])], [c]
            return f"list all {word}", [Q("sv_students", sel=["student_name", "section", "gender"], w=[("gender", "eq", gd)])], []

        @add
        def contact_attr():
            s = stu()
            col, ph = rng.choice([("email", ["email", "email address", "email id"]), ("phone", ["phone number", "mobile number"]),
                                  ("guardian_name", ["guardian", "parent's name", "guardian name"]), ("guardian_phone", ["guardian's phone number", "parent's phone number"]),
                                  ("address", ["address", "home address"]), ("date_of_birth", ["date of birth", "birthday"]),
                                  ("roll_no", ["roll number"]), ("admission_no", ["admission number"]), ("gender", ["gender"]), ("admission_date", ["admission date"])])
            q = rng.choice([f"what is {pp(s)}'s {rng.choice(ph)}", f"{rng.choice(ph)} of {pp(s)}", f"tell me the {rng.choice(ph)} of {pp(s)}", f"{pp(s)}'s {rng.choice(ph)}"])
            return q, [Q("sv_students", sel=["student_name", col], w=[("student_name", "eq", s)])], [s]

        @add
        def student_profile():
            s = stu()
            q = rng.choice([f"tell me about {pp(s)}", f"student profile of {pp(s)}", f"details of student {pp(s)}", f"who is {pp(s)}", f"show details for {pp(s)}"])
            return q, [Q("sv_students", w=[("student_name", "eq", s)])], [s]

        @add
        def teacher_attr():
            f = fac()
            col, ph = rng.choice([("qualification", ["qualification", "degree"]), ("joining_date", ["joining date", "date of joining"]), ("designation", ["designation", "position"]),
                                  ("employee_no", ["employee number", "employee id"]), ("email", ["email", "email address"]), ("phone", ["phone number"])])
            q = rng.choice([f"what is {pp(f)}'s {rng.choice(ph)}", f"{rng.choice(ph)} of {pp(f)}", f"when did {pp(f)} join" if col == "joining_date" else f"{pp(f)}'s {rng.choice(ph)}"])
            return q, [Q("sv_faculty", sel=["faculty_name", col], w=[("faculty_name", "eq", f)])], [f]

        @add
        def teacher_profile():
            f = fac()
            q = rng.choice([f"tell me about teacher {pp(f)}", f"profile of {pp(f)}", f"details of {pp(f)}", f"who is {pp(f)}"])
            return q, [Q("sv_faculty", w=[("faculty_name", "eq", f)])], [f]

        @add
        def principal():
            role = rng.choice(dom("sv_users", "role") or ["Admin"])
            if role == "Admin":
                q = rng.choice(["who is the principal", "who is the administrator", "who is the admin", "who has the admin role", "who are the administrators"])
            else:
                q = rng.choice([f"list all {rng.choice(lang.ENUM_PHRASES.get(('sv_users', 'role', role), [role.lower()]))} users", f"users with role {role.lower()}", f"show all {role.lower()} accounts"])
            return q, [Q("sv_users", w=[("role", "eq", role)])], []

        @add
        def timetable_day():
            c, d = sct(), rng.choice(dom("sv_timetable", "day_of_week") + ["Wednesday", "Thursday"])
            q = rng.choice([f"what is {sec(c)}'s {lang.surface_weekday(d, rng)} timetable", f"show {lang.surface_weekday(d, rng)} timetable for {sec(c)}",
                            f"{sec(c)} schedule on {lang.surface_weekday(d, rng)}", f"what does {sec(c)} have on {lang.surface_weekday(d, rng)}", f"what classes does {sec(c)} have on {lang.surface_weekday(d, rng)}"])
            return q, [Q("sv_timetable", w=[("section", "eq", c), ("day_of_week", "eq", d)])], [c]

        @add
        def timetable_class():
            c = sct()
            q = rng.choice([f"timetable of {sec(c)}", f"show the timetable for {sec(c)}", f"what is the weekly schedule of {sec(c)}", f"{sec(c)} time table"])
            return q, [Q("sv_timetable", w=[("section", "eq", c)])], [c]

        @add
        def timetable_teacher():
            f = fac()
            if rng.random() < 0.5:
                d = rng.choice(dom("sv_timetable", "day_of_week"))
                q = rng.choice([f"what classes does {pp(f)} have on {d}", f"timetable of {pp(f)} on {d}", f"{pp(f)}'s {d} schedule"])
                return q, [Q("sv_timetable", w=[("teacher", "eq", f), ("day_of_week", "eq", d)])], [f]
            q = rng.choice([f"timetable of {pp(f)}", f"when does {pp(f)} teach", f"{pp(f)}'s schedule", f"which periods does {pp(f)} have"])
            return q, [Q("sv_timetable", w=[("teacher", "eq", f)])], [f]

        @add
        def timetable_bare():
            q = rng.choice(["show the timetable", "timetable", "what is the schedule", "show me the time table", "class schedule", "daily timetable"])
            return q, [Q("sv_timetable")], []

        @add
        def timetable_day_only():
            d = rng.choice(dom("sv_timetable", "day_of_week") + ["Wednesday", "Friday"])
            q = rng.choice([f"what is scheduled on {d}", f"show the {d} timetable", f"which periods are there on {d}"])
            return q, [Q("sv_timetable", w=[("day_of_week", "eq", d)])], []

        @add
        def timetable_room():
            c, s = sct(), sbj()
            q = rng.choice([f"which room is {sub(s)} in for {sec(c)}", f"where is the {sub(s)} class of {sec(c)}"])
            return q, [Q("sv_course_allocation", sel=["section", "subject", "room"], w=[("subject", "eq", s), ("section", "eq", c)])], [s, c]

        @add
        def rank_of():
            s = rng.choice(dom("sv_progress", "student_name"))
            q = rng.choice([f"what is the rank of {pp(s)}", f"{pp(s)}'s class rank", f"where does {pp(s)} stand in class", f"what rank did {pp(s)} get"])
            return q, [Q("sv_progress", sel=["student_name", "exam_name", "class_rank", "percentage"], w=[("student_name", "eq", s)])], [s]

        @add
        def topper():
            q = rng.choice(["who is the class topper", "who got first rank", "who ranked first", "who stands first", "show the rank 1 student", "toppers list"])
            return q, [Q("sv_progress", sel=["student_name", "exam_name", "class_rank", "percentage", "grade"], w=[("class_rank", "eq", "1")])], []

        @add
        def report_card():
            s = rng.choice(dom("sv_progress", "student_name"))
            q = rng.choice([f"show progress report of {pp(s)}", f"report card for {pp(s)}", f"{pp(s)}'s progress report", f"teacher remarks for {pp(s)}"])
            return q, [Q("sv_progress", w=[("student_name", "eq", s)])], [s]

        @add
        def ranks_all():
            q = rng.choice(["show the rank list", "rank list", "progress reports of all students", "ranking of students", "class ranks"])
            return q, [Q("sv_progress", o=("class_rank", "asc"))], []

        @add
        def marks_student():
            s = rng.choice(dom("sv_marks", "student_name"))
            if rng.random() < 0.5:
                sj = rng.choice(dom("sv_marks", "subject"))
                q = rng.choice([f"what did {pp(s)} score in {sub(sj)}", f"{pp(s)}'s {sub(sj)} marks", f"how many marks did {pp(s)} get in {sub(sj)}", f"what grade did {pp(s)} get in {sub(sj)}"])
                return q, [Q("sv_marks", sel=["student_name", "subject", "marks_obtained", "max_marks", "grade"], w=[("student_name", "eq", s), ("subject", "eq", sj)])], [s, sj]
            q = rng.choice([f"show marks of {pp(s)}", f"exam results of {pp(s)}", f"subject wise marks of {pp(s)}", f"how did {pp(s)} perform in the exam", f"{pp(s)}'s marks"])
            return q, [Q("sv_marks", w=[("student_name", "eq", s)])], [s]

        @add
        def marks_extreme():
            s = rng.choice(dom("sv_marks", "student_name"))
            hi = rng.random() < 0.5
            q = rng.choice([f"which subject did {pp(s)} score the {'highest' if hi else 'lowest'}", f"{pp(s)}'s {'best' if hi else 'weakest'} subject", f"in which subject is {pp(s)} {'strongest' if hi else 'weakest'}"])
            return q, [Q("sv_marks", sel=["student_name", "subject", "marks_obtained", "max_marks"], w=[("student_name", "eq", s)], o=("marks_obtained", "desc" if hi else "asc"), n=1)], [s]

        @add
        def marks_topscorer():
            sj = rng.choice(dom("sv_marks", "subject"))
            hi = rng.random() < 0.65
            q = rng.choice([f"who scored the {'highest' if hi else 'lowest'} in {sub(sj)}", f"{'top' if hi else 'lowest'} scorer in {sub(sj)}", f"who got the {'best' if hi else 'worst'} marks in {sub(sj)}"])
            return q, [Q("sv_marks", sel=["student_name", "subject", "marks_obtained", "max_marks"], w=[("subject", "eq", sj)], o=("marks_obtained", "desc" if hi else "asc"), n=1)], [sj]

        @add
        def marks_avg():
            s = rng.choice(dom("sv_marks", "student_name"))
            q = rng.choice([f"what is the average marks of {pp(s)}", f"average score of {pp(s)}"])
            return q, [Q("sv_marks", w=[("student_name", "eq", s)], a=("avg", "marks_obtained"))], [s]

        @add
        def marks_result():
            r = rng.choice(["Pass", "Fail"])
            ph = rng.choice(lang.ENUM_PHRASES[("sv_marks", "result", r)])
            q = rng.choice([f"who {ph}", f"which students {ph}", f"students who {ph} the exam", f"who {ph} in the mid term"])
            return q, [Q("sv_marks", w=[("result", "eq", r)])], []

        @add
        def marks_class():
            c = rng.choice(["Grade 10", "Grade 9", "Grade 10-A", "Grade 9-A"])
            q = rng.choice([f"show the exam marks for {sec(c)} students", f"exam results of {sec(c)}", f"marks of students in {sec(c)}", f"how did {sec(c)} do in the exam"])
            col = ("section", "eq", c) if "-" in c else ("class_name", "eq", c)
            return q, [Q("sv_marks", w=[col])], [c]

        @add
        def marks_bare():
            q = rng.choice(["show marks", "exam results", "show the exam marks", "marks list", "show the exam report", "list all marks"])
            return q, [Q("sv_marks")], []

        @add
        def compare():
            a, b = rng.sample(dom("sv_marks", "student_name"), 2) if len(dom("sv_marks", "student_name")) > 1 else (stu(), stu())
            view, noun = rng.choice([("sv_marks", "marks"), ("sv_progress", "performance"), ("sv_progress", "progress reports"), ("sv_marks", "exam results"), ("sv_attendance_summary", "attendance")])
            q = rng.choice([f"compare {pp(a)} and {pp(b)}'s {noun}", f"compare {noun} of {pp(a)} and {pp(b)}", f"{pp(a)} vs {pp(b)} {noun}", f"who scored higher {pp(a)} or {pp(b)}" if view != "sv_attendance_summary" else f"{pp(a)} or {pp(b)} who attended more"])
            return q, [Q(view, w=[("student_name" if view != "sv_attendance_summary" else "student_name", "in", f"{a} / {b}")])], [a, b]

        @add
        def hw_due():
            v = rng.choice(["sv_homework", "sv_assignments"])
            t = rng.choice(dom(v, "title"))
            q = rng.choice([f"what is the due date of {lang.surface_title(t, rng)}", f"when is {lang.surface_title(t, rng)} due", f"deadline for {lang.surface_title(t, rng)}"])
            return q, [Q(v, sel=["title", "section", "due_date"], w=[("title", "eq", t)])], [t]

        @add
        def hw_submitted():
            v, vs, tc = rng.choice([("sv_homework", "sv_homework_submissions", "homework_title"), ("sv_assignments", "sv_assignment_submissions", "assignment_title")])
            t = rng.choice(dom(v, "title"))
            if rng.random() < 0.5:
                q = rng.choice([f"who submitted {lang.surface_title(t, rng)}", f"who has submitted {lang.surface_title(t, rng)}", f"list submissions for {lang.surface_title(t, rng)}"])
                return q, [Q(vs, w=[(tc, "eq", t), ("status", "ne", "Pending")])], [t]
            q = rng.choice([f"who hasn't submitted {lang.surface_title(t, rng)}", f"who has not submitted {lang.surface_title(t, rng)}", f"pending submissions for {lang.surface_title(t, rng)}"])
            return q, [Q(vs, w=[(tc, "eq", t), ("status", "eq", "Pending")])], [t]

        @add
        def hw_student():
            v, tc = rng.choice([("sv_homework_submissions", "homework_title"), ("sv_assignment_submissions", "assignment_title")])
            s = rng.choice(dom(v, "student_name"))
            word = "homework" if v == "sv_homework_submissions" else "assignment"
            q = rng.choice([f"has {pp(s)} submitted {word}", f"{word} status of {pp(s)}", f"did {pp(s)} do the {word}", f"{pp(s)}'s {word} submissions"])
            return q, [Q(v, w=[("student_name", "eq", s)])], [s]

        @add
        def hw_pending():
            v, word = rng.choice([("sv_homework_submissions", "homework"), ("sv_assignment_submissions", "assignments")])
            q = rng.choice([f"who has not submitted {word}", f"pending {word} submissions", f"which students are yet to submit {word}", f"who hasn't submitted {word} yet"])
            return q, [Q(v, w=[("status", "eq", "Pending")])], []

        @add
        def hw_class():
            v, word = rng.choice([("sv_homework", "homework"), ("sv_assignments", "assignments")])
            c = sct()
            q = rng.choice([f"what {word} is assigned for {sec(c)}", f"{word} for {sec(c)}", f"show {word} of {sec(c)}", f"what {word} does {sec(c)} have"])
            return q, [Q(v, w=[("section", "eq", c)])], [c]

        @add
        def hw_both():
            c = sct()
            q = rng.choice([f"what homework and assignments are assigned to {sec(c)}", f"homework and assignments for {sec(c)}", f"show homework and assignments of {sec(c)}"])
            return q, [Q("sv_homework", w=[("section", "eq", c)]), Q("sv_assignments", w=[("section", "eq", c)])], [c]

        @add
        def exam_when():
            s = sbj()
            q = rng.choice([f"when is the {sub(s)} exam", f"what time does the {sub(s)} exam begin", f"{sub(s)} exam date", f"which room is the {sub(s)} exam in", f"who is the invigilator for {sub(s)}"])
            sel = ["exam_name", "subject", "exam_date", "start_time", "end_time"]
            if "room" in q:
                sel = ["exam_name", "subject", "room"]
            if "invigilator" in q:
                sel = ["exam_name", "subject", "invigilator"]
            return q, [Q("sv_exams", sel=sel, w=[("subject", "eq", s)])], [s]

        @add
        def exam_class():
            c = rng.choice(["Grade 9", "Grade 10"])
            q = rng.choice([f"exam schedule for {sec(c)}", f"which exams are there for {sec(c)}", f"is there an exam for {sec(c)}", f"when are the exams of {sec(c)}"])
            return q, [Q("sv_exams", w=[("class_name", "eq", c)])], [c]

        @add
        def exam_all():
            q = rng.choice(["what exams are scheduled", "list of exams", "show the exam schedule", "upcoming exams", "exam dates"])
            return q, [Q("sv_exams")], []

        @add
        def exam_pass():
            s = sbj()
            q = rng.choice([f"what are the pass marks for {sub(s)}", f"maximum marks in the {sub(s)} exam", f"pass marks of {sub(s)}"])
            return q, [Q("sv_exams", sel=["exam_name", "subject", "max_marks", "pass_marks"], w=[("subject", "eq", s)])], [s]

        @add
        def grade_lookup():
            x = rng.choice([35, 45, 55, 65, 75, 85, 92, 98])
            q = rng.choice([f"which grade is {x} percent", f"what grade is {x}%", f"grade for {x} percent", f"what grade does {x}% get"])
            return q, [Q("sv_grade_scale", w=[("min_percent", "le", str(x)), ("max_percent", "ge", str(x))])], []

        @add
        def grade_scale():
            q = rng.choice(["what is the grading scale", "show the grade scale", "grade boundaries", "list the grades and their percentages", "grade points for each grade"])
            return q, [Q("sv_grade_scale")], []

        @add
        def notif_unread():
            p = rng.choice(dom("sv_notifications", "recipient"))
            q = rng.choice([f"unread notifications of {pp(p)}", f"does {pp(p)} have unread notifications", f"show unread messages for {pp(p)}"])
            return q, [Q("sv_notifications", w=[("recipient", "eq", p), ("is_read", "eq", "false")])], [p]

        @add
        def notif_person():
            p = rng.choice(dom("sv_notifications", "recipient"))
            q = rng.choice([f"notifications for {pp(p)}", f"what messages did {pp(p)} receive", f"show alerts for {pp(p)}"])
            return q, [Q("sv_notifications", w=[("recipient", "eq", p)])], [p]

        @add
        def sections_summary():
            q = rng.choice(["how many students are in each class", "students per class", "class strength", "list the classes and sections", "what classes does the school have", "how many sections are there"])
            return q, [Q("sv_sections", sel=["section", "class_teacher", "student_count"])], []

        @add
        def rooms():
            r = rng.choice(dom("sv_classrooms", "room"))
            q = rng.choice([f"what is the capacity of {lang.surface_room(r, rng)}", f"how many students can {lang.surface_room(r, rng)} hold", f"what type of room is {lang.surface_room(r, rng)}"])
            sel = ["room", "room_type"] if "type" in q else ["room", "capacity"]
            return q, [Q("sv_classrooms", sel=sel, w=[("room", "eq", r)])], [r]

        @add
        def labs():
            q = rng.choice(["which rooms are labs", "list the labs", "show all laboratories"])
            return q, [Q("sv_classrooms", w=[("room_type", "eq", "Lab")])], []

        @add
        def period_time():
            n = rng.choice([1, 2, 3])
            q = rng.choice([f"what time does period {n} start", f"when does period {n} end", f"timing of period {n}", f"what time is period {n}"])
            return q, [Q("sv_periods", w=[("period_no", "eq", str(n))])], []

        @add
        def academic_year():
            q = rng.choice(["what is the current academic year", "when does the academic year start", "when does the school year end", "academic year dates"])
            return q, [Q("sv_academic_years", w=[("is_current", "eq", "true")])], []

        @add
        def subjects_list():
            q = rng.choice(["what subjects are taught", "list all subjects", "which subjects are offered", "show subject codes", "how many subjects are there"])
            if "how many" in q:
                return q, [Q("sv_subjects", a=("count", "*"))], []
            return q, [Q("sv_subjects")], []

        @add
        def subject_desc():
            s = sbj()
            q = rng.choice([f"what does {sub(s)} cover", f"describe the subject {sub(s)}", f"what topics are in {sub(s)}", f"what is {sub(s)} about"])
            return q, [Q("sv_subjects", sel=["subject_name", "description"], w=[("subject_name", "eq", s)])], [s]

        return S

    # ---- follow-ups -------------------------------------------------------------
    def followup(self, base_q: str, items: Items) -> Optional[Example]:
        rng = self.rng
        q0 = items[0]
        if not isinstance(q0, Query) or len(items) != 1:
            return None
        view = VIEW_BY_NAME[q0.view]
        ctx = format_items(items)
        mode = rng.choices(["swap", "add", "count", "pronoun", "same_sel"], [0.35, 0.2, 0.1, 0.3, 0.05])[0]
        swappable = [c for c in q0.where if view.col(c.col) and view.col(c.col).kind in ("student", "faculty", "person", "section", "subject", "weekday", "date", "room", "title") and c.op == "eq"]
        if mode == "swap" and swappable:
            c = rng.choice(swappable)
            spec = view.col(c.col)
            if spec.kind == "date":
                d = rng.choice(self.dates)
                s2, iso = lang.surface_date(d, rng)
                nv, surf = iso, iso
            else:
                dom = [v for v in self.dom(q0.view, c.col) if v != c.value]
                if not dom:
                    return None
                nv = rng.choice(dom)
                surf = {"student": lang.surface_person, "faculty": lang.surface_person, "person": lang.surface_person, "section": lang.surface_section,
                        "subject": lang.surface_subject, "weekday": lang.surface_weekday, "room": lang.surface_room, "title": lang.surface_title}[spec.kind](nv, rng) \
                    if spec.kind != "date" else nv
            lead = rng.choice(["what about {x}", "and {x}", "how about {x}", "same for {x}", "what about for {x}", "{x} too", "and for {x}", "now {x}", "ok and {x}", "what about {x} then", "{x}?"])
            q = lead.format(x=surf)
            new = Query(q0.view, list(q0.sel), [Cond(w.col, w.op, nv if w is c else w.value) for w in q0.where], q0.group, q0.agg, q0.order, q0.limit)
            return q, ctx, new.fmt()
        if mode == "add":
            cands = [c for c in self.fcols(view) if c not in [w.col for w in q0.where] and view.col(c).kind in ("weekday", "subject", "section", "date", "enum")]
            if not cands:
                return None
            picked = self.pick_cond(view, exclude=[w.col for w in q0.where], cols=cands)
            if not picked:
                return None
            c, info = picked
            spec = view.col(c.col)
            if spec.kind == "weekday":
                phr = rng.choice(["only {x}", "and on {x}", "just {x}", "what about {x}", "for {x} only", "and {x}?"]).format(x=lang.surface_weekday(c.value, rng))
            elif spec.kind == "date":
                phr = rng.choice(["only on {x}", "and on {x}?", "just {x}", "for {x} only"]).format(x=info["surface"])
            elif spec.kind == "enum":
                p = rng.choice(lang.ENUM_PHRASES.get((q0.view, c.col, c.value), [c.value.lower()]))
                phr = rng.choice(["only the {x} ones", "just the {x}", "only {x}", "show only {x}"]).format(x=p)
            elif spec.kind == "section":
                phr = rng.choice(["only {x}", "just for {x}", "for {x} only", "in {x}"]).format(x=lang.surface_section(c.value, rng))
            else:
                phr = rng.choice(["only {x}", "just {x}", "for {x} only"]).format(x=lang.surface_subject(c.value, rng))
            new = Query(q0.view, list(q0.sel), list(q0.where) + [c], q0.group, q0.agg, q0.order, q0.limit)
            return phr, ctx, new.fmt()
        if mode == "count" and not q0.agg:
            q = rng.choice(["how many are there", "how many?", "count them", "what is the count", "how many of them", "total number?"])
            new = Query(q0.view, [], list(q0.where), None, ("count", "*"))
            return q, ctx, new.fmt()
        if mode == "pronoun" and q0.where:
            ent = next((c for c in q0.where if view.col(c.col).kind in PEOPLE_KINDS + ("section",) and c.op == "eq"), None)
            if not ent:
                return None
            kind = view.col(ent.col).kind
            targets = []
            for tv in VIEWS:
                if tv.name == q0.view:
                    continue
                tcol = next((c for c in tv.cols if c.kind == kind and (kind != "person" or c.name in ("applicant", "recipient", "full_name"))), None)
                if tcol and self.dom(tv.name, tcol.name) and tv.nouns and tv.name not in ("sv_progress_items", "sv_users", "sv_course_allocation") and \
                        (tv.name, tcol.name) not in MULTI:
                    targets.append((tv, tcol))
            if kind in ("student", "person"):
                targets += [(tv, tv.col(cn)) for tv, cn in [(VIEW_BY_NAME["sv_marks"], "student_name"), (VIEW_BY_NAME["sv_attendance_summary"], "student_name")] if tv.name != q0.view]
            if not targets:
                return None
            tv, tcol = rng.choice(targets)
            if kind in ("student", "faculty", "person"):
                pron = rng.choice(PRONOUNS) if kind != "faculty" else rng.choice(PRONOUNS)
                leads = ["what are {p} {N}", "show {p} {N}", "and {p} {N}?", "what about {p} {N}", "how are {p} {N}", "tell me {p} {N}", "{p} {N} please"]
                q = rng.choice(leads).format(p=pron, N=rng.choice(tv.nouns))
            else:
                leads = ["show its {N}", "and the {N} of that class?", "what about the {N} for the same class", "show the {N} of that class", "what are its {N}"]
                q = rng.choice(leads).format(N=rng.choice(tv.nouns))
            new = Query(tv.name, [], [Cond(tcol.name, "eq", ent.value)])
            return q, ctx, new.fmt()
        if mode == "same_sel" and q0.sel:
            return None
        return None

    def scope_fill(self) -> Optional[Example]:
        rng = self.rng
        views = [v for v in VIEWS if v.scope]
        view = rng.choice(views)
        base_conds: List[Tuple[str, str, str]] = []
        if rng.random() < 0.4:
            others = [c for c in self.fcols(view) if c not in view.scope and view.col(c).kind in ("weekday", "subject", "enum", "date")]
            if others:
                picked = self.pick_cond(view, cols=others)
                if picked:
                    base_conds.append((picked[0].col, picked[0].op, picked[0].value))
        scope_col = rng.choice([c for c in view.scope if self.dom(view.name, c)] or [None])
        if not scope_col:
            return None
        val = rng.choice(self.dom(view.name, scope_col))
        kind = view.col(scope_col).kind
        s = {"student": lang.surface_person, "faculty": lang.surface_person, "section": lang.surface_section, "subject": lang.surface_subject,
             "title": lang.surface_title}.get(kind, lambda v, r: v)(val, rng)
        reply = rng.choice(["{s}", "for {s}", "{s} please", "the one for {s}", "{s}'s", "show {s}", "i mean {s}", "it's {s}", "let's go with {s}"]).format(s=s)
        partial = Query(view.name, [], [Cond(*c) for c in base_conds])
        full = Query(view.name, [], [Cond(*c) for c in base_conds] + [Cond(scope_col, "eq", val)])
        return reply, f"pending: {partial.fmt()}", full.fmt()

    def ask_fill(self) -> Optional[Example]:
        rng = self.rng
        fam = rng.choice(["attendance", "attendance_date", "leave", "report", "schedule"])
        if fam == "attendance":
            choices = [("sv_attendance_summary", ["overall percentage", "attendance percentage", "percentages", "students overall"]),
                       ("sv_student_attendance", ["student attendance", "students", "period by period", "student records"]),
                       ("sv_faculty_attendance", ["teachers", "faculty", "teacher attendance", "staff"])]
            options = [c[0] for c in choices]
            orig = rng.choice(["show attendance", "attendance", "attendance report", "check attendance"])
            view, phr = rng.choice(choices)
            reply = rng.choice(phr + ["the first one", "number 1"] if view == options[0] else phr + (["the second one", "option 2"] if view == options[1] else ["the third one", "option 3"]))
            return reply, f"ask: {', '.join(options)} | orig: {orig}", Q(view).fmt()
        if fam == "attendance_date":
            d = rng.choice(self.dates)
            _, iso = lang.surface_date(d, rng)
            orig = rng.choice([f"who was absent on {iso}", f"attendance on {iso}", f"who was late on {iso}"])
            st = "Absent" if "absent" in orig else "Late" if "late" in orig else None
            options = ["sv_student_attendance", "sv_faculty_attendance"]
            view = rng.choice(options)
            phr = rng.choice(["students", "student attendance", "the students", "pupils", "first one"] if view == options[0] else ["teachers", "faculty", "the teachers", "staff", "second one"])
            w = [("attendance_date", "eq", iso)] + ([("status", "eq", st)] if st else [])
            return phr, f"ask: {', '.join(options)} | orig: {orig}", Q(view, w=w).fmt()
        if fam == "leave":
            options = ["sv_leaves", "sv_leave_types"]
            orig = rng.choice(["leave", "show leave", "leave details"])
            view = rng.choice(options)
            phr = rng.choice(["leave requests", "requests", "applications", "first one"] if view == options[0] else ["leave types", "types and limits", "policy", "second one"])
            return phr, f"ask: {', '.join(options)} | orig: {orig}", Q(view).fmt()
        if fam == "report":
            options = ["sv_marks", "sv_progress", "sv_attendance_summary"]
            orig = rng.choice(["show report", "report", "student report", "show the report"])
            view, phr = rng.choice([("sv_marks", ["marks", "exam marks", "subject marks"]), ("sv_progress", ["progress report", "rank and percentage", "report card"]),
                                    ("sv_attendance_summary", ["attendance", "attendance report"])])
            return phr[0] if rng.random() < 0.5 else rng.choice(phr), f"ask: {', '.join(options)} | orig: {orig}", Q(view).fmt()
        options = ["sv_timetable", "sv_exams"]
        orig = rng.choice(["schedule", "show schedule", "what is the schedule"])
        view = rng.choice(options)
        phr = rng.choice(["class timetable", "timetable", "daily periods"] if view == options[0] else ["exam schedule", "exams", "exam dates"])
        return phr, f"ask: {', '.join(options)} | orig: {orig}", Q(view).fmt()

    def vague_ask(self) -> Example:
        rng = self.rng
        sets = [
            (["schedule", "show schedule", "what is the schedule", "my schedule", "when is it scheduled"], ["sv_timetable", "sv_exams"]),
            (["leave", "show leave", "leave details", "leaves"], ["sv_leaves", "sv_leave_types"]),
            (["show report", "report", "student report", "show the report"], ["sv_marks", "sv_progress", "sv_attendance_summary"]),
            (["show attendance", "attendance", "attendance report", "check attendance", "attendance status"], ["sv_attendance_summary", "sv_student_attendance", "sv_faculty_attendance"]),
            (["performance", "how is the performance", "show performance"], ["sv_marks", "sv_progress", "sv_attendance_summary"]),
        ]
        qs, opts = rng.choice(sets)
        return rng.choice(qs), None, Control("ask", opts).fmt()

    def controls(self) -> Example:
        rng = self.rng
        kind = rng.choices(["oos", "write", "manual", "help", "greet", "thanks", "unsupported"], [0.26, 0.2, 0.2, 0.1, 0.07, 0.05, 0.12])[0]
        if kind == "oos":
            return rng.choice(OOS_Q), None, "oos"
        if kind == "write":
            if rng.random() < 0.5:
                return rng.choice(WRITE_Q), None, "write"
            verb = rng.choice(["delete", "remove", "update", "change", "edit", "insert", "add", "drop", "modify", "cancel", "truncate", "erase"])
            v = rng.choice(VIEWS)
            ent = rng.choice(self.dom("sv_students", "student_name") or ["Arjun Rao"])
            tmpl = rng.choice(["{v} all {n} from the database", "{v} the {n} of {e}", "{v} {e}", "please {v} the {n}", "{v} every {s}", "i want to {v} some {n}"])
            return clean(tmpl.format(v=verb, n=rng.choice(v.nouns), e=ent, s=v.singular or "record")), None, "write"
        if kind == "manual":
            if rng.random() < 0.7:
                return rng.choice(MANUAL_Q), None, "manual"
            topic = rng.choice(["attendance", "leave approval", "exam scheduling", "marks entry", "the timetable", "progress reports", "homework", "security", "backups", "year end", "deleting records", "adding students", "sections", "course assignment"])
            return rng.choice([f"what does the manual say about {topic}", f"according to the manual, how do I handle {topic}", f"explain the procedure for {topic}", f"user manual: {topic}"]), None, "manual"
        if kind == "help":
            return rng.choice(HELP_Q), None, "help"
        if kind == "greet":
            return rng.choice(GREET_Q), None, "greet"
        if kind == "thanks":
            return rng.choice(THANKS_Q), None, "thanks"
        concept = rng.choice(UNSUPPORTED_CONCEPTS)
        s = rng.choice(self.dom("sv_students", "student_name") or ["Arjun Rao"])
        tmpl = rng.choice(["what are the {c} of {s}", "show {c} details", "how much is the {c}", "when is the {c}", "{s}'s {c}", "tell me about the {c}", "list all {c}", "what is the {c} status for {s}"])
        return clean(tmpl.format(c=concept, s=s)), None, f"unsupported: {concept}"


def build(vi: ValueIndex, ay: Tuple[dt.date, dt.date], n: int, seed: int) -> List[Dict]:
    gen = Gen(vi, ay, seed)
    rng = gen.rng
    specials = gen.specials()
    views = [v for v in VIEWS if gen.fcols(v) or v.name in VIEW_BY_NAME]
    out: Dict[str, Dict] = {}
    rejected = 0

    def norm(q: str) -> str:
        return normalize_question(q, ay[0], ay[1])

    def push(q: str, ctx: Optional[str], target: str, protect: Sequence[str] = (), noise: bool = True, tag: str = "gen") -> None:
        nonlocal rejected
        # gold must compile (queries) -> the corpus can never teach the model something the runtime rejects
        try:
            for it in parse(target):
                if isinstance(it, Query):
                    compile_query(it, vi)
        except (Clarify, ValueError) as exc:
            rejected += 1
            if rejected <= 5:
                print("  rejected gold:", target, "->", getattr(exc, "kind", exc))
            return
        text = add_noise_q(q, protect) if noise else q
        text = norm(text)
        inp = f"ctx: {ctx} || q: {text}" if ctx else f"q: {text}"
        out.setdefault(inp, {"input": inp, "target": target, "tag": tag})

    add_noise_q = lambda q, protect: lang.add_noise(q, rng, list(protect))  # noqa: E731

    # hand-written gold pairs (weighted x4 with different noise)
    for q, ctx, target in BANK:
        for _ in range(4):
            push(q, ctx, target, noise=rng.random() < 0.7, tag="bank")
        push(q, ctx, target, noise=False, tag="bank")

    attempts = 0
    while len(out) < n and attempts < n * 12:
        attempts += 1
        r = rng.random()
        if r < 0.40:
            v = rng.choice(views)
            res = gen.generic(v)
            if res:
                q, items, protect = res
                push(q, None, format_items(items), protect)
        elif r < 0.62:
            res = rng.choice(specials)()
            if res:
                q, items, protect = res
                push(q, None, format_items(items), protect, tag="special")
        elif r < 0.74:
            base = None
            if rng.random() < 0.55:
                base = rng.choice(specials)()
            else:
                base = gen.generic(rng.choice(views))
            if base:
                fu = gen.followup(base[0], base[1])
                if fu:
                    push(fu[0], fu[1], fu[2], noise=True, tag="followup")
        elif r < 0.80:
            ex = gen.scope_fill()
            if ex:
                push(ex[0], ex[1], ex[2], tag="scope")
        elif r < 0.84:
            ex = gen.ask_fill()
            if ex:
                push(ex[0], ex[1], ex[2], tag="askfill")
        elif r < 0.87:
            ex = gen.vague_ask()
            push(ex[0], ex[1], ex[2], tag="ask")
        elif r < 0.91:
            a = rng.choice(specials)()
            b = rng.choice(specials)()
            if a and b and isinstance(a[1][0], Query) and isinstance(b[1][0], Query) and a[1][0].view != b[1][0].view and len(a[1]) == 1 and len(b[1]) == 1:
                joiner = rng.choice([" and ", " and also ", ", and ", "? and ", " also "])
                q = a[0].rstrip("?") + joiner + b[0]
                push(q, None, format_items(a[1] + b[1]), a[2] + b[2], tag="compound")
        else:
            q, ctx, target = gen.controls()
            push(q, ctx, target, tag="control")
    print(f"generated {len(out)} unique examples in {attempts} attempts ({rejected} gold rejected)")
    rows = list(out.values())
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--out", default=str(ROOT / "backend" / "ml" / "data"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from app.database import run_readonly_query

    vi = load_value_index()
    ay_row = run_readonly_query("SELECT start_date, end_date FROM school.academic_years WHERE is_current ORDER BY start_date DESC LIMIT 1")["rows"][0]
    ay = (dt.date.fromisoformat(ay_row["start_date"]), dt.date.fromisoformat(ay_row["end_date"]))
    rows = build(vi, ay, args.n, args.seed)

    cut = int(len(rows) * 0.03)
    val, train = rows[:cut], rows[cut:]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        with open(outdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tags: Dict[str, int] = {}
    for r in rows:
        tags[r["tag"]] = tags.get(r["tag"], 0) + 1
    print("train", len(train), "val", len(val), tags)


if __name__ == "__main__":
    main()
