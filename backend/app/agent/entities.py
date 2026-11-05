"""Entity linking: find the people, classes, dates, etc. mentioned in a question.

Everything is resolved against the live :class:`Catalog`, never against a hard-coded
list. The extractor also returns the surface spans it recognised so the question can
be *masked* (``"Divya's marks"`` -> ``"xxperson marks"``) before it reaches the topic
classifier; that is what lets the model generalise to any student, class or date.
"""
from __future__ import annotations

import datetime as dt
import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .catalog import Catalog, Exam, Klass, LeaveType, Person, Room, Section, Subject, Titled

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
_DAY_RE = re.compile(r"\b(monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|thur|thu|friday|fri|saturday)\b", re.I)
_DAY_FULL = {d[:3].lower(): d for d in WEEKDAYS}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# Placeholder tokens used to mask recognised entities before classification.
MASK = {
    "person": "xxperson", "section": "xxclass", "subject": "xxsubject", "weekday": "xxday",
    "date": "xxdate", "title": "xxtitle", "room": "xxroom", "exam": "xxexam", "leavetype": "xxleavetype",
}

EXTRA_STOP = {
    "show", "tell", "give", "list", "please", "want", "like", "need", "know", "see", "get", "find", "display",
    "kindly", "hey", "hi", "hello", "ok", "okay", "thanks", "thank", "also", "about", "details", "info",
    "information", "me", "us", "i", "can", "could", "would", "will", "shall", "does", "did", "do",
}
STOPWORDS: Set[str] = set(ENGLISH_STOP_WORDS) | EXTRA_STOP

# status vocabulary -> canonical value (the schema's allowed enum values).
_STATUS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(p, re.I), v)
    for p, v in [
        (r"\bhalf[- ]?day\b", "Half Day"),
        (r"\bon leave\b", "On Leave"),
        (r"\b(?:absent|absentees?|absence|absentee|missed|bunk\w*)\b", "Absent"),
        (r"\b(?:late|tardy|late ?comers?)\b", "Late"),
        (r"\bexcused\b", "Excused"),
        (r"\b(?:present|attended|attending)\b", "Present"),
        (r"\b(?:approved|accepted|granted|sanctioned)\b", "Approved"),
        (r"\b(?:rejected|declined|denied|refused|turned down)\b", "Rejected"),
        (r"\b(?:pending|awaiting|waiting|unapproved|outstanding|unsubmitted|overdue|not yet)\b", "Pending"),
        (r"\b(?:checked|reviewed)\b", "Checked"),
        (r"\b(?:graded|marked)\b", "Graded"),
        (r"\b(?:passed|pass|passing)\b", "Pass"),
        (r"\b(?:failed|fail|failing)\b", "Fail"),
    ]
]
_NEG_SUBMIT = re.compile(
    r"\b(?:not|never|yet to|haven'?t|hasn'?t|didn'?t|did not|have not|has not|without|no)\s+(?:yet\s+)?"
    r"(?:submit\w*|turn\w* in|hand\w* in|done|finish\w*|complet\w*|upload\w*)",
    re.I,
)
_POS_SUBMIT = re.compile(r"\b(?:submit\w*|handed in|turned in|uploaded)\b", re.I)

_ATTR_PATTERNS: Dict[str, re.Pattern] = {
    k: re.compile(v, re.I)
    for k, v in {
        "email": r"\be-?mail\b",
        "phone": r"\b(?:phone|mobile|cell|contact (?:no|number|details?))\b",
        "address": r"\b(?:address|resid\w+|lives?|home ?town)\b",
        "guardian": r"\b(?:guardians?|parents?|father|mother)\b",
        "dob": r"\b(?:birth ?days?|date of birth|dob|born|how old|age)\b",
        "roll": r"\broll\b",
        "admission": r"\b(?:admission|admitted|enrol\w*)\b",
        "gender": r"\b(?:gender|sex)\b",
        "qualification": r"\b(?:qualifications?|degrees?|qualified|educat\w+)\b",
        "joining": r"\b(?:joined|joining|join date|experience|since when|how long)\b",
        "designation": r"\b(?:designation|position|job title)\b",
        "username": r"\b(?:username|user name|login)\b",
        "employee": r"\b(?:employee (?:no|number|id|code)|emp (?:no|id|code))\b",
    }.items()
}

_SUBJECT_STOP = {"and", "the", "of", "for", "with", "in"}
_TITLE_GENERIC = {"project", "assignment", "assignments", "homework", "worksheet", "report", "essay_", "my", "the", "and", "of"}


@dataclass
class Entities:
    students: List[Person] = field(default_factory=list)
    faculty: List[Person] = field(default_factory=list)
    admins: List[Person] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)  # explicitly named (Grade 10-A)
    classes: List[Klass] = field(default_factory=list)  # grade-only (Grade 10)
    subjects: List[Subject] = field(default_factory=list)
    weekdays: List[str] = field(default_factory=list)
    dates: List[dt.date] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    exams: List[Exam] = field(default_factory=list)
    leave_types: List[LeaveType] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    titles: List[Titled] = field(default_factory=list)
    gender: Optional[str] = None
    threshold: Optional[Tuple[str, float]] = None
    attrs: List[str] = field(default_factory=list)
    # flags
    count: bool = False
    top: bool = False
    bottom: bool = False
    compare: bool = False
    unread: bool = False
    wants_all: bool = False
    negated_submit: bool = False
    # reference words resolved later against the conversation
    pron_person: bool = False
    pron_plural: bool = False
    pron_class: bool = False
    # problems worth telling the user about
    ambiguous: List[Tuple[str, List[Person]]] = field(default_factory=list)
    unknown_names: List[str] = field(default_factory=list)
    unknown_classes: List[str] = field(default_factory=list)
    spans: List[Tuple[int, int, str]] = field(default_factory=list)

    # ---- helpers ---------------------------------------------------------
    @property
    def people(self) -> List[Person]:
        return self.students + self.faculty + self.admins

    def section_ids(self, cat: Catalog) -> List[int]:
        """Explicit sections plus every section of any grade-only mention."""
        ids: List[int] = [s.section_id for s in self.sections]
        for k in self.classes:
            ids.extend(i for i in k.section_ids if i not in ids)
        return ids

    def class_ids(self) -> List[int]:
        ids = [s.class_id for s in self.sections]
        ids.extend(k.class_id for k in self.classes if k.class_id not in ids)
        return list(dict.fromkeys(ids))

    def has_scope(self) -> bool:
        return bool(self.people or self.sections or self.classes)

    def is_empty(self) -> bool:
        return not (
            self.people or self.sections or self.classes or self.subjects or self.weekdays or self.dates
            or self.statuses or self.exams or self.leave_types or self.rooms or self.titles
            or self.threshold or self.roles
        )

    def merged_over(self, base: "Entities") -> "Entities":
        """Follow-up merge: self (new message) overrides base (previous question)."""
        out = Entities()
        for name in ("students", "faculty", "admins", "sections", "classes", "subjects", "weekdays", "dates",
                     "statuses", "exams", "leave_types", "roles", "rooms", "titles", "attrs"):
            mine, theirs = getattr(self, name), getattr(base, name)
            setattr(out, name, list(mine) if mine else list(theirs))
        # a newly named person replaces an old class scope, and vice versa
        if self.people:
            out.sections, out.classes = [], []
            for name in ("students", "faculty", "admins"):
                setattr(out, name, list(getattr(self, name)))
        elif self.sections or self.classes:
            out.students, out.faculty, out.admins = [], [], []
        out.gender = self.gender or base.gender
        out.threshold = self.threshold or base.threshold
        for flag in ("count", "top", "bottom", "compare", "unread", "wants_all", "negated_submit"):
            setattr(out, flag, getattr(self, flag))
        return out


# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _word_re(phrase: str) -> re.Pattern:
    parts = [re.escape(p) for p in phrase.split()]
    return re.compile(r"(?<![\w])" + r"[\s\-]+".join(parts) + r"(?![\w])", re.I)


def _subject_aliases(sub: Subject) -> Set[str]:
    words = set(re.findall(r"[a-z]+", sub.name.lower()))
    words |= set(re.findall(r"[a-z]+", sub.description.lower()))
    words.add(sub.code.lower())
    return {w for w in words if len(w) >= 3 and w not in _SUBJECT_STOP and w not in STOPWORDS}


def _infer_year(month: int, day: int, cat: Catalog, explicit: Optional[int]) -> Optional[dt.date]:
    years = [explicit] if explicit else []
    if not years:
        ay = cat.academic_year
        try:
            start = dt.date.fromisoformat(ay["start"])
            end = dt.date.fromisoformat(ay["end"])
            years = [start.year, end.year]
        except Exception:
            years = [dt.date.today().year]
    best: Optional[dt.date] = None
    for y in years:
        try:
            cand = dt.date(y, month, day)
        except ValueError:
            continue
        if best is None:
            best = cand
        ay = cat.academic_year
        if ay and not explicit:
            try:
                if dt.date.fromisoformat(ay["start"]) <= cand <= dt.date.fromisoformat(ay["end"]):
                    return cand
            except Exception:
                pass
    return best


def _find_dates(text: str, cat: Catalog, spans: List[Tuple[int, int, str]]) -> List[dt.date]:
    found: List[Tuple[int, dt.date]] = []

    def add(m: re.Match, d: Optional[dt.date]) -> None:
        if d:
            found.append((m.start(), d))
            spans.append((m.start(), m.end(), "date"))

    for m in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        try:
            add(m, dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text):
        try:
            add(m, dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b\.?(?:,?\s*(\d{{4}}))?", text, re.I):
        add(m, _infer_year(_MONTHS[m.group(2).lower()], int(m.group(1)), cat, int(m.group(3)) if m.group(3) else None))
    for m in re.finditer(rf"\b({_MONTH_ALT})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(\d{{4}}))?", text, re.I):
        if any(s <= m.start() < e for s, e, k in spans if k == "date"):
            continue
        add(m, _infer_year(_MONTHS[m.group(1).lower()], int(m.group(2)), cat, int(m.group(3)) if m.group(3) else None))
    today = dt.date.today()
    for word, delta in (("today", 0), ("yesterday", -1), ("tomorrow", 1)):
        for m in re.finditer(rf"\b{word}\b", text, re.I):
            add(m, today + dt.timedelta(days=delta))
    found.sort(key=lambda t: t[0])
    return list(dict.fromkeys(d for _, d in found))


def _find_sections(text: str, cat: Catalog, ents: Entities) -> None:
    klass_by_number = {k.number: k for k in cat.classes if k.number is not None}
    taken: List[Tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        return not any(s < m.end() and m.start() < e for s, e in taken)

    def resolve(number: int, letter: Optional[str], m: re.Match) -> None:
        taken.append((m.start(), m.end()))
        ents.spans.append((m.start(), m.end(), "section"))
        k = klass_by_number.get(number)
        if k is None:
            ents.unknown_classes.append(f"Grade {number}")
            return
        if letter:
            sec = next((s for s in cat.sections if s.class_id == k.class_id and s.letter.lower() == letter.lower()), None)
            if sec is None:
                ents.unknown_classes.append(f"{k.name}-{letter.upper()}")
            elif sec not in ents.sections:
                ents.sections.append(sec)
        elif k not in ents.classes:
            ents.classes.append(k)

    for m in re.finditer(r"\b(?:grade|class|std|standard)\s*-?\s*(\d{1,2})(?:\s*-?\s*([a-z])\b(?!['’]?\w))?", text, re.I):
        resolve(int(m.group(1)), m.group(2), m)
    for m in re.finditer(r"(?<![\w-])(\d{1,2})-?([a-z])\b", text, re.I):
        if free(m) and int(m.group(1)) in klass_by_number:
            resolve(int(m.group(1)), m.group(2), m)
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)\b", text, re.I):
        if not free(m):
            continue
        before = text[: m.start()].rstrip().lower()
        after = text[m.end():].lstrip().lower()
        if re.match(rf"(?:of\s+)?(?:{_MONTH_ALT})\b", after) or re.search(rf"\b(?:{_MONTH_ALT})\.?$", before):
            continue  # an ordinal day of a month, not a grade
        resolve(int(m.group(1)), None, m)


def _find_people(text: str, cat: Catalog, ents: Entities, known: Set[str]) -> None:
    people = cat.people
    taken: List[Tuple[int, int]] = []
    resolved: List[Person] = []

    def add(p: Person, start: int, end: int) -> None:
        taken.append((start, end))
        ents.spans.append((start, end, "person"))
        if p in resolved:
            return
        resolved.append(p)
        {"student": ents.students, "faculty": ents.faculty, "admin": ents.admins}[p.kind].append(p)

    def free(start: int, end: int) -> bool:
        return not any(s < end and start < e for s, e in taken)

    # 1. emails, usernames, admission / employee numbers
    for p in people:
        for ident in filter(None, {p.email, p.username, p.code}):
            for m in re.finditer(re.escape(ident), text, re.I):
                if free(m.start(), m.end()):
                    add(p, m.start(), m.end())
    # 2. full names
    for p in people:
        if not p.last:
            continue
        for m in re.finditer(rf"\b{re.escape(p.first)}\s+{re.escape(p.last)}\b", text, re.I):
            if free(m.start(), m.end()):
                add(p, m.start(), m.end())
    # 3. first or last name alone (ambiguous names are reported, never guessed)
    token_map: Dict[str, List[Person]] = {}
    for p in people:
        for part in (p.first, p.last):
            if len(part) >= 3:
                token_map.setdefault(part.lower(), []).append(p)
    for m in re.finditer(r"[A-Za-z]+", text):
        tok = m.group().lower()
        if tok in token_map and free(m.start(), m.end()):
            cands = token_map[tok]
            if len(cands) == 1:
                add(cands[0], m.start(), m.end())
            else:
                ents.ambiguous.append((m.group(), cands))
                taken.append((m.start(), m.end()))
    # 4. typo tolerance for tokens that look like a name but match nothing else
    vocab = list(token_map)
    for m in re.finditer(r"[A-Za-z]+", text):
        tok = m.group().lower()
        if len(tok) < 4 or tok in known or tok in token_map or not free(m.start(), m.end()):
            continue
        close = difflib.get_close_matches(tok, vocab, n=2, cutoff=0.8)
        if close:
            owners = token_map[close[0]]
            if len(owners) == 1 and (len(close) == 1 or difflib.SequenceMatcher(None, tok, close[0]).ratio() > 0.9):
                add(owners[0], m.start(), m.end())


def _find_unknown_names(text: str, cat: Catalog, ents: Entities, known: Set[str]) -> None:
    """Capitalised words that are not part of the school vocabulary (e.g. "Rahul")."""
    covered = ents.spans
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        word = m.group()
        low = word.lower()
        if low in known or low in STOPWORDS:
            continue
        if any(s <= m.start() < e for s, e, _ in covered):
            continue
        before = text[: m.start()].rstrip()
        sentence_start = not before or before[-1] in ".?!"
        possessive = text[m.end(): m.end() + 2] in ("'s", "’s")
        if sentence_start and not possessive:
            continue
        ents.unknown_names.append(word)


def _find_titles(text: str, cat: Catalog, ents: Entities) -> None:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    sig: Dict[str, Set[str]] = {}
    for t in cat.titles:
        sig[f"{t.kind}:{t.item_id}"] = {w for w in re.findall(r"[a-z]+", t.title.lower()) if w not in _TITLE_GENERIC and w not in STOPWORDS and len(w) > 2}
    for t in cat.titles:
        key = f"{t.kind}:{t.item_id}"
        words = sig[key]
        hits = words & tokens
        unique_hits = {w for w in hits if sum(1 for o in sig.values() if w in o) == 1}
        full = re.search(re.escape(t.title.lower()), lowered)
        if full or len(hits) >= 2 or unique_hits:
            if t not in ents.titles:
                ents.titles.append(t)
            for w in hits:
                for m in re.finditer(rf"\b{re.escape(w)}\b", lowered):
                    ents.spans.append((m.start(), m.end(), "title"))


def extract_entities(text: str, cat: Catalog, known_words: Optional[Set[str]] = None) -> Entities:
    known = (known_words or set()) | STOPWORDS | set(_MONTHS) | {d.lower() for d in WEEKDAYS}
    ents = Entities()
    lowered = text.lower()

    _find_people(text, cat, ents, known)
    _find_sections(text, cat, ents)
    ents.dates = _find_dates(text, cat, ents.spans)

    for m in _DAY_RE.finditer(text):
        day = _DAY_FULL.get(m.group(1).lower()[:3])
        if day and day not in ents.weekdays:
            ents.weekdays.append(day)
            ents.spans.append((m.start(), m.end(), "weekday"))

    # subjects (aliases come from the subjects table: name, code and description words)
    for sub in cat.subjects:
        for alias in _subject_aliases(sub):
            for m in re.finditer(rf"\b{re.escape(alias)}\b|\b{re.escape(alias)}s\b", lowered):
                if sub not in ents.subjects:
                    ents.subjects.append(sub)
                ents.spans.append((m.start(), m.end(), "subject"))
        for m in re.finditer(r"\bmaths?\b", lowered):  # common abbreviation of the subject name
            if sub.name.lower().startswith("math"):
                if sub not in ents.subjects:
                    ents.subjects.append(sub)
                ents.spans.append((m.start(), m.end(), "subject"))

    # rooms
    for room in cat.rooms:
        m = _word_re(room.name).search(text)
        if m:
            ents.rooms.append(room)
            ents.spans.append((m.start(), m.end(), "room"))

    # exams: by full name or by exam type ("mid term", "unit test", "final")
    for ex in cat.exams:
        m = _word_re(ex.name).search(text)
        if m:
            ents.exams.append(ex)
            ents.spans.append((m.start(), m.end(), "exam"))
    for tname in cat.exam_types:
        base = re.sub(r"\bexam\b", "", tname, flags=re.I).strip()
        pat = re.compile(r"(?<![\w])" + r"[\s\-]*".join(re.escape(p) for p in base.lower().split()) + r"(?![\w])", re.I)
        m = pat.search(text)
        if m:
            for ex in cat.exams:
                if ex.type_name == tname and ex not in ents.exams:
                    ents.exams.append(ex)
            ents.spans.append((m.start(), m.end(), "exam"))

    # leave types ("sick leave", "casual", "earned")
    for lt in cat.leave_types:
        first = lt.name.split()[0].lower()
        m = re.search(rf"\b{re.escape(first)}\b", lowered)
        if m:
            ents.leave_types.append(lt)
            ents.spans.append((m.start(), m.end(), "leavetype"))

    _find_titles(text, cat, ents)

    # roles
    role_words = {"Admin": r"admins?|administrators?|principal", "Faculty": r"teachers?|faculty|staff|instructors?|professors?|teaching staff", "Student": r"students?|pupils?|learners?"}
    for role in cat.roles:
        pat = role_words.get(role, re.escape(role.lower()))
        if re.search(rf"\b(?:{pat})\b", lowered):
            ents.roles.append(role)

    # statuses
    negated = bool(_NEG_SUBMIT.search(text))
    ents.negated_submit = negated
    for pat, value in _STATUS_PATTERNS:
        if pat.search(text) and value not in ents.statuses:
            ents.statuses.append(value)
    if negated:
        ents.statuses = [s for s in ents.statuses if s not in ("Present",)]
        if "Pending" not in ents.statuses:
            ents.statuses.append("Pending")
    elif _POS_SUBMIT.search(text) and "Submitted" not in ents.statuses and "Pending" not in ents.statuses:
        ents.statuses.append("Submitted")
    if re.search(r"\bunread\b|\bnew notifications?\b", lowered):
        ents.unread = True

    # numeric threshold: "below 75%", "at least 90"
    m = re.search(r"(below|under|less than|lower than|fewer than|at most|<=?)\s*(\d+(?:\.\d+)?)", lowered)
    if m:
        ents.threshold = ("<", float(m.group(2)))
    else:
        m = re.search(r"(above|over|more than|greater than|higher than|at least|>=?|exceed\w*)\s*(\d+(?:\.\d+)?)\s*%", lowered)
        if m:
            ents.threshold = (">=" if "least" in m.group(1) or ">=" in m.group(1) else ">", float(m.group(2)))
    if re.search(r"\b(?:100\s*%|perfect|full attendance|hundred percent)", lowered):
        ents.threshold = (">=", 100.0)

    # gender filter ("girls", "female students")
    if re.search(r"\b(?:girls?|females?|women)\b", lowered):
        ents.gender = "Female"
    elif re.search(r"\b(?:boys?|males?|men)\b", lowered):
        ents.gender = "Male"

    for attr, pat in _ATTR_PATTERNS.items():
        if pat.search(text):
            ents.attrs.append(attr)

    ents.count = bool(re.search(r"\b(?:how many|number of|count|total number)\b", lowered))
    ents.top = bool(re.search(r"\b(?:top|highest|best|topper|toppers|maximum|most|first rank|rank(?:ed)? (?:1|first|one)|#1|1st)\b", lowered))
    ents.bottom = bool(re.search(r"\b(?:lowest|worst|least|minimum|bottom|weakest|poorest|last rank)\b", lowered))
    ents.compare = bool(re.search(r"\b(?:compar\w+|versus|vs\.?|difference|better than|against|side by side)\b|who (?:scored|did) (?:better|higher)|who scored more", lowered))
    ents.wants_all = bool(re.search(r"\b(?:all|everyone|everybody|everything|whole|entire|both|each|any)\b", lowered))

    ents.pron_person = bool(re.search(r"\b(?:he|she|his|her|hers|him|that (?:student|teacher|person|boy|girl)|this (?:student|teacher))\b", lowered))
    ents.pron_plural = bool(re.search(r"\b(?:they|them|their|theirs|those|these|both of them)\b", lowered))
    ents.pron_class = bool(re.search(r"\b(?:that|this|the same|same) (?:class|section|grade)\b", lowered))

    _find_unknown_names(text, cat, ents, known)
    return ents


def mask_question(text: str, ents: Entities) -> str:
    """Replace recognised entities with placeholder tokens for the classifier."""
    chosen: List[Tuple[int, int, str]] = []
    for s, e, kind in sorted(ents.spans, key=lambda t: (t[0], -(t[1] - t[0]))):
        if any(s < ce and cs < e for cs, ce, _ in chosen):
            continue
        chosen.append((s, e, kind))
    out, pos = [], 0
    for s, e, kind in chosen:
        out.append(text[pos:s])
        out.append(f" {MASK[kind]} ")
        pos = e
    out.append(text[pos:])
    masked = re.sub(r"\s+", " ", "".join(out)).strip().lower()
    return re.sub(r"\b(xx\w+)(?:\s+\1\b)+", r"\1", masked)  # collapse repeated placeholders
