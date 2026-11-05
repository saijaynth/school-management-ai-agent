"""The compact query language the parser model speaks, and its safe compiler.

Model output (single line, T5-tokenizer friendly)::

    sv_marks | sel: student_name, subject, marks_obtained | w: student_name eq Kiran Patel ; subject eq Science | o: marks_obtained desc | n: 1
    sv_students | a: count | w: section eq Grade 10-A
    sv_attendance_summary | g: section | a: avg attendance_pct

Control outputs (no query): ``oos``, ``write``, ``manual``, ``help``, ``greet``, ``thanks``,
``unsupported: fees``, ``ask: <view> , <view>`` (ask the user which topic they mean).
Several queries for one message are separated by `` ## ``.

The model never writes SQL. This module parses the line, validates every view / column / operator
against the semantic layer, snaps values to what really exists in the database, and only then
builds a parameterised SELECT. Anything it cannot validate is turned into a clarification.
"""
from __future__ import annotations

import datetime as dt
import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .semantic_layer import VIEW_BY_NAME, View, cte_prefix

OPS = {"eq", "ne", "lt", "gt", "le", "ge", "starts", "has", "in", "isnull", "notnull"}
AGGS = {"count", "avg", "sum", "max", "min"}
CONTROLS = {"oos", "write", "manual", "help", "greet", "thanks", "unsupported", "ask"}
SNAP_KINDS = {"student", "faculty", "person", "section", "class", "subject", "weekday", "enum", "room", "title"}
MULTI = {("sv_faculty", "subjects_taught"), ("sv_faculty", "class_teacher_of"), ("sv_subjects", "teachers")}
MAX_ROWS = 200


@dataclass
class Cond:
    col: str
    op: str
    value: str = ""

    def fmt(self) -> str:
        return f"{self.col} {self.op}" + (f" {self.value}" if self.op not in ("isnull", "notnull") else "")


@dataclass
class Query:
    view: str
    sel: List[str] = field(default_factory=list)
    where: List[Cond] = field(default_factory=list)
    group: Optional[str] = None
    agg: Optional[Tuple[str, str]] = None  # (fn, column or "*")
    order: Optional[Tuple[str, str]] = None
    limit: Optional[int] = None

    def fmt(self) -> str:
        parts = [self.view]
        if self.sel:
            parts.append("sel: " + ", ".join(self.sel))
        if self.where:
            parts.append("w: " + " ; ".join(c.fmt() for c in self.where))
        if self.group:
            parts.append(f"g: {self.group}")
        if self.agg:
            parts.append("a: " + (self.agg[0] if self.agg[1] == "*" else f"{self.agg[0]} {self.agg[1]}"))
        if self.order:
            parts.append(f"o: {self.order[0]} {self.order[1]}")
        if self.limit:
            parts.append(f"n: {self.limit}")
        return " | ".join(parts)

    def cond_for(self, col: str) -> Optional[Cond]:
        return next((c for c in self.where if c.col == col), None)


@dataclass
class Control:
    kind: str
    args: List[str] = field(default_factory=list)

    def fmt(self) -> str:
        return self.kind + (": " + ", ".join(self.args) if self.args else "")


Item = object  # Query | Control


class DslError(ValueError):
    pass


def parse(text: str) -> List[Item]:
    """Parse model output into queries/controls. Raises DslError on anything malformed."""
    items: List[Item] = []
    for chunk in [c.strip() for c in text.strip().split("##") if c.strip()]:
        head = re.split(r"[\s:]", chunk, maxsplit=1)[0].lower()
        if head in CONTROLS:
            rest = chunk[len(head):].lstrip(": ").strip()
            items.append(Control(head, [a.strip() for a in rest.split(",") if a.strip()]))
            continue
        segs = [s.strip() for s in chunk.split("|")]
        view = segs[0]
        if view not in VIEW_BY_NAME:
            raise DslError(f"unknown view: {view}")
        q = Query(view)
        for seg in segs[1:]:
            key, _, body = seg.partition(":")
            key, body = key.strip(), body.strip()
            if key == "sel":
                q.sel = [c.strip() for c in body.split(",") if c.strip()]
            elif key == "w":
                for part in [p.strip() for p in body.split(";") if p.strip()]:
                    m = re.match(r"^(\w+)\s+(\w+)(?:\s+(.*))?$", part)
                    if not m or m.group(2) not in OPS:
                        raise DslError(f"bad condition: {part}")
                    q.where.append(Cond(m.group(1), m.group(2), (m.group(3) or "").strip()))
            elif key == "g":
                q.group = body
            elif key == "a":
                bits = body.split()
                if not bits or bits[0] not in AGGS:
                    raise DslError(f"bad aggregate: {body}")
                q.agg = (bits[0], bits[1] if len(bits) > 1 else "*")
            elif key == "o":
                bits = body.split()
                q.order = (bits[0], bits[1].lower() if len(bits) > 1 else "asc")
            elif key == "n":
                if not body.isdigit():
                    raise DslError(f"bad limit: {body}")
                q.limit = int(body)
            else:
                raise DslError(f"unknown clause: {key}")
        items.append(q)
    if not items:
        raise DslError("empty output")
    return items


def format_items(items: Sequence[Item]) -> str:
    return " ## ".join(i.fmt() for i in items)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Value index: the real values in the database, used to snap what the model wrote.
# ---------------------------------------------------------------------------
@dataclass
class ValueIndex:
    values: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def domain(self, view: str, col: str) -> List[str]:
        return self.values.get((view, col), [])

    def kind_domain(self, kind: str) -> List[str]:
        out: List[str] = []
        for (vn, cn), vals in self.values.items():
            if VIEW_BY_NAME[vn].col(cn).kind == kind and kind != "date" and (vn, cn) not in MULTI:
                out.extend(v for v in vals if v not in out)
        return out


def load_value_index() -> ValueIndex:
    """One batched round trip: distinct categorical values + distinct dates for every semantic view."""
    from ..database import run_readonly_queries

    plan: List[Tuple[str, List[str], str]] = []
    for v in VIEW_BY_NAME.values():
        cols = [c.name for c in v.cols if (c.kind in SNAP_KINDS and (v.name, c.name) not in MULTI) or c.kind == "date"]
        if cols:
            def agg(c: str) -> str:
                kind = v.col(c).kind
                expr = f"CAST({c} AS date)" if kind == "date" else c
                return f"array_agg(DISTINCT {expr} ORDER BY {expr}) FILTER (WHERE {c} IS NOT NULL) AS {c}"
            plan.append((v.name, cols, cte_prefix([v.name]) + f"SELECT {', '.join(agg(c) for c in cols)} FROM {v.name}"))
    res = run_readonly_queries([(sql, None) for _, _, sql in plan])
    vi = ValueIndex()
    for (vname, cols, _), r in zip(plan, res):
        row = r["rows"][0] if r["rows"] else {}
        for c in cols:
            vi.values[(vname, c)] = [str(x) for x in (row.get(c) or [])]
    return vi


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _section_key(s: str) -> Optional[str]:
    m = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*-?\s*([a-z])?\b", s.lower())
    return (m.group(1) + (m.group(2) or "")) if m else None


def snap_value(view: str, col: str, value: str, vi: ValueIndex) -> Tuple[Optional[str], List[str]]:
    """Return (canonical value, candidates). value None + candidates => ambiguous / unknown."""
    col_spec = VIEW_BY_NAME[view].col(col)
    kind = col_spec.kind if col_spec else "text"
    WEEKDAYS_LIST = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if kind == "weekday":
        dom = WEEKDAYS_LIST
    else:
        dom = vi.domain(view, col) or vi.kind_domain(kind)
    if not dom:
        return value, []
    low = value.strip().lower()
    for d in dom:
        if d.lower() == low:
            return d, []
    if kind in ("section", "class"):
        key = _section_key(value)
        if key:
            hits = [d for d in dom if _section_key(d) == key]
            if len(hits) == 1:
                return hits[0], []
            if not hits and kind == "section" and key.isdigit():  # grade only -> prefix handled by caller
                return value, []
    toks = [d for d in dom if low and (low in d.lower().split() or _alnum(low) == _alnum(d))]
    if len(toks) == 1:
        return toks[0], []
    if len(toks) > 1:
        return None, toks
    pref = [d for d in dom if len(low) >= 3 and d.lower().startswith(low)]
    if len(pref) == 1:
        return pref[0], []
    close = difflib.get_close_matches(value, dom, n=3, cutoff=0.75)
    if len(close) == 1 or (close and difflib.SequenceMatcher(None, low, close[0].lower()).ratio() > 0.88):
        return close[0], []
    if kind == "title":
        words = {w for w in re.findall(r"[a-z]+", low) if len(w) > 3}
        hits = [d for d in dom if words and words & set(re.findall(r"[a-z]+", d.lower()))]
        if len(hits) == 1:
            return hits[0], []
    return None, close or []


@dataclass
class Clarify(Exception):
    """Raised by the compiler when the user must pick or restate something."""
    kind: str  # unknown_value | ambiguous_value | missing_scope | bad_query
    view: str = ""
    col: str = ""
    value: str = ""
    options: List[str] = field(default_factory=list)


@dataclass
class Compiled:
    sql: str
    params: tuple
    query: Query  # with snapped values
    columns: List[str]
    notes: List[str] = field(default_factory=list)


def _cond_sql(view: View, c: Cond, params: list) -> str:
    col = view.col(c.col)
    name = c.col
    if c.op == "isnull":
        return f"{name} IS NULL"
    if c.op == "notnull":
        return f"{name} IS NOT NULL"
    k = col.kind
    sym = {"eq": "=", "ne": "<>", "lt": "<", "gt": ">", "le": "<=", "ge": ">="}
    if k in ("number", "pct"):
        try:
            params.append(float(c.value))
        except ValueError:
            raise Clarify("bad_query", view.name, name, c.value)
        return f"{name} {sym.get(c.op, '=')} %s"
    if k == "date":
        try:
            d = dt.date.fromisoformat(c.value[:10])
        except ValueError:
            raise Clarify("bad_query", view.name, name, c.value)
        params.append(d)
        return f"CAST({name} AS date) {sym.get(c.op, '=')} %s"
    if k == "time":
        params.append(c.value)
        return f"{name} {sym.get(c.op, '=')} CAST(%s AS time)"
    if k == "bool":
        params.append(c.value.lower() in ("true", "yes", "1", "t"))
        return f"{name} {'<>' if c.op == 'ne' else '='} %s"
    if c.op == "in":
        params.append([v.strip().lower() for v in c.value.split("/") if v.strip()])
        return f"LOWER({name}) = ANY(%s)"
    if c.op == "starts":
        params.append(c.value.lower() + "%")
        return f"LOWER({name}) LIKE %s"
    if c.op == "has" or (view.name, name) in MULTI:
        params.append(c.value.lower())
        return f"POSITION(%s IN LOWER({name})) > 0"
    if c.op in ("lt", "gt", "le", "ge"):
        params.append(c.value)
        return f"{name} {sym[c.op]} %s"
    params.append(c.value.lower())
    return f"LOWER({name}) {'<>' if c.op == 'ne' else '='} %s"


def compile_query(q: Query, vi: ValueIndex) -> Compiled:
    view = VIEW_BY_NAME[q.view]
    names = set(view.colnames)
    notes: List[str] = []

    def need(col: str) -> str:
        if col not in names:
            raise Clarify("bad_query", q.view, col)
        return col

    snapped: List[Cond] = []
    for c in q.where:
        col = c.col
        if col == "class_name" and col not in names and "section" in names:  # grade-only on a section view
            snapped.append(Cond("section", "starts", c.value.strip().rstrip("-") + "-"))
            continue
        need(col)
        spec = view.col(col)
        value = c.value
        if c.op in ("eq", "ne", "starts", "in") and spec.kind in SNAP_KINDS and (q.view, col) not in MULTI:
            if c.op == "in":
                out = []
                for v in [x.strip() for x in value.split("/") if x.strip()]:
                    s, cands = snap_value(q.view, col, v, vi)
                    if s is None:
                        raise Clarify("ambiguous_value" if cands else "unknown_value", q.view, col, v, cands or vi.domain(q.view, col))
                    out.append(s)
                value = " / ".join(out)
            elif c.op != "starts":
                s, cands = snap_value(q.view, col, value, vi)
                if s is None:
                    raise Clarify("ambiguous_value" if len(cands) > 1 else "unknown_value", q.view, col, value, cands or vi.domain(q.view, col))
                if s != value:
                    notes.append(f"{col}: '{value}' -> '{s}'")
                value = s
        snapped.append(Cond(col, c.op, value))

    sel = [need(c) for c in q.sel] or [c for c in view.show]
    if q.group:
        need(q.group)
    params: list = []
    conds = [_cond_sql(view, c, params) for c in snapped]

    if q.agg:
        fn, col = q.agg
        if fn not in AGGS:
            raise Clarify("bad_query", q.view, fn)
        if fn == "count":
            expr, alias = "COUNT(*)", "count"
        else:
            need(col)
            expr, alias = f"{fn.upper()}({col})", f"{fn}_{col}"
        cols_sql = ([q.group] if q.group else []) + [f"{expr} AS {alias}"]
        out_cols = ([q.group] if q.group else []) + [alias]
        sql = f"SELECT {', '.join(cols_sql)} FROM {q.view}"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        if q.group:
            sql += f" GROUP BY {q.group} ORDER BY {q.group}"
    else:
        sql = f"SELECT {', '.join(sel)} FROM {q.view}"
        out_cols = sel
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        if q.order:
            col, direction = q.order
            need(col)
            sql += f" ORDER BY {col} {'DESC' if direction == 'desc' else 'ASC'} NULLS LAST"
        elif view.sort:
            sql += f" ORDER BY {view.sort}"
    sql += f" LIMIT {min(q.limit or MAX_ROWS, MAX_ROWS)}"
    final = Query(q.view, list(q.sel), snapped, q.group, q.agg, q.order, q.limit)
    return Compiled(cte_prefix([q.view]) + sql, tuple(params), final, out_cols, notes)


def missing_scope(q: Query) -> Optional[List[str]]:
    """Columns of which at least one is required (schema rule), if none was given."""
    view = VIEW_BY_NAME[q.view]
    if not view.scope or q.agg:
        return None
    given = {c.col for c in q.where if c.op not in ("isnull", "notnull")}
    if "class_name" in given:
        given.add("section")
    return None if given & set(view.scope) else list(view.scope)
