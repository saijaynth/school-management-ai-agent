"""Language knowledge used to verbalise database values and to add realistic noise."""
from __future__ import annotations

import datetime as dt
import random
import re
from typing import Dict, List, Tuple

# (view, column, value) -> ways people refer to that value. Falls back to value.lower().
ENUM_PHRASES: Dict[Tuple[str, str, str], List[str]] = {}


def _p(views, col, mapping):
    for v in views:
        for value, phrases in mapping.items():
            ENUM_PHRASES[(v, col, value)] = phrases


_p(["sv_leaves"], "status", {
    "Pending": ["pending", "awaiting approval", "waiting for approval", "not yet approved", "unapproved"],
    "Approved": ["approved", "accepted", "granted"], "Rejected": ["rejected", "declined", "denied"]})
_p(["sv_homework_submissions", "sv_assignment_submissions"], "status", {
    "Pending": ["pending", "not submitted", "not yet submitted", "yet to be submitted", "missing", "outstanding"],
    "Submitted": ["submitted", "handed in"], "Late": ["late", "submitted late"],
    "Checked": ["checked", "reviewed"], "Graded": ["graded", "marked"]})
_p(["sv_student_attendance"], "status", {
    "Absent": ["absent", "missing from class"], "Late": ["late", "who came late"], "Present": ["present"], "Excused": ["excused"]})
_p(["sv_faculty_attendance"], "status", {
    "Absent": ["absent"], "Late": ["late"], "Present": ["present"], "On Leave": ["on leave"], "Half Day": ["on half day", "half day"]})
_p(["sv_marks"], "result", {"Pass": ["passed", "pass"], "Fail": ["failed", "fail"]})
_p(["sv_users", "sv_leaves", "sv_notifications"], "role", {
    "Admin": ["admin", "administrator", "principal"], "Faculty": ["teacher", "faculty", "staff"], "Student": ["student"]})
_p(["sv_leaves"], "leave_type", {"Sick Leave": ["sick leave", "sick"], "Casual Leave": ["casual leave", "casual"],
                                  "Earned Leave": ["earned leave", "earned"]})
_p(["sv_leave_types"], "leave_type", {"Sick Leave": ["sick leave", "sick"], "Casual Leave": ["casual leave", "casual"],
                                       "Earned Leave": ["earned leave", "earned"]})
_p(["sv_students"], "gender", {"Male": ["male", "boys"], "Female": ["female", "girls"], "Other": ["other gender"]})
_p(["sv_classrooms"], "room_type", {"Lab": ["lab", "laboratory"], "Classroom": ["regular classroom"], "Library": ["library"], "Hall": ["hall"]})
_p(["sv_exams"], "exam_type", {"Mid Term": ["mid term", "midterm", "mid-term"], "Unit Test": ["unit test"], "Final Exam": ["final exam", "final"]})
_p(["sv_exam_types"], "exam_type", {"Mid Term": ["mid term", "midterm"], "Unit Test": ["unit test"], "Final Exam": ["final exam", "final"]})

BOOL_PHRASES: Dict[Tuple[str, bool], List[str]] = {
    ("is_read", False): ["unread", "not yet read"], ("is_read", True): ["read"],
    ("is_active", True): ["active"], ("is_active", False): ["inactive"],
    ("is_published", True): ["published"], ("is_published", False): ["unpublished"],
    ("is_current", True): ["current"], ("is_absent", True): ["absent"],
}

FIRST = ["please", "hey", "hi", "kindly", "ok so", "quick question:", "can you tell me", "i want to know", "could you show me",
         "i need to know", "tell me", "hey assistant,", "um,"]
FILLER_TAIL = ["please", "thanks", "pls", "for me", "right now", "in the database"]


def surface_person(name: str, rng: random.Random, allow_first: bool = True) -> str:
    parts = name.split()
    r = rng.random()
    if allow_first and len(parts) > 1 and r < 0.3:
        return parts[0]
    if r < 0.4:
        return name.lower()
    return name


def surface_section(value: str, rng: random.Random) -> str:
    m = re.match(r"^(.*?)(\d+)(?:-([A-Za-z]))?$", value)
    if not m:
        return value
    num, letter = m.group(2), m.group(3)
    if letter:
        return rng.choice([value, value.lower(), f"{num}-{letter}", f"{num}{letter}", f"Grade {num} {letter}", f"class {num}-{letter}",
                           f"grade {num}{letter.lower()}", f"{num}{letter.lower()}", f"Grade {num}-{letter.lower()}"])
    return rng.choice([value, value.lower(), f"class {num}", f"grade {num}", f"{num}th" if num not in ("1", "2", "3") else f"grade {num}"])


def surface_subject(value: str, rng: random.Random) -> str:
    opts = [value, value.lower(), value.lower()]
    if value.lower().startswith("math"):
        opts += ["maths", "math", "Maths"]
    if value.lower().startswith("sci"):
        opts += ["sci"]
    if value.lower().startswith("eng"):
        opts += ["eng"]
    return rng.choice(opts)


def surface_title(value: str, rng: random.Random) -> str:
    words = re.sub(r"[:]", "", value).split()
    r = rng.random()
    if r < 0.5 or len(words) < 3:
        return rng.choice([value, value.lower()])
    return " ".join(words[: rng.choice([2, 3])]).lower() if r < 0.8 else " ".join(words[-2:]).lower()


def surface_room(value: str, rng: random.Random) -> str:
    return rng.choice([value, value.lower()])


def surface_weekday(value: str, rng: random.Random) -> str:
    return rng.choice([value, value.lower(), value[:3]])


def surface_date(d: dt.date, rng: random.Random) -> Tuple[str, str]:
    """(surface form, normalised form). The model is trained mostly on the normalised form."""
    iso = d.isoformat()
    forms = [d.strftime("%B ") + str(d.day) + f", {d.year}", f"{d.day} {d.strftime('%b')} {d.year}", f"{d.strftime('%b')} {d.day}",
             f"{d.day} {d.strftime('%B')}", f"{d.day:02d}/{d.month:02d}/{d.year}", iso, f"{d.day}th {d.strftime('%B')}", "today", "yesterday"]
    return rng.choice(forms), iso


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.random()
    if kind < 0.35:
        return word[:i] + word[i + 1:]
    if kind < 0.7:
        return word[:i] + word[i] + word[i:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def add_noise(q: str, rng: random.Random, protect: List[str]) -> str:
    """Case/punctuation/politeness/typo noise. `protect` strings (entity surfaces) get typos rarely."""
    if rng.random() < 0.12:
        q = rng.choice(FIRST) + " " + q[0].lower() + q[1:]
    if rng.random() < 0.06:
        q = q + " " + rng.choice(FILLER_TAIL)
    if rng.random() < 0.55:
        q = q.lower()
    q = q.rstrip("?.! ")
    if rng.random() < 0.55:
        q += "?"
    if rng.random() < 0.18:
        words = q.split()
        idx = [i for i, w in enumerate(words) if len(w) >= 5 and not re.search(r"\d", w)]
        if idx:
            i = rng.choice(idx)
            if not any(words[i].lower().strip("?',") in p.lower() for p in protect) or rng.random() < 0.3:
                words[i] = typo(words[i], rng)
            q = " ".join(words)
    return re.sub(r"\s+", " ", q).strip()
