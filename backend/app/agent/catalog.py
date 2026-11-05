"""Live catalog of the entities that exist in the database.

Nothing here is hard-coded: people, classes, sections, subjects, rooms, exams and
homework/assignment titles are read from the `school` schema and refreshed
periodically, so the assistant stays correct when the data changes.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..database import run_readonly_queries

CATALOG_TTL_SECONDS = 120


@dataclass(frozen=True)
class Person:
    kind: str  # student | faculty | admin
    id: int  # student_id / faculty_id / admin_id
    user_id: int
    first: str
    last: str
    username: str
    email: str
    code: str  # admission_no / employee_no / ""
    section_id: Optional[int] = None  # students only

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}".strip()


@dataclass(frozen=True)
class Section:
    section_id: int
    class_id: int
    class_name: str
    letter: str
    teacher_id: Optional[int]

    @property
    def label(self) -> str:
        return f"{self.class_name}-{self.letter}"


@dataclass(frozen=True)
class Klass:
    class_id: int
    name: str
    number: Optional[int]
    section_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Subject:
    subject_id: int
    code: str
    name: str
    description: str


@dataclass(frozen=True)
class Room:
    room_id: int
    name: str
    room_type: str


@dataclass(frozen=True)
class Exam:
    exam_id: int
    name: str
    class_id: int
    type_name: str


@dataclass(frozen=True)
class LeaveType:
    leave_type_id: int
    name: str


@dataclass(frozen=True)
class Titled:
    kind: str  # homework | assignment
    item_id: int
    title: str
    section_id: int


@dataclass
class Catalog:
    students: List[Person] = field(default_factory=list)
    faculty: List[Person] = field(default_factory=list)
    admins: List[Person] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    classes: List[Klass] = field(default_factory=list)
    subjects: List[Subject] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    exams: List[Exam] = field(default_factory=list)
    exam_types: List[str] = field(default_factory=list)
    leave_types: List[LeaveType] = field(default_factory=list)
    titles: List[Titled] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    faculty_subjects: Dict[int, List[str]] = field(default_factory=dict)
    academic_year: Dict[str, str] = field(default_factory=dict)
    schema_columns: Dict[str, List[str]] = field(default_factory=dict)
    loaded_at: float = 0.0

    # ---- lookups --------------------------------------------------------
    @property
    def people(self) -> List[Person]:
        return self.students + self.faculty + self.admins

    def section(self, section_id: int) -> Optional[Section]:
        return next((s for s in self.sections if s.section_id == section_id), None)

    def klass(self, class_id: int) -> Optional[Klass]:
        return next((c for c in self.classes if c.class_id == class_id), None)

    def student(self, student_id: int) -> Optional[Person]:
        return next((p for p in self.students if p.id == student_id), None)

    def faculty_member(self, faculty_id: int) -> Optional[Person]:
        return next((p for p in self.faculty if p.id == faculty_id), None)

    def section_label(self, section_id: Optional[int]) -> str:
        s = self.section(section_id) if section_id is not None else None
        return s.label if s else ""


def load_catalog() -> Catalog:
    """Load everything in one batched, read-only round trip."""
    name_sql = "u.first_name AS first, COALESCE(u.last_name, '') AS last, u.username, u.email, u.user_id"
    queries = {
        "students": f"SELECT s.student_id AS id, s.admission_no AS code, s.section_id, {name_sql} "
        "FROM school.students s JOIN school.users u ON u.user_id = s.user_id ORDER BY s.student_id",
        "faculty": f"SELECT f.faculty_id AS id, f.employee_no AS code, {name_sql} "
        "FROM school.faculty f JOIN school.users u ON u.user_id = f.user_id ORDER BY f.faculty_id",
        "admins": f"SELECT a.admin_id AS id, '' AS code, {name_sql} "
        "FROM school.admins a JOIN school.users u ON u.user_id = a.user_id ORDER BY a.admin_id",
        "sections": "SELECT sec.section_id, sec.class_id, c.class_name, sec.section_name, sec.class_teacher_id "
        "FROM school.sections sec JOIN school.classes c ON c.class_id = sec.class_id "
        "ORDER BY c.class_name, sec.section_name",
        "classes": "SELECT class_id, class_name FROM school.classes ORDER BY class_id",
        "subjects": "SELECT subject_id, subject_code, subject_name, description FROM school.subjects ORDER BY subject_id",
        "rooms": "SELECT classroom_id, room_name, room_type FROM school.classrooms ORDER BY classroom_id",
        "exam_types": "SELECT type_name FROM school.exam_types ORDER BY exam_type_id",
        "exams": "SELECT e.exam_id, e.exam_name, e.class_id, t.type_name FROM school.exams e "
        "JOIN school.exam_types t ON t.exam_type_id = e.exam_type_id ORDER BY e.exam_id",
        "leave_types": "SELECT leave_type_id, type_name FROM school.leave_types ORDER BY leave_type_id",
        "roles": "SELECT role_name FROM school.roles ORDER BY role_id",
        "homework": "SELECT homework_id AS item_id, title, section_id FROM school.homework ORDER BY homework_id",
        "assignments": "SELECT assignment_id AS item_id, title, section_id FROM school.assignments ORDER BY assignment_id",
        "faculty_subjects": "SELECT ca.faculty_id, array_agg(DISTINCT sub.subject_name ORDER BY sub.subject_name) AS subjects "
        "FROM school.course_assignments ca JOIN school.subjects sub ON sub.subject_id = ca.subject_id "
        "GROUP BY ca.faculty_id",
        "year": "SELECT year_name, start_date, end_date FROM school.academic_years "
        "WHERE is_current ORDER BY start_date DESC LIMIT 1",
        "columns": "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'school' ORDER BY table_name, ordinal_position",
    }
    keys = list(queries)
    results = run_readonly_queries([(queries[k], None) for k in keys])
    rows = {k: res["rows"] for k, res in zip(keys, results)}

    cat = Catalog()
    for r in rows["students"]:
        cat.students.append(Person("student", r["id"], r["user_id"], r["first"], r["last"], r["username"], r["email"], r["code"], r["section_id"]))
    for r in rows["faculty"]:
        cat.faculty.append(Person("faculty", r["id"], r["user_id"], r["first"], r["last"], r["username"], r["email"], r["code"]))
    for r in rows["admins"]:
        cat.admins.append(Person("admin", r["id"], r["user_id"], r["first"], r["last"], r["username"], r["email"], r["code"]))

    cat.sections = [Section(r["section_id"], r["class_id"], r["class_name"], r["section_name"], r["class_teacher_id"]) for r in rows["sections"]]
    for r in rows["classes"]:
        m = re.search(r"\d+", r["class_name"])
        ids = tuple(s.section_id for s in cat.sections if s.class_id == r["class_id"])
        cat.classes.append(Klass(r["class_id"], r["class_name"], int(m.group()) if m else None, ids))

    cat.subjects = [Subject(r["subject_id"], r["subject_code"], r["subject_name"], r["description"] or "") for r in rows["subjects"]]
    cat.rooms = [Room(r["classroom_id"], r["room_name"], r["room_type"]) for r in rows["rooms"]]
    cat.exam_types = [r["type_name"] for r in rows["exam_types"]]
    cat.exams = [Exam(r["exam_id"], r["exam_name"], r["class_id"], r["type_name"]) for r in rows["exams"]]
    cat.leave_types = [LeaveType(r["leave_type_id"], r["type_name"]) for r in rows["leave_types"]]
    cat.roles = [r["role_name"] for r in rows["roles"]]
    cat.titles = [Titled("homework", r["item_id"], r["title"], r["section_id"]) for r in rows["homework"]]
    cat.titles += [Titled("assignment", r["item_id"], r["title"], r["section_id"]) for r in rows["assignments"]]
    cat.faculty_subjects = {r["faculty_id"]: list(r["subjects"]) for r in rows["faculty_subjects"]}
    if rows["year"]:
        y = rows["year"][0]
        cat.academic_year = {"name": y["year_name"], "start": y["start_date"], "end": y["end_date"]}
    for r in rows["columns"]:
        cat.schema_columns.setdefault(r["table_name"], []).append(r["column_name"])

    cat.loaded_at = time.time()
    return cat


_catalog: Optional[Catalog] = None
_lock = threading.Lock()


def get_catalog(force: bool = False) -> Catalog:
    """Return the cached catalog, refreshing it when stale. Keeps serving the old
    copy if a refresh fails (e.g. a transient database hiccup)."""
    global _catalog
    with _lock:
        stale = _catalog is None or force or (time.time() - _catalog.loaded_at) > CATALOG_TTL_SECONDS
        if stale:
            try:
                _catalog = load_catalog()
            except Exception:
                if _catalog is None:
                    raise
        return _catalog  # type: ignore[return-value]
