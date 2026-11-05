"""User Manual loader + keyword retriever for the School Management manual.

Priority of sources at runtime:
1. backend/data/manual_extracted.txt  (text produced by the ingestion run)
2. the original PDF at the workspace root (parsed with pypdf, pdfplumber fallback)
3. THIS FILE's embedded fallback copy (never fails)

The manual is split into numbered sections (``1. Introduction``,
``6.4a Attendance reports`` ...) and searchable with a lightweight TF-IDF
style scorer - good enough for topical retrieval without external services.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import MANUAL_PDF_PATH, MANUAL_TXT_PATH, DATA_DIR

# --------------------------------------------------------------------------
# Embedded fallback (generated from the source PDF during ingestion)
# --------------------------------------------------------------------------
FALLBACK_SECTIONS: List[tuple] = [
    (
        "1. Introduction",
        "This manual explains how to use the School Management database. The database "
        "stores everything a school needs to run day to day: users and roles, classes and "
        "sections, subjects, classrooms, timetables, student and faculty attendance, leave "
        "requests, homework, assignments, exams, marks and progress reports. The database "
        "supports three roles: Admin (sets up classes, subjects, timetables and exams; views "
        "attendance, exam and progress reports; approves or rejects leave requests), Faculty "
        "(teaches assigned subjects and sections; marks student attendance, assigns homework "
        "and assignments, enters marks, applies for leave) and Student (views own course "
        "details, timetable, attendance, homework, assignments, exam schedule, marks and "
        "progress report; applies for leave).",
    ),
    (
        "10.1 Foreign keys and insert order",
        "A row can only refer to a row that already exists. Insert parent data before child "
        "data in this order: (1) roles, academic_years, classes, subjects, classrooms, periods, "
        "leave_types, exam_types, grade_scale; (2) users; (3) admins, faculty; (4) sections; "
        "(5) students; (6) course_assignments, timetable; (7) attendance, leave_requests, "
        "homework, assignments; (8) homework_submissions, assignment_submissions; (9) exams, "
        "exam_schedules; (10) marks; (11) progress_reports, progress_report_items; "
        "(12) notifications. Rows cannot be deleted while other rows still refer to them "
        "(no cascading deletes are defined). Deactivate users instead (is_active = FALSE) "
        "to preserve history.",
    ),
    (
        "11.6 Year-end routine",
        "Year-end routine: 1. Insert the new row in academic_years and set the old one to "
        "is_current = FALSE. 2. Create the new year's sections. 3. Move students to their new "
        "sections: UPDATE students SET section_id = <new section> WHERE student_id = <id>. "
        "4. Create new course_assignments, timetable rows and exams. Past attendance, marks "
        "and reports stay in the database for history.",
    ),
]


@dataclass
class ManualSection:
    section_id: str
    title: str
    text: str


HeADING_RE = re.compile(
    r"^(?P<num>\d+(?:\.\w+)+|(?P<top>\d+))\s+(?P<title>[A-Z][^\n]{0,120})$"
)


def _load_text() -> str:
    if MANUAL_TXT_PATH.is_file():
        return MANUAL_TXT_PATH.read_text(encoding="utf-8")
    if MANUAL_PDF_PATH.is_file():
        return _pdf_text(MANUAL_PDF_PATH)
    return "\n\n".join(f"{t}\n{body}" for t, body in FALLBACK_SECTIONS)


def _pdf_text(pdf_path: Path) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                return "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            return ""


def parse_sections(text: str) -> List[ManualSection]:
    """Split manual text into numbered sections, dropping page furniture."""
    lines = text.splitlines()
    kept: List[str] = []
    for line in lines:
        s = line.strip()
        # Skip running headers / footers produced by the PDF export.
        if re.fullmatch(r"School Management Database - User Manual", s, re.IGNORECASE):
            continue
        if re.fullmatch(r"Page \d+ of 24", s, re.IGNORECASE):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)

    sections: List[ManualSection] = []
    current_id = ""
    current_title = ""
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body or current_id:
            sections.append(ManualSection(current_id, current_title, body))
        buffer = []

    for line in cleaned.splitlines():
        s = line.strip()
        m = HeADING_RE.match(s)
        is_heading = False
        if m and len(s) <= 120:
            num = m.group("num")
            title = m.group("title").strip()
            # Filter false positives such as a numbered list inside prose.
            is_heading = title[:1].isupper() and not title.startswith(("Show", "Use"))
        if is_heading:
            flush()
            current_id = num
            current_title = f"{num} {title}"
        else:
            buffer.append(line)
    flush()

    result = list(sections)
    return result


class ManualIndex:
    def __init__(self) -> None:
        self.sections: List[ManualSection] = parse_sections(_load_text())
        self._idf: dict[str, float] = {}
        self._tf: List[dict[str, int]] = [self._tokenize(s.text + " " + s.title) for s in self.sections]
        docs = len(self._tf)
        for doc in self._tf:
            for tok in doc:
                self._idf.setdefault(tok, 0)
        for tok in list(self._idf):
            doc_freq = sum(1 for d in self._tf if tok in d)
            self._idf[tok] = 1.0 + math.log(docs / max(doc_freq, 1)) if docs else 1.0

    @staticmethod
    def _tokenize(text: str) -> dict[str, int]:
        tokens = re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def search(self, query: str, top_k: int = 3) -> List[ManualSection]:
        q = self._tokenize(query)
        scores: List[float] = []
        for i, doc in enumerate(self._tf):
            score = 0.0
            for tok, cnt in q.items():
                if tok in doc:
                    score += min(cnt, 3) * self._idf.get(tok, 1.0)
            # boost title hits
            title_tokens = self._tokenize(self.sections[i].title)
            for tok in q:
                if tok in title_tokens:
                    score += 2.0
            scores.append(score)
        ranked = sorted(range(len(self.sections)), key=lambda i: -scores[i])
        hits = [i for i in ranked if scores[i] > 0.0][:top_k]
        if not hits and self.sections:
            hits = ranked[:1]
        return [self.sections[i] for i in hits]


_index: Optional[ManualIndex] = None


def get_manual_index() -> ManualIndex:
    global _index
    if _index is None:
        _index = ManualIndex()
    return _index


def lookup_manual(topic_or_query: str) -> dict:
    """Retrieve the most relevant manual sections for a topic/question."""
    idx = get_manual_index()
    hits = idx.search(topic_or_query, top_k=3)
    refs = []
    for h in hits:
        body = re.sub(r"\n{2,}", "\n", h.text)
        refs.append(
            {
                "section": (f"Section {h.section_id} - {h.title}" if h.section_id else h.title),
                "text": body[:2000],
            }
        )
    return {"query": topic_or_query, "sections": refs, "total_sections": len(idx.sections)}
