"""Entity and slot extraction engine for School Management queries."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .session_memory import SessionState


@dataclass
class ExtractedEntities:
    students: List[Dict[str, Any]]
    faculty: List[Dict[str, Any]]
    classes: List[str]
    sections: List[Dict[str, Any]]
    subjects: List[Dict[str, Any]]
    weekdays: List[str]
    dates: List[str]
    thresholds: List[float]
    is_pronoun_reference: bool = False
    refers_to_multi_submission: bool = False


# Known Database Entities
KNOWN_STUDENTS = [
    {
        "student_id": 1,
        "user_id": 5,
        "username": "stu_arjun",
        "name": "Arjun Rao",
        "first_name": "Arjun",
        "last_name": "Rao",
        "admission_no": "ADM2026001",
        "roll_no": 1,
        "section_id": 1,
        "section": "Grade 9-A",
        "class_name": "Grade 9",
    },
    {
        "student_id": 2,
        "user_id": 6,
        "username": "stu_meera",
        "name": "Meera Iyer",
        "first_name": "Meera",
        "last_name": "Iyer",
        "admission_no": "ADM2026002",
        "roll_no": 2,
        "section_id": 1,
        "section": "Grade 9-A",
        "class_name": "Grade 9",
    },
    {
        "student_id": 3,
        "user_id": 7,
        "username": "stu_kiran",
        "name": "Kiran Patel",
        "first_name": "Kiran",
        "last_name": "Patel",
        "admission_no": "ADM2026003",
        "roll_no": 1,
        "section_id": 2,
        "section": "Grade 10-A",
        "class_name": "Grade 10",
    },
    {
        "student_id": 4,
        "user_id": 8,
        "username": "stu_divya",
        "name": "Divya Nair",
        "first_name": "Divya",
        "last_name": "Nair",
        "admission_no": "ADM2026004",
        "roll_no": 2,
        "section_id": 2,
        "section": "Grade 10-A",
        "class_name": "Grade 10",
    },
]

KNOWN_FACULTY = [
    {
        "faculty_id": 1,
        "user_id": 2,
        "username": "fac_anita",
        "name": "Anita Sharma",
        "first_name": "Anita",
        "last_name": "Sharma",
        "employee_no": "EMP001",
        "designation": "Senior Teacher",
        "subject": "Mathematics",
        "subject_id": 1,
    },
    {
        "faculty_id": 2,
        "user_id": 3,
        "username": "fac_vijay",
        "name": "Vijay Reddy",
        "first_name": "Vijay",
        "last_name": "Reddy",
        "employee_no": "EMP002",
        "designation": "TGT Science",
        "subject": "Science",
        "subject_id": 2,
    },
    {
        "faculty_id": 3,
        "user_id": 4,
        "username": "fac_sara",
        "name": "Sara Thomas",
        "first_name": "Sara",
        "last_name": "Thomas",
        "employee_no": "EMP003",
        "designation": "TGT English",
        "subject": "English",
        "subject_id": 3,
    },
]

KNOWN_SUBJECTS = [
    {"subject_id": 1, "code": "MATH", "name": "Mathematics", "aliases": ["math", "maths", "mathematics", "algebra", "geometry"]},
    {"subject_id": 2, "code": "SCI", "name": "Science", "aliases": ["science", "physics", "sci"]},
    {"subject_id": 3, "code": "ENG", "name": "English", "aliases": ["english", "eng", "essay"]},
]

KNOWN_SECTIONS = [
    {"section_id": 1, "class_name": "Grade 9", "section_name": "A", "full_name": "Grade 9-A", "class_teacher_id": None},
    {"section_id": 2, "class_name": "Grade 10", "section_name": "A", "full_name": "Grade 10-A", "class_teacher_id": 2},
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def extract_entities(text: str, session: Optional[SessionState] = None) -> ExtractedEntities:
    """Extract domain entities from text and resolve pronouns using session memory."""
    lowered = text.lower()
    
    # 1. Students
    matched_students: List[Dict[str, Any]] = []
    for s in KNOWN_STUDENTS:
        if (
            s["name"].lower() in lowered
            or s["first_name"].lower() in lowered
            or s["username"].lower() in lowered
            or s["admission_no"].lower() in lowered
        ):
            matched_students.append(s)
            
    # 2. Faculty
    matched_faculty: List[Dict[str, Any]] = []
    for f in KNOWN_FACULTY:
        if (
            f["name"].lower() in lowered
            or f["first_name"].lower() in lowered
            or f["username"].lower() in lowered
            or f["employee_no"].lower() in lowered
        ):
            matched_faculty.append(f)
            
    # 3. Sections & Classes
    matched_sections: List[Dict[str, Any]] = []
    matched_classes: List[str] = []
    
    if "grade 10-a" in lowered or "10-a" in lowered or "grade 10 a" in lowered:
        matched_sections.append(KNOWN_SECTIONS[1])
        matched_classes.append("Grade 10")
    elif "grade 9-a" in lowered or "9-a" in lowered or "grade 9 a" in lowered:
        matched_sections.append(KNOWN_SECTIONS[0])
        matched_classes.append("Grade 9")
    elif "grade 10" in lowered or "10th" in lowered or "class 10" in lowered:
        matched_classes.append("Grade 10")
        matched_sections.append(KNOWN_SECTIONS[1])
    elif "grade 9" in lowered or "9th" in lowered or "class 9" in lowered:
        matched_classes.append("Grade 9")
        matched_sections.append(KNOWN_SECTIONS[0])
        
    # 4. Subjects
    matched_subjects: List[Dict[str, Any]] = []
    for sub in KNOWN_SUBJECTS:
        for alias in sub["aliases"]:
            if re.search(r"\b" + re.escape(alias) + r"\b", lowered):
                matched_subjects.append(sub)
                break
                
    # 5. Weekdays
    matched_weekdays: List[str] = []
    for day in WEEKDAYS:
        if day.lower() in lowered:
            matched_weekdays.append(day)
            
    # 6. Dates
    matched_dates: List[str] = []
    date_match = re.search(r"\b2026-\d{2}-\d{2}\b", text)
    if date_match:
        matched_dates.append(date_match.group(0))
    elif "2026-11-02" in lowered or "nov 2" in lowered or "november 2" in lowered:
        matched_dates.append("2026-11-02")
    elif "2026-11-03" in lowered or "nov 3" in lowered:
        matched_dates.append("2026-11-03")
    elif "2026-11-04" in lowered or "nov 4" in lowered:
        matched_dates.append("2026-11-04")
        
    # 7. Numeric thresholds (e.g., 75% or 75)
    thresholds: List[float] = []
    pct_match = re.search(r"(?:below|<|less than)\s*(\d+(?:\.\d+)?)\s*%?", lowered)
    if pct_match:
        thresholds.append(float(pct_match.group(1)))
    else:
        num_pct = re.findall(r"\b(\d+(?:\.\d+)?)\s*%", lowered)
        for p in num_pct:
            thresholds.append(float(p))
            
    # 8. Check for pronoun and conversational follow-ups
    is_pronoun = False
    refers_multi = False
    
    pronoun_pattern = re.search(r"\b(his|her|their|him|he|she|that student|that boy|that girl)\b", lowered)
    if pronoun_pattern and not matched_students and session and session.last_student_id:
        is_pronoun = True
        for s in KNOWN_STUDENTS:
            if s["student_id"] == session.last_student_id:
                matched_students.append(s)
                break
                
    # Pronoun for section or class (e.g., "that class", "that section", "who hasn't submitted them yet")
    if (
        ("that class" in lowered or "that section" in lowered or "who hasn't submitted" in lowered or "who has not submitted" in lowered or "pending submission" in lowered)
        and not matched_sections
        and session
        and session.last_section_id
    ):
        for sec in KNOWN_SECTIONS:
            if sec["section_id"] == session.last_section_id:
                matched_sections.append(sec)
                matched_classes.append(sec["class_name"])
                break
                
    if "submitted them" in lowered or "submit them" in lowered or "submitted both" in lowered:
        refers_multi = True
        
    return ExtractedEntities(
        students=matched_students,
        faculty=matched_faculty,
        classes=matched_classes,
        sections=matched_sections,
        subjects=matched_subjects,
        weekdays=matched_weekdays,
        dates=matched_dates,
        thresholds=thresholds,
        is_pronoun_reference=is_pronoun,
        refers_to_multi_submission=refers_multi,
    )

