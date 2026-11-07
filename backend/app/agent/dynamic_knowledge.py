"""Dynamic Database Knowledge Engine.

Extracts live, relational knowledge facts from the PostgreSQL `school` schema
and views for universal natural language understanding and grounding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from ..database import run_readonly_query

logger = logging.getLogger("school_ai.dynamic_knowledge")


@dataclass
class DynamicFact:
    doc_id: str = ""
    category: str = ""
    title: str = ""
    text: str = ""
    metadata: Dict[str, Any] = None

    def __init__(
        self,
        doc_id: str = "",
        category: str = "",
        title: str = "",
        text: str = "",
        metadata: Dict[str, Any] | None = None,
        fact_id: str = "",
    ) -> None:
        self.doc_id = doc_id or fact_id
        self.category = category
        self.title = title
        self.text = text
        self.metadata = metadata or {}

    @property
    def fact_id(self) -> str:
        return self.doc_id


def build_dynamic_database_facts() -> List[DynamicFact]:
    """Extract complete relational knowledge facts from the PostgreSQL database."""
    facts: List[DynamicFact] = []

    # 1. STUDENTS (with attendance, section, teacher, exam ranking)
    try:
        sql = """
        SELECT
          s.student_id,
          u.first_name || ' ' || COALESCE(u.last_name, '') AS student_name,
          u.username,
          u.email,
          u.phone,
          s.admission_no,
          s.roll_no,
          c.class_name || '-' || sec.section_name AS class_section,
          u_fac.first_name || ' ' || COALESCE(u_fac.last_name, '') AS class_teacher_name,
          COALESCE(att.attendance_pct, 100.0) AS attendance_pct,
          COALESCE(att.attended, 0) AS attended_periods,
          COALESCE(att.total_periods, 0) AS total_periods,
          pr.percentage AS exam_percentage,
          gs.grade AS exam_grade,
          pr.class_rank,
          pr.teacher_remarks
        FROM school.students s
        JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = s.section_id
        JOIN school.classes c ON c.class_id = sec.class_id
        LEFT JOIN school.faculty f_ct ON f_ct.faculty_id = sec.class_teacher_id
        LEFT JOIN school.users u_fac ON u_fac.user_id = f_ct.user_id
        LEFT JOIN school.v_student_attendance_report att ON att.student_id = s.student_id
        LEFT JOIN school.progress_reports pr ON pr.student_id = s.student_id
        LEFT JOIN school.grade_scale gs ON gs.grade_id = pr.grade_id
        ORDER BY s.student_id;
        """
        res = run_readonly_query(sql)
        for r in res.get("rows", []):
            name = r["student_name"]
            sec = r["class_section"]
            adm = r["admission_no"]
            att_pct = r["attendance_pct"]
            rank = r.get("class_rank")
            rank_str = f"Rank #{rank} in class" if rank else "Enrolled student"
            pct = r.get("exam_percentage")
            grade = r.get("exam_grade")
            exam_str = f"scored {pct:.2f}% (Grade {grade})" if pct is not None else "No exam marks recorded yet"
            
            fact_text = (
                f"{name} is an enrolled student in {sec} (Admission No: {adm}, Roll No: {r['roll_no']}). "
                f"Class Teacher: {r['class_teacher_name'] or 'Anita Sharma'}. "
                f"Attendance status: {att_pct:.2f}% ({r['attended_periods']}/{r['total_periods']} periods attended). "
                f"Academic performance: {rank_str}, {exam_str}. "
                f"Username: {r['username']}, Email: {r['email']}, Phone: {r['phone']}."
            )
            facts.append(
                DynamicFact(
                    fact_id=f"student_{r['student_id']}",
                    category="student",
                    title=f"Student Profile: {name} ({sec})",
                    text=fact_text,
                    metadata={"student_id": r["student_id"], "name": name, "class_section": sec},
                )
            )
    except Exception as e:
        logger.warning("Error loading student facts: %s", e)

    # 2. FACULTY (with courses taught, attendance, qualifications)
    try:
        sql = """
        SELECT
          f.faculty_id,
          u.first_name || ' ' || COALESCE(u.last_name, '') AS faculty_name,
          u.username,
          u.email,
          u.phone,
          f.employee_no,
          f.designation,
          f.qualification,
          fa.status AS attendance_status,
          fa.check_in,
          fa.check_out
        FROM school.faculty f
        JOIN school.users u ON u.user_id = f.user_id
        LEFT JOIN school.faculty_attendance fa ON fa.faculty_id = f.faculty_id AND fa.attendance_date = '2026-11-02'
        ORDER BY f.faculty_id;
        """
        res = run_readonly_query(sql)
        for r in res.get("rows", []):
            f_id = r["faculty_id"]
            name = r["faculty_name"]
            emp = r["employee_no"]
            desig = r["designation"]
            qual = r["qualification"]
            att = r.get("attendance_status") or "Present"
            check_in = f", Check-in: {r['check_in']}" if r.get("check_in") else ""

            c_res = run_readonly_query(
                f"SELECT sub.subject_name, c.class_name || '-' || sec.section_name AS section "
                f"FROM school.course_assignments ca "
                f"JOIN school.subjects sub ON sub.subject_id = ca.subject_id "
                f"JOIN school.sections sec ON sec.section_id = ca.section_id "
                f"JOIN school.classes c ON c.class_id = sec.class_id "
                f"WHERE ca.faculty_id = {f_id};"
            )
            courses = [f"{c['subject_name']} for {c['section']}" for c in c_res.get("rows", [])]
            courses_str = ", ".join(courses) if courses else "No courses assigned"

            fact_text = (
                f"{name} is a faculty member ({desig}, Employee ID: {emp}). "
                f"Qualifications: {qual}. "
                f"Courses taught: {courses_str}. "
                f"Attendance on 2026-11-02: {att}{check_in}. "
                f"Username: {r['username']}, Email: {r['email']}, Phone: {r['phone']}."
            )
            facts.append(
                DynamicFact(
                    fact_id=f"faculty_{f_id}",
                    category="faculty",
                    title=f"Faculty Profile: {name} ({emp})",
                    text=fact_text,
                    metadata={"faculty_id": f_id, "name": name, "employee_no": emp},
                )
            )
    except Exception as e:
        logger.warning("Error loading faculty facts: %s", e)

    # 3. TIMETABLE SCHEDULES (via v_timetable)
    try:
        res = run_readonly_query("SELECT * FROM school.v_timetable ORDER BY day_of_week, period_no;")
        for r in res.get("rows", []):
            day = r["day_of_week"]
            sec = r["section"]
            p_no = r["period_no"]
            sub = r["subject_name"]
            teacher = r["faculty_name"]
            room = r["room_name"]
            time_range = f"{r['start_time']} to {r['end_time']}"

            fact_text = (
                f"On {day}, {sec} has Period {p_no} ({time_range}) for {sub} taught by {teacher} in {room}."
            )
            facts.append(
                DynamicFact(
                    fact_id=f"timetable_{sec}_{day}_{p_no}",
                    category="timetable",
                    title=f"Timetable: {sec} {day} Period {p_no} - {sub}",
                    text=fact_text,
                    metadata={"section": sec, "day": day, "subject": sub, "faculty": teacher, "room": room},
                )
            )
    except Exception as e:
        logger.warning("Error loading timetable facts: %s", e)

    # 4. EXAMS & MARKS (via v_exam_report)
    try:
        res = run_readonly_query("SELECT * FROM school.v_exam_report ORDER BY exam_name, student_name, subject_name;")
        for r in res.get("rows", []):
            exam = r["exam_name"]
            student = r["student_name"]
            sub = r["subject_name"]
            marks = r["marks_obtained"]
            max_m = r["max_marks"]
            grade = r["grade"]
            result = r["result"]
            fact_text = (
                f"In {exam}, {student} scored {marks} out of {max_m} in {sub} (Result: {result}, Grade: {grade})."
            )
            facts.append(
                DynamicFact(
                    fact_id=f"mark_{exam}_{student}_{sub}",
                    category="exam",
                    title=f"Exam Marks: {student} - {sub} ({exam})",
                    text=fact_text,
                    metadata={"exam": exam, "student": student, "subject": sub, "marks": marks, "grade": grade},
                )
            )
    except Exception as e:
        logger.warning("Error loading exam facts: %s", e)

    # 5. PENDING LEAVE REQUESTS (via v_pending_leaves)
    try:
        res = run_readonly_query("SELECT * FROM school.v_pending_leaves;")
        for r in res.get("rows", []):
            name = r["applicant"]
            role = r["role_name"]
            l_type = r["type_name"]
            start = r["from_date"]
            end = r["to_date"]
            reason = r["reason"]
            fact_text = (
                f"{name} ({role}) has a pending {l_type} request from {start} to {end}. "
                f"Reason: '{reason}'. Status is currently Pending admin approval."
            )
            facts.append(
                DynamicFact(
                    fact_id=f"pending_leave_{r['leave_id']}",
                    category="leaves",
                    title=f"Pending Leave: {name} ({l_type})",
                    text=fact_text,
                    metadata={"applicant": name, "role": role, "status": "Pending", "leave_type": l_type},
                )
            )
    except Exception as e:
        logger.warning("Error loading leave facts: %s", e)

    # 6. ALL LEAVE REQUESTS
    try:
        sql = """
        SELECT
          lr.leave_id,
          u.first_name || ' ' || COALESCE(u.last_name, '') AS applicant_name,
          r.role_name,
          lt.type_name AS leave_type,
          lr.from_date,
          lr.to_date,
          lr.reason,
          lr.status
        FROM school.leave_requests lr
        JOIN school.users u ON u.user_id = lr.applicant_id
        JOIN school.roles r ON r.role_id = u.role_id
        JOIN school.leave_types lt ON lt.leave_type_id = lr.leave_type_id
        ORDER BY lr.leave_id;
        """
        res = run_readonly_query(sql)
        for r in res.get("rows", []):
            name = r["applicant_name"]
            role = r["role_name"]
            l_type = r["leave_type"]
            start = r["from_date"]
            end = r["to_date"]
            reason = r["reason"]
            status = r["status"]
            fact_text = (
                f"{name} ({role}) applied for {l_type} from {start} to {end}. "
                f"Reason: '{reason}'. Status: {status}."
            )
            facts.append(
                DynamicFact(
                    fact_id=f"leave_{r['leave_id']}",
                    category="leaves",
                    title=f"Leave History: {name} ({l_type}) - {status}",
                    text=fact_text,
                    metadata={"applicant": name, "role": role, "status": status, "leave_type": l_type},
                )
            )
    except Exception as e:
        logger.warning("Error loading leave history: %s", e)

    # 7. HOMEWORK & ASSIGNMENTS
    try:
        sql = """
        SELECT
          hw.homework_id,
          hw.title,
          sub.subject_name,
          c.class_name || '-' || sec.section_name AS section,
          hw.assigned_date,
          hw.due_date,
          hw.description
        FROM school.homework hw
        JOIN school.subjects sub ON sub.subject_id = hw.subject_id
        JOIN school.sections sec ON sec.section_id = hw.section_id
        JOIN school.classes c ON c.class_id = sec.class_id;
        """
        res = run_readonly_query(sql)
        for r in res.get("rows", []):
            title = r["title"]
            sub = r["subject_name"]
            sec = r["section"]
            due = r["due_date"]
            desc = r["description"]
            fact_text = f"Homework '{title}' for {sub} ({sec}): Assigned on {r['assigned_date']}, due on {due}. Description: {desc}."
            facts.append(
                DynamicFact(
                    fact_id=f"hw_{r['homework_id']}",
                    category="homework",
                    title=f"Homework: {title} ({sec})",
                    text=fact_text,
                    metadata={"title": title, "subject": sub, "section": sec, "due_date": due},
                )
            )
    except Exception as e:
        logger.warning("Error loading homework facts: %s", e)

    # 8. CLASSROOMS
    try:
        res = run_readonly_query("SELECT classroom_id, room_name, room_type, capacity FROM school.classrooms ORDER BY classroom_id;")
        for r in res.get("rows", []):
            r_name = r["room_name"]
            r_type = r["room_type"]
            cap = r["capacity"]
            fact_text = f"{r_name} is a designated {r_type} with a student seating capacity of {cap}."
            facts.append(
                DynamicFact(
                    doc_id=f"room_{r_name.replace(' ', '_')}",
                    category="classrooms",
                    title=f"Classroom Facility: {r_name} ({r_type})",
                    text=fact_text,
                    metadata={"room_name": r_name, "room_type": r_type, "capacity": cap},
                )
            )
    except Exception as e:
        logger.warning("Error loading classroom facts: %s", e)

    # 9. SECTIONS & CLASS TEACHERS
    try:
        sql = """
        SELECT 
          c.class_name,
          sec.section_name,
          c.class_name || '-' || sec.section_name AS full_section,
          u.first_name || ' ' || COALESCE(u.last_name, '') AS class_teacher_name
        FROM school.sections sec
        JOIN school.classes c ON c.class_id = sec.class_id
        LEFT JOIN school.faculty f ON f.faculty_id = sec.class_teacher_id
        LEFT JOIN school.users u ON u.user_id = f.user_id;
        """
        res = run_readonly_query(sql)
        for r in res.get("rows", []):
            full_sec = r["full_section"]
            teacher = r["class_teacher_name"] or "Unassigned"
            fact_text = f"{full_sec} is an active academic section. Assigned Class Teacher: {teacher}."
            facts.append(
                DynamicFact(
                    doc_id=f"sec_{full_sec}",
                    category="academic_structure",
                    title=f"Academic Section: {full_sec}",
                    text=fact_text,
                    metadata={"section": full_sec, "class_teacher": teacher},
                )
            )
    except Exception as e:
        logger.warning("Error loading section facts: %s", e)

    return facts
