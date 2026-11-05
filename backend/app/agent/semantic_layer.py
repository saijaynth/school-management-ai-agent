"""Semantic layer: wide, pre-joined, read-only views over the 29 `school` tables.

A small seq2seq model can reliably write ``SELECT ... FROM one_wide_view WHERE ...`` but not
five-way joins, so the model targets these views instead of the raw tables. They are never
created in the database (it is read-only): the executor injects the definitions of just the
views a generated query references as a ``WITH`` prefix.

This file is *schema modelling*, not per-question logic: each view says what it contains, which
columns exist, what kind of value each holds, and how people refer to it. The dataset generator,
the NLG layer and the guard all read this single source of truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

NAME = "{a}.first_name || ' ' || COALESCE({a}.last_name, '')"


def _name(alias: str) -> str:
    return NAME.format(a=alias)


@dataclass
class Col:
    name: str
    kind: str  # student|faculty|person|section|class|subject|weekday|date|time|number|pct|enum|bool|title|room|text
    phrases: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class View:
    name: str
    nouns: List[str]  # plural nouns people use ("students", "pupils")
    sql: str
    cols: List[Col]
    show: List[str]  # default columns for "list" questions
    topic: str  # short human label used in clarification options
    scope: List[str] = field(default_factory=list)  # columns one of which is needed (else ask)
    sort: Optional[str] = None
    singular: str = ""

    def col(self, name: str) -> Optional[Col]:
        for c in self.cols:
            if c.name == name:
                return c
        return None

    @property
    def colnames(self) -> List[str]:
        return [c.name for c in self.cols]


def C(name: str, kind: str, *phrases: str) -> Col:
    return Col(name, kind, list(phrases))


VIEWS: List[View] = [
    View(
        "sv_students", ["students", "pupils"],
        f"""SELECT {_name('u')} AS student_name, s.admission_no, s.roll_no,
               c.class_name || '-' || sec.section_name AS section, c.class_name, s.gender, s.date_of_birth,
               s.guardian_name, s.guardian_phone, s.address, s.admission_date, u.email, u.phone, u.username,
               u.is_active, {_name('tu')} AS class_teacher
        FROM school.students s JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = s.section_id
        JOIN school.classes c ON c.class_id = sec.class_id
        LEFT JOIN school.faculty ft ON ft.faculty_id = sec.class_teacher_id
        LEFT JOIN school.users tu ON tu.user_id = ft.user_id""",
        [C("student_name", "student", "name"), C("admission_no", "text", "admission number", "admission no"),
         C("roll_no", "number", "roll number", "roll no"), C("section", "section", "class", "section", "grade"),
         C("class_name", "class", "grade"), C("gender", "enum", "gender"), C("date_of_birth", "date", "date of birth", "birthday", "dob"),
         C("guardian_name", "text", "guardian", "parent", "guardian name"), C("guardian_phone", "text", "guardian phone", "parent phone number"),
         C("address", "text", "address"), C("admission_date", "date", "admission date", "joining date"),
         C("email", "text", "email", "email address"), C("phone", "text", "phone number", "mobile"), C("username", "text", "username"),
         C("is_active", "bool", "active status"), C("class_teacher", "faculty", "class teacher")],
        ["student_name", "section", "roll_no", "admission_no"], "Student details", sort="section, roll_no", singular="student",
    ),
    View(
        "sv_faculty", ["teachers", "faculty members", "faculty", "staff"],
        f"""SELECT {_name('u')} AS faculty_name, f.employee_no, f.designation, f.qualification, f.joining_date,
               u.email, u.phone, u.username, u.is_active,
               (SELECT string_agg(DISTINCT sub.subject_name, ', ') FROM school.course_assignments ca
                  JOIN school.subjects sub ON sub.subject_id = ca.subject_id WHERE ca.faculty_id = f.faculty_id) AS subjects_taught,
               (SELECT string_agg(c.class_name || '-' || sec.section_name, ', ') FROM school.sections sec
                  JOIN school.classes c ON c.class_id = sec.class_id WHERE sec.class_teacher_id = f.faculty_id) AS class_teacher_of
        FROM school.faculty f JOIN school.users u ON u.user_id = f.user_id""",
        [C("faculty_name", "faculty", "name"), C("employee_no", "text", "employee number", "employee id"),
         C("designation", "text", "designation", "position"), C("qualification", "text", "qualification", "degree"),
         C("joining_date", "date", "joining date", "date of joining"), C("email", "text", "email", "email address"),
         C("phone", "text", "phone number", "mobile"), C("username", "text", "username"), C("is_active", "bool", "active status"),
         C("subjects_taught", "subject", "subjects", "subject they teach"), C("class_teacher_of", "section", "class teacher of", "class in charge")],
        ["faculty_name", "employee_no", "designation", "subjects_taught"], "Teacher details", sort="employee_no", singular="teacher",
    ),
    View(
        "sv_users", ["users", "accounts", "people"],
        f"""SELECT {_name('u')} AS full_name, u.username, r.role_name AS role,
               COALESCE(a.designation, f.designation, r.role_name) AS designation, u.email, u.phone, u.is_active, u.last_login
        FROM school.users u JOIN school.roles r ON r.role_id = u.role_id
        LEFT JOIN school.admins a ON a.user_id = u.user_id LEFT JOIN school.faculty f ON f.user_id = u.user_id""",
        [C("full_name", "person", "name"), C("username", "text", "username"), C("role", "enum", "role"),
         C("designation", "text", "designation", "title"), C("email", "text", "email"), C("phone", "text", "phone number"),
         C("is_active", "bool", "active status"), C("last_login", "date", "last login")],
        ["full_name", "role", "designation", "username"], "Users and roles", sort="role, full_name", singular="user",
    ),
    View(
        "sv_sections", ["classes", "sections", "grades"],
        f"""SELECT c.class_name || '-' || sec.section_name AS section, c.class_name, sec.section_name,
               ay.year_name AS academic_year, {_name('tu')} AS class_teacher,
               (SELECT COUNT(*) FROM school.students st WHERE st.section_id = sec.section_id) AS student_count
        FROM school.sections sec JOIN school.classes c ON c.class_id = sec.class_id
        JOIN school.academic_years ay ON ay.academic_year_id = sec.academic_year_id
        LEFT JOIN school.faculty ft ON ft.faculty_id = sec.class_teacher_id
        LEFT JOIN school.users tu ON tu.user_id = ft.user_id""",
        [C("section", "section", "class", "section", "grade"), C("class_name", "class", "grade"), C("section_name", "text", "section letter"),
         C("academic_year", "text", "academic year"), C("class_teacher", "faculty", "class teacher", "class in-charge"),
         C("student_count", "number", "number of students", "strength")],
        ["section", "class_teacher", "student_count"], "Classes and class teachers", sort="section", singular="class",
    ),
    View(
        "sv_subjects", ["subjects", "courses"],
        f"""SELECT sub.subject_code, sub.subject_name, sub.description,
               (SELECT string_agg(DISTINCT {_name('tu')}, ', ') FROM school.course_assignments ca
                  JOIN school.faculty f ON f.faculty_id = ca.faculty_id JOIN school.users tu ON tu.user_id = f.user_id
                  WHERE ca.subject_id = sub.subject_id) AS teachers
        FROM school.subjects sub""",
        [C("subject_code", "text", "subject code", "code"), C("subject_name", "subject", "name", "subject"),
         C("description", "text", "description", "topics covered", "syllabus"), C("teachers", "faculty", "teachers", "who teaches it")],
        ["subject_name", "subject_code", "teachers"], "Subjects", sort="subject_name", singular="subject",
    ),
    View(
        "sv_course_allocation", ["course allocations", "teaching assignments", "subject allocations"],
        f"""SELECT c.class_name || '-' || sec.section_name AS section, sub.subject_name AS subject,
               {_name('tu')} AS teacher, cr.room_name AS room, ca.periods_per_week, ay.year_name AS academic_year
        FROM school.course_assignments ca JOIN school.sections sec ON sec.section_id = ca.section_id
        JOIN school.classes c ON c.class_id = sec.class_id JOIN school.subjects sub ON sub.subject_id = ca.subject_id
        JOIN school.faculty f ON f.faculty_id = ca.faculty_id JOIN school.users tu ON tu.user_id = f.user_id
        JOIN school.classrooms cr ON cr.classroom_id = ca.classroom_id
        JOIN school.academic_years ay ON ay.academic_year_id = ca.academic_year_id""",
        [C("section", "section", "class", "section"), C("subject", "subject", "subject"), C("teacher", "faculty", "teacher", "faculty"),
         C("room", "room", "room", "classroom"), C("periods_per_week", "number", "periods per week", "weekly periods"),
         C("academic_year", "text", "academic year")],
        ["section", "subject", "teacher", "room", "periods_per_week"], "Who teaches what", sort="section, subject", singular="course allocation",
    ),
    View(
        "sv_timetable", ["timetable", "schedule", "periods", "classes"],
        f"""SELECT c.class_name || '-' || sec.section_name AS section, t.day_of_week, p.period_no, p.start_time, p.end_time,
               sub.subject_name AS subject, {_name('tu')} AS teacher, cr.room_name AS room
        FROM school.timetable t JOIN school.sections sec ON sec.section_id = t.section_id
        JOIN school.classes c ON c.class_id = sec.class_id JOIN school.periods p ON p.period_id = t.period_id
        JOIN school.subjects sub ON sub.subject_id = t.subject_id JOIN school.faculty f ON f.faculty_id = t.faculty_id
        JOIN school.users tu ON tu.user_id = f.user_id JOIN school.classrooms cr ON cr.classroom_id = t.classroom_id""",
        [C("section", "section", "class", "section"), C("day_of_week", "weekday", "day"), C("period_no", "number", "period number", "period"),
         C("start_time", "time", "start time"), C("end_time", "time", "end time"), C("subject", "subject", "subject"),
         C("teacher", "faculty", "teacher"), C("room", "room", "room", "classroom")],
        ["day_of_week", "period_no", "start_time", "end_time", "subject", "teacher", "room"], "Timetable",
        scope=["section", "teacher"], sort="section, CASE day_of_week WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3 WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 ELSE 6 END, period_no",
        singular="period",
    ),
    View(
        "sv_classrooms", ["classrooms", "rooms", "labs"],
        "SELECT room_name AS room, room_type, capacity FROM school.classrooms",
        [C("room", "room", "room", "name"), C("room_type", "enum", "type"), C("capacity", "number", "capacity", "seating capacity")],
        ["room", "room_type", "capacity"], "Classrooms", sort="room", singular="room",
    ),
    View(
        "sv_periods", ["periods", "period timings", "bell schedule"],
        "SELECT period_no, start_time, end_time FROM school.periods",
        [C("period_no", "number", "period number"), C("start_time", "time", "start time"), C("end_time", "time", "end time")],
        ["period_no", "start_time", "end_time"], "Period timings", sort="period_no", singular="period",
    ),
    View(
        "sv_academic_years", ["academic years", "sessions"],
        "SELECT year_name AS academic_year, start_date, end_date, is_current FROM school.academic_years",
        [C("academic_year", "text", "academic year"), C("start_date", "date", "start date"), C("end_date", "date", "end date"),
         C("is_current", "bool", "current")],
        ["academic_year", "start_date", "end_date", "is_current"], "Academic year", sort="start_date", singular="academic year",
    ),
    View(
        "sv_student_attendance", ["student attendance records", "period attendance", "attendance records"],
        f"""SELECT sa.attendance_date, {_name('u')} AS student_name, c.class_name || '-' || sec.section_name AS section,
               p.period_no, sub.subject_name AS subject, sa.status, {_name('mu')} AS marked_by, sa.remarks
        FROM school.student_attendance sa JOIN school.students s ON s.student_id = sa.student_id
        JOIN school.users u ON u.user_id = s.user_id JOIN school.sections sec ON sec.section_id = s.section_id
        JOIN school.classes c ON c.class_id = sec.class_id JOIN school.periods p ON p.period_id = sa.period_id
        JOIN school.subjects sub ON sub.subject_id = sa.subject_id JOIN school.faculty f ON f.faculty_id = sa.marked_by
        JOIN school.users mu ON mu.user_id = f.user_id""",
        [C("attendance_date", "date", "date"), C("student_name", "student", "student", "name"), C("section", "section", "class", "section"),
         C("period_no", "number", "period"), C("subject", "subject", "subject"), C("status", "enum", "status"),
         C("marked_by", "faculty", "marked by", "teacher"), C("remarks", "text", "remarks", "reason")],
        ["attendance_date", "student_name", "period_no", "subject", "status", "remarks"], "Student attendance (period by period)",
        sort="attendance_date, student_name, period_no", singular="attendance record",
    ),
    View(
        "sv_attendance_summary", ["student attendance percentages", "attendance percentages", "attendance summary"],
        f"""SELECT {_name('u')} AS student_name, s.admission_no, c.class_name || '-' || sec.section_name AS section,
               COUNT(*) AS total_periods,
               SUM(CASE WHEN sa.status IN ('Present','Late') THEN 1 ELSE 0 END) AS attended,
               SUM(CASE WHEN sa.status IN ('Absent') THEN 1 ELSE 0 END) AS absent_periods,
               ROUND(100.0 * SUM(CASE WHEN sa.status IN ('Present','Late') THEN 1 ELSE 0 END) / COUNT(*), 2) AS attendance_pct
        FROM school.student_attendance sa JOIN school.students s ON s.student_id = sa.student_id
        JOIN school.users u ON u.user_id = s.user_id JOIN school.sections sec ON sec.section_id = s.section_id
        JOIN school.classes c ON c.class_id = sec.class_id
        GROUP BY s.student_id, s.admission_no, u.first_name, u.last_name, c.class_name, sec.section_name""",
        [C("student_name", "student", "student", "name"), C("admission_no", "text", "admission number"), C("section", "section", "class", "section"),
         C("total_periods", "number", "total periods"), C("attended", "number", "periods attended"),
         C("absent_periods", "number", "periods absent"), C("attendance_pct", "pct", "attendance percentage", "attendance", "attendance rate")],
        ["student_name", "section", "attended", "total_periods", "attendance_pct"], "Student attendance (overall percentage)",
        sort="student_name", singular="attendance summary",
    ),
    View(
        "sv_faculty_attendance", ["faculty attendance records", "teacher attendance", "staff attendance"],
        f"""SELECT fa.attendance_date, {_name('u')} AS faculty_name, f.employee_no, fa.status, fa.check_in, fa.check_out
        FROM school.faculty_attendance fa JOIN school.faculty f ON f.faculty_id = fa.faculty_id
        JOIN school.users u ON u.user_id = f.user_id""",
        [C("attendance_date", "date", "date"), C("faculty_name", "faculty", "teacher", "faculty", "name"), C("employee_no", "text", "employee number"),
         C("status", "enum", "status"), C("check_in", "time", "check in time"), C("check_out", "time", "check out time")],
        ["attendance_date", "faculty_name", "status", "check_in", "check_out"], "Teacher attendance",
        sort="attendance_date, employee_no", singular="faculty attendance record",
    ),
    View(
        "sv_leaves", ["leave requests", "leaves", "leave applications"],
        f"""SELECT {_name('u')} AS applicant, r.role_name AS role, lt.type_name AS leave_type, lr.from_date, lr.to_date,
               (lr.to_date - lr.from_date + 1) AS days, lr.reason, lr.status, lr.applied_on,
               {_name('ru')} AS reviewed_by, lr.reviewed_on, lr.admin_remarks
        FROM school.leave_requests lr JOIN school.users u ON u.user_id = lr.applicant_id
        JOIN school.roles r ON r.role_id = u.role_id JOIN school.leave_types lt ON lt.leave_type_id = lr.leave_type_id
        LEFT JOIN school.users ru ON ru.user_id = lr.reviewed_by""",
        [C("applicant", "person", "applicant", "name", "person"), C("role", "enum", "role"), C("leave_type", "enum", "leave type", "type"),
         C("from_date", "date", "start date", "from date"), C("to_date", "date", "end date", "to date"), C("days", "number", "number of days", "duration"),
         C("reason", "text", "reason"), C("status", "enum", "status"), C("applied_on", "date", "application date"),
         C("reviewed_by", "person", "reviewed by", "approved by"), C("reviewed_on", "date", "review date"), C("admin_remarks", "text", "admin remarks", "remarks")],
        ["applicant", "role", "leave_type", "from_date", "to_date", "reason", "status"], "Leave requests",
        sort="applied_on", singular="leave request",
    ),
    View(
        "sv_leave_types", ["leave types", "leave policies"],
        "SELECT type_name AS leave_type, max_days_year AS max_days_per_year FROM school.leave_types",
        [C("leave_type", "enum", "leave type", "type"), C("max_days_per_year", "number", "maximum days per year", "yearly limit", "allowed days")],
        ["leave_type", "max_days_per_year"], "Leave types", sort="leave_type", singular="leave type",
    ),
    View(
        "sv_homework", ["homework", "homework assignments"],
        f"""SELECT h.title, h.description, c.class_name || '-' || sec.section_name AS section, sub.subject_name AS subject,
               {_name('tu')} AS teacher, h.assigned_date, h.due_date,
               (SELECT COUNT(*) FROM school.homework_submissions hs WHERE hs.homework_id = h.homework_id) AS students_total,
               (SELECT COUNT(*) FROM school.homework_submissions hs WHERE hs.homework_id = h.homework_id AND hs.status <> 'Pending') AS students_submitted
        FROM school.homework h JOIN school.sections sec ON sec.section_id = h.section_id JOIN school.classes c ON c.class_id = sec.class_id
        JOIN school.subjects sub ON sub.subject_id = h.subject_id JOIN school.faculty f ON f.faculty_id = h.faculty_id
        JOIN school.users tu ON tu.user_id = f.user_id""",
        [C("title", "title", "title", "name"), C("description", "text", "description", "details"), C("section", "section", "class", "section"),
         C("subject", "subject", "subject"), C("teacher", "faculty", "teacher", "assigned by"), C("assigned_date", "date", "assigned date"),
         C("due_date", "date", "due date", "deadline"), C("students_total", "number", "number of students"),
         C("students_submitted", "number", "number of submissions", "submitted count")],
        ["title", "section", "subject", "teacher", "assigned_date", "due_date"], "Homework", sort="due_date", singular="homework",
    ),
    View(
        "sv_homework_submissions", ["homework submissions"],
        f"""SELECT h.title AS homework_title, {_name('u')} AS student_name, c.class_name || '-' || sec.section_name AS section,
               sub.subject_name AS subject, h.due_date, hs.status, hs.submitted_on, hs.faculty_remarks
        FROM school.homework_submissions hs JOIN school.homework h ON h.homework_id = hs.homework_id
        JOIN school.students s ON s.student_id = hs.student_id JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = s.section_id JOIN school.classes c ON c.class_id = sec.class_id
        JOIN school.subjects sub ON sub.subject_id = h.subject_id""",
        [C("homework_title", "title", "homework", "title"), C("student_name", "student", "student", "name"), C("section", "section", "class", "section"),
         C("subject", "subject", "subject"), C("due_date", "date", "due date"), C("status", "enum", "status"),
         C("submitted_on", "date", "submission time", "submitted on"), C("faculty_remarks", "text", "teacher remarks", "remarks", "feedback")],
        ["homework_title", "student_name", "status", "submitted_on", "faculty_remarks"], "Homework submissions",
        sort="homework_title, student_name", singular="homework submission",
    ),
    View(
        "sv_assignments", ["assignments", "projects"],
        f"""SELECT a.title, a.description, c.class_name || '-' || sec.section_name AS section, sub.subject_name AS subject,
               {_name('tu')} AS teacher, a.max_marks, a.assigned_date, a.due_date,
               (SELECT COUNT(*) FROM school.assignment_submissions s2 WHERE s2.assignment_id = a.assignment_id) AS students_total,
               (SELECT COUNT(*) FROM school.assignment_submissions s2 WHERE s2.assignment_id = a.assignment_id AND s2.status <> 'Pending') AS students_submitted
        FROM school.assignments a JOIN school.sections sec ON sec.section_id = a.section_id JOIN school.classes c ON c.class_id = sec.class_id
        JOIN school.subjects sub ON sub.subject_id = a.subject_id JOIN school.faculty f ON f.faculty_id = a.faculty_id
        JOIN school.users tu ON tu.user_id = f.user_id""",
        [C("title", "title", "title", "name"), C("description", "text", "description", "details"), C("section", "section", "class", "section"),
         C("subject", "subject", "subject"), C("teacher", "faculty", "teacher", "assigned by"), C("max_marks", "number", "maximum marks", "total marks"),
         C("assigned_date", "date", "assigned date"), C("due_date", "date", "due date", "deadline"), C("students_total", "number", "number of students"),
         C("students_submitted", "number", "number of submissions")],
        ["title", "section", "subject", "teacher", "max_marks", "due_date"], "Assignments", sort="due_date", singular="assignment",
    ),
    View(
        "sv_assignment_submissions", ["assignment submissions"],
        f"""SELECT a.title AS assignment_title, {_name('u')} AS student_name, c.class_name || '-' || sec.section_name AS section,
               sub.subject_name AS subject, a.max_marks, s2.status, s2.submitted_on, s2.marks_obtained, s2.feedback
        FROM school.assignment_submissions s2 JOIN school.assignments a ON a.assignment_id = s2.assignment_id
        JOIN school.students s ON s.student_id = s2.student_id JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = s.section_id JOIN school.classes c ON c.class_id = sec.class_id
        JOIN school.subjects sub ON sub.subject_id = a.subject_id""",
        [C("assignment_title", "title", "assignment", "title"), C("student_name", "student", "student", "name"), C("section", "section", "class", "section"),
         C("subject", "subject", "subject"), C("max_marks", "number", "maximum marks"), C("status", "enum", "status"),
         C("submitted_on", "date", "submission time"), C("marks_obtained", "number", "marks obtained", "marks", "score"), C("feedback", "text", "feedback", "remarks")],
        ["assignment_title", "student_name", "status", "marks_obtained", "max_marks", "feedback"], "Assignment submissions",
        sort="assignment_title, student_name", singular="assignment submission",
    ),
    View(
        "sv_exams", ["exams", "exam schedule", "exam timetable"],
        f"""SELECT e.exam_name, et.type_name AS exam_type, c.class_name, ay.year_name AS academic_year, e.start_date, e.end_date,
               sub.subject_name AS subject, es.exam_date, es.start_time, es.end_time, cr.room_name AS room,
               {_name('iu')} AS invigilator, es.max_marks, es.pass_marks
        FROM school.exams e JOIN school.exam_types et ON et.exam_type_id = e.exam_type_id
        JOIN school.classes c ON c.class_id = e.class_id JOIN school.academic_years ay ON ay.academic_year_id = e.academic_year_id
        JOIN school.exam_schedules es ON es.exam_id = e.exam_id JOIN school.subjects sub ON sub.subject_id = es.subject_id
        JOIN school.classrooms cr ON cr.classroom_id = es.classroom_id
        LEFT JOIN school.faculty f ON f.faculty_id = es.invigilator_id LEFT JOIN school.users iu ON iu.user_id = f.user_id""",
        [C("exam_name", "title", "exam", "name"), C("exam_type", "enum", "exam type", "type"), C("class_name", "class", "grade", "class"),
         C("academic_year", "text", "academic year"), C("start_date", "date", "start date"), C("end_date", "date", "end date"),
         C("subject", "subject", "subject"), C("exam_date", "date", "exam date", "date"), C("start_time", "time", "start time"),
         C("end_time", "time", "end time"), C("room", "room", "room", "hall"), C("invigilator", "faculty", "invigilator"),
         C("max_marks", "number", "maximum marks"), C("pass_marks", "number", "pass marks")],
        ["exam_name", "subject", "exam_date", "start_time", "end_time", "room", "invigilator", "max_marks", "pass_marks"], "Exam schedule",
        sort="exam_date", singular="exam",
    ),
    View(
        "sv_marks", ["marks", "exam results", "scores", "results"],
        f"""SELECT e.exam_name, {_name('u')} AS student_name, c.class_name || '-' || sec.section_name AS section, sub.subject_name AS subject,
               m.marks_obtained, es.max_marks, ROUND(100.0 * m.marks_obtained / es.max_marks, 2) AS percentage, g.grade,
               g.remark AS grade_remark, CASE WHEN m.marks_obtained >= es.pass_marks THEN 'Pass' ELSE 'Fail' END AS result,
               m.is_absent, {_name('eu')} AS entered_by, m.remarks
        FROM school.marks m JOIN school.exam_schedules es ON es.exam_schedule_id = m.exam_schedule_id
        JOIN school.exams e ON e.exam_id = es.exam_id JOIN school.subjects sub ON sub.subject_id = es.subject_id
        JOIN school.students s ON s.student_id = m.student_id JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = s.section_id JOIN school.classes c ON c.class_id = sec.class_id
        LEFT JOIN school.grade_scale g ON g.grade_id = m.grade_id JOIN school.faculty f ON f.faculty_id = m.entered_by
        JOIN school.users eu ON eu.user_id = f.user_id""",
        [C("exam_name", "title", "exam"), C("student_name", "student", "student", "name"), C("section", "section", "class", "section"),
         C("subject", "subject", "subject"), C("marks_obtained", "number", "marks obtained", "marks", "score"), C("max_marks", "number", "maximum marks"),
         C("percentage", "pct", "percentage"), C("grade", "enum", "grade"), C("grade_remark", "text", "grade remark"),
         C("result", "enum", "result", "pass or fail"), C("is_absent", "bool", "absent"), C("entered_by", "faculty", "entered by"),
         C("remarks", "text", "remarks")],
        ["student_name", "subject", "marks_obtained", "max_marks", "grade", "result"], "Exam marks",
        scope=["student_name", "section", "subject", "exam_name"], sort="student_name, subject", singular="mark",
    ),
    View(
        "sv_progress", ["progress reports", "report cards", "rank list", "class ranks"],
        f"""SELECT {_name('u')} AS student_name, c.class_name || '-' || sec.section_name AS section, e.exam_name,
               pr.total_max_marks, pr.total_obtained, pr.percentage, g.grade, pr.attendance_pct, pr.class_rank,
               pr.teacher_remarks, pr.is_published
        FROM school.progress_reports pr JOIN school.students s ON s.student_id = pr.student_id JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = s.section_id JOIN school.classes c ON c.class_id = sec.class_id
        JOIN school.exams e ON e.exam_id = pr.exam_id LEFT JOIN school.grade_scale g ON g.grade_id = pr.grade_id""",
        [C("student_name", "student", "student", "name"), C("section", "section", "class", "section"), C("exam_name", "title", "exam"),
         C("total_max_marks", "number", "total maximum marks"), C("total_obtained", "number", "total marks", "total obtained"),
         C("percentage", "pct", "percentage", "overall percentage"), C("grade", "enum", "grade"), C("attendance_pct", "pct", "attendance percentage"),
         C("class_rank", "number", "rank", "class rank", "position"), C("teacher_remarks", "text", "teacher remarks", "remarks"),
         C("is_published", "bool", "published")],
        ["student_name", "exam_name", "total_obtained", "total_max_marks", "percentage", "grade", "class_rank"], "Progress reports and ranks",
        sort="class_rank", singular="progress report",
    ),
    View(
        "sv_progress_items", ["subject wise report card marks", "progress report subject marks"],
        f"""SELECT {_name('u')} AS student_name, e.exam_name, sub.subject_name AS subject, pri.marks_obtained, pri.max_marks, g.grade
        FROM school.progress_report_items pri JOIN school.progress_reports pr ON pr.report_id = pri.report_id
        JOIN school.students s ON s.student_id = pr.student_id JOIN school.users u ON u.user_id = s.user_id
        JOIN school.exams e ON e.exam_id = pr.exam_id JOIN school.subjects sub ON sub.subject_id = pri.subject_id
        LEFT JOIN school.grade_scale g ON g.grade_id = pri.grade_id""",
        [C("student_name", "student", "student"), C("exam_name", "title", "exam"), C("subject", "subject", "subject"),
         C("marks_obtained", "number", "marks"), C("max_marks", "number", "maximum marks"), C("grade", "enum", "grade")],
        ["student_name", "subject", "marks_obtained", "max_marks", "grade"], "Report card subject marks",
        sort="student_name, subject", singular="report card subject mark",
    ),
    View(
        "sv_grade_scale", ["grades", "grade scale", "grading scale"],
        "SELECT grade, min_percent, max_percent, grade_point, remark FROM school.grade_scale",
        [C("grade", "enum", "grade"), C("min_percent", "pct", "minimum percentage"), C("max_percent", "pct", "maximum percentage"),
         C("grade_point", "number", "grade point"), C("remark", "text", "remark", "meaning")],
        ["grade", "min_percent", "max_percent", "grade_point", "remark"], "Grading scale", sort="min_percent DESC", singular="grade",
    ),
    View(
        "sv_exam_types", ["exam types"],
        "SELECT type_name AS exam_type, weightage FROM school.exam_types",
        [C("exam_type", "enum", "exam type", "type"), C("weightage", "pct", "weightage", "weight")],
        ["exam_type", "weightage"], "Exam types", sort="weightage", singular="exam type",
    ),
    View(
        "sv_notifications", ["notifications", "messages", "alerts"],
        f"""SELECT {_name('u')} AS recipient, r.role_name AS role, n.title, n.message, n.is_read, n.created_at
        FROM school.notifications n JOIN school.users u ON u.user_id = n.user_id JOIN school.roles r ON r.role_id = u.role_id""",
        [C("recipient", "person", "recipient", "user"), C("role", "enum", "role"), C("title", "title", "title"), C("message", "text", "message"),
         C("is_read", "bool", "read status"), C("created_at", "date", "date sent")],
        ["recipient", "title", "message", "is_read", "created_at"], "Notifications", sort="created_at", singular="notification",
    ),
]

VIEW_BY_NAME: Dict[str, View] = {v.name: v for v in VIEWS}

# raw tables each view reads, for documentation / coverage checks
ALL_TABLES = (
    "roles users academic_years classes subjects classrooms admins faculty sections students course_assignments periods timetable "
    "student_attendance faculty_attendance leave_types leave_requests homework homework_submissions assignments assignment_submissions "
    "exam_types exams exam_schedules grade_scale marks progress_reports progress_report_items notifications"
).split()


def cte_prefix(names: Sequence[str]) -> str:
    """WITH-clause defining only the semantic views a query references."""
    parts = [f"{n} AS (\n{re.sub(r'\\s+', ' ', VIEW_BY_NAME[n].sql).strip()}\n)" for n in names if n in VIEW_BY_NAME]
    return ("WITH " + ",\n".join(parts) + "\n") if parts else ""


def schema_outline() -> str:
    lines = []
    for v in VIEWS:
        lines.append(f"{v.name}({', '.join(v.colnames)})")
    return "\n".join(lines)
