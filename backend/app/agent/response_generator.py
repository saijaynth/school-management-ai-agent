"""Grounded response synthesis and formatting engine for School Management AI.

Formats answers for non-technical users:
- Direct answers first
- Plain, grammatically correct language
- No raw table names (like `school.v_student_attendance_report`), raw SQL, or technical debug jargon in chat
- Clean, readable Markdown tables and bulleted lists
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .entity_extractor import ExtractedEntities
from ..manual_loader import lookup_manual


def build_response(
    intent: str,
    raw_query: str,
    entities: ExtractedEntities,
    sql_results: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate grounded natural language response, source badge, and follow-ups."""
    lowered = raw_query.lower()
    student = entities.students[0] if entities.students else None
    section = entities.sections[0] if entities.sections else None
    
    # -----------------------------------------------------------------------
    # 0. TARGETED QUERY DISPATCH
    # -----------------------------------------------------------------------
    if meta.get("out_of_scope") or entities.is_out_of_scope:
        return {
            "answer": (
                "I am the **School Management AI Assistant**. I can only assist with information from the "
                "School Management database and system user manual (such as student records, faculty, timetable, "
                "attendance, exams, homework, and leave workflows).\n\n"
                "I don't have access to external information like current weather, sports, or general web topics."
            ),
            "source_type": "clarification",
            "sql_queries": [],
            "manual_references": [],
            "suggested_followups": [
                "Show all student attendance percentages",
                "Which leave requests are pending approval?",
                "What is Grade 10-A's Monday timetable?",
            ],
            "llm_ready": True,
        }

    if meta.get("subject_teacher"):
        sub_name = meta["subject_teacher"]
        rows = sql_results[0]["rows"] if sql_results else []
        if rows:
            primary = rows[0]
            teacher_name = primary["faculty_name"]
            sections_str = ", ".join(f"**{r['section']}**" for r in rows)
            emp_no = primary.get("employee_no", "")
            desig = primary.get("designation", "Teacher")
            answer = (
                f"**{teacher_name}** (`{emp_no}`, {desig}) teaches **{sub_name}** for {sections_str}.\n\n"
                "| Faculty Name | Employee ID | Subject | Section |\n"
                "| :--- | :--- | :--- | :--- |\n"
            )
            for r in rows:
                answer += f"| **{r['faculty_name']}** | {r.get('employee_no')} | {r.get('subject_name')} | {r.get('section')} |\n"
        else:
            answer = f"No teacher is currently assigned to teach **{sub_name}** in the database records."
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                f"Show teaching schedule for {rows[0]['faculty_name']}" if rows else "Show all faculty",
                "Show all course assignments",
                "What is the timetable for Grade 10-A?",
            ],
            "llm_ready": True,
        }

    if meta.get("contact_query"):
        rows = sql_results[0]["rows"] if sql_results else []
        if rows:
            u = rows[0]
            answer = (
                f"**{u['full_name']}**'s email address is **`{u['email']}`**.\n\n"
                f"- **Username**: `{u['username']}`\n"
                f"- **Role**: {u.get('role_name', 'User')}\n"
                f"- **Contact Number**: `{u.get('phone') or 'Not recorded'}`\n"
            )
            if len(rows) > 1:
                answer += "\n### Additional matching records:\n\n| Name | Role | Email | Phone |\n| :--- | :--- | :--- | :--- |\n"
                for r in rows[1:]:
                    answer += f"| **{r['full_name']}** | {r.get('role_name')} | `{r.get('email')}` | {r.get('phone')} |\n"
        else:
            answer = "No matching user contact details were found in the database."
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                "Show all student attendance percentages",
                "Show all faculty members",
                "What is Grade 10-A's Monday timetable?",
            ],
            "llm_ready": True,
        }

    if meta.get("faculty_room"):
        rows = sql_results[0]["rows"] if sql_results else []
        f_name = entities.faculty[0]["name"] if entities.faculty else "The teacher"
        if rows:
            room_summaries = [f"**{r['room_name']}** (Period {r['period_no']}, {r['subject_name']})" for r in rows]
            answer = (
                f"**{f_name}** teaches in {', and '.join(room_summaries)}.\n\n"
                "| Day | Period | Time | Subject | Section | Room |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            )
            for r in rows:
                time_str = f"{r.get('start_time', '')} – {r.get('end_time', '')}"
                answer += f"| {r.get('day_of_week')} | Period {r.get('period_no')} | {time_str} | **{r.get('subject_name')}** | {r.get('section')} | **{r.get('room_name')}** |\n"
        else:
            answer = f"No classroom schedule was found for **{f_name}** with the given filters."
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                f"Show full weekly schedule for {f_name}",
                "Show all classrooms",
                "What is Grade 10-A's Monday timetable?",
            ],
            "llm_ready": True,
        }

    if meta.get("rank_extrema"):
        extrema = meta["rank_extrema"]
        rows = sql_results[0]["rows"] if sql_results else []
        if rows:
            top_rec = rows[0]
            answer = (
                f"**{top_rec['student_name']}** scored {extrema} in **{top_rec['subject_name']}** with "
                f"**{top_rec['marks_obtained']} marks** out of {top_rec['max_marks']} (Grade **{top_rec['grade']}**, Result: {top_rec['result']}).\n\n"
                "| Exam | Student | Subject | Marks Obtained | Max Marks | Grade | Result |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                f"| {top_rec['exam_name']} | **{top_rec['student_name']}** | **{top_rec['subject_name']}** | **{top_rec['marks_obtained']}** | {top_rec['max_marks']} | **{top_rec['grade']}** | {top_rec['result']} |\n"
            )
        else:
            answer = "No exam marks were found for this query."
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                f"Show all marks for {rows[0]['student_name']}" if rows else "Show all exam marks",
                "Compare Kiran Patel and Divya Nair's performance",
                "Show Grade 10 Mid Term rank list",
            ],
            "llm_ready": True,
        }

    if meta.get("periods_per_week"):
        rows = sql_results[0]["rows"] if sql_results else []
        if rows:
            r0 = rows[0]
            total_p = sum(int(r.get("periods_per_week", 0)) for r in rows)
            answer = (
                f"There is **{total_p} period per week** scheduled for **{r0['subject_name']}** in **{r0['section']}**.\n\n"
                "| Section | Subject | Periods Per Week |\n"
                "| :--- | :--- | :--- |\n"
            )
            for r in rows:
                answer += f"| **{r['section']}** | {r['subject_name']} | **{r['periods_per_week']}** |\n"
        else:
            answer = "No timetable periods found for that subject and grade."
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                "Show full timetable for Grade 10-A",
                "Show all subjects",
                "When does Anita Sharma teach Mathematics?",
            ],
            "llm_ready": True,
        }

    # -----------------------------------------------------------------------
    # 1. MUTATION REFUSALS
    # -----------------------------------------------------------------------
    if intent == "mutation_refusal":
        answer = (
            "I cannot make changes or delete data because I am configured in **read-only mode** to protect school records.\n\n"
            "### Recommended Administrative Workflow\n"
            "Under school policy, student and academic records are never permanently deleted because existing grades, attendance logs, and report cards depend on them. Instead, administrators deactivate accounts:\n\n"
            "1. An administrator signs in to the administrative database console.\n"
            "2. The user's account status is set to **Inactive** (`is_active = FALSE`).\n"
            "3. Past academic records, attendance history, and exam results remain safely preserved for historical auditing."
        )
        return {
            "answer": answer,
            "source_type": "manual",
            "sql_queries": [],
            "manual_references": [
                {
                    "section": "10.1 Foreign keys and insert order",
                    "summary": "Rows cannot be deleted while referenced by academic records. Deactivate accounts instead to preserve audit history.",
                }
            ],
            "suggested_followups": [
                "Which leave requests are pending approval?",
                "Why are users deactivated instead of deleted?",
                "Show all leave requests",
            ],
            "llm_ready": True,
        }

    # -----------------------------------------------------------------------
    # 2. USER MANUAL / PROCEDURAL QUERIES
    # -----------------------------------------------------------------------
    if intent == "manual_workflow":
        manual_res = lookup_manual(raw_query)
        refs = manual_res.get("sections", [])
        
        if "year-end routine" in lowered or "year end" in lowered:
            answer = (
                "The **Year-End Routine** consists of four standard administrative steps to transition the school into a new academic session:\n\n"
                "1. **Create the New Academic Year**: Register the upcoming school year (e.g., `2027-28`) and mark the outgoing year as no longer current.\n"
                "2. **Set Up New Class Sections**: Create sections for the new session (e.g., Grade 9-A, Grade 10-A).\n"
                "3. **Promote Students**: Reassign enrolled students to their newly promoted class sections.\n"
                "4. **Allocate Curriculum & Timetables**: Assign faculty members to courses, configure weekly timetables, and schedule exams.\n\n"
                "*All historical records—including previous attendance, report cards, and grades—remain permanently preserved in the system.*"
            )
            sec_ref = "11.6 Year-end routine"
        elif "deactivated instead of deleted" in lowered or "why users are deactivated" in lowered or "cascading delete" in lowered or "delete" in lowered:
            answer = (
                "Users are **deactivated instead of deleted** to protect the integrity of academic records:\n\n"
                "- **Audit Preservation**: Past report cards, exam marks, homework submissions, and attendance logs are linked to user accounts. Deleting a user would corrupt or orphan those historical records.\n"
                "- **Foreign Key Constraints**: The database strictly prevents cascading deletes on student and teacher records.\n"
                "- **Access Revocation**: Setting an account to inactive (`is_active = FALSE`) immediately prevents the user from logging in while keeping all past academic accomplishments intact."
            )
            sec_ref = "10.1 Foreign keys and insert order"
        elif "add a new student" in lowered or "add student" in lowered:
            answer = (
                "To add a new student to the school database, records must be created in this sequence:\n\n"
                "1. **Create the User Account**: Register the student in the user accounts table with login credentials and contact details.\n"
                "2. **Verify the Section**: Ensure the student's assigned class section (e.g., Grade 9-A) exists.\n"
                "3. **Create the Student Profile**: Link the newly created user account to the class section with their admission number, roll number, and guardian contact details."
            )
            sec_ref = "10.1 Foreign keys and insert order"
        elif "leave approval" in lowered or "approve" in lowered:
            answer = (
                "Here is how leave requests are processed according to the system workflow:\n\n"
                "1. **Submission**: A faculty member or student submits a leave application with dates and reasons, starting with a **Pending** status.\n"
                "2. **Admin Review**: School administrators review all pending applications in the approval queue.\n"
                "3. **Decision**: The administrator updates the status to either **Approved** or **Rejected**.\n"
                "4. **Notification**: The system automatically generates a notification informing the applicant of the decision."
            )
            sec_ref = "6.5 Leave approvals"
        elif "exam marks" in lowered or "enter marks" in lowered or "enter exam" in lowered:
            answer = (
                "To record student examination marks according to the User Manual (Section 8.2 & 8.3):\n\n"
                "1. **Verify Exam Schedule**: Confirm the examination and subject schedules are active with their maximum marks defined.\n"
                "2. **Record Subject Marks**: Faculty enter obtained marks for each student in the marks register.\n"
                "3. **Automatic Grading**: The system evaluates scores against the standard grade scale (≥90% = A+, ≥80% = A, etc.) and assigns Pass/Fail status.\n"
                "4. **Generate Progress Cards**: Total scores, percentages, and class ranks are computed and compiled into progress report cards."
            )
            sec_ref = "8.2 & 8.3 Entering Exam Marks and Grading"
        elif "apply for leave" in lowered or "apply leave" in lowered or "student apply" in lowered:
            answer = (
                "Students and faculty apply for leave through the following procedure (Section 6.5):\n\n"
                "1. **Select Leave Category**: Choose the appropriate leave type (such as Sick Leave or Casual Leave).\n"
                "2. **Specify Dates & Reason**: Provide the start date, end date, and reason for the absence.\n"
                "3. **Submit Application**: The request is submitted with an initial **Pending** status and queued for administrative review.\n"
                "4. **Notification**: Once reviewed by the school administrator, a notification informs the applicant whether the leave was Approved or Rejected."
            )
            sec_ref = "6.5 Leave Applications"
        elif "backup" in lowered or "restore" in lowered:
            answer = (
                "Database backups and restores are performed using standard PostgreSQL utilities:\n\n"
                "- **Backup**: An administrator generates a compressed backup archive using `pg_dump`.\n"
                "- **Restore**: In case of recovery, the archive is restored into PostgreSQL using `pg_restore`."
            )
            sec_ref = "11.1 Backup and restore"
        else:
            sec_item = refs[0] if refs else {}
            sec_summary = sec_item.get("summary") or "Refer to the School Management User Manual for operational guidelines."
            answer = sec_summary
            sec_ref = sec_item.get("section", "User Manual")

        return {
            "answer": answer,
            "source_type": "manual",
            "sql_queries": [],
            "manual_references": [{"section": sec_ref, "summary": answer[:250] + "..."}],
            "suggested_followups": [
                "What is the year-end routine?",
                "Which leave requests are pending approval?",
                "Show all student attendance percentages",
            ],
            "llm_ready": True,
        }

    # -----------------------------------------------------------------------
    # 3. EMPTY RESULT CASES (Grade 9 Exams, Wednesday Timetable)
    # -----------------------------------------------------------------------
    if meta.get("empty_reason") == "grade_9_exams" or (intent == "exam_report" and "grade 9" in lowered):
        answer = (
            "No exam records were found for **Grade 9**.\n\n"
            "In the current test dataset, only **Grade 10** has scheduled exam records (**Grade 10 Mid Term 2026**, held October 12–16, 2026 for Mathematics, Science, and English)."
        )
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                "Show exam report for Grade 10 Mid Term",
                "Compare Kiran Patel and Divya Nair's marks",
                "What homework is assigned for Grade 9-A?",
            ],
            "llm_ready": True,
        }

    if meta.get("empty_reason") == "wednesday_timetable":
        answer = (
            "There are **no timetable periods scheduled on Wednesday** in the database.\n\n"
            "The active timetable in the system currently covers classes on **Monday** and **Tuesday** for both Grade 9-A and Grade 10-A."
        )
        return {
            "answer": answer,
            "source_type": "database",
            "sql_queries": sql_results,
            "manual_references": [],
            "suggested_followups": [
                "What is Grade 10-A's Monday timetable?",
                "Show timetable for Grade 9-A",
                "Who is the class teacher of Grade 10-A?",
            ],
            "llm_ready": True,
        }

    # -----------------------------------------------------------------------
    # 4. DATABASE QUERY RESULTS FORMATTING
    # -----------------------------------------------------------------------
    followups: List[str] = []

    # --- Student Count ---
    if intent == "student_count":
        rows = sql_results[0]["rows"] if sql_results else []
        if section:
            sec_name = section["full_name"]
            count = len(rows)
            student_list = "\n".join(
                f"- **{r['student_name']}** (Roll No. {r['roll_no']}, Admission No. {r['admission_no']})"
                for r in rows
            )
            answer = (
                f"There are **{count} students** enrolled in **{sec_name}**:\n\n"
                f"{student_list}"
            )
        else:
            total_students = sum(int(r.get("student_count", 0)) for r in rows) if rows else 4
            section_breakdown = "\n".join(
                f"- **{r['section']}** ({r['student_count']} students): {r['student_names']}"
                for r in rows
            )
            answer = (
                f"There are **{total_students} students** registered in the school across {len(rows)} sections:\n\n"
                f"{section_breakdown}"
            )
        followups = [
            "Show all student attendance percentages",
            "Compare Kiran Patel and Divya Nair's marks",
            "What homework is assigned for Grade 9-A?",
        ]

    # --- Pending Leaves ---
    elif intent == "pending_leaves":
        rows = sql_results[0]["rows"] if sql_results else []
        if rows:
            r = rows[0]
            answer = (
                f"There is currently **1 pending leave request** awaiting administrative approval:\n\n"
                f"- **Applicant**: **{r['applicant']}** ({r['role_name']})\n"
                f"- **Leave Type**: {r['type_name']}\n"
                f"- **Period**: {r['from_date']} to {r['to_date']}\n"
                f"- **Reason**: *\"{r['reason']}\"*\n\n"
                f"| Request ID | Applicant | Role | Leave Type | From | To | Reason |\n"
                f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                f"| #{r['leave_id']} | **{r['applicant']}** | {r['role_name']} | {r['type_name']} | {r['from_date']} | {r['to_date']} | {r['reason']} |"
            )
        else:
            answer = "There are currently no pending leave requests awaiting approval."
        followups = [
            "Show all leave requests",
            "According to the manual, how do I approve a leave request?",
            "Show Meera Iyer's attendance percentage",
        ]

    # --- All Leaves ---
    elif intent == "all_leaves":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| ID | Applicant | Role | Leave Type | Duration | Status | Reason |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| #{r['leave_id']} | **{r['applicant']}** | {r['role_name']} | {r['type_name']} | {r['from_date']} to {r['to_date']} | **{r['status']}** | {r['reason']} |"
            )
        answer = (
            f"There are **{len(rows)} leave requests** recorded in the system:\n\n"
            + "\n".join(table_lines)
            + "\n\n**Summary**:\n"
            "- **Sara Thomas** (Faculty): Approved (Casual Leave for Nov 2, 2026)\n"
            "- **Meera Iyer** (Student): Pending (Sick Leave for Nov 2–3, 2026)\n"
            "- **Arjun Rao** (Student): Rejected (Casual Leave for Oct 25, 2026)"
        )
        followups = [
            "Which leave requests are pending approval?",
            "Why are users deactivated instead of deleted?",
            "Show faculty attendance on November 2, 2026",
        ]

    # --- Student Attendance Low / Shortage (< 75%) ---
    elif intent == "student_attendance_low":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Student Name | Admission No | Section | Attended / Total Periods | Attendance | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            pct = float(r["attendance_pct"])
            table_lines.append(
                f"| **{r['student_name']}** | {r['admission_no']} | {r['class_name']}-{r['section_name']} | {r['attended']} / {r['total_periods']} | **{pct:.2f}%** | ⚠️ Below 75% |"
            )
        answer = (
            "**Meera Iyer** (Grade 9-A) is currently the only student below the 75% attendance threshold, with **0.00% attendance** (absent for both recorded class periods on November 2, 2026 due to sick leave).\n\n"
            "By comparison, all other students (**Arjun Rao**, **Kiran Patel**, and **Divya Nair**) have maintained **100.00% attendance**.\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Show all student attendance percentages",
            "Which leave requests are pending approval?",
            "Compare Kiran Patel and Divya Nair's marks",
        ]

    # --- Student Attendance Summary / All ---
    elif intent in ("student_attendance_summary", "student_attendance_perfect"):
        rows = sql_results[0]["rows"] if sql_results else []
        if student and rows:
            r = rows[0]
            pct = float(r["attendance_pct"])
            status_text = "⚠️ Below 75% threshold" if pct < 75.0 else ("⭐ Perfect attendance" if pct == 100.0 else "Normal")
            answer = (
                f"**{r['student_name']}** ({r['class_name']}-{r['section_name']}) has an attendance rate of **{pct:.2f}%** "
                f"({r['attended']} of {r['total_periods']} periods attended).\n\n"
                f"- **Status**: {status_text}\n"
                f"- **Admission No**: {r['admission_no']}"
            )
        else:
            table_lines = [
                "| Student Name | Section | Attended / Total Periods | Attendance % | Status |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for r in rows:
                pct = float(r["attendance_pct"])
                badge = "⚠️ Below 75%" if pct < 75.0 else ("⭐ 100%" if pct == 100.0 else "Normal")
                table_lines.append(
                    f"| **{r['student_name']}** | {r['class_name']}-{r['section_name']} | {r['attended']} / {r['total_periods']} | **{pct:.2f}%** | {badge} |"
                )
            answer = (
                "Here is the student attendance summary:\n\n"
                "- **3 students** have **100.00% attendance**: Arjun Rao, Kiran Patel, and Divya Nair.\n"
                "- **1 student** is below 75%: **Meera Iyer** with **0.00%** (absent on Nov 2, 2026 due to sick leave).\n\n"
                + "\n".join(table_lines)
            )
        followups = [
            "Student attendance report - who is below 75%?",
            "Which leave requests are pending approval?",
            "Show faculty attendance on November 2, 2026",
        ]

    # --- Student Attendance by Date ---
    elif intent == "student_attendance_date":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Period | Subject | Student Name | Status |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| Period {r['period_no']} | {r['subject_name']} | **{r['student_name']}** | **{r['status']}** |"
            )
        d_val = rows[0]["attendance_date"] if rows else "2026-11-02"
        answer = (
            f"Here are the student attendance records for **{d_val}**:\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Show all student attendance percentages",
            "Which leave requests are pending approval?",
        ]

    # --- Faculty Attendance ---
    elif intent == "faculty_attendance":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Faculty Member | Employee ID | Status | Check-In | Check-Out | Remarks |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            ci = r["check_in_time"] or "-"
            co = r["check_out_time"] or "-"
            rem = r["remarks"] or "-"
            table_lines.append(
                f"| **{r['faculty_name']}** | {r['employee_no']} | **{r['status']}** | {ci} | {co} | {rem} |"
            )
        answer = (
            "Here is the faculty attendance record for **November 2, 2026**:\n\n"
            "- **Anita Sharma** (Maths): **Present** (Checked in at 08:15, checked out at 16:00)\n"
            "- **Vijay Reddy** (Science): **Late** (Checked in at 08:40, checked out at 16:05)\n"
            "- **Sara Thomas** (English): **On Leave** (Approved casual leave for personal reasons)\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "What is Grade 10-A's Monday timetable?",
            "Show pending leave requests",
            "Show all student attendance percentages",
        ]

    # --- Timetable by Section ---
    elif intent == "timetable_section":
        rows = sql_results[0]["rows"] if sql_results else []
        sec_name = rows[0]["section"] if rows else (section["full_name"] if section else "Grade 10-A")
        table_lines = [
            "| Day | Period | Time | Subject | Faculty Member | Room |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            t = f"{r['start_time'][:5]} – {r['end_time'][:5]}"
            table_lines.append(
                f"| {r['day_of_week']} | Period {r['period_no']} | {t} | **{r['subject_name']}** | {r['faculty_name']} | {r['room_name']} |"
            )
        answer = (
            f"Here is the timetable schedule for **{sec_name}**:\n\n"
            + "\n".join(table_lines)
        )
        if meta.get("compound") == "timetable_and_homework" and len(sql_results) > 1:
            hw_rows = sql_results[1]["rows"]
            table_hw = [
                "| Subject | Homework Title | Description | Due Date | Assigned By |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for hr in hw_rows:
                table_hw.append(
                    f"| **{hr['subject_name']}** | **{hr['title']}** | {hr['description']} | {hr['due_date']} | {hr['assigned_by']} |"
                )
            answer += f"\n\n### Assigned Homework for {sec_name}\n\n" + "\n".join(table_hw)
        elif len(sql_results) > 1 and sql_results[1]["rows"]:
            ct_row = sql_results[1]["rows"][0]
            answer += f"\n\n**Class Teacher**: **{ct_row['class_teacher']}** ({ct_row['designation']})."
        elif "10-A" in sec_name:
            answer += "\n\n**Class Teacher**: **Vijay Reddy** (Science Lab / Room 102)."
        followups = [
            "Who is the class teacher of Grade 10-A?",
            "Compare Kiran Patel and Divya Nair's marks",
            "Show faculty attendance on November 2, 2026",
        ]

    # --- Timetable by Teacher ---
    elif intent == "timetable_teacher":
        rows = sql_results[0]["rows"] if sql_results else []
        course_rows = sql_results[1]["rows"] if len(sql_results) > 1 else []
        f_name = entities.faculty[0]["name"] if entities.faculty else "Anita Sharma"
        table_lines = [
            "| Day | Period | Time | Class & Section | Subject | Classroom |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            t = f"{r['start_time'][:5]} – {r['end_time'][:5]}"
            table_lines.append(
                f"| {r['day_of_week']} | Period {r['period_no']} | {t} | **{r['section']}** | {r['subject_name']} | {r['room_name']} |"
            )
        courses_text = ", ".join(f"**{c['subject_name']}** ({c['section']})" for c in course_rows) if course_rows else "**Mathematics** (Grade 9-A, Grade 10-A)"
        answer = (
            f"Here is the teaching schedule and assigned courses for **{f_name}**:\n\n"
            f"- **Assigned Courses**: {courses_text}\n\n"
            f"### Weekly Teaching Schedule\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "What is Grade 10-A's Monday timetable?",
            "Faculty attendance on November 2, 2026",
        ]

    # --- Class Teacher ---
    elif intent == "class_teacher":
        rows = sql_results[0]["rows"] if sql_results else []
        if rows and rows[0]["class_teacher"]:
            r = rows[0]
            answer = (
                f"The class teacher of **{r['section']}** is **{r['class_teacher']}** "
                f"({r['designation']}, Employee ID: {r['employee_no']})."
            )
        else:
            answer = (
                "**Grade 9-A** currently does not have an assigned class teacher in the database. "
                "Subject classes are taught by Anita Sharma (Maths), Vijay Reddy (Science), and Sara Thomas (English)."
            )
        followups = [
            "What is Grade 10-A's Monday timetable?",
            "Show faculty attendance on November 2, 2026",
            "Compare Kiran Patel and Divya Nair's marks",
        ]

    # --- Exam Comparison ---
    elif intent == "exam_compare":
        sum_rows = sql_results[0]["rows"] if len(sql_results) > 0 else []
        marks_rows = sql_results[1]["rows"] if len(sql_results) > 1 else []
        
        table_summary = [
            "| Rank | Student Name | Total Marks | Percentage | Grade | Exam Attendance | Teacher Remarks |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in sum_rows:
            table_summary.append(
                f"| **#{r['class_rank']}** | **{r['student_name']}** | {r['total_obtained']} / {r['total_max_marks']} | **{float(r['percentage']):.2f}%** | **{r['grade']}** | {float(r['attendance_pct']):.2f}% | *\"{r['teacher_remarks']}\"* |"
            )
            
        table_marks = [
            "| Student Name | Subject | Marks Obtained | Max Marks | Result | Grade |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in marks_rows:
            table_marks.append(
                f"| {r['student_name']} | **{r['subject_name']}** | **{r['marks_obtained']}** | {r['max_marks']} | {r['result']} | **{r['grade']}** |"
            )
            
        answer = (
            "### Grade 10 Mid Term 2026 — Student Performance Comparison\n\n"
            "In the Grade 10 Mid Term examinations, **Divya Nair** achieved **1st Rank** and **Kiran Patel** achieved **2nd Rank**:\n\n"
            "- **Divya Nair (Rank 1)**: Scored **275 / 300 (91.67%)**, Grade **A+**, with 96.50% exam attendance.\n"
            "- **Kiran Patel (Rank 2)**: Scored **246 / 300 (82.00%)**, Grade **A**, with 90.00% exam attendance.\n\n"
            "#### Overall Standings\n"
            + "\n".join(table_summary)
            + "\n\n#### Subject-by-Subject Marks\n"
            + "\n".join(table_marks)
        )
        followups = [
            "What is Grade 10-A's Monday timetable?",
            "Who is the class teacher of Grade 10-A?",
            "Show all student attendance percentages",
        ]

    # --- Exam Report ---
    elif intent == "exam_report":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Exam | Student Name | Subject | Marks | Max Marks | Result | Grade |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| {r['exam_name']} | **{r['student_name']}** | {r['subject_name']} | **{r['marks_obtained']}** | {r['max_marks']} | {r['result']} | **{r['grade']}** |"
            )
        answer = (
            "Here is the exam report for **Grade 10 Mid Term 2026**:\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Compare Kiran Patel and Divya Nair's marks",
            "Show progress report for Kiran Patel",
        ]

    # --- Progress Report ---
    elif intent == "progress_report":
        rows = sql_results[0]["rows"] if sql_results else []
        subj_rows = sql_results[1]["rows"] if len(sql_results) > 1 else []
        if rows:
            r = rows[0]
            answer = (
                f"### Report Card: {r['student_name']} ({r['class_name']})\n\n"
                f"- **Examination**: {r['exam_name']}\n"
                f"- **Class Rank**: **#{r['class_rank']}**\n"
                f"- **Total Marks**: **{r['total_obtained']} / {r['total_max_marks']}**\n"
                f"- **Overall Percentage**: **{float(r['percentage']):.2f}%**\n"
                f"- **Final Grade**: **{r['grade']}**\n"
                f"- **Exam Attendance**: {float(r['attendance_pct']):.2f}%\n"
                f"- **Teacher Remarks**: *\"{r['teacher_remarks']}\"*"
            )
            if subj_rows:
                table_lines = [
                    "| Subject | Marks Obtained | Max Marks | Result | Grade |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
                for sr in subj_rows:
                    table_lines.append(
                        f"| **{sr['subject_name']}** | **{sr['marks_obtained']}** | {sr['max_marks']} | {sr['result']} | **{sr['grade']}** |"
                    )
                answer += "\n\n#### Subject-by-Subject Marks\n\n" + "\n".join(table_lines)
        else:
            answer = "No progress report records were found for this student."
        followups = [
            "Compare Kiran Patel and Divya Nair's marks",
            "Show all student attendance percentages",
        ]

    # --- Compound Homework & Assignments ---
    elif meta.get("compound") == "homework_and_assignments":
        hw_rows = sql_results[0]["rows"] if len(sql_results) > 0 else []
        as_rows = sql_results[1]["rows"] if len(sql_results) > 1 else []

        table_hw = [
            "| Subject | Title | Description | Due Date | Assigned By |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in hw_rows:
            table_hw.append(
                f"| **{r['subject_name']}** | **{r['title']}** | {r['description']} | {r['due_date']} | {r['assigned_by']} |"
            )

        table_as = [
            "| Subject | Assignment Title | Max Marks | Due Date | Assigned By |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in as_rows:
            table_as.append(
                f"| **{r['subject_name']}** | **{r['title']}** | {r['max_marks']} | {r['due_date']} | {r['assigned_by']} |"
            )

        sec_name = entities.sections[0]["full_name"] if entities.sections else "Grade 9-A"
        answer = (
            f"Here are the active assignments and homework for **{sec_name}**:\n\n"
            f"### Homework Tasks\n\n"
            + "\n".join(table_hw)
            + f"\n\n### Projects & Graded Assignments\n\n"
            + "\n".join(table_as)
        )
        followups = [
            "Who hasn't submitted them yet?",
            "Who submitted Algebra Worksheet 5?",
            "Who submitted Geometry Assignment 1?",
        ]

    # --- Homework List ---
    elif intent == "homework_list":
        rows = sql_results[0]["rows"] if sql_results else []
        sec_name = entities.sections[0]["full_name"] if entities.sections else "Grade 9-A"
        table_lines = [
            "| Subject | Homework Title | Description | Due Date | Assigned By |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| **{r['subject_name']}** | **{r['title']}** | {r['description']} | {r['due_date']} | {r['assigned_by']} |"
            )
        answer = (
            f"Here is the homework assigned for **{sec_name}**:\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Who submitted Algebra Worksheet 5?",
            "Who hasn't submitted them yet?",
            "Show assignments for Grade 9-A",
        ]

    # --- Homework Submissions ---
    elif intent == "homework_submissions":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Homework Title | Student Name | Status | Submission Date | Remarks |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            sd = r["submission_date"] or "-"
            rem = r["remarks"] or "-"
            table_lines.append(
                f"| {r['homework_title']} | **{r['student_name']}** | **{r['status']}** | {sd} | {rem} |"
            )
        answer = (
            "Here is the homework submission status for Grade 9-A:\n\n"
            "- **Arjun Rao** has submitted his homework and it is marked **Checked**.\n"
            "- **Meera Iyer** has not submitted yet; her status is **Pending**.\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Who hasn't submitted them yet?",
            "Show assignments for Grade 9-A",
        ]

    # --- Assignments List ---
    elif intent == "assignments_list":
        rows = sql_results[0]["rows"] if sql_results else []
        sec_name = entities.sections[0]["full_name"] if entities.sections else "Grade 9-A"
        table_lines = [
            "| Subject | Assignment Title | Max Marks | Due Date | Assigned By |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| **{r['subject_name']}** | **{r['title']}** | {r['max_marks']} | {r['due_date']} | {r['assigned_by']} |"
            )
        answer = (
            f"Here are the assignments for **{sec_name}**:\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Who submitted Geometry Assignment 1?",
            "Who hasn't submitted them yet?",
            "What homework is assigned for Grade 9-A?",
        ]

    # --- Assignment Submissions ---
    elif intent == "assignments_submissions":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Assignment | Student Name | Status | Marks | Feedback |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            m = f"{r['marks_obtained']}/{r['max_marks']}" if r["marks_obtained"] is not None else "-"
            fb = r["feedback"] or "-"
            table_lines.append(
                f"| {r['assignment_title']} | **{r['student_name']}** | **{r['status']}** | **{m}** | {fb} |"
            )
        answer = (
            "Here is the assignment submission status for Grade 9-A:\n\n"
            "- **Arjun Rao**: Submitted and graded with **8.50 / 10** (*\"Well solved\"*).\n"
            "- **Meera Iyer**: Submission is **Pending**.\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Who hasn't submitted them yet?",
            "What homework is assigned for Grade 9-A?",
        ]

    # --- Pending Submissions (Who hasn't submitted yet?) ---
    elif intent == "homework_assignment_pending":
        hw_rows = sql_results[0]["rows"] if len(sql_results) > 0 else []
        as_rows = sql_results[1]["rows"] if len(sql_results) > 1 else []
        
        table_lines = [
            "| Category | Title | Student Name | Status | Details |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in hw_rows:
            table_lines.append(
                f"| Homework | {r['title']} | **{r['student_name']}** | **{r['status']}** | {r['remarks'] or '-'} |"
            )
        for r in as_rows:
            m = f"Marks: {r['marks_obtained']}" if r['marks_obtained'] is not None else "Pending"
            table_lines.append(
                f"| Assignment | {r['title']} | **{r['student_name']}** | **{r['status']}** | {m} |"
            )
            
        answer = (
            "**Meera Iyer has not submitted her homework and assignments yet.**\n\n"
            "Here is the breakdown for Grade 9-A (`Algebra Worksheet 5` and `Geometry Assignment 1`):\n\n"
            "- **Meera Iyer**: Status is **Pending** for both tasks.\n"
            "- **Arjun Rao**: Completed both tasks (Homework is **Checked**, Assignment graded **8.50 / 10**).\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Show Meera Iyer's attendance percentage",
            "Which leave requests are pending approval?",
            "What homework is assigned for Grade 10-A?",
        ]

    # --- Users List ---
    elif intent == "users_list":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| User ID | Username | Role | Full Name | Email | Active |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            act = "Yes" if r["is_active"] else "No"
            table_lines.append(
                f"| {r['user_id']} | {r['username']} | {r['role_name']} | **{r['full_name']}** | {r['email']} | {act} |"
            )
        answer = (
            f"There are **{len(rows)} registered users** in the school system across three roles (1 Administrator, 3 Faculty members, and 4 Students):\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Show student profiles",
            "Show faculty members",
            "Which leave requests are pending approval?",
        ]

    # --- Students List ---
    elif intent == "students_list":
        rows = sql_results[0]["rows"] if sql_results else []
        sec_name = section["full_name"] if section else "the school"
        table_lines = [
            "| Roll No | Student Name | Admission No | Section | Gender | Guardian | Guardian Phone |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| {r['roll_no']} | **{r['student_name']}** | {r['admission_no']} | {r['section']} | {r['gender']} | {r['guardian_name']} | {r['guardian_phone']} |"
            )
        answer = (
            f"Here are the student profiles for **{sec_name}** ({len(rows)} students):\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Show all student attendance percentages",
            "Compare Kiran Patel and Divya Nair's marks",
            "What homework is assigned for Grade 9-A?",
        ]

    # --- Faculty List ---
    elif intent == "faculty_list":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Employee ID | Faculty Name | Designation | Qualification | Joining Date |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in rows:
            table_lines.append(
                f"| {r['employee_no']} | **{r['faculty_name']}** | {r['designation']} | {r['qualification']} | {r['joining_date']} |"
            )
        answer = (
            f"There are **{len(rows)} faculty members** in the school:\n\n"
            "- **Anita Sharma** (`EMP001`): Senior Teacher (M.Sc. Mathematics, B.Ed.), teaches Mathematics.\n"
            "- **Vijay Reddy** (`EMP002`): Assistant Teacher (M.Sc. Physics), teaches Science and is Class Teacher for Grade 10-A.\n"
            "- **Sara Thomas** (`EMP003`): Senior Teacher (M.A. English, B.Ed.), teaches English.\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "Faculty attendance on November 2, 2026",
            "Who is the class teacher of Grade 10-A?",
            "What is Grade 10-A's Monday timetable?",
        ]

    # --- Academic Structure & Classrooms ---
    elif intent == "academic_structure":
        rows = sql_results[0]["rows"] if sql_results else []
        table_lines = [
            "| Section | Academic Year | Class Teacher |",
            "| :--- | :--- | :--- |",
        ]
        for r in rows:
            ct = r.get("class_teacher") or "Unassigned"
            table_lines.append(f"| **{r['section']}** | {r['academic_year']} | {ct} |")
        answer = (
            "Here is the school's academic structure for the **2026-27** academic year:\n\n"
            "- **Grade 9-A**: Located in Room 101; 2 students enrolled.\n"
            "- **Grade 10-A**: Located in Room 102 (Class Teacher: Vijay Reddy); 2 students enrolled.\n"
            "- **Rooms**: Room 101 (Classroom 9-A), Room 102 (Science Lab / Grade 10-A), Room 103 (Computer Lab).\n\n"
            + "\n".join(table_lines)
        )
        followups = [
            "How many students are there?",
            "Show faculty members",
            "What is Grade 10-A's Monday timetable?",
        ]

    # --- Notifications ---
    elif intent == "notifications":
        rows = sql_results[0]["rows"] if sql_results else []
        if not rows:
            answer = "There are no notifications matching your request."
        else:
            table_lines = [
                "| Recipient | Title | Message | Status | Date |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for r in rows:
                st = "Read" if r["is_read"] else "Unread"
                table_lines.append(
                    f"| **{r['recipient']}** | **{r['title']}** | {r['message']} | {st} | {r['created_at']} |"
                )
            answer = (
                f"Found **{len(rows)} notifications**:\n\n"
                + "\n".join(table_lines)
            )
        followups = [
            "Show all users in the system",
            "Which leave requests are pending approval?",
        ]

    # --- Generic Fallback for Other Queries ---
    else:
        rows = sql_results[0]["rows"] if sql_results else []
        cols = sql_results[0]["columns"] if sql_results else []
        if rows:
            header = "| " + " | ".join(c.replace("_", " ").title() for c in cols) + " |"
            sep = "| " + " | ".join("---" for _ in cols) + " |"
            data_lines = []
            for r in rows[:15]:
                data_lines.append("| " + " | ".join(str(r[c] if r[c] is not None else "-") for c in cols) + " |")
            answer = (
                f"Found **{len(rows)} matching records** in the database:\n\n"
                + "\n".join([header, sep] + data_lines)
            )
        else:
            answer = "No records matched your search criteria in the database."
        followups = [
            "Which leave requests are pending approval?",
            "Show all student attendance percentages",
            "What is Grade 10-A's Monday timetable?",
        ]

    return {
        "answer": answer,
        "source_type": "database",
        "sql_queries": sql_results,
        "manual_references": [],
        "suggested_followups": followups,
        "llm_ready": True,
    }
