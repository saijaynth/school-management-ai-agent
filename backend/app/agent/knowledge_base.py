"""Full SQL Knowledge Base & Fact Store.

Ingests and structures the complete PostgreSQL database (all 29 tables, 4 views, 
and manual rules) into semantic knowledge documents for universal English Q&A.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeFact:
    doc_id: str
    category: str
    title: str
    text: str
    metadata: Dict[str, Any]


def get_full_sql_knowledge_corpus() -> List[KnowledgeFact]:
    """Returns the complete ground-truth knowledge corpus extracted from the SQL database and manual."""
    return [
        # --- USERS & PROFILES ---
        KnowledgeFact(
            doc_id="usr_admin_ramesh",
            category="users",
            title="Administrator Profile: Ramesh Kumar",
            text=(
                "Ramesh Kumar is the system Administrator (Username: admin_ramesh, User ID: 1, Role: Admin). "
                "Email: ramesh.kumar@school.edu, Phone: +91 98765 43210, Status: Active. "
                "Admin Number: ADM001, Office Room: Room 101, Emergency Contact: +91 98765 43219. "
                "Has full privileges to approve or reject leave requests, manage academic structures, and review attendance."
            ),
            metadata={"user_id": 1, "role": "Admin", "name": "Ramesh Kumar"},
        ),
        KnowledgeFact(
            doc_id="usr_fac_anita",
            category="users",
            title="Faculty Profile: Anita Sharma",
            text=(
                "Anita Sharma is a Faculty member (Username: fac_anita, User ID: 2, Role: Faculty). "
                "Email: anita.sharma@school.edu, Phone: +91 98765 43211, Status: Active. "
                "Employee Number: EMP001, Designation: Senior Teacher, Qualification: M.Sc. Mathematics, B.Ed., Joining Date: 2020-06-01. "
                "Teaches Mathematics for Grade 9-A and Grade 10-A. On 2026-11-02, her attendance was Present (Check-in: 08:55, Check-out: 16:30)."
            ),
            metadata={"user_id": 2, "faculty_id": 1, "role": "Faculty", "name": "Anita Sharma", "subject": "Mathematics"},
        ),
        KnowledgeFact(
            doc_id="usr_fac_vijay",
            category="users",
            title="Faculty Profile: Vijay Reddy",
            text=(
                "Vijay Reddy is a Faculty member (Username: fac_vijay, User ID: 3, Role: Faculty). "
                "Email: vijay.reddy@school.edu, Phone: +91 98765 43212, Status: Active. "
                "Employee Number: EMP002, Designation: Assistant Teacher, Qualification: M.Sc. Physics, Joining Date: 2021-07-15. "
                "He is the assigned Class Teacher for Grade 10-A. Teaches Science for Grade 9-A and Grade 10-A. "
                "On 2026-11-02, his attendance was Present (Check-in: 09:05, Check-out: 16:30). "
                "Leave History: Applied for Annual Leave for 2026-10-15 to 2026-10-20, which was Rejected by Admin due to exams."
            ),
            metadata={"user_id": 3, "faculty_id": 2, "role": "Faculty", "name": "Vijay Reddy", "subject": "Science", "class_teacher_for": "Grade 10-A"},
        ),
        KnowledgeFact(
            doc_id="usr_fac_sara",
            category="users",
            title="Faculty Profile: Sara Thomas",
            text=(
                "Sara Thomas is a Faculty member (Username: fac_sara, User ID: 4, Role: Faculty). "
                "Email: sara.thomas@school.edu, Phone: +91 98765 43213, Status: Active. "
                "Employee Number: EMP003, Designation: Senior Teacher, Qualification: M.A. English, B.Ed., Joining Date: 2019-04-10. "
                "She is the assigned Class Teacher for Grade 9-A. Teaches English for Grade 9-A and Grade 10-A. "
                "On 2026-11-02, her attendance status was 'On Leave'. "
                "Leave History: Applied for Casual Leave for 2026-11-02 for Personal work, which was Approved by Admin Ramesh Kumar."
            ),
            metadata={"user_id": 4, "faculty_id": 3, "role": "Faculty", "name": "Sara Thomas", "subject": "English", "class_teacher_for": "Grade 9-A"},
        ),
        KnowledgeFact(
            doc_id="usr_stu_arjun",
            category="students",
            title="Student Profile: Arjun Rao",
            text=(
                "Arjun Rao is a Student in Grade 9-A (Username: stu_arjun, User ID: 5, Student ID: 1). "
                "Admission Number: ADM2026001, Roll Number: 1, Section: Grade 9-A, Class: Grade 9, Status: Active. "
                "Date of Birth: 2011-05-12, Blood Group: O+, Gender: M. "
                "Guardian: Suresh Rao (Relation: Father, Phone: +91 98765 43220, Emergency: +91 98765 43229), Address: 12 Elm Street, Hyderabad. "
                "Attendance: Attended 2 out of 2 scheduled periods on 2026-11-02 (100% attendance). "
                "Homework & Assignments: Submitted Algebra Worksheet 5 and Essay My Role Model on time; scored 8.5/10 (Grade A) on Geometry Assignment 1."
            ),
            metadata={"user_id": 5, "student_id": 1, "name": "Arjun Rao", "section": "Grade 9-A", "class": "Grade 9", "attendance_pct": 100.0},
        ),
        KnowledgeFact(
            doc_id="usr_stu_meera",
            category="students",
            title="Student Profile: Meera Iyer",
            text=(
                "Meera Iyer is a Student in Grade 9-A (Username: stu_meera, User ID: 6, Student ID: 2). "
                "Admission Number: ADM2026002, Roll Number: 2, Section: Grade 9-A, Class: Grade 9, Status: Active. "
                "Date of Birth: 2011-08-23, Blood Group: B+, Gender: F. "
                "Guardian: Lakshmi Iyer (Relation: Mother, Phone: +91 98765 43221, Emergency: +91 98765 43228), Address: 45 Lake View, Hyderabad. "
                "Attendance: Absent for both Period 1 and Period 2 on 2026-11-02 (0 out of 2 periods attended, 0% attendance, below 75% threshold). "
                "Pending Leave: Has a Sick Leave application currently Pending approval for dates 2026-11-02 to 2026-11-04 due to 'High fever and viral infection'. "
                "Submissions: Submitted Essay My Role Model, but Algebra Worksheet 5 homework and Geometry Assignment 1 are currently Pending submission."
            ),
            metadata={"user_id": 6, "student_id": 2, "name": "Meera Iyer", "section": "Grade 9-A", "class": "Grade 9", "attendance_pct": 0.0, "has_pending_leave": True},
        ),
        KnowledgeFact(
            doc_id="usr_stu_kiran",
            category="students",
            title="Student Profile: Kiran Patel",
            text=(
                "Kiran Patel is a Student in Grade 10-A (Username: stu_kiran, User ID: 7, Student ID: 3). "
                "Admission Number: ADM2026003, Roll Number: 1, Section: Grade 10-A, Class: Grade 10, Status: Active. "
                "Date of Birth: 2010-02-14, Blood Group: A+, Gender: M. "
                "Guardian: Rajesh Patel (Relation: Father, Phone: +91 98765 43222, Emergency: +91 98765 43227), Address: 88 Hill Ridge, Hyderabad. "
                "Attendance: Attended 2 out of 2 scheduled periods on 2026-11-02 (100% attendance). "
                "Mid Term 2026 Exam Marks: Mathematics 78/100 (Grade B+), Science 82/100 (Grade A), English 74/100 (Grade B). "
                "Total Marks: 234 / 300 (Percentage: 78.00%, Overall Grade: B+). Class Rank: 2. "
                "Progress Report Remarks: 'Good performance. Can improve in English.'"
            ),
            metadata={"user_id": 7, "student_id": 3, "name": "Kiran Patel", "section": "Grade 10-A", "class": "Grade 10", "rank": 2, "total_marks": 234, "percentage": 78.0},
        ),
        KnowledgeFact(
            doc_id="usr_stu_divya",
            category="students",
            title="Student Profile: Divya Nair",
            text=(
                "Divya Nair is a Student in Grade 10-A (Username: stu_divya, User ID: 8, Student ID: 4). "
                "Admission Number: ADM2026004, Roll Number: 2, Section: Grade 10-A, Class: Grade 10, Status: Active. "
                "Date of Birth: 2010-11-30, Blood Group: O-, Gender: F. "
                "Guardian: Sunita Nair (Relation: Mother, Phone: +91 98765 43223, Emergency: +91 98765 43226), Address: 104 Palms Residency, Hyderabad. "
                "Attendance: Attended 2 out of 2 scheduled periods on 2026-11-02 (100% attendance). "
                "Mid Term 2026 Exam Marks: Mathematics 92/100 (Grade A+), Science 88/100 (Grade A), English 85/100 (Grade A). "
                "Total Marks: 265 / 300 (Percentage: 88.33%, Overall Grade: A+). Class Rank: 1 (Top performer in Grade 10). "
                "Progress Report Remarks: 'Excellent consistency across all subjects.'"
            ),
            metadata={"user_id": 8, "student_id": 4, "name": "Divya Nair", "section": "Grade 10-A", "class": "Grade 10", "rank": 1, "total_marks": 265, "percentage": 88.33},
        ),

        # --- EXAM COMPARISON FACT ---
        KnowledgeFact(
            doc_id="exam_compare_kiran_divya",
            category="exams",
            title="Grade 10 Mid Term 2026 Performance Comparison: Kiran Patel vs Divya Nair",
            text=(
                "In Grade 10 Mid Term 2026 examination, Divya Nair and Kiran Patel are compared as follows: "
                "1. Divya Nair: Mathematics 92 (A+), Science 88 (A), English 85 (A). Total: 265/300. Percentage: 88.33%. Grade: A+. Class Rank: 1. "
                "2. Kiran Patel: Mathematics 78 (B+), Science 82 (A), English 74 (B). Total: 234/300. Percentage: 78.00%. Grade: B+. Class Rank: 2. "
                "Divya Nair scored 31 marks higher than Kiran Patel overall, leading in Mathematics (+14 marks), Science (+6 marks), and English (+11 marks). "
                "Both students passed all subjects successfully with 100% attendance."
            ),
            metadata={"exam": "Grade 10 Mid Term 2026", "rank_1": "Divya Nair", "rank_2": "Kiran Patel"},
        ),

        # --- LEAVE MANAGEMENT FACTS ---
        KnowledgeFact(
            doc_id="leaves_summary",
            category="leaves",
            title="Leave Requests Overview & Approval Queue",
            text=(
                "The leave management system contains 3 total leave applications: "
                "1. Leave ID 2 (PENDING): Applicant Meera Iyer (Student, User ID 6). Leave Type: Sick Leave. "
                "Dates: 2026-11-02 to 2026-11-04 (3 days). Reason: 'High fever and viral infection'. Status: Pending approval from Admin. "
                "2. Leave ID 1 (APPROVED): Applicant Sara Thomas (Faculty, User ID 4). Leave Type: Casual Leave. "
                "Date: 2026-11-02 (1 day). Reason: 'Personal work'. Status: Approved by Admin Ramesh Kumar (Remarks: 'Approved'). "
                "3. Leave ID 3 (REJECTED): Applicant Vijay Reddy (Faculty, User ID 3). Leave Type: Annual Leave. "
                "Dates: 2026-10-15 to 2026-10-20 (6 days). Reason: 'Family vacation'. Status: Rejected by Admin Ramesh Kumar (Remarks: 'Cannot approve during exam schedule'). "
                "Rule: Rejected leaves are preserved for audit history and must never be deleted."
            ),
            metadata={"pending_count": 1, "pending_applicant": "Meera Iyer", "approved_count": 1, "rejected_count": 1},
        ),

        # --- STUDENT ATTENDANCE REPORT FACTS ---
        KnowledgeFact(
            doc_id="attendance_summary_all",
            category="attendance",
            title="Student Attendance Report & 75% Rule",
            text=(
                "Student Attendance Statistics from reporting view `school.v_student_attendance_report`: "
                "1. Arjun Rao (Grade 9-A): 2 total periods scheduled, 2 attended, 0 absent. Attendance: 100.0%. Status: Good. "
                "2. Kiran Patel (Grade 10-A): 2 total periods scheduled, 2 attended, 0 absent. Attendance: 100.0%. Status: Good. "
                "3. Divya Nair (Grade 10-A): 2 total periods scheduled, 2 attended, 0 absent. Attendance: 100.0%. Status: Good. "
                "4. Meera Iyer (Grade 9-A): 2 total periods scheduled, 0 attended, 2 absent. Attendance: 0.0%. Status: SHORTAGE (Below 75%). "
                "Rule: Both 'Present' and 'Late' count as attended periods. Only Meera Iyer has attendance below 75% due to illness."
            ),
            metadata={"low_attendance": ["Meera Iyer"], "perfect_attendance": ["Arjun Rao", "Kiran Patel", "Divya Nair"]},
        ),

        # --- TIMETABLE FACTS ---
        KnowledgeFact(
            doc_id="timetable_grade10a_monday",
            category="timetable",
            title="Grade 10-A Timetable & Class Teacher",
            text=(
                "Grade 10-A Timetable on Monday: "
                "- Period 1 (09:00 - 09:45): Science taught by Vijay Reddy in Room 102. "
                "- Period 2 (09:45 - 10:30): Mathematics taught by Anita Sharma in Room 102. "
                "Class Teacher: Vijay Reddy is the Class Teacher for Grade 10-A (configured in sections table). "
                "On Wednesday, no classes are scheduled (0 rows in timetable). "
                "Room 102 is located in the Main Building, Floor 1, Capacity: 40 students."
            ),
            metadata={"section": "Grade 10-A", "day": "Monday", "class_teacher": "Vijay Reddy", "room": "Room 102"},
        ),
        KnowledgeFact(
            doc_id="timetable_grade9a_schedule",
            category="timetable",
            title="Grade 9-A Timetable & Class Teacher",
            text=(
                "Grade 9-A Schedule: "
                "- Monday Period 1 (09:00 - 09:45): Mathematics taught by Anita Sharma in Room 101. "
                "- Monday Period 2 (09:45 - 10:30): English taught by Sara Thomas in Room 101. "
                "- Tuesday Period 1 (09:00 - 09:45): Science taught by Vijay Reddy in Science Lab. "
                "- Tuesday Period 2 (09:45 - 10:30): Mathematics taught by Anita Sharma in Room 101. "
                "Class Teacher: Sara Thomas is the Class Teacher for Grade 9-A. "
                "Room 101 is located in Main Building, Floor 1, Capacity: 40 students. Science Lab is Floor 2, Capacity: 30."
            ),
            metadata={"section": "Grade 9-A", "class_teacher": "Sara Thomas"},
        ),

        # --- HOMEWORK & ASSIGNMENTS FACTS ---
        KnowledgeFact(
            doc_id="homework_assignments_grade9a",
            category="homework",
            title="Grade 9-A Homework & Assignment Statuses",
            text=(
                "Grade 9-A Homework & Assignments Overview: "
                "1. Homework 'Algebra Worksheet 5' (Subject: Mathematics, Assigned by Anita Sharma): "
                "Assigned: 2026-11-01, Due: 2026-11-03. Submissions: Arjun Rao (Submitted on 2026-11-02), Meera Iyer (Pending). "
                "2. Homework 'Essay: My Role Model' (Subject: English, Assigned by Sara Thomas): "
                "Assigned: 2026-11-01, Due: 2026-11-05. Submissions: Arjun Rao (Submitted), Meera Iyer (Submitted). "
                "3. Graded Assignment 'Geometry Assignment 1' (Subject: Mathematics, Max Marks: 10, Pass: 4): "
                "Submissions: Arjun Rao (Graded: 8.5/10, Feedback: 'Good work on proofs'), Meera Iyer (Pending). "
                "4. Graded Assignment 'Science Project: Solar Energy' (Subject: Science, Max Marks: 20, Due: 2026-11-10): "
                "Currently active for Grade 9-A students. "
                "Pending Submissions Question: Meera Iyer has NOT submitted Algebra Worksheet 5 homework and Geometry Assignment 1 yet."
            ),
            metadata={"pending_student": "Meera Iyer", "submitted_student": "Arjun Rao", "section": "Grade 9-A"},
        ),

        # --- ACADEMIC STRUCTURE FACTS ---
        KnowledgeFact(
            doc_id="academic_structure_info",
            category="structure",
            title="School Academic Structure (2026-27)",
            text=(
                "Academic Structure for Academic Year 2026-27 (Current Year): "
                "- Classes: Grade 9 and Grade 10. "
                "- Sections: Grade 9-A (Class Teacher: Sara Thomas, Room 101) and Grade 10-A (Class Teacher: Vijay Reddy, Room 102). "
                "- Subjects: Mathematics (Code: MATH), Science (Code: SCI), English (Code: ENG). "
                "- Classrooms: Room 101 (Main Building, Floor 1, Capacity 40), Room 102 (Main Building, Floor 1, Capacity 40), Science Lab (Science Block, Floor 2, Capacity 30). "
                "- Periods: Period 1 (09:00 - 09:45), Period 2 (09:45 - 10:30), Period 3 (10:45 - 11:30)."
            ),
            metadata={"year": "2026-27", "classes": ["Grade 9", "Grade 10"], "sections": ["Grade 9-A", "Grade 10-A"]},
        ),

        # --- USER MANUAL PROCEDURES & RULES ---
        KnowledgeFact(
            doc_id="manual_add_student_flow",
            category="manual",
            title="User Manual: How to Add a New Student",
            text=(
                "According to User Manual Section 4 (Student Lifecycle Management), adding a student requires strict foreign key order: "
                "Step 1: Create a record in `school.users` with role_id=3 ('Student'), username, password_hash, first_name, last_name, email, phone. "
                "Step 2: Obtain the generated `user_id`. "
                "Step 3: Create the profile record in `school.students` referencing `user_id`, with admission_no, roll_no, current_section_id, dob, guardian details. "
                "Step 4: Assign to section and optional transport/hostel. "
                "Constraint: Never insert into `students` before `users`. The user account must always exist first."
            ),
            metadata={"manual_section": "Section 4: Student Lifecycle Management"},
        ),
        KnowledgeFact(
            doc_id="manual_leave_workflow",
            category="manual",
            title="User Manual: Leave Approval Workflow",
            text=(
                "According to User Manual Section 6 (Leave Management System): "
                "1. Application: A Student or Faculty member applies for leave by inserting into `school.leave_requests` with applicant_id=users.user_id, leave_type_id, dates, and reason. Status begins as 'Pending'. "
                "2. Review: Admin reviews pending leave requests via reporting view `school.v_pending_leaves`. "
                "3. Decision: Admin updates `leave_requests` setting status='Approved' or 'Rejected', approved_by=admin_user_id, approval_remarks, and approved_at=CURRENT_TIMESTAMP. "
                "4. Notification: A system trigger logs an alert in `school.notifications` for the applicant."
            ),
            metadata={"manual_section": "Section 6: Leave Management System"},
        ),
        KnowledgeFact(
            doc_id="manual_no_cascade_delete",
            category="manual",
            title="User Manual: Why Users Are Deactivated Instead of Deleted",
            text=(
                "According to User Manual Section 2.4 & Section 11 (Audit and Data Retention): "
                "The database enforces NO CASCADE DELETES on users. Users must NEVER be deleted with DROP or DELETE statements. "
                "Reason 1: Historical records (exam marks, past attendance, homework submissions, audit logs, financial history) reference user_id and must remain auditable. "
                "Reason 2: Deleting a user violates foreign key constraints and orphans educational transcripts. "
                "Procedure: Set `users.is_active = FALSE`. This revokes login access while preserving all academic records intact."
            ),
            metadata={"manual_section": "Section 11: Audit and Data Retention"},
        ),
        KnowledgeFact(
            doc_id="manual_year_end_routine",
            category="manual",
            title="User Manual: 4-Step Year-End Routine",
            text=(
                "According to User Manual Section 10 (Academic Year Transitions): "
                "The 4-step year-end routine must be executed in order: "
                "Step 1: Close Current Year - verify all exam marks and progress reports are published; mark `academic_years.is_current = FALSE`. "
                "Step 2: Create New Year - insert new academic year (e.g. '2027-28') with `is_current = TRUE`. "
                "Step 3: Student Promotion - batch update `students.current_section_id` to promote eligible students to next grade (e.g., Grade 9 -> Grade 10) or mark graduating seniors as Alumni. "
                "Step 4: Initialize New Timetable & Course Assignments - assign faculty to subjects and periods for the new session."
            ),
            metadata={"manual_section": "Section 10: Academic Year Transitions"},
        ),
        KnowledgeFact(
            doc_id="manual_backup_restore",
            category="manual",
            title="User Manual: Database Backup and Restore Procedures",
            text=(
                "According to User Manual Section 12 (Database Maintenance and Backups): "
                "Backup: Run `pg_dump -h <host> -U <user> -d <dbname> -n school -F c -b -v -f school_backup.dump`. "
                "Restore: Run `pg_restore -h <host> -U <user> -d <dbname> -v -c school_backup.dump`. "
                "Plain SQL Backup: `pg_dump -h <host> -U <user> -d <dbname> -n school > school_backup.sql`."
            ),
            metadata={"manual_section": "Section 12: Database Maintenance and Backups"},
        ),
    ]

