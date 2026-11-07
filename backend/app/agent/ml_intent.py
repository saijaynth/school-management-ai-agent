"""Intent classification engine using Scikit-Learn.

Trains a fast, calibrated TF-IDF + LogisticRegression pipeline on startup or loads
cached weights from backend/data/intent_model.joblib.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_PATH = MODEL_DIR / "intent_model.joblib"

TRAINING_DATA: List[Tuple[str, str]] = [
    # --- PENDING LEAVES ---
    ("Which leave requests are pending approval?", "pending_leaves"),
    ("Show all pending leave requests", "pending_leaves"),
    ("Are there any leaves awaiting approval?", "pending_leaves"),
    ("Pending leaves in the queue", "pending_leaves"),
    ("Who applied for leave and is waiting for admin approval?", "pending_leaves"),
    ("Show pending leaves for admin", "pending_leaves"),
    ("List pending leave applications", "pending_leaves"),
    ("What leave requests need review?", "pending_leaves"),
    ("v_pending_leaves report", "pending_leaves"),
    
    # --- ALL LEAVES ---
    ("Show all leave requests", "all_leaves"),
    ("Leave history for faculty and students", "all_leaves"),
    ("Show approved and rejected leaves", "all_leaves"),
    ("Who had their leave rejected?", "all_leaves"),
    ("List all leave applications", "all_leaves"),
    ("Past leave requests", "all_leaves"),
    
    # --- STUDENT ATTENDANCE ---
    ("Show all student attendance percentages", "student_attendance_summary"),
    ("Student attendance report", "student_attendance_summary"),
    ("What is the attendance percentage of all students?", "student_attendance_summary"),
    ("List students and their attendance rates", "student_attendance_summary"),
    ("v_student_attendance_report", "student_attendance_summary"),
    ("Show Meera Iyer attendance", "student_attendance_summary"),
    ("Arjun Rao attendance percentage", "student_attendance_summary"),
    ("Kiran Patel attendance", "student_attendance_summary"),
    ("Divya Nair attendance record", "student_attendance_summary"),
    ("Who has low attendance?", "student_attendance_low"),
    ("Which students have attendance below 75%?", "student_attendance_low"),
    ("Show students below 75 percent attendance", "student_attendance_low"),
    ("Show students with attendance less than 75%", "student_attendance_low"),
    ("Who is below 75%?", "student_attendance_low"),
    ("Students with 100% attendance", "student_attendance_perfect"),
    ("Who attended all classes?", "student_attendance_perfect"),
    ("Student attendance on 2026-11-02", "student_attendance_date"),
    ("Show period attendance for Grade 9-A on 2026-11-02", "student_attendance_date"),
    
    # --- FACULTY ATTENDANCE ---
    ("Faculty attendance on 2026-11-02", "faculty_attendance"),
    ("Show faculty attendance for Nov 2 2026", "faculty_attendance"),
    ("Who was present among teachers on 2026-11-02?", "faculty_attendance"),
    ("Was Sara Thomas on leave on 2026-11-02?", "faculty_attendance"),
    ("Faculty check in check out times", "faculty_attendance"),
    ("Anita Sharma check in time on 2026-11-02", "faculty_attendance"),
    ("Vijay Reddy attendance status", "faculty_attendance"),
    ("List faculty attendance records", "faculty_attendance"),
    
    # --- TIMETABLE & CLASSES ---
    ("What is Grade 10-A's Monday timetable?", "timetable_section"),
    ("Show Monday timetable for Grade 10-A", "timetable_section"),
    ("Show timetable for Grade 9-A", "timetable_section"),
    ("What is the Tuesday schedule for Grade 9-A?", "timetable_section"),
    ("When does Anita Sharma teach Mathematics?", "timetable_teacher"),
    ("Sara Thomas timetable schedule", "timetable_teacher"),
    ("Vijay Reddy classes on Monday", "timetable_teacher"),
    ("What periods are scheduled on Monday?", "timetable_section"),
    ("v_timetable for Grade 10-A", "timetable_section"),
    ("Show timetable for Grade 10-A on Monday", "timetable_section"),
    ("Wednesday timetable for Grade 10-A", "timetable_section"),
    ("Which rooms are used on Monday?", "timetable_section"),
    
    # --- CLASS TEACHER ---
    ("Who is the class teacher of Grade 10-A?", "class_teacher"),
    ("Who is the class teacher of Grade 10?", "class_teacher"),
    ("Grade 10-A class teacher", "class_teacher"),
    ("Who is the class teacher of Grade 9-A?", "class_teacher"),
    ("Who is Grade 10 class teacher?", "class_teacher"),
    
    # --- EXAMS & MARKS ---
    ("Show exam report for Grade 10 Mid Term", "exam_report"),
    ("Show marks for Grade 10 Mid Term 2026", "exam_report"),
    ("v_exam_report", "exam_report"),
    ("What did Kiran Patel score in Mathematics?", "exam_report"),
    ("Divya Nair exam marks and grades", "exam_report"),
    ("Did Divya pass all subjects in the exam?", "exam_report"),
    ("List all marks obtained in Mid Term 2026", "exam_report"),
    ("Subject marks of Grade 10 students", "exam_report"),
    
    # --- EXAM COMPARISON ---
    ("Compare Kiran Patel and Divya Nair's marks", "exam_compare"),
    ("Compare Kiran and Divya marks", "exam_compare"),
    ("Compare Kiran and Divya exam performance", "exam_compare"),
    ("Who scored higher Kiran or Divya?", "exam_compare"),
    ("Compare marks between Kiran Patel and Divya Nair", "exam_compare"),
    ("Rank list for Grade 10 Mid Term", "exam_compare"),
    ("Compare progress reports of Kiran and Divya", "exam_compare"),
    
    # --- PROGRESS REPORTS ---
    ("Show progress report for Kiran Patel", "progress_report"),
    ("Divya Nair progress report", "progress_report"),
    ("Report card for Divya Nair", "progress_report"),
    ("Kiran Patel class rank and percentage", "progress_report"),
    ("Show student progress reports", "progress_report"),
    
    # --- HOMEWORK ---
    ("What homework is assigned for Grade 9-A?", "homework_list"),
    ("What homework and assignments are assigned to Grade 9-A?", "homework_list"),
    ("Show homework and assignments for Grade 9", "homework_list"),
    ("List homework for Grade 9", "homework_list"),
    ("Show homework for Grade 10-A", "homework_list"),
    ("Algebra Worksheet 5 details", "homework_list"),
    ("Essay My Role Model homework", "homework_list"),
    ("Who submitted Algebra Worksheet 5?", "homework_submissions"),
    ("Homework submission statuses for Grade 9-A", "homework_submissions"),
    ("Has Meera Iyer submitted homework?", "homework_submissions"),
    ("Arjun Rao homework submission status", "homework_submissions"),
    
    # --- ASSIGNMENTS ---
    ("Show assignments for Grade 10-A", "assignments_list"),
    ("List assignments for Grade 9-A", "assignments_list"),
    ("Science Project Solar Energy assignment", "assignments_list"),
    ("Geometry Assignment 1 details", "assignments_list"),
    ("Who submitted Geometry Assignment 1?", "assignments_submissions"),
    ("Assignment submissions and marks for Grade 9-A", "assignments_submissions"),
    ("Did Arjun Rao get marks on Geometry Assignment?", "assignments_submissions"),
    ("Is Meera's assignment pending?", "assignments_submissions"),
    
    # --- MULTI-TURN HOMEWORK + ASSIGNMENT PENDING ---
    ("Who hasn't submitted them yet?", "homework_assignment_pending"),
    ("Who has not submitted yet?", "homework_assignment_pending"),
    ("Who hasn't submitted both?", "homework_assignment_pending"),
    ("Pending submissions for Grade 9-A", "homework_assignment_pending"),
    ("Which students have pending homework or assignments?", "homework_assignment_pending"),
    ("Who hasn't submitted the homework and assignment?", "homework_assignment_pending"),
    
    # --- USERS, ROLES & PROFILES ---
    ("Show all users in the system", "users_list"),
    ("List all users and roles", "users_list"),
    ("Who are the users?", "users_list"),
    ("List of logins", "users_list"),
    ("Show student profiles", "students_list"),
    ("List all students with admission numbers", "students_list"),
    ("Show guardian details of students", "students_list"),
    ("List all faculty members", "faculty_list"),
    ("Show teachers qualifications and designations", "faculty_list"),
    ("Employee numbers of faculty", "faculty_list"),
    ("Show classes and sections", "academic_structure"),
    ("List subjects and classrooms", "academic_structure"),
    
    # --- NOTIFICATIONS ---
    ("Show all notifications", "notifications"),
    ("List unread notifications", "notifications"),
    ("Notifications for users", "notifications"),
    
    # --- USER MANUAL PROCEDURAL & POLICY ---
    ("How do I add a new student?", "manual_workflow"),
    ("How to add a student according to the manual?", "manual_workflow"),
    ("What is the foreign key insert order?", "manual_workflow"),
    ("How are leave requests approved?", "manual_workflow"),
    ("Explain the leave approval workflow", "manual_workflow"),
    ("Why are users deactivated instead of deleted?", "manual_workflow"),
    ("Why can't we delete users?", "manual_workflow"),
    ("Explain no cascading deletes", "manual_workflow"),
    ("What is the year-end routine?", "manual_workflow"),
    ("Explain the 4-step year-end routine", "manual_workflow"),
    ("How do I backup and restore the database?", "manual_workflow"),
    ("pg_dump backup instructions", "manual_workflow"),
    ("How to mark student attendance according to manual?", "manual_workflow"),
    ("How does the grading scale work?", "manual_workflow"),
    ("What are the 3 roles in the system?", "manual_workflow"),
    
    # --- MUTATION REFUSALS ---
    ("Delete all rejected leave requests", "mutation_refusal"),
    ("Delete rejected leaves", "mutation_refusal"),
    ("Drop table students", "mutation_refusal"),
    ("Update student marks to 100", "mutation_refusal"),
    ("Insert new user into users table", "mutation_refusal"),
    ("Can you remove Meera Iyer from students?", "mutation_refusal"),
    ("Delete absent attendance records", "mutation_refusal"),
    ("Approve Meera Iyer's leave in database", "mutation_refusal"),
    
    # --- EMPTY RESULT QUESTIONS ---
    ("Show Grade 9 exam marks", "exam_report"),
    ("What exams does Grade 9 have?", "exam_report"),
    ("Wednesday timetable for Grade 10-A", "timetable_section"),

    # --- PERSON PROFILE / WHO IS ---
    ("who is divya", "person_profile"),
    ("who is divya nair", "person_profile"),
    ("who is arjun", "person_profile"),
    ("who is arjun rao", "person_profile"),
    ("who is meera", "person_profile"),
    ("who is meera iyer", "person_profile"),
    ("who is kiran", "person_profile"),
    ("who is kiran patel", "person_profile"),
    ("who is anita", "person_profile"),
    ("who is anita sharma", "person_profile"),
    ("who is vijay", "person_profile"),
    ("who is vijay reddy", "person_profile"),
    ("who is sara", "person_profile"),
    ("who is sara thomas", "person_profile"),
    ("who is admin", "person_profile"),
    ("who is admin01", "person_profile"),
    ("tell me about divya", "person_profile"),
    ("tell me about divya nair", "person_profile"),
    ("tell me about student arjun", "person_profile"),
    ("tell me about arjun rao", "person_profile"),
    ("tell me about meera", "person_profile"),
    ("tell me about meera iyer", "person_profile"),
    ("tell me about kiran patel", "person_profile"),
    ("tell me about teacher anita", "person_profile"),
    ("tell me about vijay reddy", "person_profile"),
    ("tell me about sara thomas", "person_profile"),
    ("profile of divya", "person_profile"),
    ("profile of divya nair", "person_profile"),
    ("profile of arjun rao", "person_profile"),
    ("profile of meera iyer", "person_profile"),
    ("profile of kiran patel", "person_profile"),
    ("profile of vijay reddy", "person_profile"),
    ("profile of anita sharma", "person_profile"),
    ("profile of sara thomas", "person_profile"),
    ("details of divya", "person_profile"),
    ("details of divya nair", "person_profile"),
    ("details of arjun rao", "person_profile"),
    ("details of anita sharma", "person_profile"),
    ("details of vijay reddy", "person_profile"),
    ("details of sara thomas", "person_profile"),
    ("information about divya nair", "person_profile"),
    ("information on arjun rao", "person_profile"),
    ("give me details of meera iyer", "person_profile"),
    ("give me info on kiran patel", "person_profile"),
    ("student details of divya", "person_profile"),
    ("student info for arjun", "person_profile"),
    ("teacher info for anita", "person_profile"),
    ("faculty details for vijay", "person_profile"),
]


def generate_universal_sentence_variations() -> List[Tuple[str, str]]:
    """Generates hundreds of rich natural language sentence variations grounded in the database."""
    samples: List[Tuple[str, str]] = []
    
    # 1. Students
    students = [
        ("Divya", "Divya Nair", "Grade 10-A", "ADM2025004"),
        ("Arjun", "Arjun Rao", "Grade 9-A", "ADM2025001"),
        ("Meera", "Meera Iyer", "Grade 9-A", "ADM2025002"),
        ("Kiran", "Kiran Patel", "Grade 10-A", "ADM2025003"),
    ]
    
    for short, full, sec, adm in students:
        patterns = [
            f"who is {short.lower()}",
            f"who is {full.lower()}",
            f"tell me about {short.lower()}",
            f"tell me about {full.lower()}",
            f"profile of {short.lower()}",
            f"profile of {full.lower()}",
            f"can you give me details on {short.lower()}",
            f"could you check information for student {full.lower()}",
            f"what class is {short.lower()} in",
            f"which section does {full.lower()} study in",
            f"what is the roll number of {short.lower()}",
            f"admission number for {full.lower()}",
        ]
        target = f"intent: person_profile | person: {full} | section: {sec} | role: student"
        for p in patterns:
            samples.append((p, target))

        # Attendance variations
        att_patterns = [
            f"what is {short.lower()}'s attendance rate",
            f"how many classes has {full.lower()} attended",
            f"is {short.lower()} present or absent",
            f"attendance report for {full.lower()}",
            f"attendance percentage of {short.lower()}",
            f"did {full.lower()} attend all lectures",
        ]
        att_target = f"intent: student_attendance_summary | student: {full}"
        for p in att_patterns:
            samples.append((p, att_target))

        # Exam / Mark variations
        exam_patterns = [
            f"what did {short.lower()} score in exams",
            f"marks obtained by {full.lower()}",
            f"report card of {short.lower()}",
            f"grade and marks for {full.lower()}",
            f"progress report of {short.lower()}",
            f"did {full.lower()} pass the exam",
        ]
        exam_target = f"intent: exam_report | student: {full}"
        for p in exam_patterns:
            samples.append((p, exam_target))

        # Homework / Assignment variations
        hw_patterns = [
            f"has {short.lower()} submitted homework",
            f"homework submission status for {full.lower()}",
            f"did {short.lower()} turn in the assignment",
            f"is {full.lower()}'s assignment pending",
        ]
        hw_target = f"intent: homework_submissions | student: {full}"
        for p in hw_patterns:
            samples.append((p, hw_target))

    # 2. Faculty
    faculty = [
        ("Anita", "Anita Sharma", "Senior Teacher", "Mathematics"),
        ("Vijay", "Vijay Reddy", "Assistant Teacher", "Science"),
        ("Sara", "Sara Thomas", "Teacher", "English"),
    ]
    for short, full, desig, subj in faculty:
        patterns = [
            f"who is {short.lower()}",
            f"who is {full.lower()}",
            f"tell me about teacher {short.lower()}",
            f"profile of {full.lower()}",
            f"what does {short.lower()} teach",
            f"which subject is taught by {full.lower()}",
            f"designation of {short.lower()}",
            f"qualification of teacher {full.lower()}",
        ]
        target = f"intent: person_profile | person: {full} | designation: {desig} | subject: {subj} | role: faculty"
        for p in patterns:
            samples.append((p, target))

        # Faculty schedule & attendance
        fac_patterns = [
            f"when does {short.lower()} teach on monday",
            f"what is {full.lower()}'s timetable on monday",
            f"which room does {short.lower()} teach in",
            f"schedule of {full.lower()} on monday",
            f"was {short.lower()} present on 2026-11-02",
            f"check in time for {full.lower()} on nov 2",
        ]
        fac_target = f"intent: timetable_teacher | faculty: {full}"
        for p in fac_patterns:
            samples.append((p, fac_target))

    # 3. Sections & Academic
    sections = [
        ("Grade 9-A", "Anita Sharma", "Room 101"),
        ("Grade 10-A", "Vijay Reddy", "Room 102"),
    ]
    for sec, teacher, room in sections:
        sec_patterns = [
            f"who is the class teacher of {sec.lower()}",
            f"class teacher for {sec.lower()}",
            f"what is the monday schedule for {sec.lower()}",
            f"timetable for {sec.lower()} on monday",
            f"what homework is assigned to {sec.lower()}",
            f"list all assignments for {sec.lower()}",
            f"which classroom does {sec.lower()} use",
        ]
        for p in sec_patterns:
            samples.append((p, f"intent: timetable_section | section: {sec}"))

    # 4. Superlatives, Aggregates & Policy
    general_queries = [
        ("who is below 75% attendance?", "intent: student_attendance_low | threshold: 75"),
        ("which students have attendance shortage?", "intent: student_attendance_low"),
        ("who has perfect 100% attendance?", "intent: student_attendance_perfect"),
        ("who got first rank in class?", "intent: progress_report | rank: 1"),
        ("top performer in grade 10 mid term", "intent: progress_report | rank: 1"),
        ("compare kiran and divya exam performance", "intent: exam_compare | students: Kiran Patel, Divya Nair"),
        ("who scored higher between kiran and divya?", "intent: exam_compare | students: Kiran Patel, Divya Nair"),
        ("who hasn't submitted their homework yet?", "intent: homework_assignment_pending"),
        ("pending submissions for algebra worksheet", "intent: homework_submissions"),
        ("which leave requests are awaiting admin approval?", "intent: pending_leaves | status: Pending"),
        ("why shouldn't we delete rejected leave requests?", "intent: manual_workflow | topic: no_delete"),
        ("how do i backup the postgresql database?", "intent: manual_workflow | topic: backup"),
        ("explain the 4-step year-end routine", "intent: manual_workflow | topic: year_end_routine"),
        ("how to add a new student step by step", "intent: manual_workflow | topic: add_student"),
        ("how does admin approve or reject leave?", "intent: manual_workflow | topic: leave_approval"),
        ("how many students are currently enrolled?", "intent: student_count"),
        ("total count of faculty members", "intent: faculty_list"),
        ("list all classrooms and their seating capacity", "intent: academic_structure"),
        ("who is the system administrator?", "intent: person_profile | person: Ramesh Kumar | role: admin"),
        ("profile of admin_ramesh", "intent: person_profile | person: Ramesh Kumar | role: admin"),
    ]
    for q, t in general_queries:
        samples.append((q, t))

    return samples


_pipeline: Pipeline | None = None
_lock = threading.Lock()


def get_intent_classifier() -> Pipeline:
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.is_file():
            try:
                _pipeline = joblib.load(MODEL_PATH)
                return _pipeline
            except Exception:
                pass

        # Train new model on base dataset plus universal sentence variations
        texts = [t[0] for t in TRAINING_DATA]
        labels = [t[1] for t in TRAINING_DATA]
        for q, t in generate_universal_sentence_variations():
            lbl = t.split("|")[0].replace("intent:", "").strip()
            texts.append(q)
            labels.append(lbl)

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 3),
                        analyzer="word",
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=10.0,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(texts, labels)
        try:
            joblib.dump(pipeline, MODEL_PATH)
        except Exception:
            pass

        _pipeline = pipeline
        return _pipeline


def predict_intent(text: str) -> Tuple[str, float]:
    """Predict query intent and confidence score with semantic routing."""
    lowered = text.lower()
    
    # 1. Mutation Refusal Security Guardrail
    if (
        "delete" in lowered
        or "drop table" in lowered
        or "truncate" in lowered
        or "alter table" in lowered
        or "insert into" in lowered
    ) and not ("why" in lowered or "shouldn't" in lowered or "should not" in lowered or "according to" in lowered):
        return "mutation_refusal", 1.0

    # 2. Manual / Procedural Questions (ONLY when asking explicit policy or how-to instructions)
    if (
        "year-end routine" in lowered
        or "year end routine" in lowered
        or "why users are deactivated" in lowered
        or "why shouldn't we delete" in lowered
        or "why should not we delete" in lowered
        or "deactivated instead of deleted" in lowered
        or "how do i add a" in lowered
        or "how to add a" in lowered
        or "how to approve" in lowered
        or "how are leave requests approved" in lowered
        or "leave approval workflow" in lowered
        or "apply for leave" in lowered
        or "how can a student apply" in lowered
        or "how to apply for leave" in lowered
        or "enter exam marks" in lowered
        or "how do i enter exam marks" in lowered
        or "foreign key order" in lowered
        or "how do i backup" in lowered
        or "backup and restore" in lowered
        or "pg_dump" in lowered
        or "according to the manual" in lowered
    ):
        return "manual_workflow", 1.0

    # 3. Domain Context Indicators
    has_attendance = any(w in lowered for w in ["attendance", "absent", "present", "below 75", "attended", "missed", "shortage", "75%"])
    has_leave = any(w in lowered for w in ["leave", "leaves", "sick leave", "casual leave", "annual leave"])
    has_homework = any(w in lowered for w in ["homework", "assignment", "submission", "worksheet", "project"])
    has_exam = any(w in lowered for w in ["exam", "marks", "score", "scores", "progress report", "mid term", "rank list"]) or (
        "grade" in lowered and not any(g in lowered for g in ["grade 9", "grade 10", "grade-9", "grade-10", "9a", "10a", "9-a", "10-a"])
    )

    # 4. Student Counts and Rosters
    if any(p in lowered for p in ["how many student", "how many pupils", "number of student", "total student", "count of student"]):
        return "student_count", 1.0
    if ("student" in lowered or "students" in lowered) and any(p in lowered for p in ["how many", "count", "number of", "total"]) and not (has_attendance or has_exam or has_leave or has_homework):
        return "student_count", 1.0
    if ("student" in lowered or "students" in lowered) and not (has_attendance or has_exam or has_leave or has_homework):
        if any(p in lowered for p in ["list", "who are", "show", "profiles", "enrolled", "names", "in 9a", "in 10a", "in grade 9", "in grade 10", "roster"]):
            return "students_list", 1.0

    # 5. Faculty Counts and Rosters
    if any(p in lowered for p in ["how many teacher", "how many faculty", "number of teacher", "total teacher", "who are the teacher", "list teacher", "show faculty", "faculty list", "who teaches"]):
        if not ("timetable" in lowered or "schedule" in lowered or "when" in lowered):
            return "faculty_list", 1.0

    # 6. Classrooms & Academic Structure
    if any(p in lowered for p in ["how many room", "how many classroom", "list room", "list classroom", "what room", "which room", "classrooms"]):
        return "academic_structure", 1.0

    # 7. Leave Management
    if "pending leave" in lowered or "leave requests are pending" in lowered or "leaves are pending" in lowered:
        return "pending_leaves", 1.0
    if has_leave and not has_attendance:
        return "all_leaves", 1.0

    # 8. Homework & Assignment Submissions
    if (
        "who hasn't submitted" in lowered
        or "who has not submitted" in lowered
        or "submitted them yet" in lowered
        or "pending submissions" in lowered
        or "who hasn't turned in" in lowered
        or "who has not turned in" in lowered
        or "turned in yet" in lowered
    ):
        return "homework_assignment_pending", 1.0
    if "homework" in lowered and "assignment" in lowered:
        return "homework_list", 1.0

    # 9. Class Teacher
    if "class teacher" in lowered and "timetable" not in lowered:
        return "class_teacher", 1.0

    # 10. Timetable & Assigned Courses
    if "timetable" in lowered or "schedule" in lowered or "course" in lowered or "assigned" in lowered:
        if any(f in lowered for f in ["anita", "sharma", "vijay", "reddy", "sara", "thomas", "teacher schedule", "faculty schedule"]):
            return "timetable_teacher", 1.0
        return "timetable_section", 1.0

    # 11. Exam Comparisons
    if ("compare" in lowered or "vs" in lowered) and ("kiran" in lowered or "divya" in lowered or "marks" in lowered):
        return "exam_compare", 1.0

    # 12. Attendance Specifics
    if has_attendance:
        if "below 75" in lowered or "less than 75" in lowered or "under 75" in lowered or "shortage" in lowered or "low" in lowered:
            return "student_attendance_low", 1.0
        if "faculty" in lowered or "teacher" in lowered:
            return "faculty_attendance", 1.0
        return "student_attendance_summary", 1.0

    # 13. Person Profile / Identity / Who is / Tell me about
    if any(p in lowered for p in ["who is", "who are", "tell me about", "profile of", "details of", "information on", "info on", "about student", "about teacher", "about faculty", "who was"]):
        if not ("class teacher" in lowered or "users" in lowered or "students" in lowered or "teachers" in lowered or "leaves" in lowered):
            return "person_profile", 1.0

    # 14. Performance Ranking / Top Student / Rank
    if any(p in lowered for p in ["top student", "first rank", "highest marks", "rank 1", "rank #1", "who scored top", "topper", "highest score"]):
        return "progress_report", 1.0

    # Fallback to ML ensemble classifier
    clf = get_intent_classifier()
    pred = clf.predict([text])[0]
    probs = clf.predict_proba([text])[0]
    classes = list(clf.classes_)
    confidence = float(probs[classes.index(pred)])
    return pred, confidence
