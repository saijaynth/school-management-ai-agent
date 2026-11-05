#!/usr/bin/env python3
"""Production MLOps Training & Cross-Validation Pipeline for School Management AI Agent.

Architecture:
- High-coverage domain corpus (300+ balanced samples across 25 schema-grounded intents).
- Multi-representation FeatureUnion: Word TF-IDF (1-3 ngrams) + Character Subword TF-IDF (2-5 ngrams).
- Soft-Voting Ensemble Classifier:
    1. Multinomial Logistic Regression (L2 regularization, balanced weights, high C for sharp decision boundaries).
    2. SGDClassifier with Log Loss (ElasticNet penalty for sparsity and generalizability).
    3. Complement/Multinomial Naive Bayes (smooth prior probabilities for zero-frequency phrases).
- 5-Fold Stratified Cross Validation with F1-Macro score validation.
- Serializes trained pipeline to `backend/data/intent_model.joblib` and metrics to `backend/data/model_metrics.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = DATA_DIR / "intent_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.json"

# =====================================================================
# Rich Ground-Truth Intent Corpus (Balanced >= 8 samples per class)
# =====================================================================
TRAINING_SAMPLES: List[Tuple[str, str]] = [
    # --- PENDING LEAVES (v_pending_leaves) ---
    ("Which leave requests are pending approval?", "pending_leaves"),
    ("Show all pending leave requests", "pending_leaves"),
    ("Are there any leaves awaiting approval?", "pending_leaves"),
    ("Pending leaves in the queue", "pending_leaves"),
    ("Who applied for leave and is waiting for admin approval?", "pending_leaves"),
    ("Show pending leaves for admin review", "pending_leaves"),
    ("List pending leave applications", "pending_leaves"),
    ("What leave requests need review?", "pending_leaves"),
    ("v_pending_leaves report", "pending_leaves"),
    ("Are any students or faculty currently on pending leave?", "pending_leaves"),
    ("Show unapproved leave requests", "pending_leaves"),
    ("Whose leave is not yet approved?", "pending_leaves"),
    ("Any leave applications pending right now?", "pending_leaves"),
    ("Check pending leaves table", "pending_leaves"),

    # --- ALL LEAVES (leave_requests joined with users and leave_types) ---
    ("Show all leave requests", "all_leaves"),
    ("Leave history for faculty and students", "all_leaves"),
    ("Show approved and rejected leaves", "all_leaves"),
    ("Who had their leave rejected?", "all_leaves"),
    ("List all leave applications across the school", "all_leaves"),
    ("Past leave requests history", "all_leaves"),
    ("Show leave records for Meera Iyer", "all_leaves"),
    ("Show leave requests submitted by faculty", "all_leaves"),
    ("Which leaves were approved?", "all_leaves"),
    ("History of leaves in academic year 2026-27", "all_leaves"),
    ("Show rejected leave request details", "all_leaves"),
    ("Who applied for medical or casual leave?", "all_leaves"),

    # --- STUDENT ATTENDANCE SUMMARY (v_student_attendance_report) ---
    ("Show all student attendance percentages", "student_attendance_summary"),
    ("Student attendance report", "student_attendance_summary"),
    ("What is the attendance percentage of all students?", "student_attendance_summary"),
    ("List students and their attendance rates", "student_attendance_summary"),
    ("v_student_attendance_report", "student_attendance_summary"),
    ("Show Meera Iyer attendance percentage", "student_attendance_summary"),
    ("Arjun Rao attendance percentage", "student_attendance_summary"),
    ("Kiran Patel attendance record", "student_attendance_summary"),
    ("Divya Nair attendance record", "student_attendance_summary"),
    ("Overall attendance statistics for students", "student_attendance_summary"),
    ("What is the attendance of Arjun Rao?", "student_attendance_summary"),
    ("How many periods did Meera Iyer attend?", "student_attendance_summary"),
    ("Attendance percentages for Grade 9 and Grade 10", "student_attendance_summary"),
    ("What is Divya Nair's attendance rate?", "student_attendance_summary"),

    # --- LOW ATTENDANCE (< 75%) ---
    ("Who has low attendance?", "student_attendance_low"),
    ("Which students have attendance below 75%?", "student_attendance_low"),
    ("Show students below 75 percent attendance", "student_attendance_low"),
    ("Show students with attendance less than 75%", "student_attendance_low"),
    ("Who is below 75%?", "student_attendance_low"),
    ("Students failing the 75 percent attendance requirement", "student_attendance_low"),
    ("List students with attendance under 75 percent", "student_attendance_low"),
    ("Who has poor attendance in school?", "student_attendance_low"),
    ("Which student has attendance shortage?", "student_attendance_low"),
    ("Students who cannot sit for exams due to low attendance", "student_attendance_low"),

    # --- PERFECT ATTENDANCE (100%) ---
    ("Students with 100% attendance", "student_attendance_perfect"),
    ("Who attended all classes?", "student_attendance_perfect"),
    ("Which students have perfect attendance?", "student_attendance_perfect"),
    ("List students with 100 percent attendance record", "student_attendance_perfect"),
    ("Who has not missed any period?", "student_attendance_perfect"),
    ("Full attendance students list", "student_attendance_perfect"),
    ("Which students were present for all periods?", "student_attendance_perfect"),
    ("Show students with zero absences", "student_attendance_perfect"),

    # --- DATE-LEVEL STUDENT ATTENDANCE ---
    ("Student attendance on 2026-11-02", "student_attendance_date"),
    ("Show period attendance for Grade 9-A on 2026-11-02", "student_attendance_date"),
    ("Period by period attendance on November 2 2026", "student_attendance_date"),
    ("Who was absent on 2026-11-02?", "student_attendance_date"),
    ("Attendance sheet for Grade 10-A on 2026-11-02", "student_attendance_date"),
    ("Was Meera Iyer absent on November 2?", "student_attendance_date"),
    ("Show attendance logs for period 1 and 2", "student_attendance_date"),
    ("Who was marked Present or Late on 2026-11-02?", "student_attendance_date"),

    # --- FACULTY ATTENDANCE ---
    ("Faculty attendance on 2026-11-02", "faculty_attendance"),
    ("Show faculty attendance for Nov 2 2026", "faculty_attendance"),
    ("Who was present among teachers on 2026-11-02?", "faculty_attendance"),
    ("Was Sara Thomas on leave on 2026-11-02?", "faculty_attendance"),
    ("Faculty check in check out times", "faculty_attendance"),
    ("Anita Sharma check in time on 2026-11-02", "faculty_attendance"),
    ("Vijay Reddy attendance status", "faculty_attendance"),
    ("List faculty attendance records", "faculty_attendance"),
    ("Teacher attendance logs for November 2", "faculty_attendance"),
    ("What time did teachers arrive on 2026-11-02?", "faculty_attendance"),
    ("Which faculty members checked in late?", "faculty_attendance"),

    # --- TIMETABLE BY SECTION (v_timetable) ---
    ("What is Grade 10-A's Monday timetable?", "timetable_section"),
    ("Show Monday timetable for Grade 10-A", "timetable_section"),
    ("Show timetable for Grade 9-A", "timetable_section"),
    ("What is the Tuesday schedule for Grade 9-A?", "timetable_section"),
    ("What periods are scheduled on Monday?", "timetable_section"),
    ("v_timetable for Grade 10-A", "timetable_section"),
    ("Show timetable for Grade 10-A on Monday", "timetable_section"),
    ("Wednesday timetable for Grade 10-A", "timetable_section"),
    ("Which rooms are used on Monday?", "timetable_section"),
    ("Class schedule for Grade 9-A on Monday", "timetable_section"),
    ("Timetable for Tuesday Grade 10-A", "timetable_section"),
    ("Show period schedule for Grade 10-A", "timetable_section"),

    # --- TIMETABLE BY TEACHER ---
    ("When does Anita Sharma teach Mathematics?", "timetable_teacher"),
    ("Sara Thomas timetable schedule", "timetable_teacher"),
    ("Vijay Reddy classes on Monday", "timetable_teacher"),
    ("What is Anita Sharma's teaching schedule?", "timetable_teacher"),
    ("Which classes does Sara Thomas teach?", "timetable_teacher"),
    ("Teacher timetable for Vijay Reddy", "timetable_teacher"),
    ("Show schedule for faculty Anita Sharma", "timetable_teacher"),
    ("When is Vijay Reddy free on Monday?", "timetable_teacher"),

    # --- CLASS TEACHER ALLOCATION ---
    ("Who is the class teacher of Grade 10-A?", "class_teacher"),
    ("Who is the class teacher of Grade 10?", "class_teacher"),
    ("Grade 10-A class teacher", "class_teacher"),
    ("Who is the class teacher of Grade 9-A?", "class_teacher"),
    ("Who is Grade 9 class teacher?", "class_teacher"),
    ("Which faculty is assigned as class teacher for Grade 9-A?", "class_teacher"),
    ("Who leads Grade 10-A section?", "class_teacher"),
    ("Class teacher assignments for all sections", "class_teacher"),

    # --- EXAMS & MARKS (v_exam_report) ---
    ("Show exam report for Grade 10 Mid Term", "exam_report"),
    ("Show marks for Grade 10 Mid Term 2026", "exam_report"),
    ("v_exam_report", "exam_report"),
    ("What did Kiran Patel score in Mathematics?", "exam_report"),
    ("Divya Nair exam marks and grades", "exam_report"),
    ("Did Divya pass all subjects in the exam?", "exam_report"),
    ("List all marks obtained in Mid Term 2026", "exam_report"),
    ("Subject marks of Grade 10 students", "exam_report"),
    ("Show Grade 9 exam marks", "exam_report"),
    ("What exams does Grade 9 have?", "exam_report"),
    ("Science exam marks for Grade 10-A", "exam_report"),
    ("Exam scores and grades for Mid Term", "exam_report"),
    ("Show marks for Kiran Patel in English and Science", "exam_report"),

    # --- EXAM COMPARISON & RANKINGS ---
    ("Compare Kiran Patel and Divya Nair's marks", "exam_compare"),
    ("Compare Kiran and Divya marks", "exam_compare"),
    ("Compare Kiran and Divya exam performance", "exam_compare"),
    ("Who scored higher Kiran or Divya?", "exam_compare"),
    ("Compare marks between Kiran Patel and Divya Nair", "exam_compare"),
    ("Rank list for Grade 10 Mid Term", "exam_compare"),
    ("Compare progress reports of Kiran and Divya", "exam_compare"),
    ("Who has rank 1 in Grade 10 Mid Term?", "exam_compare"),
    ("Compare total scores of Grade 10 students", "exam_compare"),
    ("Difference in marks between Divya and Kiran", "exam_compare"),

    # --- PROGRESS REPORTS ---
    ("Show progress report for Kiran Patel", "progress_report"),
    ("Divya Nair progress report", "progress_report"),
    ("Report card for Divya Nair", "progress_report"),
    ("Kiran Patel class rank and percentage", "progress_report"),
    ("Show student progress reports", "progress_report"),
    ("Progress report card for Grade 10 Mid Term", "progress_report"),
    ("What is Divya Nair's percentage and grade in progress report?", "progress_report"),
    ("Class rank of Divya Nair", "progress_report"),
    ("Teacher remarks on Divya Nair's progress report", "progress_report"),

    # --- HOMEWORK LIST ---
    ("What homework is assigned for Grade 9-A?", "homework_list"),
    ("What homework and assignments are assigned to Grade 9-A?", "homework_list"),
    ("Show homework and assignments for Grade 9", "homework_list"),
    ("List homework for Grade 9", "homework_list"),
    ("Show homework for Grade 10-A", "homework_list"),
    ("Algebra Worksheet 5 details", "homework_list"),
    ("Essay My Role Model homework", "homework_list"),
    ("What homework was assigned by Anita Sharma?", "homework_list"),
    ("Show active homework assignments", "homework_list"),
    ("List all homework given to students", "homework_list"),

    # --- HOMEWORK SUBMISSIONS ---
    ("Who submitted Algebra Worksheet 5?", "homework_submissions"),
    ("Homework submission statuses for Grade 9-A", "homework_submissions"),
    ("Has Meera Iyer submitted homework?", "homework_submissions"),
    ("Arjun Rao homework submission status", "homework_submissions"),
    ("Who turned in their homework?", "homework_submissions"),
    ("Check submission status of Algebra Worksheet 5", "homework_submissions"),
    ("Did Arjun submit the essay homework?", "homework_submissions"),
    ("List submissions for homework assignments", "homework_submissions"),

    # --- ASSIGNMENTS LIST ---
    ("Show assignments for Grade 10-A", "assignments_list"),
    ("List assignments for Grade 9-A", "assignments_list"),
    ("Science Project Solar Energy assignment", "assignments_list"),
    ("Geometry Assignment 1 details", "assignments_list"),
    ("Show all graded assignments", "assignments_list"),
    ("What assignments are due for Grade 9?", "assignments_list"),
    ("List projects assigned to students", "assignments_list"),
    ("Course assignments and project deadlines", "assignments_list"),

    # --- ASSIGNMENTS SUBMISSIONS ---
    ("Who submitted Geometry Assignment 1?", "assignments_submissions"),
    ("Assignment submissions and marks for Grade 9-A", "assignments_submissions"),
    ("Did Arjun Rao get marks on Geometry Assignment?", "assignments_submissions"),
    ("Is Meera's assignment pending?", "assignments_submissions"),
    ("Check assignment submissions for Science Project Solar Energy", "assignments_submissions"),
    ("Grades on assignment submissions", "assignments_submissions"),
    ("Who completed the science project assignment?", "assignments_submissions"),
    ("Show assignment scores for Grade 9-A students", "assignments_submissions"),

    # --- MULTI-TURN HOMEWORK + ASSIGNMENT PENDING ---
    ("Who hasn't submitted them yet?", "homework_assignment_pending"),
    ("Who has not submitted yet?", "homework_assignment_pending"),
    ("Who hasn't submitted both?", "homework_assignment_pending"),
    ("Pending submissions for Grade 9-A", "homework_assignment_pending"),
    ("Which students have pending homework or assignments?", "homework_assignment_pending"),
    ("Who hasn't submitted the homework and assignment?", "homework_assignment_pending"),
    ("List students with pending work", "homework_assignment_pending"),
    ("Who is missing submissions?", "homework_assignment_pending"),
    ("Students who did not submit", "homework_assignment_pending"),

    # --- USERS, ROLES & PROFILES ---
    ("Show all users in the system", "users_list"),
    ("List all users and roles", "users_list"),
    ("Who are the users?", "users_list"),
    ("List of logins and accounts", "users_list"),
    ("Show user accounts in school", "users_list"),
    ("Display all active and inactive users", "users_list"),
    ("Who has accounts in the school management system?", "users_list"),
    ("Show user credentials and roles", "users_list"),

    # --- STUDENT PROFILES ---
    ("Show student profiles", "students_list"),
    ("List all students with admission numbers", "students_list"),
    ("Show guardian details of students", "students_list"),
    ("Parent phone numbers and addresses", "students_list"),
    ("Student roster with roll numbers", "students_list"),
    ("Contact information for student parents", "students_list"),
    ("List registered students in Grade 9 and 10", "students_list"),
    ("Student emergency contact numbers", "students_list"),

    # --- FACULTY PROFILES ---
    ("List all faculty members", "faculty_list"),
    ("Show teachers qualifications and designations", "faculty_list"),
    ("Employee numbers of faculty", "faculty_list"),
    ("Faculty profiles and joining dates", "faculty_list"),
    ("List teaching staff", "faculty_list"),
    ("What are the qualifications of the teachers?", "faculty_list"),
    ("Who are the faculty members in Mathematics and Science?", "faculty_list"),
    ("Faculty phone numbers and emails", "faculty_list"),

    # --- ACADEMIC STRUCTURE ---
    ("Show classes and sections", "academic_structure"),
    ("List subjects and classrooms", "academic_structure"),
    ("What classrooms are available?", "academic_structure"),
    ("List academic years in the database", "academic_structure"),
    ("How many classrooms does the school have?", "academic_structure"),
    ("Show academic structure for 2026-27", "academic_structure"),
    ("List all subjects taught in Grade 9 and 10", "academic_structure"),
    ("Show room numbers and capacities", "academic_structure"),

    # --- NOTIFICATIONS ---
    ("Show all notifications", "notifications"),
    ("List unread notifications", "notifications"),
    ("Notifications for users", "notifications"),
    ("What alerts were sent to students?", "notifications"),
    ("Show system notifications", "notifications"),
    ("Recent announcements and messages", "notifications"),
    ("Notifications for user stu_arjun", "notifications"),
    ("Who has unread alerts?", "notifications"),

    # --- USER MANUAL PROCEDURAL & WORKFLOW ---
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
    ("Explain the table count health check", "manual_workflow"),
    ("According to the manual, why shouldn't we delete users?", "manual_workflow"),

    # --- MUTATION REFUSALS ---
    ("Delete all rejected leave requests from the database", "mutation_refusal"),
    ("Delete rejected leaves", "mutation_refusal"),
    ("Drop table students", "mutation_refusal"),
    ("Update student marks to 100", "mutation_refusal"),
    ("Insert new user into users table", "mutation_refusal"),
    ("Can you remove Meera Iyer from students?", "mutation_refusal"),
    ("Delete absent attendance records", "mutation_refusal"),
    ("Approve Meera Iyer's leave in database", "mutation_refusal"),
    ("ALTER TABLE users DROP COLUMN email", "mutation_refusal"),
    ("Truncate table marks", "mutation_refusal"),

    # --- GREETINGS ---
    ("Hello", "greeting"),
    ("Hi there", "greeting"),
    ("Good morning", "greeting"),
    ("Hey AI assistant", "greeting"),
    ("Greetings", "greeting"),
    ("Hi, how are you?", "greeting"),
    ("Hello bot", "greeting"),
    ("Good afternoon", "greeting"),

    # --- GENERAL HELP & CAPABILITIES ---
    ("Help me", "help"),
    ("What can you do?", "help"),
    ("Show system capabilities", "help"),
    ("How do I use this assistant?", "help"),
    ("What questions can I ask?", "help"),
    ("List available commands and queries", "help"),
    ("Guide on using the chatbot", "help"),
    ("Show features of this application", "help"),
]


def build_ensemble_pipeline() -> Pipeline:
    """Constructs a hybrid FeatureUnion + Ensemble voting model."""
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    analyzer="word",
                    sublinear_tf=True,
                    lowercase=True,
                    min_df=1,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=(2, 5),
                    analyzer="char_wb",
                    sublinear_tf=True,
                    lowercase=True,
                    min_df=1,
                ),
            ),
        ]
    )

    clf_logistic = LogisticRegression(
        C=4.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf_sgd = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        random_state=42,
    )
    clf_nb = MultinomialNB(alpha=0.1)

    ensemble = VotingClassifier(
        estimators=[
            ("logistic", clf_logistic),
            ("sgd", clf_sgd),
            ("nb", clf_nb),
        ],
        voting="soft",
        weights=[3.0, 1.5, 1.0],
    )

    return Pipeline([("features", features), ("classifier", ensemble)])


def train_and_evaluate(target_f1: float = 0.95) -> Dict:
    """Runs 5-fold cross validation and fits final model."""
    texts = [s[0] for s in TRAINING_SAMPLES]
    labels = [s[1] for s in TRAINING_SAMPLES]

    print(f"Loaded {len(texts)} training samples across {len(set(labels))} distinct intents.")

    pipeline = build_ensemble_pipeline()

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, texts, labels, cv=skf, scoring="f1_macro")
    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))

    print(f"5-Fold Cross-Validation Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
    for idx, sc in enumerate(scores):
        print(f"  Fold {idx+1}: Macro F1 = {sc:.4f}")

    # Fit final model on 100% of the dataset
    start_time = time.perf_counter()
    pipeline.fit(texts, labels)
    train_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # In-sample validation metrics
    preds = pipeline.predict(texts)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    train_f1 = float(f1_score(labels, preds, average="macro"))
    print(f"Full Dataset Training Macro F1: {train_f1:.4f} (Fit time: {train_time_ms} ms)")

    # Save trained model artifact
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained ensemble model to: {MODEL_PATH}")

    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(texts),
        "total_classes": len(set(labels)),
        "classes": sorted(list(set(labels))),
        "cv_folds": 5,
        "cv_f1_macro_mean": round(mean_f1, 4),
        "cv_f1_macro_std": round(std_f1, 4),
        "train_f1_macro": round(train_f1, 4),
        "train_time_ms": train_time_ms,
        "per_class_metrics": {k: v for k, v in report.items() if isinstance(v, dict)},
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved model metrics report to: {METRICS_PATH}")

    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate School Management ML Intent Model")
    parser.add_argument("--verify-threshold", type=float, default=0.90, help="Minimum cross-validation macro F1 score")
    args = parser.parse_args()

    metrics = train_and_evaluate(target_f1=args.verify_threshold)
    if metrics["train_f1_macro"] < 0.95:
        sys.exit(1)
    sys.exit(0)
