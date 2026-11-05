"""Training phrasings for the topic router.

Seed phrases use placeholders for the things the entity linker masks at inference
time ({p}=person, {c}=class/section, {s}=subject, {d}=weekday, {t}=date,
{h}=homework/assignment title, {r}=room, {x}=exam, {l}=leave type). More examples
are generated from the live database schema (column names) and from the clarifications
users have confirmed (``learned_examples.jsonl``), and anything you put in
``backend/data/training_examples.json`` is merged in too:

    {"timetable": ["when does {p} teach next?"], "marks": ["how did {p} do?"]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ..config import CUSTOM_EXAMPLES_PATH, LEARNED_EXAMPLES_PATH
from .entities import MASK

SEEDS: Dict[str, List[str]] = {
    "student_info": [
        "who are the students in {c}", "list all students", "how many students are there", "how many students are in {c}",
        "show student details", "tell me about student {p}", "what is {p}'s email", "what is the phone number of {p}",
        "who is the guardian of {p}", "what is {p}'s address", "when was {p} born", "what is the date of birth of {p}",
        "what is the roll number of {p}", "admission number of {p}", "which class is {p} in", "which section does {p} belong to",
        "show me the girls in {c}", "list the boys in {c}", "student profile for {p}", "contact details of {p}",
        "what is the admission date of {p}", "students enrolled in {c}", "who studies in {c}", "give me student information",
        "is {p} a student", "parent name and phone of {p}", "gender of {p}", "students of {c}",
    ],
    "faculty_info": [
        "who are the teachers", "list all faculty members", "how many teachers are there", "tell me about teacher {p}",
        "what is the qualification of {p}", "when did {p} join the school", "what is {p}'s designation", "employee number of {p}",
        "what is the email of teacher {p}", "who teaches {s}", "which teacher handles {s}", "which subjects does {p} teach",
        "faculty details", "staff members", "show faculty profile for {p}", "teachers who teach {s} in {c}",
        "teacher contact details {p}", "what does {p} teach", "how many faculty members", "senior teachers",
        "experience of {p}", "faculty list with qualifications", "who is the {s} teacher", "teacher information",
    ],
    "class_teacher": [
        "who is the class teacher of {c}", "class teacher for {c}", "which class does {p} handle as class teacher",
        "is {p} a class teacher", "who is in charge of {c}", "list the class teachers", "who is the homeroom teacher of {c}",
        "class teacher", "which section is {p} the class teacher of", "who looks after {c}", "who is the form tutor of {c}",
        "class teachers of all sections", "section in charge for {c}",
    ],
    "directory": [
        "who is the administrator", "who is the principal", "who is the admin", "list all users", "how many users are there",
        "show all user accounts", "what is the username of {p}", "what are the usernames", "list users by role",
        "which roles exist in the system", "how many admins are there", "who are the administrators", "user roles",
        "is {p} an active user", "show active users", "when did {p} last log in", "what role does {p} have",
        "who has the admin role", "show system users", "people in the school database", "list everyone in the school",
    ],
    "subjects": [
        "what subjects are taught", "list all subjects", "how many subjects are there", "what is the subject code for {s}",
        "describe the subject {s}", "what does {s} cover", "show subject details", "which subjects are offered",
        "subjects offered in {c}", "what topics are in {s}", "subject list with teachers", "subjects taught by {p}",
        "what is {s} about", "subject codes",
    ],
    "course_allocation": [
        "who teaches {s} in {c}", "which teacher teaches {s} to {c}", "how many periods per week for {s} in {c}",
        "periods per week of {s}", "which room is {s} taught in for {c}", "course assignments for {c}",
        "what courses does {p} teach", "which classes does {p} teach", "subject allocation", "which teacher is assigned to {s}",
        "who is assigned to teach {c}", "teaching assignments", "course allocation for {c}", "weekly periods for {s}",
        "which sections does {p} teach {s} to", "allocation of teachers to subjects", "in which classroom does {p} teach {c}",
        "who handles {s} for {c}", "who teaches {c}", "teachers of {c}",
    ],
    "timetable": [
        "what is the timetable for {c}", "show {d} timetable for {c}", "{c} schedule on {d}", "what is {c}'s {d} timetable",
        "timetable of {p}", "when does {p} teach", "what classes does {p} have on {d}", "what is the schedule on {d}",
        "show the timetable", "class schedule", "which periods are there on {d}", "what is the first period on {d}",
        "what is scheduled on {d}", "which subject is in period 2 on {d}", "daily timetable", "weekly timetable for {c}",
        "which room is used on {d}", "what does {c} study on {d}", "show me the time table", "{p} schedule this week",
        "is there a class on {d}", "which periods does {c} have on {d}", "when is the {s} class", "who is teaching on {d}",
        "timetable for {r}", "what is happening in {r} on {d}", "teacher timetable", "period schedule for {c} for {d}",
    ],
    "classrooms": [
        "how many classrooms are there", "list the rooms", "what is the capacity of {r}", "which rooms are labs",
        "show all classrooms", "room capacity", "where is the science lab", "what type of room is {r}",
        "how many students can {r} hold", "list labs and classrooms", "rooms available", "what rooms does the school have",
    ],
    "periods": [
        "what time does period 1 start", "what are the period timings", "when does the first period end", "bell schedule",
        "period timings", "how long is each period", "what time is period 3", "show the periods of the day",
        "school period times", "how many periods are there in a day", "when is the break",
    ],
    "classes_sections": [
        "what classes are there", "how many classes does the school have", "list the classes and sections", "which sections exist",
        "how many sections are there", "show all grades", "what grades does the school have", "students per class",
        "how many students in each class", "list sections with class teachers", "class strength", "grades and sections",
    ],
    "academic_year": [
        "what is the current academic year", "when does the academic year start", "when does the school year end",
        "academic year dates", "what year is it in the school", "show academic years", "current session", "term dates",
    ],
    "student_attendance": [
        "show attendance percentage of students", "attendance percentage of {p}", "what is {p}'s attendance", "who has low attendance",
        "which students have attendance below 75 percent", "students with 100 percent attendance", "who was absent on {t}",
        "was {p} present on {t}", "which students were absent", "who was late", "attendance report for {c}",
        "student attendance on {t}", "how many periods did {p} attend", "why was {p} absent", "absent students",
        "who missed class on {t}", "attendance record of {p}", "student attendance report", "who attended all classes",
        "which student has the lowest attendance", "attendance of students in {c}", "late students on {t}",
        "period wise attendance of {p}", "did {p} attend class on {t}", "students absent today", "pupils attendance",
        "attendance below {t}", "how many students were present on {t}", "absent in {s} period",
    ],
    "faculty_attendance": [
        "faculty attendance on {t}", "which teachers were absent on {t}", "was {p} present on {t}", "who was late among teachers",
        "teacher attendance", "check in time of {p}", "when did {p} check out", "which teachers are on leave today",
        "staff attendance report", "show faculty attendance", "who came late on {t}", "teachers present on {t}",
        "was {p} on leave on {t}", "faculty check in and check out times", "which faculty were absent", "what time did {p} arrive",
        "teacher attendance status", "faculty attendance for {t}", "staff who were late", "teachers on half day",
    ],
    "attendance_any": [
        "show attendance", "attendance", "attendance report", "who was absent", "who is absent today", "who was present",
        "attendance status", "show me attendance records", "who is late", "attendance on {t}", "who was absent on {t}",
        "attendance details", "present and absent list", "absentees", "show the attendance sheet", "attendance for {t}",
    ],
    "leave_requests": [
        "which leave requests are pending approval", "show all leave requests", "pending leaves", "who applied for leave",
        "has {p} applied for leave", "was {p}'s leave approved", "show rejected leave requests", "approved leaves",
        "leave history of {p}", "who is on leave", "how many leave requests are pending", "{l} requests", "leave applications",
        "why was the leave rejected", "who rejected the leave", "reason for {p}'s leave", "leave requests from teachers",
        "student leave requests", "leave requests for {t}", "leaves awaiting admin approval", "what is the status of {p}'s leave",
        "who has applied for {l}", "list of leaves", "which leaves were rejected and why", "leave status", "leave request details",
        "admin remarks on leave", "leaves that need review", "how many days of leave did {p} apply for",
    ],
    "leave_types": [
        "what types of leave are there", "how many sick leaves are allowed in a year", "maximum days of {l}",
        "leave policy types", "list leave types and limits", "how many casual leaves per year", "leave entitlement",
        "what is the leave quota", "leave types available", "how many days of {l} per year",
    ],
    "homework": [
        "what homework is assigned for {c}", "show homework", "homework for {c}", "what is the due date of {h}", "homework due on {t}",
        "list homework assigned by {p}", "what homework do we have", "{s} homework for {c}", "describe the homework {h}",
        "which homework is due soon", "homework given today", "what homework was assigned on {t}", "details of {h}",
        "latest homework", "homework list", "homework for {p}", "what did {p} assign as homework", "when is {h} due",
        "homework and assignments for {c}", "pending homework",
    ],
    "homework_submissions": [
        "who submitted {h}", "who has not submitted homework", "homework submission status for {c}", "has {p} submitted homework",
        "homework submissions", "which students are yet to submit homework", "was {p}'s homework checked", "homework status of {p}",
        "who hasn't submitted {h}", "who submitted the homework late", "submitted homework list", "faculty remarks on homework of {p}",
        "pending homework submissions", "how many students submitted {h}", "did {p} do the homework", "homework not done",
        "who has submitted them", "homework checked by teacher",
    ],
    "assignments": [
        "what assignments are there for {c}", "show assignments", "list assignments for {c}", "details of {h}",
        "what is the due date of assignment {h}", "maximum marks for assignment {h}", "assignments given by {p}", "science assignments",
        "when is the project due", "assignment list", "what is the {s} assignment for {c}", "upcoming assignments",
        "describe the assignment {h}", "graded work for {c}", "projects for {c}", "assignments due on {t}",
    ],
    "assignment_submissions": [
        "who submitted {h}", "assignment submissions for {c}", "marks of {p} in {h}", "did {p} get marks on the assignment",
        "has {p} submitted the assignment", "who has not submitted the assignment", "assignment grades", "assignment feedback for {p}",
        "pending assignment submissions", "how many marks did {p} get in {h}", "was the assignment graded", "assignment marks list",
        "who has not submitted the project", "assignment status of {p}", "feedback on {p}'s assignment", "submitted assignments",
    ],
    "exams": [
        "when is the exam", "exam schedule for {c}", "what exams are scheduled", "when is the {s} exam", "mid term schedule",
        "which exams are there for {c}", "who is the invigilator for {s}", "which room is the {s} exam in", "exam dates",
        "what are the pass marks for {s}", "maximum marks in the {s} exam", "list of exams", "when does {x} start",
        "exam timetable", "what time does the {s} exam begin", "upcoming exams", "exam details", "how many exams are scheduled",
        "is there an exam for {c}", "exam hall for {x}", "exam types and weightage", "what is the weightage of exam types",
    ],
    "marks": [
        "show marks of {p}", "what did {p} score in {s}", "exam marks for {c}", "marks obtained in {x}", "results of {c}",
        "who scored the highest in {s}", "who scored the lowest", "did {p} pass", "who failed", "subject wise marks of {p}",
        "which subject did {p} score the lowest", "what grade did {p} get in {s}", "compare marks of {p} and {p}", "exam results",
        "who passed in {x}", "marks list", "show the exam report", "top scorer in {s}", "total marks of {p}",
        "how did {p} perform in the exam", "compare {p} and {p}", "performance of {p} in {s}", "{s} marks for {c}",
        "grades of students", "result of {x}", "who got the best marks", "average marks in {s}", "exam marks",
    ],
    "progress_reports": [
        "show progress report of {p}", "who is the class topper", "what is the rank of {p}", "report card for {p}",
        "rank list for {c}", "who ranked first in {x}", "percentage of {p}", "teacher remarks for {p}", "overall grade of {p}",
        "progress reports", "who got first rank", "class rank and percentage", "is the progress report published",
        "overall percentage in the mid term", "compare progress reports of {p} and {p}", "ranking of students", "toppers list",
        "what remarks did the teacher give {p}", "overall performance of {p}", "who stands first in {c}", "total marks and rank",
        "attendance percentage in {p}'s report card", "published reports", "rank 1 student",
    ],
    "grade_scale": [
        "what is the grading scale", "which grade is 85 percent", "what percentage is an A", "grade points for each grade",
        "what are the grade boundaries", "what marks are needed for A plus", "grade scale", "what does grade B mean",
        "minimum percentage for grade C", "how are grades assigned", "grade remarks",
    ],
    "notifications": [
        "show notifications for {p}", "do I have any unread notifications", "notifications", "what notifications were sent",
        "unread notifications of {p}", "what messages did {p} receive", "show alerts for {p}", "latest notifications",
        "which notifications are unread", "notification list", "was {p} notified about the progress report",
        "messages sent to {p}", "recent notifications",
    ],
    "manual": [
        "according to the manual how do I approve a leave request", "what does the user manual say about attendance",
        "how do I add a new student", "how do I enter exam marks", "what is the year end procedure", "how to schedule an exam",
        "how can a student apply for leave", "how do I reset the database to sample data", "what are the data rules",
        "can records be deleted from the database", "why shouldn't we delete records", "how do I back up the database",
        "how do faculty mark attendance", "what is the order for inserting data", "who can approve leave requests",
        "how are timetable clashes prevented", "explain the database relationships", "what is the procedure for promoting students",
        "how do I assign homework", "how do I build the timetable", "what does the manual say about security",
        "how do I create a progress report", "according to the manual", "manual instructions", "what are the roles and permissions",
        "how do I restore a backup", "what are the rules for deleting data", "what is the leave workflow", "how does leave approval work",
        "how do I set up classes and sections", "what happens at the end of the academic year", "user manual guidance",
        "how do I assign a teacher to a subject", "what should I do before year end", "what is the process for grading assignments",
    ],
    "write_request": [
        "delete all rejected leave requests", "remove student {p}", "update {p}'s marks to 90", "change the timetable for {c}",
        "insert a new student", "approve {p}'s leave request", "reject the pending leave", "mark {p} present", "drop the table",
        "add a new teacher", "edit the attendance of {p}", "delete the records from the database", "modify the grades",
        "create a new class", "assign homework to {c}", "update the class teacher of {c}", "set {p} as absent", "cancel the exam",
        "delete {p}", "change {p}'s email", "truncate all tables", "remove the notification", "add marks for {p}", "register a new student",
    ],
    "help": [
        "what can you do", "help", "what can I ask you", "what data do you have", "what information is available", "how do you work",
        "who are you", "what is this assistant", "which tables are in the database", "how many tables are there", "give me an overview",
        "what questions can I ask", "show me examples", "what topics do you know", "what is in the school database",
        "show database overview", "what are your capabilities", "guide me", "what does this database contain", "give me some sample questions",
    ],
    "greeting": [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello assistant", "hey there", "namaste", "yo",
        "hii", "heyy", "hello how are you", "how are you",
    ],
    "thanks": [
        "thanks", "thank you", "thanks a lot", "great thanks", "ok thanks", "got it", "awesome", "cool", "nice", "perfect", "bye",
        "goodbye", "see you", "that is all", "okay", "ok", "that helps", "that's helpful", "wonderful",
    ],
    "out_of_scope": [
        "what is the weather today", "tell me a joke", "who is the prime minister", "write a poem", "what is the capital of France",
        "translate hello to spanish", "what is two plus two", "recommend a movie", "write python code", "explain quantum physics",
        "who won the cricket match", "stock price of apple", "latest news", "what is the meaning of life", "solve this equation",
        "how do I cook pasta", "tell me a story", "what is the time in London", "play some music", "who is the president",
        "what is the population of India", "give me a recipe", "what is artificial intelligence", "how tall is mount everest",
        "who invented the telephone", "book a flight for me", "order pizza", "what is bitcoin", "write an essay on climate change",
        "can you sing", "what is the best phone", "tell me about football", "how to lose weight", "what is the square root of 144",
        "who is the richest man", "define photosynthesis", "do you like pizza", "how do I fix my car", "what is love",
        "what is the date today in tokyo", "tell me about the moon", "movie showtimes", "tips for exam stress", "how do I learn guitar",
    ],
}

_PLACEHOLDERS = {
    "p": MASK["person"],
    "c": MASK["section"],
    "s": MASK["subject"],
    "d": MASK["weekday"],
    "t": MASK["date"],
    "h": MASK["title"],
    "r": MASK["room"],
    "x": MASK["exam"],
    "l": MASK["leavetype"],
}


def _replace_placeholders(text: str) -> str:
    out = text
    for k, v in _PLACEHOLDERS.items():
        out = re.sub(rf"\{{{k}\}}", v, out)
    return out


def get_training_corpus() -> List[Tuple[str, str]]:
    """Return list of (masked_text, topic) pairs for training the classifier."""
    pairs: List[Tuple[str, str]] = []
    for topic, phrases in SEEDS.items():
        for p in phrases:
            masked = _replace_placeholders(p)
            pairs.append((masked, topic))
            if masked != p:
                pairs.append((p, topic))

    # Merge learned examples from JSONL if present
    if LEARNED_EXAMPLES_PATH.is_file():
        try:
            with open(LEARNED_EXAMPLES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "text" in item and "topic" in item:
                        pairs.append((item["text"], item["topic"]))
        except Exception:
            pass

    # Merge custom examples from JSON if present
    if CUSTOM_EXAMPLES_PATH.is_file():
        try:
            with open(CUSTOM_EXAMPLES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for topic, plist in data.items():
                        for p in plist:
                            pairs.append((_replace_placeholders(p), topic))
        except Exception:
            pass

    return pairs
