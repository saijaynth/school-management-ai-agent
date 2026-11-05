"""Hand-written training pairs and phrase banks (language knowledge, not database facts).

Facts - names, classes, dates, titles - always come from the database through the generator. What is
written here is *how people phrase things* that templates alone would not capture. None of the
questions in ``eval_set.py`` appear here, so evaluation measures generalisation, not memory.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

Pair = Tuple[str, Optional[str], str]

BANK: List[Pair] = [
    ("any leave applications waiting for the principal's decision?", None, "sv_leaves | w: status eq Pending"),
    ("show me leaves that were turned down", None, "sv_leaves | w: status eq Rejected"),
    ("which leaves got approved", None, "sv_leaves | w: status eq Approved"),
    ("did the principal approve sara's leave?", None, "sv_leaves | w: applicant eq Sara Thomas"),
    ("give me every student's attendance rate", None, "sv_attendance_summary"),
    ("how many classes did meera miss", None, "sv_attendance_summary | sel: student_name, absent_periods, total_periods | w: student_name eq Meera Iyer"),
    ("who is the topper of grade 10", None, "sv_progress | sel: student_name, exam_name, class_rank, percentage, grade | w: class_rank eq 1"),
    ("which student came first in the mid term", None, "sv_progress | sel: student_name, exam_name, class_rank, percentage, grade | w: class_rank eq 1"),
    ("who teaches maths", None, "sv_course_allocation | sel: subject, section, teacher | w: subject eq Mathematics"),
    ("who takes science for 10a", None, "sv_course_allocation | sel: section, subject, teacher | w: subject eq Science ; section eq Grade 10-A"),
    ("what's the first period on monday for 9a", None, "sv_timetable | w: section eq Grade 9-A ; day_of_week eq Monday ; period_no eq 1"),
    ("is anyone teaching on wednesday", None, "sv_timetable | w: day_of_week eq Wednesday"),
    ("which teacher is the class teacher of grade nine", None, "sv_sections | sel: section, class_teacher | w: class_name eq Grade 9"),
    ("what's divya's overall percentage in the mid term", None, "sv_progress | sel: student_name, exam_name, percentage, grade | w: student_name eq Divya Nair"),
    ("did kiran pass everything", None, "sv_marks | w: student_name eq Kiran Patel"),
    ("kiran's results", None, "sv_marks | w: student_name eq Kiran Patel"),
    ("list the assignments that are still pending", None, "sv_assignment_submissions | w: status eq Pending"),
    ("which submissions have been graded", None, "sv_assignment_submissions | w: status eq Graded"),
    ("how many marks did arjun get in his geometry assignment", None, "sv_assignment_submissions | sel: assignment_title, student_name, marks_obtained, max_marks | w: student_name eq Arjun Rao ; assignment_title eq Geometry Assignment 1"),
    ("who is the invigilator for the english paper", None, "sv_exams | sel: exam_name, subject, invigilator | w: subject eq English"),
    ("which classroom hosts the science exam", None, "sv_exams | sel: exam_name, subject, room | w: subject eq Science"),
    ("are there any exams for grade 9", None, "sv_exams | w: class_name eq Grade 9"),
    ("what's the weightage of a final exam", None, "sv_exam_types | w: exam_type eq Final Exam"),
    ("what email does vijay use", None, "sv_faculty | sel: faculty_name, email | w: faculty_name eq Vijay Reddy"),
    ("how long has anita been teaching here", None, "sv_faculty | sel: faculty_name, joining_date | w: faculty_name eq Anita Sharma"),
    ("who all are in the school", None, "sv_users"),
    ("show all teachers and what they teach", None, "sv_faculty | sel: faculty_name, designation, subjects_taught"),
    ("which room is the biggest", None, "sv_classrooms | o: capacity desc | n: 1"),
    ("which parents' phone numbers do you have for grade 9-a", None, "sv_students | sel: student_name, guardian_name, guardian_phone | w: section eq Grade 9-A"),
    # follow-ups
    ("and his attendance?", "sv_marks | w: student_name eq Kiran Patel", "sv_attendance_summary | w: student_name eq Kiran Patel"),
    ("what about her homework", "sv_attendance_summary | w: student_name eq Meera Iyer", "sv_homework_submissions | w: student_name eq Meera Iyer"),
    ("same for kiran", "sv_marks | w: student_name eq Divya Nair", "sv_marks | w: student_name eq Kiran Patel"),
    ("who teaches them?", "sv_homework | w: section eq Grade 9-A", "sv_course_allocation | w: section eq Grade 9-A"),
    ("who hasn't submitted them yet?", "sv_homework | w: section eq Grade 9-A ## sv_assignments | w: section eq Grade 9-A",
     "sv_homework_submissions | w: section eq Grade 9-A ; status eq Pending ## sv_assignment_submissions | w: section eq Grade 9-A ; status eq Pending"),
    ("and who has submitted?", "sv_homework | w: section eq Grade 9-A ## sv_assignments | w: section eq Grade 9-A",
     "sv_homework_submissions | w: section eq Grade 9-A ; status ne Pending ## sv_assignment_submissions | w: section eq Grade 9-A ; status ne Pending"),
    ("and on tuesday?", "sv_timetable | w: section eq Grade 9-A ; day_of_week eq Monday", "sv_timetable | w: section eq Grade 9-A ; day_of_week eq Tuesday"),
    ("what about wednesday", "sv_timetable | w: section eq Grade 10-A ; day_of_week eq Monday", "sv_timetable | w: section eq Grade 10-A ; day_of_week eq Wednesday"),
    ("and the class teacher?", "sv_timetable | w: section eq Grade 10-A ; day_of_week eq Monday", "sv_sections | sel: section, class_teacher | w: section eq Grade 10-A"),
    ("who is the teacher for that class", "sv_students | w: section eq Grade 9-A", "sv_sections | sel: section, class_teacher | w: section eq Grade 9-A"),
    ("what about grade 9", "sv_exams | w: class_name eq Grade 10", "sv_exams | w: class_name eq Grade 9"),
    ("how about the other class", "sv_students | w: section eq Grade 9-A", "sv_students | w: section eq Grade 10-A"),
    ("only the absent ones", "sv_student_attendance | w: attendance_date eq 2026-11-02", "sv_student_attendance | w: attendance_date eq 2026-11-02 ; status eq Absent"),
    ("which of them are girls", "sv_students | w: section eq Grade 10-A", "sv_students | w: section eq Grade 10-A ; gender eq Female"),
    ("how many?", "sv_leaves | w: status eq Pending", "sv_leaves | a: count | w: status eq Pending"),
    ("why", "sv_leaves | w: status eq Rejected", "sv_leaves | sel: applicant, leave_type, status, reviewed_by, admin_remarks | w: status eq Rejected"),
    ("sort them by percentage", "sv_progress", "sv_progress | o: percentage desc"),
    # clarification replies
    ("grade 10-a", "pending: sv_timetable", "sv_timetable | w: section eq Grade 10-A"),
    ("vijay reddy", "pending: sv_timetable", "sv_timetable | w: teacher eq Vijay Reddy"),
    ("for monday i mean grade 9a", "pending: sv_timetable | w: day_of_week eq Monday", "sv_timetable | w: day_of_week eq Monday ; section eq Grade 9-A"),
    ("divya please", "pending: sv_marks", "sv_marks | w: student_name eq Divya Nair"),
    ("the science one", "pending: sv_marks", "sv_marks | w: subject eq Science"),
    ("everyone", "pending: sv_marks", "sv_marks"),
    ("all classes", "pending: sv_timetable", "sv_timetable"),
    ("all of them", "pending: sv_timetable", "sv_timetable"),
]

MANUAL_Q: List[str] = [
    "according to the manual how do I approve a leave request", "what does the user manual say about attendance", "how do I add a new student",
    "how do I enter exam marks", "what is the year end procedure", "how to schedule an exam", "how can a student apply for leave",
    "how do I reset the database to sample data", "what are the data rules", "can records be deleted from the database",
    "why shouldn't we delete records", "why shouldn't we delete rejected leave requests", "how do I back up the database",
    "how do faculty mark attendance", "what is the order for inserting data", "who can approve leave requests",
    "how are timetable clashes prevented", "explain the database relationships", "what is the procedure for promoting students to the next year",
    "how do I assign homework", "how do I build the timetable", "what does the manual say about security", "how do I create a progress report",
    "according to the manual", "what are the roles and permissions", "how do I restore a backup", "what are the rules for deleting data",
    "what is the leave workflow", "how does leave approval work", "how do I set up classes and sections", "what happens at the end of the academic year",
    "how do I assign a teacher to a subject", "what should I do before year end", "what is the process for grading assignments",
    "according to the manual, why shouldn't we delete rows that other tables refer to", "what does the manual recommend instead of deleting users",
    "how do I move students to new sections", "which tables must be filled first", "what is the foreign key insert order", "how do I check database health",
    "how do I apply for leave as a faculty member", "where do I find my progress report as a student", "how do I run the quick health check",
]

OOS_Q: List[str] = [
    "what is the weather today", "tell me a joke", "who is the prime minister of india", "write a poem about school", "what is the capital of france",
    "translate hello to spanish", "what is two plus two", "recommend a movie", "write python code to sort a list", "explain quantum physics",
    "who won the cricket match yesterday", "stock price of apple", "latest news", "what is the meaning of life", "solve x squared plus 2x plus 1",
    "how do I cook pasta", "tell me a story", "what is the time in london", "play some music", "who is the president of the usa",
    "what is the population of india", "give me a recipe for cake", "what is artificial intelligence", "how tall is mount everest",
    "who invented the telephone", "book a flight for me", "order pizza", "what is bitcoin", "write an essay on climate change",
    "can you sing", "what is the best phone to buy", "tell me about football", "how to lose weight", "what is the square root of 144",
    "who is the richest man in the world", "define photosynthesis", "do you like pizza", "how do I fix my car", "what is love",
    "tell me about the moon", "movie showtimes near me", "tips for exam stress", "how do I learn guitar", "what's the best way to study for exams",
    "who is elon musk", "what is the speed of light", "help me write an email to my boss", "give me a fun fact", "what is the tallest building",
    "how many days are in a year", "who wrote hamlet", "what is machine learning", "can you write me a cover letter", "what's trending on twitter",
    "is it going to rain tomorrow", "plan my holiday", "explain the french revolution", "what is the gdp of japan", "what is the meaning of the word serendipity",
]

HELP_Q: List[str] = [
    "what can you do", "help", "what can I ask you", "what data do you have", "what information is available", "how do you work", "who are you",
    "what is this assistant", "which tables are in the database", "how many tables are there", "give me an overview", "what questions can I ask",
    "show me examples", "what topics do you know", "what is in the school database", "show database overview", "what are your capabilities", "guide me",
    "what does this database contain", "give me some sample questions", "how can you help me", "what do you know", "tell me what you can answer",
    "i don't know what to ask", "help me get started",
]

GREET_Q: List[str] = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello assistant", "hey there", "namaste", "yo",
                      "hii", "heyy", "hello how are you", "how are you", "howdy", "hey bot"]
THANKS_Q: List[str] = ["thanks", "thank you", "thanks a lot", "great thanks", "ok thanks", "got it", "awesome", "cool", "nice", "perfect", "bye", "goodbye",
                       "see you", "that is all", "okay", "ok", "that helps", "that's helpful", "wonderful", "thank you so much", "cheers", "great"]

WRITE_Q: List[str] = [
    "delete all rejected leave requests from the database", "delete all rejected leave requests", "remove the student arjun rao", "update kiran's marks to 90",
    "change the timetable for grade 10-a", "insert a new student", "approve meera's leave request", "reject the pending leave", "mark divya present today",
    "drop the students table", "add a new teacher named ravi", "edit the attendance of meera", "delete the records from the database", "modify the grades",
    "create a new class", "assign homework to grade 9-a", "update the class teacher of grade 10-a", "set kiran as absent", "cancel the mid term exam",
    "delete divya", "change anita's email", "truncate all tables", "remove the notification", "add marks for arjun", "register a new student",
    "approve all pending leaves", "wipe the database", "update the due date of algebra worksheet 5", "give kiran 100 marks", "change meera's attendance to present",
    "remove all homework", "rename grade 9 to grade 8", "reset the passwords", "export the whole database",
]

UNSUPPORTED_CONCEPTS: List[str] = [
    "fees", "tuition fee", "salary", "payroll", "canteen menu", "bus route", "transport", "hostel", "blood group", "religion", "nationality",
    "medical records", "sports", "library books", "uniform", "disciplinary records", "parent meetings", "scholarships", "vaccination", "bus timings",
    "cafeteria", "extracurricular activities", "clubs", "awards", "photos", "id card", "lockers", "school events", "holidays", "fee dues", "complaints",
    "insurance", "mid day meal", "pta meetings", "field trips", "swimming", "cultural programs", "lost and found", "wifi password", "parking",
]
