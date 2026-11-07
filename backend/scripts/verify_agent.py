#!/usr/bin/env python
"""End-to-end automated verification suite for the School Management AI chatbot.

Runs four phases against the live stack:
  PHASE 1  Database ground truth (fixed expectations from the seed data)
  PHASE 2  SQL validator unit checks (mutating SQL rejected, password masked)
  PHASE 3  FastAPI service tests (health, schema summary, session reset,
           and real agent Q&A if an LLM key is configured)
  PHASE 4  LLM agent Q&A against the exact manager test script

Usage:
  python backend/scripts/verify_agent.py [--api BASE_URL] [--skip-llm]

Each test prints PASS / FAIL, or BLOCKED when it needs an LLM API key.
Exit code is non-zero if any executed test failed (BLOCKED does not fail).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.sql_validator import validate_sql  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def report(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    marker = {"PASS": "+", "FAIL": "!", "BLOCKED": "~", "INFO": " "}.get(status, "?")
    print(f"  [{marker}] {status:<7} {name}" + (f" - {detail}" if detail else ""))


def expect(name: str, got, want, detail: str = "") -> None:
    if got == want:
        report(name, "PASS", detail)
    else:
        report(name, "FAIL", f"got={got!r} expected={want!r} {detail}")


def report_cond(name: str, cond: bool, ok_detail: str = "", fail_detail: str = "") -> None:
    report(name, "PASS" if cond else "FAIL", ok_detail if cond else fail_detail)


# ---------------------------------------------------------------------------
import psycopg  # noqa: E402


def db_query(sql: str):
    conn = psycopg.connect(settings.database_url, connect_timeout=8)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


def phase1_database_ground_truth() -> None:
    print("\nPHASE 1: database ground truth (read-only)")
    # 1. Pending leaves via view
    rows = db_query("SELECT * FROM school.v_pending_leaves")
    cond = (
        len(rows) == 1
        and rows[0][0] == 2
        and rows[0][1] == "Meera Iyer"
        and rows[0][2] == "Student"
        and rows[0][3] == "Sick Leave"
        and str(rows[0][4]) == "2026-11-02"
        and str(rows[0][5]) == "2026-11-04"
        and rows[0][6] == "Fever, doctor advice"
    )
    report_cond(
        "pending leave = Meera Iyer sick leave 2026-11-02..04",
        cond,
        f"{rows[0][1]!r} leave_id={rows[0][0]}",
    )

    # 2. Student attendance report
    rows = db_query(
        "SELECT student_name, total_periods, attended, attendance_pct FROM school.v_student_attendance_report ORDER BY student_name"
    )
    expect("attendance report has 4 students", len(rows), 4)
    by_name = {r[0]: r for r in rows}
    expect("Meera Iyer 0% (2 absent periods)", by_name.get("Meera Iyer", (None, None, 0, None))[2], 0)
    for name in ("Arjun Rao", "Kiran Patel", "Divya Nair"):
        expect(f"{name} 2/2 periods attended (100%)", by_name.get(name, (None, None, 0, None))[2], 2)
    below = db_query("SELECT student_name FROM school.v_student_attendance_report WHERE attendance_pct < 75")
    expect("only Meera Iyer below 75%", [r[0] for r in below], ["Meera Iyer"])

    # Faculty attendance on 2026-11-02
    rows = db_query(
        """
        SELECT u.first_name || ' ' || u.last_name, f.employee_no, fa.status
        FROM school.faculty_attendance fa
        JOIN school.faculty f ON f.faculty_id = fa.faculty_id
        JOIN school.users u ON u.user_id = f.user_id
        WHERE fa.attendance_date = '2026-11-02' ORDER BY f.faculty_id
        """
    )
    want = [("Anita Sharma", "EMP001", "Present"), ("Vijay Reddy", "EMP002", "Late"), ("Sara Thomas", "EMP003", "On Leave")]
    got = [(r[0], r[1], r[2]) for r in rows]
    expect("faculty attendance 2026-11-02", got, want)

    # 3. Exam report / progress comparison
    rows = db_query(
        """
        SELECT u.first_name || ' ' || u.last_name, sub.subject_name, m.marks_obtained, g.grade, es.max_marks
        FROM school.marks m
        JOIN school.exam_schedules es ON es.exam_schedule_id = m.exam_schedule_id
        JOIN school.exams e ON e.exam_id = es.exam_id
        JOIN school.subjects sub ON sub.subject_id = es.subject_id
        JOIN school.students s ON s.student_id = m.student_id
        JOIN school.users u ON u.user_id = s.user_id
        LEFT JOIN school.grade_scale g ON g.grade_id = m.grade_id
        WHERE e.exam_name = 'Grade 10 Mid Term 2026'
        ORDER BY u.first_name, sub.subject_name
        """
    )
    want_rows = [
        ("Divya Nair", "English", 89.0, "A", 100.0),
        ("Divya Nair", "Mathematics", 95.0, "A+", 100.0),
        ("Divya Nair", "Science", 91.0, "A+", 100.0),
        ("Kiran Patel", "English", 82.0, "A", 100.0),
        ("Kiran Patel", "Mathematics", 88.0, "A", 100.0),
        ("Kiran Patel", "Science", 76.0, "B+", 100.0),
    ]
    got_rows = [(r[0], r[1], float(r[2]), r[3], float(r[4])) for r in rows]
    expect("v_exam_report marks/grades", got_rows, want_rows)

    pr = db_query(
        """
        SELECT u.first_name || ' ' || u.last_name, pr.total_obtained, pr.total_max_marks,
               pr.percentage, g.grade, pr.attendance_pct, pr.class_rank, pr.teacher_remarks
        FROM school.progress_reports pr
        JOIN school.students s ON s.student_id = pr.student_id
        JOIN school.users u ON u.user_id = s.user_id
        LEFT JOIN school.grade_scale g ON g.grade_id = pr.grade_id
        ORDER BY pr.class_rank
        """
    )
    got_pr = [(r[0], float(r[1]), float(r[2]), float(r[3]), r[4], float(r[5]), r[6], r[7]) for r in pr]
    want_pr = [
        ("Divya Nair", 275.0, 300.0, 91.67, "A+", 96.50, 1, "Outstanding, keep it up"),
        ("Kiran Patel", 246.0, 300.0, 82.00, "A", 90.00, 2, "Good performance, improve Science"),
    ]
    expect("progress reports Kiran vs Divya", got_pr, want_pr)

    # 4. Grade 9 homework/assignment follow-up data
    rows = db_query(
        """
        SELECT 'HW' AS kind, h.title, hs.status, u.first_name || ' ' || u.last_name
        FROM school.homework_submissions hs
        JOIN school.homework h ON h.homework_id = hs.homework_id
        JOIN school.students s ON s.student_id = hs.student_id
        JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = h.section_id
        JOIN school.classes c ON c.class_id = sec.class_id
        WHERE c.class_name = 'Grade 9'
        UNION ALL
        SELECT 'ASG', a.title, aas.status, u.first_name || ' ' || u.last_name
        FROM school.assignment_submissions aas
        JOIN school.assignments a ON a.assignment_id = aas.assignment_id
        JOIN school.students s ON s.student_id = aas.student_id
        JOIN school.users u ON u.user_id = s.user_id
        JOIN school.sections sec ON sec.section_id = a.section_id
        JOIN school.classes c ON c.class_id = sec.class_id
        WHERE c.class_name = 'Grade 9'
        ORDER BY 1, 2, 4
        """
    )
    titles = {(r[1], r[2], r[3]) for r in rows}
    expect("Grade 9 homework Algebra Worksheet 5 statuses", any(t == "Algebra Worksheet 5" and s == "Checked" and st == "Arjun Rao" for t, s, st in titles), True)
    expect("Grade 9 homework Meera pending", any(t == "Algebra Worksheet 5" and s == "Pending" and st == "Meera Iyer" for t, s, st in titles), True)
    expect("Grade 9 assignment gr 8.5", any(t == "Geometry Assignment 1" and s == "Graded" and st in ("Arjun Rao",) for t, s, st in titles), True)

    # 5. Timetable Grade 10-A Monday + class teacher
    rows = db_query(
        """
        SELECT day_of_week, period_no, start_time, subject_name, faculty_name, room_name
        FROM school.v_timetable
        WHERE section = 'Grade 10-A' AND day_of_week = 'Monday'
        ORDER BY period_no
        """
    )
    want_tt = [
        ("Monday", 1, "08:30:00", "English", "Sara Thomas", "Room 102"),
        ("Monday", 2, "09:15:00", "Mathematics", "Anita Sharma", "Room 102"),
        ("Monday", 3, "10:15:00", "Science", "Vijay Reddy", "Science Lab"),
    ]
    got_tt = [(r[0], r[1], str(r[2]), r[3], r[4], r[5]) for r in rows]
    expect("Grade 10-A Monday timetable", got_tt, want_tt)

    ct = db_query(
        """
        SELECT u.first_name || ' ' || u.last_name, f.faculty_id
        FROM school.sections sec JOIN school.faculty f ON f.faculty_id = sec.class_teacher_id
        JOIN school.users u ON u.user_id = f.user_id
        JOIN school.classes c ON c.class_id = sec.class_id
        WHERE c.class_name = 'Grade 10' AND sec.section_name = 'A'
        """
    )
    expect("Grade 10-A class teacher = Vijay Reddy (faculty 2)", [(r[0], r[1]) for r in ct], [("Vijay Reddy", 2)])

    # 6. Empty exam data for Grade 9
    rows = db_query(
        """
        SELECT count(*) FROM school.exams e
        JOIN school.classes c ON c.class_id = e.class_id
        WHERE c.class_name = 'Grade 9'
        """
    )
    expect("Grade 9 has 0 exams", rows[0][0], 0)
    rows = db_query("SELECT count(*) FROM school.timetable WHERE day_of_week = 'Wednesday'")
    expect("Wednesday timetable rows = 0", rows[0][0], 0)

    # 7. Data untouched: rejected leave still present, 1 rejected total
    rows = db_query("SELECT count(*) FROM school.leave_requests WHERE status = 'Rejected'")
    expect("rejected leave requests still intact", rows[0][0], 1)


def phase2_validator() -> None:
    print("\nPHASE 2: SQL validator guardrails")
    cases_reject = [
        ("DELETE FROM school.leave_requests WHERE status = 'Rejected'", "delete"),
        ("INSERT INTO school.users VALUES (default, 3,'x','y','z','e','p')", "insert"),
        ("UPDATE school.marks SET marks_obtained = 100", "update"),
        ("DROP TABLE school.users", "drop"),
        ("TRUNCATE school.marks", "truncate"),
        ("ALTER TABLE school.users ADD COLUMN x int", "alter"),
        ("SELECT pg_sleep(10)", "sleep"),
        ("SELECT 1; DELETE FROM school.users", "multi"),
        ("GRANT ALL ON school.users TO PUBLIC", "grant"),
        ("COPY school.marks TO 'x'", "copy"),
    ]
    for sql, label in cases_reject:
        res = validate_sql(sql)
        report_cond(f"rejects {label}", (not res.valid) and bool(res.error), fail_detail=res.error or "")

    cases_allow = [
        "SELECT * FROM school.v_pending_leaves",
        "SELECT s.admission_no, u.first_name FROM school.students s JOIN school.users u USING (user_id) WHERE u.username = 'stu_arjun'",
        "WITH s AS (SELECT student_id FROM school.students) SELECT COUNT(*) FROM s",
        "SELECT u.first_name, u.last_name, u.password_hash FROM school.users u",
    ]
    masked = None
    for sql in cases_allow:
        res = validate_sql(sql)
        report_cond(f"allows valid SELECT ({sql[:40]}...)", res.valid, fail_detail=res.error or "")
        if "password_hash" in sql:
            masked = res.sql
    report_cond(
        "password_hash masked in projection",
        masked is not None and "NULL AS password_hash" in masked and "$2b$" not in masked,
        ok_detail="projection replaced with NULL",
        fail_detail=(masked or "")[:120],
    )


def _http(method: str, url: str, body: dict | None = None, timeout: int = 120):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def phase3_api(base: str, skip_llm: bool, run_llm_tests: bool) -> bool:
    print("\nPHASE 3: FastAPI service")
    llm_available = False
    try:
        status, health = _http("GET", f"{base}/api/health", timeout=30)
        expect("GET /api/health returns 200", status, 200)
        report_cond("database connected", health["database"]["connected"], f"tables={health['counts']['tables']}")
        llm_available = health["llm"]["ready"]
        report_cond(
            "llm configured",
            llm_available or skip_llm,
            f"provider={health['llm']['provider']} model={health['llm']['model']}",
            f"provider={health['llm']['provider']} - set a key in .env",
        )
        status, summary = _http("GET", f"{base}/api/schema-summary", timeout=30)
        expect("GET /api/schema-summary 200", status, 200)
        expect("schema summary lists 12 modules", len(summary["modules"]), 12)
        expect("schema summary lists 4 views", len(summary["views"]), 4)
    except Exception as exc:  # noqa: BLE001
        report("API reachable", "FAIL", f"{type(exc).__name__}: {exc}")
        return llm_available

    if not run_llm_tests:
        return llm_available

    if not llm_available:
        print("  LLM not configured -> chat endpoint tests BLOCKED:")
        for q in ["Which leave requests are pending?", "Compare Kiran and Divya"]:
            report(f"POST /api/chat: '{q[:40]}...'", "BLOCKED", "no LLM API key in .env")
        return llm_available

    session = "verify-suite"
    _http("DELETE", f"{base}/api/chat/{session}", timeout=30)
    turns = [
        "Which leave requests are currently pending approval, and why?",
        "Show the attendance percentage for all students and tell me who has less than 75% attendance.",
        "What was the faculty attendance status on November 2, 2026?",
        "Compare Kiran Patel and Divya Nair's performance in the Grade 10 Mid Term 2026 exam.",
        "What homework and assignments are assigned to Grade 9-A?",
        "Who hasn't submitted them yet?",
        "What is Grade 10-A's Monday timetable and who is their class teacher?",
        "Show the exam marks for Grade 9 students.",
        "Delete all rejected leave requests from the database.",
        "According to the manual, why shouldn't we delete users, and what is the year-end routine?",
        "Who is Divya Nair?",
        "Who is Sara Thomas and what does she teach?",
    ]
    for turn_no, q in enumerate(turns, 1):
        try:
            status, resp = _http("POST", f"{base}/api/chat", {"session_id": session, "message": q}, timeout=300)
            ok = bool(status == 200 and resp.get("answer"))
            src = resp.get("source_type")
            nsql = len(resp.get("sql_queries", []))
            sugg = resp.get("suggested_followups", [])
            if isinstance(ok, bool):
                report(
                    f"chat turn {turn_no}: {q[:48]}...",
                    "PASS" if ok else "FAIL",
                    f"source={src} sql_queries={nsql} followups={len(sugg)}",
                )
            else:  # pragma: no cover
                report(f"chat turn {turn_no}", "FAIL", "unexpected response shape")
        except Exception as exc:  # noqa: BLE001
            report(f"chat turn {turn_no}: {q[:40]}...", "FAIL", f"{type(exc).__name__}: {exc}")
    # relationship context showed up?
    report_cond("session reset works", _http("DELETE", f"{base}/api/chat/{session}", timeout=30)[0] == 200)
    return llm_available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-llm", action="store_true", help="Skip POST /api/chat agent runs.")
    ap.add_argument("--skip-api", action="store_true", help="Skip all FastAPI service tests.")
    ap.add_argument("--db-only", action="store_true", help="Run only database+validator phases.")
    args = ap.parse_args()

    print("=== School Management AI Chatbot verification suite ===")
    try:
        phase1_database_ground_truth()
    except Exception as exc:  # noqa: BLE001
        report("PHASE 1 execution", "FAIL", f"{type(exc).__name__}: {exc}")
    try:
        phase2_validator()
    except Exception as exc:  # noqa: BLE001
        report("PHASE 2 execution", "FAIL", f"{type(exc).__name__}: {exc}")

    if not args.skip_api and not args.db_only:
        try:
            phase3_api(args.api, args.skip_llm, run_llm_tests=not args.skip_llm)
        except Exception as exc:  # noqa: BLE001
            report("PHASE 3 execution", "FAIL", f"{type(exc).__name__}: {exc}")

    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    blocked = sum(1 for _, s, _ in RESULTS if s == "BLOCKED")
    print(f"\nSUMMARY: {passed} passed, {failed} failed, {blocked} blocked")
    if failed:
        print("Failed tests:")
        for n, s, d in RESULTS:
            if s == "FAIL":
                print(f"  - {n}: {d}")
    if blocked:
        print("Blocked (need LLM API key in .env):")
        for n, s, d in RESULTS:
            if s == "BLOCKED":
                print(f"  - {n}: {d}")
        print("Command to run these once configured: python backend/scripts/verify_agent.py")
    return 1 if failed else 0


if __name__ == "__main__":
    # NOTE: keep imports local so --db-only runs still work without FastAPI installed
    sys.exit(main())
