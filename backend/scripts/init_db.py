#!/usr/bin/env python
"""Bootstrap the `school` schema.

Connects with DATABASE_URL, executes school_management_postgres92.sql when the
schema is missing (or --reset), then verifies the seed: 29 tables, 4 views,
8 users, 4 students, 3 faculty, 8 timetable rows, 6 marks and 1 pending leave.

Usage:
  python backend/scripts/init_db.py             # create if missing
  python backend/scripts/init_db.py --reset     # drop & reload sample data
  python backend/scripts/init_db.py --verify-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import SQL_SCRIPT_PATH, settings  # noqa: E402


def _connect(db_url: str | None = None):
    import psycopg

    return psycopg.connect(db_url or settings.database_url, autocommit=False)


def load_schema(db_url: str | None = None) -> None:
    script = SQL_SCRIPT_PATH.read_text(encoding="utf-8")
    conn = _connect(db_url)
    try:
        with conn.cursor() as cur:
            # No parameters -> psycopg3 runs the whole multi-statement script.
            cur.execute(script)
        conn.commit()
        print(f"Executed {SQL_SCRIPT_PATH.name} successfully.")
    finally:
        conn.close()


def install(db_url: str | None = None) -> bool:
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'school' AND table_type = 'BASE TABLE'),
                  (SELECT COUNT(*) FROM information_schema.views
                    WHERE table_schema = 'school')
                """
            )
            tables, views = cur.fetchone()
    if tables < 29 or views < 4:
        print(f"school schema incomplete (tables={tables}, views={views}) -> loading {SQL_SCRIPT_PATH.name}")
        load_schema(db_url)
    else:
        print(f"school schema already installed: {tables} tables, {views} views")
    return verify_health(db_url)


def verify_health(db_url: str | None = None) -> bool:
    checks = [
        ("tables", "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='school' AND table_type='BASE TABLE'", 29),
        ("views", "SELECT COUNT(*) FROM information_schema.views WHERE table_schema='school'", 4),
        ("users", "SELECT COUNT(*) FROM school.users", 8),
        ("students", "SELECT COUNT(*) FROM school.students", 4),
        ("faculty", "SELECT COUNT(*) FROM school.faculty", 3),
        ("timetable", "SELECT COUNT(*) FROM school.timetable", 8),
        ("marks", "SELECT COUNT(*) FROM school.marks", 6),
        ("pending_leaves", "SELECT COUNT(*) FROM school.leave_requests WHERE status='Pending'", 1),
    ]
    ok = True
    print("\n=== school schema health check ===")
    with _connect(db_url) as conn, conn.cursor() as cur:
        for label, sql, expected in checks:
            cur.execute(sql)
            got = cur.fetchone()[0]
            status = "OK" if got == expected else "MISMATCH"
            if got != expected:
                ok = False
            print(f"  {label:<15} expected={expected:<3} got={got:<3} {status}")
    print("=== health check PASSED ===" if ok else "=== health check FAILED ===")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap/verify the school schema.")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection URL (defaults to DATABASE_URL in .env).")
    parser.add_argument("--reset", action="store_true", help="Drop and reload sample data.")
    parser.add_argument("--verify-only", action="store_true", help="Only run the health check.")
    args = parser.parse_args()

    target_url = args.db_url or settings.database_url
    if args.verify_only:
        sys.exit(0 if verify_health(target_url) else 1)
    if args.reset:
        conn = _connect(target_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS school CASCADE")
        conn.close()
        print("Dropped school schema.")
    sys.exit(0 if install(target_url) else 1)
