"""PostgreSQL connection layer.

Every application connection runs with:
  - search_path = school
  - default_transaction_read_only = on  (hard stop against mutations)
  - statement_timeout = 5000 ms

The database bootstrap script (scripts/init_db.py) uses its own dedicated
connection because it must be able to CREATE objects and INSERT sample data.
"""
from __future__ import annotations

import decimal
import datetime
import json
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings


def _connection_kwargs() -> dict:
    kwargs: dict[str, Any] = {"row_factory": dict_row}
    # Neon PgBouncer pooler rejects all startup parameters in options.
    # Non-pooled or local databases accept startup options.
    if "-pooler" not in settings.database_url:
        kwargs["options"] = (
            f"-c search_path={settings.db_schema} "
            f"-c default_transaction_read_only=on "
            f"-c statement_timeout={settings.statement_timeout_ms} "
            f"-c application_name=school-ai-chatbot"
        )
    return kwargs


_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                kwargs=_connection_kwargs(),
                open=True,
            )
    return _pool


@contextmanager
def db_connection(retries: int = 2) -> Iterator[psycopg.Connection]:
    """Checkout a pooled read-only connection, reconnecting on dead sockets."""
    pool = get_pool()
    last_err: Optional[BaseException] = None
    attempt = 0
    while attempt <= retries:
        conn = pool.getconn()
        try:
            # Ensure schema search_path, timeout, and read-only mode for every session checkout
            with conn.cursor() as cur:
                cur.execute(
                    f"SET search_path TO {settings.db_schema}; "
                    f"SET default_transaction_read_only = on; "
                    f"SET statement_timeout = '{settings.statement_timeout_ms}ms';"
                )
            yield conn
            return
        except psycopg.Error as err:
            last_err = err
            try:
                conn.rollback()
            except Exception:
                pass
            if isinstance(err, psycopg.OperationalError):
                attempt += 1
                pool.putconn(conn, close=True)
                time.sleep(0.25 * attempt)
                continue
            raise
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                pass
    raise last_err if last_err else RuntimeError("database unavailable")


def serialize_value(value: Any) -> Any:
    """JSON-safe conversion for DATE/TIME/TIMESTAMP/NUMERIC/DECIMAL."""
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, decimal.Decimal):
        f = float(value)
        return f
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (datetime.timedelta,)):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return value.hex() if bytes(value).isascii() else "<binary>"
    return str(value)


def serialize_row(row: dict) -> dict:
    return {k: serialize_value(v) for k, v in row.items()}


def run_readonly_query(sql: str, params: Optional[tuple] = None) -> dict:
    """Execute a validated read-only SELECT and normalize the result rows."""
    started = time.perf_counter()
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            raise ValueError("Statement returned no result set (SELECT required).")
        rows_raw = cur.fetchall()
        columns = [d.name for d in cur.description]
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    rows = [serialize_row(r) for r in rows_raw]
    # Guard: never let a password hash escape to the API layer.
    for r in rows:
        for k in list(r.keys()):
            if "password" in k.lower() and r[k]:
                r[k] = "***MASKED***"
    return {
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "execution_time_ms": elapsed_ms,
    }


def check_connectivity() -> dict:
    """Cheap health probe used by /api/health and startup checks."""
    try:
        with db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = %s AND table_type = 'BASE TABLE') AS tables,
                  (SELECT COUNT(*) FROM information_schema.views
                    WHERE table_schema = %s) AS views,
                  (SELECT COUNT(*) FROM users)                            AS users
                """,
                (settings.db_schema, settings.db_schema),
            )
            stats = cur.fetchone()
        return {
            "ok": True,
            "schema": settings.db_schema,
            "tables": stats["tables"],
            "views": stats["views"],
            "users": stats["users"],
        }
    except Exception as exc:  # noqa: BLE001 - surfaced as a health status
        return {"ok": False, "error": f"{type(exc).__name__}"}


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
