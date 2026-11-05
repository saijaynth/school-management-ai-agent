"""PostgreSQL access layer (read-only, one network round trip per batch).

Every batch of queries is sent as a single ``BEGIN READ ONLY; ...; COMMIT`` script
with a ``statement_timeout``. That gives a hard read-only guarantee at the
database level (writes fail even if validation were bypassed) while keeping
latency to one round trip, which matters for a remote Neon database. Settings
are applied with ``SET LOCAL`` so it is also safe behind PgBouncer-style poolers
that do not preserve session state.

Parameters are bound client-side by psycopg (properly escaped literals), which is
what allows the whole script to travel in one simple-protocol message.
"""
from __future__ import annotations

import datetime
import decimal
import json
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, List, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row, tuple_row
from psycopg_pool import ConnectionPool

from .config import settings
from .sql_validator import validate_sql

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                kwargs={"application_name": "school-ai-chatbot"},
                max_idle=240,
                max_lifetime=1800,
                timeout=25,
                open=False,
            )
            pool.open()
            _pool = pool
    return _pool


def serialize_value(value: Any) -> Any:
    """JSON-safe conversion for DATE/TIME/TIMESTAMP/NUMERIC values."""
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return "<binary>"
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return str(value)


def _normalize(sql: str, columns: List[str], raw_rows: list, elapsed_ms: float) -> dict:
    rows = []
    for r in raw_rows:
        if isinstance(r, dict):
            row = {c: serialize_value(r.get(c)) for c in columns}
        else:
            row = {c: serialize_value(v) for c, v in zip(columns, r)}
        for k in row:
            if "password" in k.lower() and row[k]:
                row[k] = "***MASKED***"
        rows.append(row)
    return {"sql": sql, "columns": columns, "rows": rows, "row_count": len(rows), "execution_time_ms": elapsed_ms}


def _execute_batch(queries: Sequence[Tuple[str, Optional[tuple]]]) -> List[dict]:
    started = time.perf_counter()
    with get_pool().connection() as conn:
        conn.autocommit = True
        conn.row_factory = tuple_row
        try:
            with psycopg.ClientCursor(conn) as cur:
                statements = [cur.mogrify(sql, params) for sql, params in queries]
                script = (
                    "BEGIN READ ONLY; "
                    f"SET LOCAL search_path TO {settings.db_schema}; "
                    f"SET LOCAL statement_timeout = {int(settings.statement_timeout_ms)}; "
                    + "; ".join(statements)
                    + "; COMMIT"
                )
                cur.execute(script)
                sets: List[Tuple[List[str], list]] = []
                while True:
                    if cur.description is not None:
                        sets.append(([d.name for d in cur.description], cur.fetchall()))
                    if not cur.nextset():
                        break
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
    elapsed = round((time.perf_counter() - started) * 1000.0, 2)
    if len(sets) != len(queries):
        raise RuntimeError("Unexpected result-set count from batched query.")
    return [_normalize(sql, cols, rows, elapsed) for (sql, _), (cols, rows) in zip(queries, sets)]


def run_readonly_queries(queries: Sequence[Tuple[str, Optional[tuple]]], retries: int = 2) -> List[dict]:
    """Validate and run several SELECTs in one round trip. Retries dead connections."""
    checked: List[Tuple[str, Optional[tuple]]] = []
    for sql, params in queries:
        verdict = validate_sql(sql)
        if not verdict.valid:
            raise ValueError(f"Blocked SQL: {verdict.error}")
        checked.append((verdict.sql, params))
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return _execute_batch(checked)
        except (psycopg.OperationalError, psycopg.InterfaceError, TimeoutError) as err:
            last_err = err
            time.sleep(0.3 * (attempt + 1))
    raise last_err if last_err else RuntimeError("database unavailable")


def run_readonly_query(sql: str, params: Optional[tuple] = None, retries: int = 2) -> dict:
    return run_readonly_queries([(sql, params)], retries=retries)[0]


def check_connectivity() -> dict:
    """Cheap health probe used by /api/health and startup checks."""
    try:
        res = run_readonly_query(
            """
            SELECT
              (SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE') AS tables,
              (SELECT COUNT(*) FROM information_schema.views
                WHERE table_schema = %s) AS views,
              (SELECT COUNT(*) FROM school.users) AS users
            """,
            (settings.db_schema, settings.db_schema),
            retries=1,
        )
        return {"ok": True, "schema": settings.db_schema, **res["rows"][0]}
    except Exception as exc:  # noqa: BLE001 - surfaced as a health status
        return {"ok": False, "error": type(exc).__name__}


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def db_connection():
    """Context manager yielding a pooled connection with dict_row factory."""
    with get_pool().connection() as conn:
        conn.row_factory = dict_row
        yield conn



def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
