"""Agent tools: SQL execution, schema inspection and manual retrieval.

Each tool returns a plain JSON-serializable dict so the tool result can be
appended directly to the LLM message history.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..database import run_readonly_query
from ..config import settings
from ..sql_validator import validate_sql
from ..manual_loader import lookup_manual

MAX_ROWS_RETURNED = 100


def execute_school_sql(sql_query: str, rationale: str = "") -> Dict[str, Any]:
    """Validate + execute one read-only SELECT against the `school` schema."""
    validation = validate_sql(sql_query)
    if not validation.valid:
        return {
            "tool": "execute_school_sql",
            "ok": False,
            "error": validation.error,
            "hint": "Rewrite the statement as a single read-only SELECT on schema `school`.",
        }
    sql = validation.sql
    try:
        result = run_readonly_query(sql)
    except Exception as exc:  # psycopg errors surface to the agent for self-correction
        pg_message = ""
        try:
            pg_message = str(getattr(exc, "diag", None) or "") or ""
        except Exception:
            pg_message = ""
        detail = getattr(exc, "detail", None) or ""
        hint = getattr(exc, "hint", None) or ""
        message = f"{type(exc).__name__}: {exc}"
        if detail or hint:
            message += f" | {detail} {hint}".strip()
        return {
            "tool": "execute_school_sql",
            "ok": False,
            "sql": sql,
            "rationale": rationale,
            "error": message[:800],
        }

    if result["row_count"] > MAX_ROWS_RETURNED:
        result["rows"] = result["rows"][:MAX_ROWS_RETURNED]
        result["truncated"] = True
        result["note"] = (
            f"Result truncated to first {MAX_ROWS_RETURNED} of {result['row_count']} rows. "
            "Refine the query with LIMIT or aggregations if you need a different slice."
        )
    result["ok"] = True
    result["rationale"] = rationale
    result["tool"] = "execute_school_sql"
    if validation.masking_notes:
        result["masking_notes"] = validation.masking_notes
    return result


def inspect_table_schema(table_or_view_names: List[str]) -> Dict[str, Any]:
    """Return column metadata for the requested school-schema objects."""
    if not table_or_view_names:
        return {"tool": "inspect_table_schema", "ok": False, "error": "No table names given."}

    qualified = []
    params: List[str] = []
    valid_names = []
    for name in table_or_view_names[:8]:
        short = name.strip().lower().lstrip("school. ").replace("school.", "")
        if not short.isidentifier() and not (short.startswith("v_") and short.replace("_", "").isalnum()):
            continue
        valid_names.append(short)
        params.append(short)
    if not valid_names:
        return {"tool": "inspect_table_schema", "ok": False, "error": "No valid identifiers."}

    placeholders = ", ".join(f"%s" for _ in valid_names)
    sql = f"""
    SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default
    FROM information_schema.columns c
    WHERE c.table_schema = %s AND c.table_name IN ({placeholders})
    ORDER BY c.table_name, c.ordinal_position
    """
    args = [settings.db_schema] + valid_names
    try:
        base = run_readonly_query(
            sql, tuple(args)  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        return {"tool": "inspect_table_schema", "ok": False, "error": str(exc)[:400]}

    # Foreign keys for these tables
    fk_sql = """
    SELECT kcu.table_name, kcu.column_name,
           ccu.table_name AS foreign_table, ccu.column_name AS foreign_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
      ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s
      AND kcu.table_name IN ({placeholders})
    """
    fks: List[Dict[str, Any]] = []
    try:
        fk_result = run_readonly_query(fk_sql.format(placeholders=placeholders), tuple(args))  # type: ignore[arg-type]
        fks = [
            {
                "table": r["table_name"],
                "column": r["column_name"],
                "references": f'{r["foreign_table"]}({r["foreign_column"]})',
            }
            for r in fk_result["rows"]
        ]
    except Exception:
        fks = []

    return {
        "tool": "inspect_table_schema",
        "ok": True,
        "tables": valid_names,
        "columns": base["rows"],
        "foreign_keys": fks,
    }


def lookup_school_manual(topic_or_query: str) -> Dict[str, Any]:
    """Retrieve matching sections of the School Management User Manual."""
    result = lookup_manual(topic_or_query)
    result["tool"] = "lookup_school_manual"
    result["ok"] = True
    return result


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_school_sql",
            "description": (
                "Execute ONE read-only SQL SELECT statement (optionally WITH ... SELECT) "
                "against the `school` PostgreSQL schema. Only SELECT is allowed - INSERT, "
                "UPDATE, DELETE, DDL and admin statements are rejected by the validator. "
                "Use this for every question about actual records/data; prefer the built-in "
                "views v_student_attendance_report, v_exam_report, v_pending_leaves and "
                "v_timetable when they answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The complete SQL SELECT statement to run.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short sentence on why this query is being run.",
                    },
                },
                "required": ["sql_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_table_schema",
            "description": (
                "Look up exact column names, types, nullability and foreign keys for "
                "one or more tables/views of the `school` schema via information_schema. "
                "Use when unsure about column names or constraint details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_or_view_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Table or view names without schema prefix, max 8.",
                    }
                },
                "required": ["table_or_view_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_school_manual",
            "description": (
                "Search the School Management Database - User Manual (Sections 1-13) for "
                "workflows, role permissions, constraints, maintenance procedures, "
                "troubleshooting steps and reference SQL templates. Use for any "
                "'how does the system work' or 'according to the manual' question, and "
                "when the user asks to perform changes (insert/update/delete workflows)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_or_query": {
                        "type": "string",
                        "description": "The topic, question or keywords to look up.",
                    }
                },
                "required": ["topic_or_query"],
            },
        },
    },
]
