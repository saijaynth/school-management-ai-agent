"""Strict read-only SQL validation for LLM-generated statements.

Enforced in three layers before anything reaches PostgreSQL:
1. Statement shape: exactly one statement, starting with SELECT or WITH.
2. Keyword blacklist: zero mutating/administrative keywords (checked after
   stripping string literals, comments and dollar-quoted bodies).
3. Projection masking: any reference to password_hash in the top SELECT
   projection is replaced with ``NULL AS password_hash`` so secret hashes
   can never be returned. The database layer additionally masks any value
   that might otherwise escape through another naming path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "COPY", "VACUUM", "ANALYZE", "REINDEX", "CALL",
    "COMMENT", "LOCK", "NOCHECK",
]

FORBIDDEN_FUNCTIONS = [
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file",
    "pg_ls_dir", "pg_read_binary_file", "lo_import", "lo_export",
    "dblink", "pg_terminate_backend", "pg_cancel_backend",
]

_PASSWORD_TOKEN_RE = re.compile(r"(?<!\w)(?:\w+\.)?password_hash(?![\w])", re.IGNORECASE)
_PASSWORD_ALIAS_RE = re.compile(
    r"(?P<ref>(?:\w+\.)?\bpassword_hash\b)(?P<alias>\s+AS\s+\w+)?", re.IGNORECASE
)
_DOLLAR_RE = re.compile(r"\$[A-Za-z_]*\$")


@dataclass
class SqlValidationResult:
    valid: bool
    sql: str = ""
    error: Optional[str] = None
    masking_notes: List[str] = field(default_factory=list)


def _strip_literals_and_comments(sql: str) -> str:
    """Remove string literals, quoted identifiers, dollar quotes and comments
    so keyword scans only see SQL structure, never data values such as
    ``status = 'Pending'``."""
    out: List[str] = []
    i, n = 0, len(sql)
    state = None  # None | sq | dq | line | block | dollar
    dollar_tag = ""
    while i < n:
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "-" and nxt == "-":
                state = "line"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if c == "'":
                state = "sq"
                i += 1
                continue
            if c == '"':
                state = "dq"
                i += 1
                continue
            m = _DOLLAR_RE.match(sql, i)
            if c == "$" and m:
                dollar_tag = m.group(0)
                state = "dollar"
                i += len(dollar_tag)
                continue
            out.append(c)
            i += 1
            continue
        if state == "sq":
            if c == "'":
                if nxt == "''":
                    i += 2
                    continue
                state = None
            i += 1
        elif state == "dq":
            if c == '"':
                if nxt == '""':
                    i += 2
                    continue
                state = None
            i += 1
        elif state == "dollar":
            if sql.startswith(dollar_tag, i):
                i += len(dollar_tag)
                state = None
            else:
                i += 1
        elif state == "line":
            if c == "\n":
                state = None
                out.append("\n")
            i += 1
        else:  # block comment
            if c == "*" and nxt == "/":
                state = None
                i += 2
                continue
            i += 1
    return "".join(out)


def _statement_count(sql: str) -> int:
    rugged = _strip_literals_and_comments(sql)
    depth = 0
    count = 0
    saw_content = False
    for ch in rugged:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            count += 1
            saw_content = False
        elif not ch.isspace():
            saw_content = True
    if saw_content:
        count += 1
    return count


def _mask_password_hash_in_projection(sql: str, notes: List[str]) -> str:
    """Mask password_hash inside the outermost SELECT ... FROM projection."""
    m = re.search(r"\bSELECT\b", sql, re.IGNORECASE)
    if not m:
        return sql
    select_start = m.start()
    # Skip WITH ... header by preferring the SELECT that follows CTE bodies.
    for mm in re.finditer(r"\bSELECT\b", sql, re.IGNORECASE):
        select_start = mm.start()
    depth = 0
    idx = select_start
    from_pos = -1
    in_literal: Optional[str] = None
    while idx < len(sql):
        ch = sql[idx]
        if in_literal:
            if ch == in_literal:
                in_literal = None
        elif ch in "'\"":
            in_literal = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and re.match(r"\bFROM\b", sql[idx:], re.IGNORECASE):
            from_pos = idx
            break
        idx += 1
    if from_pos <= select_start:
        return sql
    projection = sql[select_start:from_pos]
    if _PASSWORD_TOKEN_RE.search(projection):
        notes.append("password_hash in SELECT projection -> masked as NULL")

        def _mask(match: re.Match) -> str:
            alias = (match.group("alias") or "").strip()
            if alias:
                # `u.password_hash AS ph` -> `NULL AS ph`
                return f"NULL {alias}"
            return "NULL AS password_hash"

        projection = _PASSWORD_ALIAS_RE.sub(_mask, projection)
        return sql[:select_start] + projection + sql[from_pos:]
    return sql


def validate_sql(sql: str) -> SqlValidationResult:
    """Validate a candidate SQL statement as a safe read-only SELECT."""
    notes: List[str] = []
    original = (sql or "").strip().rstrip(";").strip()

    if not original:
        return SqlValidationResult(False, error="Empty SQL statement.")

    if _statement_count(original) > 1:
        return SqlValidationResult(False, original, "Exactly ONE statement is allowed.")

    clean = re.sub(r"\s+", " ", _strip_literals_and_comments(original)).strip()

    first = re.match(r"^(SELECT|WITH)\b", clean, re.IGNORECASE)
    if not first:
        return SqlValidationResult(
            False, original,
            "Only SELECT statements (optionally WITH ... SELECT) are allowed.",
        )
    if first.group(1).upper() == "WITH" and not re.search(r"\)\s*SELECT\b", clean, re.IGNORECASE):
        return SqlValidationResult(
            False, original, "WITH must be followed by a final SELECT statement."
        )

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"(?<![\w]){kw}(?![\w])", clean, re.IGNORECASE):
            return SqlValidationResult(
                False, original,
                f"Forbidden keyword detected: {kw}. Only read-only SELECT queries "
                "can run against the school database.",
            )
    for fn in FORBIDDEN_FUNCTIONS:
        if re.search(rf"(?<![\w]){fn}\s*\(", clean, re.IGNORECASE):
            return SqlValidationResult(
                False, original,
                f"Forbidden function detected: {fn}(). Read-only queries only.",
            )

    masked_sql = _mask_password_hash_in_projection(original, notes)
    return SqlValidationResult(True, masked_sql, masking_notes=notes)
