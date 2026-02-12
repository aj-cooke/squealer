from __future__ import annotations

import re
from dataclasses import dataclass

BLOCKED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
    "attach", "detach", "vacuum", "pragma", "reindex", "grant", "revoke", "merge",
}

@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def validate_read_only_sql(sql: str) -> ValidationResult:
    if not sql or not sql.strip():
        return ValidationResult(False, "SQL is empty.")

    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        return ValidationResult(False, "SQL is empty after removing comments.")

    if ";" in cleaned.rstrip(";"):
        return ValidationResult(False, "Only a single SQL statement is allowed.")

    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return ValidationResult(False, "Only SELECT queries are allowed.")

    tokens = re.findall(r"[a-zA-Z_]+", lowered)
    found_blocked = sorted({t for t in tokens if t in BLOCKED_KEYWORDS})
    if found_blocked:
        return ValidationResult(False, f"Blocked SQL keyword(s): {', '.join(found_blocked)}")

    return ValidationResult(True)
