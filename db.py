from __future__ import annotations

import sqlite3
from typing import Any

from sql_guardrails import validate_read_only_sql


def execute_select(db_path: str, sql: str, row_limit: int = 100) -> list[dict[str, Any]]:
    validation = validate_read_only_sql(sql)
    if not validation.ok:
        raise ValueError(validation.error)

    normalized = sql.strip().rstrip(";")
    limited_sql = f"SELECT * FROM ({normalized}) LIMIT {int(row_limit)}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(limited_sql)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
