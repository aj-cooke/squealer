from __future__ import annotations

import sqlite3


def get_schema_context(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        tables = [r[0] for r in cur.fetchall()]

        lines: list[str] = []
        for table in tables:
            lines.append(f"Table: {table}")
            cur.execute(f"PRAGMA table_info({table})")
            for col in cur.fetchall():
                _, name, coltype, notnull, default_value, pk = col
                pk_flag = " PK" if pk else ""
                nn_flag = " NOT NULL" if notnull else ""
                default_flag = f" DEFAULT {default_value}" if default_value is not None else ""
                lines.append(f"  - {name}: {coltype}{pk_flag}{nn_flag}{default_flag}")
            lines.append("")
        return "\n".join(lines).strip()
    finally:
        conn.close()
