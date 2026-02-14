#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cached value profiles for RAG value hints.")
    p.add_argument("--db-path", default="race_team.db")
    p.add_argument("--out", default="artifacts/value_profiles.json")
    p.add_argument("--top-n-categorical", type=int, default=10)
    p.add_argument("--max-categorical-distinct", type=int, default=200)
    p.add_argument("--max-numeric-sample", type=int, default=200000)
    return p.parse_args()


def _resolve_output_path(out_arg: str) -> Path:
    out_path = Path(out_arg)
    if out_path.exists() and out_path.is_dir():
        return out_path / "value_profiles.json"
    return out_path


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _safe_table_columns(conn: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    tables = [str(r[0]) for r in cur.fetchall()]
    out: dict[str, list[tuple[str, str]]] = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({_quote_ident(table)})")
        out[table] = [(str(row[1]), str(row[2] or "")) for row in cur.fetchall()]
    return out


def _infer_kind(declared_type: str, values: list[Any], col_name: str) -> str:
    dt = declared_type.upper()
    name = col_name.lower()
    if any(tok in dt for tok in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
        return "numeric"
    if any(tok in dt for tok in ("DATE", "TIME")) or name.endswith("_date") or name.endswith("_at"):
        return "temporal"

    numeric_hits = 0
    temporal_hits = 0
    probe = values[:50]
    for v in probe:
        if v is None:
            continue
        s = str(v)
        try:
            float(s)
            numeric_hits += 1
        except Exception:
            pass
        if _looks_iso_date(s):
            temporal_hits += 1

    if probe and numeric_hits >= max(3, int(0.8 * len(probe))):
        return "numeric"
    if probe and temporal_hits >= max(3, int(0.7 * len(probe))):
        return "temporal"
    return "categorical"


def _looks_iso_date(s: str) -> bool:
    if len(s) < 8:
        return False
    if "-" not in s:
        return False
    head = s.split("T", 1)[0]
    parts = head.split("-")
    if len(parts) != 3:
        return False
    return all(p.isdigit() for p in parts)


def _quantile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    idx = p * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _build_column_profile(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    declared_type: str,
    top_n_categorical: int,
    max_categorical_distinct: int,
    max_numeric_sample: int,
) -> dict[str, Any]:
    cur = conn.cursor()
    t = _quote_ident(table)
    c = _quote_ident(column)

    cur.execute(f"SELECT COUNT(*) FROM {t}")
    row_count = int(cur.fetchone()[0])

    cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {c} IS NULL")
    null_count = int(cur.fetchone()[0])

    cur.execute(f"SELECT COUNT(DISTINCT {c}) FROM {t}")
    distinct_count = int(cur.fetchone()[0])

    cur.execute(f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT 200")
    probe_values = [row[0] for row in cur.fetchall()]
    kind = _infer_kind(declared_type, probe_values, column)

    out: dict[str, Any] = {
        "declared_type": declared_type,
        "inferred_kind": kind,
        "row_count": row_count,
        "null_count": null_count,
        "null_rate": round((null_count / row_count), 6) if row_count else 0.0,
        "distinct_count": distinct_count,
        "sample_tokens": [column.lower(), table.lower()],
    }

    if kind == "numeric":
        cur.execute(f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT {int(max_numeric_sample)}")
        values: list[float] = []
        for row in cur.fetchall():
            try:
                values.append(float(row[0]))
            except Exception:
                continue
        if values:
            values.sort()
            out["numeric_stats"] = {
                "min": round(values[0], 6),
                "p25": round(_quantile(values, 0.25), 6),
                "p50": round(_quantile(values, 0.5), 6),
                "p75": round(_quantile(values, 0.75), 6),
                "max": round(values[-1], 6),
            }
    elif kind == "temporal":
        cur.execute(f"SELECT MIN({c}), MAX({c}) FROM {t} WHERE {c} IS NOT NULL")
        mn, mx = cur.fetchone()
        out["temporal_range"] = {"min": mn, "max": mx}
    else:
        if distinct_count <= max_categorical_distinct:
            cur.execute(
                f"""
                SELECT {c} AS value, COUNT(*) AS freq
                FROM {t}
                WHERE {c} IS NOT NULL
                GROUP BY {c}
                ORDER BY freq DESC, value
                LIMIT {int(top_n_categorical)}
                """
            )
            out["top_values"] = [{"value": row[0], "count": int(row[1])} for row in cur.fetchall()]

    return out


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db_path)
    try:
        table_cols = _safe_table_columns(conn)
        payload: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": args.db_path,
            "tables": {},
        }

        for table, cols in table_cols.items():
            table_info: dict[str, Any] = {"columns": {}}
            for col_name, declared_type in cols:
                profile = _build_column_profile(
                    conn,
                    table,
                    col_name,
                    declared_type,
                    top_n_categorical=max(1, args.top_n_categorical),
                    max_categorical_distinct=max(1, args.max_categorical_distinct),
                    max_numeric_sample=max(100, args.max_numeric_sample),
                )
                table_info["columns"][col_name] = profile
            payload["tables"][table] = table_info

        out_path = _resolve_output_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        print(f"Wrote profiles to: {out_path}")
        print(f"Tables profiled: {len(payload['tables'])}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
