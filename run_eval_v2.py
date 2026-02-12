#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

from llm_client import OpenAIClient
from schema_context import get_schema_context
from sql_guardrails import validate_read_only_sql


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark v2 against canonical SQL outputs.")
    p.add_argument("--db-path", default="race_team.db")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--benchmark", default="benchmarks/v1/questions_v1.json")
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--out", default="eval_report_v2.json")
    p.add_argument("--strict-columns", action="store_true")
    return p.parse_args()


def _normalize_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    return v


def _rows_to_tuples(rows: list[dict[str, Any]]) -> tuple[list[str], list[tuple[Any, ...]]]:
    if not rows:
        return [], []
    cols = list(rows[0].keys())
    tuples = [tuple(_normalize_value(r.get(c)) for c in cols) for r in rows]
    return cols, tuples


def _execute_sql(db_path: str, sql: str, max_rows: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        wrapped = f"SELECT * FROM ({sql.strip().rstrip(';')}) LIMIT {int(max_rows)}"
        cur = conn.cursor()
        cur.execute(wrapped)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _compare_rows(
    check: dict[str, Any],
    agent_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    strict_columns: bool,
) -> tuple[bool, str]:
    check_type = check.get("type", "exact_rows")
    agent_cols, agent_tuples = _rows_to_tuples(agent_rows)
    canon_cols, canon_tuples = _rows_to_tuples(canonical_rows)

    if check_type == "exact_rows":
        if strict_columns and agent_cols != canon_cols:
            return False, f"column mismatch: agent={agent_cols}, canonical={canon_cols}"
        return (agent_tuples == canon_tuples, "rows match" if agent_tuples == canon_tuples else "row mismatch")

    if check_type == "set_match":
        if strict_columns and agent_cols != canon_cols:
            return False, f"column mismatch: agent={agent_cols}, canonical={canon_cols}"
        return (
            Counter(agent_tuples) == Counter(canon_tuples),
            "set match" if Counter(agent_tuples) == Counter(canon_tuples) else "set mismatch",
        )

    if check_type == "ordered_top_k":
        if strict_columns and agent_cols != canon_cols:
            return False, f"column mismatch: agent={agent_cols}, canonical={canon_cols}"
        k = int(check.get("k", 1))
        return (
            agent_tuples[:k] == canon_tuples[:k],
            f"top {k} match" if agent_tuples[:k] == canon_tuples[:k] else f"top {k} mismatch",
        )

    if check_type == "numeric":
        tol = float(check.get("tolerance", 0.0))
        if not agent_tuples or not canon_tuples:
            return False, "expected numeric rows but one side is empty"
        av = agent_tuples[0][0]
        cv = canon_tuples[0][0]
        try:
            avf = float(av)
            cvf = float(cv)
        except Exception:
            return False, f"numeric parse failed: agent={av}, canonical={cv}"
        ok = abs(avf - cvf) <= tol
        return ok, f"numeric diff={abs(avf-cvf):.6f}, tol={tol}"

    if check_type == "boolean":
        if not agent_tuples or not canon_tuples:
            return False, "expected boolean rows but one side is empty"
        av = bool(agent_tuples[0][0])
        cv = bool(canon_tuples[0][0])
        return (av == cv, "boolean match" if av == cv else "boolean mismatch")

    if check_type == "non_empty":
        return (
            (len(agent_rows) > 0) == (len(canonical_rows) > 0),
            "non-empty behavior match" if (len(agent_rows) > 0) == (len(canonical_rows) > 0) else "non-empty behavior mismatch",
        )

    return False, f"unsupported check type: {check_type}"


def main() -> None:
    args = parse_args()
    with open(args.benchmark, "r", encoding="utf-8") as f:
        cases = json.load(f)

    llm = OpenAIClient(model=args.model)
    schema = get_schema_context(args.db_path)

    results: list[dict[str, Any]] = []

    for case in cases:
        start = time.time()
        cid = case["id"]
        q = case["question"]
        canonical_sql = case["canonical_sql"]
        check = case.get("check", {"type": "exact_rows"})

        row: dict[str, Any] = {
            "id": cid,
            "question": q,
            "difficulty": case.get("difficulty", "unknown"),
            "tags": case.get("tags", []),
            "check": check,
            "canonical_sql": canonical_sql,
            "generated_sql": "",
            "ok": False,
            "reason": "",
            "elapsed_sec": 0.0,
        }

        try:
            generated_sql = llm.generate_sql(question=q, schema=schema).strip()
            row["generated_sql"] = generated_sql

            valid = validate_read_only_sql(generated_sql)
            if not valid.ok:
                row["reason"] = f"guardrail_block: {valid.error}"
            else:
                canonical_rows = _execute_sql(args.db_path, canonical_sql, args.max_rows)
                agent_rows = _execute_sql(args.db_path, generated_sql, args.max_rows)
                ok, reason = _compare_rows(check, agent_rows, canonical_rows, args.strict_columns)
                row["ok"] = ok
                row["reason"] = reason
                row["canonical_row_count"] = len(canonical_rows)
                row["agent_row_count"] = len(agent_rows)
        except Exception as exc:
            row["reason"] = f"exception: {exc}"

        row["elapsed_sec"] = round(time.time() - start, 3)
        results.append(row)
        print(f"[{'OK' if row['ok'] else 'FAIL'}] {cid} {q}")

    total = len(results)
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = total - ok_count
    latencies = [r["elapsed_sec"] for r in results]

    by_difficulty: dict[str, dict[str, Any]] = {}
    groups = defaultdict(list)
    for r in results:
        groups[r["difficulty"]].append(r)
    for k, g in groups.items():
        ok_k = sum(1 for x in g if x["ok"])
        by_difficulty[k] = {
            "total": len(g),
            "ok": ok_k,
            "failed": len(g) - ok_k,
            "success_rate": round(ok_k / len(g), 3) if g else 0.0,
        }

    by_tag: dict[str, dict[str, Any]] = {}
    tag_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        for tag in r.get("tags", []):
            tag_groups[tag].append(r)
    for tag, g in sorted(tag_groups.items()):
        ok_t = sum(1 for x in g if x["ok"])
        by_tag[tag] = {
            "total": len(g),
            "ok": ok_t,
            "failed": len(g) - ok_t,
            "success_rate": round(ok_t / len(g), 3) if g else 0.0,
        }

    summary = {
        "total": total,
        "ok": ok_count,
        "failed": fail_count,
        "success_rate": round(ok_count / total, 3) if total else 0.0,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95": round(statistics.quantiles(latencies, n=20)[18], 3) if len(latencies) >= 20 else None,
        "latency_max": round(max(latencies), 3) if latencies else None,
        "by_difficulty": by_difficulty,
        "by_tag": by_tag,
    }

    report = {
        "benchmark": args.benchmark,
        "model": args.model,
        "summary": summary,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
