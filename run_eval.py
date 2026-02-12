#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

from agent import run_once


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simple eval over benchmark questions.")
    p.add_argument("--db-path", default="race_team.db")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--questions-file", default="eval_questions.json")
    p.add_argument("--row-limit", type=int, default=50)
    p.add_argument("--out", default="eval_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        started = time.time()
        result = run_once(args.db_path, args.model, q, args.row_limit)
        result["eval_elapsed_sec"] = round(time.time() - started, 3)
        results.append(result)
        print(f"[{'OK' if result['ok'] else 'FAIL'}] {q}")

    ok_count = sum(1 for r in results if r["ok"])
    latencies = [r["elapsed_sec"] for r in results]
    summary = {
        "total": len(results),
        "ok": ok_count,
        "failed": len(results) - ok_count,
        "success_rate": round(ok_count / len(results), 3) if results else 0.0,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_max": round(max(latencies), 3) if latencies else None,
    }

    report = {"summary": summary, "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
