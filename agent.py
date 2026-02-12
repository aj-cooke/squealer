#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from db import execute_select
from llm_client import OpenAIClient
from schema_context import get_schema_context
from sql_guardrails import validate_read_only_sql


def run_once(db_path: str, model: str, question: str, row_limit: int) -> dict[str, Any]:
    started = time.time()
    try:
        llm = OpenAIClient(model=model)
    except Exception as exc:
        return {"ok": False, "error": f"LLM client init failed: {exc}", "question": question, "sql": "", "rows": [], "elapsed_sec": round(time.time() - started, 3)}

    schema = get_schema_context(db_path)

    try:
        sql = llm.generate_sql(question=question, schema=schema).strip()
    except Exception as exc:
        return {"ok": False, "error": f"SQL generation failed: {exc}", "question": question, "sql": "", "rows": [], "elapsed_sec": round(time.time() - started, 3)}

    validation = validate_read_only_sql(sql)
    if not validation.ok:
        return {"ok": False, "error": f"SQL blocked by guardrails: {validation.error}", "question": question, "sql": sql, "rows": [], "elapsed_sec": round(time.time() - started, 3)}

    try:
        rows = execute_select(db_path=db_path, sql=sql, row_limit=row_limit)
    except Exception as exc:
        return {"ok": False, "error": f"SQL execution failed: {exc}", "question": question, "sql": sql, "rows": [], "elapsed_sec": round(time.time() - started, 3)}

    try:
        answer = llm.generate_answer(question=question, sql=sql, rows=rows)
    except Exception as exc:
        return {"ok": False, "error": f"Answer generation failed: {exc}", "question": question, "sql": sql, "rows": rows, "elapsed_sec": round(time.time() - started, 3)}

    return {"ok": True, "error": None, "question": question, "sql": sql, "rows": rows, "answer": answer, "elapsed_sec": round(time.time() - started, 3)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask NL questions over SQLite race-team data.")
    parser.add_argument("--db-path", default="race_team.db")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--question", help="Single question to run.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return
    print(f"ok: {result['ok']}")
    print(f"question: {result['question']}")
    print(f"elapsed_sec: {result['elapsed_sec']}")
    if result.get("sql"):
        print("\nSQL:")
        print(result["sql"])
    if result["ok"]:
        print("\nanswer:")
        print(result["answer"])
        print("\nrows_preview:")
        for row in result["rows"][:10]:
            print(row)
    else:
        print("\nerror:")
        print(result["error"])


def main() -> None:
    args = parse_args()
    if args.interactive:
        while True:
            q = input("question> ").strip()
            if q.lower() in {"quit", "exit"}:
                break
            if not q:
                continue
            print_result(run_once(args.db_path, args.model, q, args.row_limit), args.json)
            print()
        return
    if not args.question:
        raise SystemExit("Provide --question or use --interactive")
    print_result(run_once(args.db_path, args.model, args.question, args.row_limit), args.json)


if __name__ == "__main__":
    main()
