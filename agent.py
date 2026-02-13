#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any

from db import execute_select
from llm_client import OpenAIClient
from schema_context import get_schema_context
from sql_guardrails import validate_read_only_sql


def _exception_details(exc: Exception) -> dict[str, Any]:
    cause = exc.__cause__
    return {
        "error_type": type(exc).__name__,
        "error_repr": repr(exc),
        "error_cause_type": type(cause).__name__ if cause is not None else "",
        "error_cause_repr": repr(cause) if cause is not None else "",
        "error_traceback": traceback.format_exc(),
    }


def _validate_or_error(sql: str) -> str | None:
    validation = validate_read_only_sql(sql)
    if validation.ok:
        return None
    return f"SQL blocked by guardrails: {validation.error}"


def _execute_with_optional_retry(
    llm: OpenAIClient,
    db_path: str,
    question: str,
    schema: str,
    sql: str,
    row_limit: int,
    max_sql_retries: int,
) -> tuple[bool, str, list[dict[str, Any]], int]:
    err = _validate_or_error(sql)
    if err is not None:
        return False, err, [], 1

    sql_attempts = 1
    current_sql = sql
    last_exec_error: Exception | None = None
    while True:
        try:
            rows = execute_select(db_path=db_path, sql=current_sql, row_limit=row_limit)
            return True, current_sql, rows, sql_attempts
        except Exception as exec_exc:
            last_exec_error = exec_exc

        if (sql_attempts - 1) >= max_sql_retries:
            return False, f"SQL execution failed: {last_exec_error}", [], sql_attempts

        try:
            repaired_sql = llm.generate_sql_repair(
                question=question,
                schema=schema,
                failed_sql=current_sql,
                error=str(last_exec_error),
            ).strip()
        except Exception as repair_exc:
            return False, f"SQL execution failed: {last_exec_error}; retry generation failed: {repair_exc}", [], sql_attempts

        sql_attempts += 1
        err = _validate_or_error(repaired_sql)
        if err is not None:
            return False, f"SQL execution failed: {last_exec_error}; retry failed: {err}", [], sql_attempts
        current_sql = repaired_sql


def run_once(db_path: str, model: str, question: str, row_limit: int, max_sql_retries: int = 1) -> dict[str, Any]:
    started = time.time()
    try:
        llm = OpenAIClient(model=model)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"LLM client init failed: {exc}",
            "question": question,
            "sql": "",
            "rows": [],
            "elapsed_sec": round(time.time() - started, 3),
            **_exception_details(exc),
        }

    schema = get_schema_context(db_path)

    try:
        sql = llm.generate_sql(question=question, schema=schema).strip()
    except Exception as exc:
        return {
            "ok": False,
            "error": f"SQL generation failed: {exc}",
            "question": question,
            "sql": "",
            "rows": [],
            "elapsed_sec": round(time.time() - started, 3),
            **_exception_details(exc),
        }

    ok, final_sql, rows, sql_attempts = _execute_with_optional_retry(
        llm=llm,
        db_path=db_path,
        question=question,
        schema=schema,
        sql=sql,
        row_limit=row_limit,
        max_sql_retries=max_sql_retries,
    )
    if not ok:
        return {
            "ok": False,
            "error": final_sql,
            "question": question,
            "sql": sql,
            "rows": [],
            "sql_attempts": sql_attempts,
            "elapsed_sec": round(time.time() - started, 3),
        }

    try:
        answer = llm.generate_answer(question=question, sql=final_sql, rows=rows)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Answer generation failed: {exc}",
            "question": question,
            "sql": final_sql,
            "rows": rows,
            "sql_attempts": sql_attempts,
            "elapsed_sec": round(time.time() - started, 3),
            **_exception_details(exc),
        }

    return {
        "ok": True,
        "error": None,
        "question": question,
        "sql": final_sql,
        "rows": rows,
        "answer": answer,
        "sql_attempts": sql_attempts,
        "elapsed_sec": round(time.time() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask NL questions over SQLite race-team data.")
    parser.add_argument("--db-path", default="race_team.db")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--max-sql-retries", type=int, default=1)
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
    if "sql_attempts" in result:
        print(f"sql_attempts: {result['sql_attempts']}")
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
            print_result(
                run_once(args.db_path, args.model, q, args.row_limit, max_sql_retries=max(0, args.max_sql_retries)),
                args.json,
            )
            print()
        return
    if not args.question:
        raise SystemExit("Provide --question or use --interactive")
    print_result(
        run_once(
            args.db_path,
            args.model,
            args.question,
            args.row_limit,
            max_sql_retries=max(0, args.max_sql_retries),
        ),
        args.json,
    )


if __name__ == "__main__":
    main()
