#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any

from db import execute_select
from llm_client import OpenAIClient
from retriever import RAGContextBuilder, extract_tables_from_sql
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


def run_once(
    db_path: str,
    model: str,
    question: str,
    row_limit: int,
    max_sql_retries: int = 1,
    max_semantic_retries: int = 1,
    enable_rag: bool = False,
    rag_examples_path: str = "benchmarks/v1/questions_v1.json",
    rag_profiles_path: str = "artifacts/value_profiles.json",
    rag_example_top_k: int = 4,
    rag_schema_top_k: int = 4,
    rag_value_top_k: int = 8,
    rag_use_embeddings: bool = False,
    rag_embedding_model: str = "text-embedding-3-small",
    rag_embedding_index_path: str = "artifacts/example_embeddings.json",
    rag_schema_embedding_index_path: str = "artifacts/schema_embeddings.json",
) -> dict[str, Any]:
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
    rag_builder: RAGContextBuilder | None = None
    rag_info: dict[str, Any] = {}
    if enable_rag:
        try:
            rag_builder = RAGContextBuilder(
                db_path=db_path,
                examples_path=rag_examples_path,
                profiles_path=rag_profiles_path,
                example_top_k=max(1, rag_example_top_k),
                schema_top_k=max(1, rag_schema_top_k),
                value_top_k=max(1, rag_value_top_k),
                use_embeddings=rag_use_embeddings,
                embedding_model=rag_embedding_model,
                embedding_index_path=rag_embedding_index_path,
                schema_embedding_index_path=rag_schema_embedding_index_path,
            )
            if rag_builder.init_warnings:
                rag_info["rag_warning"] = " ; ".join(rag_builder.init_warnings)
        except Exception as exc:
            rag_info["rag_error"] = f"RAG disabled due to init error: {exc}"

    semantic_feedback = ""
    semantic_attempts: list[dict[str, Any]] = []
    total_sql_attempts = 0
    last_sql = ""
    last_error = "No attempt executed."

    for semantic_idx in range(max_semantic_retries + 1):
        semantic_attempt_no = semantic_idx + 1
        attempt_info: dict[str, Any] = {
            "semantic_attempt": semantic_attempt_no,
            "intent_spec": {},
            "precise_question": "",
            "generated_sql": "",
            "adequacy": {},
            "status": "incomplete",
            "rag": {},
        }

        rag_context_text = ""
        if rag_builder is not None:
            rag_tables = extract_tables_from_sql(last_sql)
            try:
                rag_payload = rag_builder.build_context(question=question, extra_tables=rag_tables)
                rag_context_text = rag_payload.get("context_text", "")
                attempt_info["rag"] = {
                    "candidate_tables": rag_payload.get("candidate_tables", []),
                    "examples_used": len(rag_payload.get("examples", [])),
                    "schema_hits_used": len(rag_payload.get("schema_hits", [])),
                    "value_hints_used": len(rag_payload.get("value_hints", [])),
                }
            except Exception as exc:
                attempt_info["rag"] = {"error": str(exc)}

        try:
            intent_spec = llm.generate_intent_spec(
                question=question,
                schema=schema,
                prior_feedback=semantic_feedback,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Intent normalization failed: {exc}",
                "question": question,
                "sql": last_sql,
                "rows": [],
                "semantic_attempts": semantic_attempts,
                "semantic_attempts_used": semantic_attempt_no,
                "sql_attempts": total_sql_attempts,
                "elapsed_sec": round(time.time() - started, 3),
                **_exception_details(exc),
            }

        precise_question = (intent_spec.get("precise_question") or question).strip()
        sql_requirements = intent_spec.get("sql_requirements", [])
        attempt_info["intent_spec"] = intent_spec
        attempt_info["precise_question"] = precise_question

        try:
            retry_guidance = ""
            if sql_requirements:
                retry_guidance = " ; ".join(str(req) for req in sql_requirements if str(req).strip())
            if semantic_feedback:
                retry_guidance = f"{retry_guidance} ; {semantic_feedback}".strip(" ;")
            sql = llm.generate_sql(
                question=precise_question,
                schema=schema,
                retry_guidance=retry_guidance,
                rag_context=rag_context_text,
            ).strip()
        except Exception as exc:
            attempt_info["status"] = "sql_generation_failed"
            semantic_attempts.append(attempt_info)
            return {
                "ok": False,
                "error": f"SQL generation failed: {exc}",
                "question": question,
                "sql": last_sql,
                "rows": [],
                "semantic_attempts": semantic_attempts,
                "semantic_attempts_used": semantic_attempt_no,
                "sql_attempts": total_sql_attempts,
                "elapsed_sec": round(time.time() - started, 3),
                **_exception_details(exc),
            }

        last_sql = sql
        attempt_info["generated_sql"] = sql

        ok, final_sql, rows, sql_attempts = _execute_with_optional_retry(
            llm=llm,
            db_path=db_path,
            question=precise_question,
            schema=schema,
            sql=sql,
            row_limit=row_limit,
            max_sql_retries=max_sql_retries,
        )
        total_sql_attempts += sql_attempts
        last_sql = final_sql if ok else sql
        if not ok:
            last_error = final_sql
            attempt_info["status"] = "sql_execution_failed"
            attempt_info["error"] = final_sql
            semantic_attempts.append(attempt_info)
            if semantic_idx >= max_semantic_retries:
                return {
                    "ok": False,
                    "error": final_sql,
                    "question": question,
                    "sql": sql,
                    "rows": [],
                    "semantic_attempts": semantic_attempts,
                    "semantic_attempts_used": semantic_attempt_no,
                    "sql_attempts": total_sql_attempts,
                    "elapsed_sec": round(time.time() - started, 3),
                }
            semantic_feedback = f"Previous attempt failed to execute SQL: {final_sql}"
            continue

        try:
            answer = llm.generate_answer(question=question, sql=final_sql, rows=rows)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Answer generation failed: {exc}",
                "question": question,
                "sql": final_sql,
                "rows": rows,
                "semantic_attempts": semantic_attempts,
                "semantic_attempts_used": semantic_attempt_no,
                "sql_attempts": total_sql_attempts,
                "elapsed_sec": round(time.time() - started, 3),
                **_exception_details(exc),
            }

        adequacy = {
            "is_sufficient": True,
            "reason_code": "sufficient",
            "missing_piece": "",
            "next_action": "keep",
        }
        adequacy_error = ""
        try:
            adequacy = llm.evaluate_answer_adequacy(
                original_question=question,
                intent_spec=intent_spec,
                sql=final_sql,
                rows=rows,
                answer=answer,
            )
        except Exception as exc:
            adequacy_error = str(exc)

        attempt_info["adequacy"] = adequacy
        if adequacy_error:
            attempt_info["adequacy_error"] = adequacy_error

        should_refine = (not adequacy.get("is_sufficient", False)) and adequacy.get("next_action") == "refine_and_retry"
        if should_refine and semantic_idx < max_semantic_retries:
            attempt_info["status"] = "refine_and_retry"
            semantic_attempts.append(attempt_info)
            semantic_feedback = (
                f"Reason={adequacy.get('reason_code', 'other')}; "
                f"Missing={adequacy.get('missing_piece', '')}; "
                f"SQLFix={adequacy.get('sql_fix_hint', '')}; "
                f"Previous SQL={final_sql}"
            )
            continue

        attempt_info["status"] = "completed"
        semantic_attempts.append(attempt_info)
        return {
            "ok": True,
            "error": None,
            "question": question,
            "sql": final_sql,
            "rows": rows,
            "answer": answer,
            "intent_spec": intent_spec,
            "adequacy": adequacy,
            "semantic_attempts": semantic_attempts,
            "semantic_attempts_used": semantic_attempt_no,
            "sql_attempts": total_sql_attempts,
            "rag_enabled": rag_builder is not None,
            **rag_info,
            "elapsed_sec": round(time.time() - started, 3),
        }

    return {
        "ok": False,
        "error": last_error,
        "question": question,
        "sql": last_sql,
        "rows": [],
        "semantic_attempts": semantic_attempts,
        "semantic_attempts_used": len(semantic_attempts),
        "sql_attempts": total_sql_attempts,
        "rag_enabled": rag_builder is not None,
        **rag_info,
        "elapsed_sec": round(time.time() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask NL questions over SQLite race-team data.")
    parser.add_argument("--db-path", default="race_team.db")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--max-sql-retries", type=int, default=1)
    parser.add_argument("--max-semantic-retries", type=int, default=1)
    parser.add_argument("--enable-rag", action="store_true")
    parser.add_argument("--rag-examples-path", default="benchmarks/v1/questions_v1.json")
    parser.add_argument("--rag-profiles-path", default="artifacts/value_profiles.json")
    parser.add_argument("--rag-example-top-k", type=int, default=4)
    parser.add_argument("--rag-schema-top-k", type=int, default=4)
    parser.add_argument("--rag-value-top-k", type=int, default=8)
    parser.add_argument("--rag-use-embeddings", action="store_true")
    parser.add_argument("--rag-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--rag-embedding-index-path", default="artifacts/example_embeddings.json")
    parser.add_argument("--rag-schema-embedding-index-path", default="artifacts/schema_embeddings.json")
    parser.add_argument("--question", help="Single question to run.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-thinking", action="store_true", help="Print structured reasoning trace.")
    return parser.parse_args()


def print_result(result: dict[str, Any], as_json: bool, show_thinking: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return
    print(f"ok: {result['ok']}")
    print(f"question: {result['question']}")
    print(f"elapsed_sec: {result['elapsed_sec']}")
    if "sql_attempts" in result:
        print(f"sql_attempts: {result['sql_attempts']}")
    if "semantic_attempts_used" in result:
        print(f"semantic_attempts_used: {result['semantic_attempts_used']}")
    if "rag_enabled" in result:
        print(f"rag_enabled: {result['rag_enabled']}")
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

    if show_thinking:
        if result.get("intent_spec"):
            print("\nintent_spec:")
            print(f"precise_question: {result['intent_spec'].get('precise_question', '')}")
            reqs = result["intent_spec"].get("sql_requirements", [])
            if reqs:
                print("sql_requirements:")
                for req in reqs:
                    print(f"- {req}")
        if result.get("adequacy"):
            print("\nadequacy:")
            print(f"is_sufficient: {result['adequacy'].get('is_sufficient')}")
            print(f"reason_code: {result['adequacy'].get('reason_code', '')}")
            if result["adequacy"].get("missing_piece"):
                print(f"missing_piece: {result['adequacy'].get('missing_piece', '')}")
            if result["adequacy"].get("sql_fix_hint"):
                print(f"sql_fix_hint: {result['adequacy'].get('sql_fix_hint', '')}")
        if result.get("semantic_attempts"):
            print("\nsemantic_attempts:")
            for attempt in result["semantic_attempts"]:
                print(
                    f"- attempt={attempt.get('semantic_attempt')} status={attempt.get('status')} "
                    f"question={attempt.get('precise_question', '')}"
                )
                rag_summary = attempt.get("rag", {})
                if rag_summary:
                    if rag_summary.get("error"):
                        print(f"  rag_error: {rag_summary.get('error')}")
                    else:
                        print(
                            "  rag: "
                            f"tables={rag_summary.get('candidate_tables', [])} "
                            f"examples={rag_summary.get('examples_used', 0)} "
                            f"schema_hits={rag_summary.get('schema_hits_used', 0)} "
                            f"value_hints={rag_summary.get('value_hints_used', 0)}"
                        )
                if attempt.get("adequacy", {}).get("missing_piece"):
                    print(f"  adequacy_missing: {attempt['adequacy'].get('missing_piece', '')}")


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
                run_once(
                    args.db_path,
                    args.model,
                    q,
                    args.row_limit,
                    max_sql_retries=max(0, args.max_sql_retries),
                    max_semantic_retries=max(0, args.max_semantic_retries),
                    enable_rag=args.enable_rag,
                    rag_examples_path=args.rag_examples_path,
                    rag_profiles_path=args.rag_profiles_path,
                    rag_example_top_k=max(1, args.rag_example_top_k),
                    rag_schema_top_k=max(1, args.rag_schema_top_k),
                    rag_value_top_k=max(1, args.rag_value_top_k),
                    rag_use_embeddings=args.rag_use_embeddings,
                    rag_embedding_model=args.rag_embedding_model,
                    rag_embedding_index_path=args.rag_embedding_index_path,
                    rag_schema_embedding_index_path=args.rag_schema_embedding_index_path,
                ),
                args.json,
                show_thinking=args.show_thinking,
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
            max_semantic_retries=max(0, args.max_semantic_retries),
            enable_rag=args.enable_rag,
            rag_examples_path=args.rag_examples_path,
            rag_profiles_path=args.rag_profiles_path,
            rag_example_top_k=max(1, args.rag_example_top_k),
            rag_schema_top_k=max(1, args.rag_schema_top_k),
            rag_value_top_k=max(1, args.rag_value_top_k),
            rag_use_embeddings=args.rag_use_embeddings,
            rag_embedding_model=args.rag_embedding_model,
            rag_embedding_index_path=args.rag_embedding_index_path,
            rag_schema_embedding_index_path=args.rag_schema_embedding_index_path,
        ),
        args.json,
        show_thinking=args.show_thinking,
    )


if __name__ == "__main__":
    main()
