# Agent Build Plan (v1, Framework-Free)

## Key Decision
- Use no orchestration framework for v1.
- Keep architecture modular so a framework or multi-agent runtime can be added later.

## Safety Rule for SQL
- SQL is validated before execution.
- Reject if query is empty, multi-statement, non-SELECT/WITH, or contains blocked write/admin keywords.

## Deliverables
- `agent.py`
- `llm_client.py`
- `prompts.py`
- `schema_context.py`
- `sql_guardrails.py`
- `db.py`
- `seed_data.py`
- `eval_questions.json`
- `run_eval.py`

## Definition of Done
- End-to-end question answering against `race_team.db`.
- Non-read-only SQL always blocked with clear errors.
- Eval script produces summary metrics.
