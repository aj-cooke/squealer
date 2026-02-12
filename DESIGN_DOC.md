# Design Doc: Squealer NL-to-SQL Agent (v1)

## Objective
Build and iterate a local natural-language-to-SQL agent over a synthetic race-team SQLite database, with deterministic evaluation to compare agent versions.

## Scope
- Read-only QA over relational data (`sqlite3`).
- Single-agent flow (no orchestration framework in v1).
- Strong SQL safety guardrails.
- Versioned benchmark with deterministic pass/fail scoring.

## Architecture
1. User question input (`agent.py`).
2. Schema introspection (`schema_context.py`) to produce table/column context.
3. SQL generation via OpenAI (`llm_client.py`, `prompts.py`).
4. Guardrail validation (`sql_guardrails.py`):
   - single statement only
   - must start with `SELECT`/`WITH`
   - blocks write/admin keywords
5. Query execution (`db.py`) with result row cap.
6. Answer synthesis via OpenAI from SQL + row preview.

## Data Layer
- Database: SQLite (`race_team.db`).
- Seed pipeline: `seed_data.py`.
- Entities:
  - `teams`, `drivers`, `cars`, `races`, `weather`, `race_results`, `lap_times`, `pit_stops`

## Benchmark + Evaluation
### v1 Benchmark
- File: `benchmarks/v1/questions_v1.json`
- Cases: 100
- Each case contains:
  - `id`, `question`, `canonical_sql`, `check`, `tags`, `difficulty`

### Evaluator
- Script: `run_eval_v2.py`
- Flow per case:
  1. generate SQL from question
  2. validate with guardrails
  3. execute generated SQL and canonical SQL
  4. compare outputs using check type

### Check Types
- `numeric` (with tolerance)
- `exact_rows`
- `ordered_top_k`
- `set_match`
- `boolean`
- `non_empty`

### Scoring Policy
- Default: value-based row comparison (column alias-insensitive).
- Optional strict mode: `--strict-columns`.

### Metrics
- overall success rate
- pass/fail by difficulty
- pass/fail by tag
- p50/p95/max latency

## Security and Secrets
- `.env` holds `OPENAI_API_KEY` locally.
- `.env`, key files, DB files, and eval report artifacts are git-ignored.
- `.env.example` provides safe template for collaborators.

## Versioning and Comparison Strategy
1. Keep benchmark dataset stable per version (`benchmarks/v1`).
2. Evaluate each agent revision on same benchmark/model/db snapshot.
3. Compare reports on:
   - overall pass rate
   - per-tag/per-difficulty deltas
   - latency regressions
4. Only promote complexity when measurable gains justify added maintenance.

## Known Gaps (Current v1)
- Medium/hard query performance remains below target.
- No SQL self-repair retry loop yet.
- No prompt-example retrieval or planning stage yet.

## Next Steps
1. Add one-shot retry when generated SQL fails guardrails/execution.
2. Add targeted prompt exemplars for weak tags (`ranking`, `season`, multi-join aggregations).
3. Introduce per-case failure categorization in report for tighter iteration loops.
