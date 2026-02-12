# Benchmark v1

This benchmark contains 100 NL questions with canonical SQL and deterministic checks.

## File
- `benchmarks/v1/questions_v1.json`

Each case includes:
- `id`
- `question`
- `canonical_sql`
- `check`
- `tags`
- `difficulty`

## Check Types
- `exact_rows`: columns and rows must match exactly (order-sensitive).
- `set_match`: columns must match and row multiset must match (order-insensitive).
- `ordered_top_k`: first `k` rows must match exactly.
- `numeric`: compare first cell numeric values with optional `tolerance`.
- `boolean`: compare first cell truthiness.
- `non_empty`: compare whether result presence/absence matches canonical query.

## Run
```bash
python3 run_eval_v2.py \
  --db-path race_team.db \
  --model gpt-4.1-mini \
  --benchmark benchmarks/v1/questions_v1.json \
  --out eval_report_v2.json
```

## Scoring
`run_eval_v2.py` computes per-case pass/fail and aggregate metrics:
- overall success rate
- latency p50/p95/max
- success breakdown by difficulty
- success breakdown by tag


Optional: add `--strict-columns` to require exact output column names/aliases.
