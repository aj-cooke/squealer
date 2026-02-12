# First-Pass Plan: Race Team QA Agent (SQLite + Python)

## Goal
Build a local MVP where a user asks a natural-language question about race-team data and gets a correct answer backed by SQL query results.

## Constraints
- Database should have minimal setup overhead.
- Use Python-friendly tooling.
- Minimize dependencies for fake-data generation.
- Use OpenAI models for question-to-SQL behavior.

## Tech Stack (First Pass)
- Database: `sqlite3` (Python standard library)
- Data generation: `random`, `datetime`, `itertools`, `uuid` (stdlib) + `numpy` + `pandas`
- Agent app: Python script/service
- LLM: OpenAI model for SQL generation

## Phases
1. Define target questions and success metrics.
2. Build SQLite schema.
3. Generate fake data with stdlib + numpy + pandas.
4. Implement NL -> SQL -> DB -> answer flow.
5. Enforce strict SQL guardrails (SELECT only).
6. Add benchmark eval harness.
7. Iterate only when metrics justify extra complexity.
