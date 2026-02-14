SQL_SYSTEM_PROMPT = """You translate user questions into SQLite SQL.
Rules:
- Return exactly one SQL statement.
- Return SQL only, no markdown, no prose.
- Query must be read-only and start with SELECT or WITH.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, DETACH.
- Prefer explicit JOIN conditions.
- Use table/column names exactly from schema.
"""

SQL_USER_PROMPT_TEMPLATE = """Schema:
{schema}

Question:
{question}

Return a single SQLite SELECT query.
"""

SQL_RETRY_GUIDANCE_TEMPLATE = """
Retry guidance:
{retry_guidance}
"""

INTENT_SPEC_SYSTEM_PROMPT = """You convert user questions into a precise data-analysis plan.
Rules:
- Return plain text only with the exact keys requested.
- Preserve user intent; do not invent constraints.
- If details are missing, call them out briefly.
"""

INTENT_SPEC_USER_PROMPT_TEMPLATE = """Schema:
{schema}

Original question:
{question}

Prior adequacy feedback:
{prior_feedback}

Return exactly this format:
PRECISE_QUESTION: <one sentence>
SQL_REQUIREMENTS: <semicolon-separated SQL requirements the next query must satisfy>
"""

SQL_REPAIR_USER_PROMPT_TEMPLATE = """Schema:
{schema}

Question:
{question}

Previous SQL (failed):
{failed_sql}

Execution error:
{error}

Fix the SQL so it answers the same question and runs on SQLite.
Return a single corrected SQLite SELECT query.
"""

ANSWER_SYSTEM_PROMPT = """You are a data assistant.
Given a question, SQL, and SQL result rows, return a concise factual answer.
If no rows are returned, clearly say no matching data was found.
"""

ANSWER_USER_PROMPT_TEMPLATE = """Question:
{question}

SQL:
{sql}

Rows:
{rows}
"""

ANSWER_ADEQUACY_SYSTEM_PROMPT = """You judge whether the SQL result sufficiently answers the original user question.
Rules:
- Return plain text only with the exact keys requested.
- Evaluate against the original question first, then the intent spec.
- If insufficient, state what is missing and how to refine next attempt.
"""

ANSWER_ADEQUACY_USER_PROMPT_TEMPLATE = """Original question:
{original_question}

Intent spec:
{intent_spec}

Executed SQL:
{sql}

Rows:
{rows}

Draft answer:
{answer}

Return exactly this format:
SUFFICIENT: yes|no
REASON_CODE: sufficient|insufficient_scope|insufficient_filtering|wrong_metric|wrong_grain|empty_result|ambiguous_request|other
MISSING_PIECE: <short text>
NEXT_ACTION: keep|refine_and_retry
SQL_FIX_HINT: <short SQL-focused instruction for next attempt>
"""
