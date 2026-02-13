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
