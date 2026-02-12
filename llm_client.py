from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency. Install with: pip install openai") from exc

from prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
    SQL_SYSTEM_PROMPT,
    SQL_USER_PROMPT_TEMPLATE,
)


class LLMClient(ABC):
    @abstractmethod
    def generate_sql(self, question: str, schema: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_answer(self, question: str, sql: str, rows: list[dict]) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY") or _load_openai_api_key_from_dotenv()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call_text(self, system: str, user: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        text = getattr(resp, "output_text", "")
        if not text:
            raise RuntimeError("Model returned empty text output.")
        return text.strip()

    def generate_sql(self, question: str, schema: str) -> str:
        return self._call_text(SQL_SYSTEM_PROMPT, SQL_USER_PROMPT_TEMPLATE.format(schema=schema, question=question))

    def generate_answer(self, question: str, sql: str, rows: list[dict]) -> str:
        return self._call_text(
            ANSWER_SYSTEM_PROMPT,
            ANSWER_USER_PROMPT_TEMPLATE.format(question=question, sql=sql, rows=rows[:20]),
        )


def _load_openai_api_key_from_dotenv(dotenv_path: str = ".env") -> str:
    path = Path(dotenv_path)
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""
