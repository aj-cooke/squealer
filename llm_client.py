from __future__ import annotations

import os
import random
import socket
import time
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
        api_key = os.getenv("OPENAI_API_KEY") or _load_openai_api_key_from_dotenv() or _load_openai_api_key_from_file()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.api_key = api_key
        self.model = model
        self.max_attempts = int(os.getenv("OPENAI_CALL_MAX_ATTEMPTS", "6"))
        self.base_backoff_sec = float(os.getenv("OPENAI_CALL_BASE_BACKOFF_SEC", "0.5"))
        self.max_backoff_sec = float(os.getenv("OPENAI_CALL_MAX_BACKOFF_SEC", "8.0"))

    def _call_text(self, system: str, user: str) -> str:
        # Use a fresh client per call to avoid stale connection-pool state during long eval loops.
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                client = OpenAI(api_key=self.api_key)
                resp = client.responses.create(
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
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_attempts:
                    break
                if not _is_transient_connectivity_error(exc):
                    break
                # Best-effort DNS probe before retry to make resolver issues explicit in logs.
                _probe_dns("api.openai.com")
                backoff = min(self.base_backoff_sec * (2 ** (attempt - 1)), self.max_backoff_sec)
                time.sleep(backoff + random.uniform(0.0, 0.2))
        if last_exc is not None:
            raise RuntimeError(f"OpenAI call failed after {self.max_attempts} attempt(s): {last_exc!r}") from last_exc
        raise RuntimeError("OpenAI call failed unexpectedly with no captured exception.")

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


def _load_openai_api_key_from_file(path_str: str = "bot_key.txt") -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _is_transient_connectivity_error(exc: Exception) -> bool:
    message = str(exc).lower()
    tokens = (
        "connection error",
        "temporary failure in name resolution",
        "timed out",
        "timeout",
        "name resolution",
        "connecterror",
        "apiconnectionerror",
    )
    return any(t in message for t in tokens)


def _probe_dns(hostname: str) -> None:
    try:
        socket.getaddrinfo(hostname, 443)
    except Exception:
        # Probe is informational only; retry logic still decides next step.
        pass
