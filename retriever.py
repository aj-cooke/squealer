from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_FROM_JOIN_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_QUOTED_FROM_JOIN_RE = re.compile(r"\b(?:from|join)\s+\"([^\"]+)\"", re.IGNORECASE)


@dataclass
class ExampleCase:
    case_id: str
    question: str
    canonical_sql: str
    tags: list[str]
    difficulty: str


@dataclass
class SchemaChunk:
    chunk_id: str
    table: str
    column: str
    chunk_type: str
    text: str


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _lexical_score(query: str, doc: str) -> float:
    q = _tokenize(query)
    d = _tokenize(doc)
    if not q or not d:
        return 0.0
    overlap = len(q & d)
    return overlap / math.sqrt(len(q) * len(d))


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return -1.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return -1.0
    return dot / (n1 * n2)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class _EmbeddingClient:
    def __init__(self, model: str) -> None:
        if OpenAI is None:
            raise ValueError("openai dependency is not installed")
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY") or _load_openai_api_key_from_dotenv() or _load_openai_api_key_from_file()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set for embedding retrieval.")
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def embed_texts(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            for item in resp.data:
                all_embeddings.append(list(item.embedding))
        return all_embeddings


class RAGContextBuilder:
    def __init__(
        self,
        db_path: str,
        examples_path: str,
        profiles_path: str,
        example_top_k: int = 4,
        schema_top_k: int = 4,
        value_top_k: int = 8,
        use_embeddings: bool = False,
        embedding_model: str = "text-embedding-3-small",
        embedding_index_path: str = "artifacts/example_embeddings.json",
        schema_embedding_index_path: str = "artifacts/schema_embeddings.json",
    ) -> None:
        self.db_path = db_path
        self.examples_path = examples_path
        self.profiles_path = profiles_path
        self.example_top_k = max(1, example_top_k)
        self.schema_top_k = max(1, schema_top_k)
        self.value_top_k = max(1, value_top_k)
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.embedding_index_path = embedding_index_path
        self.schema_embedding_index_path = schema_embedding_index_path

        self.examples = self._load_examples(examples_path)
        self.schema_chunks = self._load_schema_chunks(db_path)
        self.profiles = self._load_profiles(profiles_path)

        self._embedder: _EmbeddingClient | None = None
        self._example_embeddings: list[list[float]] = []
        self._schema_embeddings: list[list[float]] = []
        self.init_warnings: list[str] = []
        if use_embeddings:
            try:
                self._embedder = _EmbeddingClient(embedding_model)
            except Exception as exc:
                self._embedder = None
                self.use_embeddings = False
                self.init_warnings.append(f"Embedding retrieval disabled: {exc}")

            if self._embedder is not None:
                try:
                    self._example_embeddings = self._load_or_build_example_embeddings()
                except Exception as exc:
                    self._example_embeddings = []
                    self.init_warnings.append(f"Example embeddings disabled: {exc}")

                try:
                    self._schema_embeddings = self._load_or_build_schema_embeddings()
                except Exception as exc:
                    self._schema_embeddings = []
                    self.init_warnings.append(f"Schema embeddings disabled: {exc}")

    def build_context(self, question: str, extra_tables: list[str] | None = None) -> dict[str, Any]:
        examples = self.retrieve_examples(question)
        schema_hits = self.retrieve_schema(question)
        selected_tables = self._select_candidate_tables(question, schema_hits, extra_tables or [])
        value_hints = self.retrieve_value_hints(question, selected_tables)
        return {
            "examples": examples,
            "schema_hits": schema_hits,
            "candidate_tables": selected_tables,
            "value_hints": value_hints,
            "context_text": self._format_context_text(examples, schema_hits, value_hints),
        }

    def retrieve_examples(self, question: str) -> list[dict[str, Any]]:
        if not self.examples:
            return []
        scored: list[tuple[float, ExampleCase]] = []
        if self.use_embeddings and self._embedder is not None and self._example_embeddings:
            qv = self._embedder.embed_text(question)
            for ex, vec in zip(self.examples, self._example_embeddings):
                scored.append((_cosine_similarity(qv, vec), ex))
        else:
            for ex in self.examples:
                score = _lexical_score(question, f"{ex.question} {' '.join(ex.tags)}")
                scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.example_top_k]
        return [
            {
                "score": round(score, 4),
                "id": ex.case_id,
                "question": ex.question,
                "sql": ex.canonical_sql,
                "tags": ex.tags,
                "difficulty": ex.difficulty,
            }
            for score, ex in top
        ]

    def retrieve_schema(self, question: str) -> list[dict[str, Any]]:
        scored: list[tuple[float, SchemaChunk]] = []
        if self.use_embeddings and self._embedder is not None and self._schema_embeddings:
            qv = self._embedder.embed_text(question)
            for chunk, vec in zip(self.schema_chunks, self._schema_embeddings):
                emb_score = _cosine_similarity(qv, vec)
                lex_score = _lexical_score(question, chunk.text)
                score = (0.8 * emb_score) + (0.2 * lex_score)
                scored.append((score, chunk))
        else:
            for chunk in self.schema_chunks:
                scored.append((_lexical_score(question, chunk.text), chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.schema_top_k]
        selected_chunks = [chunk for _, chunk in top]
        selected_ids = {chunk.chunk_id for chunk in selected_chunks}

        # Ensure table context is always present for any selected column chunk.
        table_chunk_by_table: dict[str, tuple[float, SchemaChunk]] = {}
        for score, chunk in scored:
            if chunk.chunk_type == "table" and chunk.table not in table_chunk_by_table:
                table_chunk_by_table[chunk.table] = (score, chunk)

        for chunk in list(selected_chunks):
            if chunk.chunk_type != "column":
                continue
            table_entry = table_chunk_by_table.get(chunk.table)
            if table_entry is None:
                continue
            _, table_chunk = table_entry
            if table_chunk.chunk_id not in selected_ids:
                selected_chunks.append(table_chunk)
                selected_ids.add(table_chunk.chunk_id)

        score_by_chunk_id = {chunk.chunk_id: score for score, chunk in scored}
        return [
            {
                "score": round(score_by_chunk_id.get(chunk.chunk_id, 0.0), 4),
                "table": chunk.table,
                "column": chunk.column,
                "type": chunk.chunk_type,
                "text": chunk.text,
            }
            for chunk in selected_chunks
        ]

    def retrieve_value_hints(self, question: str, tables: list[str]) -> list[dict[str, Any]]:
        if not self.profiles:
            return []
        qtokens = _tokenize(question)
        candidates: list[tuple[float, dict[str, Any]]] = []

        table_map = self.profiles.get("tables", {})
        for table in tables:
            tinfo = table_map.get(table)
            if not isinstance(tinfo, dict):
                continue
            cols = tinfo.get("columns", {})
            if not isinstance(cols, dict):
                continue
            for col_name, col_info in cols.items():
                if not isinstance(col_info, dict):
                    continue
                hint = self._build_column_hint(table, col_name, col_info)
                if not hint:
                    continue
                score = self._column_relevance_score(qtokens, table, col_name, col_info)
                candidates.append((score, hint))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[: self.value_top_k]
        return [hint for _, hint in top]

    def _select_candidate_tables(
        self,
        question: str,
        schema_hits: list[dict[str, Any]],
        extra_tables: list[str],
    ) -> list[str]:
        tables: list[str] = []
        seen: set[str] = set()

        for table in extra_tables:
            if table not in seen:
                seen.add(table)
                tables.append(table)

        for hit in schema_hits:
            table = hit.get("table", "")
            if table and table not in seen:
                seen.add(table)
                tables.append(table)

        qtokens = _tokenize(question)
        for chunk in self.schema_chunks:
            if chunk.table in seen:
                continue
            if chunk.table.lower() in qtokens:
                seen.add(chunk.table)
                tables.append(chunk.table)

        return tables

    def _build_column_hint(self, table: str, column: str, col_info: dict[str, Any]) -> dict[str, Any] | None:
        kind = col_info.get("inferred_kind", "")
        if kind == "numeric" and isinstance(col_info.get("numeric_stats"), dict):
            stats = col_info["numeric_stats"]
            return {
                "table": table,
                "column": column,
                "kind": kind,
                "summary": (
                    f"{table}.{column} numeric stats: min={stats.get('min')}, p25={stats.get('p25')}, "
                    f"p50={stats.get('p50')}, p75={stats.get('p75')}, max={stats.get('max')}"
                ),
                "distinct_count": col_info.get("distinct_count"),
                "null_rate": col_info.get("null_rate"),
            }
        if kind == "temporal" and isinstance(col_info.get("temporal_range"), dict):
            rng = col_info["temporal_range"]
            return {
                "table": table,
                "column": column,
                "kind": kind,
                "summary": f"{table}.{column} date range: min={rng.get('min')}, max={rng.get('max')}",
                "distinct_count": col_info.get("distinct_count"),
                "null_rate": col_info.get("null_rate"),
            }
        top_values = col_info.get("top_values")
        if isinstance(top_values, list) and top_values:
            rendered = ", ".join(f"{item.get('value')} ({item.get('count')})" for item in top_values[:5])
            return {
                "table": table,
                "column": column,
                "kind": "categorical",
                "summary": f"{table}.{column} frequent values: {rendered}",
                "distinct_count": col_info.get("distinct_count"),
                "null_rate": col_info.get("null_rate"),
            }
        return None

    def _column_relevance_score(self, qtokens: set[str], table: str, column: str, col_info: dict[str, Any]) -> float:
        score = 0.0
        names = _tokenize(f"{table} {column} {' '.join(col_info.get('sample_tokens', []))}")
        score += 0.7 * len(qtokens & names)

        for item in col_info.get("top_values", [])[:8]:
            value = str(item.get("value", "")).lower()
            if value and value in qtokens:
                score += 2.0

        kind = col_info.get("inferred_kind", "")
        if kind == "temporal" and any(tok in qtokens for tok in {"date", "season", "year", "latest", "earliest"}):
            score += 1.0
        if kind == "numeric" and any(tok in qtokens for tok in {"avg", "average", "max", "min", "median", "percentile", "sum"}):
            score += 1.0
        return score

    def _format_context_text(
        self,
        examples: list[dict[str, Any]],
        schema_hits: list[dict[str, Any]],
        value_hints: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []

        if examples:
            lines.append("Example question-SQL pairs (use as patterns, adapt tables/filters carefully):")
            for ex in examples:
                lines.append(f"- [{ex['id']}] Q: {ex['question']}")
                lines.append(f"  SQL: {ex['sql']}")

        if schema_hits:
            lines.append("Relevant schema snippets:")
            for chunk in schema_hits:
                prefix = chunk.get("type", "schema")
                if chunk.get("column"):
                    lines.append(f"- ({prefix}) {chunk['table']}.{chunk['column']}: {chunk['text']}")
                else:
                    lines.append(f"- ({prefix}) {chunk['table']}: {chunk['text']}")

        if value_hints:
            lines.append("Likely relevant value/profile hints:")
            for hint in value_hints:
                lines.append(f"- {hint['summary']}")

        return "\n".join(lines).strip()

    def _load_examples(self, path: str) -> list[ExampleCase]:
        p = Path(path)
        if not p.exists():
            return []
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: list[ExampleCase] = []
        for item in raw:
            out.append(
                ExampleCase(
                    case_id=str(item.get("id", "")),
                    question=str(item.get("question", "")),
                    canonical_sql=str(item.get("canonical_sql", "")),
                    tags=[str(t) for t in item.get("tags", [])],
                    difficulty=str(item.get("difficulty", "unknown")),
                )
            )
        return out

    def _load_schema_chunks(self, db_path: str) -> list[SchemaChunk]:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            tables = [str(r[0]) for r in cur.fetchall()]

            chunks: list[SchemaChunk] = []
            for table in tables:
                cur.execute(f"PRAGMA table_info({_quote_ident(table)})")
                cols = cur.fetchall()
                col_defs = []
                pk_cols: list[str] = []
                for col in cols:
                    _, name, coltype, notnull, _, pk = col
                    col_type = str(coltype or "")
                    bit = f"{name} {col_type}".strip()
                    if pk:
                        bit += " PK"
                        pk_cols.append(str(name))
                    if notnull:
                        bit += " NOT NULL"
                    col_defs.append(bit)

                cur.execute(f"PRAGMA foreign_key_list({_quote_ident(table)})")
                fk_rows = cur.fetchall()
                fk_bits = []
                fk_map: dict[str, tuple[str, str]] = {}
                for fk in fk_rows:
                    _, _, ref_table, from_col, to_col, *_ = fk
                    from_c = str(from_col)
                    ref_t = str(ref_table)
                    to_c = str(to_col)
                    fk_bits.append(f"{from_c}->{ref_t}.{to_c}")
                    fk_map[from_c] = (ref_t, to_c)

                table_desc = self._describe_table(table, [str(c[1]) for c in cols], pk_cols, fk_bits)
                table_text = f"{table_desc}; columns: {', '.join(col_defs)}"
                if fk_bits:
                    table_text += f"; foreign_keys: {', '.join(fk_bits)}"

                chunks.append(
                    SchemaChunk(
                        chunk_id=f"table:{table}",
                        table=table,
                        column="",
                        chunk_type="table",
                        text=table_text,
                    )
                )

                for col in cols:
                    _, name, coltype, notnull, _, pk = col
                    col_name = str(name)
                    col_type = str(coltype or "")
                    ref = fk_map.get(col_name)
                    col_desc = self._describe_column(table, col_name, col_type, bool(notnull), bool(pk), ref)
                    chunks.append(
                        SchemaChunk(
                            chunk_id=f"column:{table}.{col_name}",
                            table=table,
                            column=col_name,
                            chunk_type="column",
                            text=col_desc,
                        )
                    )
            return chunks
        finally:
            conn.close()

    def _describe_table(self, table: str, columns: list[str], pk_cols: list[str], fk_bits: list[str]) -> str:
        col_sample = ", ".join(columns[:6]) if columns else "none"
        pk_text = ", ".join(pk_cols) if pk_cols else "none"
        fk_text = ", ".join(fk_bits) if fk_bits else "none"
        return (
            f"Table {table} stores records for {table.replace('_', ' ')}"
            f"; key columns include: {col_sample}"
            f"; primary key columns: {pk_text}"
            f"; join relationships: {fk_text}"
        )

    def _describe_column(
        self,
        table: str,
        column: str,
        declared_type: str,
        notnull: bool,
        is_pk: bool,
        ref: tuple[str, str] | None,
    ) -> str:
        tokens = column.lower().split("_")
        semantic = " ".join(tokens)
        role_bits: list[str] = []
        if is_pk:
            role_bits.append("primary key")
        if column.lower().endswith("_id"):
            role_bits.append("identifier")
        if any(tok in column.lower() for tok in ("date", "time", "season", "year")):
            role_bits.append("time-like field")
        if any(tok in column.lower() for tok in ("count", "total", "avg", "position", "points", "lap")):
            role_bits.append("metric/filter field")
        if notnull:
            role_bits.append("non-null")
        if ref is not None:
            role_bits.append(f"foreign key to {ref[0]}.{ref[1]}")
        role_text = ", ".join(role_bits) if role_bits else "general attribute"
        dtype = declared_type if declared_type else "unknown type"
        return (
            f"Column {table}.{column} ({dtype}) represents {semantic}; "
            f"usage hints: {role_text}"
        )

    def _load_profiles(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if not path or not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}
        return payload

    def _load_or_build_example_embeddings(self) -> list[list[float]]:
        if self._embedder is None or not self.examples:
            return []

        p = Path(self.embedding_index_path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    isinstance(cached, dict)
                    and cached.get("embedding_model") == self.embedding_model
                    and cached.get("examples_path") == self.examples_path
                    and cached.get("content_signature") == self._example_content_signature()
                    and isinstance(cached.get("vectors"), list)
                    and len(cached["vectors"]) == len(self.examples)
                ):
                    return [list(map(float, vec)) for vec in cached["vectors"]]
            except Exception:
                pass

        questions = [ex.question for ex in self.examples]
        vectors = self._embedder.embed_texts(questions)

        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedding_model,
            "examples_path": self.examples_path,
            "content_signature": self._example_content_signature(),
            "count": len(vectors),
            "vectors": vectors,
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

        return vectors

    def _load_or_build_schema_embeddings(self) -> list[list[float]]:
        if self._embedder is None or not self.schema_chunks:
            return []

        p = Path(self.schema_embedding_index_path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    isinstance(cached, dict)
                    and cached.get("embedding_model") == self.embedding_model
                    and cached.get("db_path") == self.db_path
                    and cached.get("content_signature") == self._schema_content_signature()
                    and isinstance(cached.get("vectors"), list)
                    and len(cached["vectors"]) == len(self.schema_chunks)
                ):
                    return [list(map(float, vec)) for vec in cached["vectors"]]
            except Exception:
                pass

        texts = [chunk.text for chunk in self.schema_chunks]
        vectors = self._embedder.embed_texts(texts)

        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedding_model,
            "db_path": self.db_path,
            "content_signature": self._schema_content_signature(),
            "chunk_ids": [chunk.chunk_id for chunk in self.schema_chunks],
            "count": len(vectors),
            "vectors": vectors,
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

        return vectors

    def _example_content_signature(self) -> str:
        return "|".join(f"{ex.case_id}:{ex.question}" for ex in self.examples)

    def _schema_content_signature(self) -> str:
        return "|".join(f"{chunk.chunk_id}:{chunk.text}" for chunk in self.schema_chunks)


def extract_tables_from_sql(sql: str) -> list[str]:
    if not sql:
        return []
    tables: list[str] = []
    seen: set[str] = set()

    for match in _QUOTED_FROM_JOIN_RE.finditer(sql):
        table = match.group(1).strip()
        if table and table not in seen:
            seen.add(table)
            tables.append(table)

    for match in _FROM_JOIN_RE.finditer(sql):
        table = match.group(1).strip()
        if table and table not in seen:
            seen.add(table)
            tables.append(table)

    return tables


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
