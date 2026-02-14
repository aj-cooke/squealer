# squealer
Text -> SQL -> Data agent

## RAG Quickstart

Build cached value profiles:

```bash
python3 build_profiles.py --db-path race_team.db --out artifacts/value_profiles.json
```

Run one question with RAG:

```bash
python3 agent.py \\
  --db-path race_team.db \\
  --question \"Which driver scored the most points in 2024?\" \\
  --enable-rag \\
  --rag-examples-path benchmarks/v1/questions_v1.json \\
  --rag-profiles-path artifacts/value_profiles.json
```

Enable embedding retrieval for example questions:

```bash
python3 agent.py \\
  --db-path race_team.db \\
  --question \"Which team has the best average finish?\" \\
  --enable-rag \\
  --rag-use-embeddings \\
  --rag-embedding-model text-embedding-3-small \\
  --rag-embedding-index-path artifacts/example_embeddings.json \\
  --rag-schema-embedding-index-path artifacts/schema_embeddings.json
```
