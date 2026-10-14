# Knowledge Ingestion Harness

A professional knowledge ingestion and review workbench, with a standalone web interface and optional Agnes Harness tools.

## Features

- Live RAGFlow library charts, document tables and actual chunk inspection.
- Public-source discovery through OpenAlex/arXiv and parallel local-file ingestion.
- Source, duplicate, parsing and chunk checks with per-item progress.
- Persisted job history and recovery that checks existing document IDs without re-uploading.
- Writes limited to new libraries created and registered by this workbench instance.
- A 16-tool AGH plugin for planning, ingestion, review and evidence-backed knowledge packets.

## Run

Python 3.10+ and a RAGFlow instance with a configured embedding model are required. Set RAGFLOW_BASE_URL and RAGFLOW_API_KEY, then run:

```
python examples/knowledge-workbench-ui/server.py --port 4189
```

Open http://127.0.0.1:4189/ . Create a new target library from an available template before importing documents.

See [deployment](docs/deployment.md), [AGH integration](docs/agnes-integration.md), and [domain adaptation](docs/domain-adapters.md). Existing electromagnetic CLI and optional retrieval-assistant examples remain available. MIT license; third-party services and documents retain their own licenses.
