# Knowledge ingestion and RAG workflow

A traceable workflow for discovering documents, preserving originals, parsing PDFs with MinerU, reviewing extracted claims against page-level evidence, routing approved material to RAGFlow datasets, and evaluating retrieval quality.

The design is domain-adaptable. The runnable reference implementation included here comes from an electromagnetic-simulation project and still contains domain-specific scoring, extraction fields, and answer policies. Other domains need their own profile/extractor; changing search terms alone is not sufficient. See [domain adapters](docs/domain-adapters.md).

```text
sources → deduplicate/rank → preserve original + hash → MinerU parse
        → structured extraction with evidence → quality/admission gates
        → route to a domain dataset → verify parse/chunks
        → retrieval benchmark → analyze and iterate
```

Key principles: preserve source provenance, require evidence for extracted facts, quarantine uncertainty, keep domains separated, make retries idempotent, and compare a new parser/chunking configuration against a benchmark before replacing a working version.

The repository contains no user PDFs, Zotero database, runtime index, API keys, private endpoints, personal paths, or RAGFlow corpus. See [README.md](README.md) for setup and [SECURITY.md](SECURITY.md) for the safety boundary.
