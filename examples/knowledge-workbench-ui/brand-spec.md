# Knowledge Ingestion Harness — visual direction

## Product intent

- Audience: researchers building a specialist knowledge base.
- Primary tasks: inspect the current library; collect a small, traceable batch from public open-access literature or user-selected local files; parse it in an isolated RAGFlow sandbox; inspect real document and chunk results.
- Provide a fully automated intake mode with parallel source discovery and per-document acquisition; show each live stage and review gate in a task table.
- Existing collections are read-only. Sandbox creation copies parser and embedding settings from a selected existing collection, then all uploads and parse requests are scoped to the new marked sandbox ID.
- If a copied template selects MinerU while the local MinerU service is unavailable, new sandboxes use RAGFlow's built-in DeepDOC parser.
- Post-parse rule checks use actual stored chunks to check empty content and sample-length distribution; flagged items remain visible for manual review. Do not present these checks as model-based factual review.
- The interface distinguishes live task state from manual review. It does not represent sample data as live task state.

## Visual system

- Direction: calm evidence workbench based on the `tufte-dataink` recipe.
- Ground: `#FBFAF6`; ink: `#20211F`; rules: `#D9D5CA`.
- Slate identifies neutral process metadata; green indicates a completed connection or gate; amber indicates work in progress; red indicates a held or failed gate.
- Georgia / Times for editorial headings and evidence; system sans serif for controls and operational metadata.
- Preserve the sidebar-and-workbench frame, with a compact intake workflow, a prominent five-stage process view, per-item review table, live library charts, a searchable document table, and chunk inspector.
- Charts, document records, and chunk excerpts use live RAGFlow data. Only explicitly labeled readiness states describe unavailable integrations.

## Integration boundary

The local web server uses repository OpenAlex and arXiv discovery adapters, accepts browser-selected local files, and calls RAGFlow create/upload/parse endpoints only for new datasets marked `HIH-SANDBOX-`. Existing datasets can be chosen only as read-only parser templates. The rule review is performed by the local Harness against the actual RAGFlow document states and stored chunks; no model-based factual reviewer, website-wide crawler, or benchmark evaluation API is configured.
