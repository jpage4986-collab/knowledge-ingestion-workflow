"""Backfill deterministic metadata for already-parsed documents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sync_library import (
    FIELD_TERMS,
    PaperAttachment,
    PaperIndex,
    SyncRunner,
    evaluate_chunks,
    extract_simulation_summary,
    persist_evidence_manifest,
    split_markdown,
)


def reconcile_metadata(config: Mapping[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    storage = config["storage"]
    index = PaperIndex(Path(storage["index_db"]), Path(storage["summary_csv"]))
    empty_source = type("Source", (), {"list_pdf_attachments": lambda self, limit=None: []})()
    runner = SyncRunner(config, empty_source, index, None)
    duplicates = index.mark_title_duplicates()
    # Enrich legacy discovered records from the durable candidate table. Only
    # empty paper fields are filled, so Zotero metadata remains authoritative.
    index.connection.execute(
        """
        UPDATE papers
        SET authors = COALESCE(NULLIF(authors, ''), (SELECT c.authors FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)),
            year = COALESCE(NULLIF(year, ''), (SELECT c.year FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)),
            doi = COALESCE(NULLIF(doi, ''), (SELECT c.doi FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)),
            abstract = COALESCE(NULLIF(abstract, ''), (SELECT c.abstract FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)),
            landing_url = COALESCE(NULLIF(landing_url, ''), (SELECT c.landing_url FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)),
            pdf_url = COALESCE(NULLIF(pdf_url, ''), (SELECT c.pdf_url FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)),
            source_provider = COALESCE(NULLIF(source_provider, ''), (SELECT c.source FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key))
        WHERE EXISTS (SELECT 1 FROM literature_candidates c WHERE c.candidate_id = papers.zotero_key)
        """
    )
    index.connection.commit()
    rows = index.connection.execute(
        "SELECT * FROM papers WHERE mineru_status = 'done' AND status != 'duplicate_existing' ORDER BY updated_at LIMIT ?",
        (limit or 1000000,),
    ).fetchall()
    results: list[dict[str, Any]] = [
        {"doc_id": item["doc_id"], "status": "duplicate_marked", "canonical_doc_id": item["canonical_doc_id"]}
        for item in duplicates
    ]
    for item in duplicates:
        runner.record_quarantine(item["doc_id"], "duplicate_existing", item["reason"])
    try:
        for raw_row in rows:
            row = dict(raw_row)
            doc_id = str(row["doc_id"])
            markdown_path = Path(str(row.get("mineru_markdown_path") or ""))
            if not markdown_path.is_file():
                results.append({"doc_id": doc_id, "status": "missing_mineru_markdown"})
                continue
            paper = PaperAttachment(
                zotero_key=str(row.get("zotero_key") or ""), parent_key=str(row.get("parent_key") or ""),
                title=str(row.get("title") or ""), authors=str(row.get("authors") or ""),
                year=str(row.get("year") or ""), doi=str(row.get("doi") or ""), tags=str(row.get("tags") or ""),
                filename=str(row.get("filename") or markdown_path.name),
                local_path=Path(str(row.get("raw_path") or markdown_path)),
            )
            markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
            summary = extract_simulation_summary(
                markdown, paper, doc_id, str(markdown_path),
                original_pdf_path=str(row.get("raw_path") or paper.local_path),
            )
            evidence = persist_evidence_manifest(Path(storage["root"]), doc_id, summary["evidence_ref"])
            summary["evidence_ref"] = json.dumps(evidence, ensure_ascii=False)
            if str(row.get("status") or "") == "uploaded_to_ragflow":
                summary["notes"] += " Historical RAGFlow upload detected; admission review remains unverified."
                index.upsert_review(doc_id, {
                    "accepted": False,
                    "relevance_score": 0.0,
                    "quality_score": 0.0,
                    "reason": "历史测试上传未经过模型准入审查；该记录不得作为已验证成品使用。",
                    "missing_fields": ["verified_admission_review"],
                    "simulation_parameters": {},
                })
                runner.record_quarantine(
                    doc_id,
                    "legacy_upload_without_verified_admission",
                    "历史测试上传未经过模型审查；保留 RAGFlow 文档，但不得作为已验证成品使用。",
                )
            elif str(row.get("status") or "") == "review_needs_review":
                existing_review = index.connection.execute(
                    "SELECT 1 FROM review_results WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if existing_review is None:
                    index.upsert_review(doc_id, {
                        "accepted": False,
                        "relevance_score": 0.0,
                        "quality_score": 0.0,
                        "reason": "模型评审尚未执行；保留在待审隔离区。",
                        "missing_fields": [],
                        "simulation_parameters": {},
                    })
            chunk_size = runner.adaptive_chunk_size(markdown_path)
            chunk_report = evaluate_chunks(markdown, split_markdown(markdown, chunk_size)).as_dict()
            chunk_report["max_chars"] = chunk_size
            index.upsert_summary(summary)
            index.upsert_chunk_quality(doc_id, chunk_report)
            results.append({"doc_id": doc_id, "status": "metadata_reconciled", "chunk_size": chunk_size})
        # Rebuild packets after all summaries/chunk reports are persisted. This
        # also upgrades records created by older versions of the workflow.
        packet_statuses = {
            "review_needs_review", "review_rejected", "uploaded_to_ragflow",
            "ragflow_failed", "chunk_needs_review", "duplicate_existing",
        }
        for packet_row in index.connection.execute("SELECT * FROM papers ORDER BY doc_id"):
            status = str(packet_row["status"] or "")
            if status in packet_statuses:
                reason = str(packet_row["last_error"] or "需要补充入库审查或历史记录核验")
                runner.record_quarantine(str(packet_row["doc_id"]), status, reason)
        index.export_csv()
        return results
    finally:
        index.close()


if __name__ == "__main__":
    raise SystemExit("请通过 sync_library.py reconcile-metadata 调用")
