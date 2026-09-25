"""Re-check historical RAGFlow uploads without uploading a second copy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ingest_model import build_ingest_reviewer
from sync_library import (
    FIELD_TERMS,
    PaperAttachment,
    PaperIndex,
    SyncRunner,
    evaluate_chunks,
    extract_simulation_summary,
    persist_evidence_manifest,
    split_markdown,
    resolve_api_secret,
    secret_selector_label,
    utc_now,
)


def recheck_admission(
    config: Mapping[str, Any],
    limit: int | None = None,
    model_base_url: str | None = None,
    model_protocol: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Review legacy uploads in place; never call the RAGFlow upload endpoint."""
    if model_base_url or model_protocol:
        effective_config = dict(config)
        effective_model = dict(config.get("model", {}))
        if model_base_url:
            effective_model["base_url"] = model_base_url.strip()
        if model_protocol:
            effective_model["protocol"] = model_protocol.strip()
        effective_config["model"] = effective_model
        config = effective_config
    storage = config["storage"]
    index = PaperIndex(Path(storage["index_db"]), Path(storage["summary_csv"]))
    reviewer = build_ingest_reviewer(config)
    if reviewer is None:
        index.close()
        raise RuntimeError("review.enabled 必须为 true，且模型配置完整")
    model_cfg = config.get("model", {})
    api_key_env = str(model_cfg.get("api_key_env", "MODEL_API_KEY")) if isinstance(model_cfg, Mapping) else "MODEL_API_KEY"
    if not resolve_api_secret(api_key_env):
        index.close()
        raise RuntimeError(
            "模型 API key 未通过环境变量或本地配置设置："
            f"{secret_selector_label(api_key_env)}；历史上传保持原状态，不会重审或重复上传。"
        )
    if doc_ids:
        # An explicit document id is an operator-requested recheck.  Permit
        # already-published rows here as well: this is still an in-place
        # admission review and never calls the RAGFlow upload endpoint.  This
        # lets the autonomous loop repair a missing review record or refresh a
        # stale model assessment without creating a duplicate document.
        placeholders = ",".join("?" for _ in doc_ids)
        rows = index.connection.execute(
            f"SELECT * FROM papers WHERE doc_id IN ({placeholders}) ORDER BY updated_at",
            tuple(doc_ids),
        ).fetchall()
    else:
        rows = index.connection.execute(
            "SELECT * FROM papers WHERE status = 'uploaded_to_ragflow' ORDER BY updated_at LIMIT ?",
            (limit or 1000,),
        ).fetchall()
    empty_source = type("Source", (), {"list_pdf_attachments": lambda self, limit=None: []})()
    runner = SyncRunner(config, empty_source, index, reviewer)
    results: list[dict[str, Any]] = []
    try:
        for raw_row in rows:
            row = dict(raw_row)
            doc_id = str(row["doc_id"])
            markdown_path = Path(str(row.get("mineru_markdown_path") or ""))
            if not markdown_path.is_file():
                row["status"] = "review_needs_review"
                row["last_error"] = "历史 RAGFlow 文档缺少 MinerU Markdown，无法重审"
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                runner.record_quarantine(doc_id, row["status"], row["last_error"])
                results.append({"doc_id": doc_id, "status": row["status"], "error": row["last_error"]})
                continue
            paper = PaperAttachment(
                zotero_key=str(row.get("zotero_key") or ""), parent_key=str(row.get("parent_key") or ""),
                title=str(row.get("title") or ""), authors=str(row.get("authors") or ""),
                year=str(row.get("year") or ""), doi=str(row.get("doi") or ""), tags=str(row.get("tags") or ""),
                filename=str(row.get("filename") or markdown_path.name),
                local_path=Path(str(row.get("raw_path") or markdown_path)),
                dataset_key=next(
                    (
                        key for key, value in config.get("ragflow", {}).get("dataset_ids", {}).items()
                        if str(value) == str(row.get("ragflow_dataset_id") or "")
                    ),
                    "",
                ),
                abstract=str(row.get("abstract") or ""), landing_url=str(row.get("landing_url") or ""),
                pdf_url=str(row.get("pdf_url") or ""), source_provider=str(row.get("source_provider") or ""),
            )
            markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
            summary = extract_simulation_summary(
                markdown, paper, doc_id, str(markdown_path),
                original_pdf_path=str(row.get("raw_path") or paper.local_path),
            )
            summary["evidence_ref"] = json.dumps(
                persist_evidence_manifest(Path(storage["root"]), doc_id, summary["evidence_ref"]),
                ensure_ascii=False,
            )
            # For an already-published document, prefer the authoritative
            # RAGFlow chunk-quality record.  Re-splitting the local Markdown
            # with the pre-upload heuristic can produce a lower score and
            # would incorrectly downgrade a document whose live chunks have
            # already passed the positioned coverage gate.
            existing_quality = index.connection.execute(
                "SELECT * FROM chunk_quality WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if (
                str(row.get("ragflow_document_id") or "")
                and existing_quality is not None
                and float(existing_quality["score"] or 0) >= runner.chunk_min_score
            ):
                try:
                    violations = json.loads(str(existing_quality["violations"] or "[]"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    violations = []
                chunk_report = {
                    "chunk_count": int(existing_quality["chunk_count"] or 0),
                    "coverage": float(existing_quality["coverage"] or 0),
                    "boundary_integrity": float(existing_quality["boundary_integrity"] or 0),
                    "structure_integrity": float(existing_quality["structure_integrity"] or 0),
                    "score": float(existing_quality["score"] or 0),
                    "violations": violations,
                    "source": "existing_ragflow_positioned_quality",
                }
            else:
                chunk_size = runner.adaptive_chunk_size(markdown_path)
                chunk_report = evaluate_chunks(markdown, split_markdown(markdown, chunk_size)).as_dict()
                chunk_report["max_chars"] = chunk_size
            index.upsert_chunk_quality(doc_id, chunk_report)
            index.upsert_summary(summary)
            if runner.chunk_gate_enabled and chunk_report["score"] < runner.chunk_min_score:
                row["status"] = "chunk_needs_review"
                row["last_error"] = f"chunk quality {chunk_report['score']:.4f} < {runner.chunk_min_score:.4f}"
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                runner.record_quarantine(doc_id, row["status"], row["last_error"])
                results.append({"doc_id": doc_id, "status": row["status"], "quality": chunk_report["score"]})
                continue
            try:
                review = reviewer.review(
                    title=paper.title, authors=paper.authors, year=paper.year, doi=paper.doi,
                    abstract=paper.abstract, markdown=markdown, heuristic_summary=summary,
                )
                review = runner.normalize_official_case_evidence_warning(paper, review, chunk_report)
            except Exception as exc:
                review = {"accepted": False, "relevance_score": 0, "quality_score": 0, "reason": str(exc), "missing_fields": []}
                index.upsert_review(doc_id, review)
                row["status"] = "review_needs_review"
                row["last_error"] = str(exc)
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                runner.record_quarantine(doc_id, row["status"], row["last_error"])
                results.append({"doc_id": doc_id, "status": row["status"], "error": str(exc)})
                continue
            index.upsert_review(doc_id, review)
            parameters = review.get("simulation_parameters", {})
            if isinstance(parameters, Mapping):
                for field in FIELD_TERMS:
                    value = parameters.get(field)
                    if isinstance(value, str) and value.strip():
                        summary[field] = value.strip()
            if review.get("quality_score"):
                summary["quality_score"] = round(float(review["quality_score"]), 3)
            summary["notes"] = "Historical upload admission recheck: " + str(review.get("reason", ""))
            index.upsert_summary(summary)
            accepted = review.get("accepted") is True or str(review.get("accepted", "")).lower() == "true"
            relevance = float(review.get("relevance_score", 0) or 0)
            quality = float(review.get("quality_score", 0) or 0)
            min_relevance, min_quality, allow_evidence_warning = runner.admission_policy(paper)
            warning_admission = review.get("validation") == "evidence_warning" and allow_evidence_warning
            if (
                not accepted
                or relevance < min_relevance
                or quality < min_quality
                or (review.get("validation") == "evidence_gate" and not warning_admission)
            ):
                row["status"] = "review_rejected" if not accepted else "review_needs_review"
                row["last_error"] = str(review.get("reason", "历史上传未达到准入阈值"))
            elif not str(row.get("ragflow_document_id") or ""):
                row["status"] = "review_needs_review"
                row["last_error"] = "模型接受，但历史记录缺少 RAGFlow document ID；禁止猜测或重复上传"
            else:
                row["status"] = "published_ragflow"
                row["last_error"] = ""
            row["updated_at"] = utc_now()
            index.upsert_paper(row)
            # Keep an audit packet even when the old upload is accepted; this is
            # an admission record, not a request to delete the quarantine folder.
            runner.record_quarantine(doc_id, row["status"], row["last_error"] or "历史上传已完成当前准入复核")
            results.append({"doc_id": doc_id, "status": row["status"], "ragflow_document_id": row.get("ragflow_document_id", "")})
        index.export_csv()
        return results
    finally:
        index.close()


if __name__ == "__main__":
    raise SystemExit("请通过 sync_library.py recheck-admission 调用")
