"""Compensating retry path for parsed documents held outside the corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ingest_model import build_ingest_reviewer
from sync_library import (
    FIELD_TERMS,
    PaperAttachment,
    PaperIndex,
    RAGFlowClient,
    SyncRunner,
    extract_simulation_summary,
    persist_evidence_manifest,
    published_document_name,
    reserve_ragflow_upload_attempt,
    sha256_file,
    load_config,
    split_markdown,
    evaluate_chunks,
    utc_now,
)


RETRYABLE = {"review_needs_review", "ragflow_failed", "stored_raw"}


def retry_failed(
    config: Mapping[str, Any],
    limit: int | None = None,
    model_base_url: str | None = None,
    model_protocol: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
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
    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        rows = index.connection.execute(
            f"SELECT * FROM papers WHERE status IN ('review_needs_review','review_rejected','ragflow_failed','stored_raw') AND doc_id IN ({placeholders}) ORDER BY updated_at",
            tuple(doc_ids),
        ).fetchall()
    else:
        rows = index.connection.execute(
            "SELECT * FROM papers WHERE status IN ('review_needs_review','ragflow_failed','stored_raw') ORDER BY updated_at LIMIT ?",
            (limit or 1000,),
        ).fetchall()
    results: list[dict[str, Any]] = []
    empty_source = type("Source", (), {"list_pdf_attachments": lambda self, limit=None: []})()
    retry_runner = SyncRunner(config, empty_source, index, reviewer)
    try:
        for raw_row in rows:
            row = dict(raw_row)
            doc_id = str(row["doc_id"])
            markdown_path = Path(str(row.get("mineru_markdown_path") or ""))
            if not markdown_path.is_file():
                # Discovery/seed ingestion may have safely retained the
                # original PDF but not reached MinerU yet. Treat that state
                # as recoverable instead of leaving it outside the retry
                # queue forever.
                raw_path = Path(str(row.get("raw_path") or ""))
                if row.get("status") == "stored_raw" and raw_path.is_file():
                    try:
                        parsed = retry_runner.mineru.parse(raw_path, doc_id)
                    except Exception as exc:
                        parsed = {"status": "mineru_failed", "error": str(exc)}
                    row["mineru_status"] = parsed.get("status", "mineru_failed")
                    row["mineru_markdown_path"] = parsed.get("markdown_path", "")
                    row["status"] = "parsed" if parsed.get("status") == "done" else parsed.get("status", "mineru_failed")
                    row["last_error"] = parsed.get("error", "")
                    row["updated_at"] = utc_now()
                    index.upsert_paper(row)
                    if parsed.get("status") == "done":
                        markdown_path = Path(str(parsed.get("markdown_path") or ""))
                if markdown_path.is_file():
                    pass
                else:
                    retry_runner.record_quarantine(doc_id, "retry_missing_mineru_output", "MinerU Markdown 不存在")
                    results.append({"doc_id": doc_id, "status": "retry_missing_mineru_output"})
                    continue
            paper = PaperAttachment(
                zotero_key=str(row.get("zotero_key") or ""), parent_key=str(row.get("parent_key") or ""),
                title=str(row.get("title") or ""), authors=str(row.get("authors") or ""),
                year=str(row.get("year") or ""), doi=str(row.get("doi") or ""), tags=str(row.get("tags") or ""),
                filename=str(row.get("filename") or markdown_path.name), local_path=Path(str(row.get("raw_path") or markdown_path)),
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
                markdown,
                paper,
                doc_id,
                str(markdown_path),
                original_pdf_path=str(row.get("raw_path") or paper.local_path),
            )
            summary["evidence_ref"] = json.dumps(
                persist_evidence_manifest(Path(storage["root"]), doc_id, summary["evidence_ref"]),
                ensure_ascii=False,
            )
            chunk_size = retry_runner.adaptive_chunk_size(markdown_path)
            chunk_report = evaluate_chunks(markdown, split_markdown(markdown, chunk_size)).as_dict()
            chunk_report["max_chars"] = chunk_size
            index.upsert_chunk_quality(doc_id, chunk_report)
            # Persist deterministic extraction/evidence before the external
            # model call, so a missing endpoint cannot discard local evidence.
            index.upsert_summary(summary)
            if bool(config.get("chunk_quality", {}).get("enabled", False)) and chunk_report["score"] < float(config.get("chunk_quality", {}).get("min_score", 0.98)):
                # Route chunk-only failures to the adaptive repair loop.  They
                # are not model-admission failures and should not be stranded
                # in the ordinary review queue.
                row["status"] = "chunk_needs_review"
                row["last_error"] = f"chunk quality {chunk_report['score']:.4f} below gate"
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                retry_runner.record_quarantine(doc_id, row["status"], row["last_error"])
                results.append({"doc_id": doc_id, "status": row["status"], "quality": chunk_report["score"]})
                continue
            rag_id = ""
            try:
                review = reviewer.review(title=paper.title, authors=paper.authors, year=paper.year, doi=paper.doi, abstract=paper.abstract, markdown=markdown, heuristic_summary=summary)
            except Exception as exc:
                row["status"] = "review_needs_review"
                row["last_error"] = str(exc)
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                index.upsert_review(doc_id, {"accepted": False, "relevance_score": 0, "quality_score": 0, "reason": str(exc), "missing_fields": []})
                retry_runner.record_quarantine(doc_id, row["status"], row["last_error"])
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
            summary["notes"] = "Retry model review: " + str(review.get("reason", ""))
            index.upsert_summary(summary)
            accepted = review.get("accepted") is True or str(review.get("accepted", "")).lower() == "true"
            relevance = float(review.get("relevance_score", 0) or 0)
            quality = float(review.get("quality_score", 0) or 0)
            review_cfg = config.get("review", {})
            min_relevance, min_quality, allow_evidence_warning = retry_runner.admission_policy(paper)
            warning_admission = review.get("validation") == "evidence_warning" and allow_evidence_warning
            if not accepted or relevance < min_relevance or quality < min_quality or (review.get("validation") == "evidence_gate" and not warning_admission):
                row["status"] = "review_rejected" if not accepted else "review_needs_review"
                row["last_error"] = str(review.get("reason", "retry review below admission gate"))
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                retry_runner.record_quarantine(doc_id, row["status"], row["last_error"])
                results.append({"doc_id": doc_id, "status": row["status"], "quality": summary["quality_score"]})
                continue
            published = Path(storage["root"]) / "published" / f"{doc_id}.md"
            published.parent.mkdir(parents=True, exist_ok=True)
            published.write_text("# Paper simulation summary\n\n" + "\n".join(f"- {key}: {summary.get(key, '')}" for key in FIELD_TERMS) + "\n\n# MinerU full-text output\n\n" + markdown, encoding="utf-8")
            try:
                dataset_id = str(row.get("ragflow_dataset_id") or "")
                if not dataset_id:
                    dataset_id = retry_runner.dataset_id_for(paper)
                if not dataset_id:
                    raise RuntimeError("未配置目标 RAGFlow dataset_id")
                client = RAGFlowClient(str(config["ragflow"]["base_url"]), dataset_id, str(config["ragflow"]["api_key_env"]))
                upload_original_pdf = bool(config.get("ragflow", {}).get("upload_original_pdf", False))
                upload_mode = "original_pdf" if upload_original_pdf else "local_mineru_markdown"
                upload_path = Path(str(row.get("raw_path") or "")) if upload_original_pdf else published
                if upload_original_pdf and not upload_path.is_file():
                    raise FileNotFoundError(f"原始 PDF 不存在：{upload_path}")
                display_name = (
                    f"{doc_id}--{sha256_file(upload_path)[:12]}.pdf"
                    if upload_original_pdf
                    else published_document_name(doc_id, markdown)
                )
                existing = client.find_active_document(display_name)
                reused_existing = bool(existing)
                if existing:
                    rag_id = str(existing["id"])
                else:
                    reserve_ragflow_upload_attempt(index, row, is_original_pdf=upload_original_pdf)
                    rag_id = client.upload(upload_path, display_name=display_name)
                parser_update = (
                    retry_runner.original_pdf_parser_update()
                    if upload_original_pdf
                    else retry_runner.preparsed_markdown_update()
                )
                if parser_update:
                    client.update_document(rag_id, parser_update)
                    row["ragflow_parser_config"] = json.dumps(parser_update, ensure_ascii=False)
                client.trigger_parse(rag_id)
                client.wait_done(rag_id)
                if not client.list_chunks(rag_id):
                    raise RuntimeError(f"RAGFlow 文档解析完成但没有生成分块：{rag_id}")
            except Exception as exc:
                fallback_enabled = bool(config.get("ragflow", {}).get("fallback_to_preparsed_markdown", True))
                if upload_original_pdf and fallback_enabled:
                    try:
                        # Preserve the original-PDF document even when its
                        # service-side parser fails. The raw PDF is a required
                        # visual fallback asset; the Markdown upload is only a
                        # parallel retrieval fallback, never a replacement.
                        # The failed PDF remains auditable in RAGFlow instead
                        # of being destructively deleted.
                        fallback_name = published_document_name(doc_id, markdown)
                        fallback_existing = client.find_active_document(fallback_name)
                        fallback_id = (
                            str(fallback_existing["id"])
                            if fallback_existing
                            else _upload_fallback(index, client, published, fallback_name, row)
                        )
                        fallback_update = retry_runner.preparsed_markdown_update()
                        if fallback_update:
                            client.update_document(fallback_id, fallback_update)
                            row["ragflow_parser_config"] = json.dumps(fallback_update, ensure_ascii=False)
                        client.trigger_parse(fallback_id)
                        client.wait_done(fallback_id)
                        if not client.list_chunks(fallback_id):
                            raise RuntimeError(f"RAGFlow Markdown 兜底也没有生成分块：{fallback_id}")
                        rag_id = fallback_id
                        upload_mode = "local_mineru_markdown_fallback"
                    except Exception as fallback_exc:
                        row["status"] = "ragflow_failed"
                        row["last_error"] = f"原始 PDF 解析失败：{exc}；Markdown 兜底失败：{fallback_exc}"
                        row["updated_at"] = utc_now()
                        index.upsert_paper(row)
                        retry_runner.record_quarantine(doc_id, row["status"], row["last_error"])
                        results.append({"doc_id": doc_id, "status": row["status"], "error": row["last_error"]})
                        continue
                else:
                    row["status"] = "ragflow_failed"
                    row["last_error"] = str(exc)
                    row["updated_at"] = utc_now()
                    index.upsert_paper(row)
                    retry_runner.record_quarantine(doc_id, row["status"], row["last_error"])
                    results.append({"doc_id": doc_id, "status": row["status"], "error": row["last_error"]})
                    continue
            row["ragflow_document_id"] = rag_id
            row["ragflow_dataset_id"] = dataset_id
            row["status"] = "published_ragflow"
            row["last_error"] = ""
            row["updated_at"] = utc_now()
            index.upsert_paper(row)
            results.append({"doc_id": doc_id, "status": row["status"], "ragflow_document_id": rag_id, "reused_existing_document": reused_existing, "ragflow_upload_mode": upload_mode})
        index.export_csv()
        return results
    finally:
        index.close()


def _upload_fallback(index: PaperIndex, client: RAGFlowClient, published: Path, fallback_name: str, row: dict[str, Any]) -> str:
    reserve_ragflow_upload_attempt(index, row, is_original_pdf=False)
    return client.upload(published, display_name=fallback_name)
