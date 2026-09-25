"""Read-only audit of live RAGFlow documents referenced by the local index."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from retrieval_benchmark import text_fragmentation_diagnostic
from sync_library import RAGFlowClient, resolve_api_secret, secret_selector_label


def audit_ragflow_runtime(config: Mapping[str, Any], dataset_key: str | None = None) -> dict[str, Any]:
    storage = config["storage"]
    root = Path(storage["root"])
    db_path = Path(storage["index_db"])
    rag = config["ragflow"]
    reverse_datasets = {str(value): str(key) for key, value in rag.get("dataset_ids", {}).items()}
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT doc_id, title, status, ragflow_document_id, ragflow_pdf_document_id, ragflow_dataset_id, ragflow_parser_config "
            "FROM papers WHERE ragflow_document_id IS NOT NULL AND ragflow_document_id != '' ORDER BY doc_id"
        ).fetchall()
    finally:
        connection.close()
    if dataset_key:
        rows = [row for row in rows if reverse_datasets.get(str(row["ragflow_dataset_id"] or ""), "") == dataset_key]
    clients: dict[str, RAGFlowClient] = {}
    documents: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pdf_assets_indexed = sum(
        1 for row in rows if row["ragflow_pdf_document_id"]
    )
    pdf_assets_checked = 0
    api_key_env = str(rag.get("api_key_env", "RAGFLOW_API_KEY"))
    if not resolve_api_secret(api_key_env):
        report = {
            "status": "needs_credentials",
            "dataset_key_filter": dataset_key,
            "api_key_source": secret_selector_label(api_key_env),
            "documents_indexed": len(rows),
            "documents_checked": 0,
            "documents": [],
            "violations": [],
            "errors": [{"type": "missing_api_key", "api_key_source": secret_selector_label(api_key_env)}],
        }
        return _write_report(root, report)
    for row in rows:
        live_dataset_id = str(row["ragflow_dataset_id"] or "")
        live_dataset_key = reverse_datasets.get(live_dataset_id, "")
        document_id = str(row["ragflow_document_id"])
        item: dict[str, Any] = {
            "doc_id": str(row["doc_id"]),
            "title": str(row["title"] or ""),
            "status": str(row["status"] or ""),
            "dataset_key": live_dataset_key or None,
            "dataset_id": live_dataset_id,
            "document_id": document_id,
        }
        try:
            if live_dataset_id not in clients:
                clients[live_dataset_id] = RAGFlowClient(
                    str(rag["base_url"]), live_dataset_id, api_key_env,
                    timeout=float(rag.get("retrieval_timeout_seconds", 20)),
                )
            client = clients[live_dataset_id]
            live = client.document(document_id)
            if not live:
                violation = {"doc_id": row["doc_id"], "type": "live_document_not_found", "document_id": document_id}
                violations.append(violation)
                item["violations"] = [violation["type"]]
                documents.append(item)
                continue
            chunks = client.list_chunks(document_id)
            content = "\n".join(str(chunk.get("content", "")) for chunk in chunks)
            parser_config = live.get("parser_config") if isinstance(live.get("parser_config"), Mapping) else {}
            live_chunk_count = int(live.get("chunk_count") or 0)
            item.update({
                "run": live.get("run"),
                "progress": live.get("progress"),
                "metadata_chunk_count": live_chunk_count,
                "chunk_count": len(chunks),
                "parser_layout": parser_config.get("layout_recognize"),
                "delimiter_is_real_newline": parser_config.get("delimiter") == "\n",
                "text_diagnostic": text_fragmentation_diagnostic(content),
            })
            item_violations: list[str] = []
            if str(live.get("run") or live.get("status") or "") not in {"DONE", "3"}:
                item_violations.append("live_document_not_done")
            if not chunks:
                item_violations.append("live_document_zero_chunks")
            # RAGFlow can expose a stale metadata count after a parse in
            # either direction. When the document is DONE and the paginated
            # endpoint returns usable chunks, that live set is authoritative;
            # retain metadata drift as a warning instead of rejecting content.
            item["chunk_count_delta"] = live_chunk_count - len(chunks)
            if live_chunk_count and abs(live_chunk_count - len(chunks)) > 2:
                if chunks and str(live.get("run") or live.get("status") or "") in {"DONE", "3"}:
                    warning = {
                        "doc_id": row["doc_id"],
                        "type": "metadata_chunk_count_stale",
                        "document_id": document_id,
                        "metadata_chunk_count": live_chunk_count,
                        "list_chunk_count": len(chunks),
                    }
                    warnings.append(warning)
                    item.setdefault("warnings", []).append(warning["type"])
                else:
                    item_violations.append("chunk_count_mismatch")
            if row["ragflow_parser_config"]:
                try:
                    configured_container = json.loads(str(row["ragflow_parser_config"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    configured_container = {}
                configured_parser = (
                    configured_container.get("parser_config", configured_container)
                    if isinstance(configured_container, Mapping)
                    else {}
                )
                configured_layout = str(configured_parser.get("layout_recognize") or "") if isinstance(configured_parser, Mapping) else ""
                live_layout = str(parser_config.get("layout_recognize") or "")
                if configured_layout and live_layout != configured_layout:
                    item_violations.append("parser_layout_mismatch")
                # Plain Text is required only for a pre-parsed Markdown upload.
                # PDF uploads intentionally use the configured MinerU layout.
                if configured_layout.lower() == "plain text" and live_layout != "Plain Text":
                    item_violations.append("preparsed_document_layout_not_plain_text")
                configured_delimiter = configured_parser.get("delimiter") if isinstance(configured_parser, Mapping) else None
                if configured_delimiter is not None and parser_config.get("delimiter") != configured_delimiter:
                    item_violations.append("preparsed_document_delimiter_not_newline")
            if item["text_diagnostic"].get("suspected_reparse_damage"):
                item_violations.append("suspected_character_fragmentation")
            item["violations"] = item_violations
            if item_violations:
                violations.extend({"doc_id": row["doc_id"], "type": violation, "document_id": document_id} for violation in item_violations)
            pdf_document_id = str(row["ragflow_pdf_document_id"] or "")
            if pdf_document_id:
                pdf_assets_checked += 1
                pdf_asset: dict[str, Any] = {"document_id": pdf_document_id}
                try:
                    same_as_serving_document = pdf_document_id == document_id
                    pdf_live = live if same_as_serving_document else client.document(pdf_document_id)
                    if not pdf_live:
                        pdf_asset["violations"] = ["original_pdf_not_found"]
                        violations.append({
                            "doc_id": row["doc_id"], "type": "original_pdf_not_found",
                            "document_id": pdf_document_id,
                        })
                    else:
                        pdf_chunks = chunks if same_as_serving_document else client.list_chunks(pdf_document_id)
                        pdf_parser = pdf_live.get("parser_config") if isinstance(pdf_live.get("parser_config"), Mapping) else {}
                        pdf_run = str(pdf_live.get("run") or pdf_live.get("status") or "")
                        pdf_layout = str(pdf_parser.get("layout_recognize") or "")
                        pdf_violations: list[str] = []
                        if pdf_run not in {"DONE", "3"}:
                            pdf_violations.append("original_pdf_not_done")
                        if not pdf_chunks:
                            pdf_violations.append("original_pdf_zero_chunks")
                        if not pdf_layout or "mineru" not in pdf_layout.lower():
                            pdf_violations.append("original_pdf_not_mineru")
                        pdf_asset.update({
                            "name": pdf_live.get("name"),
                            "run": pdf_live.get("run"),
                            "status": pdf_live.get("status"),
                            "retrieval_enabled": str(pdf_live.get("status") or "").lower() not in {"0", "disabled", "false"},
                            "size": pdf_live.get("size"),
                            "metadata_chunk_count": int(pdf_live.get("chunk_count") or 0),
                            "chunk_count": len(pdf_chunks),
                            "parser_layout": pdf_layout,
                            "violations": pdf_violations,
                        })
                        if pdf_violations:
                            violations.extend({
                                "doc_id": row["doc_id"], "type": violation,
                                "document_id": pdf_document_id,
                            } for violation in pdf_violations)
                        pdf_count = int(pdf_live.get("chunk_count") or 0)
                        if not same_as_serving_document and pdf_count and abs(pdf_count - len(pdf_chunks)) > 2 and not pdf_violations:
                            warning = {
                                "doc_id": row["doc_id"], "type": "original_pdf_chunk_count_stale",
                                "document_id": pdf_document_id,
                                "metadata_chunk_count": pdf_count,
                                "list_chunk_count": len(pdf_chunks),
                            }
                            warnings.append(warning)
                            pdf_asset.setdefault("warnings", []).append(warning["type"])
                        if not str(pdf_live.get("name") or "").lower().endswith(".pdf"):
                            pdf_asset["violations"].append("original_pdf_name_not_pdf")
                            violations.append({
                                "doc_id": row["doc_id"], "type": "original_pdf_name_not_pdf",
                                "document_id": pdf_document_id,
                            })
                except Exception as exc:
                    error = {
                        "doc_id": row["doc_id"], "document_id": pdf_document_id,
                        "type": f"original_pdf_{type(exc).__name__}", "error": str(exc),
                    }
                    errors.append(error)
                    pdf_asset["error"] = error["error"]
                item["original_pdf_asset"] = pdf_asset
        except Exception as exc:
            error = {"doc_id": row["doc_id"], "document_id": document_id, "type": type(exc).__name__, "error": str(exc)}
            errors.append(error)
            item["error"] = error["error"]
        documents.append(item)
    report = {
        "status": "pass" if not violations and not errors else "needs_attention",
        "dataset_key_filter": dataset_key,
        "documents_indexed": len(rows),
        "documents_checked": len(documents),
        "pdf_assets_indexed": pdf_assets_indexed,
        "pdf_assets_checked": pdf_assets_checked,
        "documents": documents,
        "violations": violations,
        "warnings": warnings,
        "errors": errors,
    }
    return _write_report(root, report)


def _write_report(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = root / "evaluation" / "ragflow_runtime_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    return report
