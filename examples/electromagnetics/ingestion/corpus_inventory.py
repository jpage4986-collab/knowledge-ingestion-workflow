"""Build a read-only, actionable inventory of the local literature corpus."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def corpus_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    storage = config["storage"]
    db_path = Path(storage["index_db"])
    root = Path(storage["root"])
    configured = config.get("ragflow", {}).get("dataset_ids", {}) or {}
    reverse_datasets = {str(value): str(key) for key, value in configured.items() if str(value).strip()}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not db_path.is_file():
        errors.append(f"missing_index: {db_path}")
    else:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(db_path)
            with connection:
                connection.row_factory = sqlite3.Row
                for row in connection.execute("SELECT * FROM papers ORDER BY title, doc_id"):
                    item = dict(row)
                    item["dataset_key"] = reverse_datasets.get(str(item.get("ragflow_dataset_id") or ""), "")
                    item["raw_exists"] = Path(str(item.get("raw_path") or "")).is_file()
                    item["mineru_markdown_exists"] = Path(str(item.get("mineru_markdown_path") or "")).is_file()
                    review = connection.execute(
                        "SELECT accepted, relevance_score, quality_score, reason FROM review_results WHERE doc_id = ?",
                        (item["doc_id"],),
                    ).fetchone()
                    item["review"] = dict(review) if review else None
                    quality = connection.execute(
                        "SELECT score, coverage, structure_integrity, violations FROM chunk_quality WHERE doc_id = ?",
                        (item["doc_id"],),
                    ).fetchone()
                    item["chunk_quality"] = dict(quality) if quality else None
                    summary = connection.execute(
                        "SELECT quality_score, extraction_status, evidence_ref FROM simulation_summaries WHERE doc_id = ?",
                        (item["doc_id"],),
                    ).fetchone()
                    item["summary_quality_score"] = float(summary[0]) if summary and summary[0] is not None else None
                    item["extraction_status"] = str(summary[1] or "") if summary else ""
                    item["evidence_manifest_path"] = ""
                    if summary and summary[2]:
                        try:
                            evidence = json.loads(str(summary[2]))
                            if isinstance(evidence, dict):
                                item["evidence_manifest_path"] = str(evidence.get("manifest_path") or "")
                        except (TypeError, ValueError, json.JSONDecodeError):
                            pass
                    packet_path = root / "quarantine" / str(item["doc_id"]) / "review_packet.json"
                    item["review_packet_path"] = str(packet_path)
                    item["review_packet_exists"] = packet_path.is_file()
                    # Keep the inventory compact and never expose full source
                    # text or secrets in this quick-lookup artifact.
                    rows.append({
                        key: item.get(key)
                        for key in (
                            "doc_id", "title", "authors", "year", "doi", "status", "dataset_key",
                            "abstract", "landing_url", "pdf_url", "source_provider",
                            "ragflow_dataset_id", "ragflow_document_id", "ragflow_parser_config", "mineru_status", "raw_path",
                            "mineru_markdown_path", "raw_exists", "mineru_markdown_exists", "last_error",
                            "selection_score", "summary_quality_score", "extraction_status", "evidence_manifest_path",
                            "review_packet_path", "review_packet_exists", "review", "chunk_quality",
                        )
                    })
        except (OSError, sqlite3.Error) as exc:
            errors.append(f"index_read_error: {type(exc).__name__}: {exc}")
        finally:
            if connection is not None:
                connection.close()

    status_counts = dict(Counter(str(row.get("status") or "unknown") for row in rows))
    dataset_counts = dict(Counter(str(row.get("dataset_key") or row.get("ragflow_dataset_id") or "unassigned") for row in rows))
    action_counts: Counter[str] = Counter()
    for row in rows:
        status = str(row.get("status") or "")
        action = {
            "review_needs_review": "retry_after_model_key",
            "ragflow_failed": "retry_after_ragflow_key_or_service",
            "duplicate_existing": "keep_quarantined",
            "review_rejected": "manual_recheck_or_keep_rejected",
            "published_ragflow": "complete",
            "uploaded_to_ragflow": "legacy_upload_needs_admission_audit",
        }.get(status, "inspect")
        row["next_action"] = action
        action_counts[action] += 1
    report = {
        "status": "needs_attention" if errors or any(row.get("next_action") not in {"complete", "keep_quarantined"} for row in rows) else "pass",
        "index_path": str(db_path),
        "literature_root": str(root),
        "paper_count": len(rows),
        "canonical_paper_count": sum(str(row.get("status") or "") != "duplicate_existing" for row in rows),
        "distinct_title_count": len({str(row.get("title") or "").casefold() for row in rows if row.get("title")}),
        "status_counts": status_counts,
        "dataset_counts": dataset_counts,
        "next_action_counts": dict(action_counts),
        "errors": errors,
        "papers": rows,
        "created_at": _now(),
    }
    output = root / "evaluation" / "corpus_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(output)
    return report


if __name__ == "__main__":
    raise SystemExit("请通过 sync_library.py corpus-inventory 调用")
