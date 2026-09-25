"""Read-only audit for the local literature index and evidence invariants."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit_index(config: Mapping[str, Any]) -> dict[str, Any]:
    storage = config["storage"]
    db_path = Path(storage["index_db"])
    required_drive = str(storage.get("required_drive", "G:")).rstrip("\\/").upper()
    configured_dataset_ids = {
        str(value) for value in (config.get("ragflow", {}).get("dataset_ids", {}) or {}).values() if str(value).strip()
    }
    review_cfg = config.get("review", {})
    quality_cfg = config.get("quality", {})
    review_min_relevance = float(review_cfg.get("min_relevance_score", 0.65))
    review_min_quality = float(review_cfg.get("min_quality_score", 0.65))
    dataset_thresholds = review_cfg.get("dataset_thresholds", {}) if isinstance(review_cfg, Mapping) else {}
    dataset_ids = config.get("ragflow", {}).get("dataset_ids", {}) if isinstance(config.get("ragflow", {}), Mapping) else {}

    def admission_thresholds(dataset_id: str) -> tuple[float, float]:
        dataset_key = next((key for key, value in dataset_ids.items() if str(value) == dataset_id), "")
        policy = dataset_thresholds.get(dataset_key, {}) if isinstance(dataset_thresholds, Mapping) else {}
        policy = policy if isinstance(policy, Mapping) else {}
        return (
            float(policy.get("min_relevance_score", review_min_relevance)),
            float(policy.get("min_quality_score", review_min_quality)),
        )
    chunk_min_score = float(config.get("chunk_quality", {}).get("min_score", quality_cfg.get("structural_min", 0.98)))
    violations: list[dict[str, str]] = []
    rows: list[sqlite3.Row] = []
    counts: dict[str, int] = {}
    processed_statuses = {
        "parsed", "review_needs_review", "review_rejected", "published_ragflow",
        "uploaded_to_ragflow", "ragflow_failed", "chunk_needs_review",
    }
    review_statuses = {"review_needs_review", "review_rejected", "published_ragflow", "uploaded_to_ragflow"}
    if not db_path.is_file():
        violations.append({"type": "missing_index", "path": str(db_path)})
    else:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            paper_columns = {str(item[1]) for item in connection.execute("PRAGMA table_info(papers)").fetchall()}
            has_parser_config_column = "ragflow_parser_config" in paper_columns
            for table in ("papers", "simulation_summaries", "chunk_quality", "review_results", "discovery_runs"):
                try:
                    counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except sqlite3.OperationalError:
                    counts[table] = 0
            try:
                rows = connection.execute("SELECT * FROM papers ORDER BY doc_id").fetchall()
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
                doc_id = str(row["doc_id"])
                status = str(row["status"] or "")
                raw_path = Path(str(row["raw_path"] or ""))
                if raw_path and raw_path.drive.upper() != required_drive:
                    violations.append({"doc_id": doc_id, "type": "raw_not_on_required_drive", "path": str(raw_path)})
                if status in processed_statuses and not raw_path.is_file():
                    violations.append({"doc_id": doc_id, "type": "missing_raw_pdf", "path": str(raw_path)})
                dataset_id = str(row["ragflow_dataset_id"] or "")
                if dataset_id and configured_dataset_ids and dataset_id not in configured_dataset_ids:
                    violations.append({"doc_id": doc_id, "type": "dataset_id_not_in_config", "dataset_id": dataset_id})
                if status in processed_statuses:
                    markdown_path = Path(str(row["mineru_markdown_path"] or ""))
                    if not markdown_path.is_file():
                        violations.append({"doc_id": doc_id, "type": "missing_mineru_markdown", "path": str(markdown_path)})
                    summary = connection.execute("SELECT evidence_ref FROM simulation_summaries WHERE doc_id = ?", (doc_id,)).fetchone()
                    if summary is None:
                        violations.append({"doc_id": doc_id, "type": "missing_simulation_summary"})
                    else:
                        try:
                            evidence = json.loads(str(summary[0] or ""))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            evidence = {}
                        manifest = Path(str(evidence.get("manifest_path") or "")) if isinstance(evidence, dict) else Path()
                        if not manifest.is_file():
                            violations.append({"doc_id": doc_id, "type": "missing_evidence_manifest", "path": str(manifest)})
                        elif isinstance(evidence, dict):
                            for evidence_key in ("original_pdf", "content_list_path"):
                                evidence_path = Path(str(evidence.get(evidence_key) or ""))
                                if evidence_key == "content_list_path" and not evidence_path:
                                    continue
                                if evidence_key == "original_pdf" and not evidence_path.is_file():
                                    violations.append({"doc_id": doc_id, "type": "missing_evidence_file", "path": str(evidence_path)})
                                if evidence_key == "content_list_path" and evidence_path and not evidence_path.is_file():
                                    violations.append({"doc_id": doc_id, "type": "missing_mineru_content_list", "path": str(evidence_path)})
                    quality = connection.execute("SELECT 1 FROM chunk_quality WHERE doc_id = ?", (doc_id,)).fetchone()
                    if quality is None:
                        violations.append({"doc_id": doc_id, "type": "missing_chunk_quality"})
                if status in review_statuses:
                    review = connection.execute("SELECT 1 FROM review_results WHERE doc_id = ?", (doc_id,)).fetchone()
                    if review is None:
                        violations.append({"doc_id": doc_id, "type": "missing_review_result"})
                # Review packets are quarantine artifacts. Published records
                # retain their admission review in SQLite and their PDF,
                # MinerU evidence, and RAGFlow document; they are not expected
                # to have a file under the quarantine directory.
                if status in {"review_needs_review", "review_rejected", "uploaded_to_ragflow", "duplicate_existing", "ragflow_failed", "chunk_needs_review"}:
                    packet = Path(storage["root"]) / "quarantine" / doc_id / "review_packet.json"
                    if not packet.is_file():
                        violations.append({"doc_id": doc_id, "type": "missing_review_packet", "path": str(packet)})
                if status == "published_ragflow":
                    admission_min_relevance, admission_min_quality = admission_thresholds(dataset_id)
                    review = connection.execute(
                        "SELECT accepted, relevance_score, quality_score FROM review_results WHERE doc_id = ?",
                        (doc_id,),
                    ).fetchone()
                    if review is None:
                        violations.append({"doc_id": doc_id, "type": "published_without_admission_review"})
                    else:
                        if int(review[0] or 0) != 1:
                            violations.append({"doc_id": doc_id, "type": "published_without_accepted_review"})
                        if float(review[1] or 0) < admission_min_relevance:
                            violations.append({"doc_id": doc_id, "type": "published_below_relevance_gate"})
                        if float(review[2] or 0) < admission_min_quality:
                            violations.append({"doc_id": doc_id, "type": "published_below_model_quality_gate"})
                    chunk = connection.execute("SELECT score FROM chunk_quality WHERE doc_id = ?", (doc_id,)).fetchone()
                    if chunk is None or float(chunk[0] or 0) < chunk_min_score:
                        violations.append({"doc_id": doc_id, "type": "published_below_chunk_gate"})
                    if not has_parser_config_column or not str(row["ragflow_parser_config"] or ""):
                        violations.append({"doc_id": doc_id, "type": "published_without_parser_config_audit"})
                if status == "uploaded_to_ragflow":
                    violations.append({"doc_id": doc_id, "type": "legacy_upload_without_verified_admission"})
                if status == "published_ragflow" and not str(row["ragflow_document_id"] or ""):
                    violations.append({"doc_id": doc_id, "type": "published_without_ragflow_document"})
        # sqlite3's context manager commits/rolls back but does not close the
        # connection; close it explicitly so Windows can remove the DB.
        connection.close()
    report = {
        "status": "pass" if not violations else "needs_attention",
        "index_path": str(db_path),
        "required_drive": required_drive,
        "counts": counts,
        "paper_count": len(rows),
        "violations": violations,
        "created_at": _now(),
    }
    output = Path(storage["root"]) / "evaluation" / "index_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(output)
    return report


if __name__ == "__main__":
    raise SystemExit("请通过 sync_library.py audit-index 调用")
