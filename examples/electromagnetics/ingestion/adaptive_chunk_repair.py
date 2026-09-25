"""Safely retry RAGFlow chunking with progressively more formula-friendly profiles."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from chunk_quality import evaluate_positioned_chunks
from ingest_model import build_ingest_reviewer
from sync_library import (
    PaperIndex,
    RAGFlowClient,
    load_config,
    reserve_ragflow_upload_attempt,
    sha256_file,
    utc_now,
)


def _normalise_parser_config(config: Mapping[str, Any], token_num: int) -> dict[str, Any]:
    setting = config.get("ragflow", {}).get("original_pdf_parser", {})
    parser_config = dict(setting.get("parser_config", {})) if isinstance(setting, Mapping) else {}
    parser_config["chunk_token_num"] = int(token_num)
    parser_config["delimiter"] = "\n\n"
    parser_config["children_delimiter"] = "\n\n"
    return {"chunk_method": str(setting.get("chunk_method", "naive")), "parser_config": parser_config}


def _history_path(root: Path) -> Path:
    return root / "evaluation" / "adaptive_chunk_repair_history.json"


def _load_history(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_history(path: Path, history: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _profile_duplicate_ids(
    attempts: list[Mapping[str, Any]], profiles: tuple[int, ...], keep_id: str, current_id: str = ""
) -> list[str]:
    """Return only losing PDF-profile IDs for this same source record."""
    ids = {
        str(item.get("new_ragflow_document_id") or "")
        for item in attempts
        if str(item.get("new_ragflow_document_id") or "")
        and str(item.get("profile") or "").isdigit()
        and int(item["profile"]) in profiles
    }
    if current_id:
        ids.add(str(current_id))
    ids.discard(str(keep_id or ""))
    return sorted(item for item in ids if item)


def _profile_completion_state(
    prior_document_id: str,
    best_pdf_id: str,
    best_score: float,
    threshold: float,
    prior_pdf_document_id: str = "",
) -> dict[str, Any]:
    """Keep the previous serving source when a replacement misses the gate."""
    accepted = bool(best_pdf_id and best_score >= threshold)
    if accepted:
        return {
            "candidate_accepted": True,
            "serving_document_id": best_pdf_id,
            "pdf_document_id": best_pdf_id,
            "status": "published_ragflow",
            "retire_best_pdf": False,
        }
    prior_id = str(prior_document_id or "")
    prior_pdf_id = str(prior_pdf_document_id or "")
    serving_id = prior_id if prior_id and prior_id != best_pdf_id else ""
    # A prior PDF is the durable original asset. If the prior serving source is
    # Markdown, retain the best parsed PDF separately for visual page fallback.
    pdf_id = prior_pdf_id or (best_pdf_id if best_pdf_id else "")
    return {
        "candidate_accepted": False,
        "serving_document_id": serving_id,
        "pdf_document_id": pdf_id,
        "status": "published_ragflow" if serving_id else "chunk_quality_exhausted",
        "retire_best_pdf": bool(best_pdf_id and prior_pdf_id and prior_pdf_id != best_pdf_id),
    }


def _preparsed_markdown_update(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build the no-second-layout-parse config for a MinerU Markdown fallback."""
    setting = config.get("ragflow", {}).get("preparsed_markdown", {})
    if not isinstance(setting, Mapping) or not setting.get("enabled", False):
        return None
    parser_config = setting.get("parser_config", {})
    if not isinstance(parser_config, Mapping):
        raise ValueError("ragflow.preparsed_markdown.parser_config must be an object")
    parser_config = dict(parser_config)
    for key in ("delimiter", "children_delimiter"):
        if parser_config.get(key) == r"\n":
            parser_config[key] = "\n"
    parent_child = parser_config.get("parent_child")
    if isinstance(parent_child, Mapping):
        parent_child = dict(parent_child)
        if parent_child.get("children_delimiter") == r"\n":
            parent_child["children_delimiter"] = "\n"
        parser_config["parent_child"] = parent_child
    return {
        "chunk_method": str(setting.get("chunk_method", "naive")),
        "parser_config": parser_config,
    }


def run_adaptive_chunk_repair(config: Mapping[str, Any], limit: int = 1) -> list[dict[str, Any]]:
    """Try one unaccepted chunk-quality document without replacing its old copy prematurely."""
    storage = config["storage"]
    root = Path(str(storage["root"]))
    index = PaperIndex(Path(str(storage["index_db"])), Path(str(storage["summary_csv"])))
    history_file = _history_path(root)
    history = _load_history(history_file)
    profiles = (1024, 1536, 2048)
    rows = index.connection.execute(
        "SELECT * FROM papers WHERE status = 'chunk_needs_review' ORDER BY updated_at LIMIT ?",
        (max(1, int(limit)),),
    ).fetchall()
    reports: list[dict[str, Any]] = []
    try:
        for raw_row in rows:
            row = dict(raw_row)
            doc_id = str(row.get("doc_id") or "")
            attempts = history.setdefault(doc_id, [])
            # A prior worker version could record a profile as completed while
            # reusing another profile's RAGFlow document.  Choose the first
            # profile with a distinct completed document instead of using the
            # raw history length, so that a bad historical record cannot skip
            # the real 1536 attempt.
            distinct_profiles: set[int] = set()
            for previous in attempts:
                try:
                    previous_profile = int(previous.get("profile"))
                except (TypeError, ValueError):
                    continue
                # RAGFlow can fail before returning a usable document record
                # (for example when MinerU is unavailable), so a failed
                # attempt may have no new_ragflow_document_id.  It is still a
                # completed profile attempt and must not be retried forever.
                previous_status = str(previous.get("status") or "")
                if (
                    previous_profile in profiles
                    and previous.get("finished_at")
                    and previous_status not in {"started", "reparse_in_progress"}
                    and (
                        previous_status.startswith("repair_failed")
                        or previous_status.startswith("quality_failed")
                        or previous_status.startswith("published")
                    )
                ):
                    distinct_profiles.add(previous_profile)
            profile_index = next((index for index, profile in enumerate(profiles) if profile not in distinct_profiles), len(profiles))
            attempt_number = len(attempts)
            if profile_index >= len(profiles):
                # Do not create a second text-only retrieval copy after the
                # PDF profile sweep. Reuse the best existing PDF, preserve its
                # evidence trail, and leave a terminal review state if it does
                # not meet the quality gate.
                candidates = []
                for previous in attempts:
                    try:
                        previous_profile = int(previous.get("profile"))
                    except (TypeError, ValueError):
                        continue
                    quality = previous.get("quality")
                    previous_id = str(previous.get("new_ragflow_document_id") or "")
                    if previous_profile in profiles and previous_id and isinstance(quality, Mapping):
                        candidates.append((previous, quality))
                best_attempt, best_quality = max(
                    candidates,
                    key=lambda pair: (
                        float(pair[1].get("score", 0) or 0),
                        float(pair[1].get("boundary_integrity", 0) or 0),
                        float(pair[1].get("coverage", 0) or 0),
                    ),
                    default=(None, {}),
                )
                prior_document_id = str(row.get("ragflow_document_id") or "")
                best_id = str((best_attempt or {}).get("new_ragflow_document_id") or "")
                best_score = float(best_quality.get("score", 0) or 0)
                threshold = float(config.get("chunk_quality", {}).get("min_score", 0.98))
                prior_pdf_document_id = str(row.get("ragflow_pdf_document_id") or "")
                completion_state = _profile_completion_state(
                    prior_document_id,
                    best_id,
                    best_score,
                    threshold,
                    prior_pdf_document_id,
                )
                retired_duplicate_ids: list[str] = []
                duplicate_cleanup_errors: list[str] = []
                if best_id:
                    client = RAGFlowClient(
                        str(config["ragflow"]["base_url"]),
                        str(row.get("ragflow_dataset_id") or ""),
                        str(config["ragflow"]["api_key_env"]),
                        timeout=60,
                    )
                    best_document = client.document(best_id)
                    if str(best_document.get("name") or "").lower().endswith(".pdf"):
                        if prior_document_id and not prior_pdf_document_id:
                            prior_document = client.document(prior_document_id)
                            if str(prior_document.get("name") or "").lower().endswith(".pdf"):
                                prior_pdf_document_id = prior_document_id
                                completion_state = _profile_completion_state(
                                    prior_document_id,
                                    best_id,
                                    best_score,
                                    threshold,
                                    prior_pdf_document_id,
                                )
                        parser_config = best_document.get("parser_config") or {}
                        # Keep a parsed PDF available for visual/source use, but
                        # do not expose a profile below the chunk-quality gate
                        # to retrieval. Never replace the previous serving ID
                        # unless the candidate actually passes that gate.
                        client.update_document(best_id, {"enabled": int(completion_state["candidate_accepted"])})
                        row["ragflow_document_id"] = completion_state["serving_document_id"]
                        row["ragflow_pdf_document_id"] = completion_state["pdf_document_id"]
                        if completion_state["candidate_accepted"] or not prior_document_id:
                            row["ragflow_parser_config"] = json.dumps(
                                {
                                    "chunk_method": best_document.get("chunk_method") or "naive",
                                    "parser_config": parser_config,
                                },
                                ensure_ascii=False,
                            )
                        duplicate_ids = _profile_duplicate_ids(
                            attempts, profiles, best_id, prior_document_id
                        )
                        keep_ids = {
                            str(completion_state.get("serving_document_id") or ""),
                            str(completion_state.get("pdf_document_id") or ""),
                        }
                        duplicate_ids = [duplicate_id for duplicate_id in duplicate_ids if duplicate_id not in keep_ids]
                        if completion_state["retire_best_pdf"] and best_id not in duplicate_ids:
                            duplicate_ids.append(best_id)
                        for duplicate_id in duplicate_ids:
                            try:
                                duplicate = client.document(duplicate_id)
                                duplicate_name = str(duplicate.get("name") or "")
                                if duplicate_name.startswith(doc_id) and duplicate_name.lower().endswith(".pdf"):
                                    response = client.delete_documents([duplicate_id])
                                    if isinstance(response, Mapping) and response.get("code") not in (None, 0, "0"):
                                        raise RuntimeError(f"RAGFlow delete returned {response}")
                                    retired_duplicate_ids.append(duplicate_id)
                            except Exception as exc:
                                duplicate_cleanup_errors.append(f"{duplicate_id}: {type(exc).__name__}: {exc}")
                row["status"] = str(completion_state["status"])
                if completion_state["candidate_accepted"]:
                    row["last_error"] = ""
                elif completion_state["serving_document_id"]:
                    row["last_error"] = (
                        f"新 PDF 候选分块质量 {best_score:.4f} 低于门槛 {threshold:.4f}；"
                        f"继续使用已发布来源 {completion_state['serving_document_id']}，未切换到低质量版本。"
                    )
                else:
                    row["last_error"] = (
                        f"PDF 已保留用于溯源；最佳分块质量 {best_score:.4f} 低于门槛 {threshold:.4f}，"
                        "没有可用的已发布 serving 文档，需纳入交付缺口清单。"
                    )
                row["updated_at"] = utc_now()
                index.upsert_paper(row)
                report = {
                    "doc_id": doc_id,
                    "status": row["status"],
                    "profile": "profile_sweep_complete",
                    "best_ragflow_document_id": best_id,
                    "serving_document_id": completion_state["serving_document_id"],
                    "tracked_pdf_document_id": completion_state["pdf_document_id"],
                    "best_quality": dict(best_quality),
                    "threshold": threshold,
                    "candidate_retrieval_enabled": bool(completion_state["candidate_accepted"]),
                    "retired_duplicate_document_ids": retired_duplicate_ids,
                    "duplicate_cleanup_errors": duplicate_cleanup_errors,
                    "finished_at": utc_now(),
                }
                attempts.append(report)
                reports.append(report)
                continue

                # Legacy fallback retained temporarily for migration history;
                # the terminal branch above prevents further uploads.
                raw_markdown = Path(str(row.get("mineru_markdown_path") or ""))
                fallback_report: dict[str, Any] = {
                    "doc_id": doc_id,
                    "status": "preparsed_fallback_started",
                    "profile": "preparsed_markdown",
                    "old_ragflow_document_id": str(row.get("ragflow_document_id") or ""),
                    "started_at": utc_now(),
                }
                try:
                    dataset_id = str(row.get("ragflow_dataset_id") or "")
                    fallback_update = _preparsed_markdown_update(config)
                    if not raw_markdown.is_file() or not dataset_id or not fallback_update:
                        raise RuntimeError("MinerU Markdown 或 preparsed_markdown 配置不存在")
                    source = raw_markdown.read_text(encoding="utf-8", errors="replace")
                    client = RAGFlowClient(
                        str(config["ragflow"]["base_url"]),
                        dataset_id,
                        str(config["ragflow"]["api_key_env"]),
                        timeout=60,
                    )
                    display_name = f"{doc_id}--preparsed-{sha256_file(raw_markdown)[:12]}.md"
                    existing = client.find_active_document(display_name)
                    if existing:
                        fallback_id = str(existing["id"])
                    else:
                        reserve_ragflow_upload_attempt(index, row, is_original_pdf=False)
                        fallback_id = client.upload(raw_markdown, display_name=display_name)
                    client.update_document(fallback_id, fallback_update)
                    existing_run = str(existing.get("run") or "").upper() if existing else ""
                    document = existing if existing_run == "DONE" else None
                    if document is None:
                        client.trigger_parse(fallback_id)
                        document = client.wait_done(fallback_id, timeout=1800, interval=5)
                    chunks = [chunk for chunk in client.list_chunks(fallback_id) if str(chunk.get("content") or "").strip()]
                    quality = evaluate_positioned_chunks(source, chunks) if chunks else {
                        "chunk_count": 0,
                        "coverage": 0.0,
                        "boundary_integrity": 0.0,
                        "structure_integrity": 0.0,
                        "position_coverage": 0.0,
                        "score": 0.0,
                        "violations": ["missing_source_or_chunks"],
                    }
                    fallback_report.update({
                        "new_ragflow_document_id": fallback_id,
                        "run": document.get("run") if isinstance(document, Mapping) else "",
                        "chunk_count": len(chunks),
                        "quality": quality,
                    })
                    index.upsert_chunk_quality(doc_id, quality)
                    threshold = float(config.get("chunk_quality", {}).get("min_score", 0.98))
                    if len(chunks) == 0 or float(quality.get("score", 0.0)) < threshold:
                        client.update_document(fallback_id, {"enabled": 0})
                        row["last_error"] = f"preparsed Markdown quality {float(quality.get('score', 0.0)):.4f} < {threshold:.4f}"
                        fallback_report["status"] = "preparsed_fallback_quality_failed"
                    else:
                        # Chunk quality is necessary but not sufficient for
                        # admission.  Run the same model review gate before
                        # making the fallback the active retrieval document.
                        reviewer = build_ingest_reviewer(config)
                        if reviewer is None:
                            raise RuntimeError("review.enabled 必须为 true，且模型配置完整")
                        review = reviewer.review(
                            title=str(row.get("title") or ""),
                            authors=str(row.get("authors") or ""),
                            year=str(row.get("year") or ""),
                            doi=str(row.get("doi") or ""),
                            abstract=str(row.get("abstract") or ""),
                            markdown=source,
                            heuristic_summary={},
                        )
                        index.upsert_review(doc_id, review)
                        accepted = review.get("accepted") is True or str(review.get("accepted", "")).lower() == "true"
                        relevance = float(review.get("relevance_score", 0) or 0)
                        model_quality = float(review.get("quality_score", 0) or 0)
                        dataset_key = next(
                            (
                                key for key, value in config.get("ragflow", {}).get("dataset_ids", {}).items()
                                if str(value) == dataset_id
                            ),
                            "papers",
                        )
                        review_cfg = config.get("review", {})
                        thresholds = review_cfg.get("dataset_thresholds", {}).get(dataset_key, {}) if isinstance(review_cfg, Mapping) else {}
                        thresholds = thresholds if isinstance(thresholds, Mapping) else {}
                        min_relevance = float(thresholds.get("min_relevance_score", review_cfg.get("min_relevance_score", 0.65)))
                        min_quality = float(thresholds.get("min_quality_score", review_cfg.get("min_quality_score", 0.65)))
                        allow_warning = bool(thresholds.get("allow_evidence_warning", False))
                        warning_admission = review.get("validation") == "evidence_warning" and allow_warning
                        admission_ok = (
                            accepted
                            and relevance >= min_relevance
                            and model_quality >= min_quality
                            and not (review.get("validation") == "evidence_gate" and not warning_admission)
                        )
                        fallback_report["review"] = {
                            "accepted": accepted,
                            "relevance_score": relevance,
                            "quality_score": model_quality,
                            "validation": review.get("validation"),
                            "reason": str(review.get("reason") or ""),
                        }
                        if admission_ok:
                            row["ragflow_document_id"] = fallback_id
                            row["ragflow_parser_config"] = json.dumps(fallback_update, ensure_ascii=False)
                            row["status"] = "published_ragflow"
                            row["last_error"] = ""
                            fallback_report["status"] = "published_ragflow_preparsed_fallback"
                        else:
                            client.update_document(fallback_id, {"enabled": 0})
                            row["status"] = "review_rejected" if not accepted else "review_needs_review"
                            row["last_error"] = str(review.get("reason") or "preparsed fallback admission below gate")
                            fallback_report["status"] = "preparsed_fallback_admission_failed"
                    row["updated_at"] = utc_now()
                    index.upsert_paper(row)
                except Exception as exc:
                    fallback_report.update({
                        "status": "preparsed_fallback_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                fallback_report["finished_at"] = utc_now()
                attempts.append(fallback_report)
                reports.append(fallback_report)
                continue
            token_num = profiles[profile_index]
            raw_pdf = Path(str(row.get("raw_path") or ""))
            dataset_id = str(row.get("ragflow_dataset_id") or "")
            old_id = str(row.get("ragflow_document_id") or "")
            # Newly discovered PDFs can fail the local chunk gate before they
            # ever receive a RAGFlow document id.  They are still repairable:
            # upload the adaptive candidate directly and only disable an old
            # document when one actually exists.
            if not raw_pdf.is_file() or not dataset_id:
                reports.append({"doc_id": doc_id, "status": "missing_source_or_ragflow_id"})
                continue
            parser_update = _normalise_parser_config(config, token_num)
            display_name = f"{doc_id}--{sha256_file(raw_pdf)[:12]}.pdf"
            report: dict[str, Any] = {
                "doc_id": doc_id,
                "status": "started",
                "profile": token_num,
                "old_ragflow_document_id": old_id,
                "started_at": utc_now(),
            }
            client = RAGFlowClient(str(config["ragflow"]["base_url"]), dataset_id, str(config["ragflow"]["api_key_env"]), timeout=60)
            new_id = ""
            # Prefer the currently indexed PDF, then reuse a PDF emitted by
            # older repair runs, and finally reconcile against RAGFlow. Every
            # chunk profile below operates on this one document ID.
            if old_id:
                old_document = client.document(old_id)
                if str(old_document.get("name") or "").lower().endswith(".pdf"):
                    new_id = old_id
            if not new_id:
                for previous in reversed(attempts):
                    previous_id = str(previous.get("new_ragflow_document_id") or "")
                    try:
                        previous_profile = int(previous.get("profile"))
                    except (TypeError, ValueError):
                        continue
                    if previous_id and previous_profile in profiles:
                        previous_document = client.document(previous_id)
                        if str(previous_document.get("name") or "").lower().endswith(".pdf"):
                            new_id = previous_id
                            break
            if not new_id:
                existing = client.find_active_document(display_name)
                new_id = str(existing.get("id") or "") if existing else ""
            try:
                if not new_id:
                    reserve_ragflow_upload_attempt(index, row, is_original_pdf=True)
                    new_id = client.upload(raw_pdf, display_name=display_name)
                current_document = client.document(new_id)
                current_run = str(current_document.get("run") or "").upper()
                if current_run in {"RUNNING", "UNSTART"}:
                    reports.append({"doc_id": doc_id, "status": "reparse_in_progress", "ragflow_document_id": new_id})
                    continue
                current_config = current_document.get("parser_config") or {}
                current_token_num = int(current_config.get("chunk_token_num") or 0)
                if current_run != "DONE" or current_token_num != token_num:
                    client.update_document(new_id, {**parser_update, "enabled": 1})
                    client.trigger_parse(new_id)
                    document = client.wait_done(new_id, timeout=1800, interval=5)
                else:
                    document = current_document
                chunks = client.list_chunks(new_id)
                markdown_path = Path(str(row.get("mineru_markdown_path") or ""))
                source = markdown_path.read_text(encoding="utf-8", errors="replace") if markdown_path.is_file() else ""
                positioned_chunks = [chunk for chunk in chunks if str(chunk.get("content") or "").strip()]
                quality = evaluate_positioned_chunks(source, positioned_chunks) if source and positioned_chunks else {
                    "chunk_count": len(positioned_chunks), "coverage": 0.0, "boundary_integrity": 0.0,
                    "structure_integrity": 0.0, "score": 0.0, "violations": ["missing_source_or_chunks"],
                }
                report.update({
                    "new_ragflow_document_id": new_id,
                    "run": document.get("run"),
                    "chunk_count": len(positioned_chunks),
                    "quality": quality,
                })
                index.upsert_chunk_quality(doc_id, quality)
                threshold = float(config.get("chunk_quality", {}).get("min_score", 0.98))
                row["ragflow_document_id"] = new_id
                row["ragflow_parser_config"] = json.dumps(parser_update, ensure_ascii=False)
                row["updated_at"] = utc_now()
                if len(positioned_chunks) == 0 or float(quality.get("score", 0.0)) < threshold:
                    row["last_error"] = f"adaptive profile {token_num} quality {float(quality.get('score', 0.0)):.4f} < {threshold:.4f}"
                    index.upsert_paper(row)
                    report["status"] = "quality_failed_old_preserved"
                else:
                    if old_id and old_id != new_id:
                        client.update_document(old_id, {"enabled": 0})
                    row["status"] = "published_ragflow"
                    row["last_error"] = ""
                    index.upsert_paper(row)
                    report["status"] = "published_ragflow"
            except Exception as exc:
                report.update({"status": "repair_failed_old_preserved", "error": f"{type(exc).__name__}: {exc}"})
                if new_id:
                    row["ragflow_document_id"] = new_id
                    row["last_error"] = str(exc)
                    row["updated_at"] = utc_now()
                    index.upsert_paper(row)
            report["finished_at"] = utc_now()
            attempts.append(report)
            reports.append(report)
        _save_history(history_file, history)
        index.export_csv()
        return reports
    finally:
        index.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="adaptive RAGFlow chunk repair")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_adaptive_chunk_repair(load_config(args.config), args.limit), ensure_ascii=False, indent=2))
