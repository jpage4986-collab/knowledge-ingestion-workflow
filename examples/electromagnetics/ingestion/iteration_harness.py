"""Repeatable quality loop for the electromagnetic literature ingestion workflow.

The loop is intentionally local and non-destructive: it evaluates existing
MinerU Markdown, tests chunk round-tripping, scores evidence density, and
records adaptive parameter choices. Uploading remains the responsibility of
the admission-gated sync workflow, so 50 iterations cannot pollute RAGFlow.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from chunk_quality import evaluate_chunks, split_markdown


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_score(markdown: str) -> float:
    """Estimate whether a parsed document contains usable simulation evidence.

    This is a gate signal, not a claim of semantic truth. The model review and
    human benchmark remain the final authority for admission.
    """
    text = markdown.lower()
    has_equation = "$$" in markdown or bool(re.search(r"(?<!\\)\\(?:\(|\[)", markdown))
    has_table = "|" in markdown or bool(re.search(r"<table\b", text))
    has_numeric_units = bool(re.search(
        r"\b\d+(?:\.\d+)?\s*(?:\[[^\]\n]{1,20}\]|(?:nm|µm|μm|um|mm|m|fs|ps|ns|ghz|mhz|thz|hz|s|v/m|db)\b)",
        text,
        flags=re.I,
    ))
    has_simulation_terms = sum(
        term in text for term in ("fdtd", "mesh", "pml", "boundary", "source", "monitor", "solver")
    ) >= 2
    has_procedure_or_results = bool(re.search(
        r"\b(?:results?|discussion|computed|measured|procedure|settings?|compute|select|click)\b|\n\s*\d+\s+(?:in|select|click|from|under)\b",
        text,
        flags=re.I,
    ))
    has_traceability = bool(re.search(r"page|第\s*\d+\s*页|p\.\s*\d+|references|bibliography|参考文献", text))
    # A procedural vendor case can be highly useful without a bibliography or
    # explicit page labels. Keep traceability as a half-weight bonus while
    # requiring structure, numerical evidence, simulation context, and either
    # equations/tables/images or procedure/results.
    weighted_signals = {
        "document_structure": bool(re.search(r"^#{1,6}\s+", markdown, re.M)),
        "structured_or_visual_evidence": has_table or has_equation or bool(re.search(r"!\[[^\]]*\]\([^)]*\)", markdown)),
        "numeric_parameter_evidence": has_numeric_units or has_equation,
        "simulation_context": has_simulation_terms,
        "procedure_or_results": has_procedure_or_results,
        "traceability": has_traceability,
    }
    weights = {
        "document_structure": 1.0,
        "structured_or_visual_evidence": 1.0,
        "numeric_parameter_evidence": 1.0,
        "simulation_context": 1.0,
        "procedure_or_results": 1.0,
        "traceability": 0.5,
    }
    return round(sum(weights[key] for key, present in weighted_signals.items() if present) / sum(weights.values()), 4)


def discover_markdown(
    root: Path,
    explicit: Iterable[str] | None = None,
    excluded_paths: set[str] | None = None,
) -> list[Path]:
    excluded = {str(Path(item).resolve()).casefold() for item in (excluded_paths or set())}
    paths = [Path(item) for item in explicit or []]
    if not paths:
        # The configured root may contain per-source namespaces such as
        # comsol_official/published and comsol_official/mineru.
        paths = [path for path in root.rglob("*.md") if path.name.lower() != "readme.md"] if root.exists() else []
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file() and str(path.resolve()).casefold() not in excluded:
            unique.setdefault(_sha256(path), path)
    return sorted(unique.values(), key=lambda value: str(value).lower())


def duplicate_markdown_paths(index_db: Path) -> set[str]:
    """Read duplicate paths from the authoritative index without changing it."""
    if not index_db.is_file():
        return set()
    try:
        connection = sqlite3.connect(index_db)
        rows = connection.execute(
            "SELECT mineru_markdown_path FROM papers WHERE status = 'duplicate_existing'"
        ).fetchall()
        connection.close()
    except (OSError, sqlite3.Error):
        return set()
    return {str(row[0]) for row in rows if row and row[0]}


def evaluate_document(path: Path, max_chars: int, evidence_min: float = 0.80, structural_min: float = 0.98) -> dict[str, Any]:
    markdown = path.read_text(encoding="utf-8", errors="replace")
    chunks = split_markdown(markdown, max_chars=max_chars)
    report = evaluate_chunks(markdown, chunks).as_dict()
    evidence = evidence_score(markdown)
    total = round(0.70 * float(report["score"]) + 0.30 * evidence, 4)
    checks = {
        "round_trip": float(report["coverage"]) >= 1.0,
        "non_empty": all(bool(chunk.strip()) for chunk in chunks),
        "structure": float(report["structure_integrity"]) >= structural_min,
        "chunk_gate": float(report["score"]) >= structural_min,
        "evidence_gate": evidence >= evidence_min,
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "max_chars": max_chars,
        "chunk_report": report,
        "evidence_score": evidence,
        "total_score": total,
        "tests_passed": all(checks.values()),
        "checks": checks,
    }


def run_iterations(config: Mapping[str, Any], iterations: int = 50, document_paths: Iterable[str] | None = None) -> dict[str, Any]:
    if iterations < 50:
        raise ValueError("为满足稳定性目标，iterations 至少为 50")
    storage = config["storage"]
    root = Path(storage["root"])
    documents = discover_markdown(root, document_paths, duplicate_markdown_paths(Path(storage.get("index_db", ""))))
    if not documents:
        raise RuntimeError(f"没有找到可评估的 MinerU Markdown：{root}")

    sizes = [512, 700, 900, 1200, 1600]
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_by_document: dict[str, dict[str, Any]] = {}
    quality = config.get("quality", {})
    evidence_min = float(quality.get("evidence_min", 0.80)) if isinstance(quality, Mapping) else 0.80
    structural_min = float(quality.get("structural_min", 0.98)) if isinstance(quality, Mapping) else 0.98
    exploration_rounds = len(documents) * len(sizes)
    for index in range(iterations):
        path = documents[index % len(documents)]
        document_digest = _sha256(path)
        # Every document gets every supported size once. This avoids using a
        # global winner for documents with very different MinerU layouts.
        if index < exploration_rounds:
            max_chars = sizes[(index // len(documents)) % len(sizes)]
        else:
            document_best = best_by_document.get(document_digest)
            anchor = int((document_best or best or {}).get("max_chars", sizes[-1]))
            max_chars = sizes[(sizes.index(anchor) + (index % 3) - 1) % len(sizes)] if anchor in sizes else sizes[index % len(sizes)]
        result = evaluate_document(path, max_chars, evidence_min=evidence_min, structural_min=structural_min)
        result.update({"iteration": index + 1, "timestamp": _now()})
        history.append(result)
        current_key = result["sha256"]
        if result["tests_passed"] and (best is None or result["total_score"] > best["total_score"]):
            best = result
        if result["tests_passed"] and (current_key not in best_by_document or result["total_score"] > best_by_document[current_key]["total_score"]):
            best_by_document[current_key] = result

    output_root = root / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    history_path = output_root / "iteration_history.jsonl"
    with history_path.open("w", encoding="utf-8") as stream:
        for item in history:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "goal": "50+ adaptive iterations with structural and evidence gates",
        "iterations": iterations,
        "documents_evaluated": len(documents),
        "passed_iterations": sum(bool(item["tests_passed"]) for item in history),
        "best_total_score": best["total_score"] if best else 0.0,
        "best_chunk_size": best["max_chars"] if best else None,
        "best_by_document": list(best_by_document.values()),
        "history_path": str(history_path),
        "semantic_accuracy_status": "requires labelled retrieval benchmark and model/human review",
        "created_at": _now(),
    }
    (output_root / "iteration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
