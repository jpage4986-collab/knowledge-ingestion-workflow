"""Small labelled benchmark for real RAGFlow retrieval regression checks."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from sync_library import RAGFlowClient


def build_client(rag: Mapping[str, Any], dataset_id: str) -> RAGFlowClient:
    """Construct a client with an explicit benchmark request timeout."""
    timeout_value = rag.get("retrieval_timeout_seconds")
    if timeout_value is None:
        return RAGFlowClient(str(rag["base_url"]), dataset_id, str(rag["api_key_env"]))
    return RAGFlowClient(
        str(rag["base_url"]),
        dataset_id,
        str(rag["api_key_env"]),
        timeout=float(timeout_value),
    )


def resolve_source_document_ids(config: Mapping[str, Any], source_doc_ids: list[str]) -> list[str]:
    """Resolve local paper IDs to admitted RAGFlow document IDs, if available."""
    if not source_doc_ids:
        return []
    raw_index_path = str(config.get("storage", {}).get("index_db", "")).strip()
    index_path = Path(raw_index_path) if raw_index_path else Path(str(config.get("storage", {}).get("root", ""))) / "index.sqlite"
    if not index_path.exists():
        return []
    connection = None
    try:
        connection = sqlite3.connect(index_path)
        placeholders = ",".join("?" for _ in source_doc_ids)
        rows = connection.execute(
            f"SELECT ragflow_document_id FROM papers WHERE doc_id IN ({placeholders})",
            source_doc_ids,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        if connection is not None:
            connection.close()
    return sorted({str(row[0]) for row in rows if row[0]})


def load_document_aliases(config: Mapping[str, Any]) -> dict[str, str]:
    """Load local old-to-new RAGFlow IDs created by verified migrations."""
    root = Path(str(config.get("storage", {}).get("root", "")))
    path = root / "evaluation" / "ragflow_document_aliases.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in payload.items() if str(key) and str(value)} if isinstance(payload, Mapping) else {}


def normalize_retrieval_text(value: Any) -> str:
    """Normalize common PDF/RAGFlow punctuation and unit formatting."""
    text = str(value or "").lower()
    text = re.sub(r"\[\s*([a-zµμ]+)\s*\]", r" \1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def required_term_present(content: str, term: str) -> bool:
    normalized_content = normalize_retrieval_text(content)
    normalized_term = normalize_retrieval_text(term)
    if normalized_term in normalized_content:
        return True
    compact_content = re.sub(r"[^a-z0-9\u3400-\u9fff.]+", "", normalized_content)
    compact_term = re.sub(r"[^a-z0-9\u3400-\u9fff.]+", "", normalized_term)
    return bool(compact_term) and compact_term in compact_content


def text_fragmentation_diagnostic(content: str) -> dict[str, Any]:
    """Measure character-level word breaks commonly introduced by re-parsing."""
    raw = str(content or "")
    alpha_tokens = re.findall(r"[A-Za-z]{2,}", raw)
    # Only a newline between alphabetic fragments is suspicious. Counting all
    # whitespace falsely labels ordinary word pairs such as "simulation
    # summary" as parser damage.
    broken_spans = re.findall(r"(?<![.!?:;,)\\]])[A-Za-z]{1,12}\n\s?[A-Za-z]{1,12}(?![A-Za-z])", raw)
    # Keep only short right fragments; ordinary line wrapping between complete
    # words such as ``simulation\\nsummary`` is not parser damage.
    broken_spans = [span for span in broken_spans if re.search(r"\\n\\s?[A-Za-z]{1,4}$", span)]
    # Recompute using physical lines to avoid regex escaping ambiguity. A
    # short next-line token is unlikely to be an intentional line wrap.
    broken_spans = []
    lines = raw.splitlines()
    for previous, current in zip(lines, lines[1:]):
        next_token = current.strip().split(maxsplit=1)[0] if current.strip() else ""
        if (previous and previous[-1].isalpha() and next_token and next_token[0].isalpha()
                and len(next_token.rstrip(".,;:!?)]}")) <= 4):
            broken_spans.append(previous[-12:] + chr(10) + next_token[:12])
    marker_count = len(broken_spans)
    token_count = max(1, len(alpha_tokens))
    rate = round(marker_count / token_count, 4)
    return {
        "alpha_token_count": len(alpha_tokens),
        "broken_word_marker_count": marker_count,
        "fragmentation_rate": rate,
        "suspected_reparse_damage": rate >= 0.08,
    }


def evaluate_case(
    client: RAGFlowClient,
    case: Mapping[str, Any],
    use_required_terms_fallback: bool = False,
    retry_on_timeout: bool = False,
) -> dict[str, Any]:
    query = str(case["question"])
    # Use the answer assistant's deterministic English query when one is
    # supplied. This benchmarks the real serving path rather than raw Chinese
    # retrieval against an English-language corpus.
    retrieval_query = str(case.get("retrieval_query") or "").strip()
    initial_query = retrieval_query or query
    query_attempts = [initial_query]
    query_attempts.extend(str(value) for value in case.get("query_variants", []) if str(value).strip())
    if retrieval_query and query not in query_attempts:
        query_attempts.append(query)
    if use_required_terms_fallback:
        terms_query = " ".join(str(value).strip() for value in case.get("required_terms", []) if str(value).strip())
        if terms_query and terms_query not in query_attempts:
            query_attempts.append(terms_query)
    required_terms = [str(term) for term in case.get("required_terms", [])]
    gold_ids = {str(value) for value in case.get("gold_document_ids", [])}
    chunks: list[dict[str, Any]] = []
    best_score = -1.0
    attempt_counts: list[dict[str, Any]] = []
    effective_query = initial_query
    timeout_retries = 0
    for attempt_index, attempt in enumerate(query_attempts):
        attempt_error = ""
        while True:
            try:
                candidate_chunks = client.retrieve(
                    attempt,
                    top_k=int(case.get("top_k", 10)),
                    similarity_threshold=float(case.get("similarity_threshold", 0.2)),
                    vector_similarity_weight=float(case.get("vector_similarity_weight", 0.5)),
                    keyword=bool(case.get("keyword", True)),
                    doc_ids=[str(value) for value in case.get("doc_ids", [])] or None,
                )
                break
            except TimeoutError:
                if not retry_on_timeout or timeout_retries >= 1:
                    if attempt_index + 1 >= len(query_attempts):
                        raise
                    attempt_error = "TimeoutError: timed out after retry"
                    candidate_chunks = []
                    break
                timeout_retries += 1
        candidate_content = "\n".join(str(chunk.get("content", "")) for chunk in candidate_chunks)
        candidate_hits = [term for term in required_terms if required_term_present(candidate_content, term)]
        candidate_recall = len(candidate_hits) / max(1, len(required_terms))
        candidate_ids = {str(chunk.get("document_id") or chunk.get("doc_id") or "") for chunk in candidate_chunks}
        candidate_gold_hit = bool(gold_ids & candidate_ids) if gold_ids else None
        candidate_score = (
            round(0.7 * candidate_recall + 0.3 * float(candidate_gold_hit), 4)
            if candidate_gold_hit is not None else candidate_recall
        )
        attempt_result = {
            "query": attempt,
            "chunk_count": len(candidate_chunks),
            "term_recall": round(candidate_recall, 4),
            "gold_hit": candidate_gold_hit,
        }
        if attempt_error:
            attempt_result["error"] = attempt_error
        attempt_counts.append(attempt_result)
        if candidate_chunks and candidate_score > best_score:
            chunks = candidate_chunks
            effective_query = attempt
            best_score = candidate_score
        # Do not stop at a merely non-empty result. Keep testing query variants
        # until the required evidence and (when available) the gold source hit.
        if candidate_chunks and candidate_score >= 1.0:
            break
    content = "\n".join(str(chunk.get("content", "")) for chunk in chunks)
    text_diagnostic = text_fragmentation_diagnostic(content)
    required_term_hits = [term for term in required_terms if required_term_present(content, term)]
    term_recall = len(required_term_hits) / max(1, len(required_terms))
    returned_ids = {str(chunk.get("document_id") or chunk.get("doc_id") or "") for chunk in chunks}
    gold_hit = bool(gold_ids & returned_ids) if gold_ids else None
    score = round(0.7 * term_recall + 0.3 * float(bool(gold_hit)) if gold_hit is not None else term_recall, 4)
    document_counts: dict[str, dict[str, int]] = {}
    gold_chunk_count = 0
    for rank, chunk in enumerate(chunks, 1):
        document_id = str(chunk.get("document_id") or chunk.get("doc_id") or "").strip()
        if not document_id:
            continue
        item = document_counts.setdefault(document_id, {"chunk_count": 0, "first_rank": rank})
        item["chunk_count"] += 1
        if document_id in gold_ids:
            gold_chunk_count += 1
    document_chunk_distribution = [
        {"document_id": document_id, **counts}
        for document_id, counts in sorted(
            document_counts.items(), key=lambda pair: (pair[1]["first_rank"], pair[0])
        )
    ]
    gold_document_rank = next(
        (
            rank
            for rank, item in enumerate(document_chunk_distribution, 1)
            if item["document_id"] in gold_ids
        ),
        None,
    )
    gold_chunk_share_at_k = (
        round(gold_chunk_count / len(chunks), 4) if gold_ids and chunks else
        (0.0 if gold_ids else None)
    )
    has_cjk = bool(re.search(r"[\u3400-\u9fff]", query))
    return {
        "id": str(case.get("id", case["question"])),
        "question": query,
        "query_language_hint": "cjk" if has_cjk else "non_cjk",
        "native_chunk_count": (
            next((item["chunk_count"] for item in attempt_counts if item["query"] == query), 0)
            if not retrieval_query else None
        ),
        "initial_query_chunk_count": attempt_counts[0]["chunk_count"] if attempt_counts else 0,
        "retrieval_query_override_used": bool(retrieval_query),
        "effective_query": effective_query,
        "query_fallback_used": effective_query != query,
        "query_attempts": attempt_counts,
        "timeout_retries": timeout_retries,
        "chunk_count": len(chunks),
        "term_recall": round(term_recall, 4),
        "required_terms_found": required_term_hits,
        "missing_required_terms": [term for term in required_terms if term not in required_term_hits],
        "gold_hit": gold_hit,
        "gold_document_rank": gold_document_rank,
        "gold_chunk_count": gold_chunk_count if gold_ids else None,
        "gold_chunk_share_at_k": gold_chunk_share_at_k,
        "unique_document_count": len(document_counts),
        "document_chunk_distribution": document_chunk_distribution,
        "score": score,
        "returned_document_ids": sorted(returned_ids),
        "text_diagnostic": text_diagnostic,
    }


def run_benchmark(
    config: Mapping[str, Any],
    benchmark_path: Path,
    dataset_id: str | None = None,
    dataset_key: str | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark must be a non-empty JSON list")
    if dataset_key:
        cases = [case for case in cases if str(case.get("dataset_key", "")) == dataset_key]
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("benchmark filter selected no cases")
    rag = config["ragflow"]
    target = dataset_id or str(rag.get("dataset_ids", {}).get("cases") or rag.get("dataset_id", ""))
    client = build_client(rag, target)
    clients: dict[str, RAGFlowClient] = {target: client}
    document_aliases = load_document_aliases(config)
    use_required_terms_fallback = bool(rag.get("retrieval_query_fallback", False))
    retry_on_timeout = bool(rag.get("retrieval_retry_on_timeout", True))
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for case in cases:
        source_doc_ids = [str(value) for value in case.get("source_doc_ids", [])]
        resolved_source_ids = resolve_source_document_ids(config, source_doc_ids)
        effective_case = dict(case)
        explicit_gold_ids = [str(value) for value in effective_case.get("gold_document_ids", [])]
        aliased_gold_ids = [document_aliases[value] for value in explicit_gold_ids if value in document_aliases]
        if aliased_gold_ids:
            effective_case["gold_document_ids"] = sorted(set(explicit_gold_ids + aliased_gold_ids))
        if not effective_case.get("gold_document_ids") and resolved_source_ids:
            effective_case["gold_document_ids"] = resolved_source_ids
        case_target = str(
            case.get("dataset_id")
            or rag.get("dataset_ids", {}).get(str(case.get("dataset_key", "")), "")
            or target
        )
        if not case_target:
            error = "未配置该 benchmark case 的 dataset_id 或 dataset_key"
            errors.append(error)
            results.append({
                "id": str(case.get("id", case.get("question", "unknown"))),
                "question": str(case.get("question", "")),
                "status": "failed",
                "error": error,
            })
            continue
        if case_target not in clients:
            clients[case_target] = build_client(rag, case_target)
        try:
            evaluated = evaluate_case(
                clients[case_target],
                effective_case,
                use_required_terms_fallback=use_required_terms_fallback,
                retry_on_timeout=retry_on_timeout,
            )
            evaluated["dataset_id"] = case_target
            evaluated["source_doc_ids"] = source_doc_ids
            evaluated["resolved_source_document_ids"] = resolved_source_ids
            evaluated["gold_document_aliases_applied"] = {
                old_id: document_aliases[old_id]
                for old_id in explicit_gold_ids
                if old_id in document_aliases
            }
            evaluated["source_resolution_status"] = (
                "resolved" if resolved_source_ids else ("pending_upload" if source_doc_ids else "not_applicable")
            )
            results.append(evaluated)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            results.append({
                "id": str(case.get("id", case.get("question", "unknown"))),
                "question": str(case.get("question", "")),
                "dataset_id": case_target,
                "source_doc_ids": source_doc_ids,
                "resolved_source_document_ids": resolved_source_ids,
                "source_resolution_status": (
                    "resolved" if resolved_source_ids else ("pending_upload" if source_doc_ids else "not_applicable")
                ),
                "status": "failed",
                "error": error,
            })
    scored = [item["score"] for item in results if "score" in item]
    source_resolution_counts: dict[str, int] = {}
    for item in results:
        resolution = str(item.get("source_resolution_status", "not_applicable"))
        source_resolution_counts[resolution] = source_resolution_counts.get(resolution, 0) + 1
    dataset_summary: dict[str, dict[str, Any]] = {}
    damaged_text_cases: list[str] = []
    for item in results:
        key = str(item.get("dataset_id") or "unknown")
        summary = dataset_summary.setdefault(key, {"case_count": 0, "mean_score": None, "zero_chunk_cases": 0, "cjk_zero_chunk_cases": 0})
        summary["case_count"] += 1
        if int(item.get("chunk_count", 0) or 0) == 0:
            summary["zero_chunk_cases"] += 1
            if item.get("query_language_hint") == "cjk":
                summary["cjk_zero_chunk_cases"] += 1
        diagnostic = item.get("text_diagnostic", {})
        if isinstance(diagnostic, Mapping) and diagnostic.get("suspected_reparse_damage"):
            damaged_text_cases.append(str(item.get("id", "unknown")))
    for key, summary in dataset_summary.items():
        matching = [item for item in results if str(item.get("dataset_id") or "unknown") == key]
        scores_for_dataset = [float(item["score"]) for item in matching if "score" in item]
        summary["mean_score"] = round(sum(scores_for_dataset) / len(scores_for_dataset), 4) if scores_for_dataset else None
        gold_shares = [
            float(item["gold_chunk_share_at_k"])
            for item in matching
            if item.get("gold_chunk_share_at_k") is not None
        ]
        unique_document_counts = [
            int(item["unique_document_count"])
            for item in matching
            if "unique_document_count" in item
        ]
        summary["gold_document_hit_rate"] = round(
            sum(bool(item.get("gold_hit")) for item in matching if item.get("gold_hit") is not None)
            / max(1, sum(item.get("gold_hit") is not None for item in matching)), 4
        ) if any(item.get("gold_hit") is not None for item in matching) else None
        summary["mean_gold_source_chunk_share_at_k"] = round(sum(gold_shares) / len(gold_shares), 4) if gold_shares else None
        summary["mean_unique_documents_per_query"] = round(
            sum(unique_document_counts) / len(unique_document_counts), 2
        ) if unique_document_counts else None
    zero_chunk_cases = [item for item in results if int(item.get("chunk_count", 0) or 0) == 0]
    recommendations: list[str] = []
    if any(item.get("source_resolution_status") == "pending_upload" for item in results):
        recommendations.append("先完成 source_doc_ids 对应文档的准入与上传，再评估论文题目。")
    resolved_zero_cjk = [
        item for item in zero_chunk_cases
        if item.get("source_resolution_status") in {"resolved", "not_applicable"}
        and item.get("query_language_hint") == "cjk"
    ]
    if resolved_zero_cjk:
        recommendations.append("已上传文档对中文问题无召回；应启用中文/英文术语扩展，或检查 RAGFlow 数据集语言与嵌入配置。")
    if zero_chunk_cases and not recommendations:
        recommendations.append("存在无召回题目；检查文档解析状态、索引块数量和检索阈值。")
    if damaged_text_cases:
        recommendations.append("部分返回 chunk 存在字符级断词，疑似 Markdown 被 RAGFlow 二次布局解析；应检查上传格式或按文本文件禁用二次 MinerU。")
    report_status = "ok" if not errors else ("failed" if not scored else "partial_failed")
    if report_status == "ok" and zero_chunk_cases:
        report_status = "needs_attention"
    report = {
        "dataset_id": target,
        "dataset_ids": sorted({str(item.get("dataset_id")) for item in results if item.get("dataset_id")}),
        "case_count": len(results),
        "benchmark_path": str(benchmark_path),
        "dataset_key_filter": dataset_key,
        "max_cases": max_cases,
        "status": report_status,
        "mean_score": round(sum(scored) / len(scored), 4) if scored else None,
        "source_resolution_counts": source_resolution_counts,
        "retrieval_query_fallback": use_required_terms_fallback,
        "retrieval_timeout_seconds": float(rag.get("retrieval_timeout_seconds", 60.0)),
        "retrieval_retry_on_timeout": retry_on_timeout,
        "dataset_summary": dataset_summary,
        "suspected_reparse_damage_cases": damaged_text_cases,
        "recommendations": recommendations,
        "errors": errors,
        "results": results,
    }
    output_root = Path(config["storage"]["root"]) / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "retrieval_benchmark_latest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
