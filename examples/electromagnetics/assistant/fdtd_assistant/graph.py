from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List

from .adapters import MockFixtureRetriever, Retriever, WebSearcher
from .evaluation import evaluate_evidence, evidence_group_matches
from .models import AssistantState, Evidence, QuestionPart, evidence_from_dict
from .query_translation import coverage_alias_groups, translate_for_retrieval


def _decompose(state: AssistantState) -> AssistantState:
    question = state["question"].strip()
    chunks = [chunk.strip(" ，,；;。？?") for chunk in question.replace("以及", "；").replace("并且", "；").split("；") if chunk.strip()]
    if len(chunks) == 1:
        chunks = [chunk.strip(" ，,。") for chunk in question.replace("、", "，").split("，") if chunk.strip()]
    if not chunks:
        chunks = [question]
    parts = []
    for index, chunk in enumerate(chunks, 1):
        terms = [term for term in ("FDTD", "CFL", "Yee", "稳定", "单位", "公式", "边界", "色散") if term.lower() in chunk.lower() or term in chunk]
        groups = coverage_alias_groups(chunk)
        parts.append(QuestionPart(f"q{index}", chunk, terms, groups))
    state["subquestions"] = [part.__dict__ for part in parts]
    state.setdefault("trace", []).append("decompose")
    return state


def _retrieve(state: AssistantState, retriever: Retriever) -> AssistantState:
    # Keep the original domain context on every sub-query.  Splitting a Chinese
    # question by commas is useful for coverage, but sending bare fragments
    # such as “端口” can pull an unrelated software manual from a shared RAG.
    original = state["question"]
    # Always retain the complete question; it often carries the document/topic
    # name that disappears when a Chinese question is split at commas.
    queries = [original]
    for part in state.get("subquestions", []):
        translated_part, changed = translate_for_retrieval(str(part["text"]))
        query_part = translated_part if changed else str(part["text"])
        queries.append(f"{original}；{query_part}")
    found: list[Evidence] = []
    with ThreadPoolExecutor(max_workers=min(2, len(queries))) as pool:
        futures = {pool.submit(retriever.search, query, top_k=4): query for query in queries}
        for future in as_completed(futures):
            try:
                found.extend(future.result())
            except Exception as exc:
                state.setdefault("retrieval_errors", []).append(type(exc).__name__)
    unique = {item.evidence_id: item for item in found}
    def priority(item: Evidence) -> tuple[float, float, str]:
        relevance = min(1.0, max(0.0, float(item.retrieval_score or 0.0)))
        quality = min(1.0, max(0.0, float(item.source_quality or 0.0)))
        return (-(0.72 * relevance + 0.28 * quality), -relevance, item.evidence_id)

    state["candidates"] = [item.as_dict() for item in sorted(unique.values(), key=priority)]
    existing = {item["evidence_id"]: item for item in state.get("selected_evidence", [])}
    existing.update({item["evidence_id"]: item for item in state["candidates"]})
    ordered = sorted(
        (evidence_from_dict(item) for item in existing.values()),
        key=priority,
    )
    parts = [QuestionPart(**part) for part in state.get("subquestions", [])]
    specific_parts = [part for part in parts if part.required_term_groups]
    if not specific_parts:
        selected = ordered[:6]
    else:
        selected_by_id: dict[str, Evidence] = {}
        for part in specific_parts:
            remaining = set(range(len(part.required_term_groups)))
            selected_for_part = 0
            while remaining and selected_for_part < 4 and len(selected_by_id) < 6:
                best: tuple[int, float, tuple[float, float, str], Evidence, list[bool]] | None = None
                for item in ordered:
                    if item.evidence_id in selected_by_id:
                        continue
                    matches = evidence_group_matches(part, item) or []
                    newly_covered = sum(bool(matches[index]) for index in remaining)
                    if newly_covered <= 0:
                        continue
                    alignment = sum(matches) / max(1, len(matches))
                    candidate = (newly_covered, alignment, priority(item), item, matches)
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
                if best is None:
                    break
                _, _, _, item, matches = best
                selected_by_id[item.evidence_id] = item
                remaining.difference_update(index for index, matched in enumerate(matches) if matched)
                selected_for_part += 1

            # A second independent, fully aligned source is useful corroboration;
            # repeated adjacent chunks from the same document add little value.
            if not remaining and selected_for_part < 3 and len(selected_by_id) < 6:
                def provenance_id(item: Evidence) -> str:
                    return str(item.upstream.get("source_index") or item.source_id)

                chosen_sources = {provenance_id(item) for item in selected_by_id.values()}
                for item in ordered:
                    if item.evidence_id in selected_by_id or provenance_id(item) in chosen_sources:
                        continue
                    matches = evidence_group_matches(part, item) or []
                    if matches and all(matches):
                        selected_by_id[item.evidence_id] = item
                        break
        selected = list(selected_by_id.values())
        # Preserve page-level fallback evidence already obtained from a cited source.
        for item in ordered:
            if item.is_page_fallback and item.evidence_id not in selected_by_id and len(selected) < 6:
                selected.append(item)
    state["selected_evidence"] = [item.as_dict() for item in selected]
    state.setdefault("trace", []).append("retrieve")
    return state


def _assess(state: AssistantState) -> AssistantState:
    parts = [QuestionPart(**part) for part in state.get("subquestions", [])]
    evidence = [evidence_from_dict(item) for item in state.get("selected_evidence", [])]
    result = evaluate_evidence(parts, evidence, state.get("iteration", 0), state.get("max_iterations", 1), state.get("allow_web", False))
    state["evaluation"] = result.as_dict()
    state["route"] = result.route
    state.setdefault("trace", []).append("assess")
    return state


def _page_fallback(state: AssistantState, retriever: Retriever) -> AssistantState:
    seen = {(item["source_id"], item.get("page")) for item in state.get("selected_evidence", [])}
    added: list[Evidence] = []
    for item in state.get("selected_evidence", []):
        page_value = item.get("page")
        try:
            page = int(page_value)
        except (TypeError, ValueError):
            # An unresolved locator is not a request for page zero or page
            # one. Preserve the chunk evidence and skip visual-page fallback.
            continue
        if page <= 0:
            continue
        key = (item["source_id"], page)
        if key in seen:
            upstream = item.get("upstream", {})
            try:
                added.extend(retriever.get_page(
                    item["source_id"],
                    page,
                    raw_path=upstream.get("raw_path"),
                    ragflow_document_id=upstream.get("ragflow_document_id"),
                ))
            except TypeError:
                added.extend(retriever.get_page(item["source_id"], page))
            seen.remove(key)
    unique = {item["evidence_id"]: item for item in state.get("selected_evidence", [])}
    unique.update({item.evidence_id: item.as_dict() for item in added})
    state["fallback_pages"] = [item for item in unique.values() if item.get("is_page_fallback")]
    state["selected_evidence"] = list(unique.values())
    state["iteration"] = state.get("iteration", 0) + 1
    state.setdefault("trace", []).append("page_fallback")
    return state


def _web_retrieve(state: AssistantState, web_searcher: WebSearcher | None) -> AssistantState:
    if web_searcher is None:
        state["web_search_status"] = "not_configured"
    else:
        try:
            items = web_searcher.search(state["question"], top_k=3)
            state["web_evidence"] = [item.as_dict() for item in items]
            state["selected_evidence"] = state.get("selected_evidence", []) + state["web_evidence"]
            state["web_search_status"] = "used"
        except Exception as exc:
            state["web_search_status"] = "failed"
            state["web_search_error"] = f"{type(exc).__name__}: web search unavailable"
    state["iteration"] = state.get("iteration", 0) + 1
    state.setdefault("trace", []).append("web_retrieve")
    return state


def _rereetrieve(state: AssistantState) -> AssistantState:
    """Advance the retry budget in the compiled graph before retrieval."""
    state["iteration"] = state.get("iteration", 0) + 1
    state.setdefault("trace", []).append("rereetrieve")
    return state


def _generate(state: AssistantState) -> AssistantState:
    raw_evidence = [evidence_from_dict(item) for item in state.get("selected_evidence", [])]
    # Prefer the full-page recovery for a locator and avoid repeating the same citation.
    by_locator: dict[str, Evidence] = {}
    for item in raw_evidence:
        current = by_locator.get(item.locator)
        if current is None or (item.is_page_fallback and not current.is_page_fallback):
            by_locator[item.locator] = item
    evidence = list(by_locator.values())
    state.setdefault("web_search_status", "not_needed")
    evaluation = state.get("evaluation", {})
    if evaluation.get("evidence_coverage", 0.0) <= 0.0:
        state["answer"] = "当前知识库没有覆盖这个问题的可靠证据，拒绝生成无依据结论；请补充相关论文证据或启用经过授权的检索。"
        state["citations"] = []
        state["page_context"] = []
        state["refusal"] = True
        state.setdefault("trace", []).append("generate")
        return state
    lines = ["基于当前离线知识库，结论如下："]
    if evidence:
        for index, item in enumerate(evidence[:6], 1):
            lines.append(f"- {item.content} [{index}]")
    else:
        lines.append("- 当前没有检索到足够证据，不能可靠作答。")
    if evaluation.get("missing_parts"):
        lines.append("- 仍缺少：" + ", ".join(evaluation["missing_parts"]) + "；请补充资料或启用联网检索。")
    if evaluation.get("conflict_notes"):
        lines.append("- 来源口径冲突：" + "；".join(evaluation["conflict_notes"]))
    lines.append("引用：" + (" ".join(f"[{i}] {item.locator}" for i, item in enumerate(evidence[:6], 1)) or "无"))
    state["answer"] = "\n".join(lines)
    state["citations"] = [item.locator for item in evidence[:6]]
    state["page_context"] = [
        {
            "type": "pdf_page" if item.document_type.lower() == "pdf" else "document_page",
            "source_id": item.source_id,
            "title": item.title,
            "document_type": item.document_type,
            "page": item.page,
            "locator": item.locator,
            "text": item.content,
            "is_full_page": True,
            "upstream": item.upstream,
            "page_asset": item.upstream.get("page_asset", {
                "kind": "document_page_reference",
                "raw_path": item.upstream.get("raw_path"),
                "page": item.page,
                "document_type": item.document_type,
                "dataset_id": item.upstream.get("dataset_id"),
                "dataset_key": item.upstream.get("dataset_key"),
                "ragflow_document_id": item.upstream.get("ragflow_document_id"),
                "page_image_path": item.upstream.get("page_image_path"),
                "mode": "reference_only",
            }),
        }
        for item in evidence
        if item.is_page_fallback
    ]
    if state["page_context"]:
        state["page_context_instruction"] = "以下是检索到的相关文档整页文本。请先阅读整页再回答；只能引用页内可支持的结论，不得补写页内不存在的参数或场值。"
    state.setdefault("trace", []).append("generate")
    return state


class LocalGraph:
    """Small deterministic executor used when langgraph is not installed."""

    def __init__(self, retriever: Retriever, web_searcher: WebSearcher | None = None) -> None:
        self.retriever, self.web_searcher = retriever, web_searcher

    def invoke(self, initial: AssistantState) -> AssistantState:
        state: AssistantState = dict(initial)
        state.setdefault("iteration", 0)
        state.setdefault("max_iterations", 2)
        state.setdefault("allow_web", False)
        state.setdefault("trace", [])
        _decompose(state)
        while True:
            _retrieve(state, self.retriever)
            _assess(state)
            route = state.get("route", "answer")
            if route == "page_fallback" and state["iteration"] < state["max_iterations"]:
                _page_fallback(state, self.retriever)
                continue
            if route == "reretrieve" and state["iteration"] < state["max_iterations"]:
                state["iteration"] += 1
                continue
            if route == "web_retrieve" and state.get("allow_web") and state["iteration"] < state["max_iterations"]:
                _web_retrieve(state, self.web_searcher)
                continue
            _generate(state)
            return state


def _langgraph_graph(retriever: Retriever, web_searcher: WebSearcher | None):
    """Compile the same node contract with LangGraph when available."""
    from langgraph.graph import END, StateGraph  # type: ignore

    graph = StateGraph(AssistantState)
    graph.add_node("decompose", _decompose)
    graph.add_node("retrieve", lambda s: _retrieve(s, retriever))
    graph.add_node("assess", _assess)
    graph.add_node("page_fallback", lambda s: _page_fallback(s, retriever))
    graph.add_node("web_retrieve", lambda s: _web_retrieve(s, web_searcher))
    graph.add_node("rereetrieve", _rereetrieve)
    graph.add_node("generate", _generate)
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges("assess", lambda s: s.get("route", "answer"), {
        "page_fallback": "page_fallback", "reretrieve": "rereetrieve", "web_retrieve": "web_retrieve", "answer": "generate"
    })
    graph.add_edge("page_fallback", "assess")
    graph.add_edge("web_retrieve", "assess")
    graph.add_edge("rereetrieve", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


_DEFAULT_GRAPH = None


def build_graph(retriever: Retriever | None = None, web_searcher: WebSearcher | None = None):
    global _DEFAULT_GRAPH
    if retriever is None and web_searcher is None and _DEFAULT_GRAPH is not None:
        return _DEFAULT_GRAPH
    retriever = retriever or MockFixtureRetriever()
    try:
        graph = _langgraph_graph(retriever, web_searcher)
    except ImportError:
        graph = LocalGraph(retriever, web_searcher)
    if retriever.__class__ is MockFixtureRetriever and web_searcher is None and _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = graph
    return graph


def run_assistant(question: str, *, retriever: Retriever | None = None, web_searcher: WebSearcher | None = None, allow_web: bool = False, max_iterations: int = 2, include_full_page: bool = True) -> AssistantState:
    graph = build_graph(retriever, web_searcher)
    state = graph.invoke({"question": question, "allow_web": allow_web, "max_iterations": max_iterations})
    if not include_full_page:
        state.pop("page_context", None)
        state.pop("page_context_instruction", None)
    return state
