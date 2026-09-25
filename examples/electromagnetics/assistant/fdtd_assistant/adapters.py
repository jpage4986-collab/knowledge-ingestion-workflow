"""Retrieval and parsing seams.

Only MockFixtureRetriever is active by default. The other adapters deliberately
raise a clear error instead of pretending to be connected to a service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .models import Evidence

DOMAIN_TERMS = ("FDTD", "CFL", "Yee", "稳定", "单位", "公式", "边界", "色散", "网格", "时间", "空间", "电场", "磁场", "步长")
DOMAIN_ALIASES = {
    "单位": ("秒", "米", "m/s"),
    "公式": ("Δt", "c"),
}


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> Sequence[Evidence]: ...

    def get_page(self, source_id: str, page: int) -> Sequence[Evidence]: ...


class RAGFlowRetriever:
    """RAGFlow boundary; only an injected client may perform I/O."""

    def __init__(self, client: object | None = None) -> None:
        self.client = client
        self._document_metadata: dict[str, dict] = {}

    def search(self, query: str, top_k: int = 5) -> Sequence[Evidence]:
        if self.client is None or not hasattr(self.client, "search"):
            raise RuntimeError("RAGFlow is not connected. Configure and inject a client explicitly.")
        try:
            raw = self.client.search(query, top_k=top_k)
        except TypeError:
            raw = self.client.search(query)
        results = [self._coerce(item, index, False) for index, item in enumerate(_result_items(raw), 1)]
        for item in results:
            self._document_metadata[item.source_id] = dict(item.upstream)
        return results

    def get_page(self, source_id: str, page: int | None, *, raw_path: str | None = None, ragflow_document_id: str | None = None) -> Sequence[Evidence]:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            return []
        if page_number <= 0:
            return []
        if self.client is None or not hasattr(self.client, "get_page"):
            raise RuntimeError("RAGFlow is not connected. Configure and inject a client explicitly.")
        metadata = dict(self._document_metadata.get(source_id, {}))
        if raw_path:
            metadata["raw_path"] = raw_path
        if ragflow_document_id:
            metadata["ragflow_document_id"] = ragflow_document_id
        try:
            raw = self.client.get_page(source_id, page_number, raw_path=metadata.get("raw_path"), ragflow_document_id=metadata.get("ragflow_document_id"))
        except TypeError:
            raw = self.client.get_page(source_id, page_number)
        results = [self._coerce(item, index, True) for index, item in enumerate(_result_items(raw), 1)]
        for item in results:
            merged = dict(metadata)
            merged.update(item.upstream)
            merged.setdefault("doc_id", source_id)
            merged.setdefault("page", page)
            merged["page_asset"] = {
                "kind": "document_page_reference",
                "raw_path": merged.get("raw_path"),
            "page": merged.get("page", page_number),
                "document_type": item.document_type,
                "dataset_id": merged.get("dataset_id"),
                "dataset_key": merged.get("dataset_key"),
                "ragflow_document_id": merged.get("ragflow_document_id"),
                "page_image_path": merged.get("page_image_path"),
                "mode": "reference_only",
            }
            item.upstream.clear()
            item.upstream.update(merged)
        return results

    @staticmethod
    def _coerce(value: Evidence | dict, index: int, fallback: bool) -> Evidence:
        if isinstance(value, Evidence):
            return value
        if not isinstance(value, dict):
            raise TypeError("RAGFlow client results must be Evidence or mapping objects")
        source_id = str(value.get("doc_id") or value.get("source_id") or value.get("document_id") or "ragflow-document")
        page_value = value.get("page")
        page = int(page_value) if page_value is not None else None
        chunk_id = str(value.get("chunk_id") or value.get("id") or index)
        locator = str(value.get("evidence_ref") or value.get("locator") or f"{source_id}:{page or chunk_id}")
        upstream = {key: value.get(key) for key in (
            "doc_id", "dataset_key", "dataset_id", "title", "source_path", "raw_path",
            "mineru_markdown_path", "content_list_path", "evidence_ref", "ragflow_document_id",
            "page", "chunk_id", "retrieval_score", "source_quality", "page_image_path",
            "page_locator", "answer_priority", "source_class", "source_url", "repo_name", "commit", "license",
            "provenance_verified", "admission_score", "parse_quality", "source_index",
        ) if key in value}
        if fallback:
            upstream["page_asset"] = {
                "kind": "document_page_reference",
                "raw_path": upstream.get("raw_path"),
                "page": page,
                "document_type": str(value.get("document_type") or "pdf"),
                "dataset_id": upstream.get("dataset_id"),
                "dataset_key": upstream.get("dataset_key"),
                "ragflow_document_id": upstream.get("ragflow_document_id"),
                "page_image_path": upstream.get("page_image_path"),
                "mode": "reference_only",
            }
        return Evidence(
            evidence_id=f"{source_id}:{chunk_id}:{page or 0}{'-full' if fallback else ''}",
            source_id=source_id,
            title=str(value.get("title") or source_id),
            content=str(value.get("content") or value.get("text") or ""),
            document_type=str(value.get("document_type") or "pdf"),
            page=page,
            source_quality=float(value.get("source_quality") or 0.0),
            retrieval_score=float(value.get("retrieval_score") or 0.0),
            locator=locator,
            is_page_fallback=fallback,
            upstream=upstream,
        )


def _result_items(value: object) -> list[Evidence | dict]:
    if isinstance(value, dict):
        for key in ("data", "results", "items", "evidence"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError("retriever client must return a list or a mapping containing results")


class MinerUParser:
    """Integration placeholder for layout-aware PDF/PPT/image parsing."""

    def __init__(self, client: object | None = None) -> None:
        self.client = client

    def parse_page(self, source_id: str, page: int) -> Sequence[Evidence]:
        raise RuntimeError("MinerU is not connected. Configure and inject a client explicitly.")


class WebSearcher:
    """Optional web seam; no network call is made by the offline implementation."""

    def search(self, query: str, top_k: int = 3) -> Sequence[Evidence]:
        raise RuntimeError("Web search is disabled in offline mode; inject a searcher explicitly.")


@dataclass(frozen=True)
class FixturePage:
    source_id: str
    title: str
    document_type: str
    page: int
    text: str
    source_quality: float


def default_fixtures() -> list[FixturePage]:
    return [
        FixturePage(
            "fdtd-notes", "FDTD 电磁计算讲义", "pdf", 4,
            "Yee 网格将电场 E 与磁场 H 交错放置在空间和时间上。空间步长为 Δx, Δy, Δz，时间步长为 Δt。",
            0.78,
        ),
        FixturePage(
            "fdtd-notes", "FDTD 电磁计算讲义", "pdf", 5,
            "三维均匀介质中的 CFL 稳定条件为 Δt ≤ 1/(c√((1/Δx)^2+(1/Δy)^2+(1/Δz)^2))；真空中 c≈2.998×10^8 m/s。二维或一维情形删去不用的项。",
            0.86,
        ),
        FixturePage(
            "fdtd-lab", "FDTD 实验记录", "pptx", 7,
            "网格收敛检查：减小 Δx 后应比较反射系数、谐振频率与能量误差；Δt 应以秒计，长度应统一为米。",
            0.62,
        ),
        FixturePage(
            "fdtd-figure", "稳定性示意图", "png", 1,
            "图示说明：当时间步长超过 CFL 上限，数值场会出现非物理增长；处于上限以内通常保持稳定，但仍需检查色散与边界反射。",
            0.70,
        ),
    ]


def _terms(text: str) -> set[str]:
    latin = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower()))
    cjk = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    domain = {term.lower() for term in DOMAIN_TERMS if term.lower() in text.lower() or term in text}
    for trigger, aliases in DOMAIN_ALIASES.items():
        if trigger in text:
            domain.update(alias.lower() for alias in aliases)
    return latin | cjk | domain


class MockFixtureRetriever:
    """Deterministic lexical retriever with whole-page fallback support."""

    def __init__(self, pages: Iterable[FixturePage] | None = None) -> None:
        self.pages = list(pages or default_fixtures())

    def search(self, query: str, top_k: int = 5) -> Sequence[Evidence]:
        q = _terms(query)
        scored: list[tuple[float, FixturePage]] = []
        for page in self.pages:
            overlap = len(q & _terms(page.text + " " + page.title))
            bonus = 0.15 if any(alias in query.lower() and alias in page.text.lower() for alias in ("cfl", "yee", "稳定", "单位", "公式")) else 0.0
            score = min(1.0, 0.18 + 0.14 * overlap + bonus)
            if overlap or bonus:
                scored.append((score, page))
        scored.sort(key=lambda item: (-item[0], item[1].source_id, item[1].page))
        return [self._evidence(page, score, False, index) for index, (score, page) in enumerate(scored[:top_k], 1)]

    def get_page(self, source_id: str, page: int) -> Sequence[Evidence]:
        return [self._evidence(item, 1.0, True, 1) for item in self.pages if item.source_id == source_id and item.page == page]

    @staticmethod
    def _evidence(page: FixturePage, score: float, fallback: bool, index: int) -> Evidence:
        return Evidence(
            evidence_id=f"{page.source_id}-p{page.page}{'-full' if fallback else ''}",
            source_id=page.source_id,
            title=page.title,
            content=page.text,
            document_type=page.document_type,
            page=page.page,
            source_quality=page.source_quality,
            retrieval_score=score,
            locator=f"{page.source_id}:{page.page}",
            is_page_fallback=fallback,
        )
