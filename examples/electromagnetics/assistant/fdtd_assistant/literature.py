"""Read-only adapters for the upstream literature index.

This module deliberately does not build a second knowledge base. It reads the
upstream JSON/SQLite index, resolves evidence text from the paths or references
owned by the upstream pipeline, and exposes the existing Retriever contract.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlparse

from .adapters import _terms
from .models import Evidence


UPSTREAM_FIELDS = (
    "doc_id",
    "dataset_key",
    "dataset_id",
    "title",
    "source_path",
    "raw_path",
    "mineru_markdown_path",
    "content_list_path",
    "evidence_ref",
    "ragflow_document_id",
    "page",
    "chunk_id",
    "retrieval_score",
    "source_quality",
)


@dataclass(frozen=True)
class LiteratureIndexRecord:
    """Stable cross-project row shared by the upstream and answer projects."""

    doc_id: str
    dataset_key: str
    dataset_id: str | None
    title: str
    source_path: str | None
    raw_path: str | None
    mineru_markdown_path: str | None
    content_list_path: str | None
    evidence_ref: str
    ragflow_document_id: str | None
    page: int | None
    chunk_id: str
    retrieval_score: float
    source_quality: float
    text: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "LiteratureIndexRecord":
        missing = [key for key in ("doc_id", "dataset_key", "title", "evidence_ref", "chunk_id") if not value.get(key)]
        if missing:
            raise ValueError(f"upstream evidence row missing required fields: {', '.join(missing)}")
        page = value.get("page")
        try:
            page = int(page) if page is not None and page != "" else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid page in upstream evidence row: {page!r}") from exc
        return cls(
            doc_id=str(value["doc_id"]),
            dataset_key=str(value["dataset_key"]),
            dataset_id=_optional_text(value.get("dataset_id")),
            title=str(value["title"]),
            source_path=_optional_text(value.get("source_path")),
            raw_path=_optional_text(value.get("raw_path")),
            mineru_markdown_path=_optional_text(value.get("mineru_markdown_path")),
            content_list_path=_optional_text(value.get("content_list_path")),
            evidence_ref=str(value["evidence_ref"]),
            ragflow_document_id=_optional_text(value.get("ragflow_document_id")),
            page=page,
            chunk_id=str(value["chunk_id"]),
            retrieval_score=float(value.get("retrieval_score") or 0.0),
            source_quality=float(value.get("source_quality") or 0.0),
            text=_optional_text(value.get("text") or value.get("content") or value.get("markdown")),
        )

    def upstream_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("text", None)
        return value


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records") or payload.get("evidence") or payload.get("items")
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("upstream JSON must be a list or an object containing records/evidence/items")
    return rows


class JSONLiteratureIndex:
    """Read-only JSON/JSONL loader for the upstream evidence index."""

    def __init__(self, records: Sequence[LiteratureIndexRecord], *, base_dir: str | Path | None = None) -> None:
        self.records = tuple(records)
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    @classmethod
    def from_path(cls, path: str | Path) -> "JSONLiteratureIndex":
        index_path = Path(path).resolve()
        if index_path.suffix.lower() in {".jsonl", ".ndjson"}:
            rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = _json_rows(json.loads(index_path.read_text(encoding="utf-8")))
        return cls([LiteratureIndexRecord.from_mapping(row) for row in rows], base_dir=index_path.parent)


class SQLiteLiteratureIndex:
    """Read-only SQLite adapter; the upstream owns schema creation and writes."""

    def __init__(self, path: str | Path, *, table: str = "evidence_index") -> None:
        self.path = Path(path).resolve()
        self.table = _safe_identifier(table)

    def records(self) -> tuple[LiteratureIndexRecord, ...]:
        connection = sqlite3.connect(self.path)
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(f"SELECT * FROM {self.table}").fetchall()
        finally:
            connection.close()
        return tuple(LiteratureIndexRecord.from_mapping(dict(row)) for row in rows)


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"unsafe SQLite table identifier: {value!r}")
    return value


class UpstreamLiteratureRetriever:
    """Lexical retriever over upstream rows with page-level content fallback."""

    def __init__(self, records: Iterable[LiteratureIndexRecord], *, base_dir: str | Path | None = None, dataset_key: str | None = None) -> None:
        self.records = tuple(records)
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.dataset_key = dataset_key
        self._content_cache: dict[tuple[str, int | None, bool], str] = {}

    @classmethod
    def from_json(cls, path: str | Path, *, dataset_key: str | None = None) -> "UpstreamLiteratureRetriever":
        index = JSONLiteratureIndex.from_path(path)
        return cls(index.records, base_dir=index.base_dir, dataset_key=dataset_key)

    @classmethod
    def from_sqlite(cls, path: str | Path, *, table: str = "evidence_index", dataset_key: str | None = None) -> "UpstreamLiteratureRetriever":
        index_path = Path(path).resolve()
        return cls(SQLiteLiteratureIndex(index_path, table=table).records(), base_dir=index_path.parent, dataset_key=dataset_key)

    def search(self, query: str, top_k: int = 5) -> Sequence[Evidence]:
        query_terms = _terms(query)
        scored: list[tuple[float, LiteratureIndexRecord]] = []
        for record in self.records:
            if self.dataset_key is not None and record.dataset_key != self.dataset_key:
                continue
            searchable = " ".join((record.title, record.dataset_key, record.evidence_ref, record.chunk_id, record.text or ""))
            overlap = len(query_terms & _terms(searchable))
            if overlap:
                score = max(record.retrieval_score, min(1.0, 0.10 + 0.12 * overlap + 0.25 * record.source_quality))
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].doc_id, item[1].page or -1, item[1].chunk_id))
        return [self._evidence(record, score, False) for score, record in scored[:top_k]]

    def get_page(self, source_id: str, page: int) -> Sequence[Evidence]:
        matches = [record for record in self.records if record.doc_id == source_id and record.page == page]
        if not matches:
            return []
        return [self._evidence(record, 1.0, True) for record in matches]

    def _evidence(self, record: LiteratureIndexRecord, score: float, fallback: bool) -> Evidence:
        content = self._resolve_content(record, full_page=fallback)
        upstream = record.upstream_dict()
        if fallback:
            upstream["page_asset"] = {
                "kind": "document_page_reference",
                "raw_path": record.raw_path or record.source_path,
                "page": record.page,
                "document_type": "pdf" if (record.raw_path or record.source_path or "").lower().endswith(".pdf") else "document",
                "dataset_id": record.dataset_id,
                "dataset_key": record.dataset_key,
                "ragflow_document_id": record.ragflow_document_id,
                "page_image_path": None,
                "mode": "reference_only",
            }
        return Evidence(
            evidence_id=f"{record.doc_id}:{record.chunk_id}:{record.page or 0}{'-full' if fallback else ''}",
            source_id=record.doc_id,
            title=record.title,
            content=content,
            document_type="pdf" if (record.raw_path or record.source_path or "").lower().endswith(".pdf") else "document",
            page=record.page,
            source_quality=record.source_quality,
            retrieval_score=score,
            locator=record.evidence_ref or f"{record.doc_id}:{record.page}",
            is_page_fallback=fallback,
            upstream=upstream,
        )

    def _resolve_content(self, record: LiteratureIndexRecord, *, full_page: bool) -> str:
        cache_key = (record.doc_id, record.page, full_page)
        if cache_key in self._content_cache:
            return self._content_cache[cache_key]
        content = ""
        if full_page and record.content_list_path:
            content = _read_content_list(record.content_list_path, record.page, self.base_dir)
        if full_page and not content and record.mineru_markdown_path:
            content = _read_text_path(record.mineru_markdown_path, self.base_dir)
        if not content:
            content = record.text or ""
        if not content and record.evidence_ref:
            content = _read_reference(record.evidence_ref, self.base_dir)
        if not content and record.content_list_path:
            content = _read_content_list(record.content_list_path, record.page, self.base_dir)
        if not content and record.mineru_markdown_path:
            content = _read_text_path(record.mineru_markdown_path, self.base_dir)
        self._content_cache[cache_key] = content
        return content


def _resolve_path(value: str, base_dir: Path | None) -> Path:
    parsed = urlparse(value)
    raw = unquote(parsed.path) if parsed.scheme == "file" else value
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path if base_dir else path)


def _read_text_path(value: str, base_dir: Path | None) -> str:
    path = _resolve_path(value, base_dir)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _read_reference(value: str, base_dir: Path | None) -> str:
    if value.startswith("file:") or Path(value).suffix.lower() in {".txt", ".md", ".markdown"}:
        return _read_text_path(value, base_dir)
    return ""


def _read_content_list(value: str, page: int | None, base_dir: Path | None) -> str:
    path = _resolve_path(value, base_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    candidates = payload if isinstance(payload, list) else payload.get("pages", payload.get("content", [])) if isinstance(payload, dict) else []
    if not isinstance(candidates, list):
        return ""
    selected: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        item_page = item.get("page") or item.get("page_idx") or item.get("page_no")
        if page is not None and item_page is not None:
            try:
                if int(item_page) not in {page, page - 1}:
                    continue
            except (TypeError, ValueError):
                continue
        text = item.get("text") or item.get("content") or item.get("markdown") or item.get("text_content")
        if text:
            selected.append(str(text))
    return "\n".join(selected)
