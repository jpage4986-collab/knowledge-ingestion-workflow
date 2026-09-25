from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from chunk_quality import evaluate_chunks, split_markdown


def looks_like_inline_secret(value: str) -> bool:
    """Recognize a secret supplied directly in config without ever logging it."""
    candidate = str(value or "").strip()
    return len(candidate) >= 24 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate) is None


def resolve_api_secret(selector: str) -> str:
    """Resolve an environment-variable selector or a backwards-compatible inline secret."""
    candidate = str(selector or "").strip()
    if not candidate:
        return ""
    environment_value = os.environ.get(candidate, "").strip()
    if environment_value:
        return environment_value
    return candidate if looks_like_inline_secret(candidate) else ""


def secret_selector_label(selector: str) -> str:
    """Return a safe diagnostic label which never exposes an inline secret."""
    candidate = str(selector or "").strip()
    if looks_like_inline_secret(candidate):
        return "inline_config_secret"
    return candidate or "unset"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_c_drive_health(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the configured C: health state for every ingest entry point."""
    from maintenance import disk_health

    storage = config.get("storage", {})
    return disk_health(
        "C:",
        float(storage.get("c_drive_warning_gb", 20)),
        float(storage.get("c_drive_critical_gb", 10)),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def published_document_name(doc_id: str, markdown: str) -> str:
    """Return a stable RAGFlow filename for idempotent retry and recovery."""
    content_hash = hashlib.sha256(markdown.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{doc_id}--{content_hash}.md"


def canonical_ragflow_name(name: str) -> str:
    """Strip only RAGFlow's collision suffix, e.g. ``(2)`` before extension."""
    return re.sub(r"\s*\(\d+\)(?=\.[^.]+$)", "", str(name or "").strip())


def reserve_ragflow_upload_attempt(index: Any, row: dict[str, Any], *, is_original_pdf: bool) -> None:
    """Persist upload limits before sending bytes to RAGFlow.

    A literature item may make at most three new upload requests in total;
    its original PDF is a large visual asset and may be sent only once.
    Existing documents are looked up by stable content-derived name before
    this function is called, so retries reuse them without consuming a slot.
    """
    total_attempts = int(row.get("ragflow_upload_attempts") or 0)
    pdf_attempts = int(row.get("ragflow_pdf_upload_attempts") or 0)
    if total_attempts >= 3:
        raise RuntimeError("该文献已达到 3 次 RAGFlow 上传请求上限，已停止再次上传。")
    if is_original_pdf and pdf_attempts >= 1:
        raise RuntimeError("该原始 PDF 已发送过一次上传请求，按大文件规则不再重传。")
    row["ragflow_upload_attempts"] = total_attempts + 1
    if is_original_pdf:
        row["ragflow_pdf_upload_attempts"] = pdf_attempts + 1
    index.upsert_paper(row)


class StorageQuotaError(RuntimeError):
    pass


def directory_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


class StorageQuota:
    """Protect the user-specified G: literature budget before new writes."""

    def __init__(self, config: Mapping[str, Any]):
        self.root = Path(config["root"])
        self.quota_bytes = int(float(config.get("quota_gb", 150)) * 1024**3)
        self.reserve_bytes = int(float(config.get("reserve_gb", 10)) * 1024**3)
        required_drive = str(config.get("required_drive", "")).strip().rstrip("\\/").upper()
        if required_drive and self.root.drive.upper() != required_drive:
            raise ValueError(f"文献库必须放在 {required_drive}，当前路径是 {self.root}")

    @property
    def used_bytes(self) -> int:
        return directory_size_bytes(self.root)

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.quota_bytes - self.used_bytes - self.reserve_bytes)

    def ensure_capacity(self, estimated_new_bytes: int, label: str = "new literature") -> None:
        if estimated_new_bytes > self.remaining_bytes:
            used_gb = self.used_bytes / 1024**3
            estimate_gb = estimated_new_bytes / 1024**3
            raise StorageQuotaError(
                f"G: 文献库配额不足：已用 {used_gb:.2f} GB，预计新增 {estimate_gb:.2f} GB，"
                f"配额 {self.quota_bytes / 1024**3:.0f} GB，预留 {self.reserve_bytes / 1024**3:.0f} GB；已阻止 {label}。"
            )


def slug(value: str, limit: int = 100) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return (cleaned or "paper")[:limit]


def normalise_title(value: str) -> str:
    """Create a provider-independent title key for cross-source deduplication."""
    return re.sub(r"[^\w]+", " ", str(value or "").casefold(), flags=re.UNICODE).strip()


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def http_request(
    url: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def http_json(
    url: str,
    method: str = "GET",
    payload: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
) -> Any:
    merged = {"Accept": "application/json"}
    merged.update(headers or {})
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        merged.setdefault("Content-Type", "application/json")
    _, raw, _ = http_request(url, method, body, merged, timeout)
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


@dataclass(frozen=True)
class PaperAttachment:
    zotero_key: str
    parent_key: str
    title: str
    authors: str
    year: str
    doi: str
    tags: str
    filename: str
    local_path: Path
    dataset_key: str = ""
    abstract: str = ""
    landing_url: str = ""
    pdf_url: str = ""
    source_provider: str = ""


@dataclass(frozen=True)
class LiteratureCandidate:
    """Metadata returned by a literature discovery provider."""

    candidate_id: str
    query: str
    title: str
    authors: str
    year: str
    doi: str
    landing_url: str
    pdf_url: str
    abstract: str = ""
    source: str = "openalex"
    citation_count: int = 0


def candidate_priority(candidate: LiteratureCandidate) -> float:
    """Rank candidates before consuming download, MinerU, and RAGFlow budget."""
    text = f"{candidate.title} {candidate.abstract}".lower()
    weighted_terms = {
        "fdtd": 0.28,
        "finite difference time domain": 0.26,
        "finite-difference time-domain": 0.26,
        "pml": 0.10,
        "perfectly matched layer": 0.10,
        "electromagnetic": 0.10,
        "photonic": 0.08,
        "lumerical": 0.06,
        "tidy3d": 0.03,
        "mesh": 0.05,
        "boundary condition": 0.05,
        "monitor": 0.03,
        "material": 0.16,
        "materials": 0.16,
        "dielectric": 0.16,
        "permittivity": 0.16,
        "permeability": 0.14,
        "conductivity": 0.10,
        "loss tangent": 0.10,
        "dispersive": 0.10,
        "microwave properties": 0.10,
        "debye": 0.08,
        "drude": 0.08,
        "lorentz": 0.08,
        "cole-cole": 0.08,
        # Prioritize the user's current metasurface / transmission-reflection
        # scope instead of silently screening it out for lacking the literal
        # token "FDTD". Model review and evidence gates still decide admission.
        "metasurface": 0.24,
        "metamaterial": 0.18,
        "space-time-coding": 0.14,
        "transmitarray": 0.16,
        "reflectarray": 0.16,
        "frequency selective surface": 0.16,
        "floquet": 0.12,
        "unit cell": 0.10,
        "surface impedance": 0.10,
        "transmission": 0.08,
        "reflection": 0.08,
        "s-parameter": 0.08,
        "scattering": 0.06,
    }
    score = sum(weight for term, weight in weighted_terms.items() if term in text)
    if candidate.doi:
        score += 0.08
    if candidate.pdf_url:
        score += 0.08
    score += min(max(candidate.citation_count, 0), 1000) / 1000 * 0.10
    return round(min(score, 1.0), 4)


def attachment_priority(paper: PaperAttachment) -> float:
    """Rank a Zotero PDF while preserving curated first-party sources.

    First-party cases and manuals have their own dataset policy.  They are
    allowed through the paper-specific FDTD relevance threshold because a
    COMSOL application note may document an electromagnetic workflow without
    literally containing the string ``FDTD``.
    """
    text = f"{paper.title} {paper.tags} {paper.filename} {paper.abstract}".lower()
    source = str(paper.source_provider or "").lower()
    tags = str(paper.tags or "").lower()
    if source in {"official", "comsol-official", "first-party"} or "official" in source or "first-party" in tags:
        if paper.dataset_key in {"cases", "software_manuals"}:
            return 0.95
    terms = {
        "fdtd": 0.40,
        "finite difference": 0.28,
        "electromagnetic": 0.18,
        "comsol": 0.16,
        "lumerical": 0.16,
        "cst": 0.12,
        "hfss": 0.12,
        "pml": 0.12,
        "material": 0.08,
        "optics": 0.06,
        "photonics": 0.06,
    }
    score = sum(weight for term, weight in terms.items() if term in text)
    if paper.doi:
        score += 0.05
    return round(min(score, 1.0), 4)


class LiteratureDiscoverer(Protocol):
    def search(self, query: str, limit: int = 10) -> list[LiteratureCandidate]: ...


class OpenAlexDiscoverer:
    """Search OpenAlex and expose only candidates with a usable OA PDF URL."""

    def __init__(self, base_url: str = "https://api.openalex.org", mailto: str = "", timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.mailto = mailto
        self.timeout = timeout

    @staticmethod
    def _abstract(value: Any) -> str:
        if not isinstance(value, Mapping):
            return ""
        inverted = value.get("abstract_inverted_index")
        if not isinstance(inverted, Mapping):
            return ""
        words: dict[int, str] = {}
        for token, positions in inverted.items():
            if isinstance(positions, list):
                for position in positions:
                    if isinstance(position, int):
                        words[position] = str(token)
        return " ".join(words[index] for index in sorted(words))

    @staticmethod
    def _pdf_location(work: Mapping[str, Any]) -> tuple[str, str]:
        locations: list[Any] = []
        for key in ("best_oa_location", "primary_location"):
            if work.get(key):
                locations.append(work[key])
        if isinstance(work.get("locations"), list):
            locations.extend(work["locations"])
        for location in locations:
            if not isinstance(location, Mapping):
                continue
            pdf_url = str(location.get("pdf_url") or "").strip()
            landing = str(location.get("landing_page_url") or "").strip()
            if pdf_url:
                return landing or pdf_url, pdf_url
            if landing.lower().endswith(".pdf"):
                return landing, landing
        return "", ""

    def search(self, query: str, limit: int = 10) -> list[LiteratureCandidate]:
        params: dict[str, Any] = {"search": query, "per-page": max(1, min(limit, 100)), "mailto": self.mailto}
        url = self.base_url + "/works?" + urllib.parse.urlencode({k: v for k, v in params.items() if v})
        payload = http_json(url, timeout=self.timeout)
        results = payload.get("results", []) if isinstance(payload, Mapping) else []
        candidates: list[LiteratureCandidate] = []
        for work in results:
            if not isinstance(work, Mapping):
                continue
            landing, pdf_url = self._pdf_location(work)
            authors = "; ".join(
                str((authorship.get("author") or {}).get("display_name") or "")
                for authorship in (work.get("authorships") or [])
                if isinstance(authorship, Mapping) and isinstance(authorship.get("author"), Mapping)
            )
            work_id = str(work.get("id") or work.get("doi") or "")
            if not work_id:
                continue
            candidates.append(
                LiteratureCandidate(
                    candidate_id=slug(work_id.replace("https://openalex.org/", "openalex-")),
                    query=query,
                    title=str(work.get("title") or ""),
                    authors=authors,
                    year=str(work.get("publication_year") or ""),
                    doi=str(work.get("doi") or "").removeprefix("https://doi.org/"),
                    landing_url=landing or str(work.get("id") or ""),
                    pdf_url=pdf_url,
                    abstract=self._abstract(work),
                    citation_count=int(work.get("cited_by_count") or 0),
                )
            )
        return candidates


class ArxivDiscoverer:
    """Query arXiv's public Atom feed as a quota-independent OA source."""

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query", timeout: float = 45):
        self.base_url = base_url
        self.timeout = timeout

    @staticmethod
    def _text(element: ET.Element | None) -> str:
        return " ".join((element.text or "").split()) if element is not None else ""

    @classmethod
    def _parse_feed(cls, raw: bytes, query: str) -> list[LiteratureCandidate]:
        root = ET.fromstring(raw)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        candidates: list[LiteratureCandidate] = []
        for entry in root.findall("a:entry", ns):
            landing = cls._text(entry.find("a:id", ns)).replace("http://", "https://")
            if not landing:
                continue
            arxiv_id = landing.rsplit("/", 1)[-1]
            pdf_url = ""
            for link in entry.findall("a:link", ns):
                if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf":
                    pdf_url = str(link.attrib.get("href") or "").replace("http://", "https://")
                    break
            authors = "; ".join(cls._text(author.find("a:name", ns)) for author in entry.findall("a:author", ns))
            published = cls._text(entry.find("a:published", ns))
            candidates.append(
                LiteratureCandidate(
                    candidate_id=slug(f"arxiv-{arxiv_id}"), query=query,
                    title=cls._text(entry.find("a:title", ns)), authors=authors,
                    year=published[:4], doi="", landing_url=landing, pdf_url=pdf_url,
                    abstract=cls._text(entry.find("a:summary", ns)), source="arxiv",
                )
            )
        return candidates

    @staticmethod
    def _search_query(query: str) -> str:
        """Turn a human query into a forgiving arXiv field query.

        arXiv treats a quoted multi-word ``all:`` expression as one exact
        phrase. That is too strict for normal discovery queries such as
        ``FDTD PML electromagnetic simulation``. Keep the method name in
        the query, then add at most one useful context term.
        """
        lowered = query.lower()
        tokens = re.findall(r"[a-z0-9]+", lowered)
        if "fdtd" in tokens or "finite difference time domain" in lowered:
            method_terms = ["FDTD"]
        else:
            stopwords = {"and", "or", "the", "for", "with", "using", "simulation", "method", "study"}
            method_terms = list(dict.fromkeys(token for token in tokens if token not in stopwords))[:2]
            if not method_terms:
                method_terms = ["electromagnetic"]
        context_terms = ("electromagnetic", "pml", "photonic", "metamaterial", "antenna", "scattering")
        context = next((term for term in context_terms if term in tokens and term.lower() not in {value.lower() for value in method_terms}), "")
        parts = [f"all:{term}" for term in method_terms]
        if context:
            parts.append(f"all:{context}")
        return " AND ".join(parts)

    def search(self, query: str, limit: int = 10) -> list[LiteratureCandidate]:
        params = urllib.parse.urlencode({"search_query": self._search_query(query), "start": 0, "max_results": max(1, min(limit, 50))})
        url = self.base_url + "?" + params
        request = urllib.request.Request(url, headers={"Accept": "application/atom+xml", "User-Agent": "EM-FDTD-Knowledge-Collector/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except Exception as exc:
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if not curl:
                raise RuntimeError(f"arXiv request failed: {exc}") from exc
            completed = subprocess.run([curl, "--fail", "--location", "--max-time", str(int(self.timeout)), "-A", "EM-FDTD-Knowledge-Collector/0.1", url], capture_output=True, timeout=self.timeout + 10)
            if completed.returncode != 0:
                raise RuntimeError(f"arXiv request failed via Python and curl: {exc}") from exc
            raw = completed.stdout
        return self._parse_feed(raw, query)


def download_open_pdf(url: str, destination: Path, timeout: float = 90) -> Path:
    """Download a candidate PDF without pretending HTML is a PDF."""
    if not url:
        raise ValueError("candidate has no PDF URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        status, body, headers = http_request(url, timeout=timeout, headers={"User-Agent": "EM-FDTD-Knowledge-Collector/0.1"})
        content_type = str(headers.get("Content-Type") or "").lower()
        if not body.startswith(b"%PDF") and "pdf" not in content_type:
            raise RuntimeError(f"downloaded resource is not a PDF: HTTP {status}, content-type={content_type}")
        destination.write_bytes(body)
    except RuntimeError as exc:
        # On Windows, Python's CA bundle can be incomplete while the system
        # curl trust store works. Fall back only to curl's normal certificate
        # validation; never use --insecure and never accept HTML as a PDF.
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if not curl:
            raise
        completed = subprocess.run(
            [curl, "--fail", "--location", "--max-time", str(int(timeout)), "-A", "EM-FDTD-Knowledge-Collector/0.1", "-o", str(destination), url],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 10,
        )
        if completed.returncode != 0 or not destination.exists():
            raise RuntimeError(f"PDF download failed via Python and curl: {exc}; curl={completed.stderr[-500:]}") from exc
        body = destination.read_bytes()
        if not body.startswith(b"%PDF"):
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded resource is not a PDF after curl fallback: {url}") from exc
        return destination
    if not destination.read_bytes().startswith(b"%PDF"):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded resource is not a PDF: {url}")
    return destination


class ZoteroSource(Protocol):
    def list_pdf_attachments(self, limit: int | None = None) -> list[PaperAttachment]: ...


def file_url_to_path(value: str) -> Path:
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme.lower() != "file":
        raise ValueError(f"Zotero attachment is not a local file URL: {value}")
    path = urllib.parse.unquote(parsed.path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    if parsed.netloc and parsed.netloc.lower() not in {"", "localhost"}:
        path = "//" + parsed.netloc + path
    result = Path(path)
    if not result.exists():
        raise FileNotFoundError(result)
    return result


class ZoteroLocalApi:
    """Read-only Zotero desktop adapter. It never writes to Zotero."""

    def __init__(self, base_url: str, library_prefix: str = "/users/0", timeout: float = 20):
        self.base_url = base_url.rstrip("/")
        self.library_prefix = library_prefix.strip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {})
        url = join_url(self.base_url, path)
        if query:
            url += "?" + query
        return http_json(url, timeout=self.timeout)

    def _get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        query = urllib.parse.urlencode(params or {})
        url = join_url(self.base_url, path)
        if query:
            url += "?" + query
        _, raw, _ = http_request(url, timeout=self.timeout)
        return raw.decode("utf-8", errors="replace")

    def _all_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        start = 0
        limit = 100
        while True:
            page = self._get(
                f"{self.library_prefix}/items",
                {"format": "json", "limit": limit, "start": start},
            )
            if not isinstance(page, list):
                raise RuntimeError("Unexpected Zotero response: items is not a list")
            items.extend(page)
            if len(page) < limit:
                return items
            start += len(page)

    def list_pdf_attachments(self, limit: int | None = None) -> list[PaperAttachment]:
        items = self._all_items()
        parents: dict[str, dict[str, Any]] = {}
        attachments: list[dict[str, Any]] = []
        for item in items:
            data = item.get("data", {})
            if data.get("itemType") == "attachment":
                attachment = dict(data)
                attachment["_key"] = item.get("key")
                attachments.append(attachment)
            elif item.get("key"):
                parents[item["key"]] = data

        found: list[PaperAttachment] = []
        for attachment in attachments:
            filename = str(attachment.get("filename") or "")
            content_type = str(attachment.get("contentType") or "")
            if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
                continue
            parent_key = str(attachment.get("parentItem") or "")
            parent = parents.get(parent_key, {})
            try:
                url = self._get_text(
                    f"{self.library_prefix}/items/{attachment['_key']}/file/view/url"
                )
                local_path = file_url_to_path(str(url))
            except (KeyError, RuntimeError, ValueError, FileNotFoundError):
                continue
            creators = parent.get("creators", [])
            authors = "; ".join(
                " ".join(str(x) for x in (creator.get("firstName"), creator.get("lastName")) if x)
                or str(creator.get("name") or "")
                for creator in creators
            )
            tags = "; ".join(str(tag.get("tag") or "") for tag in parent.get("tags", []))
            found.append(
                PaperAttachment(
                    zotero_key=str(attachment.get("key") or ""),
                    parent_key=parent_key,
                    title=str(parent.get("title") or attachment.get("title") or filename),
                    authors=authors,
                    year=str(parent.get("date") or "")[:4],
                    doi=str(parent.get("DOI") or ""),
                    tags=tags,
                    filename=filename or local_path.name,
                    local_path=local_path,
                    abstract=str(parent.get("abstractNote") or ""),
                    source_provider="zotero",
                )
            )
            if limit and len(found) >= limit:
                break
        return found


class ZoteroSqlite:
    """Read-only fallback for a local Zotero data directory."""

    def __init__(self, db_path: str | Path, storage_root: str | Path, timeout: float = 20):
        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root)
        self.timeout = timeout

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        # Zotero can hold a write lock while updating its database. The
        # immutable read-only view never writes or waits on that lock. It may
        # lag behind a live WAL update, so the desktop API remains preferred
        # when enabled and the next cycle refreshes the snapshot.
        connection = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True, timeout=0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _item_values(connection: sqlite3.Connection, item_id: int) -> dict[str, str]:
        rows = connection.execute(
            """
            SELECT f.fieldName, v.value
            FROM itemData d
            JOIN fields f ON f.fieldID = d.fieldID
            JOIN itemDataValues v ON v.valueID = d.valueID
            WHERE d.itemID = ?
            """,
            (item_id,),
        ).fetchall()
        return {str(row["fieldName"]): str(row["value"] or "") for row in rows}

    def _attachment_path(self, key: str, stored_path: str) -> Path:
        value = str(stored_path or "")
        if value.startswith("storage:"):
            return self.storage_root / key / value.split(":", 1)[1]
        if value.startswith("file:"):
            return file_url_to_path(value)
        return Path(value)

    def list_pdf_attachments(self, limit: int | None = None) -> list[PaperAttachment]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT ai.itemID AS attachment_id, ai.key AS attachment_key,
                       a.parentItemID, a.contentType, a.path,
                       pi.key AS parent_key
                FROM itemAttachments a
                JOIN items ai ON ai.itemID = a.itemID
                LEFT JOIN items pi ON pi.itemID = a.parentItemID
                ORDER BY ai.key
                """
            ).fetchall()
            found: list[PaperAttachment] = []
            for row in rows:
                filename = str(row["path"] or "").split(":", 1)[-1]
                content_type = str(row["contentType"] or "")
                if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
                    continue
                try:
                    local_path = self._attachment_path(str(row["attachment_key"]), str(row["path"] or ""))
                except (ValueError, FileNotFoundError):
                    continue
                if not local_path.exists() or not local_path.is_file():
                    continue
                parent_id = row["parentItemID"]
                parent_values = self._item_values(connection, int(parent_id)) if parent_id else {}
                creator_rows = connection.execute(
                    """
                    SELECT c.firstName, c.lastName, c.fieldMode
                    FROM itemCreators ic
                    JOIN creators c ON c.creatorID = ic.creatorID
                    LEFT JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
                    WHERE ic.itemID = ? AND (ct.creatorType = 'author' OR ct.creatorType IS NULL)
                    ORDER BY ic.orderIndex
                    """,
                    (parent_id,),
                ).fetchall() if parent_id else []
                authors = "; ".join(
                    " ".join(str(value or "") for value in (creator["firstName"], creator["lastName"]) if value)
                    for creator in creator_rows
                )
                tag_rows = connection.execute(
                    """
                    SELECT t.name
                    FROM itemTags it JOIN tags t ON t.tagID = it.tagID
                    WHERE it.itemID = ? ORDER BY t.name
                    """,
                    (parent_id,),
                ).fetchall() if parent_id else []
                tags = "; ".join(str(tag["name"] or "") for tag in tag_rows)
                found.append(
                    PaperAttachment(
                        zotero_key=str(row["attachment_key"] or ""),
                        parent_key=str(row["parent_key"] or ""),
                        title=parent_values.get("title") or filename,
                        authors=authors,
                        year=parent_values.get("date", "")[:4],
                        doi=parent_values.get("DOI", ""),
                        tags=tags,
                        filename=filename or local_path.name,
                        local_path=local_path,
                        abstract=parent_values.get("abstractNote", ""),
                        source_provider="zotero",
                    )
                )
                if limit and len(found) >= limit:
                    break
            return found
        finally:
            connection.close()


class ZoteroAutoSource:
    """Use the desktop API when available, otherwise use the read-only SQLite fallback."""

    def __init__(self, api: ZoteroSource, sqlite_source: ZoteroSource):
        self.api = api
        self.sqlite_source = sqlite_source
        self.last_source = "unknown"
        self.last_api_error = ""

    def list_pdf_attachments(self, limit: int | None = None) -> list[PaperAttachment]:
        try:
            result = self.api.list_pdf_attachments(limit=limit)
            self.last_source = "local_api"
            self.last_api_error = ""
            return result
        except Exception as exc:
            self.last_api_error = str(exc)
            result = self.sqlite_source.list_pdf_attachments(limit=limit)
            self.last_source = "sqlite"
            return result


class MinerUParser:
    """Run every PDF through MinerU CLI with an explicit all-pages request."""

    def __init__(
        self,
        command: str | list[str],
        output_root: Path,
        extra_args: list[str] | None = None,
        temp_root: Path | None = None,
        timeout_seconds: float = 1800,
    ):
        self.command = [command] if isinstance(command, str) else list(command)
        self.output_root = output_root
        self.extra_args = extra_args or []
        self.temp_root = temp_root
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        if self.temp_root:
            self.temp_root.mkdir(parents=True, exist_ok=True)
        if "--pages" in self.extra_args:
            raise ValueError("Do not override --pages; this workflow always parses all PDF pages")

    def _env(self) -> dict[str, str] | None:
        if not self.temp_root:
            return None
        env = os.environ.copy()
        env.update({"TEMP": str(self.temp_root), "TMP": str(self.temp_root), "TMPDIR": str(self.temp_root)})
        return env

    def is_available(self) -> bool:
        executable = self.command[0]
        return bool(shutil.which(executable) or Path(executable).exists())

    def cli_help(self) -> str:
        if not self.is_available():
            return ""
        try:
            completed = subprocess.run(self.command + ["--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=self._env(), timeout=20)
        except subprocess.TimeoutExpired:
            return ""
        return completed.stdout + "\n" + completed.stderr

    def cli_style(self) -> str:
        help_text = self.cli_help()
        return "modern" if "parse" in help_text and "--pages" in help_text else "legacy"

    def build_command(self, pdf_path: Path, run_dir: Path) -> list[str]:
        if self.cli_style() == "modern":
            return self.command + ["parse", str(pdf_path), "--pages", "all", "-o", str(run_dir)] + self.extra_args
        # MinerU 3.4.x installed in the current environment uses the legacy
        # -p/-o CLI. A file path means the complete PDF; it has no page filter.
        return self.command + ["-p", str(pdf_path), "-o", str(run_dir)] + self.extra_args

    def parse(self, pdf_path: Path, doc_id: str) -> dict[str, Any]:
        if not self.is_available():
            return {"status": "blocked", "error": f"MinerU command not found: {self.command[0]}"}
        run_dir = self.output_root / doc_id / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(pdf_path, run_dir)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._env(),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
            (run_dir / "mineru.stdout.txt").write_text(stdout, encoding="utf-8")
            (run_dir / "mineru.stderr.txt").write_text(stderr, encoding="utf-8")
            return {
                "status": "failed",
                "run_dir": str(run_dir),
                "error": f"MinerU timed out after {self.timeout_seconds:g} seconds",
            }
        (run_dir / "mineru.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "mineru.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        markdown_files = sorted(run_dir.rglob("*.md")) + sorted(run_dir.rglob("*.mmd"))
        markdown_path: Path | None = markdown_files[0] if markdown_files else None
        if markdown_path is None and completed.stdout.strip():
            markdown_path = run_dir / "mineru.stdout.md"
            markdown_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            return {"status": "failed", "run_dir": str(run_dir), "error": completed.stderr[-2000:]}
        if markdown_path is None:
            return {"status": "failed", "run_dir": str(run_dir), "error": "MinerU completed without Markdown output"}
        return {
            "status": "done",
            "run_dir": str(run_dir),
            "markdown_path": str(markdown_path),
            "stdout": completed.stdout[-2000:],
        }


FIELD_TERMS: dict[str, tuple[str, ...]] = {
    "target_wavelength": ("wavelength", "波长", "nm", "µm", "μm"),
    "frequency_range": ("frequency", "频率", "THz", "GHz"),
    "materials": ("material", "材料", "SiO2", "silicon", "silica", "permittivity", "折射率"),
    "geometry": ("geometry", "结构", "厚度", "width", "wide", "height", "period", "周期", "radius", "thick", "thickness", "length"),
    "solver": ("FDTD", "finite-difference", "finite difference", "Lumerical", "Meep", "solver"),
    "mesh": ("mesh", "grid", "网格", "spatial resolution", "cell size"),
    "cfl_timestep": ("time step", "时间步长", "timestep", "Δt", "dt", "Courant", "CFL"),
    "boundary_pml": ("PML", "boundary", "边界", "absorbing", "Bloch", "symmetric"),
    "source": ("source", "源", "mode source", "plane wave", "Gaussian", "port"),
    "monitors": ("monitor", "监视器", "probe", "mode expansion", "power monitor", "field monitor"),
    "stop_condition": ("auto shutoff", "autoshutoff", "stop condition", "停止条件", "simulation time"),
    "reported_metrics": ("transmission", "reflection", "efficiency", "S-parameter", "透射", "反射", "耦合效率"),
}

FIELD_CONTEXT_HINTS: dict[str, tuple[str, ...]] = {
    "target_wavelength": ("incident", "laser", "wavelength", "illumination", "nm", "µm", "μm"),
    "frequency_range": ("source", "excitation", "bandwidth", "pulse", "frequency", "GHz", "MHz", "THz"),
    "materials": ("material", "medium", "permittivity", "permeability", "silicon", "air", "water"),
    "geometry": ("geometry", "domain", "cavity", "period", "width", "wide", "height", "length", "long", "radius", "position"),
    "solver": ("FDTD", "Yee", "finite difference", "time-marching", "solver", "stability"),
    "mesh": ("mesh", "grid", "cell", "spatial", "coarse", "fine", "subgrid", "resolution"),
    "cfl_timestep": ("CFL", "Courant", "time step", "timestep", "delta t", "Δt", "dt"),
    "boundary_pml": ("PML", "PEC", "boundary", "absorbing", "perfect electric", "Bloch"),
    "source": ("source", "excitation", "Gaussian", "pulse", "incident", "plane wave", "port"),
    "monitors": ("monitor", "probe", "record", "field", "power", "observation"),
    "stop_condition": ("simulation time", "time step", "stop", "shutoff", "convergence", "iteration"),
    "reported_metrics": ("SAR", "transmission", "reflection", "coefficient", "error", "accuracy", "efficiency"),
}

FIELD_PRIMARY_TERMS: dict[str, tuple[str, ...]] = {
    "target_wavelength": ("wavelength", "波长"),
    "frequency_range": ("bandwidth", "frequency", "频率"),
    "geometry": ("geometry", "domain", "cavity", "period", "width", "wide", "height", "length", "long", "thick"),
    "solver": ("proposed", "scheme", "method", "FDTD", "finite difference", "Yee", "solver"),
    "mesh": ("mesh", "grid", "cell size", "spatial resolution", "网格"),
    "cfl_timestep": ("time step", "timestep", "时间步长", "Δt"),
    "boundary_pml": ("PML", "PEC", "absorbing"),
    "source": ("source", "excitation", "Gaussian", "pulse"),
    "monitors": ("monitor", "probe", "power monitor", "field monitor"),
    "stop_condition": ("simulation time", "shutoff", "convergence"),
    "reported_metrics": ("SAR", "transmission", "reflection", "efficiency", "accuracy"),
}

FIELD_CONFLICT_TERMS: dict[str, tuple[str, ...]] = {
    "target_wavelength": ("fwhm", "resonance", "mesh", "resolution"),
    "geometry": ("mesh size", "spatial resolution", "runtime", "memory"),
    "source": ("fwhm", "resonance", "free spectral range", "runtime", "memory"),
    "stop_condition": ("elapsed", "runtime", "iterations", "memory", "discretization"),
}


def detect_extraction_warnings(values: Mapping[str, Any]) -> list[str]:
    """Flag field values that look like neighboring result/context fields."""
    warnings: list[str] = []
    for field, conflict_terms in FIELD_CONFLICT_TERMS.items():
        value = str(values.get(field, "")).lower()
        hits = [term for term in conflict_terms if term in value]
        if hits:
            warnings.append(f"{field}:疑似包含冲突词({', '.join(hits)})")
    return warnings


def _contains_term(text: str, term: str) -> bool:
    """Match short technical tokens without treating them as substrings."""
    normalized = term.strip().lower()
    if not normalized:
        return False
    if len(normalized) <= 3 and re.fullmatch(r"[a-z0-9]+", normalized):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text, flags=re.I))
    return normalized in text


def extract_context(
    text: str,
    terms: Iterable[str],
    window: int = 500,
    hints: Iterable[str] = (),
    primary_terms: Iterable[str] = (),
    exclude_terms: Iterable[str] = (),
    hard_exclude: bool = False,
) -> str:
    """Return a concise, parameter-like context instead of the first hit.

    MinerU output often mentions a generic word (for example ``frequency``
    for CPU hardware) before the actual simulation setup. Score candidate
    sentences/paragraphs by field hints and nearby numeric/unit evidence so
    the deterministic fallback remains useful while still being explicitly
    non-authoritative until model review.
    """
    term_list = [str(term) for term in terms if str(term).strip()]
    hint_list = [str(hint).lower() for hint in hints if str(hint).strip()]
    primary_list = [str(term) for term in primary_terms if str(term).strip()]
    exclude_list = [str(term).lower() for term in exclude_terms if str(term).strip()]
    if not term_list:
        return ""
    segments = [
        segment.strip()
        for segment in re.split(r"\n\s*\n|(?<=[.!?。！？；;])\s+", text)
        if segment.strip()
    ]
    candidates: list[tuple[float, int, str]] = []
    for segment_index, segment in enumerate(segments):
        lowered = segment.lower()
        matched = [(index, term.lower()) for index, term in enumerate(term_list) if _contains_term(lowered, term)]
        if not matched:
            continue
        excluded_hits = sum(1 for term in exclude_list if term in lowered)
        if hard_exclude and excluded_hits:
            continue
        hint_hits = sum(1 for hint in hint_list if hint in lowered)
        primary_hits = sum(1 for term in primary_list if _contains_term(lowered, term))
        numeric_hits = len(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:nm|µm|μm|um|mm|m|fs|ps|ns|GHz|MHz|THz|Hz|s|V/m|dB)\b", segment, flags=re.I))
        unitless_numbers = min(3, len(re.findall(r"\b\d+(?:\.\d+)?\b", segment)))
        term_specificity = sum(len(term_list) - index for index, _ in matched) / len(term_list)
        length_penalty = max(0.0, (len(segment) - 650) / 650)
        score = 5.0 * primary_hits + 3.0 * hint_hits + 2.5 * min(4, numeric_hits) + 0.35 * unitless_numbers + term_specificity - length_penalty - 2.0 * excluded_hits
        candidates.append((score, -segment_index, segment))
    if not candidates:
        return ""
    _, _, selected = max(candidates)
    selected = re.sub(r"\s+", " ", selected).strip()
    if len(selected) <= window:
        return selected
    positions = [selected.lower().find(term.lower()) for term in term_list]
    positions = [position for position in positions if position >= 0]
    position = min(positions) if positions else 0
    start = max(0, position - min(140, window // 3))
    end = min(len(selected), start + window)
    return selected[start:end].strip()


def extract_target_wavelength(text: str) -> str:
    """Extract operating wavelengths, excluding resonance/FWHM result values."""
    pattern = r"\b\d+(?:\.\d+)?\s*(?:nm|µm|μm|um|micron)\b"
    values: list[str] = []
    for match in re.finditer(pattern, text, flags=re.I):
        boundary_positions = [text.rfind(delimiter, 0, match.start()) for delimiter in (".", "!", "?", ";", "\n")]
        sentence_start = max(boundary_positions, default=-1) + 1
        before = text[sentence_start:match.start()]
        lowered = before.lower()
        if any(term in lowered for term in ("fwhm", "resonance", "q-factor", "quality factor", "mesh", "resolution")):
            continue
        keyword_matches = list(re.finditer(
            r"\b(?:wavelength|波长|center(?:ed)?|designed|target|operat(?:e|ing))\b",
            before,
            flags=re.I,
        ))
        direct_at = bool(re.search(r"\bat\s*$", before, flags=re.I))
        if not keyword_matches and not direct_at:
            continue
        if keyword_matches:
            last_keyword_tail = before[keyword_matches[-1].start():]
            if re.search(pattern, last_keyword_tail, flags=re.I):
                continue
        elif not direct_at:
            continue
        values.append(match.group(0))
    return "; ".join(dict.fromkeys(values[:12]))


def extract_evidence(text: str, terms: Iterable[str], window: int = 500) -> dict[str, Any]:
    """Locate a field and nearby MinerU image references for audit/review."""
    positions = [text.lower().find(term.lower()) for term in terms]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return {}
    position = min(positions)
    start = max(0, position - 120)
    end = min(len(text), position + window)
    nearby_images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text[start:end])
    return {
        "char_offset": position,
        "line": text.count("\n", 0, position) + 1,
        "excerpt": re.sub(r"\s+", " ", text[start:end]).strip(),
        "image_refs": list(dict.fromkeys(nearby_images)),
    }


def extract_value_evidence(text: str, value: str, window: int = 500) -> dict[str, Any]:
    """Locate the sentence/paragraph containing a selected extracted value."""
    needle = str(value or "").strip()
    if not needle:
        return {}
    position = text.lower().find(needle.lower())
    if position < 0:
        return {}
    boundary_positions = [text.rfind(delimiter, 0, position) for delimiter in (".", "!", "?", ";", "\n")]
    start = max(boundary_positions, default=-1) + 1
    end_candidates = [text.find(delimiter, position + len(needle)) for delimiter in (".", "!", "?", ";", "\n")]
    end_candidates = [candidate for candidate in end_candidates if candidate >= 0]
    end = min(end_candidates, default=min(len(text), position + window)) + 1
    end = min(len(text), max(end, position + len(needle)))
    excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
    nearby_images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text[start:min(len(text), end + window)])
    return {
        "char_offset": position,
        "line": text.count("\n", 0, position) + 1,
        "excerpt": excerpt,
        "image_refs": list(dict.fromkeys(nearby_images)),
    }


def mineru_page_evidence(markdown_path: Path, field_terms: Mapping[str, Iterable[str]]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Read MinerU's page-indexed content list without reparsing the PDF."""
    candidates = sorted(markdown_path.parent.glob("*_content_list.json"))
    if not candidates:
        return "", {}
    content_list_path = candidates[0]
    try:
        items = json.loads(content_list_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return str(content_list_path), {}
    if not isinstance(items, list):
        return str(content_list_path), {}
    pages: dict[int, list[tuple[str, str]]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        asset = str(item.get("img_path") or "").strip()
        pages.setdefault(page_idx, []).append((text, asset))
    result: dict[str, dict[str, Any]] = {}
    for field, terms in field_terms.items():
        term_list = [str(term).lower() for term in terms]
        matches: list[tuple[int, str, list[str]]] = []
        for page_idx, blocks in pages.items():
            page_text = " ".join(block[0] for block in blocks)
            if any(term in page_text.lower() for term in term_list):
                assets = [block[1] for block in blocks if block[1]]
                matches.append((page_idx, page_text, assets))
        if matches:
            page_numbers = [page_idx + 1 for page_idx, _, _ in matches]
            assets: list[str] = []
            excerpts: list[str] = []
            for _, page_text, page_assets in matches[:3]:
                excerpts.append(re.sub(r"\s+", " ", page_text)[:800])
                for asset in page_assets:
                    asset_path = (markdown_path.parent / asset).resolve()
                    assets.append(str(asset_path) if asset_path.is_file() else asset)
            result[field] = {
                "pages": page_numbers,
                "review_pages": page_numbers[:3],
                "page_excerpt": "\n\n".join(excerpts),
                "page_assets": list(dict.fromkeys(assets)),
            }
    return str(content_list_path), result


def persist_evidence_manifest(storage_root: Path, doc_id: str, evidence_ref: str) -> dict[str, Any]:
    """Persist a standalone, review-friendly evidence manifest on G:."""
    try:
        payload = json.loads(evidence_ref)
        if not isinstance(payload, dict):
            payload = {"raw_evidence_ref": evidence_ref}
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {"raw_evidence_ref": evidence_ref}
    target = storage_root / "evidence" / doc_id / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload["manifest_path"] = str(target)
    payload["generated_at"] = utc_now()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def materialize_page_evidence(
    original_pdf_path: str | Path,
    page_numbers: Iterable[int],
    target: Path,
    pdftoppm_command: str = "pdftoppm",
) -> list[dict[str, Any]]:
    """Export cited PDF pages as compact raster PDF/PNG evidence.

    This is intentionally limited to pages already identified by MinerU field
    evidence. It creates a compact review fallback on G:, not a second copy of
    the full document. Poppler renders directly from the original PDF; using
    pypdf to copy a page can duplicate shared resources and create a 50+ MB
    file for every page of a large manual. Failures are returned as data so
    ingestion can continue.
    """
    pdf_path = Path(str(original_pdf_path or ""))
    page_set: set[int] = set()
    for page in page_numbers:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            page_set.add(page_number)
    pages = sorted(page_set)
    if not pdf_path.is_file() or not pages:
        return []
    target.mkdir(parents=True, exist_ok=True)
    try:
        from pypdf import PdfReader
        page_count = len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        return [{"status": "failed", "error": f"PDF page export unavailable: {type(exc).__name__}: {exc}"}]
    executable = shutil.which(pdftoppm_command)
    if not executable and Path(pdftoppm_command).is_file():
        executable = str(Path(pdftoppm_command))
    results: list[dict[str, Any]] = []
    for page_number in pages:
        item: dict[str, Any] = {"page": page_number, "status": "ok"}
        page_pdf = target / f"page-{page_number:04d}.pdf"
        page_png = target / f"page-{page_number:04d}.png"
        if page_number > page_count:
            item.update({"status": "failed", "error": f"page {page_number} exceeds PDF page count {page_count}"})
            results.append(item)
            continue
        if executable and not page_png.is_file():
            prefix = target / f"page-{page_number:04d}"
            try:
                subprocess.run(
                    [executable, "-png", "-r", "150", "-singlefile", "-f", str(page_number), "-l", str(page_number), str(pdf_path), str(prefix)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                item["png_error"] = f"page PNG export failed: {type(exc).__name__}: {exc}"
        item["png_path"] = str(page_png)
        item["png_exists"] = page_png.is_file()
        if not executable:
            item["png_error"] = f"pdftoppm unavailable: {pdftoppm_command}"
        if page_png.is_file() and not page_pdf.is_file():
            try:
                from PIL import Image
                with Image.open(page_png) as image:
                    image.convert("RGB").save(page_pdf, "PDF", resolution=150.0)
                item["pdf_source"] = "rasterized_png"
            except Exception as exc:
                item["pdf_error"] = f"compact page PDF export failed: {type(exc).__name__}: {exc}"
        if not page_pdf.is_file():
            # Poppler is optional.  Keep the page-level fallback functional on
            # machines that do not have pdftoppm by copying the cited page
            # directly from the preserved original PDF.
            try:
                from pypdf import PdfReader, PdfWriter
                reader = PdfReader(str(pdf_path))
                writer = PdfWriter()
                writer.add_page(reader.pages[page_number - 1])
                with page_pdf.open("wb") as stream:
                    writer.write(stream)
                item["pdf_source"] = "original_pdf_page"
            except Exception as exc:
                item["pdf_error"] = f"original PDF page export failed: {type(exc).__name__}: {exc}"
        item["pdf_path"] = str(page_pdf)
        item["pdf_exists"] = page_pdf.is_file()
        if not item["png_exists"] and not item["pdf_exists"]:
            item["status"] = "failed"
            item.setdefault("error", "no page-level PDF or PNG fallback was produced")
        results.append(item)
    return results


def extract_simulation_summary(
    markdown: str,
    paper: PaperAttachment,
    doc_id: str,
    markdown_path: str,
    original_pdf_path: str | None = None,
) -> dict[str, Any]:
    values: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    for field, terms in FIELD_TERMS.items():
        context_terms = terms
        exclude_terms: tuple[str, ...] = ()
        hard_exclude = False
        if field == "target_wavelength":
            context_terms = ("wavelength", "波长")
            exclude_terms = ("mesh", "resolution", "bandwidth")
            hard_exclude = True
        elif field == "geometry":
            exclude_terms = ("mesh", "resolution", "bandwidth", "runtime", "memory")
            hard_exclude = True
        elif field == "source":
            exclude_terms = ("mesh size", "spatial resolution", "runtime", "memory", "resonance", "free spectral range", "fwhm")
            hard_exclude = True
        elif field == "stop_condition":
            exclude_terms = ("elapsed", "runtime", "iterations", "memory", "discretization")
            hard_exclude = True
        values[field] = extract_context(
            markdown,
            context_terms,
            hints=FIELD_CONTEXT_HINTS.get(field, ()),
            primary_terms=FIELD_PRIMARY_TERMS.get(field, ()),
            exclude_terms=exclude_terms,
            hard_exclude=hard_exclude,
        )
        if field == "target_wavelength" and not values[field]:
            values[field] = extract_context(
                markdown,
                ("nm", "µm", "μm", "um", "micron"),
                hints=FIELD_CONTEXT_HINTS.get(field, ()),
                primary_terms=FIELD_PRIMARY_TERMS.get(field, ()),
                exclude_terms=exclude_terms,
                hard_exclude=False,
            )
        selected_value = str(values.get(field, "")).strip()
        selected_evidence = extract_value_evidence(markdown, selected_value[:160]) if selected_value else {}
        evidence[field] = selected_evidence or extract_evidence(markdown, terms)
    target_wavelength = extract_target_wavelength(markdown)
    if target_wavelength:
        values["target_wavelength"] = target_wavelength
        evidence["target_wavelength"] = extract_value_evidence(markdown, target_wavelength) or evidence.get("target_wavelength", {})
    materials = re.findall(r"\b(?:SiO2|Si|silicon|silica|silicon dioxide|air|water|Au|Ag|GaAs)\b", markdown, flags=re.I)
    if materials:
        values["materials"] = "; ".join(dict.fromkeys(materials[:20]))
    content_list_path, page_evidence = mineru_page_evidence(Path(markdown_path), FIELD_TERMS)
    for field, page_data in page_evidence.items():
        evidence.setdefault(field, {}).update(page_data)
    critical = ("target_wavelength", "materials", "geometry", "mesh", "boundary_pml", "source", "monitors", "reported_metrics")
    present = sum(bool(values.get(field)) for field in critical)
    extraction_warnings = detect_extraction_warnings(values)
    quality = round(present / len(critical), 3)
    if extraction_warnings:
        quality = round(max(0.0, quality - min(0.25, 0.05 * len(extraction_warnings))), 3)
    evidence_ref = {
        "document_id": doc_id,
        "markdown_path": markdown_path,
        "content_list_path": content_list_path,
        "original_pdf": str(original_pdf_path or paper.local_path),
        "fields": evidence,
        "review_instruction": "若字段不确定，优先回看 original_pdf 对应页及 image_refs，不把启发式摘录当作事实。",
    }
    return {
        "doc_id": doc_id,
        "zotero_key": paper.zotero_key,
        "title": paper.title,
        "doi": paper.doi,
        "authors": paper.authors,
        "year": paper.year,
        **values,
        "quality_score": quality,
        "extraction_status": "ok" if quality >= 0.625 and not extraction_warnings else "needs_review",
        "extraction_warnings": extraction_warnings,
        "evidence_ref": json.dumps(evidence_ref, ensure_ascii=False),
        "markdown_path": markdown_path,
        "notes": (
            "Heuristic field extraction; verify against MinerU page JSON/images before using as a design fact."
            + (" Warnings: " + "; ".join(extraction_warnings) if extraction_warnings else "")
        ),
        "updated_at": utc_now(),
    }


MATERIAL_PROPERTY_FIELDS = (
    "relative_permittivity", "permeability", "loss_tangent", "conductivity",
    "refractive_index", "dispersion_model", "frequency_range", "temperature",
)


def canonical_material_name(value: str) -> str:
    """Return a conservative registry key, or empty for prose noise."""
    candidate = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;()[]")
    if not candidate:
        return ""
    if ":" in candidate and re.search(r"(?:epsilon|eps|μ|mu|σ|sigma|permittivity|permeability|conductivity)", candidate, re.I):
        candidate = candidate.split(":", 1)[0].strip()
    lowered = candidate.lower()
    known = (
        (r"\bair\b|空气", "Air"),
        (r"\bvacuum\b|真空", "Vacuum"),
        (r"\bsilicon\b|\bsi\b|硅", "Silicon"),
        (r"\bsio2\b|二氧化硅", "SiO2"),
        (r"\bquartz\b|石英", "Quartz"),
        (r"\bptfe\b|聚四氟乙烯", "PTFE"),
        (r"\bcopper\b|铜", "Copper"),
        (r"\bpolycarbonate\b|聚碳酸酯", "Polycarbonate"),
        (r"\bsubstrate\b|基底|介质基板", "Substrate"),
        (r"\bpec\b|perfect electric conductor|理想电导体", "PEC"),
        (r"\bhuman head\b|人体头部", "Human-head tissue model"),
        (r"\bwater[- ]like\b|水样", "Water-like dispersive dielectric"),
    )
    for pattern, name in known:
        if re.search(pattern, lowered, re.I):
            return name
    noisy_prefixes = (
        "this document", "the proposed", "the tensors", "first,", "at least",
        "personal use", "the material parameters", "conductive regions",
        "maxwell’s equations", "maxwell's equations", "motion alters",
    )
    if len(candidate) > 96 or lowered.startswith(noisy_prefixes):
        return ""
    if candidate.count(" ") > 10 or re.search(r"[.!?]", candidate):
        return ""
    return candidate


def extract_material_properties(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only property values co-located with a named material in source evidence.

    Document-wide context windows caused unrelated frequencies and graph-axis
    values to be attributed to nearby material names.  Evidence is now bounded
    to a MinerU content block/sentence; comparisons and contrasts are not
    treated as a material's own property value.
    """
    raw_names = [item.strip() for item in str(summary.get("materials") or "").split(";") if item.strip()]
    names: list[tuple[str, str]] = []
    # Heuristic summaries sometimes attach several domains or parameter
    # expressions to one material label. Split only at a likely material-name
    # boundary, then discard parameter prose instead of registering it as a
    # material. Numeric values remain evidence in the Markdown context.
    material_boundary = re.compile(
        r",\s*(?=(?:air|silicon|sio2|quartz|ptfe|copper|gold|silver|water|ferrite|dielectric|substrate|pec)\b)",
        flags=re.IGNORECASE,
    )
    noise_start = re.compile(r"^(?:matching studied at|example params?|personal use|the material parameters)\b", re.I)
    property_hint = re.compile(r"(?:\b(?:eps|epsilon|mu|sigma|ms|h0|tau|alpha|beta)\b|ε|μ|σ)\s*[_{}^\w]*\s*=", re.I)
    for raw_name in raw_names:
        for part in material_boundary.split(raw_name):
            candidate = re.sub(r"\s+", " ", part).strip(" ,.;")
            lowered = candidate.casefold()
            if not candidate or noise_start.search(candidate):
                continue
            if "water-like" in lowered or "water like" in lowered:
                names.append(("Water-like dispersive dielectric", raw_name))
                continue
            # Remove a parenthesized parameter tail, including incomplete OCR
            # parentheses, but keep a real material name before it.
            opening = candidate.find("(")
            if opening >= 0 and property_hint.search(candidate[opening:]):
                candidate = candidate[:opening].strip(" ,.;")
            parameter_start = re.search(
                r"(?:relative\s+permittivity|dielectric\s+constant|permittivity|conductivity|permeability|"
                r"refractive\s+index|loss\s+tangent|epsilon|eps|mu|sigma|tan\s*δ|εr|μr|σ)"
                r"\s*[_{}^\w]*\s*(?:=|:|≈|~)",
                candidate,
                flags=re.IGNORECASE,
            )
            if parameter_start:
                candidate = candidate[:parameter_start.start()].strip(" ,.;:()[]")
            if property_hint.search(candidate) and len(candidate) > 32:
                candidate = re.split(r"\s+(?:with|and)\s+", candidate, maxsplit=1, flags=re.I)[0].strip(" ,.;")
            if candidate and not noise_start.search(candidate):
                canonical = canonical_material_name(candidate)
                if canonical and canonical.casefold() not in {"absorber", "absorbers", "dielectric rod", "dielectric material"}:
                    names.append((canonical, raw_name))
    markdown_path = Path(str(summary.get("markdown_path") or ""))
    if not names or not markdown_path.is_file():
        return []
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    try:
        evidence_ref = json.loads(str(summary.get("evidence_ref") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        evidence_ref = {}
    page_units: list[tuple[int | None, str]] = []
    content_list_path = Path(str(evidence_ref.get("content_list_path") or "")) if isinstance(evidence_ref, Mapping) else Path()
    if not content_list_path.is_file():
        candidates = sorted(markdown_path.parent.glob("*_content_list.json"))
        content_list_path = candidates[0] if candidates else Path()

    def add_evidence_units(raw_text: str, page: int | None) -> None:
        text = re.sub(r"(?i)<br\s*/?>|</(?:p|div|tr|li)\s*>", "\n", str(raw_text or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&nbsp;", " ").replace("&#160;", " ")
        for line in re.split(r"[\r\n]+", text):
            for sentence in re.split(r"(?<=[.!?。！？])\s+", line):
                unit = re.sub(r"\s+", " ", sentence).strip(" \t|;；")
                if unit and len(unit) <= 2400:
                    page_units.append((page, unit))

    if content_list_path.is_file():
        try:
            content_items = json.loads(content_list_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            content_items = []
        if isinstance(content_items, list):
            for item in content_items:
                if not isinstance(item, Mapping) or not str(item.get("text") or "").strip():
                    continue
                try:
                    page_number = int(item.get("page_idx")) + 1
                except (TypeError, ValueError):
                    page_number = None
                add_evidence_units(str(item.get("text") or ""), page_number)
    if not page_units:
        add_evidence_units(markdown, None)
    patterns: dict[str, tuple[str, ...]] = {
        "relative_permittivity": (r"(?:relative\s+)?permittivity", r"dielectric\s+constant", r"epsilon[_\s-]*r", r"eps[_\s-]*r", r"ε[_\s-]*r", r"epsilon"),
        # The registry stores relative permeability. A bare “permeability”
        # near μ0 is often an absolute vacuum constant, and OCR can mistake a
        # citation superscript for its numeric value. Require explicit μr
        # notation or a relative-property label.
        "permeability": (r"relative\s+permeability", r"mu[_\s-]*r", r"μ[_\s-]*r"),
        "loss_tangent": (r"loss\s+tangent", r"tan\s*δ", r"tan\s+delta"),
        "conductivity": (r"conductivity", r"sigma", r"σ"),
        "refractive_index": (r"refractive\s+index", r"index\s+of\s+refraction"),
    }
    numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    numeric_value = rf"{numeric}(?:\s*(?:±|to|[-–])\s*{numeric})?(?:\s*(?:S\s*/\s*m|S/m|\u03a9\s*\.?m|Ohm\s*\.?m|F\s*/\s*m|H\s*/\s*m|dB|%))?"
    frequency_value = rf"{numeric}\s*(?:kHz|MHz|GHz|THz|Hz)(?:\s*(?:to|[-–])\s*{numeric}\s*(?:kHz|MHz|GHz|THz|Hz))?"
    temperature_value = rf"-?{numeric}\s*(?:°\s*C|K|deg(?:ree)?s?\s*C)"
    rows: list[dict[str, Any]] = []

    def normalize_candidate(field: str, candidate: str) -> str:
        """Reject markup/placeholders while keeping explicit source values."""
        value = re.sub(r"\s+", " ", candidate).strip(" ,().")
        if not value or value.lower() in {"n/a", "na", "none", "unknown", "not specified"}:
            return ""
        if re.fullmatch(r"[${}\\]+", value) or re.match(r"^[${}\\]", value) or value in {"?", "--", "…"}:
            return ""
        if field == "dispersion_model":
            return value[:160] if re.search(r"drude|lorentz|sellmeier|debye|cole[- ]?cole|constant", value, re.IGNORECASE) else ""
        if field in {"relative_permittivity", "permeability", "loss_tangent", "conductivity", "refractive_index"}:
            return value[:160] if re.match(r"^[+-]?(?:\d|\.\d)", value) else ""
        return value[:160]

    def extract_after_terms(text: str, field: str, terms: tuple[str, ...], value_pattern: str) -> str:
        """Extract a value only when it is local to a property mention.

        The old implementation captured the rest of a sentence after a unit
        or property keyword.  That made prose such as ``... PEC ...`` look
        like a frequency value.  Require a numeric/value token on the same
        line and keep the evidence window deliberately small.
        """
        for term in terms:
            for term_match in re.finditer(term, text, flags=re.IGNORECASE):
                tail = text[term_match.end():].split("\n", 1)[0]
                tail = tail.split(";", 1)[0][:140]
                value_match = re.search(value_pattern, tail, flags=re.IGNORECASE)
                if not value_match:
                    continue
                raw_value = value_match.group(0).strip()
                suffix = tail[value_match.end():]
                if raw_value.endswith(".") or re.match(r"\s*\.\s*\d", suffix) or re.match(r"\s+\.\s*(?:\d|$)", suffix):
                    continue
                if field in {"relative_permittivity", "permeability", "loss_tangent", "refractive_index"}:
                    units = r"(?:nm|μm|µm|um|mm|cm|m|hz|khz|mhz|ghz|thz|s\s*/\s*m|dB|%)"
                    if re.search(units, raw_value, flags=re.IGNORECASE) or re.match(rf"\s*{units}\b", suffix, flags=re.IGNORECASE):
                        continue
                # A distant number is usually an unrelated number in the
                # same sentence.  Accept it only with an explicit relation.
                prefix = tail[:value_match.start()]
                if re.search(r"(?:\$\$|\\[A-Za-z]+)", prefix):
                    continue
                if ("[" in prefix and "]" not in prefix.rsplit("[", 1)[-1]) or tail[value_match.end():].lstrip().startswith("]"):
                    continue
                if re.search(r"\b(?:contrast|difference|ratio|compared|between|on the order|greater than|less than|up to)\b", prefix, flags=re.IGNORECASE):
                    continue
                related = re.search(r"(?:=|:|≈|~|\bis\b|\bwas\b|\bwere\b|\bof\b)\s*$", prefix[-56:], flags=re.IGNORECASE)
                window_start = max(0, len(prefix) - 56)
                if value_match.start() <= 20 or (related and value_match.start() - (window_start + related.start()) <= 28):
                    return normalize_candidate(field, raw_value)
        return ""

    def material_mentioned(text: str, name: str) -> bool:
        escaped = re.escape(name)
        if re.fullmatch(r"[A-Za-z0-9+.-]+", name):
            return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, flags=re.IGNORECASE))
        return bool(re.search(escaped, text, flags=re.IGNORECASE))

    unique_names: dict[str, list[str]] = {}
    for name, raw_name in names:
        unique_names.setdefault(name, []).append(raw_name)

    for name in unique_names:
        name_units = [(page, text) for page, text in page_units if material_mentioned(text, name)]
        candidate_values: dict[str, dict[str, list[tuple[int | None, str]]]] = {field: {} for field in MATERIAL_PROPERTY_FIELDS}
        for page, text in name_units:
            for field, terms in patterns.items():
                candidate = extract_after_terms(text, field, terms, rf"({numeric_value})")
                if candidate:
                    candidate_values[field].setdefault(candidate, []).append((page, text))
            frequency_context = re.search(r"\b(?:frequency\s+range|frequency\s+of|measured\s+at)\b", text, flags=re.IGNORECASE)
            if frequency_context:
                value = re.search(rf"(?<![\w.])({frequency_value})\b", text[frequency_context.start():], flags=re.IGNORECASE)
                if value:
                    normalized = normalize_candidate("frequency_range", value.group(1))
                    if normalized:
                        candidate_values["frequency_range"].setdefault(normalized, []).append((page, text))
            temperature_context = re.search(r"\b(?:temperature|measured\s+at|at\s+a\s+temperature\s+of)\b", text, flags=re.IGNORECASE)
            if temperature_context:
                value = re.search(rf"(?<![\w.])({temperature_value})\b", text[temperature_context.start():], flags=re.IGNORECASE)
                if value:
                    normalized = normalize_candidate("temperature", value.group(1))
                    if normalized:
                        candidate_values["temperature"].setdefault(normalized, []).append((page, text))
            model = re.search(r"\b(?:dispersion\s+model\s*(?:is|=|:)?\s*)?(Drude|Lorentz|Sellmeier|Debye|Cole[- ]?Cole)\b", text, flags=re.IGNORECASE)
            if model and re.search(r"\b(?:dispersion|model|modeled|modelled|fit|fitted)\b", text, flags=re.IGNORECASE):
                normalized = normalize_candidate("dispersion_model", model.group(1))
                if normalized:
                    candidate_values["dispersion_model"].setdefault(normalized, []).append((page, text))

        values: dict[str, str] = {}
        conflicts: list[str] = []
        evidence_snippets: list[str] = []
        evidence_pages: list[int] = []
        for field, options in candidate_values.items():
            if len(options) == 1:
                value, evidence = next(iter(options.items()))
                values[field] = value
                evidence_snippets.extend(text for _, text in evidence)
                evidence_pages.extend(page for page, _ in evidence if page is not None)
            elif len(options) > 1:
                conflicts.append(field)
                for evidence in options.values():
                    evidence_snippets.extend(text for _, text in evidence)
                    evidence_pages.extend(page for page, _ in evidence if page is not None)
        if not evidence_snippets:
            evidence_snippets = [text for _, text in name_units[:3]]
            evidence_pages.extend(page for page, _ in name_units if page is not None)
        rows.append({
            "doc_id": summary.get("doc_id", ""),
            "material_name": name,
            **{field: values.get(field, "") for field in MATERIAL_PROPERTY_FIELDS},
            "evidence_pages": json.dumps(sorted(set(evidence_pages)), ensure_ascii=False),
            "evidence_text": "\n\n".join(dict.fromkeys(evidence_snippets))[:1200],
            "source_pdf": evidence_ref.get("original_pdf", "") if isinstance(evidence_ref, Mapping) else "",
            "extraction_status": "conflict_in_source" if conflicts else "property_observed" if values else "identified_only",
            "updated_at": utc_now(),
        })
    return rows


class RAGFlowClient:
    def __init__(self, base_url: str, dataset_id: str, api_key_env: str, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.api_key_env = api_key_env
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        key = resolve_api_secret(self.api_key_env)
        if not key:
            raise RuntimeError(
                "RAGFlow API key is not configured via environment or inline config: "
                f"{secret_selector_label(self.api_key_env)}"
            )
        return {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    def list_datasets(self, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]:
        url = join_url(self.base_url, "/api/v1/datasets") + "?" + urllib.parse.urlencode({"page": page, "page_size": page_size})
        payload = http_json(url, headers=self._headers(), timeout=self.timeout)
        data = payload.get("data") if isinstance(payload, Mapping) else []
        if isinstance(data, Mapping):
            for key in ("kbs", "datasets", "docs"):
                if isinstance(data.get(key), list):
                    return [dict(item) for item in data[key] if isinstance(item, Mapping)]
            return [dict(data)] if data.get("id") else []
        return [dict(item) for item in (data or []) if isinstance(item, Mapping)]

    def create_dataset(
        self,
        name: str,
        embedding_model: str,
        description: str = "",
        chunk_method: str = "naive",
        parser_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a dataset with an explicit, server-side embedding model ID."""
        if not embedding_model or embedding_model.startswith("FILL_"):
            raise ValueError("创建 RAGFlow 数据集前必须填写服务端实际 embedding_model 标识")
        payload = {
            "name": name,
            "description": description,
            "embedding_model": embedding_model,
            "permission": "me",
            "chunk_method": chunk_method,
        }
        if parser_config is not None:
            payload["parser_config"] = dict(parser_config)
        response = http_json(
            join_url(self.base_url, "/api/v1/datasets"),
            method="POST",
            payload=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        data = response.get("data") if isinstance(response, Mapping) else None
        if not isinstance(data, Mapping) or not data.get("id"):
            raise RuntimeError(f"RAGFlow 创建数据集未返回 ID: {response}")
        return dict(data)

    @staticmethod
    def _multipart(file_path: Path, display_name: str | None = None) -> tuple[bytes, str]:
        boundary = "----CodexBoundary" + uuid.uuid4().hex
        content = file_path.read_bytes()
        name = display_name or file_path.name
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("ascii")
        return body, f"multipart/form-data; boundary={boundary}"

    def upload(self, file_path: Path, display_name: str | None = None) -> str:
        body, content_type = self._multipart(file_path, display_name)
        headers = self._headers()
        headers["Content-Type"] = content_type
        _, raw, _ = http_request(
            join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/documents"),
            method="POST",
            body=body,
            headers=headers,
            timeout=self.timeout,
        )
        payload = json.loads(raw.decode("utf-8"))
        data = payload.get("data") or []
        if isinstance(data, dict):
            data = [data]
        if not data or not data[0].get("id"):
            raise RuntimeError(f"RAGFlow upload returned no document id: {payload}")
        return str(data[0]["id"])

    def list_documents(self, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]:
        """List documents in this dataset for safe retry reconciliation."""
        page_size = max(1, min(int(page_size), 100))
        url = join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/documents")
        url += "?" + urllib.parse.urlencode({"page": max(1, int(page)), "page_size": page_size})
        payload = http_json(url, headers=self._headers(), timeout=self.timeout)
        data = payload.get("data") if isinstance(payload, Mapping) else []
        if isinstance(data, Mapping):
            documents = data.get("docs") or data.get("documents") or data.get("data") or []
        else:
            documents = data or []
        return [dict(item) for item in documents if isinstance(item, Mapping)]

    def find_active_document(self, display_name: str, max_pages: int = 100) -> dict[str, Any] | None:
        """Find an exact or server-renamed duplicate, ignoring deleted records.

        RAGFlow appends ``(N)`` to a duplicate filename. Exact matching alone
        made every retry look new and caused unbounded duplicate uploads.
        Prefer an exact canonical name, then reuse the first active collision.
        """
        page_size = 100
        canonical_name = canonical_ragflow_name(display_name)
        collision_matches: list[dict[str, Any]] = []
        for page in range(1, max(1, int(max_pages)) + 1):
            documents = self.list_documents(page=page, page_size=page_size)
            for document in documents:
                name = str(document.get("name") or "")
                if not document.get("id"):
                    continue
                status = str(document.get("status") or "").strip().lower()
                if status in {"0", "disabled", "deleted", "deleting"}:
                    continue
                if name == display_name:
                    return document
                if canonical_ragflow_name(name) == canonical_name:
                    collision_matches.append(document)
            if len(documents) < page_size:
                break
        if not collision_matches:
            return None
        collision_matches.sort(
            key=lambda item: (
                str(item.get("run") or "").upper() == "DONE",
                int(item.get("chunk_count") or 0) > 0,
                str(item.get("name") or ""),
            ),
            reverse=True,
        )
        return collision_matches[0]

    def trigger_parse(self, document_id: str) -> None:
        response = http_json(
            join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/chunks"),
            method="POST",
            payload={"document_ids": [document_id]},
            headers=self._headers(),
            timeout=self.timeout,
        )
        if isinstance(response, Mapping) and response.get("code") not in (None, 0, "0"):
            raise RuntimeError(f"RAGFlow 文档解析任务未能启动：{response}")

    def update_document(self, document_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Update per-document parsing before triggering RAGFlow ingestion."""
        response = http_json(
            join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/documents/{document_id}"),
            method="PUT",
            payload=dict(payload),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if isinstance(response, Mapping) and response.get("code") not in (None, 0, "0"):
            raise RuntimeError(f"RAGFlow 文档解析配置被拒绝：{response}")
        requested_config = payload.get("parser_config") if isinstance(payload, Mapping) else None
        requested_layout = requested_config.get("layout_recognize") if isinstance(requested_config, Mapping) else None
        if requested_layout:
            stored = self.document(document_id)
            stored_config = stored.get("parser_config") if isinstance(stored, Mapping) else None
            stored_layout = stored_config.get("layout_recognize") if isinstance(stored_config, Mapping) else None
            if stored_layout and str(stored_layout) != str(requested_layout):
                raise RuntimeError(
                    "RAGFlow 未保存请求的 PDF 解析器："
                    f"requested={requested_layout}, stored={stored_layout}"
                )
        return dict(response) if isinstance(response, Mapping) else {"data": response}

    def list_chunks(self, document_id: str, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]:
        page_size = max(1, min(int(page_size), 100))
        all_chunks: list[dict[str, Any]] = []
        current_page = max(1, int(page))
        while True:
            payload = self.list_chunks_response(document_id, page=current_page, page_size=page_size)
            data = payload.get("data") if isinstance(payload, Mapping) else []
            if isinstance(data, Mapping):
                chunks = data.get("chunks") or data.get("docs") or []
                total = int(data.get("total") or 0)
            else:
                chunks = data or []
                total = 0
            batch = [dict(item) for item in chunks if isinstance(item, Mapping)]
            all_chunks.extend(batch)
            if not batch or len(batch) < page_size or (total and len(all_chunks) >= total):
                return all_chunks
            current_page += 1

    def list_chunks_response(self, document_id: str, page: int = 1, page_size: int = 100) -> dict[str, Any]:
        """Return the raw chunk-list response for diagnostics and audit trails."""
        url = join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/documents/{document_id}/chunks")
        url += "?" + urllib.parse.urlencode({"page": page, "page_size": page_size})
        payload = http_json(url, headers=self._headers(), timeout=self.timeout)
        return dict(payload) if isinstance(payload, Mapping) else {"raw": payload}

    def delete_documents(self, document_ids: list[str]) -> dict[str, Any]:
        if not document_ids:
            return {"code": 0}
        return http_json(
            join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/documents"),
            method="DELETE",
            payload={"ids": list(document_ids)},
            headers=self._headers(),
            timeout=self.timeout,
        )

    def document(self, document_id: str) -> dict[str, Any]:
        url = join_url(self.base_url, f"/api/v1/datasets/{self.dataset_id}/documents")
        url += "?" + urllib.parse.urlencode({"id": document_id, "page": 1, "page_size": 10})
        payload = http_json(url, headers=self._headers(), timeout=self.timeout)
        data = payload.get("data") or []
        if isinstance(data, dict):
            # RAGFlow v0.27 returns {data: {docs: [...], total: ...}}.
            docs = data.get("docs")
            if isinstance(docs, list):
                return docs[0] if docs else {}
            return data
        if data:
            return data[0]
        return {}

    def retrieve(self, question: str, top_k: int = 10, similarity_threshold: float = 0.2, vector_similarity_weight: float = 0.5, keyword: bool = True, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Call the dataset-scoped retrieval endpoint and return ranked chunks."""
        payload: dict[str, Any] = {
            "question": question,
            "dataset_ids": [self.dataset_id],
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "keyword": keyword,
        }
        if doc_ids:
            payload["doc_ids"] = doc_ids
        response = http_json(
            join_url(self.base_url, "/api/v1/retrieval"),
            method="POST",
            payload=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        data = response.get("data") if isinstance(response, Mapping) else None
        chunks = data.get("chunks", []) if isinstance(data, Mapping) else []
        return [dict(chunk) for chunk in chunks if isinstance(chunk, Mapping)]

    def wait_done(self, document_id: str, timeout: float = 1800, interval: float = 5) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.document(document_id)
            run = str(status.get("run") or status.get("status") or "")
            if run in {"DONE", "3"}:
                # RAGFlow can report DONE even when a parser plugin failed
                # before producing chunks. Treat that as a failed ingestion so
                # callers can invoke their configured fallback instead of
                # publishing an unusable zero-chunk document.
                if "chunk_count" in status and int(status.get("chunk_count") or 0) == 0:
                    raise RuntimeError(f"RAGFlow parsing finished with zero chunks for {document_id}: {status.get('progress_msg') or 'empty parser output'}")
                return status
            if run in {"FAIL", "4", "CANCEL", "2"}:
                raise RuntimeError(f"RAGFlow parsing failed for {document_id}: {status}")
            time.sleep(interval)
        raise TimeoutError(f"RAGFlow parsing timed out for {document_id}")


class PaperIndex:
    def __init__(self, db_path: Path, summary_csv: Path, material_catalog_csv: Path | None = None):
        self.db_path = db_path
        self.summary_csv = summary_csv
        self.material_properties_csv = summary_csv.with_name("material_properties.csv")
        self.material_registry_csv = summary_csv.with_name("material_registry.csv")
        self.material_catalog_csv = summary_csv.with_name("material_catalog.csv")
        default_catalog = self.material_catalog_csv if self.material_catalog_csv.is_file() else None
        self.source_material_catalog_csv = material_catalog_csv or default_catalog
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
              doc_id TEXT PRIMARY KEY, zotero_key TEXT NOT NULL, parent_key TEXT,
              title TEXT, authors TEXT, year TEXT, doi TEXT, tags TEXT, filename TEXT,
              abstract TEXT, landing_url TEXT, pdf_url TEXT, source_provider TEXT,
              sha256 TEXT NOT NULL UNIQUE, source_path TEXT, raw_path TEXT,
              mineru_status TEXT, mineru_markdown_path TEXT, ragflow_document_id TEXT,
              ragflow_pdf_document_id TEXT,
              ragflow_dataset_id TEXT, ragflow_parser_config TEXT, selection_score REAL,
              ragflow_upload_attempts INTEGER NOT NULL DEFAULT 0,
              ragflow_pdf_upload_attempts INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL, last_error TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS simulation_summaries (
              doc_id TEXT PRIMARY KEY, zotero_key TEXT, title TEXT, doi TEXT, authors TEXT,
              year TEXT, target_wavelength TEXT, frequency_range TEXT, materials TEXT,
              geometry TEXT, solver TEXT, mesh TEXT, cfl_timestep TEXT, boundary_pml TEXT,
              source TEXT, monitors TEXT, stop_condition TEXT, reported_metrics TEXT,
              quality_score REAL, extraction_status TEXT, evidence_ref TEXT,
              markdown_path TEXT, notes TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS literature_candidates (
              candidate_id TEXT PRIMARY KEY, query TEXT, source TEXT, title TEXT,
              authors TEXT, year TEXT, doi TEXT, landing_url TEXT, pdf_url TEXT, abstract TEXT,
              priority_score REAL, status TEXT NOT NULL, doc_id TEXT, decision_json TEXT, last_error TEXT,
              created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS discovery_runs (
              run_id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL,
              providers TEXT, status TEXT NOT NULL, candidate_count INTEGER NOT NULL DEFAULT 0,
              error TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS review_results (
              doc_id TEXT PRIMARY KEY, accepted INTEGER NOT NULL, relevance_score REAL,
              quality_score REAL, reason TEXT, missing_fields TEXT, raw_json TEXT,
              created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS chunk_quality (
              doc_id TEXT PRIMARY KEY, chunk_count INTEGER, coverage REAL,
              boundary_integrity REAL, structure_integrity REAL, score REAL,
              violations TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS material_properties (
              doc_id TEXT NOT NULL, material_name TEXT NOT NULL,
              relative_permittivity TEXT, permeability TEXT, loss_tangent TEXT,
              conductivity TEXT, refractive_index TEXT, dispersion_model TEXT,
              frequency_range TEXT, temperature TEXT, evidence_pages TEXT,
              evidence_text TEXT, source_pdf TEXT, extraction_status TEXT,
              updated_at TEXT, PRIMARY KEY (doc_id, material_name)
            );
            CREATE TABLE IF NOT EXISTS material_registry (
              material_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL,
              relative_permittivity TEXT, permeability TEXT, loss_tangent TEXT,
              conductivity TEXT, refractive_index TEXT, dispersion_model TEXT,
              frequency_range TEXT, temperature TEXT, source_count INTEGER NOT NULL DEFAULT 0,
              source_doc_ids TEXT, evidence_pages TEXT, property_variants TEXT,
              evidence_status TEXT NOT NULL, conflict_fields TEXT, updated_at TEXT,
              data_types TEXT, wavelength_ranges TEXT, conditions TEXT, source_references TEXT,
              source_files TEXT, source_archives TEXT, comments TEXT, numeric_rows TEXT,
              data_previews TEXT, card_texts TEXT, catalog_record_count INTEGER NOT NULL DEFAULT 0,
              parameter_coverage TEXT, evidence_locator TEXT
            );
            CREATE TABLE IF NOT EXISTS material_catalog (
              record_id TEXT PRIMARY KEY, material_name TEXT NOT NULL,
              source_file TEXT, source_archive TEXT, data_types TEXT,
              wavelength_ranges TEXT, comments TEXT, conditions TEXT, source_references TEXT,
              numeric_rows TEXT, data_preview TEXT, card_text TEXT, source_path TEXT,
              evidence_status TEXT NOT NULL, updated_at TEXT
            );
            """
        )
        try:
            self.connection.execute("ALTER TABLE papers ADD COLUMN ragflow_dataset_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.connection.execute("ALTER TABLE papers ADD COLUMN ragflow_parser_config TEXT")
        except sqlite3.OperationalError:
            pass
        for column in ("ragflow_upload_attempts", "ragflow_pdf_upload_attempts"):
            try:
                self.connection.execute(f"ALTER TABLE papers ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
        try:
            self.connection.execute("ALTER TABLE papers ADD COLUMN ragflow_pdf_document_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.connection.execute("ALTER TABLE literature_candidates ADD COLUMN priority_score REAL")
        except sqlite3.OperationalError:
            pass
        try:
            self.connection.execute("ALTER TABLE papers ADD COLUMN selection_score REAL")
        except sqlite3.OperationalError:
            pass
        for column in ("abstract", "landing_url", "pdf_url", "source_provider"):
            try:
                self.connection.execute(f"ALTER TABLE papers ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass
        try:
            self.connection.execute("ALTER TABLE literature_candidates ADD COLUMN abstract TEXT")
        except sqlite3.OperationalError:
            pass
        registry_columns = {
            "data_types": "TEXT", "wavelength_ranges": "TEXT", "conditions": "TEXT",
            "source_references": "TEXT", "source_files": "TEXT", "source_archives": "TEXT",
            "comments": "TEXT", "numeric_rows": "TEXT", "data_previews": "TEXT",
            "card_texts": "TEXT", "catalog_record_count": "INTEGER NOT NULL DEFAULT 0",
            "parameter_coverage": "TEXT", "evidence_locator": "TEXT",
        }
        for column, column_type in registry_columns.items():
            try:
                self.connection.execute(f"ALTER TABLE material_registry ADD COLUMN {column} {column_type}")
            except sqlite3.OperationalError:
                pass
        self.connection.commit()
        if self.source_material_catalog_csv and self.source_material_catalog_csv.is_file():
            self.refresh_material_registry()

    def close(self) -> None:
        self.connection.close()

    def find_sha(self, digest: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM papers WHERE sha256 = ?", (digest,)).fetchone()

    def find_doi(self, doi: str) -> sqlite3.Row | None:
        if not doi:
            return None
        return self.connection.execute("SELECT * FROM papers WHERE lower(doi) = lower(?)", (doi,)).fetchone()

    def find_candidate(self, candidate_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM literature_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()

    def find_title(self, title: str, year: str = "") -> sqlite3.Row | None:
        """Find an existing paper by normalized title, tolerating missing years."""
        key = normalise_title(title)
        if not key:
            return None
        target_year = str(year or "")[:4]
        rows = self.connection.execute("SELECT * FROM papers ORDER BY rowid").fetchall()
        for row in rows:
            if normalise_title(str(row["title"] or "")) != key:
                continue
            existing_year = str(row["year"] or "")[:4]
            if target_year and existing_year and target_year != existing_year:
                continue
            if str(row["status"] or "") != "duplicate_existing":
                return row
        return None

    def mark_title_duplicates(self) -> list[dict[str, str]]:
        """Mark later same-title records as duplicates without deleting evidence."""
        rows = self.connection.execute("SELECT rowid, * FROM papers ORDER BY rowid").fetchall()
        canonical: dict[str, sqlite3.Row] = {}
        duplicates: list[dict[str, str]] = []
        for row in rows:
            if str(row["status"] or "") == "duplicate_existing":
                continue
            key = normalise_title(str(row["title"] or ""))
            if not key:
                continue
            previous = canonical.get(key)
            if previous is None:
                canonical[key] = row
                continue
            previous_year = str(previous["year"] or "")[:4]
            current_year = str(row["year"] or "")[:4]
            if previous_year and current_year and previous_year != current_year:
                continue
            reason = f"标题与已有文献重复，保留规范记录 {previous['doc_id']}；未删除原始证据。"
            self.connection.execute(
                "UPDATE papers SET status = ?, last_error = ?, updated_at = ? WHERE doc_id = ?",
                ("duplicate_existing", reason, utc_now(), row["doc_id"]),
            )
            self.connection.execute(
                "UPDATE literature_candidates SET status = ?, doc_id = ?, last_error = ?, updated_at = ? WHERE doc_id = ?",
                ("duplicate_existing", previous["doc_id"], reason, utc_now(), row["doc_id"]),
            )
            duplicates.append({"doc_id": str(row["doc_id"]), "canonical_doc_id": str(previous["doc_id"]), "reason": reason})
        self.connection.commit()
        return duplicates

    def upsert_candidate(self, candidate: LiteratureCandidate, status: str, **updates: Any) -> None:
        row = {
            "candidate_id": candidate.candidate_id,
            "query": candidate.query,
            "source": candidate.source,
            "title": candidate.title,
            "authors": candidate.authors,
            "year": candidate.year,
            "doi": candidate.doi,
            "landing_url": candidate.landing_url,
            "pdf_url": candidate.pdf_url,
            "abstract": candidate.abstract,
            "priority_score": candidate_priority(candidate),
            "status": status,
            "doc_id": "",
            "decision_json": "",
            "last_error": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        row.update(updates)
        self.connection.execute(
            "INSERT INTO literature_candidates (" + ",".join(row) + ") VALUES (" + ",".join("?" for _ in row) + ") "
            "ON CONFLICT(candidate_id) DO UPDATE SET "
            + ",".join(f"{field}=excluded.{field}" for field in row if field != "candidate_id"),
            list(row.values()),
        )
        self.connection.commit()

    def record_discovery_run(
        self,
        query: str,
        providers: Iterable[str],
        status: str,
        candidate_count: int = 0,
        error: str = "",
    ) -> None:
        self.connection.execute(
            "INSERT INTO discovery_runs (query, providers, status, candidate_count, error, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (query, json.dumps(list(providers), ensure_ascii=False), status, int(candidate_count), error, utc_now()),
        )
        self.connection.commit()

    def upsert_review(self, doc_id: str, review: Mapping[str, Any]) -> None:
        missing = review.get("missing_fields", [])
        missing_json = json.dumps(missing, ensure_ascii=False) if isinstance(missing, list) else str(missing)
        self.connection.execute(
            "INSERT INTO review_results (doc_id, accepted, relevance_score, quality_score, reason, missing_fields, raw_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET "
            "accepted=excluded.accepted, relevance_score=excluded.relevance_score, quality_score=excluded.quality_score, "
            "reason=excluded.reason, missing_fields=excluded.missing_fields, raw_json=excluded.raw_json, updated_at=excluded.updated_at",
            (
                doc_id,
                int(bool(review.get("accepted"))),
                float(review.get("relevance_score", 0) or 0),
                float(review.get("quality_score", 0) or 0),
                str(review.get("reason", "")),
                missing_json,
                json.dumps(dict(review), ensure_ascii=False),
                utc_now(),
                utc_now(),
            ),
        )
        self.connection.commit()

    def upsert_chunk_quality(self, doc_id: str, report: Mapping[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO chunk_quality (doc_id, chunk_count, coverage, boundary_integrity, structure_integrity, score, violations, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET "
            "chunk_count=excluded.chunk_count, coverage=excluded.coverage, boundary_integrity=excluded.boundary_integrity, "
            "structure_integrity=excluded.structure_integrity, score=excluded.score, violations=excluded.violations, updated_at=excluded.updated_at",
            (
                doc_id,
                int(report.get("chunk_count", 0) or 0),
                float(report.get("coverage", 0) or 0),
                float(report.get("boundary_integrity", 0) or 0),
                float(report.get("structure_integrity", 0) or 0),
                float(report.get("score", 0) or 0),
                json.dumps(report.get("violations", []), ensure_ascii=False),
                utc_now(),
            ),
        )
        self.connection.commit()

    def upsert_paper(self, row: dict[str, Any]) -> None:
        fields = [
            "doc_id", "zotero_key", "parent_key", "title", "authors", "year", "doi", "tags",
            "filename", "abstract", "landing_url", "pdf_url", "source_provider", "sha256", "source_path", "raw_path", "mineru_status", "mineru_markdown_path",
            "ragflow_document_id", "ragflow_pdf_document_id", "ragflow_dataset_id", "ragflow_parser_config", "selection_score",
            "ragflow_upload_attempts", "ragflow_pdf_upload_attempts", "status", "last_error", "created_at", "updated_at",
        ]
        attempt_fields = {"ragflow_upload_attempts", "ragflow_pdf_upload_attempts"}
        values = [int(row.get(field) or 0) if field in attempt_fields else row.get(field) for field in fields]
        self.connection.execute(
            f"INSERT INTO papers ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)}) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            + ",".join(f"{field}=excluded.{field}" for field in fields if field != "doc_id"),
            values,
        )
        self.connection.commit()

    def upsert_summary(self, row: dict[str, Any]) -> None:
        fields = [
            "doc_id", "zotero_key", "title", "doi", "authors", "year", "target_wavelength",
            "frequency_range", "materials", "geometry", "solver", "mesh", "cfl_timestep",
            "boundary_pml", "source", "monitors", "stop_condition", "reported_metrics",
            "quality_score", "extraction_status", "evidence_ref", "markdown_path", "notes", "updated_at",
        ]
        self.connection.execute(
            f"INSERT INTO simulation_summaries ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)}) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            + ",".join(f"{field}=excluded.{field}" for field in fields if field != "doc_id"),
            [row.get(field) for field in fields],
        )
        self.connection.execute("DELETE FROM material_properties WHERE doc_id = ?", (row.get("doc_id"),))
        material_rows = extract_material_properties(row)
        material_fields = [
            "doc_id", "material_name", "relative_permittivity", "permeability", "loss_tangent",
            "conductivity", "refractive_index", "dispersion_model", "frequency_range", "temperature",
            "evidence_pages", "evidence_text", "source_pdf", "extraction_status", "updated_at",
        ]
        for material_row in material_rows:
            self.connection.execute(
                f"INSERT INTO material_properties ({','.join(material_fields)}) VALUES ({','.join('?' for _ in material_fields)})",
                [material_row.get(field, "") for field in material_fields],
            )
        self.connection.commit()
        self.refresh_material_registry()

    def rebuild_material_observations(self) -> int:
        """Re-extract material observations from all stored summaries.

        This is intentionally separate from ``upsert_summary`` so that parser
        and name-normalization improvements can be applied to the existing
        library without re-running MinerU or changing the paper summaries.
        The source summary remains the evidence boundary; no values are
        inferred from missing fields.
        """
        summaries = self.connection.execute(
            "SELECT * FROM simulation_summaries ORDER BY doc_id"
        ).fetchall()
        material_fields = [
            "doc_id", "material_name", "relative_permittivity", "permeability", "loss_tangent",
            "conductivity", "refractive_index", "dispersion_model", "frequency_range", "temperature",
            "evidence_pages", "evidence_text", "source_pdf", "extraction_status", "updated_at",
        ]
        rows_to_insert: list[list[Any]] = []
        for summary in summaries:
            for material_row in extract_material_properties(dict(summary)):
                rows_to_insert.append([material_row.get(field, "") for field in material_fields])
        self.connection.execute("DELETE FROM material_properties")
        if rows_to_insert:
            self.connection.executemany(
                f"INSERT INTO material_properties ({','.join(material_fields)}) VALUES ({','.join('?' for _ in material_fields)})",
                rows_to_insert,
            )
        self.connection.commit()
        self.refresh_material_registry()
        return len(rows_to_insert)

    def import_material_catalog(self) -> int:
        """Idempotently import the existing G: material-card table.

        The catalog is treated as source-backed evidence, not as a manually
        verified truth.  Optical wavelength data is kept in its own field and
        is never converted to an electromagnetic frequency implicitly.
        """
        path = self.source_material_catalog_csv
        if not path or not path.is_file():
            return 0
        fields = (
            "record_id", "material_name", "source_file", "source_archive", "data_types",
            "wavelength_ranges", "comments", "conditions", "source_references", "numeric_rows",
            "data_preview", "card_text", "source_path", "evidence_status", "updated_at",
        )
        count = 0
        # This table is a materialized copy of the configured catalog source.
        # Rebuild only rows originating from this source (and its exported
        # mirror), so repeated worker entry points remain idempotent.
        source_paths = (str(path), str(self.material_catalog_csv))
        self.connection.execute(
            "DELETE FROM material_catalog WHERE source_path IN (?, ?)",
            tuple(source_paths),
        )
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as stream:
            reader = csv.DictReader(stream)
            for raw in reader:
                material_name = str(raw.get("material_name") or "").strip()
                source_file = str(raw.get("source_file") or "").strip()
                source_archive = str(raw.get("source_archive") or "").strip()
                references = str(raw.get("references") or "").strip()
                seed = str(raw.get("record_id") or raw.get("material_id") or "").strip()
                record_id = seed or hashlib.sha1(
                    "|".join((source_file, material_name, references)).encode("utf-8", errors="replace")
                ).hexdigest()
                if not material_name:
                    continue
                row = {
                    "record_id": record_id,
                    "material_name": material_name,
                    "source_file": source_file,
                    "source_archive": source_archive,
                    "data_types": str(raw.get("data_types") or "").strip(),
                    "wavelength_ranges": str(raw.get("wavelength_ranges") or "").strip(),
                    "comments": str(raw.get("comments") or "").strip(),
                    "conditions": str(raw.get("conditions") or "").strip(),
                    "source_references": references,
                    "numeric_rows": str(raw.get("numeric_rows") or "").strip(),
                    "data_preview": str(raw.get("data_preview") or "").strip(),
                    "card_text": str(raw.get("card_text") or "").strip(),
                    "source_path": str(path),
                    "evidence_status": "catalog_source_supported",
                    "updated_at": utc_now(),
                }
                self.connection.execute(
                    "INSERT INTO material_catalog (" + ",".join(fields) + ") VALUES (" + ",".join("?" for _ in fields) + ") "
                    "ON CONFLICT(record_id) DO UPDATE SET "
                    + ",".join(f"{field}=excluded.{field}" for field in fields if field != "record_id"),
                    [row[field] for field in fields],
                )
                count += 1
        self.connection.commit()
        return count

    def refresh_material_registry(self) -> None:
        """Rebuild the deduplicated material view without overwriting conflicts."""
        self.import_material_catalog()
        rows = self.connection.execute("SELECT * FROM material_properties ORDER BY doc_id, material_name").fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            canonical = canonical_material_name(str(row["material_name"] or ""))
            if not canonical:
                continue
            group_key = canonical.casefold()
            item = grouped.setdefault(group_key, {
                "canonical_name": canonical,
                "docs": set(), "pages": {}, "variants": {field: set() for field in MATERIAL_PROPERTY_FIELDS},
            })
            item["docs"].add(str(row["doc_id"] or ""))
            try:
                page_values = json.loads(str(row["evidence_pages"] or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                page_values = []
            item["pages"].setdefault(str(row["doc_id"] or ""), page_values)
            for field in MATERIAL_PROPERTY_FIELDS:
                value = str(row[field] or "").strip()
                if value:
                    item["variants"][field].add(value)

        catalog_rows = self.connection.execute("SELECT * FROM material_catalog ORDER BY material_name, record_id").fetchall()
        catalog_fields = (
            "data_types", "wavelength_ranges", "conditions", "source_references", "source_file",
            "source_archive", "comments", "numeric_rows", "data_preview", "card_text",
        )
        for row in catalog_rows:
            canonical = canonical_material_name(str(row["material_name"] or ""))
            if not canonical:
                continue
            group_key = canonical.casefold()
            item = grouped.setdefault(group_key, {
                "canonical_name": canonical,
                "docs": set(), "pages": {}, "variants": {field: set() for field in MATERIAL_PROPERTY_FIELDS},
                "catalog": {field: set() for field in catalog_fields}, "catalog_ids": set(), "catalog_locators": {},
            })
            item.setdefault("canonical_name", canonical)
            item.setdefault("catalog", {field: set() for field in catalog_fields})
            item.setdefault("catalog_ids", set())
            item.setdefault("catalog_locators", {})
            item["catalog_ids"].add(str(row["record_id"] or ""))
            item["catalog_locators"][str(row["record_id"] or "")] = {
                "source_file": str(row["source_file"] or ""),
                "source_archive": str(row["source_archive"] or ""),
                "page": "not_applicable_catalog_record",
            }
            for field in catalog_fields:
                value = str(row[field] or "").strip()
                if value:
                    item["catalog"][field].add(value)

        self.connection.execute("DELETE FROM material_registry")
        for _, item in grouped.items():
            canonical = str(item.get("canonical_name") or "")
            if not canonical:
                continue
            variants = {field: sorted(values) for field, values in item["variants"].items() if values}
            conflicts = sorted(field for field, values in variants.items() if len(values) > 1)
            values = {field: (vals[0] if len(vals) == 1 else "") for field, vals in variants.items()}
            catalog = item.get("catalog", {})
            catalog_values = {
                field: sorted(values) for field, values in catalog.items() if values
            }
            if conflicts:
                status = "conflict_needs_review"
            elif catalog_values:
                status = "catalog_source_supported"
            elif variants:
                status = "source_supported"
            else:
                status = "identified_only"
            material_id = hashlib.sha1(canonical.lower().encode("utf-8")).hexdigest()[:16]
            coverage = {}
            for field in MATERIAL_PROPERTY_FIELDS:
                if field in variants:
                    coverage[field] = "conflict" if len(variants[field]) > 1 else "provided"
                else:
                    coverage[field] = "not_reported"
            if coverage["frequency_range"] == "not_reported" and catalog_values.get("wavelength_ranges"):
                coverage["frequency_range"] = "wavelength_only_not_frequency"
            if coverage["relative_permittivity"] == "not_reported" and catalog_values.get("data_types"):
                if any(re.search(r"\bnk\b|refractive", value, re.I) for value in catalog_values["data_types"]):
                    coverage["relative_permittivity"] = "n_k_available_not_permittivity"
            if coverage["dispersion_model"] == "not_reported" and catalog_values.get("data_types"):
                coverage["dispersion_model"] = "data_type_available_model_not_normalized"
            locator = {
                "paper_documents": sorted(item["docs"]),
                "paper_pages": item["pages"],
                "catalog_records": item.get("catalog_locators", {}),
            }
            self.connection.execute(
                "INSERT INTO material_registry (material_id, canonical_name, relative_permittivity, permeability, loss_tangent, conductivity, refractive_index, dispersion_model, frequency_range, temperature, source_count, source_doc_ids, evidence_pages, property_variants, evidence_status, conflict_fields, updated_at, data_types, wavelength_ranges, conditions, source_references, source_files, source_archives, comments, numeric_rows, data_previews, card_texts, catalog_record_count, parameter_coverage, evidence_locator) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    material_id, canonical,
                    values.get("relative_permittivity", ""), values.get("permeability", ""),
                    values.get("loss_tangent", ""), values.get("conductivity", ""),
                    values.get("refractive_index", ""), values.get("dispersion_model", ""),
                    values.get("frequency_range", ""), values.get("temperature", ""),
                    len(item["docs"]) + len(item.get("catalog_ids", set())),
                    json.dumps(sorted(item["docs"]), ensure_ascii=False),
                    json.dumps(item["pages"], ensure_ascii=False), json.dumps(variants, ensure_ascii=False),
                    status, json.dumps(conflicts, ensure_ascii=False), utc_now(),
                    json.dumps(catalog_values.get("data_types", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("wavelength_ranges", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("conditions", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("source_references", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("source_file", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("source_archive", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("comments", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("numeric_rows", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("data_preview", []), ensure_ascii=False),
                    json.dumps(catalog_values.get("card_text", []), ensure_ascii=False),
                    len(item.get("catalog_ids", set())),
                    json.dumps(coverage, ensure_ascii=False),
                    json.dumps(locator, ensure_ascii=False),
                ),
            )
        self.connection.commit()

    def export_csv(self) -> None:
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = self.connection.execute("SELECT * FROM simulation_summaries ORDER BY title, doc_id").fetchall()
        if rows:
            with self.summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(dict(row) for row in rows)
        material_rows = self.connection.execute(
            "SELECT * FROM material_properties ORDER BY material_name, doc_id"
        ).fetchall()
        with self.material_properties_csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(material_rows[0].keys()) if material_rows else [
                "doc_id", "material_name", "relative_permittivity", "permeability", "loss_tangent",
                "conductivity", "refractive_index", "dispersion_model", "frequency_range", "temperature",
                "evidence_pages", "evidence_text", "source_pdf", "extraction_status", "updated_at",
            ])
            writer.writeheader()
            writer.writerows(dict(row) for row in material_rows)
        self.refresh_material_registry()
        catalog_rows = self.connection.execute(
            "SELECT * FROM material_catalog ORDER BY material_name, record_id"
        ).fetchall()
        with self.material_catalog_csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(catalog_rows[0].keys()) if catalog_rows else [
                "record_id", "material_name", "source_file", "source_archive", "data_types",
                "wavelength_ranges", "comments", "conditions", "source_references", "numeric_rows",
                "data_preview", "card_text", "source_path", "evidence_status", "updated_at",
            ])
            writer.writeheader()
            writer.writerows(dict(row) for row in catalog_rows)
        registry_rows = self.connection.execute(
            "SELECT * FROM material_registry ORDER BY canonical_name"
        ).fetchall()
        with self.material_registry_csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(registry_rows[0].keys()) if registry_rows else [
                "material_id", "canonical_name", *MATERIAL_PROPERTY_FIELDS, "source_count",
                "source_doc_ids", "evidence_pages", "property_variants", "evidence_status",
                "conflict_fields", "updated_at", "data_types", "wavelength_ranges", "conditions",
                "source_references", "source_files", "source_archives", "comments", "numeric_rows",
                "data_previews", "card_texts", "catalog_record_count", "parameter_coverage", "evidence_locator",
            ])
            writer.writeheader()
            writer.writerows(dict(row) for row in registry_rows)


class SyncRunner:
    def __init__(self, config: dict[str, Any], source: ZoteroSource, index: PaperIndex, reviewer: Any | None = None):
        self.config = config
        self.source = source
        storage = config["storage"]
        self.storage_root = Path(storage["root"])
        self.quarantine_root = self.storage_root / "quarantine"
        self.storage_quota = StorageQuota(storage)
        self.storage_multiplier = float(storage.get("estimated_output_multiplier", 8))
        self.index = index
        mineru_cfg = config["mineru"]
        temp_root_value = config.get("storage", {}).get("temp_root", "")
        self.mineru = MinerUParser(
            mineru_cfg["command"],
            Path(mineru_cfg["output_root"]),
            mineru_cfg.get("extra_args"),
            Path(temp_root_value) if temp_root_value else None,
            timeout_seconds=mineru_cfg.get("timeout_seconds", 1800),
        )
        rag_cfg = config["ragflow"]
        self.ragflow_config = rag_cfg
        self.ragflow_enabled = bool(rag_cfg.get("enabled"))
        # Formal installations route documents by dataset_ids. Keep a client
        # sentinel whenever RAGFlow is enabled so that routing is not skipped
        # just because the legacy single-dataset_id is empty.
        self.ragflow = RAGFlowClient(rag_cfg["base_url"], rag_cfg.get("dataset_id", ""), rag_cfg["api_key_env"]) if rag_cfg.get("enabled") else None
        self.reviewer = reviewer
        self.zotero_min_priority = float(config.get("zotero", {}).get("min_priority_score", 0.0))
        review_cfg = config.get("review", {})
        self.review_min_relevance = float(review_cfg.get("min_relevance_score", 0.65))
        self.review_min_quality = float(review_cfg.get("min_quality_score", 0.65))
        self.review_dataset_thresholds = review_cfg.get("dataset_thresholds", {}) if isinstance(review_cfg, Mapping) else {}
        chunk_cfg = config.get("chunk_quality", {})
        self.chunk_gate_enabled = bool(chunk_cfg.get("enabled", False))
        self.chunk_min_score = float(chunk_cfg.get("min_score", 0.98))
        self.chunk_max_chars = int(chunk_cfg.get("max_chars", 6000))
        self.chunk_auto_tune = bool(chunk_cfg.get("auto_tune", True))
        self.chunk_profile_path = self.storage_root / "evaluation" / "iteration_summary.json"

    def admission_policy(self, paper: PaperAttachment) -> tuple[float, float, bool]:
        policy = self.review_dataset_thresholds.get(paper.dataset_key, {}) if isinstance(self.review_dataset_thresholds, Mapping) else {}
        policy = policy if isinstance(policy, Mapping) else {}
        return (
            float(policy.get("min_relevance_score", self.review_min_relevance)),
            float(policy.get("min_quality_score", self.review_min_quality)),
            bool(policy.get("allow_evidence_warning", False)),
        )

    @staticmethod
    def normalize_official_case_evidence_warning(
        paper: PaperAttachment,
        review: dict[str, Any],
        chunk_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Downgrade OCR/table alignment noise for trusted official case PDFs."""
        official_domains = {"comsol.com"}
        official_source = str(paper.source_provider or "").strip().lower() == "comsol-official"
        for source_url in (paper.landing_url, paper.pdf_url):
            try:
                hostname = (urllib.parse.urlsplit(str(source_url or "")).hostname or "").lower().rstrip(".")
            except ValueError:
                hostname = ""
            if any(hostname == domain or hostname.endswith("." + domain) for domain in official_domains):
                official_source = True
                break
        if (
            paper.dataset_key == "cases"
            and official_source
            and review.get("validation") == "evidence_gate"
            and float(chunk_report.get("score", 0) or 0) >= 0.98
            and float(review.get("relevance_score", 0) or 0) >= 0.70
            and float(review.get("quality_score", 0) or 0) >= 0.70
        ):
            warnings = list(dict.fromkeys(str(item) for item in review.get("missing_fields", []) if str(item).strip()))
            review["accepted"] = True
            review["validation"] = "evidence_warning"
            review["evidence_warnings"] = warnings
            review["reason"] = "官方 COMSOL 案例已通过相关性、质量和分块门禁；数值/表格证据对齐保留为警告：" + ", ".join(warnings)
        return review

    def preparsed_markdown_update(self) -> dict[str, Any] | None:
        """Return the explicit no-second-layout-parse policy for Markdown."""
        setting = self.ragflow_config.get("preparsed_markdown", {})
        if not isinstance(setting, Mapping) or not setting.get("enabled", False):
            return None
        parser_config = setting.get("parser_config", {})
        if not isinstance(parser_config, Mapping):
            raise ValueError("ragflow.preparsed_markdown.parser_config 必须是对象")
        parser_config = dict(parser_config)
        # Older configs used the JSON value "\\\\n", which loads as the
        # literal two-character delimiter "\\n". RAGFlow interprets that
        # differently from a real newline and can split words character-wise.
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

    def original_pdf_parser_update(self) -> dict[str, Any] | None:
        """Return the RAGFlow parser settings for direct original-PDF upload."""
        setting = self.ragflow_config.get("original_pdf_parser", {})
        if not isinstance(setting, Mapping) or not setting.get("enabled", False):
            return None
        parser_config = setting.get("parser_config", {})
        if not isinstance(parser_config, Mapping):
            raise ValueError("ragflow.original_pdf_parser.parser_config 必须是对象")
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

    def adaptive_chunk_size(self, markdown_path: Path) -> int:
        """Use the last verified per-document profile when available.

        A new document keeps the configured conservative default. Existing
        documents with a matching Markdown hash reuse the best size found by
        the 50+ iteration harness, so later retries are actually adaptive.
        """
        if not self.chunk_auto_tune or not self.chunk_profile_path.is_file():
            return self.chunk_max_chars
        try:
            profile = json.loads(self.chunk_profile_path.read_text(encoding="utf-8"))
            digest = sha256_file(markdown_path)
            for item in profile.get("best_by_document", []):
                if item.get("sha256") == digest and int(item.get("max_chars", 0)) > 0:
                    return int(item["max_chars"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return self.chunk_max_chars
        return self.chunk_max_chars

    def record_quarantine(self, doc_id: str, status: str, reason: str) -> Path:
        target = self.quarantine_root / doc_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "status.json").write_text(
            json.dumps({"doc_id": doc_id, "status": status, "reason": reason, "updated_at": utc_now()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.write_review_packet(doc_id, status, reason, target)

    @staticmethod
    def _decode_json_object(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if not value:
            return {}
        try:
            decoded = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}

    @staticmethod
    def _artifact(path_value: Any, *, kind: str, indexed_sha256: str = "") -> dict[str, Any]:
        path_text = str(path_value or "")
        path = Path(path_text) if path_text else Path()
        exists = bool(path_text) and path.is_file()
        artifact: dict[str, Any] = {"kind": kind, "path": path_text, "exists": exists}
        if indexed_sha256 and kind == "original_pdf":
            artifact["indexed_sha256"] = indexed_sha256
            if exists:
                try:
                    artifact["actual_sha256"] = sha256_file(path)
                    artifact["sha256_matches_index"] = artifact["actual_sha256"] == indexed_sha256
                except OSError as exc:
                    artifact["hash_error"] = f"{type(exc).__name__}: {exc}"
        return artifact

    def write_review_packet(self, doc_id: str, status: str, reason: str, target: Path | None = None) -> Path:
        """Collect every review input into one auditable, non-destructive packet.

        The packet intentionally stores paths and structured metadata, not a second
        copy of the PDF or full Markdown.  This keeps the 150 GB budget meaningful
        while making a human/model review reproducible from the canonical artifacts.
        """
        target = target or (self.quarantine_root / doc_id)
        target.mkdir(parents=True, exist_ok=True)
        paper_row = self.index.connection.execute(
            "SELECT * FROM papers WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        paper = dict(paper_row) if paper_row else {}
        summary_row = self.index.connection.execute(
            "SELECT * FROM simulation_summaries WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        summary = dict(summary_row) if summary_row else {}
        review_row = self.index.connection.execute(
            "SELECT * FROM review_results WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        review = dict(review_row) if review_row else {}
        if review.get("raw_json"):
            decoded_review = self._decode_json_object(review["raw_json"])
            if decoded_review:
                review["model_output"] = decoded_review
            review.pop("raw_json", None)
        chunk_row = self.index.connection.execute(
            "SELECT * FROM chunk_quality WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        chunk_quality = dict(chunk_row) if chunk_row else {}
        if chunk_quality.get("violations"):
            try:
                chunk_quality["violations"] = json.loads(str(chunk_quality["violations"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        evidence = self._decode_json_object(summary.get("evidence_ref"))
        markdown_path = str(paper.get("mineru_markdown_path") or summary.get("markdown_path") or "")
        content_list_path = str(evidence.get("content_list_path") or "")
        if not content_list_path and markdown_path:
            candidates = sorted(Path(markdown_path).parent.glob("*_content_list.json"))
            if candidates:
                content_list_path = str(candidates[0])
        manifest_path = str(evidence.get("manifest_path") or "")
        published_path = self.storage_root / "published" / f"{doc_id}.md"
        indexed_sha = str(paper.get("sha256") or "")
        artifacts = [
            self._artifact(paper.get("source_path"), kind="zotero_source"),
            self._artifact(paper.get("raw_path") or evidence.get("original_pdf"), kind="original_pdf", indexed_sha256=indexed_sha),
            self._artifact(markdown_path, kind="mineru_markdown"),
            self._artifact(content_list_path, kind="mineru_content_list"),
            self._artifact(manifest_path, kind="evidence_manifest"),
            self._artifact(published_path, kind="published_markdown"),
        ]
        cited_pages: set[int] = set()
        for field_evidence in evidence.get("fields", {}).values() if isinstance(evidence.get("fields"), Mapping) else []:
            if not isinstance(field_evidence, Mapping):
                continue
            review_pages = field_evidence.get("review_pages")
            if review_pages is None:
                review_pages = list(field_evidence.get("pages", []))[:3]
            for page in review_pages:
                try:
                    page_number = int(page)
                except (TypeError, ValueError):
                    continue
                if page_number > 0:
                    cited_pages.add(page_number)
        page_artifacts = materialize_page_evidence(
            paper.get("raw_path") or evidence.get("original_pdf"),
            cited_pages,
            target / "page_evidence",
            str(self.config.get("evidence", {}).get("pdftoppm_command", "pdftoppm")),
        )
        evidence["page_artifacts"] = page_artifacts
        if manifest_path and page_artifacts:
            try:
                manifest_payload = self._decode_json_object(Path(manifest_path).read_text(encoding="utf-8"))
                manifest_payload["page_artifacts"] = page_artifacts
                manifest_payload["updated_at"] = utc_now()
                Path(manifest_path).write_text(
                    json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # The review packet still carries the generated paths.
                pass
        for page_artifact in page_artifacts:
            if page_artifact.get("pdf_path"):
                artifacts.append(self._artifact(page_artifact["pdf_path"], kind="cited_page_pdf"))
            if page_artifact.get("png_path"):
                artifacts.append(self._artifact(page_artifact["png_path"], kind="cited_page_png"))
        packet = {
            "schema_version": 1,
            "doc_id": doc_id,
            "status": status,
            "reason": reason,
            "updated_at": utc_now(),
            "review_instruction": "先核对 original_pdf 对应页和 MinerU 页面证据；只有证据与摘要一致且分块门禁通过，才允许重新入库。",
                "paper": {
                key: paper.get(key)
                for key in ("doc_id", "zotero_key", "title", "authors", "year", "doi", "tags", "filename", "abstract", "landing_url", "pdf_url", "source_provider", "sha256", "selection_score")
                if key in paper
            },
            "artifacts": artifacts,
            "evidence": evidence,
            "simulation_summary": summary,
            "model_review": review or None,
            "chunk_quality": chunk_quality or None,
            "quarantine_status_path": str(target / "status.json"),
        }
        packet_path = target / "review_packet.json"
        packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        return packet_path

    def dataset_id_for(self, paper: PaperAttachment) -> str:
        mapping = self.ragflow_config.get("dataset_ids", {})
        if not isinstance(mapping, Mapping):
            mapping = {}
        # The materials dataset is a curated parameter registry, not a paper
        # corpus.  Material-related papers must remain in the papers dataset;
        # their extracted observations are rebuilt into the material table by
        # material_catalog_sync.py.  This guard also protects Zotero/seed
        # records that carry dataset_key="materials".
        requested_key = str(paper.dataset_key or "").strip()
        if requested_key == "materials":
            requested_key = "papers"
        if requested_key and mapping.get(requested_key):
            return str(mapping[requested_key])
        text = f"{paper.title} {paper.tags}".lower()
        if any(term in text for term in ("manual", "user guide", "手册", "操作", "comsol", "lumerical", "cst", "hfss")):
            key = "software_manuals"
        elif any(term in text for term in ("case", "example", "案例", "仿真设置", "simulation setup")) and not paper.doi:
            key = "cases"
        elif paper.doi or any(term in text for term in ("paper", "论文", "method", "方法", "openalex")):
            key = "papers"
        else:
            key = str(self.ragflow_config.get("default_dataset_key", "fundamentals"))
        return str(mapping.get(key) or self.ragflow_config.get("dataset_id", ""))

    def run(self, dry_run: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        if not dry_run:
            health = current_c_drive_health(self.config)
            if health["state"] == "critical":
                return [{
                    "status": "c_drive_critical",
                    "error": f"C: 可用空间 {health['free_gb']:.2f} GB 低于临界阈值 {health['critical_free_gb']:.2f} GB",
                }]
        attachments = self.source.list_pdf_attachments(limit=limit)
        results: list[dict[str, Any]] = []
        for paper in attachments:
            selection_score = attachment_priority(paper)
            dataset_minimums = self.config.get("zotero", {}).get("dataset_min_priority", {})
            dataset_minimum = float(dataset_minimums.get(paper.dataset_key, self.zotero_min_priority)) if isinstance(dataset_minimums, Mapping) else self.zotero_min_priority
            if selection_score < dataset_minimum:
                results.append({"status": "screened_out", "title": paper.title, "selection_score": selection_score})
                continue
            digest = sha256_file(paper.local_path)
            existing = self.index.find_sha(digest)
            existing = dict(existing) if existing else None
            retry_existing = bool(
                existing
                and str(existing.get("mineru_status") or "") not in {"done"}
                and str(existing.get("status") or "") in {"stored_raw", "failed", "mineru_failed", "blocked"}
            )
            if existing and not retry_existing:
                results.append({"doc_id": existing["doc_id"], "status": "skipped_existing", "title": paper.title})
                continue
            doc_id = str(existing["doc_id"]) if retry_existing else f"zotero-{slug(paper.zotero_key)}-{digest[:12]}"
            existing_raw_path = str(existing.get("raw_path") or "") if retry_existing else ""
            raw_path = Path(existing_raw_path) if existing_raw_path else self.storage_root / "raw" / doc_id / slug(paper.filename)
            if not existing_raw_path and retry_existing:
                raw_path = self.storage_root / "raw" / doc_id / slug(paper.filename)
            if dry_run:
                estimated = int(paper.local_path.stat().st_size * self.storage_multiplier)
                self.storage_quota.ensure_capacity(estimated, paper.title)
                results.append({"doc_id": doc_id, "status": "planned", "title": paper.title, "sha256": digest, "selection_score": selection_score})
                continue
            estimated = int(paper.local_path.stat().st_size * self.storage_multiplier)
            try:
                self.storage_quota.ensure_capacity(estimated, paper.title)
            except StorageQuotaError as exc:
                results.append({"doc_id": doc_id, "status": "storage_quota_blocked", "title": paper.title, "error": str(exc)})
                continue
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if not retry_existing or not raw_path.is_file():
                shutil.copy2(paper.local_path, raw_path)
            row = dict(existing) if retry_existing else {
                "doc_id": doc_id, "zotero_key": paper.zotero_key, "parent_key": paper.parent_key,
                        "title": paper.title, "authors": paper.authors, "year": paper.year, "doi": paper.doi,
                "tags": paper.tags, "filename": paper.filename, "abstract": paper.abstract,
                "landing_url": paper.landing_url, "pdf_url": paper.pdf_url,
                "source_provider": paper.source_provider, "sha256": digest,
                "source_path": str(paper.local_path), "raw_path": str(raw_path),
                "mineru_status": "queued", "mineru_markdown_path": "", "ragflow_document_id": "",
                "ragflow_pdf_document_id": "",
                "ragflow_dataset_id": self.dataset_id_for(paper),
                "selection_score": selection_score,
                "status": "stored_raw", "last_error": "", "created_at": utc_now(), "updated_at": utc_now(),
            }
            if retry_existing:
                row.update({
                    "zotero_key": paper.zotero_key,
                    "parent_key": paper.parent_key,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "doi": paper.doi,
                    "tags": paper.tags,
                    "filename": paper.filename,
                    "abstract": paper.abstract,
                    "landing_url": paper.landing_url,
                    "pdf_url": paper.pdf_url,
                    "source_provider": paper.source_provider,
                    "source_path": str(paper.local_path),
                    "raw_path": str(raw_path),
                    "selection_score": selection_score,
                    "status": "stored_raw",
                    "last_error": "",
                    "updated_at": utc_now(),
                })
            self.index.upsert_paper(row)
            parsed = self.mineru.parse(raw_path, doc_id)
            row["mineru_status"] = parsed["status"]
            row["mineru_markdown_path"] = parsed.get("markdown_path", "")
            row["status"] = "parsed" if parsed["status"] == "done" else parsed["status"]
            row["last_error"] = parsed.get("error", "")
            row["updated_at"] = utc_now()
            self.index.upsert_paper(row)
            if parsed["status"] != "done":
                self.record_quarantine(doc_id, parsed["status"], parsed.get("error", "MinerU failed"))
                results.append({"doc_id": doc_id, "status": parsed["status"], "title": paper.title, "error": parsed.get("error", "")})
                continue
            markdown = Path(parsed["markdown_path"]).read_text(encoding="utf-8", errors="replace")
            summary = extract_simulation_summary(
                markdown, paper, doc_id, parsed["markdown_path"], original_pdf_path=str(raw_path)
            )
            summary["evidence_ref"] = json.dumps(
                persist_evidence_manifest(self.storage_root, doc_id, summary["evidence_ref"]),
                ensure_ascii=False,
            )
            chunk_size = self.adaptive_chunk_size(Path(parsed["markdown_path"]))
            chunk_report = evaluate_chunks(markdown, split_markdown(markdown, chunk_size)).as_dict()
            chunk_report["max_chars"] = chunk_size
            self.index.upsert_chunk_quality(doc_id, chunk_report)
            if self.chunk_gate_enabled and chunk_report["score"] < self.chunk_min_score:
                summary["extraction_status"] = "needs_review"
                summary["notes"] = "Chunk quality gate failed: " + json.dumps(chunk_report, ensure_ascii=False)
                self.index.upsert_summary(summary)
                row["status"] = "chunk_needs_review"
                row["last_error"] = f"chunk quality {chunk_report['score']:.4f} < {self.chunk_min_score:.4f}"
                row["updated_at"] = utc_now()
                self.index.upsert_paper(row)
                self.record_quarantine(doc_id, row["status"], row["last_error"])
                self.index.export_csv()
                results.append({"doc_id": doc_id, "status": row["status"], "title": paper.title, "quality": chunk_report["score"]})
                continue
            review: dict[str, Any] | None = None
            if self.reviewer is not None:
                try:
                    review = self.reviewer.review(
                        title=paper.title,
                        authors=paper.authors,
                        year=paper.year,
                        doi=paper.doi,
                        abstract=paper.abstract,
                        markdown=markdown,
                        heuristic_summary=summary,
                    )
                    review = self.normalize_official_case_evidence_warning(paper, review, chunk_report)
                    parameters = review.get("simulation_parameters", {})
                    if isinstance(parameters, dict):
                        for field in FIELD_TERMS:
                            value = parameters.get(field)
                            if isinstance(value, str) and value.strip():
                                summary[field] = value.strip()
                    model_quality = float(review.get("quality_score", 0) or 0)
                    if model_quality:
                        summary["quality_score"] = round(model_quality, 3)
                    summary["notes"] = "Model review: " + str(review.get("reason", ""))
                    self.index.upsert_review(doc_id, review)
                except Exception as exc:
                    review = {"accepted": False, "relevance_score": 0, "quality_score": 0, "reason": str(exc), "missing_fields": [], "error": str(exc)}
                    self.index.upsert_review(doc_id, review)
                    summary["extraction_status"] = "needs_review"
                    summary["notes"] = "Model review failed; manual review required."
                    self.index.upsert_summary(summary)
                    row["status"] = "review_needs_review"
                    row["last_error"] = str(exc)
                    row["updated_at"] = utc_now()
                    self.index.upsert_paper(row)
                    self.record_quarantine(doc_id, row["status"], row["last_error"])
                    self.index.export_csv()
                    results.append({"doc_id": doc_id, "status": row["status"], "title": paper.title, "error": str(exc)})
                    continue
            self.index.upsert_summary(summary)
            self.index.export_csv()
            if review is not None:
                accepted = review.get("accepted") is True or str(review.get("accepted", "")).lower() == "true"
                relevance = float(review.get("relevance_score", 0) or 0)
                quality = float(review.get("quality_score", 0) or 0)
                min_relevance, min_quality, allow_evidence_warning = self.admission_policy(paper)
                warning_admission = review.get("validation") == "evidence_warning" and allow_evidence_warning
                if not accepted or relevance < min_relevance or quality < min_quality or (review.get("validation") == "evidence_gate" and not warning_admission):
                    row["status"] = "review_rejected" if not accepted else "review_needs_review"
                    row["last_error"] = str(review.get("reason", "模型评估未达到入库阈值"))
                    row["updated_at"] = utc_now()
                    self.index.upsert_paper(row)
                    self.record_quarantine(doc_id, row["status"], row["last_error"])
                    results.append({"doc_id": doc_id, "status": row["status"], "title": paper.title, "quality": summary["quality_score"], "reason": row["last_error"]})
                    continue
            if self.ragflow_enabled and self.reviewer is None:
                summary["extraction_status"] = "needs_review"
                summary["notes"] = "RAGFlow 准入门禁：必须先启用文献审查模型；当前未上传。"
                self.index.upsert_summary(summary)
                self.index.upsert_review(
                    doc_id,
                    {
                        "accepted": False,
                        "relevance_score": 0,
                        "quality_score": 0,
                        "reason": "审查模型未提供；RAGFlow 准入门禁阻止上传。",
                        "missing_fields": [],
                    },
                )
                row["status"] = "review_needs_review"
                row["last_error"] = "RAGFlow 已启用但未提供审查器；禁止无审查入库。"
                row["updated_at"] = utc_now()
                self.index.upsert_paper(row)
                self.record_quarantine(doc_id, row["status"], row["last_error"])
                results.append({"doc_id": doc_id, "status": row["status"], "title": paper.title, "error": row["last_error"]})
                continue
            published = self.storage_root / "published" / f"{doc_id}.md"
            published.parent.mkdir(parents=True, exist_ok=True)
            published.write_text(
                "# Paper simulation summary\n\n"
                + "\n".join(f"- {key}: {summary.get(key, '')}" for key in FIELD_TERMS)
                + "\n\n# MinerU full-text output\n\n"
                + markdown,
                encoding="utf-8",
            )
            upload_mode = "local_mineru_markdown"
            if self.ragflow:
                rag_id = ""
                try:
                    dataset_id = self.dataset_id_for(paper)
                    if not dataset_id:
                        raise RuntimeError("未配置目标 RAGFlow dataset_id")
                    ragflow = RAGFlowClient(self.ragflow_config["base_url"], dataset_id, self.ragflow_config["api_key_env"])
                    upload_original_pdf = bool(self.ragflow_config.get("upload_original_pdf", False))
                    upload_mode = "original_pdf" if upload_original_pdf else "local_mineru_markdown"
                    upload_path = raw_path if upload_original_pdf else published
                    display_name = (
                        f"{doc_id}--{sha256_file(raw_path)[:12]}.pdf"
                        if upload_original_pdf
                        else published_document_name(doc_id, markdown)
                    )
                    existing = ragflow.find_active_document(display_name)
                    if existing:
                        rag_id = str(existing["id"])
                    else:
                        reserve_ragflow_upload_attempt(self.index, row, is_original_pdf=upload_original_pdf)
                        rag_id = ragflow.upload(upload_path, display_name=display_name)
                    if upload_original_pdf:
                        # Track the source PDF separately from the serving
                        # document, which may become a Markdown fallback.
                        row["ragflow_pdf_document_id"] = rag_id
                    parser_update = (
                        self.original_pdf_parser_update()
                        if upload_original_pdf
                        else self.preparsed_markdown_update()
                    )
                    if parser_update:
                        ragflow.update_document(rag_id, parser_update)
                        row["ragflow_parser_config"] = json.dumps(parser_update, ensure_ascii=False)
                    ragflow.trigger_parse(rag_id)
                    ragflow.wait_done(rag_id)
                    if not ragflow.list_chunks(rag_id):
                        raise RuntimeError(f"RAGFlow 文档解析完成但没有生成分块：{rag_id}")
                    row["ragflow_document_id"] = rag_id
                    row["status"] = "published_ragflow"
                    row["last_error"] = ""
                except Exception as exc:
                    fallback_enabled = bool(self.ragflow_config.get("fallback_to_preparsed_markdown", True))
                    if upload_mode == "original_pdf" and fallback_enabled:
                        try:
                            # Preserve the original-PDF document as the
                            # visual/audit asset. A Markdown fallback is a
                            # parallel retrieval path, never a replacement.
                            fallback_name = published_document_name(doc_id, markdown)
                            fallback_existing = ragflow.find_active_document(fallback_name)
                            fallback_id = (
                                str(fallback_existing["id"])
                                if fallback_existing
                                else self._upload_ragflow_fallback(ragflow, published, fallback_name, row)
                            )
                            fallback_update = self.preparsed_markdown_update()
                            if fallback_update:
                                ragflow.update_document(fallback_id, fallback_update)
                                row["ragflow_parser_config"] = json.dumps(fallback_update, ensure_ascii=False)
                            ragflow.trigger_parse(fallback_id)
                            ragflow.wait_done(fallback_id)
                            if not ragflow.list_chunks(fallback_id):
                                raise RuntimeError(f"RAGFlow Markdown 兜底也没有生成分块：{fallback_id}")
                            row["ragflow_document_id"] = fallback_id
                            row["status"] = "published_ragflow"
                            row["last_error"] = ""
                            upload_mode = "local_mineru_markdown_fallback"
                        except Exception as fallback_exc:
                            row["status"] = "ragflow_failed"
                            row["last_error"] = f"原始 PDF 解析失败：{exc}；Markdown 兜底失败：{fallback_exc}"
                    else:
                        row["status"] = "ragflow_failed"
                        row["last_error"] = str(exc)
                row["updated_at"] = utc_now()
                self.index.upsert_paper(row)
                if row["status"] != "published_ragflow":
                    self.record_quarantine(doc_id, row["status"], row["last_error"])
            results.append({"doc_id": doc_id, "status": row["status"], "title": paper.title, "quality": summary["quality_score"], "ragflow_upload_mode": upload_mode})
        self.index.export_csv()
        return results

    def _upload_ragflow_fallback(
        self,
        ragflow: RAGFlowClient,
        published: Path,
        fallback_name: str,
        row: dict[str, Any],
    ) -> str:
        reserve_ragflow_upload_attempt(self.index, row, is_original_pdf=False)
        return ragflow.upload(published, display_name=fallback_name)


class SinglePaperSource:
    """Adapter used to send one discovered PDF through the same safe pipeline."""

    def __init__(self, paper: PaperAttachment):
        self.paper = paper

    def list_pdf_attachments(self, limit: int | None = None) -> list[PaperAttachment]:
        return [self.paper][:limit]


class DiscoveryRunner:
    """Discover OA candidates, then route each downloaded PDF through SyncRunner."""

    @staticmethod
    def _is_quota_or_rate_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return bool(re.search(r"\bhttp\s*429\b|rate limit|quota|insufficient budget|retryafter", message))

    @staticmethod
    def _candidate_identity(candidate: LiteratureCandidate) -> str:
        title = re.sub(r"[^a-z0-9]+", " ", candidate.title.lower()).strip()
        if title:
            return f"title:{title}:{candidate.year[:4]}"
        if candidate.doi:
            return "doi:" + re.sub(r"^https?://doi.org/", "", candidate.doi.strip().lower())
        return "candidate:" + candidate.candidate_id

    @classmethod
    def _merge_candidates(cls, candidates: Iterable[LiteratureCandidate]) -> list[LiteratureCandidate]:
        """Deduplicate provider results while retaining the most actionable record."""
        merged: dict[str, LiteratureCandidate] = {}
        for candidate in candidates:
            key = cls._candidate_identity(candidate)
            previous = merged.get(key)
            if previous is None:
                merged[key] = candidate
                continue
            previous_rank = (
                bool(previous.pdf_url), bool(previous.doi), previous.citation_count,
                len(previous.abstract), previous.source == "openalex",
            )
            current_rank = (
                bool(candidate.pdf_url), bool(candidate.doi), candidate.citation_count,
                len(candidate.abstract), candidate.source == "openalex",
            )
            if current_rank > previous_rank:
                merged[key] = candidate
        return list(merged.values())

    def __init__(self, config: dict[str, Any], index: PaperIndex, reviewer: Any | None = None):
        self.config = config
        self.index = index
        self.discovery = config.get("discovery", {})
        provider_names = [str(self.discovery.get("provider", "openalex")).lower()]
        fallback = str(self.discovery.get("fallback_provider", "")).strip().lower()
        if fallback and fallback not in provider_names:
            provider_names.append(fallback)
        self.provider_names = provider_names
        self.discoverers: list[LiteratureDiscoverer] = []
        for provider in provider_names:
            if provider == "openalex":
                self.discoverers.append(
                    OpenAlexDiscoverer(
                        base_url=str(self.discovery.get("base_url", "https://api.openalex.org")),
                        mailto=str(self.discovery.get("mailto", "")),
                        timeout=float(self.discovery.get("openalex_timeout_seconds", 20)),
                    )
                )
            elif provider == "arxiv":
                self.discoverers.append(
                    ArxivDiscoverer(
                        base_url=str(self.discovery.get("arxiv_base_url", "https://export.arxiv.org/api/query")),
                        timeout=float(self.discovery.get("arxiv_timeout_seconds", 15)),
                    )
                )
            else:
                raise ValueError(f"Unsupported discovery provider: {provider}")
        if not self.discoverers:
            raise ValueError("At least one discovery provider is required")
        self.storage_quota = StorageQuota(config["storage"])
        self.discovery_max_pdf_bytes = int(float(config["storage"].get("max_discovery_pdf_gb", 1.5)) * 1024**3)
        self.reviewer = reviewer

    def run(
        self,
        queries: list[str],
        limit_per_query: int | None = None,
        dry_run: bool = False,
        dataset_key: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not dry_run:
            health = current_c_drive_health(self.config)
            if health["state"] == "critical":
                error = f"C: 可用空间 {health['free_gb']:.2f} GB 低于临界阈值 {health['critical_free_gb']:.2f} GB"
                for query in queries:
                    self.index.record_discovery_run(query, self.provider_names, "blocked", error=error)
                    results.append({"query": query, "status": "c_drive_critical", "error": error})
                return results
        limit = limit_per_query or int(self.discovery.get("limit_per_query", 10))
        min_priority = float(self.discovery.get("min_priority_score", 0.0))
        always_query_fallback = bool(self.discovery.get("always_query_fallback", False))
        storage_root = Path(self.config["storage"]["root"])
        disabled_provider_indexes: set[int] = set()
        for query in queries:
            discovered: list[LiteratureCandidate] = []
            errors: list[str] = []
            for provider_index, discoverer in enumerate(self.discoverers):
                if provider_index in disabled_provider_indexes:
                    errors.append(
                        f"{discoverer.__class__.__name__}: skipped after a prior quota/rate-limit failure in this run"
                    )
                    continue
                try:
                    found = discoverer.search(query, limit=limit)
                    discovered.extend(found)
                    # The fallback is only needed when the primary source
                    # returned no usable open PDF. This keeps normal runs
                    # cheap while repairing the common metadata-only case.
                    if provider_index == 0 and not always_query_fallback and any(
                        candidate.pdf_url and candidate_priority(candidate) >= min_priority for candidate in found
                    ):
                        break
                except Exception as exc:
                    errors.append(f"{discoverer.__class__.__name__}: {exc}")
                    if self._is_quota_or_rate_limit_error(exc):
                        disabled_provider_indexes.add(provider_index)
            candidates = self._merge_candidates(discovered)
            if not candidates and errors:
                self.index.record_discovery_run(query, self.provider_names, "failed", error="; ".join(errors))
                results.append({"query": query, "status": "discovery_failed", "errors": errors})
                continue
            if not candidates:
                self.index.record_discovery_run(query, self.provider_names, "empty")
                results.append({"query": query, "status": "discovery_empty"})
                continue
            self.index.record_discovery_run(
                query,
                self.provider_names,
                "partial" if errors else "ok",
                candidate_count=len(candidates),
                error="; ".join(errors),
            )
            candidates = sorted(candidates, key=candidate_priority, reverse=True)
            for candidate in candidates:
                priority = candidate_priority(candidate)
                previous_candidate = self.index.find_candidate(candidate.candidate_id)
                terminal_statuses = {
                    "skipped_existing", "review_needs_review", "review_rejected", "published_ragflow",
                    "ragflow_failed", "chunk_needs_review", "screened_out", "no_open_pdf", "storage_quota_blocked",
                    "duplicate_existing",
                }
                previous_status = str(previous_candidate["status"] or "") if previous_candidate else ""
                screening_threshold_changed = (
                    previous_status == "screened_out"
                    and priority >= min_priority
                ) if previous_candidate else False
                if previous_candidate and previous_status in terminal_statuses and not screening_threshold_changed:
                    results.append({
                        "candidate_id": candidate.candidate_id,
                        "status": "skipped_existing_candidate",
                        "previous_status": str(previous_candidate["status"]),
                        "title": candidate.title,
                        "priority": priority,
                    })
                    continue
                self.index.upsert_candidate(candidate, "discovered")
                if priority < min_priority:
                    self.index.upsert_candidate(
                        candidate,
                        "screened_out",
                        decision_json=json.dumps({"priority_score": priority}, ensure_ascii=False),
                    )
                    results.append({"candidate_id": candidate.candidate_id, "status": "screened_out", "title": candidate.title, "priority": priority})
                    continue
                existing = self.index.find_doi(candidate.doi)
                if existing:
                    self.index.upsert_candidate(candidate, "skipped_existing", doc_id=existing["doc_id"])
                    results.append({"candidate_id": candidate.candidate_id, "status": "skipped_existing", "title": candidate.title, "priority": priority})
                    continue
                existing = self.index.find_title(candidate.title, candidate.year)
                if existing:
                    self.index.upsert_candidate(candidate, "skipped_existing", doc_id=existing["doc_id"])
                    results.append({
                        "candidate_id": candidate.candidate_id,
                        "status": "skipped_existing_candidate",
                        "previous_status": str(existing["status"] or ""),
                        "doc_id": str(existing["doc_id"]),
                        "title": candidate.title,
                        "priority": priority,
                    })
                    continue
                if not candidate.pdf_url:
                    self.index.upsert_candidate(candidate, "no_open_pdf")
                    results.append({"candidate_id": candidate.candidate_id, "status": "no_open_pdf", "title": candidate.title, "priority": priority})
                    continue
                download_path = storage_root / "discovery" / "downloads" / f"{slug(candidate.candidate_id)}.pdf"
                if dry_run:
                    self.index.upsert_candidate(candidate, "planned")
                    results.append({"candidate_id": candidate.candidate_id, "status": "planned", "title": candidate.title, "pdf_url": candidate.pdf_url, "priority": priority})
                    continue
                try:
                    self.storage_quota.ensure_capacity(self.discovery_max_pdf_bytes, candidate.title)
                    download_open_pdf(candidate.pdf_url, download_path)
                    self.index.upsert_candidate(candidate, "downloaded")
                    paper = PaperAttachment(
                        zotero_key=candidate.candidate_id,
                        parent_key="",
                        title=candidate.title,
                        authors=candidate.authors,
                        year=candidate.year,
                        doi=candidate.doi,
                        tags="discovered;" + candidate.source,
                        filename=download_path.name,
                        local_path=download_path,
                        dataset_key=dataset_key or "papers",
                        abstract=candidate.abstract,
                        landing_url=candidate.landing_url,
                        pdf_url=candidate.pdf_url,
                        source_provider=candidate.source,
                    )
                    sync = SyncRunner(self.config, SinglePaperSource(paper), self.index, self.reviewer)
                    sync_result = sync.run(limit=1)[0]
                    self.index.upsert_candidate(candidate, sync_result["status"], doc_id=str(sync_result.get("doc_id", "")))
                    results.append({"candidate_id": candidate.candidate_id, "priority": priority, **sync_result})
                except StorageQuotaError as exc:
                    self.index.upsert_candidate(candidate, "storage_quota_blocked", last_error=str(exc))
                    results.append({"candidate_id": candidate.candidate_id, "status": "storage_quota_blocked", "title": candidate.title, "error": str(exc)})
                except Exception as exc:
                    self.index.upsert_candidate(candidate, "download_failed", last_error=str(exc))
                    results.append({"candidate_id": candidate.candidate_id, "status": "download_failed", "title": candidate.title, "error": str(exc)})
        return results


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("zotero", "mineru", "ragflow", "storage")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing sections: {', '.join(missing)}")
    return config


def preflight(config: dict[str, Any], probe_zotero: bool = False) -> int:
    print(f"Python: {sys.version.split()[0]}")
    temp_root_value = config.get("storage", {}).get("temp_root", "")
    mineru = MinerUParser(
        config["mineru"]["command"],
        Path(config["mineru"]["output_root"]),
        config["mineru"].get("extra_args"),
        Path(temp_root_value) if temp_root_value else None,
        timeout_seconds=config["mineru"].get("timeout_seconds", 1800),
    )
    print(f"MinerU command available: {mineru.is_available()} ({config['mineru']['command']})")
    root = Path(config["storage"]["root"])
    print(f"Storage root exists: {root.exists()} ({root})")
    quota = StorageQuota(config["storage"])
    print(f"Storage quota: {quota.used_bytes / 1024**3:.2f} GB used / {quota.quota_bytes / 1024**3:.0f} GB limit; {quota.remaining_bytes / 1024**3:.2f} GB available after reserve")
    rag = config["ragflow"]
    print(f"RAGFlow enabled: {bool(rag.get('enabled'))}")
    if rag.get("enabled"):
        print(f"RAGFlow API key configured: {bool(resolve_api_secret(rag['api_key_env']))} (value not shown)")
    review = config.get("review", {})
    model = config.get("model", {})
    print(f"Ingest review enabled: {bool(review.get('enabled'))}")
    if review.get("enabled"):
        print(f"Ingest model configured: {bool(model.get('base_url')) and bool(model.get('model_name'))}; API key configured: {bool(resolve_api_secret(model.get('api_key_env', 'MODEL_API_KEY')))} (value not shown)")
    chunk_cfg = config.get("chunk_quality", {})
    print(f"Chunk quality gate: {bool(chunk_cfg.get('enabled', False))}; threshold={chunk_cfg.get('min_score', 0.98)}")
    if probe_zotero:
        source = build_zotero_source(config)
        papers = source.list_pdf_attachments(limit=1)
        if isinstance(source, ZoteroAutoSource):
            detail = f"source={source.last_source}"
            if source.last_api_error:
                detail += "; desktop API unavailable, used SQLite fallback"
        else:
            detail = "source=configured"
        print(f"Zotero probe: {detail}, PDF attachments found in sample: {len(papers)}")
    return 0


def build_runner(config: dict[str, Any], dry_run: bool = False) -> SyncRunner:
    source = build_zotero_source(config)
    storage = config["storage"]
    index_path = Path(":memory:") if dry_run else Path(storage["index_db"])
    catalog_path = str(config.get("material_registry", {}).get("catalog_csv", "")).strip()
    index = PaperIndex(index_path, Path(storage["summary_csv"]), Path(catalog_path) if catalog_path else None)
    try:
        from ingest_model import build_ingest_reviewer
        reviewer = build_ingest_reviewer(config)
    except ImportError:
        reviewer = None
    return SyncRunner(config, source, index, reviewer)


def build_zotero_source(config: dict[str, Any]) -> ZoteroSource:
    zotero = config["zotero"]
    api = ZoteroLocalApi(zotero["base_url"], zotero.get("library_prefix", "/users/0"))
    mode = str(zotero.get("mode", "auto")).lower()
    if mode == "api":
        return api
    sqlite_source = ZoteroSqlite(
        zotero.get("sqlite_path", str(Path.home() / "Zotero" / "zotero.sqlite")),
        zotero.get("storage_root", str(Path.home() / "Zotero" / "storage")),
    )
    if mode == "sqlite":
        return sqlite_source
    if mode == "auto":
        return ZoteroAutoSource(api, sqlite_source)
    raise ValueError(f"Unsupported zotero.mode: {mode}; expected auto, api, or sqlite")


def run_cycle(
    config: dict[str, Any],
    queries: list[str],
    limit: int | None = None,
    dry_run: bool = False,
    dataset_key: str | None = None,
    skip_zotero: bool = False,
) -> dict[str, Any]:
    """Run one bounded supplementation cycle with health checks on both sides."""
    from maintenance import run_maintenance

    # A routine ingestion cycle reports disk health but must not remove files
    # from the user's Windows Temp directory. Cleanup remains an explicit
    # maintenance action.
    before = run_maintenance(config, perform_cleanup=False)
    if before["c_drive"]["state"] != "ok":
        return {
            "blocked": "c_drive_low_space",
            "health_before": before,
            "zotero": [],
            "discovery": [],
            "health_after": before,
        }
    storage = config["storage"]
    catalog_path = str(config.get("material_registry", {}).get("catalog_csv", "")).strip()
    index = PaperIndex(
        Path(":memory:") if dry_run else Path(storage["index_db"]),
        Path(storage["summary_csv"]),
        Path(catalog_path) if catalog_path else None,
    )
    try:
        try:
            from ingest_model import build_ingest_reviewer
            reviewer = build_ingest_reviewer(config)
        except ImportError:
            reviewer = None
        if skip_zotero:
            zotero_results = []
        else:
            zotero_runner = SyncRunner(config, build_zotero_source(config), index, reviewer)
            zotero_results = zotero_runner.run(dry_run=dry_run, limit=limit)
        discovery_results = DiscoveryRunner(config, index, reviewer).run(
            queries,
            limit_per_query=limit,
            dry_run=dry_run,
            dataset_key=dataset_key,
        )
    finally:
        index.close()
    ragflow_runtime = None
    if not dry_run and bool(config.get("ragflow", {}).get("enabled")):
        try:
            from ragflow_runtime_audit import audit_ragflow_runtime
            ragflow_runtime = audit_ragflow_runtime(config)
        except Exception as exc:
            ragflow_runtime = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    after = run_maintenance(config, perform_cleanup=False)
    return {
        "health_before": before,
        "zotero": zotero_results,
        "discovery": discovery_results,
        "ragflow_runtime": ragflow_runtime,
        "health_after": after,
    }


def run_watch(config: dict[str, Any], queries: list[str], limit: int | None = None, interval_minutes: float = 360, cycles: int | None = None, dry_run: bool = False) -> None:
    """Run bounded or continuous supplementation cycles with a disk safety stop."""
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    for cycle_number in range(cycles or 2**31 - 1):
        report = run_cycle(config, queries, limit=limit, dry_run=dry_run)
        root = Path(config["storage"]["root"]) / "evaluation"
        root.mkdir(parents=True, exist_ok=True)
        with (root / "watch_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"cycle": cycle_number + 1, "timestamp": utc_now(), "report": report}, ensure_ascii=False) + "\n")
        print(json.dumps({"cycle": cycle_number + 1, "blocked": report.get("blocked"), "zotero_count": len(report.get("zotero", [])), "discovery_count": len(report.get("discovery", [])), "c_free_gb": report["health_after"]["c_drive"]["free_gb"]}, ensure_ascii=False))
        if report.get("blocked") == "c_drive_low_space" or (cycles is not None and cycle_number + 1 >= cycles):
            return
        remaining_seconds = interval_minutes * 60
        while remaining_seconds > 0:
            sleep_seconds = min(60.0, remaining_seconds)
            time.sleep(sleep_seconds)
            remaining_seconds -= sleep_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="Electromagnetic-domain reference: Zotero/OpenAlex to MinerU to RAGFlow")
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--probe-zotero", action="store_true")
    sync = sub.add_parser("sync")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--limit", type=int)
    discover = sub.add_parser("discover-sync")
    discover.add_argument("--query", action="append", dest="queries", help="literature query; may be repeated")
    discover.add_argument("--limit", type=int)
    discover.add_argument("--dry-run", action="store_true")
    iterate = sub.add_parser("iterate", help="run adaptive chunk/evidence evaluation")
    iterate.add_argument("--iterations", type=int, default=50)
    iterate.add_argument("--document", action="append", dest="documents", help="Markdown path; may be repeated")
    sub.add_parser("audit-index", help="read-only audit of index, MinerU, evidence, and RAGFlow invariants")
    sub.add_parser("audit-config", help="audit dataset mappings, embedding, MinerU backend, and runtime credentials")
    runtime_audit = sub.add_parser("ragflow-audit", help="read-only audit of live RAGFlow document parsing and chunks")
    runtime_audit.add_argument("--dataset-key")
    sub.add_parser("corpus-inventory", help="write a compact actionable inventory of indexed literature")
    reconcile = sub.add_parser("reconcile-metadata", help="backfill local summaries, chunks, and evidence without model/RAGFlow calls")
    reconcile.add_argument("--limit", type=int)
    admission = sub.add_parser("recheck-admission", help="re-review historical RAGFlow uploads without re-uploading them")
    admission.add_argument("--limit", type=int)
    admission.add_argument("--model-base-url", help="temporary model base URL for this run; does not modify config.json")
    admission.add_argument("--model-protocol", help="temporary model protocol, e.g. anthropic_messages or openai_chat_completions")
    admission.add_argument("--doc-id", action="append", dest="doc_ids", help="re-check only this indexed document; may be repeated")
    cycle = sub.add_parser("cycle", help="one bounded Zotero + discovery + MinerU + review + RAGFlow supplementation cycle")
    cycle.add_argument("--query", action="append", dest="queries", help="discovery query; may be repeated")
    cycle.add_argument("--limit", type=int)
    cycle.add_argument("--dry-run", action="store_true")
    retrieval = sub.add_parser("retrieval-eval", help="run a labelled retrieval benchmark against RAGFlow")
    retrieval.add_argument("--benchmark", type=Path, required=True)
    retrieval.add_argument("--dataset-id")
    retrieval.add_argument("--dataset-key", help="only evaluate cases with this dataset_key")
    retrieval.add_argument("--max-cases", type=int, help="evaluate at most this many cases")
    retry = sub.add_parser("retry-failed", help="retry parsed documents held outside RAGFlow")
    retry.add_argument("--limit", type=int)
    retry.add_argument("--doc-id", action="append", dest="doc_ids", help="retry only this indexed document; may be repeated")
    retry.add_argument("--model-base-url", help="temporary model base URL for this run; does not modify config.json")
    retry.add_argument("--model-protocol", help="temporary model protocol, e.g. anthropic_messages or openai_chat_completions")
    seed = sub.add_parser("seed-ingest", help="ingest curated remote or local PDF sources through the same gate")
    seed.add_argument("--manifest", type=Path, required=True)
    seed.add_argument("--dry-run", action="store_true")
    watch = sub.add_parser("watch", help="repeatedly run bounded supplementation cycles")
    watch.add_argument("--query", action="append", dest="queries", help="discovery query; may be repeated")
    watch.add_argument("--limit", type=int)
    watch.add_argument("--interval-minutes", type=float, default=360)
    watch.add_argument("--cycles", type=int)
    watch.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        return preflight(config, args.probe_zotero)
    if args.command == "discover-sync":
        storage = config["storage"]
        index = PaperIndex(Path(":memory:") if args.dry_run else Path(storage["index_db"]), Path(storage["summary_csv"]))
        try:
            from ingest_model import build_ingest_reviewer
            reviewer = build_ingest_reviewer(config)
            discovery_cfg = config.get("discovery", {})
            queries = args.queries or list(discovery_cfg.get("queries", []))
            if not queries:
                raise SystemExit("请使用 --query 指定检索词，或在 config.discovery.queries 中配置。")
            workflow = DiscoveryRunner(config, index, reviewer)
            for result in workflow.run(queries, limit_per_query=args.limit, dry_run=args.dry_run):
                print(json.dumps(result, ensure_ascii=False))
        finally:
            index.close()
        return 0
    if args.command == "iterate":
        from iteration_harness import run_iterations
        report = run_iterations(config, iterations=args.iterations, document_paths=args.documents)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-index":
        from index_audit import audit_index
        print(json.dumps(audit_index(config), ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-config":
        from config_audit import audit_config
        print(json.dumps(audit_config(config), ensure_ascii=False, indent=2))
        return 0
    if args.command == "ragflow-audit":
        from ragflow_runtime_audit import audit_ragflow_runtime
        print(json.dumps(audit_ragflow_runtime(config, args.dataset_key), ensure_ascii=False, indent=2))
        return 0
    if args.command == "corpus-inventory":
        from corpus_inventory import corpus_inventory
        print(json.dumps(corpus_inventory(config), ensure_ascii=False, indent=2))
        return 0
    if args.command == "reconcile-metadata":
        from reconcile_metadata import reconcile_metadata
        print(json.dumps(reconcile_metadata(config, args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.command == "recheck-admission":
        from recheck_admission import recheck_admission
        print(json.dumps(recheck_admission(config, args.limit, args.model_base_url, args.model_protocol, args.doc_ids), ensure_ascii=False, indent=2))
        return 0
    if args.command == "cycle":
        discovery_cfg = config.get("discovery", {})
        queries = args.queries or list(discovery_cfg.get("queries", []))
        if not queries:
            raise SystemExit("请使用 --query 指定检索词，或在 config.discovery.queries 中配置。")
        print(json.dumps(run_cycle(config, queries, limit=args.limit, dry_run=args.dry_run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "retrieval-eval":
        from retrieval_benchmark import run_benchmark
        print(json.dumps(
            run_benchmark(config, args.benchmark, args.dataset_id, args.dataset_key, args.max_cases),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if args.command == "retry-failed":
        from retry_failed import retry_failed
        print(json.dumps(retry_failed(config, args.limit, args.model_base_url, args.model_protocol, args.doc_ids), ensure_ascii=False, indent=2))
        return 0
    if args.command == "seed-ingest":
        from seed_ingest import run_seed_ingest
        print(json.dumps(run_seed_ingest(config, args.manifest, args.dry_run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "watch":
        discovery_cfg = config.get("discovery", {})
        queries = args.queries or list(discovery_cfg.get("queries", []))
        if not queries:
            raise SystemExit("请使用 --query 指定检索词，或在 config.discovery.queries 中配置。")
        run_watch(config, queries, limit=args.limit, interval_minutes=args.interval_minutes, cycles=args.cycles, dry_run=args.dry_run)
        return 0
    runner = build_runner(config, dry_run=args.dry_run)
    try:
        for result in runner.run(dry_run=args.dry_run, limit=args.limit):
            print(json.dumps(result, ensure_ascii=False))
    finally:
        runner.index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
