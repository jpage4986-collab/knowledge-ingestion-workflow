"""Local Knowledge Ingestion workbench with existing-library write isolation."""
from __future__ import annotations

import argparse
import copy
import email.policy
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "examples" / "electromagnetics" / "ingestion" / "config.json"
RAGFLOW_BASE = os.environ.get("RAGFLOW_BASE_URL", "http://127.0.0.1:9380").rstrip("/")
RAGFLOW_KEY = os.environ.get("RAGFLOW_API_KEY", "").strip()
MINERU_HEALTH_URL = os.environ.get("MINERU_PROXY_HEALTH_URL", "http://127.0.0.1:8001/health")
AGNES_MODEL_PROVIDER = os.environ.get("AGNES_MODEL_PROVIDER", "agnes-ai")
AGNES_MODEL_VERSION = os.environ.get("AGNES_MODEL_VERSION", "")
MODEL_GATE_FILE = Path(os.environ.get("KNOWLEDGE_MODEL_GATE_FILE", str(ROOT / "model-gate.json")))
try:
    from job_history import JobHistoryStore
except ModuleNotFoundError:  # pragma: no cover
    import job_history as _job_history_module
    JobHistoryStore = _job_history_module.JobHistoryStore
HISTORY = JobHistoryStore()

class ParseGateError(Exception):
    pass


def _model_gate(dataset_id: str = "") -> dict:
    """Read model and parser configuration from RAGFlow, never from assertions."""
    records = _raw_datasets()
    if dataset_id:
        records = [item for item in records if str(item.get("id")) == dataset_id]
        if not records:
            raise ServiceError(404, "找不到待核验的知识库配置")
    configurations = []
    for item in records:
        embedding = str(item.get("embedding_model") or "").strip()
        parser = item.get("parser_config") if isinstance(item.get("parser_config"), dict) else {}
        configurations.append({
            "dataset_id": str(item.get("id")), "dataset_name": str(item.get("name") or ""),
            "embedding_model": embedding, "parser": str(parser.get("layout_recognize") or "DeepDOC"),
            "chunk_method": str(item.get("chunk_method") or "naive"),
            "ready": bool(embedding and not embedding.startswith("FILL_")),
        })
    ready = bool(configurations) and (all(c["ready"] for c in configurations) if dataset_id else any(c["ready"] for c in configurations))
    selected = configurations[0] if len(configurations) == 1 else {}
    return {"ready": ready, "source": "ragflow_dataset_metadata", "core_agent": "agnes-3.0-flash",
            "provider": selected.get("embedding_model", "").split("@")[-1],
            "version": selected.get("embedding_model", ""), "configurations": configurations,
            "message": "已读取实际解析与向量化配置" if ready else "知识库尚未配置可用向量模型"}


def _require_model_gate(dataset_id: str) -> None:
    gate = _model_gate(dataset_id)
    if not gate["ready"]:
        raise ServiceError(409, gate["message"])


def _load_model_gate_file(dataset_id: str = "") -> dict:
    # Compatibility entry point. Manual files and environment assertions are ignored.
    return _model_gate(dataset_id)


CACHE_SECONDS = 20
PAGE_SIZE_MAX = 100
SANDBOX_PREFIX = "HIH-SANDBOX-"
SANDBOX_MARKER = "[knowledge-workbench-sandbox:v1]"
SANDBOX_REGISTRY_FILE = ROOT / "sandbox-registry.json"
_sandbox_registry_lock = threading.RLock()

def _registered_sandbox_ids() -> set[str]:
    try:
        payload = json.loads(SANDBOX_REGISTRY_FILE.read_text("utf-8"))
        return {str(value) for value in payload.get("ids", [])}
    except (OSError, ValueError, AttributeError):
        return set()

def _register_sandbox(dataset_id: str) -> None:
    with _sandbox_registry_lock:
        ids = _registered_sandbox_ids() | {dataset_id}
        temporary = SANDBOX_REGISTRY_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps({"ids": sorted(ids)}, indent=2), "utf-8")
        os.replace(temporary, SANDBOX_REGISTRY_FILE)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_REQUEST_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_FILES = 10
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx", ".xlsx", ".csv", ".html"}

_cache_lock = threading.RLock()
_datasets_cache: tuple[float, list[dict]] | None = None
_documents_cache: dict[str, tuple[float, list[dict]]] = {}
_summary_cache: tuple[float, dict] | None = None
_ingestion_jobs_lock = threading.RLock()
_ingestion_jobs: dict[str, dict] = {}
_ingestion_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="harness-job")


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _request_json(path: str, *, params: dict | None = None, auth: bool = True, timeout: float = 8) -> dict:
    if auth and not RAGFLOW_KEY:
        raise ServiceError(503, "RAGFLOW_API_KEY 未配置")
    query = urlencode(params or {})
    url = f"{RAGFLOW_BASE}{path}" + (f"?{query}" if query else "")
    headers = {"Accept": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {RAGFLOW_KEY}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ServiceError(exc.code, "RAGFlow API Key 无法读取知识库") from None
        raise ServiceError(502, f"RAGFlow 返回 HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError):
        raise ServiceError(503, "无法连接 RAGFlow") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ServiceError(502, "RAGFlow 返回的数据格式无法识别") from None
    if not isinstance(payload, dict):
        raise ServiceError(502, "RAGFlow 返回的数据格式无法识别")
    if payload.get("code", 0) not in (0, None):
        raise ServiceError(502, str(payload.get("message") or "RAGFlow 请求失败"))
    return payload


def _api_data(path: str, *, params: dict | None = None, auth: bool = True, timeout: float = 8):
    payload = _request_json(path, params=params, auth=auth, timeout=timeout)
    return payload.get("data")


def _write_api_json(path: str, payload: dict, *, timeout: float = 60) -> dict:
    if not RAGFLOW_KEY:
        raise ServiceError(503, "RAGFLOW_API_KEY 未配置")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{RAGFLOW_BASE}{path}", data=body,
        headers={"Authorization": f"Bearer {RAGFLOW_KEY}", "Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ServiceError(exc.code, "RAGFlow API Key 无法执行沙盒操作") from None
        detail = ""
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            detail = str(error_payload.get("message") or "") if isinstance(error_payload, dict) else ""
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise ServiceError(502, f"RAGFlow 拒绝操作（HTTP {exc.code}）" + (f"：{detail[:240]}" if detail else "")) from None
    except (URLError, TimeoutError, OSError):
        raise ServiceError(503, "无法连接 RAGFlow") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ServiceError(502, "RAGFlow 返回的数据格式无法识别") from None
    if not isinstance(result, dict):
        raise ServiceError(502, "RAGFlow 返回的数据格式无法识别")
    if result.get("code", 0) not in (0, None):
        message = str(result.get("message") or "RAGFlow 请求失败")[:240]
        raise ServiceError(502, f"RAGFlow 操作失败：{message}")
    return result


def _write_api_multipart(path: str, filename: str, content: bytes, content_type: str) -> dict:
    if not RAGFLOW_KEY:
        raise ServiceError(503, "RAGFLOW_API_KEY 未配置")
    boundary = "----Harness" + secrets.token_hex(16)
    safe_filename = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = Request(
        f"{RAGFLOW_BASE}{path}", data=body,
        headers={"Authorization": f"Bearer {RAGFLOW_KEY}", "Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ServiceError(exc.code, "RAGFlow API Key 无法执行沙盒操作") from None
        raise ServiceError(502, f"RAGFlow 上传失败（HTTP {exc.code}）") from None
    except (URLError, TimeoutError, OSError):
        raise ServiceError(503, "上传时无法连接 RAGFlow") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ServiceError(502, "RAGFlow 上传响应无法识别") from None
    if not isinstance(result, dict) or result.get("code", 0) not in (0, None):
        message = str(result.get("message") or "RAGFlow 上传失败") if isinstance(result, dict) else "RAGFlow 上传响应无法识别"
        raise ServiceError(502, message[:240])
    return result


def get_datasets(*, force: bool = False) -> list[dict]:
    global _datasets_cache
    now = time.monotonic()
    with _cache_lock:
        if not force and _datasets_cache and now - _datasets_cache[0] < CACHE_SECONDS:
            return _datasets_cache[1]
    data = _api_data("/api/v1/datasets", params={"page": 1, "page_size": 100})
    if not isinstance(data, list):
        raise ServiceError(502, "RAGFlow 知识库列表格式无法识别")
    datasets = [
        {
            "id": str(item.get("id", "")),
            "name": str(item.get("name", "未命名知识库")),
            "description": str(item.get("description", "") or ""),
            "document_count": _int(item.get("document_count")),
            "chunk_count": _int(item.get("chunk_count")),
            "token_count": _int(item.get("token_num")),
            "status": str(item.get("status", "")),
            "chunk_method": str(item.get("chunk_method", "")),
            "update_time": _int(item.get("update_time")),
            "is_sandbox": _is_sandbox_record(item),
        }
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]
    with _cache_lock:
        _datasets_cache = (time.monotonic(), datasets)
    return datasets


def _raw_datasets() -> list[dict]:
    data = _api_data("/api/v1/datasets", params={"page": 1, "page_size": 100})
    if not isinstance(data, list):
        raise ServiceError(502, "RAGFlow 知识库列表格式无法识别")
    return [item for item in data if isinstance(item, dict) and item.get("id")]


def _is_sandbox_record(item: dict) -> bool:
    return str(item.get("name") or "").startswith(SANDBOX_PREFIX) and SANDBOX_MARKER in str(item.get("description") or "")


def _sandbox_dataset(dataset_id: str) -> dict:
    if not _safe_id(dataset_id):
        raise ServiceError(400, "知识库标识无效")
    match = next((item for item in _raw_datasets() if str(item.get("id")) == dataset_id), None)
    if not match:
        raise ServiceError(404, "找不到这个知识库")
    if not _is_sandbox_record(match):
        raise ServiceError(403, "该操作只允许写入 Harness 新建的沙盒知识库")
    if dataset_id not in _registered_sandbox_ids():
        raise ServiceError(403, "这个库不属于本工作区创建的目标库，请新建入库目标")
    return match


def list_sandboxes() -> list[dict]:
    owned = _registered_sandbox_ids()
    return [item for item in get_datasets() if item["is_sandbox"] and item["id"] in owned]


def _portable_parser_config(value: dict) -> dict:
    """Keep only ParserConfig fields accepted by the installed RAGFlow API."""
    allowed = {
        "auto_keywords", "auto_questions", "chunk_token_num", "delimiter", "graphrag",
        "html4excel", "layout_recognize", "parent_child", "enable_children",
        "children_delimiter", "raptor", "topn_tags", "filename_embd_weight",
        "task_page_size", "pages", "compilation_template_group_id", "ext",
        "table_column_mode", "table_column_roles", "table_column_names",
    }
    return {key: item for key, item in value.items() if key in allowed}


def create_sandbox(template_id: str) -> dict:
    if not _safe_id(template_id):
        raise ServiceError(400, "模板知识库标识无效")
    template = next((item for item in _raw_datasets() if str(item.get("id")) == template_id), None)
    if not template:
        raise ServiceError(404, "找不到所选模板知识库")
    embedding_model = str(template.get("embedding_model") or "")
    if not embedding_model or embedding_model.startswith("FILL_"):
        raise ServiceError(503, "所选模板没有可用的 embedding 模型，无法创建沙盒")
    name = f"{SANDBOX_PREFIX}{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    parser_config = _portable_parser_config(template.get("parser_config") if isinstance(template.get("parser_config"), dict) else {})
    parser_note = ""
    if "mineru" in str(parser_config.get("layout_recognize") or "").casefold():
        mineru = _service_snapshot().get("mineru") or {}
        if mineru.get("status") != "online":
            parser_config["layout_recognize"] = "DeepDOC"
            parser_note = "MinerU 不可用，已为新沙盒切换到 RAGFlow DeepDOC。"
    created = _write_api_json("/api/v1/datasets", {
        "name": name,
        "description": f"{SANDBOX_MARKER} 隔离知识库；解析模板：{template.get('name', '')}" + ("；PDF 解析器：DeepDOC（MinerU 不可用）" if parser_note else ""),
        "embedding_model": embedding_model,
        "permission": "me",
        "chunk_method": str(template.get("chunk_method") or "naive"),
        "parser_config": parser_config,
    })
    data = created.get("data")
    if not isinstance(data, dict) or not data.get("id"):
        raise ServiceError(502, "RAGFlow 已响应，但没有返回沙盒知识库 ID")
    _register_sandbox(str(data["id"]))
    global _datasets_cache, _summary_cache
    with _cache_lock:
        _datasets_cache = None
        _summary_cache = None
    return {"id": str(data["id"]), "name": str(data.get("name") or name), "is_sandbox": True, "template_name": str(template.get("name") or ""), "parser_note": parser_note}


def _safe_filename(value: str) -> str:
    base = str(value or "document").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "_", base).strip(" .")
    if not base:
        base = "document.txt"
    if len(base) > 180:
        suffix = Path(base).suffix[:16]
        base = base[: 180 - len(suffix)] + suffix
    if Path(base).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ServiceError(415, f"不支持的文件格式：{Path(base).suffix or '无扩展名'}")
    return base


def _trigger_parse(dataset_id: str, doc_ids: list[str]) -> None:
    _sandbox_dataset(dataset_id)
    _require_model_gate(dataset_id)
    if not doc_ids or any(not _safe_id(doc_id) for doc_id in doc_ids):
        raise ServiceError(400, "待解析文档标识无效")
    owned_ids: set[str] = set()
    page = 1
    while page <= 100:
        docs, total = _fetch_document_page(dataset_id, page, PAGE_SIZE_MAX)
        owned_ids.update(str(item.get("id")) for item in docs if isinstance(item, dict) and item.get("id"))
        if len(owned_ids) >= total or len(docs) < PAGE_SIZE_MAX:
            break
        page += 1
    if not set(doc_ids).issubset(owned_ids):
        raise ServiceError(403, "解析请求包含不属于当前沙盒知识库的文档")
    _write_api_json(f"/api/v1/datasets/{dataset_id}/chunks", {"document_ids": doc_ids}, timeout=60)
    global _summary_cache
    with _cache_lock:
        _summary_cache = None


def _upload_to_sandbox(dataset_id: str, filename: str, content: bytes, *, wait_for_parse: bool = True, on_uploaded=None) -> dict:
    dataset = _sandbox_dataset(dataset_id)
    filename = _safe_filename(filename)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ServiceError(413, "单个文件不能为空且不能超过 50 MB")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    response = _write_api_multipart(
        f"/api/v1/datasets/{dataset_id}/documents", filename, content, content_type,
    )
    raw_docs = response.get("data") or []
    if isinstance(raw_docs, dict):
        raw_docs = [raw_docs]
    docs = [item for item in raw_docs if isinstance(item, dict) and item.get("id")]
    if not docs:
        raise ServiceError(502, "文件已发送，但 RAGFlow 没有返回文档 ID")
    doc_ids = [str(item["id"]) for item in docs]
    if on_uploaded:
        on_uploaded(doc_ids)
    if not wait_for_parse:
        return {
            "dataset": {"id": dataset_id, "name": str(dataset.get("name") or "")},
            "items": [{"id": str(item["id"]), "name": str(item.get("name") or filename)} for item in docs],
            "parse_started": False,
            "message": "已上传到沙盒；模型未确认，解析暂缓",
        }
    try:
        _trigger_parse(dataset_id, doc_ids)
        parse_started = True
        message = "已上传到沙盒并启动解析"
    except ServiceError as exc:
        parse_started = False
        message = f"已上传到沙盒；解析未启动：{exc.message}"
    with _cache_lock:
        _documents_cache.pop(dataset_id, None)
    return {
        "dataset": {"id": dataset_id, "name": str(dataset.get("name") or "")},
        "items": [{"id": str(item["id"]), "name": str(item.get("name") or filename)} for item in docs],
        "parse_started": parse_started,
        "message": message,
    }


def _public_http_url(value: str) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
        host = parsed.hostname or ""
        if parsed.scheme.lower() not in {"http", "https"} or not host or parsed.username or parsed.password:
            raise ValueError
        if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local", ".internal")):
            raise ValueError
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            addresses = [ipaddress.ip_address(entry[4][0]) for entry in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)]
        if not addresses:
            raise ValueError
        if any(not address.is_global for address in addresses):
            # Some managed outbound proxies intentionally map public DNS names
            # to RFC 2544 benchmarking addresses, then resolve them upstream.
            # Accept that synthetic range only when an explicit proxy is set;
            # private, loopback, link-local, and other reserved ranges stay blocked.
            proxy_configured = bool(urllib.request.getproxies().get(parsed.scheme.lower()))
            synthetic_proxy_range = ipaddress.ip_network("198.18.0.0/15")
            if not proxy_configured or any(address not in synthetic_proxy_range for address in addresses):
                raise ValueError
    except (ValueError, OSError, socket.gaierror):
        raise ServiceError(400, "来源链接无效，或指向本机/内网地址") from None
    return parsed.geturl()


def _download_public_pdf(value: str) -> tuple[str, bytes]:
    url = _public_http_url(value)

    class PublicRedirects(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            _public_http_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    request = Request(url, headers={"Accept": "application/pdf,application/octet-stream;q=0.8,*/*;q=0.2", "User-Agent": "KnowledgeIngestionWorkbench/1.0"})
    try:
        with build_opener(PublicRedirects).open(request, timeout=45) as response:
            final_url = _public_http_url(response.geturl())
            length = _int(response.headers.get("Content-Length"))
            if length > MAX_UPLOAD_BYTES:
                raise ServiceError(413, "远程文件超过 50 MB")
            content = response.read(MAX_UPLOAD_BYTES + 1)
            content_type = response.headers.get_content_type()
    except ServiceError:
        raise
    except HTTPError as exc:
        raise ServiceError(502, f"远程下载失败（HTTP {exc.code}）") from None
    except (URLError, TimeoutError, OSError):
        raise ServiceError(502, "无法下载这个公开来源") from None
    if len(content) > MAX_UPLOAD_BYTES:
        raise ServiceError(413, "远程文件超过 50 MB")
    if not content.startswith(b"%PDF-"):
        raise ServiceError(415, f"在线来源返回的不是 PDF（{content_type}）")
    filename = Path(urlsplit(final_url).path).name or "open-access-paper.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return _safe_filename(filename), content


def discover_sources(query: str, limit: int = 8) -> dict:
    query = " ".join(str(query or "").split())[:300]
    if len(query) < 2:
        raise ServiceError(400, "请输入至少 2 个字符的主题或检索词")
    limit = max(1, min(int(limit or 8), 20))
    # Reuse the repository's existing public-literature connectors. Import lazily
    # so read-only inventory remains available without the ingestion package.
    ingest_path = str(REPO_ROOT / "examples" / "electromagnetics" / "ingestion")
    if ingest_path not in os.sys.path:
        os.sys.path.insert(0, ingest_path)
    try:
        from sync_library import ArxivDiscoverer, OpenAlexDiscoverer
        sources = [("OpenAlex", OpenAlexDiscoverer()), ("arXiv", ArxivDiscoverer())]
    except Exception:
        raise ServiceError(503, "本地文献发现模块无法加载") from None
    candidates: dict[str, dict] = {}
    errors: list[str] = []

    def search_source(entry):
        label, discoverer = entry
        try:
            found = []
            for candidate in discoverer.search(query, limit=limit):
                if not candidate.pdf_url:
                    continue
                found.append({
                    "id": candidate.candidate_id,
                    "source": candidate.source or label.lower(),
                    "title": candidate.title,
                    "authors": candidate.authors,
                    "year": candidate.year,
                    "doi": candidate.doi,
                    "abstract": candidate.abstract[:1400],
                    "landing_url": candidate.landing_url,
                    "pdf_url": candidate.pdf_url,
                    "citation_count": _int(candidate.citation_count),
                })
            return label, found
        except Exception:
            return label, None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="source-search") as pool:
        for label, found in pool.map(search_source, sources):
            if found is None:
                errors.append(label)
                continue
            for item in found:
                key = str(item.get("doi") or item.get("id") or "").casefold()
                current = candidates.get(key)
                if not current or (item["abstract"] and not current["abstract"]):
                    candidates[key] = item
    result = sorted(candidates.values(), key=lambda item: item["citation_count"], reverse=True)[: limit * 2]
    return {"items": result, "query": query, "sources": ["OpenAlex", "arXiv"], "errors": errors}


def _job_snapshot(job_id: str) -> dict | None:
    with _ingestion_jobs_lock:
        job = _ingestion_jobs.get(job_id)
        return copy.deepcopy(job) if job else None


def _update_job(job_id: str, **changes) -> None:
    with _ingestion_jobs_lock:
        job = _ingestion_jobs.get(job_id)
        if not job:
            return
        job.update(changes)
        job["updated_at"] = int(time.time())
        HISTORY.record(job)


def _update_job_item(job_id: str, item_id: str, **changes) -> None:
    with _ingestion_jobs_lock:
        job = _ingestion_jobs.get(job_id)
        if not job:
            return
        for item in job["items"]:
            if item["id"] == item_id:
                item.update(changes)
                job["updated_at"] = int(time.time())
                HISTORY.record(job)
                return


def _update_item_check(job_id: str, item_id: str, name: str, state: str, detail: str) -> None:
    with _ingestion_jobs_lock:
        job = _ingestion_jobs.get(job_id)
        if not job:
            return
        for item in job["items"]:
            if item["id"] == item_id:
                item.setdefault("checks", {})[name] = {"state": state, "detail": detail}
                job["updated_at"] = int(time.time())
                HISTORY.record(job)
                return


def list_ingestion_jobs() -> list[dict]:
    with _ingestion_jobs_lock:
        jobs = [copy.deepcopy(item) for item in _ingestion_jobs.values()]
    for item in jobs:
        if item["status"] in {"completed", "completed_with_review", "failed"}:
            HISTORY.record(item)
    jobs.sort(key=lambda item: item["created_at"], reverse=True)
    return jobs[:20]


def restore_recoverable_jobs() -> list[dict]:
    jobs = HISTORY.snapshot().values()
    with _ingestion_jobs_lock:
        active_ids = {key for key, item in _ingestion_jobs.items() if item.get("status") in {"running", "queued"}}
    return sorted([dict(item, restored=True) for item in jobs if item.get("id") not in active_ids],
                  key=lambda item: int(item.get("updated_at") or 0), reverse=True)[:50]


def load_interrupted_jobs() -> list[dict]:
    return HISTORY.load_interrupted()


def _safe_resume_job(stored: dict) -> dict:
    result = copy.deepcopy(stored)
    uploaded = any(item.get("document_ids") for item in result.get("items", []))
    result.update(resumable=uploaded, resumed=uploaded, lost_local_bytes=stored.get("mode") == "local")
    if uploaded:
        result.update(status="queued", stage="resume", message="继续检查已上传文档的解析与实际分块", updated_at=int(time.time()))
        for item in result.get("items", []):
            if item.get("document_ids"):
                item.update(status="running", stage="parse", message="恢复结果核验", progress=58)
            elif item.get("status") in {"queued", "running"}:
                item.update(status="needs_review", stage="acquire", message="尚无已保存文档标识，请重新选择本地文件或重新采集来源", progress=100)
    else:
        result["message"] = "本地原始字节已丢失，请重新选择文件" if stored.get("mode") == "local" else "尚无已上传文档，请重新启动来源采集"
    return result


def _start_resume(job_id: str) -> dict:
    with _ingestion_jobs_lock:
        live = _ingestion_jobs.get(job_id)
        if live and live.get("status") in {"queued", "running"}:
            raise ServiceError(409, "任务正在执行，无需重复恢复")
        stored = HISTORY.snapshot().get(job_id)
        if not stored:
            raise ServiceError(404, "找不到已保存任务")
        _sandbox_dataset(str(stored.get("dataset_id") or ""))
        resume = _safe_resume_job(stored)
        if not resume["resumable"]:
            raise ServiceError(409, resume["message"])
        if sum(j.get("status") in {"queued", "running"} for j in _ingestion_jobs.values()) >= 2:
            raise ServiceError(429, "并行队列已满")
        _ingestion_jobs[job_id] = resume
        HISTORY.record(resume)
    _ingestion_executor.submit(_review_uploaded_documents, job_id, True)
    return {"resumed": True, "job": _job_snapshot(job_id)}


def _normalized_title(value: str) -> str:
    stem = Path(str(value or "")).stem.casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", stem, flags=re.UNICODE)


def _new_job(dataset_id: str, mode: str, *, query: str = "", limit: int = 0, parallelism: int = 3, files: list[tuple[str, bytes]] | None = None, allow_unconfirmed_gate: bool = False) -> dict:
    dataset = _sandbox_dataset(dataset_id)
    if parallelism < 1 or parallelism > 6:
        raise ServiceError(400, "并行度须在 1 到 6 之间")
    gate = _load_model_gate_file(dataset_id)
    with _ingestion_jobs_lock:
        active = sum(job["status"] in {"queued", "running"} for job in _ingestion_jobs.values())
        if active >= 2:
            raise ServiceError(429, "并行入库队列已满，请等待当前任务完成")
        job_id = secrets.token_hex(12)
        job = {
            "id": job_id,
            "dataset_id": dataset_id,
            "dataset_name": str(dataset.get("name") or "隔离知识库"),
            "mode": mode,
            "query": query,
            "limit": limit,
            "parallelism": parallelism,
            "status": "queued",
            "stage": "queued",
            "message": "任务排队中",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "items": [],
            "errors": [],
            "local_file_count": len(files or []) if files else 0,
            "lost_local_bytes": False,
            "model_configuration": gate,
            "model_gate": {
                "ready": bool(gate.get("ready")),
                "provider": str(gate.get("provider") or ""),
                "version": str(gate.get("version") or ""),
            },
        }
        if not gate.get("ready") and not allow_unconfirmed_gate:
            # Read-only review jobs may still be created; parsing will be
            # deferred (not triggered) until the model gate is confirmed.
            job["message"] = "任务排队中；模型配置未确认，将跳过解析触发"
            job["stage"] = "queued"
        _ingestion_jobs[job_id] = job
        HISTORY.record(job)
        # Keep recent history bounded without removing active jobs.
        terminal = [key for key, value in _ingestion_jobs.items() if value["status"] not in {"queued", "running"}]
        for old_id in terminal[:-20]:
            _ingestion_jobs.pop(old_id, None)
    safe_files = [(str(name), bytes(content)) for name, content in (files or [])]
    _ingestion_executor.submit(_run_ingestion_job, job_id, safe_files)
    return _job_snapshot(job_id) or job


def _run_ingestion_job(job_id: str, local_files: list[tuple[str, bytes]]) -> None:
    job = _job_snapshot(job_id)
    if not job:
        return
    dataset_id = job["dataset_id"]
    try:
        _sandbox_dataset(dataset_id)
        _update_job(job_id, status="running", stage="source_review" if local_files else "discover", message="正在准备资料")
        existing = _all_documents(dataset_id, force=True)
        known_titles = {_normalized_title(str(doc.get("name") or "")) for doc in existing}
        items: list[dict] = []

        if local_files:
            seen = set(known_titles)
            local_content_by_id: dict[str, bytes] = {}
            for index, (filename, content) in enumerate(local_files):
                title = _safe_filename(filename)
                key = _normalized_title(title)
                item = {
                    "id": f"local-{index}-{secrets.token_hex(3)}",
                    "title": title,
                    "source": "本地文件",
                    "status": "queued",
                    "stage": "source_review",
                    "message": "等待审查",
                    "progress": 5,
                    "document_ids": [],
                    "checks": {
                        "source": {"state": "waiting", "detail": "检查文件名与重复项"},
                        "file": {"state": "waiting", "detail": "检查格式、大小与内容"},
                        "parse": {"state": "waiting", "detail": "等待解析"},
                        "chunks": {"state": "waiting", "detail": "等待分块核验"},
                    },
                }
                if key in seen:
                    item.update(status="skipped", stage="source_review", message="同名资料已存在", progress=100)
                    item["checks"]["source"] = {"state": "needs_review", "detail": "当前沙盒中存在同名文档"}
                    item["checks"]["file"] = {"state": "skipped", "detail": "未上传"}
                    seen.add(key)
                else:
                    seen.add(key)
                    item["checks"]["source"] = {"state": "passed", "detail": "文件名有效，沙盒中无同名文档"}
                    if not content or len(content) > MAX_UPLOAD_BYTES:
                        item.update(status="failed", stage="source_review", message="文件为空或超过 50 MB", progress=100)
                        item["checks"]["file"] = {"state": "failed", "detail": "单文件大小须为 1 B 至 50 MB"}
                    elif Path(title).suffix.lower() not in ALLOWED_EXTENSIONS:
                        item.update(status="failed", stage="source_review", message="文件格式不支持", progress=100)
                        item["checks"]["file"] = {"state": "failed", "detail": "格式不在支持范围内"}
                    else:
                        item["content_bytes"] = len(content)
                        item["checks"]["file"] = {"state": "passed", "detail": f"{len(content):,} B · {Path(title).suffix.lower()}"}
                        local_content_by_id[item["id"]] = content
                items.append(item)
            _update_job(job_id, items=[{k: v for k, v in item.items() if k != "content_bytes"} for item in items], stage="acquire", message="本地资料审查完成，开始并行采集")
        else:
            _update_job(job_id, stage="discover", message="并行检索 OpenAlex 与 arXiv")
            found = discover_sources(job["query"], int(job["limit"]))
            candidates = found["items"][: int(job["limit"])]
            if not candidates:
                _update_job(job_id, status="failed", stage="discover", message="未发现可获取的公开 PDF，请调整主题或选择本地资料", errors=found.get("errors") or ["公开来源检索无结果"])
                return
            seen = set(known_titles)
            for candidate in candidates:
                title = str(candidate.get("title") or "").strip()[:500]
                key = _normalized_title(title)
                item = {
                    "id": str(candidate.get("id") or secrets.token_hex(6))[:120],
                    "title": title or "未命名文献",
                    "source": str(candidate.get("source") or "开放文献"),
                    "year": _int(candidate.get("year")),
                    "doi": str(candidate.get("doi") or "")[:200],
                    "citation_count": _int(candidate.get("citation_count")),
                    "status": "queued",
                    "stage": "source_review",
                    "message": "来源审查中",
                    "progress": 8,
                    "document_ids": [],
                    "checks": {
                        "source": {"state": "running", "detail": "校验开放 PDF 来源与重复项"},
                        "file": {"state": "waiting", "detail": "等待下载校验"},
                        "parse": {"state": "waiting", "detail": "等待解析"},
                        "chunks": {"state": "waiting", "detail": "等待分块核验"},
                    },
                }
                if not title:
                    item.update(status="failed", message="缺少文献标题", progress=100)
                    item["checks"]["source"] = {"state": "failed", "detail": "来源记录没有标题"}
                elif key in seen:
                    item.update(status="skipped", message="同名文献已存在", progress=100)
                    item["checks"]["source"] = {"state": "needs_review", "detail": "当前沙盒或本批次存在同名文献"}
                    item["checks"]["file"] = {"state": "skipped", "detail": "未下载"}
                else:
                    seen.add(key)
                    try:
                        item["pdf_url"] = _public_http_url(str(candidate.get("pdf_url") or ""))
                        item["checks"]["source"] = {"state": "passed", "detail": "公开 HTTP(S) PDF 来源有效，未发现重复标题"}
                    except ServiceError as exc:
                        item.update(status="failed", message=exc.message, progress=100)
                        item["checks"]["source"] = {"state": "failed", "detail": exc.message}
                items.append(item)
            errors = found.get("errors") or []
            source_note = f" · 暂不可用：{', '.join(errors)}" if errors else ""
            _update_job(job_id, items=items, errors=errors, stage="source_review", message=f"来源审查完成 · {len(items)} 条记录{source_note}")

        # Only filenames and metadata are published in the job snapshot.
        uploadable = [item for item in items if item["status"] == "queued"]
        file_map = local_content_by_id if local_files else {}
        _update_job(job_id, stage="acquire", message=f"并行采集 · {len(uploadable)} 项 · 并行度 {job['parallelism']}")

        def acquire(item: dict) -> None:
            item_id = item["id"]
            _update_job_item(job_id, item_id, stage="acquire", message="采集中", progress=20)
            try:
                if local_files:
                    content = file_map[item_id]
                    filename = item["title"]
                else:
                    filename, content = _download_public_pdf(item["pdf_url"])
                    title_stem = re.sub(r"[\\/:*?\"<>|]+", "_", item["title"]).strip(" .")[:150] or "open-access-paper"
                    filename = _safe_filename(f"{title_stem}.pdf")
                _update_item_check(job_id, item_id, "file", "passed", f"PDF / {len(content):,} B" if filename.lower().endswith(".pdf") else f"{Path(filename).suffix.lower()} / {len(content):,} B")
                _update_job_item(job_id, item_id, message="上传至隔离知识库", progress=45)
                gate_now = _load_model_gate_file(dataset_id)
                result = _upload_to_sandbox(dataset_id, filename, content, wait_for_parse=bool(gate_now.get("ready")), on_uploaded=lambda ids: _update_job_item(job_id, item_id, document_ids=ids))
                doc_ids = [str(doc.get("id")) for doc in result.get("items", []) if doc.get("id")]
                if not doc_ids:
                    raise ServiceError(502, "RAGFlow 未返回文档标识")
                _update_job_item(job_id, item_id, status="running", stage="parse", message="已入库，等待解析", progress=58, document_ids=doc_ids)
                _update_item_check(
                    job_id, item_id, "parse",
                    "running" if result.get("parse_started") else "needs_review",
                    "解析任务已启动" if result.get("parse_started") else str(result.get("message") or "解析任务未启动"),
                )
            except ServiceError as exc:
                _update_job_item(job_id, item_id, status="failed", stage="acquire", message=exc.message, progress=100)
                _update_item_check(job_id, item_id, "file", "failed", exc.message)
            except Exception:
                _update_job_item(job_id, item_id, status="failed", stage="acquire", message="采集失败", progress=100)
                _update_item_check(job_id, item_id, "file", "failed", "下载或上传失败")

        if uploadable:
            with ThreadPoolExecutor(max_workers=int(job["parallelism"]), thread_name_prefix="ingest-item") as pool:
                list(pool.map(acquire, uploadable))
        _update_job(job_id, stage="parse", message="等待 RAGFlow 解析完成")

        _review_uploaded_documents(job_id)
    except ServiceError as exc:
        _update_job(job_id, status="failed", stage="failed", message=exc.message, errors=[exc.message])
        HISTORY.record(_job_snapshot(job_id) or {})
    except Exception:
        _update_job(job_id, status="failed", stage="failed", message="工作流执行失败", errors=["工作流执行失败"])
        HISTORY.record(_job_snapshot(job_id) or {})


def _review_uploaded_documents(job_id: str, read_only: bool = False) -> None:
    job = _job_snapshot(job_id) or {}
    dataset_id = str(job.get("dataset_id") or "")
    try:
        _sandbox_dataset(dataset_id)
        _update_job(job_id, status="running", stage="parse", message="核验远端文档状态")
        tracked = [item for item in (_job_snapshot(job_id) or {}).get("items", []) if item.get("status") == "running" and item.get("document_ids")]
        pending_ids = {doc_id for item in tracked for doc_id in item["document_ids"]}
        run_by_id: dict[str, dict] = {}
        retried: set[str] = set()
        deadline = time.monotonic() + 600
        while pending_ids and time.monotonic() < deadline:
            _sandbox_dataset(dataset_id)
            docs, total_docs = _fetch_document_page(dataset_id, 1, PAGE_SIZE_MAX)
            docs = list(docs)
            pages = min(100, (total_docs + PAGE_SIZE_MAX - 1) // PAGE_SIZE_MAX)
            page = 2
            while page <= pages and not pending_ids.issubset({str(doc.get("id")) for doc in docs if doc.get("id")}):
                more_docs, _ = _fetch_document_page(dataset_id, page, PAGE_SIZE_MAX)
                docs.extend(more_docs)
                if len(more_docs) < PAGE_SIZE_MAX:
                    break
                page += 1
            run_by_id = {str(doc.get("id")): doc for doc in docs if doc.get("id")}
            terminal = {"DONE", "FAIL", "CANCEL"}
            for item in tracked:
                ids = item["document_ids"]
                found_docs = [run_by_id[doc_id] for doc_id in ids if doc_id in run_by_id]
                for doc in found_docs:
                    doc_id = str(doc.get("id"))
                    run = str(doc.get("run") or "UNKNOWN").upper()
                    if not read_only and run == "UNSTART" and doc_id not in retried and time.monotonic() + 580 > deadline:
                        if not _load_model_gate_file(dataset_id).get("ready"):
                            retried.add(doc_id)
                            continue
                        try:
                            _trigger_parse(dataset_id, [doc_id])
                        except ServiceError:
                            pass
                        retried.add(doc_id)
                    if run in terminal or (read_only and run == "UNSTART"):
                        pending_ids.discard(doc_id)
                    if read_only and run == "UNSTART":
                        _update_job_item(job_id, item["id"], status="needs_review", stage="parse", message="远端文档尚未启动解析", progress=85)
                        _update_item_check(job_id, item["id"], "parse", "needs_review", "恢复仅核验现有结果，不重新触发解析")
                if len(found_docs) == len(ids):
                    runs = [str(doc.get("run") or "UNKNOWN").upper() for doc in found_docs]
                    if all(run in terminal for run in runs):
                        done = all(run == "DONE" for run in runs)
                        empty_docs = [doc for doc in found_docs if "chunk_count" in doc and _int(doc.get("chunk_count")) == 0]
                        diagnostics = " ".join(str(doc.get("progress_msg") or "") for doc in found_docs).casefold()
                        parser_error = "[error]" in diagnostics or "no chunk built" in diagnostics
                        okay = done and not empty_docs and not parser_error
                        if not done:
                            detail = "存在解析失败或取消项"
                        elif "mineru not found" in diagnostics:
                            detail = "MinerU 解析器不可用，RAGFlow 未生成分块"
                        elif empty_docs or parser_error:
                            detail = "RAGFlow 任务已结束，但没有生成可用分块"
                        else:
                            detail = "全部文档解析完成并生成分块"
                        _update_job_item(job_id, item["id"], status="running" if done else "needs_review", stage="chunk_review" if done else "parse", message="解析完成，检查分块" if done else "解析未全部完成", progress=76 if done else 100)
                        _update_item_check(job_id, item["id"], "parse", "passed" if okay else "needs_review" if done else "failed", detail)
                    else:
                        messages = [line.strip() for doc in found_docs for line in str(doc.get("progress_msg") or "").splitlines() if line.strip()]
                        latest = messages[-1][:180] if messages else "RAGFlow 正在解析"
                        _update_item_check(job_id, item["id"], "parse", "running", latest)
                        _update_job_item(job_id, item["id"], message=latest, progress=60)
            _update_job(job_id, stage="parse", message=f"解析处理中 · 剩余 {len(pending_ids)} 个文档")
            if pending_ids:
                time.sleep(3)

        if pending_ids:
            for item in tracked:
                if any(doc_id in pending_ids for doc_id in item["document_ids"]):
                    _update_job_item(job_id, item["id"], status="needs_review", stage="parse", message="解析仍在进行", progress=85)
                    _update_item_check(job_id, item["id"], "parse", "needs_review", "等待超时，文档仍在解析")

        _update_job(job_id, stage="chunk_review", message="读取实际解析分块并执行规则核验")
        for item in tracked:
            current = next((entry for entry in (_job_snapshot(job_id) or {}).get("items", []) if entry["id"] == item["id"]), {})
            if current.get("stage") != "chunk_review":
                continue
            sampled, nonempty, long_chunks = 0, 0, 0
            try:
                for doc_id in item["document_ids"]:
                    payload = get_document_chunks(dataset_id, doc_id, {"page": ["1"], "page_size": ["5"]})
                    chunks = payload.get("items") or []
                    sampled += len(chunks)
                    nonempty += sum(bool(str(chunk.get("content") or "").strip()) for chunk in chunks)
                    long_chunks += sum(len(str(chunk.get("content") or "").strip()) >= 40 for chunk in chunks)
                okay = sampled > 0 and nonempty == sampled and long_chunks / max(1, sampled) >= 0.5
                detail = f"抽查 {sampled} 个实际分块 · {long_chunks} 个长度不少于 40 字符"
                _update_item_check(job_id, item["id"], "chunks", "passed" if okay else "needs_review", detail if sampled else "未读取到分块")
                parse_passed = current.get("checks", {}).get("parse", {}).get("state") == "passed"
                final_pass = okay and parse_passed
                _update_job_item(job_id, item["id"], status="completed" if final_pass else "needs_review", stage="completed", message="分块核验通过" if final_pass else "存在需复核项", progress=100)
            except ServiceError as exc:
                _update_item_check(job_id, item["id"], "chunks", "needs_review", exc.message)
                _update_job_item(job_id, item["id"], status="needs_review", stage="completed", message="分块核验未完成", progress=100)

        snapshot = _job_snapshot(job_id) or {}
        counts = {key: sum(item.get("status") == key for item in snapshot.get("items", [])) for key in ("completed", "needs_review", "failed", "skipped")}
        terminal_state = "completed" if not pending_ids and counts["needs_review"] == 0 and counts["failed"] == 0 else "completed_with_review"
        source_note = f" · 来源不可用：{', '.join(snapshot.get('errors') or [])}" if snapshot.get("errors") else ""
        _update_job(job_id, status=terminal_state, stage="completed", message=f"流程完成 · 通过 {counts['completed']} · 复核 {counts['needs_review']} · 失败 {counts['failed']} · 跳过 {counts['skipped']}{source_note}")
        HISTORY.record(_job_snapshot(job_id) or {})
        with _cache_lock:
            _documents_cache.pop(dataset_id, None)
            global _summary_cache
            _summary_cache = None
    except ServiceError as exc:
        _update_job(job_id, status="failed", stage="failed", message=exc.message, errors=[exc.message])
    except Exception as exc:
        _update_job(job_id, status="failed", stage="failed", message="结果核验失败", errors=[type(exc).__name__])


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _serialize_document(item: dict) -> dict:
    """Keep the API response useful while omitting local storage paths."""
    provenance = {}
    with _ingestion_jobs_lock:
        for job in _ingestion_jobs.values():
            for entry in job.get("items", []):
                if str(item.get("id")) in entry.get("document_ids", []):
                    provenance = entry
                    break
    return {
        "id": str(item.get("id", "")),
        "name": str(item.get("name", "未命名文档")),
        "type": str(item.get("type", "")),
        "suffix": str(item.get("suffix", "")),
        "size": _int(item.get("size")),
        "chunk_count": _int(item.get("chunk_count")),
        "token_count": _int(item.get("token_count")),
        "run": _document_run_state(item),
        "progress": _int(item.get("progress")),
        "status": str(item.get("status", "")),
        "source_type": str(provenance.get("source") or item.get("source_type", "")),
        "source_url": str(provenance.get("pdf_url") or ""),
        "progress_message": str(item.get("progress_msg", "") or ""),
        "create_time": _int(item.get("create_time")),
        "update_time": _int(item.get("update_time")),
    }


def _document_run_state(item: dict) -> str:
    run = str(item.get("run") or "UNKNOWN").upper()
    if run == "DONE" and "chunk_count" in item and _int(item.get("chunk_count")) == 0:
        return "NO_CHUNKS"
    return run


def _serialize_chunk(item: dict, limit: int = 1400) -> dict:
    content = str(item.get("content", "") or "")
    positions = item.get("positions") or []
    first_position = positions[0] if isinstance(positions, list) and positions else None
    page = first_position[0] if isinstance(first_position, (list, tuple)) and first_position else first_position
    return {
        "id": str(item.get("id", "")),
        "content": content[:limit],
        "truncated": len(content) > limit,
        "page": page,
        "keywords": [str(value) for value in (item.get("important_keywords") or [])[:8]],
    }


def _safe_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value))


def _get_dataset(dataset_id: str) -> dict:
    if not _safe_id(dataset_id):
        raise ServiceError(400, "知识库标识无效")
    dataset = next((item for item in get_datasets() if item["id"] == dataset_id), None)
    if not dataset:
        raise ServiceError(404, "找不到这个知识库")
    return dataset


def _fetch_document_page(dataset_id: str, page: int, page_size: int) -> tuple[list[dict], int]:
    data = _api_data(
        f"/api/v1/datasets/{dataset_id}/documents",
        params={"page": page, "page_size": page_size},
    )
    if not isinstance(data, dict) or not isinstance(data.get("docs"), list):
        raise ServiceError(502, "RAGFlow 文档列表格式无法识别")
    return data["docs"], _int(data.get("total"))


def _all_documents(dataset_id: str, *, force: bool = False) -> list[dict]:
    now = time.monotonic()
    with _cache_lock:
        cached = _documents_cache.get(dataset_id)
        if not force and cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]
    first_page, total = _fetch_document_page(dataset_id, 1, PAGE_SIZE_MAX)
    documents = list(first_page)
    pages = (total + PAGE_SIZE_MAX - 1) // PAGE_SIZE_MAX
    for page in range(2, pages + 1):
        next_page, _ = _fetch_document_page(dataset_id, page, PAGE_SIZE_MAX)
        documents.extend(next_page)
    with _cache_lock:
        _documents_cache[dataset_id] = (time.monotonic(), documents)
    return documents


def get_document_page(dataset_id: str, query: dict[str, list[str]]) -> dict:
    dataset = _get_dataset(dataset_id)
    page = max(1, _int((query.get("page") or [1])[0]))
    page_size = min(100, max(5, _int((query.get("page_size") or [25])[0])))
    term = ((query.get("q") or [""])[0]).strip().casefold()[:100]
    run = ((query.get("run") or [""])[0]).strip().upper()

    if term or run:
        documents = _all_documents(dataset_id)
        if term:
            documents = [
                doc for doc in documents
                if term in " ".join(str(doc.get(key, "")) for key in ("name", "suffix", "source_type", "type")).casefold()
            ]
        if run:
            if run == "NO_CHUNKS":
                documents = [doc for doc in documents if _document_run_state(doc) == "NO_CHUNKS"]
            else:
                documents = [doc for doc in documents if _document_run_state(doc) == run]
        total = len(documents)
        start = (page - 1) * page_size
        selected = documents[start:start + page_size]
    else:
        selected, total = _fetch_document_page(dataset_id, page, page_size)

    return {
        "dataset": {"id": dataset["id"], "name": dataset["name"]},
        "items": [_serialize_document(item) for item in selected],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
        "filtered": bool(term or run),
    }


def get_document_chunks(dataset_id: str, document_id: str, query: dict[str, list[str]]) -> dict:
    dataset = _get_dataset(dataset_id)
    if not _safe_id(document_id):
        raise ServiceError(400, "文档标识无效")
    page = max(1, _int((query.get("page") or [1])[0]))
    page_size = min(10, max(1, _int((query.get("page_size") or [5])[0])))
    data = _api_data(
        f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
        params={"page": page, "page_size": page_size},
    )
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise ServiceError(502, "RAGFlow 分块内容格式无法识别")
    doc = data.get("doc") or {}
    return {
        "dataset": {"id": dataset["id"], "name": dataset["name"]},
        "document": _serialize_document(doc) if isinstance(doc, dict) else {},
        "items": [_serialize_chunk(item) for item in data["chunks"] if isinstance(item, dict)],
        "total": _int(data.get("total")),
        "page": page,
        "page_size": page_size,
    }


def get_summary(*, force: bool = False) -> dict:
    global _summary_cache
    now = time.monotonic()
    with _cache_lock:
        if not force and _summary_cache and now - _summary_cache[0] < CACHE_SECONDS:
            return _summary_cache[1]
    datasets = get_datasets(force=force)
    run_counts: dict[str, int] = {}
    completed = 0
    latest_update = 0
    for dataset in datasets:
        for item in _all_documents(dataset["id"], force=force):
            run = _document_run_state(item)
            run_counts[run] = run_counts.get(run, 0) + 1
            completed += run == "DONE"
            latest_update = max(latest_update, _int(item.get("update_time")))
    total_docs = sum(item["document_count"] for item in datasets)
    summary = {
        "dataset_count": len(datasets),
        "document_count": total_docs,
        "chunk_count": sum(item["chunk_count"] for item in datasets),
        "completed_count": completed,
        "run_counts": run_counts,
        "latest_update": latest_update,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _cache_lock:
        _summary_cache = (time.monotonic(), summary)
    return summary


def _service_snapshot() -> dict:
    ragflow: dict = {"status": "offline", "version": "", "dataset_count": 0, "message": "服务不可达"}
    try:
        version = _api_data("/api/v1/system/version", auth=False, timeout=3)
        datasets = get_datasets()
        ragflow = {
            "status": "online",
            "version": str(version or ""),
            "dataset_count": len(datasets),
            "message": "只读连接正常",
        }
    except ServiceError as exc:
        ragflow["message"] = exc.message

    mineru = {"status": "offline", "message": "解析服务不可达"}
    try:
        request = Request(MINERU_HEALTH_URL, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))
        if isinstance(health, dict):
            upstream = health.get("upstream") or {}
            mineru_status = "online" if health.get("status") == "healthy" or upstream.get("status") == "healthy" else "degraded"
            mineru = {"status": mineru_status, "message": str(health.get("status", mineru_status))}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        pass

    config_path = Path(os.environ.get("INGESTION_CONFIG", str(DEFAULT_CONFIG))).expanduser()
    return {
        "ragflow": ragflow,
        "mineru": mineru,
        "ingestion": {
            "status": "configured" if config_path.is_file() else "not_configured",
            "message": "CLI 配置已找到" if config_path.is_file() else "CLI 运行配置尚未创建",
            "http_api": "not_available",
        },
    }


class WorkbenchHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _is_local_request(self) -> bool:
        try:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname
        except ValueError:
            return False
        return host in {"127.0.0.1", "localhost", "::1"}

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _route_api(self, path: str, query: dict[str, list[str]]) -> bool:
        try:
            if path == "/api/status":
                payload = _service_snapshot()
                payload["model_gate"] = _load_model_gate_file()
                self._send_json(200, payload)
            elif path == "/api/ingestion-jobs":
                self._send_json(200, {"items": list_ingestion_jobs()})
            elif path == "/api/ingestion-jobs/restored":
                self._send_json(200, {"items": restore_recoverable_jobs()})
            elif path == "/api/ingestion-jobs/restored/confirm" and query.get("job_id"):
                job_id = str(query.get("job_id", [""])[0])
                if not _safe_id(job_id):
                    raise ServiceError(400, "任务标识无效")
                restored = next((item for item in HISTORY.snapshot().values() if str(item.get("id")) == job_id), None)
                if not restored:
                    raise ServiceError(404, "找不到这份已保存的入库任务")
                restored.pop("restored", None)
                self._send_json(200, restored)
            elif path == "/api/model-gate":
                self._send_json(200, _load_model_gate_file())
            elif path.startswith("/api/ingestion-jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                if not _safe_id(job_id):
                    raise ServiceError(400, "任务标识无效")
                job = _job_snapshot(job_id)
                if not job:
                    raise ServiceError(404, "找不到这个入库任务")
                self._send_json(200, job)
            elif path == "/api/datasets":
                self._send_json(200, {"items": get_datasets(force=(query.get("refresh") == ["1"]))})
            elif path == "/api/sandboxes":
                self._send_json(200, {"items": list_sandboxes()})
            elif path == "/api/summary":
                self._send_json(200, get_summary(force=(query.get("refresh") == ["1"])))
            else:
                parts = [part for part in path.split("/") if part]
                if len(parts) == 4 and parts[:2] == ["api", "datasets"] and parts[3] == "documents":
                    self._send_json(200, get_document_page(parts[2], query))
                elif len(parts) == 6 and parts[:2] == ["api", "datasets"] and parts[3] == "documents" and parts[5] == "chunks":
                    self._send_json(200, get_document_chunks(parts[2], parts[4], query))
                else:
                    return False
        except ServiceError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception:
            self._send_json(502, {"error": "读取知识库数据失败"})
        return True

    def _trusted_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
            host = urlsplit("//" + self.headers.get("Host", ""))
            return parsed.scheme == "http" and parsed.netloc.lower() == host.netloc.lower()
        except ValueError:
            return False

    def _request_body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ServiceError(400, "请求长度无效") from None
        if length <= 0:
            raise ServiceError(400, "请求内容为空")
        if length > limit:
            raise ServiceError(413, "请求超过本地工作台的大小限制")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ServiceError(400, "请求内容未完整接收")
        return body

    def _route_post(self, path: str, query: dict[str, list[str]]) -> bool:
        try:
            if path == "/api/model-gate/confirm":
                raise ServiceError(409, "模型配置由知识库元数据读取，无需手工确认")

            if path in {"/api/ingestion-jobs/resume", "/api/ingestion-jobs/retry"}:
                try:
                    payload = json.loads(self._request_body(1024 * 1024).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ServiceError(400, "恢复请求必须是有效 JSON") from None
                if not isinstance(payload, dict) or not _safe_id(str(payload.get("job_id") or "")):
                    raise ServiceError(400, "任务标识无效")
                self._send_json(202, _start_resume(str(payload["job_id"])))
                return True

            if path == "/api/discover":
                body = self._request_body(1024 * 1024)
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ServiceError(400, "检索请求必须是有效 JSON") from None
                if not isinstance(payload, dict):
                    raise ServiceError(400, "检索请求必须是对象")
                result = discover_sources(str(payload.get("query") or ""), _int(payload.get("limit") or 8))
                self._send_json(200, result)
                return True

            if path == "/api/sandboxes":
                body = self._request_body(1024 * 1024)
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ServiceError(400, "创建请求必须是有效 JSON") from None
                if not isinstance(payload, dict):
                    raise ServiceError(400, "创建请求必须是对象")
                result = create_sandbox(str(payload.get("template_id") or ""))
                self._send_json(201, result)
                return True

            parts = [part for part in path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "sandboxes"] and parts[3] in {"files", "remote", "auto", "auto-files"}:
                dataset_id = parts[2]
                _sandbox_dataset(dataset_id)
                if parts[3] == "auto":
                    body = self._request_body(1024 * 1024)
                    try:
                        payload = json.loads(body.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        raise ServiceError(400, "全自动入库请求必须是有效 JSON") from None
                    if not isinstance(payload, dict):
                        raise ServiceError(400, "全自动入库请求必须是对象")
                    query_text = " ".join(str(payload.get("query") or "").split())[:300]
                    if len(query_text) < 2:
                        raise ServiceError(400, "请输入至少 2 个字符的专业主题")
                    limit = max(1, min(_int(payload.get("limit") or 6), 12))
                    parallelism = max(1, min(_int(payload.get("parallelism") or 3), 6))
                    job = _new_job(dataset_id, "online", query=query_text, limit=limit, parallelism=parallelism)
                    self._send_json(202, job)
                    return True

                if parts[3] == "auto-files":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.lower().startswith("multipart/form-data"):
                        raise ServiceError(415, "本地文件须使用 multipart/form-data")
                    body = self._request_body(MAX_REQUEST_BYTES)
                    mime = BytesParser(policy=email.policy.default).parsebytes(
                        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
                    )
                    if not mime.is_multipart():
                        raise ServiceError(400, "无法读取本地文件表单")
                    files = []
                    total_bytes = 0
                    for part in mime.iter_parts():
                        disposition = part.get_content_disposition()
                        field = part.get_param("name", header="content-disposition")
                        filename = part.get_filename()
                        if disposition != "form-data" or field != "files" or not filename:
                            continue
                        safe_name = _safe_filename(filename)
                        content = part.get_payload(decode=True) or b""
                        if not content or len(content) > MAX_UPLOAD_BYTES:
                            raise ServiceError(413, f"{safe_name}：文件为空或超过 50 MB")
                        total_bytes += len(content)
                        if total_bytes > 95 * 1024 * 1024:
                            raise ServiceError(413, "每批文件总量不能超过 95 MB")
                        files.append((safe_name, content))
                    if not files:
                        raise ServiceError(400, "请先选择至少一个本地文件")
                    if len(files) > MAX_UPLOAD_FILES:
                        raise ServiceError(413, f"一次最多导入 {MAX_UPLOAD_FILES} 个文件")
                    parallelism = max(1, min(_int((query.get("parallelism") or [3])[0]), 6))
                    job = _new_job(dataset_id, "local", parallelism=parallelism, files=files)
                    self._send_json(202, job)
                    return True

                if parts[3] == "files":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.lower().startswith("multipart/form-data"):
                        raise ServiceError(415, "本地文件须使用 multipart/form-data")
                    body = self._request_body(MAX_REQUEST_BYTES)
                    mime = BytesParser(policy=email.policy.default).parsebytes(
                        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
                    )
                    if not mime.is_multipart():
                        raise ServiceError(400, "无法读取本地文件表单")
                    files = []
                    for part in mime.iter_parts():
                        disposition = part.get_content_disposition()
                        field = part.get_param("name", header="content-disposition")
                        filename = part.get_filename()
                        if disposition != "form-data" or field != "files" or not filename:
                            continue
                        content = part.get_payload(decode=True) or b""
                        files.append((filename, content))
                    if not files:
                        raise ServiceError(400, "请先选择至少一个本地文件")
                    if len(files) > MAX_UPLOAD_FILES:
                        raise ServiceError(413, f"一次最多导入 {MAX_UPLOAD_FILES} 个文件")
                    results = []
                    gate_now = _load_model_gate_file(dataset_id)
                    for filename, content in files:
                        try:
                            result = _upload_to_sandbox(dataset_id, filename, content, wait_for_parse=bool(gate_now.get("ready")))
                            results.append({"name": filename, "status": "started" if result["parse_started"] else "uploaded", **result})
                        except ServiceError as exc:
                            results.append({"name": filename, "status": "failed", "error": exc.message})
                    succeeded = sum(item["status"] in {"started", "uploaded"} for item in results)
                    self._send_json(200 if succeeded else 400, {
                        "items": results,
                        "success_count": succeeded,
                        "failure_count": len(results) - succeeded,
                    })
                    return True

                body = self._request_body(1024 * 1024)
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ServiceError(400, "在线导入请求必须是有效 JSON") from None
                if not isinstance(payload, dict):
                    raise ServiceError(400, "在线导入请求必须是对象")
                url = _public_http_url(str(payload.get("url") or ""))
                filename, content = _download_public_pdf(url)
                requested_name = str(payload.get("name") or "").strip()
                if requested_name:
                    if not requested_name.lower().endswith(".pdf"):
                        requested_name += ".pdf"
                    filename = _safe_filename(requested_name)
                gate_now = _load_model_gate_file(dataset_id)
                result = _upload_to_sandbox(dataset_id, filename, content, wait_for_parse=bool(gate_now.get("ready")))
                self._send_json(200, result)
                return True
        except ServiceError as exc:
            self._send_json(exc.status, {"error": exc.message})
            return True
        except Exception:
            self._send_json(502, {"error": "工作流执行失败，请检查本机服务状态"})
            return True
        return False

    def do_GET(self) -> None:
        if not self._is_local_request():
            self.send_error(403, "Local workbench requests only.")
            return
        parsed = urlsplit(self.path)
        if parsed.path.startswith("/api/"):
            if not self._route_api(parsed.path, parse_qs(parsed.query)):
                self._send_json(404, {"error": "API route not found"})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if not self._is_local_request():
            self.send_error(403, "Local workbench requests only.")
            return
        if not self._trusted_origin():
            self.send_error(403, "Cross-origin requests are blocked.")
            return
        parsed = urlsplit(self.path)
        if not parsed.path.startswith("/api/") or not self._route_post(parsed.path, parse_qs(parsed.query)):
            self.send_error(405, "Only sandbox-scoped workflow actions are available.")

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging document names, search terms, or local paths.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Knowledge Ingestion Harness workbench.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (loopback only).")
    parser.add_argument("--port", type=int, default=4173, help="Local web port (default: 4173).")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("The workbench server must bind to a loopback address.")
    # Convert active history only once, on process startup.
    with _ingestion_jobs_lock:
        for job in HISTORY.load_interrupted() + HISTORY.load_recoverable():
            _ingestion_jobs[job["id"]] = job
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"Knowledge Ingestion Harness: http://{args.host}:{args.port}/")
    print("Existing datasets are read-only. Writes are restricted to Harness-marked sandbox datasets.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
