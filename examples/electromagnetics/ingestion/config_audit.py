"""Static configuration and runtime-readiness audit for the ingestion workflow."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sync_library import RAGFlowClient, resolve_api_secret, secret_selector_label


DATASET_KEYS = ("fundamentals", "materials", "papers", "software_manuals", "cases", "code_examples")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _model_messages_url(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def _probe_model_endpoint(base_url: str, timeout: float = 8.0) -> dict[str, Any]:
    """Check TLS/DNS/reachability without sending a document or an API key."""
    url = _model_messages_url(base_url)
    if not str(base_url or "").strip():
        return {"configured": False, "reachable": False, "url": ""}
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "EM-FDTD-ConfigAudit/1.0"}, method="HEAD")
    try:
        with urlopen(request, timeout=timeout) as response:
            return {"configured": True, "reachable": True, "url": url, "http_status": int(response.status)}
    except HTTPError as exc:
        # 401/403/404/405 still prove that the remote TLS/HTTP endpoint is up.
        return {"configured": True, "reachable": True, "url": url, "http_status": int(exc.code)}
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        return {"configured": True, "reachable": False, "url": url, "error": f"{type(exc).__name__}: {reason}"}


def probe_model_endpoints(model: Mapping[str, Any], extra_candidates: list[str] | None = None) -> list[dict[str, Any]]:
    """Probe primary and optional candidates without credentials or documents.

    Candidates are diagnostic only; ingestion continues to use ``base_url``.
    """
    primary = str(model.get("base_url", "")).strip()
    configured_candidates = model.get("probe_base_urls", [])
    values: list[str] = [primary]
    if isinstance(configured_candidates, (list, tuple)):
        values.extend(str(item).strip() for item in configured_candidates if str(item).strip())
    if extra_candidates:
        values.extend(str(item).strip() for item in extra_candidates if str(item).strip())
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.rstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    timeout = float(model.get("connect_timeout_seconds", 8))
    return [
        {
            "role": "primary" if index == 0 else "candidate",
            "base_url": base_url,
            "selected_for_ingest": index == 0,
            **_probe_model_endpoint(base_url, timeout=timeout),
        }
        for index, base_url in enumerate(unique)
    ]


def audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    runtime_dataset_checks: list[dict[str, Any]] = []
    storage = config.get("storage", {})
    rag = config.get("ragflow", {})
    mineru = config.get("mineru", {})
    review = config.get("review", {})
    model = config.get("model", {})
    root = Path(str(storage.get("root", "")))
    required_drive = str(storage.get("required_drive", "G:")).rstrip("\\/").upper()
    if required_drive and root.drive.upper() != required_drive:
        violations.append({"type": "storage_drive_mismatch", "expected": required_drive, "actual": root.drive.upper()})
    if not root:
        violations.append({"type": "missing_storage_root"})
    embedding = str(rag.get("embedding_model", "")).strip()
    if embedding != "text-embedding-v4@Tongyi-Qianwen":
        violations.append({"type": "embedding_model_mismatch", "expected": "text-embedding-v4@Tongyi-Qianwen", "actual": embedding})
    mapping = rag.get("dataset_ids", {})
    if not isinstance(mapping, Mapping):
        mapping = {}
    for key in DATASET_KEYS:
        value = str(mapping.get(key, "")).strip()
        if not value or value.startswith("FILL_"):
            violations.append({"type": "missing_dataset_id", "dataset_key": key})
    command = str(mineru.get("command", "")).strip()
    if not command:
        violations.append({"type": "missing_mineru_command"})
    elif not Path(command).exists():
        warnings.append({"type": "mineru_command_not_found", "path": command})
    extra_args = [str(value) for value in mineru.get("extra_args", [])]
    if "-b" not in extra_args or "pipeline" not in extra_args:
        warnings.append({"type": "mineru_backend_not_explicit_pipeline"})
    rag_key_selector = str(rag.get("api_key_env", "RAGFLOW_API_KEY"))
    if bool(rag.get("enabled")) and not resolve_api_secret(rag_key_selector):
        warnings.append({"type": "ragflow_api_key_missing", "source": secret_selector_label(rag_key_selector)})
    if bool(rag.get("enabled")) and not bool(review.get("enabled")):
        violations.append({"type": "ragflow_requires_ingest_review", "reason": "RAGFlow 成品库禁止绕过文献准入审查"})
    model_key_selector = str(model.get("api_key_env", "MODEL_API_KEY"))
    if bool(review.get("enabled")) and not resolve_api_secret(model_key_selector):
        warnings.append({"type": "model_api_key_missing", "source": secret_selector_label(model_key_selector)})
    preparsed = rag.get("preparsed_markdown", {})
    if isinstance(preparsed, Mapping) and preparsed.get("enabled"):
        preparsed_config = preparsed.get("parser_config", {})
        if not isinstance(preparsed_config, Mapping) or preparsed_config.get("layout_recognize") not in {"Plain Text", False}:
            violations.append({
                "type": "preparsed_markdown_layout_must_be_disabled",
                "reason": "本地已由 MinerU 解析的 Markdown 不应在 RAGFlow 再次做布局识别",
            })
    model_endpoint_candidates = probe_model_endpoints(model) if bool(review.get("enabled")) else []
    model_endpoint_check = model_endpoint_candidates[0] if model_endpoint_candidates else {"configured": False, "reachable": False, "url": ""}
    if model_endpoint_candidates and not model_endpoint_check.get("reachable"):
        warnings.append({
            "type": "model_endpoint_unreachable",
            "url": model_endpoint_check.get("url", ""),
            "error": model_endpoint_check.get("error", "unknown error"),
        })
    rag_api_key_env = str(rag.get("api_key_env", "RAGFLOW_API_KEY"))
    rag_api_key = resolve_api_secret(rag_api_key_env)
    if bool(rag.get("enabled")) and rag_api_key:
        expected_embedding = embedding
        try:
            runtime_rows = RAGFlowClient(
                str(rag.get("base_url", "")), "", rag_api_key_env,
                timeout=float(rag.get("runtime_audit_timeout_seconds", 20)),
            ).list_datasets()
            by_id = {str(row.get("id")): row for row in runtime_rows if isinstance(row, Mapping) and row.get("id")}
            for dataset_key in DATASET_KEYS:
                configured_id = str(mapping.get(dataset_key, "")).strip()
                row = by_id.get(configured_id, {})
                parser_config = row.get("parser_config") if isinstance(row, Mapping) else {}
                parser_config = parser_config if isinstance(parser_config, Mapping) else {}
                layout = str(parser_config.get("layout_recognize") or row.get("layout_recognize") or "")
                check = {
                    "dataset_key": dataset_key,
                    "configured_id": configured_id,
                    "found": bool(row),
                    "name": str(row.get("name") or "") if isinstance(row, Mapping) else "",
                    "embedding_model": str(row.get("embedding_model") or "") if isinstance(row, Mapping) else "",
                    "chunk_method": str(row.get("chunk_method") or "") if isinstance(row, Mapping) else "",
                    "layout_recognize": layout,
                    "parser_config_available": bool(parser_config),
                }
                runtime_dataset_checks.append(check)
                if not row:
                    violations.append({"type": "ragflow_dataset_not_found", "dataset_key": dataset_key, "dataset_id": configured_id})
                    continue
                if expected_embedding and check["embedding_model"] != expected_embedding:
                    violations.append({
                        "type": "ragflow_embedding_model_mismatch",
                        "dataset_key": dataset_key,
                        "expected": expected_embedding,
                        "actual": check["embedding_model"],
                    })
                if not check["parser_config_available"]:
                    warnings.append({"type": "ragflow_parser_config_unavailable", "dataset_key": dataset_key})
                elif dataset_key == "code_examples" and layout != "Plain Text":
                    violations.append({
                        "type": "ragflow_code_dataset_layout_not_plain_text",
                        "dataset_key": dataset_key,
                        "expected": "Plain Text",
                        "actual": layout,
                    })
                elif dataset_key != "code_examples" and "mineru" not in layout.lower():
                    violations.append({
                        "type": "ragflow_layout_not_mineru",
                        "dataset_key": dataset_key,
                        "actual": layout,
                    })
        except Exception as exc:
            warnings.append({
                "type": "ragflow_runtime_probe_failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
    credential_warning_types = {"ragflow_api_key_missing", "model_api_key_missing"}
    only_credential_warnings = bool(warnings) and all(item.get("type") in credential_warning_types for item in warnings)
    report = {
        "status": "pass" if not violations and not warnings else ("needs_credentials" if not violations and only_credential_warnings else "needs_attention"),
        "static_status": "pass" if not violations else "needs_attention",
        "runtime_status": "ready" if not warnings else ("needs_credentials" if only_credential_warnings else "needs_attention"),
        "dataset_keys": list(DATASET_KEYS),
        "embedding_model": embedding,
        "mineru_backend": "pipeline" if "-b" in extra_args and "pipeline" in extra_args else "unspecified",
        "model_endpoint_check": model_endpoint_check,
        "model_endpoint_candidates": model_endpoint_candidates,
        "runtime_dataset_checks": runtime_dataset_checks,
        "storage_root": str(root),
        "violations": violations,
        "warnings": warnings,
        "created_at": _now(),
    }
    output = root / "evaluation" / "config_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(output)
    return report


if __name__ == "__main__":
    raise SystemExit("请通过 sync_library.py audit-config 调用")
