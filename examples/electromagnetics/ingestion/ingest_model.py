"""Model-driven screening and simulation-parameter review for ingestion.

This is deliberately separate from the answer assistant. Its output is a
machine-readable admission decision for the corpus, not a user-facing answer.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sync_library import resolve_api_secret, secret_selector_label


@dataclass
class IngestReviewModel:
    base_url: str
    model_name: str
    protocol: str = "anthropic_messages"
    api_key_env: str = "MODEL_API_KEY"
    max_tokens: int = 3000
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    auth_header: str = "x-api-key"
    evidence_mode: str = "strict"
    network_circuit_seconds: float = 300.0
    _network_blocked_until: float = field(default=0.0, init=False, repr=False)

    def _url(self) -> str:
        base = self.base_url.rstrip("/")
        if self.protocol.lower() in {"openai", "openai_chat", "openai_chat_completions", "chat_completions"}:
            if base.endswith("/chat/completions"):
                return base
            if base.endswith("/v1"):
                return base + "/chat/completions"
            return base + "/v1/chat/completions"
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def _key(self) -> str:
        key = resolve_api_secret(self.api_key_env)
        if not key:
            raise RuntimeError(
                "模型 API key 未通过环境变量或本地配置设置："
                f"{secret_selector_label(self.api_key_env)}；不会自动入库。"
            )
        return key

    @staticmethod
    def _build_review_excerpt(
        markdown: str,
        heuristic_summary: Mapping[str, Any],
        max_chars: int = 100000,
    ) -> str:
        """Build a bounded review context without dropping late-document evidence."""
        if len(markdown) <= max_chars:
            return markdown
        pieces: list[tuple[str, str]] = []

        def add(label: str, value: Any, limit: int) -> None:
            text = str(value or "").strip()
            if text:
                pieces.append((label, text[:limit]))

        head_budget = min(24000, max(1000, max_chars // 3))
        tail_budget = min(16000, max(1000, max_chars // 4))
        add("document_start", markdown[:head_budget], head_budget)
        add("document_end", markdown[-tail_budget:], tail_budget)
        evidence_ref = heuristic_summary.get("evidence_ref", "") if isinstance(heuristic_summary, Mapping) else ""
        try:
            decoded = json.loads(evidence_ref) if isinstance(evidence_ref, str) else evidence_ref
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = {}
        fields = decoded.get("fields", {}) if isinstance(decoded, Mapping) else {}
        if isinstance(fields, Mapping):
            for field, data in fields.items():
                if not isinstance(data, Mapping):
                    continue
                pages = data.get("pages", [])
                add(f"evidence_{field}_pages={pages}", data.get("page_excerpt", ""), 3500)

        anchor_terms = (
            "wavelength", "frequency", "material", "geometry", "mesh", "CFL", "time step",
            "PML", "boundary", "source", "monitor", "solver", "stop condition", "simulation",
            "网格", "边界", "激励", "监视器", "求解器",
        )
        seen_spans: set[tuple[int, int]] = set()
        for term in anchor_terms:
            for match in list(re.finditer(re.escape(term), markdown, flags=re.IGNORECASE))[:4]:
                start = max(0, match.start() - 1400)
                end = min(len(markdown), match.end() + 1800)
                span = (start, end)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                add(f"context_{term}_{start}", markdown[start:end], 3300)

        output: list[str] = []
        used = 0
        for label, text in pieces:
            block = f"\n\n[MINERU {label}]\n{text}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            output.append(block[:remaining])
            used += min(len(block), remaining)
        return "".join(output)

    def _post_payload(self, payload: Mapping[str, Any]) -> str:
        """POST one bounded review request and return the gateway JSON text."""
        key = self._key()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": "2023-06-01",
            "Connection": "close",
            "User-Agent": "EM-FDTD-Ingest/1.0",
        }
        use_openai_protocol = self.protocol.lower() in {"openai", "openai_chat", "openai_chat_completions", "chat_completions"}
        if self.auth_header.lower() == "authorization" or (use_openai_protocol and self.auth_header.lower() == "x-api-key"):
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers[self.auth_header] = key
        request = Request(self._url(), data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        raw = ""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"文献审查模型请求失败，HTTP {exc.code}") from exc
                last_error = exc
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
        if raw:
            return raw
        if isinstance(last_error, (URLError, TimeoutError, ConnectionError, OSError)):
            self._network_blocked_until = time.monotonic() + max(0.0, self.network_circuit_seconds)
        if isinstance(last_error, HTTPError):
            raise RuntimeError(f"文献审查模型请求失败，HTTP {last_error.code}") from last_error
        if isinstance(last_error, URLError):
            raise RuntimeError(f"文献审查模型网络请求失败：{last_error.reason}") from last_error
        if isinstance(last_error, TimeoutError):
            raise RuntimeError("文献审查模型请求超时") from last_error
        raise RuntimeError(f"文献审查模型连接失败：{last_error}")

    def review(self, *, title: str, authors: str, year: str, doi: str, abstract: str, markdown: str, heuristic_summary: Mapping[str, Any]) -> dict[str, Any]:
        if time.monotonic() < self._network_blocked_until:
            raise RuntimeError("文献审查模型本轮网络熔断：上一请求已确认端点不可达；本轮不重复等待。")
        # Keep requests bounded while retaining the beginning, end, extracted
        # evidence and distributed parameter windows. The original Markdown
        # remains in the local corpus and is still used by the evidence gate.
        excerpt = self._build_review_excerpt(markdown, heuristic_summary)
        user = {
            "task": "评估一篇候选文献是否应进入电磁/FDTD成品知识库，并总结可复用的仿真参数。",
            "admission_rule": "accepted=true 表示这篇资料对电磁知识库有足够的相关性和可靠性，不表示 12 个仿真参数必须齐全。缺失参数必须留空并写入 missing_fields，禁止按常识补全；缺少参数原文证据是提醒，不是自动拒绝。只有明确不相关、解析损坏，或原文与抽取出的数值/单位发生冲突时才拒绝。",
            "output_schema": {
                "accepted": "boolean",
                "relevance_score": "0到1的小数",
                "quality_score": "0到1的小数",
                "document_type": "paper | software_manual | case | fundamentals | materials",
                "reason": "简短中文原因",
                "missing_fields": ["缺失或不确定的仿真参数字段"],
                "simulation_parameters": {"仅填写原文明确支持的参数字段": ""},
                "parameter_evidence": {"参数字段": "对应 MinerU 原文短句；无证据填空"},
            },
            "metadata": {"title": title, "authors": authors, "year": year, "doi": doi, "abstract": abstract},
            "heuristic_summary": dict(heuristic_summary),
            "mineru_markdown_excerpt": excerpt,
            "mineru_excerpt_strategy": "start_end_evidence_and_parameter_windows",
        }
        system_prompt = "你是文献入库审查器。只输出一个完整且合法的JSON对象，不要输出Markdown围栏；参数值和证据短句必须来自原文。"
        user["review_instruction"] = "保持输出简洁：每个证据短句不超过240字；缺失字段留空；不要添加 schema 之外的长篇解释。"
        if self.protocol.lower() in {"openai", "openai_chat", "openai_chat_completions", "chat_completions"}:
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
            }
        else:
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            }
        raw = self._post_payload(payload)
        try:
            response = json.loads(raw)
            text = self._extract_response_text(response)
            if not text:
                raise ValueError("empty model content")
            result = self._parse_json(text)
            result = self._validate_review_schema(result)
            return self._apply_evidence_gate(result, markdown, heuristic_summary, self.evidence_mode)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            # Large manuals can cause a gateway/model to truncate the first
            # JSON response. Retry once with a compact evidence context;
            # this is semantic recovery, not a blind network retry.
            compact_excerpt = self._build_review_excerpt(markdown, heuristic_summary, max_chars=32000)
            if compact_excerpt != excerpt:
                retry_user = dict(user)
                retry_user["mineru_markdown_excerpt"] = compact_excerpt
                retry_user["mineru_excerpt_strategy"] = "compact_start_end_evidence_and_parameter_windows"
                retry_payload = dict(payload)
                retry_payload["max_tokens"] = min(self.max_tokens, 1800)
                if self.protocol.lower() in {"openai", "openai_chat", "openai_chat_completions", "chat_completions"}:
                    retry_payload["messages"] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(retry_user, ensure_ascii=False)},
                    ]
                else:
                    retry_payload["messages"] = [{"role": "user", "content": json.dumps(retry_user, ensure_ascii=False)}]
                    retry_payload["system"] = system_prompt
                try:
                    retry_response = json.loads(self._post_payload(retry_payload))
                    retry_text = self._extract_response_text(retry_response)
                    retry_result = self._validate_review_schema(self._parse_json(retry_text))
                    return self._apply_evidence_gate(retry_result, markdown, heuristic_summary, self.evidence_mode)
                except (json.JSONDecodeError, ValueError, TypeError, KeyError, RuntimeError) as retry_exc:
                    exc = retry_exc
            raise RuntimeError(
                "文献审查模型返回内容未通过 JSON/schema/evidence 门禁："
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _extract_response_text(response: Mapping[str, Any]) -> str:
        """Extract text from Anthropic and common OpenAI-compatible gateways."""
        content = response.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, Mapping) and block.get("type") in {"text", "output_text"}
            ]
            if any(part.strip() for part in parts):
                return "\n".join(part for part in parts if part.strip()).strip()
        choices = response.get("choices")
        if isinstance(choices, list):
            parts: list[str] = []
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                message = choice.get("message")
                if isinstance(message, Mapping):
                    value = message.get("content")
                    if isinstance(value, str) and value.strip():
                        parts.append(value)
                    elif isinstance(value, list):
                        parts.extend(
                            str(item.get("text", ""))
                            for item in value
                            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}
                        )
                elif isinstance(choice.get("text"), str):
                    parts.append(str(choice["text"]))
            if any(part.strip() for part in parts):
                return "\n".join(part for part in parts if part.strip()).strip()
        output_text = response.get("output_text")
        return str(output_text).strip() if isinstance(output_text, str) else ""

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").removeprefix("json").strip()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("no JSON object")
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(cleaned[start:])
        if not isinstance(value, dict):
            raise ValueError("review JSON is not an object")
        if cleaned[start + end:].strip():
            value["response_warning"] = "模型在 JSON 后追加了内容，已忽略追加文本"
        return value

    @staticmethod
    def _validate_review_schema(review: dict[str, Any]) -> dict[str, Any]:
        """Normalize the small admission schema before applying evidence gates."""
        if "accepted" not in review:
            raise ValueError("review JSON is missing accepted")
        accepted = review["accepted"]
        if isinstance(accepted, bool):
            normalized_accepted = accepted
        elif isinstance(accepted, str) and accepted.strip().lower() in {"true", "false"}:
            normalized_accepted = accepted.strip().lower() == "true"
        else:
            raise ValueError("review accepted must be boolean")
        review["accepted"] = normalized_accepted
        for field in ("relevance_score", "quality_score"):
            if field not in review:
                raise ValueError(f"review JSON is missing {field}")
            try:
                value = float(review[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"review {field} must be numeric") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"review {field} must be between 0 and 1")
            review[field] = value
        for field in ("simulation_parameters", "parameter_evidence"):
            value = review.get(field, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"review {field} must be an object")
            review[field] = dict(value)
        if not isinstance(review.get("missing_fields", []), list):
            raise ValueError("review missing_fields must be an array")
        return review

    @staticmethod
    def _apply_evidence_gate(
        review: dict[str, Any],
        markdown: str,
        heuristic_summary: Mapping[str, Any],
        evidence_mode: str = "strict",
    ) -> dict[str, Any]:
        """Prevent unsupported model parameters from being auto-admitted."""
        accepted = review.get("accepted") is True or str(review.get("accepted", "")).lower() == "true"
        if not accepted:
            return review
        missing: list[str] = []
        hard_missing: list[str] = []
        balanced = str(evidence_mode).lower() in {"balanced", "warning", "soft"}
        parameters = review.get("simulation_parameters", {})
        evidence = review.get("parameter_evidence", {})
        if isinstance(parameters, Mapping):
            for field, value in parameters.items():
                if not isinstance(value, str) or not value.strip():
                    continue
                proof = evidence.get(field, "") if isinstance(evidence, Mapping) else ""
                if not isinstance(proof, str) or not proof.strip():
                    missing.append(str(field))
                    if not balanced and re.search(r"\d", value):
                        hard_missing.append(str(field))
                    continue
                normalized_proof = re.sub(r"\s+", " ", proof).strip()
                normalized_markdown = re.sub(r"\s+", " ", markdown)
                covers_value = IngestReviewModel._evidence_covers_value(value, proof)
                proof_is_from_source = normalized_proof in normalized_markdown
                numeric_conflict = IngestReviewModel._evidence_has_numeric_conflict(value, proof)
                if not proof_is_from_source or not covers_value:
                    missing.append(str(field))
                    if not balanced or numeric_conflict:
                        hard_missing.append(str(field))
        try:
            heuristic_quality = float(heuristic_summary.get("quality_score", 1.0) or 0)
        except (TypeError, ValueError):
            heuristic_quality = 0.0
        if heuristic_quality < 0.625:
            missing.append("heuristic_summary_quality")
            hard_missing.append("heuristic_summary_quality")
        if not missing:
            return review
        prior = review.get("missing_fields", [])
        prior_values = [str(value) for value in prior] if isinstance(prior, list) else []
        if balanced and not hard_missing:
            review["validation"] = "evidence_warning"
            review["evidence_warnings"] = list(dict.fromkeys(missing))
            review["missing_fields"] = list(dict.fromkeys(prior_values + missing))
            review["reason"] = "证据完整性提醒（未发现数值/单位错配）：" + ", ".join(missing)
            return review
        review["accepted"] = False
        try:
            model_quality = float(review.get("quality_score", 0) or 0)
        except (TypeError, ValueError):
            model_quality = 0.0
        review["quality_score"] = min(model_quality, 0.79)
        review["missing_fields"] = list(dict.fromkeys(prior_values + missing))
        review["reason"] = "语义证据门禁未通过：以下字段缺少原文证据或启发式覆盖不足：" + ", ".join(missing)
        review["validation"] = "evidence_gate"
        return review

    @staticmethod
    def _evidence_covers_value(value: str, proof: str) -> bool:
        """Require model-reported numeric values to occur in their evidence."""
        normalized_value = re.sub(r"\s+", " ", value).strip().lower()
        normalized_proof = re.sub(r"\s+", " ", proof).strip().lower()
        if normalized_value and normalized_value in normalized_proof:
            return True
        numeric_pattern = r"(?<![a-z0-9])(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?(?![a-z0-9])"
        value_numbers = re.findall(numeric_pattern, normalized_value, flags=re.IGNORECASE)
        if not value_numbers:
            return True
        proof_numbers = set(re.findall(numeric_pattern, normalized_proof, flags=re.IGNORECASE))
        if not all(number in proof_numbers for number in value_numbers):
            return False
        measurement_pattern = (
            r"(?<![a-z0-9])(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?\s*"
            r"(?:\[\s*([a-z%µμΩ]+)\s*\]|([a-z%µμΩ]+))(?![a-z0-9])"
        )
        for bracket_unit, plain_unit in re.findall(measurement_pattern, normalized_value, flags=re.IGNORECASE):
            unit = bracket_unit or plain_unit
            if unit and unit not in normalized_proof:
                return False
        return True

    @staticmethod
    def _evidence_has_numeric_conflict(value: str, proof: str) -> bool:
        """Detect an actual numeric/unit disagreement, not merely wording drift."""
        normalized_value = re.sub(r"\s+", " ", value).strip().lower()
        normalized_proof = re.sub(r"\s+", " ", proof).strip().lower()
        numeric_pattern = r"(?<![a-z0-9])(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?(?![a-z0-9])"
        value_numbers = re.findall(numeric_pattern, normalized_value, flags=re.IGNORECASE)
        proof_numbers = set(re.findall(numeric_pattern, normalized_proof, flags=re.IGNORECASE))
        # A field may legitimately summarize several examples or ranges while
        # its short evidence quote covers only one of them. Treat that as an
        # incomplete citation in balanced mode, not as a contradiction. Only
        # a simple one-number vs one-number disagreement is hard evidence.
        if len(value_numbers) == 1 and len(proof_numbers) == 1 and value_numbers[0] != next(iter(proof_numbers)):
            return True
        measurement_pattern = (
            r"(?<![a-z0-9])(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?\s*"
            r"(?:\[\s*([a-z%µμΩ]+)\s*\]|([a-z%µμΩ]+))(?![a-z0-9])"
        )
        value_units = {a or b for a, b in re.findall(measurement_pattern, normalized_value, flags=re.IGNORECASE) if (a or b) and (a or b) != "e"}
        proof_units = {a or b for a, b in re.findall(measurement_pattern, normalized_proof, flags=re.IGNORECASE) if (a or b) and (a or b) != "e"}
        return bool(len(value_numbers) == 1 and len(proof_numbers) == 1 and value_units and proof_units and value_units.isdisjoint(proof_units))


def build_ingest_reviewer(config: Mapping[str, Any]) -> IngestReviewModel | None:
    review = config.get("review", {})
    if not isinstance(review, Mapping) or not review.get("enabled", False):
        return None
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("config.model 必须是对象")
    return IngestReviewModel(
        base_url=str(model.get("base_url", "")),
        model_name=str(model.get("model_name", "YOUR_MODEL_ID")),
        protocol=str(model.get("protocol", "anthropic_messages")),
        api_key_env=str(model.get("api_key_env", "MODEL_API_KEY")),
        max_tokens=int(model.get("max_tokens", 3000)),
        temperature=float(model.get("temperature", 0.0)),
        timeout_seconds=float(model.get("timeout_seconds", 120)),
        auth_header=str(model.get("auth_header", "x-api-key")),
        evidence_mode=str(review.get("evidence_mode", "strict")),
    )
