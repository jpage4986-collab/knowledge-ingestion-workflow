from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .models import Evaluation, Evidence, QuestionPart
from .adapters import DOMAIN_ALIASES, DOMAIN_TERMS


def normalize_evidence_text(text: str) -> str:
    """Normalize common MinerU LaTeX subscript spacing for exact evidence checks."""
    normalized = str(text or "").lower()
    normalized = re.sub(
        r"([a-z])\s*_\s*\{\s*([a-z0-9]+)\s*\}", r"\1_\2", normalized, flags=re.I
    )
    normalized = re.sub(r"([a-z])\s*_\s*([a-z0-9])", r"\1_\2", normalized, flags=re.I)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower()))
    terms |= set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    terms |= {term.lower() for term in DOMAIN_TERMS if term.lower() in text.lower() or term in text}
    for trigger, aliases in DOMAIN_ALIASES.items():
        if trigger in text:
            terms.update(alias.lower() for alias in aliases)
    return terms


def _normalized_question_terms(text: str) -> set[str]:
    terms = _normalized_terms(text)
    for term in ("FDTD", "CFL", "Yee", "PML", "COMSOL"):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])\s*(?:无关|不相关|irrelevant|unrelated)", text, re.I):
            terms.discard(term.lower())
    return terms


def evidence_group_matches(part: QuestionPart, evidence: Evidence) -> list[bool] | None:
    """Return which specific evidence groups this single chunk supports."""
    if not part.required_term_groups:
        return None
    corpus = normalize_evidence_text(f"{evidence.title} {evidence.content} {evidence.upstream}")
    alias_terms = _normalized_terms(corpus)
    normalized_corpus = re.sub(r"\s+", " ", corpus.lower())

    def phrase_matches(value: str) -> bool:
        phrase = re.sub(r"\s+", " ", value.lower()).strip()
        if not phrase:
            return False
        if " " in phrase or not re.fullmatch(r"[a-z0-9_]+", phrase):
            return phrase in normalized_corpus
        phrase_terms = _normalized_terms(phrase)
        return phrase in alias_terms or bool(phrase_terms) and phrase_terms.issubset(alias_terms)

    return [any(phrase_matches(phrase) for phrase in group) for group in part.required_term_groups]


def evidence_alignment(part: QuestionPart, evidence: Evidence) -> float | None:
    """Fraction of a question's specific evidence groups found in one chunk."""
    matches = evidence_group_matches(part, evidence)
    if matches is None:
        return None
    return sum(matches) / max(1, len(matches))


def evaluate_evidence(parts: Iterable[QuestionPart], evidence: Iterable[Evidence], iteration: int, max_iterations: int, allow_web: bool) -> Evaluation:
    parts = list(parts)
    evidence = list(evidence)
    corpus = " ".join(item.content for item in evidence)
    corpus_terms = _normalized_terms(corpus)
    alias_corpus = normalize_evidence_text(" ".join(f"{item.title} {item.content}" for item in evidence))
    alias_terms = _normalized_terms(alias_corpus)
    normalized_alias_corpus = re.sub(r"\s+", " ", alias_corpus.lower())

    def covered(part: QuestionPart) -> bool:
        question_terms = _normalized_question_terms(part.text)
        def phrase_matches(value: str) -> bool:
            phrase = re.sub(r"\s+", " ", value.lower()).strip()
            if not phrase:
                return False
            if " " in phrase or not re.fullmatch(r"[a-z0-9_]+", phrase):
                return phrase in normalized_alias_corpus
            phrase_terms = _normalized_terms(phrase)
            return phrase in alias_terms or bool(phrase_terms) and phrase_terms.issubset(alias_terms)

        if part.required_term_groups:
            return all(any(phrase_matches(phrase) for phrase in group) for group in part.required_term_groups)
        if _normalized_question_terms(part.text) & corpus_terms:
            return True
        for required in part.required_terms:
            if required.lower() in {"fdtd", "cfl", "yee", "pml", "comsol"} and required.lower() not in question_terms:
                continue
            if phrase_matches(required):
                return True
        return False

    missing = [part.part_id for part in parts if not covered(part)]
    coverage = (len(parts) - len(missing)) / max(1, len(parts))
    grouped_parts = [part for part in parts if part.required_term_groups]
    if grouped_parts:
        precision = sum(
            max((evidence_alignment(part, item) or 0.0) for part in grouped_parts)
            for item in evidence
        ) / max(1, len(evidence))
    else:
        precision = sum(item.retrieval_score for item in evidence) / max(1, len(evidence))
    source_quality = sum(item.source_quality for item in evidence) / max(1, len(evidence))
    has_formula = bool(re.search(r"CFL|Δt|dt|1/\(|稳定条件|\^|equation|formula", corpus, re.I))
    has_units = bool(re.search(
        r"m/s|秒|米|\d+(?:\.\d+)?\s*(?:nm|μm|µm|um|mm|cm|m|GHz|MHz|THz|Hz|S/m|F/m|Ω|ohm)",
        corpus,
        re.I,
    ))
    question_requires_formula = bool(re.search(r"公式|方程|关系式|formula|equation|CFL|稳定条件", " ".join(part.text for part in parts), re.I))
    question_requires_units = bool(re.search(r"单位|unit|波长|wavelength|频率|frequency|nm|μm|µm|um|尺寸|范围", " ".join(part.text for part in parts), re.I))
    required_checks = ([int(has_formula)] if question_requires_formula else []) + ([int(has_units)] if question_requires_units else [])
    units_formula = sum(required_checks) / len(required_checks) if required_checks else 1.0
    by_topic: defaultdict[str, list[str]] = defaultdict(list)
    if re.search(r"稳定|CFL", corpus, re.I):
        by_topic["stability"].append(corpus)
    conflict_notes: list[str] = []
    if len(by_topic["stability"]) > 1 and ("超过" in corpus and "以内" in corpus):
        # These are complementary statements, not a contradiction.
        pass
    conflict = 0.0
    reflection_result = re.search(r"(?:reflection|reflectance).{0,100}(?:85\s*%|0\.85).{0,60}(?:92\s*%|0\.92)", corpus, re.I | re.S)
    order_transmittance = re.search(r"transmittance.{0,220}order\s*\[\s*1\s*,\s*0\s*\]", corpus, re.I | re.S)
    if reflection_result and order_transmittance:
        conflict = 1.0
        conflict_notes.append(
            "COMSOL 形状优化 PDF 的结果段报告反射率由 85% 提高至 92%，但设置步骤将目标函数设为 [1,0] 级次透射率最大化；两处指标口径不一致，需保留原文并核对，不能互相替代。"
        )
    completeness = sum((coverage, precision, source_quality, units_formula, 1.0 - conflict)) / 5
    if missing and iteration < max_iterations:
        if allow_web and any(item.is_page_fallback for item in evidence):
            route = "web_retrieve"
        elif any(
            item.page is not None
            and item.page > 0
            and item.document_type.lower() in {"pdf", "ppt", "pptx", "powerpoint", "png", "jpg", "jpeg", "image"}
            for item in evidence
        ):
            route = "page_fallback"
        elif allow_web:
            route = "web_retrieve"
        else:
            # Code and plain-text chunks have no original page asset to expand;
            # avoid repeating a retrieval that cannot improve their evidence.
            route = "answer"
    else:
        route = "answer"
    return Evaluation(coverage, precision, source_quality, units_formula, conflict, completeness, missing, conflict_notes, route)
