from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, TypedDict


@dataclass(frozen=True)
class Evidence:
    """A citation-ready evidence unit returned by any retriever."""

    evidence_id: str
    source_id: str
    title: str
    content: str
    document_type: str
    page: int | None
    source_quality: float
    retrieval_score: float
    locator: str
    is_page_fallback: bool = False
    upstream: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionPart:
    part_id: str
    text: str
    required_terms: List[str] = field(default_factory=list)
    required_term_groups: List[List[str]] = field(default_factory=list)


@dataclass(frozen=True)
class Evaluation:
    evidence_coverage: float
    precision: float
    source_quality: float
    units_formula: float
    conflict: float
    completeness: float
    missing_parts: List[str]
    conflict_notes: List[str]
    route: Literal["answer", "reretrieve", "page_fallback", "web_retrieve"]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssistantState(TypedDict, total=False):
    question: str
    subquestions: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    selected_evidence: List[Dict[str, Any]]
    retrieval_errors: List[str]
    fallback_pages: List[Dict[str, Any]]
    page_context: List[Dict[str, Any]]
    page_context_instruction: str
    web_search_status: str
    web_search_error: str
    web_evidence: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    answer: str
    citations: List[str]
    refusal: bool
    iteration: int
    max_iterations: int
    allow_web: bool
    route: str
    trace: List[str]
    config: Dict[str, Any]


def evidence_from_dict(value: Dict[str, Any]) -> Evidence:
    return Evidence(**value)
