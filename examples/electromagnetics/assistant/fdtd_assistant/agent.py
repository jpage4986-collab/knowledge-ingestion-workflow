"""Stable final-agent interface for the upstream literature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters import MockFixtureRetriever, Retriever, WebSearcher
from .graph import run_assistant
from .literature import UpstreamLiteratureRetriever
from .models import AssistantState
from .pdf_lifecycle import PdfLease, PdfLeaseManager


@dataclass
class AnswerAgent:
    """Downstream answer agent; it consumes an injected retriever, never crawls."""

    retriever: Retriever
    web_searcher: WebSearcher | None = None
    pdf_manager: PdfLeaseManager | None = None

    @classmethod
    def offline(cls) -> "AnswerAgent":
        return cls(MockFixtureRetriever())

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        dataset_key: str | None = None,
        web_searcher: WebSearcher | None = None,
        pdf_manager: PdfLeaseManager | None = None,
    ) -> "AnswerAgent":
        return cls(UpstreamLiteratureRetriever.from_json(path, dataset_key=dataset_key), web_searcher, pdf_manager)

    @classmethod
    def from_sqlite(
        cls,
        path: str | Path,
        *,
        table: str = "evidence_index",
        dataset_key: str | None = None,
        web_searcher: WebSearcher | None = None,
        pdf_manager: PdfLeaseManager | None = None,
    ) -> "AnswerAgent":
        return cls(UpstreamLiteratureRetriever.from_sqlite(path, table=table, dataset_key=dataset_key), web_searcher, pdf_manager)

    def answer(self, question: str, *, allow_web: bool = False, max_iterations: int = 2, include_full_page: bool = True) -> AssistantState:
        return run_assistant(
            question,
            retriever=self.retriever,
            web_searcher=self.web_searcher,
            allow_web=allow_web,
            max_iterations=max_iterations,
            include_full_page=include_full_page,
        )

    def stage_pdf(
        self,
        source: str | Path,
        *,
        ttl_seconds: int | None = None,
        filename: str | None = None,
    ) -> PdfLease:
        """Stage a PDF supplement for downstream reading.

        The returned lease must be released after the other agent has read the
        file.  The context-manager helper ``pdf_lease`` guarantees cleanup.
        """

        manager = self.pdf_manager or PdfLeaseManager()
        return manager.acquire(source, ttl_seconds=ttl_seconds, filename=filename)

    def pdf_lease(
        self,
        source: str | Path,
        *,
        ttl_seconds: int | None = None,
        filename: str | None = None,
    ):
        """Return a context manager that deletes the staged PDF on exit."""

        manager = self.pdf_manager or PdfLeaseManager()
        return manager.lease(source, ttl_seconds=ttl_seconds, filename=filename)
