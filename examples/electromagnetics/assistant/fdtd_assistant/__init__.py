"""Offline-first electromagnetic/FDTD knowledge answer assistant."""

from .graph import build_graph, run_assistant
from .knowledge import FixtureKnowledgeProvider, SimulationSpec
from .workflow import build_workflow, run_workflow
from .client import FDTDClient, FDTDClientError
from .literature import LiteratureIndexRecord, JSONLiteratureIndex, SQLiteLiteratureIndex, UpstreamLiteratureRetriever
from .agent import AnswerAgent
from .pdf_lifecycle import PdfLease, PdfLeaseManager

__all__ = ["build_graph", "run_assistant", "build_workflow", "run_workflow", "FixtureKnowledgeProvider", "SimulationSpec", "FDTDClient", "FDTDClientError", "LiteratureIndexRecord", "JSONLiteratureIndex", "SQLiteLiteratureIndex", "UpstreamLiteratureRetriever", "AnswerAgent", "PdfLease", "PdfLeaseManager"]
