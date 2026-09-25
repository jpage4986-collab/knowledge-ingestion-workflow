import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from adaptive_chunk_repair import _profile_completion_state, _profile_duplicate_ids


class AdaptiveRepairDeduplicationTests(unittest.TestCase):
    def test_only_losing_profile_pdfs_are_selected_for_cleanup(self):
        attempts = [
            {"profile": 1024, "new_ragflow_document_id": "pdf-a"},
            {"profile": 1536, "new_ragflow_document_id": "pdf-a"},
            {"profile": 1536, "new_ragflow_document_id": "pdf-b"},
            {"profile": 2048, "new_ragflow_document_id": "pdf-c"},
            {"profile": "preparsed_markdown", "new_ragflow_document_id": "markdown"},
            {"profile": "profile_sweep_complete", "new_ragflow_document_id": "report"},
            {"profile": 4096, "new_ragflow_document_id": "unsupported"},
        ]

        self.assertEqual(
            _profile_duplicate_ids(attempts, (1024, 1536, 2048), "pdf-b", "old-pdf"),
            ["old-pdf", "pdf-a", "pdf-c"],
        )

    def test_low_quality_pdf_keeps_previous_markdown_as_serving_source(self):
        state = _profile_completion_state("markdown-old", "pdf-new", 0.9618, 0.98)
        self.assertEqual(state["serving_document_id"], "markdown-old")
        self.assertEqual(state["pdf_document_id"], "pdf-new")
        self.assertEqual(state["status"], "published_ragflow")
        self.assertFalse(state["candidate_accepted"])
        self.assertFalse(state["retire_best_pdf"])

    def test_low_quality_pdf_keeps_previous_pdf_and_marks_candidate_for_retirement(self):
        state = _profile_completion_state("pdf-old", "pdf-new", 0.9448, 0.98, "pdf-old")
        self.assertEqual(state["serving_document_id"], "pdf-old")
        self.assertEqual(state["pdf_document_id"], "pdf-old")
        self.assertEqual(state["status"], "published_ragflow")
        self.assertTrue(state["retire_best_pdf"])

    def test_low_quality_pdf_without_prior_serving_document_is_not_published(self):
        state = _profile_completion_state("", "pdf-new", 0.9448, 0.98)
        self.assertEqual(state["serving_document_id"], "")
        self.assertEqual(state["pdf_document_id"], "pdf-new")
        self.assertEqual(state["status"], "chunk_quality_exhausted")

    def test_quality_gate_pass_replaces_serving_and_pdf_ids(self):
        state = _profile_completion_state("markdown-old", "pdf-new", 0.98, 0.98)
        self.assertEqual(state["serving_document_id"], "pdf-new")
        self.assertEqual(state["pdf_document_id"], "pdf-new")
        self.assertEqual(state["status"], "published_ragflow")
        self.assertTrue(state["candidate_accepted"])


if __name__ == "__main__":
    unittest.main()
