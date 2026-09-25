import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragflow_runtime_audit import audit_ragflow_runtime
from sync_library import PaperIndex


class FakeRAGFlowClient:
    documents = {
        "markdown-id": {
            "id": "markdown-id", "name": "paper.md", "run": "DONE", "status": "1",
            "chunk_count": 1, "parser_config": {"layout_recognize": "Plain Text", "delimiter": "\n"},
        },
        "pdf-id": {
            "id": "pdf-id", "name": "paper.pdf", "run": "DONE", "status": "0",
            "size": 1000, "chunk_count": 1,
            "parser_config": {"layout_recognize": "mineru-from-env@MinerU"},
        },
    }

    def __init__(self, *_args, **_kwargs):
        pass

    def document(self, document_id):
        return self.documents.get(document_id, {})

    def list_chunks(self, document_id):
        if document_id not in self.documents:
            return []
        return [{"content": "Electromagnetic simulation source page and mesh parameters."}]


class RAGFlowPdfAssetAuditTests(unittest.TestCase):
    def test_disabled_but_parsed_original_pdf_is_verified_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path = root / "index.sqlite"
            index = PaperIndex(index_path, root / "summary.csv")
            try:
                index.upsert_paper({
                    "doc_id": "paper-1", "zotero_key": "z1", "title": "FDTD paper",
                    "sha256": "a" * 64, "status": "published_ragflow",
                    "ragflow_document_id": "markdown-id", "ragflow_pdf_document_id": "pdf-id",
                    "ragflow_dataset_id": "dataset-papers",
                    "ragflow_parser_config": json.dumps({"parser_config": {
                        "layout_recognize": "Plain Text", "delimiter": "\n",
                    }}),
                })
            finally:
                index.close()
            config = {
                "storage": {"root": str(root), "index_db": str(index_path)},
                "ragflow": {
                    "base_url": "http://unused", "api_key_env": "TEST_RAGFLOW_AUDIT_KEY",
                    "dataset_ids": {"papers": "dataset-papers"},
                },
            }
            with mock.patch.dict(os.environ, {"TEST_RAGFLOW_AUDIT_KEY": "present"}), \
                    mock.patch("ragflow_runtime_audit.RAGFlowClient", FakeRAGFlowClient):
                report = audit_ragflow_runtime(config, dataset_key="papers")

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["pdf_assets_indexed"], 1)
            self.assertEqual(report["pdf_assets_checked"], 1)
            pdf_asset = report["documents"][0]["original_pdf_asset"]
            self.assertEqual(pdf_asset["document_id"], "pdf-id")
            self.assertEqual(pdf_asset["chunk_count"], 1)
            self.assertFalse(pdf_asset["retrieval_enabled"])
            self.assertEqual(pdf_asset["parser_layout"], "mineru-from-env@MinerU")


if __name__ == "__main__":
    unittest.main()
