import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fdtd_assistant.literature import UpstreamLiteratureRetriever


class UpstreamLiteratureAdapterTests(unittest.TestCase):
    def _row(self, base: Path) -> dict[str, object]:
        return {
            "doc_id": "doc-001",
            "dataset_key": "em-public",
            "title": "FDTD stability paper",
            "source_path": "papers/doc-001.pdf",
            "raw_path": "papers/doc-001.pdf",
            "mineru_markdown_path": None,
            "content_list_path": "pages.json",
            "evidence_ref": "doc-001:p5:c1",
            "ragflow_document_id": "rag-001",
            "page": 5,
            "chunk_id": "c1",
            "retrieval_score": 0.91,
            "source_quality": 0.97,
            "text": "CFL stability time step",
        }

    def test_json_index_preserves_upstream_fields_and_resolves_full_page(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "pages.json").write_text(json.dumps([{"page_idx": 5, "text": "完整第 5 页：CFL 公式、材料条件和边界说明。"}], ensure_ascii=False), encoding="utf-8")
            (base / "index.json").write_text(json.dumps([self._row(base)], ensure_ascii=False), encoding="utf-8")
            retriever = UpstreamLiteratureRetriever.from_json(base / "index.json")
            hits = retriever.search("FDTD CFL stability", top_k=3)
            page = retriever.get_page("doc-001", 5)
            self.assertEqual(hits[0].upstream["doc_id"], "doc-001")
            self.assertEqual(hits[0].upstream["ragflow_document_id"], "rag-001")
            self.assertEqual(page[0].content, "完整第 5 页：CFL 公式、材料条件和边界说明。")
            self.assertTrue(page[0].is_page_fallback)
            self.assertEqual(page[0].upstream["page_asset"]["raw_path"], "papers/doc-001.pdf")
            self.assertEqual(page[0].upstream["page_asset"]["page"], 5)

    def test_sqlite_index_uses_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite"
            row = self._row(path.parent)
            columns = ", ".join(f'"{key}" TEXT' for key in row)
            connection = sqlite3.connect(path)
            try:
                connection.execute(f"CREATE TABLE evidence_index ({columns})")
                keys = list(row)
                placeholders = ", ".join("?" for _ in keys)
                connection.execute(f"INSERT INTO evidence_index ({', '.join(keys)}) VALUES ({placeholders})", [str(row[key]) if row[key] is not None else None for key in keys])
                connection.commit()
            finally:
                connection.close()
            retriever = UpstreamLiteratureRetriever.from_sqlite(path)
            result = retriever.search("FDTD CFL", top_k=1)[0]
            self.assertEqual(result.upstream["dataset_key"], "em-public")
            self.assertEqual(result.upstream["chunk_id"], "c1")
