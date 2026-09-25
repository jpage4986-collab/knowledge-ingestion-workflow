import sqlite3
import unittest

from ragflow_document_finalizer import find_local_document_row


class FakeIndex:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE TABLE papers (doc_id TEXT, ragflow_document_id TEXT, ragflow_pdf_document_id TEXT)"
        )
        self.connection.execute(
            "INSERT INTO papers VALUES (?, ?, ?)",
            ("local-doc", "", "pdf-document-id"),
        )


class RAGFlowDocumentFinalizerTests(unittest.TestCase):
    def test_resolves_document_through_original_pdf_asset_id(self):
        index = FakeIndex()
        try:
            row = find_local_document_row(index, "pdf-document-id")
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "local-doc")
        finally:
            index.connection.close()


if __name__ == "__main__":
    unittest.main()
