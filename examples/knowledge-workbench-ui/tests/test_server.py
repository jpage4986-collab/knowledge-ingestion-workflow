from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class ReadOnlyDashboardTests(unittest.TestCase):
    def test_document_serialization_omits_local_paths(self):
        item = {
            "id": "doc-1",
            "name": "manual.pdf",
            "location": r"C:\private\library\manual.pdf",
            "run": "DONE",
            "chunk_count": 3,
        }

        result = server._serialize_document(item)

        self.assertEqual(result["name"], "manual.pdf")
        self.assertEqual(result["chunk_count"], 3)
        self.assertNotIn("location", result)
        self.assertNotIn("private", str(result))

    def test_chunk_serialization_limits_content_and_keeps_page(self):
        result = server._serialize_chunk({
            "id": "chunk-1",
            "content": "x" * 1500,
            "positions": [12, 0, 0, 0, 0],
            "important_keywords": ["FDTD", "mesh"],
        })

        self.assertEqual(len(result["content"]), 1400)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["page"], 12)
        self.assertEqual(result["keywords"], ["FDTD", "mesh"])

    def test_search_filters_actual_documents_in_read_only_page(self):
        documents = [
            {"id": "a", "name": "width_sweep.py", "run": "DONE", "chunk_count": 2},
            {"id": "b", "name": "manual.pdf", "run": "FAIL", "chunk_count": 0},
        ]
        with patch.object(server, "get_datasets", return_value=[{"id": "ds-1", "name": "Library"}]), \
             patch.object(server, "_all_documents", return_value=documents):
            result = server.get_document_page("ds-1", {"q": ["WIDTH"], "run": ["DONE"]})

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["id"], "a")

    def test_post_requests_are_rejected(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.WorkbenchHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{httpd.server_port}/api/datasets"
            request = Request(url, data=b"{}", method="POST")
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 405)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
