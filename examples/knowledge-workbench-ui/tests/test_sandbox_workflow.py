from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class SandboxWorkflowTests(unittest.TestCase):
    def test_existing_library_is_rejected_as_a_write_target(self):
        existing = {
            "id": "legacy-dataset",
            "name": "Existing Library",
            "description": "User-owned knowledge base",
        }
        with patch.object(server, "_raw_datasets", return_value=[existing]), \
             patch.object(server, "_write_api_json") as write:
            with self.assertRaises(server.ServiceError) as raised:
                server._sandbox_dataset("legacy-dataset")
        self.assertEqual(raised.exception.status, 403)
        write.assert_not_called()

    def test_sandbox_creation_clones_read_only_template_settings(self):
        template = {
            "id": "template-dataset",
            "name": "Research papers",
            "description": "Existing library",
            "embedding_model": "embedding@provider",
            "chunk_method": "naive",
            "parser_config": {
                "chunk_token_num": 700,
                "layout_recognize": "DeepDOC",
                "image_context_size": 0,
                "llm_id": "provider-private-model-id",
                "table_context_size": 0,
            },
        }
        with patch.object(server, "_raw_datasets", return_value=[template]), \
             patch.object(server, "_write_api_json", return_value={"data": {"id": "new-sandbox"}}) as write:
            result = server.create_sandbox("template-dataset")

        self.assertEqual(result["id"], "new-sandbox")
        self.assertTrue(result["is_sandbox"])
        args, kwargs = write.call_args
        self.assertEqual(args[0], "/api/v1/datasets")
        self.assertEqual(args[1]["embedding_model"], "embedding@provider")
        self.assertEqual(args[1]["parser_config"], {"chunk_token_num": 700, "layout_recognize": "DeepDOC"})
        self.assertTrue(args[1]["name"].startswith(server.SANDBOX_PREFIX))
        self.assertIn(server.SANDBOX_MARKER, args[1]["description"])

    def test_upload_is_sandbox_scoped_and_starts_parse(self):
        sandbox = {"id": "sandbox-1", "name": "Harness sandbox"}
        upload = {"data": [{"id": "document-1", "name": "test.txt"}]}
        with patch.object(server, "_sandbox_dataset", return_value=sandbox), \
             patch.object(server, "_write_api_multipart", return_value=upload) as send_file, \
             patch.object(server, "_trigger_parse") as parse:
            result = server._upload_to_sandbox("sandbox-1", "notes.txt", b"test content")

        send_file.assert_called_once()
        parse.assert_called_once_with("sandbox-1", ["document-1"])
        self.assertTrue(result["parse_started"])
        self.assertEqual(result["items"][0]["id"], "document-1")

    def test_local_and_private_network_urls_are_rejected(self):
        for url in ("http://127.0.0.1/file.pdf", "http://localhost/file.pdf", "file:///C:/notes.pdf"):
            with self.subTest(url=url), self.assertRaises(server.ServiceError) as raised:
                server._public_http_url(url)
            self.assertEqual(raised.exception.status, 400)

    def test_managed_proxy_synthetic_dns_range_is_allowed_for_public_hosts(self):
        with patch("server.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("198.18.1.114", 443))]), \
             patch("server.urllib.request.getproxies", return_value={"https": "http://127.0.0.1:7900"}):
            self.assertEqual(
                server._public_http_url("https://arxiv.org/pdf/1609.07600"),
                "https://arxiv.org/pdf/1609.07600",
            )

        with patch("server.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("198.18.1.114", 443))]), \
             patch("server.urllib.request.getproxies", return_value={}):
            with self.assertRaises(server.ServiceError):
                server._public_http_url("https://arxiv.org/pdf/1609.07600")

    def test_file_name_is_reduced_to_a_safe_basename(self):
        self.assertEqual(server._safe_filename("..\\private\\notes.pdf"), "notes.pdf")
        with self.assertRaises(server.ServiceError):
            server._safe_filename("secrets.exe")


if __name__ == "__main__":
    unittest.main()
