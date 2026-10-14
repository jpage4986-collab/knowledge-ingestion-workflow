from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import job_history
import server


class InterruptedJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = job_history.JobHistoryStore(Path(self.temporary.name, "jobs.json"))

    def tearDown(self):
        self.temporary.cleanup()

    def test_running_job_becomes_interrupted_on_recover(self):
        job = {"id": "job-9", "status": "running", "created_at": 1, "updated_at": 2, "items": []}
        self.store.record(job)
        interrupted = self.store.load_interrupted()
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["status"], "interrupted")
        self.assertEqual(interrupted[0]["stage"], "interrupted")

    def test_terminal_jobs_not_interrupted(self):
        self.store.record({"id": "done", "status": "completed", "created_at": 1})
        self.assertEqual(self.store.load_interrupted(), [])

    def test_resume_with_uploaded_docs_is_safe(self):
        stored = {
            "id": "job-10", "status": "interrupted", "mode": "local",
            "dataset_id": "ds1", "local_file_count": 1, "lost_local_bytes": True,
            "items": [{"id": "i1", "document_ids": ["doc1"], "status": "running"}],
            "model_gate": {"ready": True},
        }
        resumed = server._safe_resume_job(stored)
        self.assertTrue(resumed["resumable"])
        self.assertTrue(resumed["resumed"])

    def test_resume_without_local_bytes_and_no_docs_is_not_resumable(self):
        stored = {
            "id": "job-11", "status": "interrupted", "mode": "local",
            "dataset_id": "ds1", "local_file_count": 2, "lost_local_bytes": True,
            "items": [{"id": "i1", "document_ids": [], "status": "failed"}],
            "model_gate": {"ready": True},
        }
        resumed = server._safe_resume_job(stored)
        self.assertFalse(resumed["resumable"])
        self.assertTrue(resumed["lost_local_bytes"])
        self.assertIn("本地原始字节已丢失", resumed["message"])


class PersistenceTests(unittest.TestCase):
    def test_job_update_persists_to_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = job_history.JobHistoryStore(Path(tmp, "j.json"))
            with patch.object(server, "HISTORY", store):
                job_id, _ = self._make_job()
                server._update_job(job_id, status="running", stage="parse", message="parsing")
                persisted = store.snapshot()[job_id]
                self.assertEqual(persisted["status"], "running")
                self.assertEqual(persisted["stage"], "parse")

    def test_item_check_update_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = job_history.JobHistoryStore(Path(tmp, "j2.json"))
            with patch.object(server, "HISTORY", store):
                job_id, item_id = self._make_job(with_item=True)
                server._update_item_check(job_id, item_id, "parse", "running", "doc parsing")
                persisted = store.snapshot()[job_id]
                self.assertEqual(persisted["items"][0]["checks"]["parse"]["state"], "running")

    def _make_job(self, with_item=False):
        import secrets
        job_id = "job-" + secrets.token_hex(4)
        job = {
            "id": job_id, "dataset_id": "ds1", "dataset_name": "Sandbox",
            "mode": "online", "query": "t", "limit": 1, "parallelism": 1,
            "status": "queued", "stage": "queued", "message": "queued",
            "created_at": 1, "updated_at": 1, "items": [], "errors": [],
            "local_file_count": 0, "lost_local_bytes": False,
            "model_gate": {"ready": True, "provider": "bge", "version": "1"},
        }
        item_id = "item-1"
        if with_item:
            job["items"].append({"id": item_id, "document_ids": [], "checks": {}})
        with server._ingestion_jobs_lock:
            server._ingestion_jobs[job_id] = job
            server.HISTORY.record(job)
        return job_id, item_id

    def test_old_library_write_rejected(self):
        # A non-sandbox dataset must be rejected for writes.
        fake_raw = [
            {"id": "old-lib", "name": "old-library", "description": "legacy"},
            {"id": "HIH-SANDBOX-1", "name": "HIH-SANDBOX-1", "description": "[knowledge-workbench-sandbox:v1]"},
        ]
        with patch.object(server, "_raw_datasets", return_value=fake_raw), patch.object(server, "_registered_sandbox_ids", return_value={"HIH-SANDBOX-1"}):
            with self.assertRaises(server.ServiceError) as ctx:
                server._sandbox_dataset("old-lib")
            self.assertEqual(ctx.exception.status, 403)
            # sandbox id works
            ds = server._sandbox_dataset("HIH-SANDBOX-1")
            self.assertEqual(ds["id"], "HIH-SANDBOX-1")


class ResumeEndpointTests(unittest.TestCase):
    def test_resume_endpoint_rejects_unresumable_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = job_history.JobHistoryStore(Path(tmp, "jr.json"))
            unresumable = {
                "id": "job-20", "status": "interrupted", "mode": "local",
                "dataset_id": "ds1", "local_file_count": 1, "lost_local_bytes": True,
                "items": [{"id": "i", "document_ids": [], "status": "failed"}],
            }
            store.record(unresumable)
            with patch.object(server, "HISTORY", store):
                result = server._safe_resume_job(unresumable)
                self.assertFalse(result["resumable"])


if __name__ == "__main__":
    unittest.main()
