from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import job_history
import server


class JobHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = job_history.JobHistoryStore(Path(self.temporary.name, "jobs.json"))

    def tearDown(self):
        self.temporary.cleanup()

    def test_seed_replay_and_recoverable_round_trip(self):
        job = {"id": "job-1", "status": "completed", "created_at": 100, "updated_at": 110, "dataset_name": "Sandbox"}
        self.assertEqual(self.store.seed_from_memory([job]), 1)
        self.assertEqual(self.store.seed_from_memory([job]), 0)
        recovered = self.store.load_recoverable()
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["id"], "job-1")
        self.assertTrue(self.store.path.is_file())
        payload = json.loads(self.store.path.read_text("utf-8"))
        self.assertEqual(payload["version"], job_history.HISTORY_VERSION)

    def test_corrupt_index_returns_empty_without_raising(self):
        self.store.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.load_recoverable(), [])

    def test_server_routes_use_persisted_history(self):
        self.assertFalse(server._model_gate()["ready"])
        with patch.object(server, "HISTORY", self.store):
            self.assertEqual(server.restore_recoverable_jobs(), [])
            completed = {"id": "job-2", "status": "failed", "created_at": 1, "updated_at": 2}
            self.store.record(completed)
            restored = server.restore_recoverable_jobs()
            self.assertEqual(len(restored), 1)
            self.assertTrue(restored[0]["restored"])
            self.assertEqual(restored[0]["id"], "job-2")


if __name__ == "__main__":
    unittest.main()
