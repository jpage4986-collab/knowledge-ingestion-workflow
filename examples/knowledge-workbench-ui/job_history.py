"""Persistent ingestion job history with restart recovery for the workbench."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

HISTORY_VERSION = 1
HISTORY_LIMIT = 100

_history_lock = threading.RLock()


ROOT_DIR = Path(__file__).resolve().parent


class JobHistoryStore:
    """Bounded JSON job index. Active in-memory state remains authoritative."""

    def __init__(self, path: str | Path | None = None):
        raw = path or os.environ.get("KNOWLEDGE_JOB_HISTORY", "")
        if not raw:
            raw = str(ROOT_DIR / "job-history.json")
        self.path = Path(raw).expanduser()

    def _load(self) -> dict[str, dict]:
        if not self.path or not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, dict):
            return {}
        return {
            key: dict(item)
            for key, item in jobs.items()
            if isinstance(item, dict) and str(item.get("id") or "")
        }

    def _save(self, jobs: dict[str, dict]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(jobs.values(), key=lambda item: int(item.get("created_at") or 0))[-HISTORY_LIMIT:]
        payload = {
            "version": HISTORY_VERSION,
            "jobs": {item["id"]: item for item in ordered},
        }
        handle, name = tempfile.mkstemp(prefix=".harness-jobs-", suffix=".json", dir=str(self.path.parent))
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as handle_file:
                json.dump(payload, handle_file, ensure_ascii=False)
            os.replace(name, self.path)
        except BaseException:
            if os.path.exists(name):
                os.unlink(name)
            raise

    def seed_from_memory(self, jobs: list[dict]) -> int:
        """Start or repair the index from live in-memory jobs."""
        with _history_lock:
            existing = self._load()
            fresh = 0
            for job in jobs:
                if str(job.get("id") or "") and job["id"] not in existing:
                    existing[job["id"]] = dict(job)
                    fresh += 1
            if fresh:
                self._save(existing)
            return fresh

    def load_recoverable(self) -> list[dict]:
        """Terminal jobs persisted before the previous process ended."""
        with _history_lock:
            jobs = self._load()
            finished = [item for item in jobs.values() if str(item.get("status")) in {
                "completed", "completed_with_review", "failed",
            }]
            finished.sort(key=lambda item: int(item.get("updated_at") or item.get("created_at") or 0), reverse=True)
            return finished[:50]

    def load_interrupted(self) -> list[dict]:
        """Active jobs persisted as running/queued before the previous process ended."""
        with _history_lock:
            jobs = self._load()
            active = [item for item in jobs.values() if str(item.get("status")) in {"queued", "running", "interrupted"}]
            for item in active:
                if item.get("status") != "interrupted":
                    item["status"] = "interrupted"
                    item["stage"] = "interrupted"
                    item["message"] = "任务在进程结束前被中断，已保存文档标识，可恢复继续检查"
                    item["updated_at"] = int(item.get("updated_at") or item.get("created_at") or 0)
                    self._persist_active(item)
            active.sort(key=lambda item: int(item.get("updated_at") or item.get("created_at") or 0), reverse=True)
            return active[:50]

    def _persist_active(self, job: dict) -> None:
        with _history_lock:
            jobs = self._load()
            jobs[job["id"]] = dict(job)
            self._save(jobs)

    def record(self, job: dict) -> None:
        if not str(job.get("id") or ""):
            return
        with _history_lock:
            jobs = self._load()
            jobs[job["id"]] = dict(job)
            self._save(jobs)

    def snapshot(self) -> dict[str, Any]:
        """Live in-memory jobs plus terminal history from the index."""
        return self._load()
