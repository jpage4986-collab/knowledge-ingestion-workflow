"""Keep per-document RAGFlow finalizers alive for the local ingestion loop.

The upload worker can enqueue a PDF and return before RAGFlow finishes parsing.
This small local supervisor discovers those rows from the G-drive SQLite index,
starts one finalizer per active document, and records only non-sensitive status.
It never uploads, deletes, or rewrites a source document.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sync_library import load_config, utc_now


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _process_has_document(document_id: str) -> bool:
    try:
        import psutil  # type: ignore
    except ImportError:
        # The normal Python runtime does not necessarily bundle psutil.  On
        # Windows, use the read-only process inventory already used by the
        # launcher so a manually started finalizer is not duplicated.
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return any("ragflow_document_finalizer.py" in line and document_id in line for line in result.stdout.splitlines())
    needle = str(document_id)
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, OSError):
            continue
        if "ragflow_document_finalizer.py" in command and needle in command:
            return True
    return False


def _dataset_key(config: dict[str, Any], dataset_id: str) -> str | None:
    for key, value in (config.get("ragflow", {}).get("dataset_ids", {}) or {}).items():
        if str(value) == str(dataset_id):
            return str(key)
    return None


def supervise(config_path: Path, poll_seconds: int, max_finalizers: int) -> int:
    config = load_config(config_path)
    root = Path(str(config["storage"]["root"]))
    evaluation = root / "evaluation"
    runtime = root / "runtime"
    status_path = evaluation / "ragflow_finalizer_supervisor_status.json"
    db_path = Path(str(config["storage"]["index_db"]))
    finalizer = config_path.with_name("ragflow_document_finalizer.py")
    while True:
        launched: list[str] = []
        skipped: list[dict[str, str]] = []
        active_count = 0
        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                "SELECT doc_id, ragflow_document_id, ragflow_dataset_id, status "
                "FROM papers WHERE ragflow_document_id IS NOT NULL AND ragflow_document_id <> '' "
                "AND status = 'uploaded_to_ragflow' ORDER BY updated_at"
            ).fetchall()
        finally:
            connection.close()

        for local_doc_id, ragflow_id, dataset_id, row_status in rows:
            ragflow_id = str(ragflow_id)
            state_path = evaluation / f"ragflow_finalizer_{ragflow_id}.json"
            state: dict[str, Any] = {}
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
            if state.get("status") in {"admission_completed", "admission_failed", "failed"}:
                skipped.append({"document_id": ragflow_id, "reason": str(state.get("status"))})
                continue
            if _process_has_document(ragflow_id):
                active_count += 1
                skipped.append({"document_id": ragflow_id, "reason": "already_running"})
                continue
            if active_count >= max(1, max_finalizers):
                skipped.append({"document_id": ragflow_id, "reason": "concurrency_limit"})
                continue
            dataset_key = _dataset_key(config, str(dataset_id))
            if not dataset_key:
                skipped.append({"document_id": ragflow_id, "reason": "dataset_id_not_mapped"})
                continue
            log_path = runtime / f"ragflow_finalizer_{ragflow_id}.log"
            log = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(finalizer),
                "--config",
                str(config_path),
                "--dataset-key",
                dataset_key,
                "--document-id",
                ragflow_id,
                "--poll-seconds",
                str(max(15, poll_seconds // 2)),
                "--timeout-seconds",
                "7200",
            ]
            subprocess.Popen(
                command,
                cwd=str(config_path.parent),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            log.close()
            active_count += 1
            launched.append(ragflow_id)

        _write_json(
            status_path,
            {
                "checked_at": utc_now(),
                "active_rows": len(rows),
                "launched": launched,
                "skipped": skipped,
                "poll_seconds": poll_seconds,
                "max_finalizers": max_finalizers,
            },
        )
        time.sleep(max(30, poll_seconds))


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise local RAGFlow document finalizers")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-finalizers", type=int, default=1)
    args = parser.parse_args()
    supervise(args.config, args.poll_seconds, args.max_finalizers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
