"""Recoverably finalize one long-running RAGFlow document.

The helper is intentionally narrow: it never uploads or deletes a document.
It waits for an already-triggered parse, records a service-side chunk snapshot,
updates the matching local row, and runs admission review for that exact row.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sync_library import PaperIndex, RAGFlowClient, load_config, utc_now


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_local_document_row(index: PaperIndex, document_id: str):
    """Resolve either the serving document ID or its separately stored PDF ID."""
    return index.connection.execute(
        "SELECT * FROM papers WHERE ragflow_document_id = ? OR ragflow_pdf_document_id = ? OR doc_id = ?",
        (document_id, document_id, document_id),
    ).fetchone()


def finalize(config_path: Path, dataset_key: str, document_id: str, poll_seconds: int, timeout_seconds: int) -> int:
    config_path = config_path.resolve()
    config = load_config(config_path)
    storage_root = Path(str(config["storage"]["root"]))
    status_path = storage_root / "evaluation" / f"ragflow_finalizer_{document_id}.json"
    dataset_id = str(config["ragflow"]["dataset_ids"][dataset_key])
    client = RAGFlowClient(str(config["ragflow"]["base_url"]), dataset_id, str(config["ragflow"]["api_key_env"]), timeout=120)
    started = time.monotonic()
    while True:
        document = client.document(document_id)
        run = str(document.get("run") or document.get("status") or "").upper()
        payload = {
            "checked_at": utc_now(),
            "dataset_key": dataset_key,
            "dataset_id": dataset_id,
            "document_id": document_id,
            "run": run,
            "progress": document.get("progress"),
            "chunk_count": document.get("chunk_count"),
        }
        _write_status(status_path, payload)
        if run in {"DONE", "FAIL", "FAILED", "CANCEL", "CANCELED", "CANCELLED"}:
            break
        if time.monotonic() - started >= timeout_seconds:
            payload.update({"status": "timeout", "error": "等待 RAGFlow 文档解析超时；任务未被删除，可稍后恢复检查"})
            _write_status(status_path, payload)
            return 2
        time.sleep(max(15, poll_seconds))

    chunks = client.list_chunks(document_id) if run == "DONE" else []
    if run != "DONE" or not chunks:
        payload.update({"status": "failed", "error": "RAGFlow 文档未完成或没有可检索分块；原始 PDF 与失败记录保留"})
        _write_status(status_path, payload)
        return 1

    index = PaperIndex(Path(config["storage"]["index_db"]), Path(config["storage"]["summary_csv"]))
    try:
        row = find_local_document_row(index, document_id)
    finally:
        index.close()
    if row is None:
        payload.update({"status": "failed", "error": "本地索引找不到对应文献；未猜测写入"})
        _write_status(status_path, payload)
        return 1

    snapshot = storage_root / "evaluation" / "live_ragflow_snapshots" / f"{document_id}.md"
    content_list = snapshot.with_name(f"{document_id}_content_list.json")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    sections = ["# RAGFlow service-side MinerU snapshot", "", f"document_id: {document_id}", ""]
    content_items: list[dict[str, Any]] = []

    # RAGFlow's positions are reliable for short documents, but for a long PDF
    # its page batches may be numbered relative to each task.  Prefer the
    # original local MinerU content_list.json, whose page_idx is global and is
    # also used by the visual page-fallback path.  Keep the source explicit.
    local_content_list_path = ""
    local_markdown_path = str(row["mineru_markdown_path"] or "")
    if local_markdown_path:
        candidates = sorted(Path(local_markdown_path).parent.glob("*_content_list.json"))
        if candidates:
            local_content_list_path = str(candidates[0])
            try:
                local_items = json.loads(candidates[0].read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                local_items = []
            if isinstance(local_items, list):
                for item in local_items:
                    if not isinstance(item, dict) or not isinstance(item.get("page_idx"), int):
                        continue
                    enriched = dict(item)
                    enriched["source"] = "local_mineru_content_list"
                    enriched["service_document_id"] = document_id
                    content_items.append(enriched)

    # If the full local page index is unavailable, retain a clearly scoped
    # service-chunk fallback rather than silently presenting local task pages
    # as global PDF pages.
    use_service_positions = not content_items
    for number, chunk in enumerate(chunks, 1):
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        sections.extend([f"## Chunk {number}", content, ""])
        if not use_service_positions:
            continue
        positions = chunk.get("positions")
        pages: set[int] = set()
        if isinstance(positions, list):
            for position in positions:
                if isinstance(position, list) and position and isinstance(position[0], int) and position[0] > 0:
                    pages.add(position[0] - 1)
        for page_idx in sorted(pages):
            content_items.append({
                "page_idx": page_idx,
                "text": content,
                "source": "ragflow_mineru_chunk_relative_task_page",
                "chunk_id": str(chunk.get("id") or ""),
                "page_scope": "relative_to_ragflow_task_batch",
            })
    snapshot.write_text("\n".join(sections), encoding="utf-8")
    content_list.write_text(json.dumps(content_items, ensure_ascii=False, indent=2), encoding="utf-8")

    index = PaperIndex(Path(config["storage"]["index_db"]), Path(config["storage"]["summary_csv"]))
    try:
        updated = dict(row)
        updated["ragflow_document_id"] = document_id
        updated["ragflow_dataset_id"] = dataset_id
        updated["ragflow_parser_config"] = json.dumps(document.get("parser_config") or {}, ensure_ascii=False)
        updated["status"] = "uploaded_to_ragflow"
        updated["last_error"] = "服务端 MinerU PDF 解析完成，等待指定文档准入复核。"
        updated["updated_at"] = utc_now()
        index.upsert_paper(updated)
    finally:
        index.close()

    admission = subprocess.run(
        [sys.executable, str(config_path.with_name("sync_library.py")), "--config", str(config_path), "recheck-admission", "--doc-id", str(updated["doc_id"])],
        cwd=str(config_path.parent), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
    )
    payload.update({
        "status": "admission_completed" if admission.returncode == 0 else "admission_failed",
        "chunk_count": len(chunks),
        "snapshot": str(snapshot),
        "content_list": str(content_list),
        "source_content_list": local_content_list_path,
        "content_list_page_scope": "global_local_mineru" if local_content_list_path and content_items else "relative_ragflow_task_batch",
        "admission_exit_code": admission.returncode,
        "admission_output_tail": (admission.stdout or admission.stderr)[-2000:],
    })
    _write_status(status_path, payload)
    return 0 if admission.returncode == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-key", required=True, choices=("fundamentals", "materials", "papers", "software_manuals", "cases"))
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    return finalize(args.config.resolve(), args.dataset_key, args.document_id, args.poll_seconds, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
