"""Repair RAGFlow documents that reported DONE but produced zero chunks."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sync_library import RAGFlowClient, load_config, published_document_name


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _preparsed_update(config: Mapping[str, Any]) -> dict[str, Any]:
    setting = config.get("ragflow", {}).get("preparsed_markdown", {})
    if not isinstance(setting, Mapping) or not setting.get("enabled"):
        raise ValueError("需要启用 ragflow.preparsed_markdown 作为自动兜底")
    parser_config = dict(setting.get("parser_config", {}))
    for key in ("delimiter", "children_delimiter"):
        if parser_config.get(key) == r"\n":
            parser_config[key] = "\n"
    parent_child = parser_config.get("parent_child")
    if isinstance(parent_child, Mapping):
        parent_child = dict(parent_child)
        if parent_child.get("children_delimiter") == r"\n":
            parent_child["children_delimiter"] = "\n"
        parser_config["parent_child"] = parent_child
    return {"chunk_method": str(setting.get("chunk_method", "naive")), "parser_config": parser_config}


def repair(config: Mapping[str, Any], doc_ids: list[str]) -> list[dict[str, Any]]:
    storage = config["storage"]
    root = Path(storage["root"])
    connection = sqlite3.connect(Path(storage["index_db"]))
    connection.row_factory = sqlite3.Row
    reports: list[dict[str, Any]] = []
    try:
        update = _preparsed_update(config)
        for doc_id in doc_ids:
            row = connection.execute(
                "SELECT doc_id, ragflow_document_id, ragflow_dataset_id, mineru_markdown_path FROM papers WHERE doc_id=?",
                (doc_id,),
            ).fetchone()
            report: dict[str, Any] = {"doc_id": doc_id, "started_at": _now()}
            if row is None:
                report.update({"status": "not_found"})
                reports.append(report)
                continue
            old_id = str(row["ragflow_document_id"] or "")
            dataset_id = str(row["ragflow_dataset_id"] or "")
            candidates = [
                root / "published" / f"{doc_id}.md",
                Path(str(row["mineru_markdown_path"] or "")),
            ]
            source = next((path for path in candidates if path.is_file()), None)
            if not old_id or not dataset_id or source is None:
                report.update({"status": "missing_source_or_ragflow_id", "source": str(source or "")})
                reports.append(report)
                continue
            client = RAGFlowClient(str(config["ragflow"]["base_url"]), dataset_id, str(config["ragflow"]["api_key_env"]))
            new_id = ""
            try:
                new_id = client.upload(source, display_name=f"{published_document_name(doc_id, source.read_text(encoding='utf-8', errors='replace'))[:-3]}repaired.md")
                client.update_document(new_id, update)
                client.trigger_parse(new_id)
                client.wait_done(new_id)
                chunks = client.list_chunks(new_id)
                if not chunks:
                    raise RuntimeError("兜底 Markdown 仍未生成分块")
                client.update_document(old_id, {"enabled": 0})
                connection.execute(
                    "UPDATE papers SET ragflow_document_id=?, ragflow_parser_config=?, last_error='', updated_at=? WHERE doc_id=?",
                    (new_id, json.dumps(update, ensure_ascii=False), _now(), doc_id),
                )
                connection.commit()
                report.update({"status": "repaired", "old_id": old_id, "new_id": new_id, "chunk_count": len(chunks), "source": str(source)})
            except Exception as exc:
                if new_id:
                    try:
                        client.delete_documents([new_id])
                    except Exception:
                        pass
                report.update({"status": "failed", "old_id": old_id, "error": f"{type(exc).__name__}: {exc}"})
            reports.append(report)
    finally:
        connection.close()
    output = root / "evaluation" / "zero_chunk_repair_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="repair zero-chunk RAGFlow documents using local MinerU Markdown")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--doc-id", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(repair(load_config(args.config), args.doc_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
