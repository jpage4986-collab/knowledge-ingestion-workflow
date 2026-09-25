"""Ingest curated remote or local PDF sources through the same gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ingest_model import build_ingest_reviewer
from sync_library import PaperAttachment, PaperIndex, SinglePaperSource, SyncRunner, download_open_pdf, load_config, slug


def run_seed_ingest(config: Mapping[str, Any], manifest_path: Path, dry_run: bool = False) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("seed manifest must be a JSON list")
    storage = config["storage"]
    root = Path(storage["root"])
    index = PaperIndex(Path(storage["index_db"]) if not dry_run else Path(":memory:"), Path(storage["summary_csv"]))
    reviewer = build_ingest_reviewer(config)
    results: list[dict[str, Any]] = []
    try:
        for item in manifest:
            if not isinstance(item, Mapping) or not item.get("title") or not (item.get("url") or item.get("path")):
                results.append({"status": "invalid_manifest_item"})
                continue
            source_id = str(item.get("id") or slug(str(item["title"])))
            destination = root / "discovery" / "first_party" / f"{source_id}.pdf"
            if dry_run:
                results.append({
                    "id": source_id,
                    "status": "planned",
                    "title": str(item["title"]),
                    "source": str(item.get("path") or item.get("url")),
                })
                continue
            try:
                local_path = Path(str(item.get("path", ""))).expanduser() if item.get("path") else None
                if local_path is not None:
                    if not local_path.is_file() or local_path.suffix.lower() != ".pdf":
                        raise FileNotFoundError(f"local PDF not found: {local_path}")
                    source_path = local_path
                else:
                    download_open_pdf(str(item["url"]), destination)
                    source_path = destination
                paper = PaperAttachment(
                    zotero_key=f"first-party-{source_id}", parent_key="", title=str(item["title"]),
                    authors=str(item.get("authors", "")), year=str(item.get("year", "")), doi=str(item.get("doi", "")),
                    tags="curated;" + str(item.get("source", "local")) + ";" + str(item.get("tags", "")), filename=source_path.name,
                    local_path=source_path, dataset_key=str(item.get("dataset_key", "cases")),
                    abstract=str(item.get("abstract", "")), landing_url=str(item.get("landing_url", item.get("url", ""))),
                    pdf_url=str(item.get("url", "")), source_provider=str(item.get("source", "local")),
                )
                result = SyncRunner(config, SinglePaperSource(paper), index, reviewer).run(limit=1)[0]
                results.append({"id": source_id, **result})
            except Exception as exc:
                results.append({"id": source_id, "status": "seed_failed", "title": str(item["title"]), "error": str(exc)})
        return results
    finally:
        index.close()
