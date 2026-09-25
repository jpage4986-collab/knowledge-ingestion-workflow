"""Create the five formal RAGFlow datasets with an explicit Embedding ID.

This script does not start RAGFlow, upload documents, or create an index
dataset. It only creates missing content datasets when run without
``--dry-run``. Use ``--write-config`` to persist the returned dataset IDs
after checking the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sync_library import RAGFlowClient, load_config


def parser_config_for(item: dict) -> dict:
    baseline = item.get("chunk_baseline", {})
    overlap = float(baseline.get("overlap", 0) or 0)
    if overlap > 1:
        overlap /= 100
    context = int(baseline.get("image_table_context", 0) or 0)
    return {
        # PDFs are converted by the local MinerU pipeline before admission,
        # and the formal datasets are also configured with RAGFlow's MinerU
        # layout identifier so a future direct-PDF upload cannot silently
        # fall back to a different layout parser.
        "layout_recognize": "mineru-from-env@mineru-from-env(1)@MinerU",
        "chunk_token_num": int(baseline.get("size", 512)),
        "delimiter": str(baseline.get("separator", "\n")),
        "html4excel": bool(baseline.get("table_to_html", False)),
        "raptor": {"use_raptor": False},
    }


def persist_dataset_ids(config: dict, structure: dict, resolved_ids: dict[str, str], config_path: Path) -> dict[str, str]:
    """Persist only resolved formal dataset IDs, preserving other config fields."""
    name_to_key = {item["name"]: item["key"] for item in structure["ragflow_datasets"]}
    rag_cfg = config.setdefault("ragflow", {})
    mapping = dict(rag_cfg.get("dataset_ids", {}))
    for name, dataset_id in resolved_ids.items():
        key = name_to_key.get(name)
        if key and dataset_id:
            mapping[key] = dataset_id
    rag_cfg["dataset_ids"] = mapping
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--structure", type=Path, default=Path("..") / "knowledge_base_structure_v2" / "library_structure.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-config", action="store_true", help="write the resolved dataset IDs back to config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    rag_cfg = config["ragflow"]
    embedding_model = str(rag_cfg.get("embedding_model", "")).strip()
    if not embedding_model or embedding_model.startswith("FILL_"):
        raise SystemExit("请在 config.json 的 ragflow.embedding_model 填写 RAGFlow 页面中显示的精确模型标识。")
    names = [item["name"] for item in structure["ragflow_datasets"]]
    if args.dry_run:
        for name in names:
            item = next(value for value in structure["ragflow_datasets"] if value["name"] == name)
            print(json.dumps({"name": name, "embedding_model": embedding_model, "parser_config": parser_config_for(item), "status": "planned"}, ensure_ascii=False))
        return 0
    client = RAGFlowClient(rag_cfg["base_url"], "", rag_cfg["api_key_env"])
    existing = {str(item.get("name")): item for item in client.list_datasets()}
    resolved_ids: dict[str, str] = {}
    for item in structure["ragflow_datasets"]:
        name = item["name"]
        if name in existing:
            dataset_id = str(existing[name].get("id", ""))
            if dataset_id:
                resolved_ids[name] = dataset_id
            print(json.dumps({"name": name, "id": dataset_id, "status": "exists"}, ensure_ascii=False))
            continue
        created = client.create_dataset(
            name,
            embedding_model,
            description=item.get("role", ""),
            chunk_method="naive",
            parser_config=parser_config_for(item),
        )
        dataset_id = str(created.get("id", ""))
        if dataset_id:
            resolved_ids[name] = dataset_id
        print(json.dumps({"name": name, "id": dataset_id, "status": "created", "embedding_model": created.get("embedding_model", embedding_model)}, ensure_ascii=False))
    if args.write_config:
        mapping = persist_dataset_ids(config, structure, resolved_ids, args.config)
        print(json.dumps({"status": "config_updated", "config": str(args.config), "dataset_ids": mapping}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
