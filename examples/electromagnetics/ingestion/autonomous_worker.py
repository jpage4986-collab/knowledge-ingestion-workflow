"""Local autonomous worker for the literature ingestion loop.

The worker is intentionally a normal foreground process: it does not start
RAGFlow, MinerU, or any model service. It can be launched by Windows Task
Scheduler or a user-owned service wrapper after the workflow is accepted.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from maintenance import run_maintenance
from retry_failed import retry_failed
from seed_ingest import run_seed_ingest
from sync_library import load_config, run_cycle, utc_now
from adaptive_chunk_repair import run_adaptive_chunk_repair


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_material_query(query: str) -> bool:
    terms = ("material", "materials", "permittivity", "permeability", "conductivity", "dielectric", "loss tangent", "debye", "drude", "lorentz", "cole cole", "cole-cole", "材料", "介电", "磁导", "电导", "损耗")
    lowered = str(query or "").lower()
    return any(term in lowered for term in terms)


def _material_gap_queries(config: Mapping[str, Any]) -> list[str]:
    """Rotate concrete material-gap searches into the next discovery cycle."""
    root = Path(str(config["storage"]["root"]))
    queue_path = root / "material_gap_queue.csv"
    if not queue_path.is_file():
        return []
    registry_cfg = config.get("material_registry", {})
    batch_size = max(1, int(registry_cfg.get("gap_query_batch", 3))) if isinstance(registry_cfg, Mapping) else 3
    state_path = root / "evaluation" / "material_gap_query_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        cursor = max(0, int(state.get("cursor", 0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        cursor = 0
    with queue_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    cursor %= len(rows)
    selected = [rows[(cursor + offset) % len(rows)] for offset in range(min(batch_size, len(rows)))]
    next_cursor = (cursor + len(selected)) % len(rows)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"cursor": next_cursor, "batch_size": batch_size, "queue_rows": len(rows), "updated_at": utc_now()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return [str(row.get("recommended_query") or "").strip() for row in selected if str(row.get("recommended_query") or "").strip()]


class WorkerLock:
    """Prevent two local workers from writing the same SQLite index at once."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        import msvcrt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"已有另一个本地 worker 正在运行：{self.path}") from exc
        self.handle.seek(0)
        self.handle.write(utc_now())
        self.handle.truncate()
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        import msvcrt

        if self.handle is not None:
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.handle.close()
                self.handle = None


def run_worker(
    config: Mapping[str, Any],
    *,
    queries: list[str],
    dataset_key: str | None = None,
    limit: int | None = None,
    interval_minutes: float = 360,
    max_cycles: int | None = None,
    seed_manifest: Path | None = None,
    retry_limit: int = 20,
    once: bool = False,
) -> int:
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    storage = config["storage"]
    root = Path(str(storage["root"]))
    evaluation = root / "evaluation"
    lock_path = evaluation / "autonomous_worker.lock"
    history_path = evaluation / "autonomous_worker_history.jsonl"
    status_path = evaluation / "autonomous_worker_status.json"
    cycle_limit = 1 if once else (max_cycles or 2**31 - 1)
    with WorkerLock(lock_path):
        for cycle_number in range(cycle_limit):
            started_at = utc_now()
            _write_json(status_path, {
                "state": "running",
                "phase": "maintenance",
                "cycle_number": cycle_number + 1,
                "started_at": started_at,
                "updated_at": utc_now(),
            })
            health = run_maintenance(config, perform_cleanup=False)
            if health["c_drive"]["state"] != "ok":
                # RAGFlow/MinerU may allocate temporary data outside the
                # configured G: workspace (notably inside Docker Desktop).
                # Stop before discovery or parsing while C: is in warning or
                # critical state; do not consume the user's remaining reserve.
                report = {"cycle": cycle_number + 1, "started_at": started_at, "finished_at": utc_now(), "blocked": "c_drive_low_space", "health": health}
                _write_json(status_path, report)
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, ensure_ascii=False) + "\n")
                return 2

            seed_results: list[dict[str, Any]] = []
            if seed_manifest and seed_manifest.is_file() and cycle_number == 0:
                _write_json(status_path, {
                    "state": "running",
                    "phase": "seed_ingest",
                    "cycle_number": cycle_number + 1,
                    "started_at": started_at,
                    "updated_at": utc_now(),
                    "seed_manifest": str(seed_manifest),
                    "health": health,
                })
                seed_results = run_seed_ingest(config, seed_manifest)
            _write_json(status_path, {
                "state": "running",
                "phase": "zotero_discovery_and_ingest",
                "cycle_number": cycle_number + 1,
                "started_at": started_at,
                "updated_at": utc_now(),
                "seed_count": len(seed_results),
                "health": health,
            })
            if dataset_key:
                cycle_report = run_cycle(config, queries, limit=limit, dataset_key=dataset_key)
            else:
                material_queries = [query for query in queries if _is_material_query(query)]
                material_queries.extend(_material_gap_queries(config))
                material_queries = list(dict.fromkeys(material_queries))
                general_queries = [query for query in queries if query not in material_queries]
                general_report = run_cycle(config, general_queries, limit=limit, dataset_key="papers") if general_queries else {"zotero": [], "discovery": [], "health_after": health}
                # Material searches discover source papers.  They must be
                # stored in the papers corpus; only the extracted parameter
                # table and source index belong to the materials dataset.
                material_report = run_cycle(config, material_queries, limit=limit, dataset_key="papers", skip_zotero=True) if material_queries else {"zotero": [], "discovery": [], "health_after": general_report.get("health_after", health)}
                cycle_report = {
                    "health_before": general_report.get("health_before", health),
                    "zotero": general_report.get("zotero", []),
                    "discovery": general_report.get("discovery", []) + material_report.get("discovery", []),
                    "material_discovery": material_report.get("discovery", []),
                    "ragflow_runtime": material_report.get("ragflow_runtime") or general_report.get("ragflow_runtime"),
                    "health_after": material_report.get("health_after", general_report.get("health_after", health)),
                }
            # Rebuild the observation layer with the current extraction and
            # name-normalization rules before publishing the registry. This
            # keeps long-running workers self-correcting when the parser
            # improves, without re-running MinerU for already stored PDFs.
            try:
                from sync_library import PaperIndex
                material_catalog_csv = str(config.get("material_registry", {}).get("catalog_csv", "")).strip()
                paper_index = PaperIndex(
                    Path(str(storage["index_db"])),
                    Path(str(storage["summary_csv"])),
                    Path(material_catalog_csv) if material_catalog_csv else None,
                )
                rebuilt_material_observations = paper_index.rebuild_material_observations()
                paper_index.export_csv()
                paper_index.close()
                cycle_report["material_observation_rebuild"] = {
                    "status": "ok",
                    "rows": rebuilt_material_observations,
                }
            except Exception as exc:
                cycle_report["material_observation_rebuild"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                from material_catalog_sync import sync_material_registry_to_ragflow
                material_registry_sync = sync_material_registry_to_ragflow(config)
            except Exception as exc:
                material_registry_sync = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            cycle_report["material_registry_sync"] = material_registry_sync
            _write_json(status_path, {
                "state": "running",
                "phase": "retry_and_finalize",
                "cycle_number": cycle_number + 1,
                "started_at": started_at,
                "updated_at": utc_now(),
                "seed_count": len(seed_results),
                "zotero_count": len(cycle_report.get("zotero", [])),
                "discovery_count": len(cycle_report.get("discovery", [])),
                "health": health,
            })
            retry_results = retry_failed(config, limit=retry_limit)
            _write_json(status_path, {
                "state": "running",
                "phase": "adaptive_chunk_repair",
                "cycle_number": cycle_number + 1,
                "started_at": started_at,
                "updated_at": utc_now(),
                "seed_count": len(seed_results),
                "zotero_count": len(cycle_report.get("zotero", [])),
                "discovery_count": len(cycle_report.get("discovery", [])),
                "retry_count": len(retry_results),
                "health": health,
            })
            chunk_repair_results = run_adaptive_chunk_repair(config, limit=1)
            # Retry and adaptive chunk repair can add or revise material
            # observations. Publish one final registry after those mutations
            # so the serving RAGFlow document always matches SQLite/CSV.
            try:
                from sync_library import PaperIndex
                material_catalog_csv = str(config.get("material_registry", {}).get("catalog_csv", "")).strip()
                final_index = PaperIndex(
                    Path(str(storage["index_db"])),
                    Path(str(storage["summary_csv"])),
                    Path(material_catalog_csv) if material_catalog_csv else None,
                )
                final_material_rows = final_index.rebuild_material_observations()
                final_index.export_csv()
                final_index.close()
                from material_catalog_sync import sync_material_registry_to_ragflow
                final_material_sync = sync_material_registry_to_ragflow(config)
                cycle_report["final_material_registry"] = {
                    "observation_rows": final_material_rows,
                    "sync": final_material_sync,
                }
            except Exception as exc:
                cycle_report["final_material_registry"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                from material_optical_data import sync_optical_curve_table
                cycle_report["material_optical_curves"] = sync_optical_curve_table(config)
            except Exception as exc:
                cycle_report["material_optical_curves"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report = {
                "cycle_number": cycle_number + 1,
                "started_at": started_at,
                "finished_at": utc_now(),
                "seed": seed_results,
                "cycle_report": cycle_report,
                "retry": retry_results,
                "adaptive_chunk_repair": chunk_repair_results,
                "health": run_maintenance(config, perform_cleanup=False),
            }
            _write_json(status_path, report)
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
            print(json.dumps({
                "cycle": cycle_number + 1,
                "blocked": cycle_report.get("blocked"),
                "seed_count": len(seed_results),
                "zotero_count": len(cycle_report.get("zotero", [])),
                "discovery_count": len(cycle_report.get("discovery", [])),
                "retry_count": len(retry_results),
                "c_free_gb": report["health"]["c_drive"]["free_gb"],
            }, ensure_ascii=False), flush=True)
            if cycle_report.get("blocked") == "c_drive_low_space" or cycle_number + 1 >= cycle_limit:
                return 0
            next_run_at = (
                datetime.now(timezone.utc) + timedelta(minutes=float(interval_minutes))
            ).isoformat(timespec="seconds")
            _write_json(status_path, {
                **report,
                "state": "sleeping",
                "phase": "waiting_next_cycle",
                "updated_at": utc_now(),
                "next_run_at": next_run_at,
                "interval_minutes": float(interval_minutes),
            })
            time.sleep(interval_minutes * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Zotero/G-drive to MinerU to review to RAGFlow worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--dataset-key", choices=("fundamentals", "materials", "papers", "software_manuals", "cases"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--interval-minutes", type=float, default=360)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    queries = args.queries or list(config.get("discovery", {}).get("queries", []))
    if not queries:
        raise SystemExit("请使用 --query 指定发现词，或在 config.discovery.queries 中配置。")
    try:
        return run_worker(
            config,
            queries=queries,
            dataset_key=args.dataset_key,
            limit=args.limit,
            interval_minutes=args.interval_minutes,
            max_cycles=args.max_cycles,
            seed_manifest=args.seed_manifest,
            retry_limit=args.retry_limit,
            once=args.once,
        )
    except Exception as exc:
        # Never leave a misleading "running" status after an uncaught worker
        # failure. Keep the error structured and let the process exit nonzero
        # so an external supervisor can restart it.
        try:
            evaluation = Path(str(config["storage"]["root"])) / "evaluation"
            _write_json(evaluation / "autonomous_worker_status.json", {
                "state": "failed",
                "finished_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
