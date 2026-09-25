"""Read-only live RAGFlow adapter for the electromagnetic datasets."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .adapters import RAGFlowRetriever
from .query_translation import TRANSLATION_RULES


_TOKEN_RE = re.compile(r"[A-Za-z]{3,}|[0-9]{2,}|[\u4e00-\u9fff]{2,}")
_QUERY_ALIASES = dict(TRANSLATION_RULES)
_MATERIAL_QUERY = re.compile(
    r"材料参数|材料卡片|介电常数|磁导率|电导率|折射率|消光系数|光学常数|介电函数|损耗角|波长范围|"
    r"permittivity|permeability|conductivity|dielectric constant|loss tangent|refractive index|"
    r"optical constants?|extinction coefficient|dielectric function|wavelength range|\bn\s*[-/,]?\s*k\b",
    re.I,
)
_CODE_QUERY = re.compile(r"代码|程序|编程|脚本|源码|源代码|python|matlab|notebook|\bapi\b|\bcode\b|\bscript\b|implementation|\.py\b", re.I)
_MANUAL_QUERY = re.compile(r"手册|操作|安装|菜单|命令|文档|manual|documentation|\bapi\b|\bcommand\b", re.I)
_CASE_QUERY = re.compile(
    r"案例|算例|仿真模型|官方模型|超表面|频率选择表面|开口谐振环|周期单元|透射带通|薄导线|等效半径|"
    r"frequency selective surface|\bfss\b|complementary split ring resonator|\bcsrr\b|"
    r"periodic unit cell|transmission bandpass|thin wire|equivalent radius|case study|simulation case|\bdemo\b|\bexample\b",
    re.I,
)
_PAPER_QUERY = re.compile(r"论文|文献|期刊|doi|引用|综述|研究进展|\bpaper\b|\bjournal\b|\bliterature\b", re.I)


def _tokens(value: str) -> set[str]:
    return {item.lower() for item in _TOKEN_RE.findall(value or "")}


def _expand_query(query: str) -> str:
    aliases = [value for key, value in _QUERY_ALIASES.items() if key in query]
    return f"{query} {' '.join(aliases)}".strip()


class MultiDatasetRAGFlowClient:
    """Expose the configured RAGFlow datasets through the assistant seam.

    The assistant only calls retrieval and reads original PDFs for page fallback;
    it never uploads, edits, parses, or deletes RAGFlow documents.
    """

    def __init__(self, config_path: str | Path, index_path: str | Path):
        upstream = Path(config_path).resolve().parent
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        from sync_library import RAGFlowClient, load_config  # type: ignore

        self.config = load_config(Path(config_path))
        rag = self.config["ragflow"]
        self.base_url = str(rag["base_url"])
        self.dataset_ids = {str(k): str(v) for k, v in rag.get("dataset_ids", {}).items()}
        self.retrieval_timeout = float(rag.get("retrieval_timeout_seconds", 20))
        self.clients = {
            key: RAGFlowClient(self.base_url, dataset_id, str(rag["api_key_env"]), timeout=self.retrieval_timeout)
            for key, dataset_id in self.dataset_ids.items()
        }
        self.metadata: dict[str, dict[str, Any]] = {}
        self.name_to_metadata: dict[str, dict[str, Any]] = {}
        self._pages: dict[str, dict[int, str]] = {}
        self._optical_curve_path: Path | None = None
        self._optical_curve_cache: tuple[int, list[dict[str, Any]]] | None = None
        self.last_dataset_keys: list[str] = []
        self._load_metadata(Path(index_path))
        storage_root = Path(self.config.get("storage", {}).get("root", ""))
        self._load_ingest_manifests(storage_root)
        self._load_material_states(storage_root)

    def _load_metadata(self, index_path: Path) -> None:
        uri = f"file:{index_path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM papers WHERE status = 'published_ragflow' "
                "AND ragflow_document_id IS NOT NULL AND ragflow_document_id <> ''"
            ).fetchall()
        finally:
            connection.close()
        id_to_key = {value: key for key, value in self.dataset_ids.items()}
        for row in rows:
            item = dict(row)
            dataset_id = str(item.get("ragflow_dataset_id") or "")
            item["dataset_key"] = id_to_key.get(dataset_id, "unknown")
            item["dataset_id"] = dataset_id
            item["source_id"] = str(item.get("ragflow_document_id") or item.get("doc_id") or "")
            item["source_class"] = self._source_class(item)
            item["source_quality"] = self._source_quality(item)
            self.metadata[item["source_id"]] = item
            pdf_document_id = str(item.get("ragflow_pdf_document_id") or "").strip()
            if pdf_document_id and pdf_document_id != item["source_id"]:
                # A row can serve Markdown for retrieval while retaining a
                # separately parsed original PDF for page-level evidence.
                # Map both RAGFlow document IDs back to the same local source.
                pdf_item = dict(item)
                pdf_item["ragflow_document_id"] = pdf_document_id
                self.metadata[pdf_document_id] = pdf_item
            for name in (item.get("title"), item.get("filename"), item.get("ragflow_document_id")):
                if name:
                    self.name_to_metadata[str(name).lower()] = item

    def _register_manifest_row(self, dataset_key: str, row: dict[str, Any]) -> None:
        document_id = str(row.get("ragflow_document_id") or "").strip()
        if not document_id:
            return
        dataset_id = self.dataset_ids.get(dataset_key, "")
        document_name = str(row.get("document_name") or row.get("file_name") or "")
        source_url = str(row.get("source_url") or row.get("repo_url") or "")
        official_case = dataset_key == "cases" and "comsol.com" in source_url.casefold()
        pinned_code = dataset_key == "code_examples" and bool(row.get("commit") and row.get("source_sha256"))
        source_class = "official_case" if official_case else "version_pinned_code_repository" if pinned_code else "community_code_repository" if dataset_key == "code_examples" else "unknown"
        source_quality = 0.97 if official_case else 0.82 if pinned_code else 0.72 if dataset_key == "code_examples" else 0.5
        raw_path = str(row.get("file_path") or "") if dataset_key == "cases" else ""
        metadata = {
            "doc_id": document_id,
            "source_id": document_id,
            "dataset_key": dataset_key,
            "dataset_id": dataset_id,
            "title": str(row.get("title") or row.get("repo_name") or document_name or document_id),
            "filename": document_name,
            "source_path": source_url or str(row.get("source_path") or ""),
            "raw_path": raw_path,
            "ragflow_document_id": document_id,
            "source_url": source_url,
            "repo_url": str(row.get("repo_url") or ""),
            "repo_name": str(row.get("repo_name") or ""),
            "commit": str(row.get("commit") or ""),
            "license": str(row.get("license_note") or row.get("license") or ""),
            "application_id": str(row.get("application_id") or ""),
            "topic": str(row.get("topic") or ""),
            "page_count": row.get("page_count"),
            "parse_status": str(row.get("parse_status") or ""),
            "source_class": source_class,
            "source_quality": source_quality,
            "admission_score": source_quality,
            "parse_quality": 0.9 if str(row.get("parse_status") or "").upper() == "DONE" else 0.65,
            "provenance_verified": bool(source_url and (official_case or pinned_code)),
            "source_sha256": str(row.get("source_sha256") or row.get("sha256") or ""),
            "quality_note": str(row.get("quality_note") or ""),
        }
        self.metadata[document_id] = metadata
        for name in (document_name, row.get("file_name"), row.get("title"), document_id):
            if name:
                self.name_to_metadata[str(name).lower()] = metadata

    def _load_ingest_manifests(self, storage_root: Path) -> None:
        """Load code/PDF sources tracked outside the Zotero paper index."""
        manifests = {
            "code_examples": storage_root / "code_examples" / "audit" / "ingest_manifest.json",
            "cases": storage_root / "comsol_cases" / "audit" / "comsol_case_pdf_manifest.json",
        }
        for dataset_key, path in manifests.items():
            if dataset_key not in self.dataset_ids or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows = payload.get("files", []) if isinstance(payload, dict) else []
            for row in rows:
                if isinstance(row, dict):
                    self._register_manifest_row(dataset_key, row)

    def _load_material_states(self, storage_root: Path) -> None:
        """Index successful material-table uploads so retrieval isn't discarded.

        These generated material references are synchronized by dedicated
        workflows rather than stored as rows in the Zotero ``papers`` table.
        The assistant is read-only and trusts only the successful serving-state
        files whose dataset ID matches the configured materials dataset.
        """
        dataset_id = self.dataset_ids.get("materials", "")
        if not dataset_id:
            return
        state_specs = (
            (
                "material_registry_ragflow_state.json",
                "已核验材料参数登记表",
                "verified_material_catalog",
                0.90,
                "",
                Path("material_registry_ragflow.md"),
            ),
            (
                "material_optical_ragflow_state.json",
                "材料折射率与消光系数 n-k 光谱曲线表",
                "verified_optical_reference_data",
                0.95,
                "https://github.com/polyanskiy/refractiveindex.info-database",
                Path("materials") / "optical_constant_reference_curves.md",
            ),
        )
        evaluation_root = storage_root / "evaluation"
        for filename, title, source_class, quality, source_url, default_source_path in state_specs:
            path = evaluation_root / filename
            if not path.is_file():
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            document_id = str(state.get("document_id") or "").strip()
            if (
                state.get("status") != "uploaded_and_parsed"
                or not document_id
                or str(state.get("dataset_id") or "") != dataset_id
            ):
                continue
            source_path = str(
                state.get("source_path")
                or state.get("path")
                or str(storage_root / default_source_path)
            )
            if source_class == "verified_optical_reference_data":
                self._optical_curve_path = Path(source_path)
            document_name = str(
                state.get("display_name")
                or (Path(source_path).name if source_path else filename)
            )
            metadata = {
                "doc_id": document_id,
                "source_id": document_id,
                "dataset_key": "materials",
                "dataset_id": dataset_id,
                "title": title,
                "filename": document_name,
                "source_path": source_path,
                "raw_path": "",
                "ragflow_document_id": document_id,
                "source_url": source_url,
                "source_class": source_class,
                "source_quality": quality,
                "admission_score": quality,
                "parse_quality": 1.0,
                "provenance_verified": True,
                "parse_status": "DONE",
                "registry_rows": state.get("rows"),
                "curve_count": state.get("curve_count"),
                "point_count": state.get("point_count"),
            }
            self.metadata[document_id] = metadata
            self.name_to_metadata[document_name.lower()] = metadata
            self.name_to_metadata[title.lower()] = metadata

    @staticmethod
    def _source_class(item: dict[str, Any]) -> str:
        existing = str(item.get("source_class") or "").strip()
        if existing:
            return existing
        provider = str(item.get("source_provider") or "").casefold()
        if "comsol" in provider or "ansys" in provider:
            return "official_manual"
        if "arxiv" in provider:
            return "preprint"
        return "unknown"

    @classmethod
    def _source_quality(cls, item: dict[str, Any]) -> float:
        for field in ("source_quality", "selection_score"):
            try:
                score = float(item.get(field) or 0)
            except (TypeError, ValueError):
                score = 0.0
            if score > 0:
                return min(1.0, max(0.0, score))
        source_class = cls._source_class(item)
        return 0.97 if source_class in {"official_case", "official_manual"} else 0.7 if source_class == "preprint" else 0.5

    def route_dataset_keys(self, query: str) -> list[str]:
        """Route by evidence type; material-parameter lookups stay isolated."""
        if _MATERIAL_QUERY.search(query):
            wanted = ["materials"]
        elif _CODE_QUERY.search(query):
            wanted = ["code_examples"]
            if _MANUAL_QUERY.search(query):
                wanted.append("software_manuals")
            if re.search(r"案例|算例|simulation case|case study", query, re.I):
                wanted.append("cases")
        elif _CASE_QUERY.search(query):
            wanted = ["cases"]
            # Keep an exact case question grounded in the case PDF. Search
            # manuals alongside it only when the user explicitly asks for
            # documentation, commands, installation, or UI/API instructions.
            if _MANUAL_QUERY.search(query):
                wanted.append("software_manuals")
        elif _MANUAL_QUERY.search(query):
            wanted = ["software_manuals"]
        elif _PAPER_QUERY.search(query):
            wanted = ["papers", "fundamentals"]
        elif re.search(r"FDTD|CFL|Yee|稳定|网格|边界|色散|Maxwell|electromagnetic", query, re.I):
            wanted = ["fundamentals", "papers"]
        else:
            wanted = ["fundamentals", "papers", "software_manuals", "cases"]
        return [key for key in wanted if key in self.clients]

    def _metadata_for_hit(self, hit: dict[str, Any], dataset_key: str) -> dict[str, Any] | None:
        candidate = str(
            hit.get("doc_id")
            or hit.get("document_id")
            or hit.get("documentId")
            or hit.get("docnm_kwd")
            or hit.get("document_keyword")
            or ""
        )
        item = self.metadata.get(candidate)
        if item is None:
            item = self.name_to_metadata.get(candidate.lower())
        return item

    def _optical_curve_rows(self) -> list[dict[str, Any]]:
        path = getattr(self, "_optical_curve_path", None)
        if not path or not path.is_file():
            return []
        try:
            modified = path.stat().st_mtime_ns
            cached = getattr(self, "_optical_curve_cache", None)
            if cached and cached[0] == modified:
                return cached[1]
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        section_title = ""
        source_index = ""
        source_url = ""
        sample_condition = ""
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            if line.startswith("## "):
                section_title = line[3:].split(" — ", 1)[0].strip()
                source_index = ""
                source_url = ""
                sample_condition = ""
                continue
            if line.startswith("- 来源索引"):
                match = re.search(r"`([^`]+)`", line)
                source_index = match.group(1).strip() if match else ""
                continue
            if line.startswith("- 来源链接"):
                match = re.search(r"https?://\S+", line)
                source_url = match.group(0).rstrip("。；;，,") if match else ""
                continue
            if line.startswith("- 样品/测量条件"):
                sample_condition = line.split("：", 1)[-1].strip().rstrip("。")
                continue
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 4:
                continue
            try:
                wavelength, refractive_index, extinction = (float(value) for value in cells[1:])
            except ValueError:
                continue
            if not source_index:
                source_match = re.search(r"\s—\s*((?:main|other)/.+)$", cells[0])
                source_index = source_match.group(1).strip() if source_match else ""
            if not source_index:
                continue
            rows.append({
                "material_title": section_title or cells[0].split(" — ", 1)[0].strip(),
                "source_index": source_index,
                "source_url": source_url,
                "sample_condition": sample_condition,
                "wavelength_um": wavelength,
                "n": refractive_index,
                "k": extinction,
                "line": line_number,
            })
        self._optical_curve_cache = (modified, rows)
        return rows

    @staticmethod
    def _query_mentions_material(query: str, row: dict[str, Any]) -> bool:
        source_parts = str(row.get("source_index") or "").split("/")
        symbol = source_parts[1] if len(source_parts) > 1 else ""
        normalized_query = query.translate(str.maketrans({"₂": "2", "₃": "3"}))
        if symbol and re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", normalized_query, re.I):
            return True
        material_name = str(row.get("material_title") or "")
        match = re.search(r"[（(]([^）)]+)[）)]", material_name)
        formula = match.group(1).split("/")[0].strip() if match else ""
        if formula and re.search(rf"(?<![A-Za-z0-9]){re.escape(formula)}(?![A-Za-z0-9])", normalized_query, re.I):
            return True
        aliases = {
            "Au": ("金", "gold"), "Ag": ("银", "silver"), "Cu": ("铜", "copper"),
            "Al": ("铝", "aluminum", "aluminium"), "Si": ("硅", "silicon"),
            "SiO2": ("二氧化硅", "石英", "silica", "silicon dioxide"),
            "TiO2": ("二氧化钛", "氧化钛", "titanium dioxide"),
            "GaN": ("氮化镓", "gallium nitride"), "Ge": ("锗", "germanium"),
            "ZnO": ("氧化锌", "zinc oxide"), "Si3N4": ("氮化硅", "silicon nitride"),
            "MgF2": ("氟化镁", "magnesium fluoride"), "ITO": ("氧化铟锡", "indium tin oxide"),
        }
        for alias in aliases.get(symbol, ()):
            if alias.isascii() and re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", normalized_query, re.I):
                return True
            if not alias.isascii():
                if len(alias) == 1:
                    if re.search(rf"{re.escape(alias)}(?=在|的|于|材料|参数|\s|[，,。?？])", normalized_query):
                        return True
                elif alias in normalized_query:
                    return True
        return False

    def _local_optical_lookup(self, query: str) -> list[dict[str, Any]]:
        wavelength_match = re.search(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*(nm|nanometers?|纳米|μm|µm|um|micrometers?|microns?|微米)(?![A-Za-z])",
            query,
            re.I,
        )
        if not wavelength_match:
            return []
        value = float(wavelength_match.group(1))
        unit = wavelength_match.group(2).lower()
        target_um = value / 1000 if unit in {"nm", "纳米"} or unit.startswith("nano") else value
        optical_metadata = next((
            item for item in self.metadata.values()
            if item.get("dataset_key") == "materials"
            and item.get("source_class") == "verified_optical_reference_data"
        ), None)
        if not optical_metadata:
            return []
        matched_rows = [row for row in self._optical_curve_rows() if self._query_mentions_material(query, row)]
        if not matched_rows:
            return []
        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in matched_rows:
            by_source.setdefault(str(row["source_index"]), []).append(row)
        target_nm = value if unit in {"nm", "纳米"} or unit.startswith("nano") else value * 1000
        line_path = Path(str(self._optical_curve_path or optical_metadata.get("source_path") or ""))
        hits: list[dict[str, Any]] = []
        for source_index, points in by_source.items():
            points.sort(key=lambda item: float(item["wavelength_um"]))
            exact = next((item for item in points if abs(float(item["wavelength_um"]) - target_um) <= 1e-7), None)
            if exact:
                evidence_points = [exact]
                explanation = "精确采样点；以下数值为源表原始记录。"
            else:
                lower = next((item for item in reversed(points) if float(item["wavelength_um"]) < target_um), None)
                upper = next((item for item in points if float(item["wavelength_um"]) > target_um), None)
                if not lower or not upper:
                    continue
                evidence_points = [lower, upper]
                explanation = "源表没有目标波长的精确采样点；仅返回相邻实测/表列点，不把插值值冒充原始数据。"
            material_title = str(evidence_points[0]["material_title"])
            source_name = source_index.rsplit("/", 1)[-1]
            point_lines = [
                f"{float(point['wavelength_um']):.6g} μm: n={point['n']:.8g}, k={point['k']:.8g} (源表第 {point['line']} 行)"
                for point in evidence_points
            ]
            doi = str(evidence_points[0].get("source_url") or "")
            sample = str(evidence_points[0].get("sample_condition") or "")
            content = (
                f"材料：{material_title}；来源记录：{source_index}；目标：{target_nm:g} nm = {target_um:.4f} μm。"
                f"表列为折射率 n 和消光系数 k。{explanation}相邻数据：" + "；".join(point_lines)
                + (f"；样品/测量条件：{sample}" if sample else "")
                + (f"；原始文献链接：{doi}" if doi else "")
            )
            row_start = evidence_points[0]["line"]
            row_end = evidence_points[-1]["line"]
            slug = re.sub(r"[^a-z0-9]+", "-", source_index.casefold()).strip("-")
            hits.append({
                "id": f"local-optical-{slug}-{target_um:.4f}",
                "document_id": str(optical_metadata["ragflow_document_id"]),
                "document_keyword": str(optical_metadata.get("filename") or line_path.name),
                "title": f"{material_title} — {source_name}",
                "content": content,
                "similarity": 1.0,
                "source_quality": 0.95,
                "dataset_key": "materials",
                "dataset_id": str(optical_metadata.get("dataset_id") or ""),
                "source_path": str(line_path),
                "source_url": doi or str(optical_metadata.get("source_url") or ""),
                "source_index": source_index,
                "source_class": "verified_optical_reference_data",
                "provenance_verified": bool(doi),
                "evidence_ref": f"{line_path.name}:L{row_start}-L{row_end}",
                "chunk_id": f"local-{slug}-{target_um:.4f}",
                "document_type": "document",
            })
        return hits

    def _search_one(self, dataset_key: str, query: str, top_k: int) -> list[dict[str, Any]]:
        client = self.clients[dataset_key]
        return client.retrieve(_expand_query(query), top_k=top_k, similarity_threshold=0.1)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        dataset_keys = self.route_dataset_keys(query)
        self.last_dataset_keys = list(dataset_keys)
        if dataset_keys == ["materials"]:
            local_optical_hits = self._local_optical_lookup(query)
            if local_optical_hits:
                return local_optical_hits[: max(1, top_k)]
        with ThreadPoolExecutor(max_workers=max(1, len(dataset_keys))) as pool:
            futures = {pool.submit(self._search_one, key, query, top_k): key for key in dataset_keys}
            for future in as_completed(futures):
                dataset_key = futures[future]
                try:
                    hits = future.result()
                except Exception:
                    continue
                for hit in hits[:top_k]:
                    hit = dict(hit)
                    meta = self._metadata_for_hit(hit, dataset_key)
                    # RAGFlow may still contain documents from an older run.
                    # The answer agent serves only the current audited index.
                    if meta is None:
                        continue
                    source_id = str(meta.get("source_id") or hit.get("doc_id") or hit.get("id") or f"ragflow-{dataset_key}")
                    content = str(hit.get("content") or hit.get("text") or "")
                    document_name = str(hit.get("document_keyword") or hit.get("docnm_kwd") or meta.get("filename") or "")
                    chunk_id = str(hit.get("id") or hit.get("chunk_id") or "chunk")
                    chunk_locator_id = chunk_id.removeprefix("chunk-")
                    is_pdf = str(meta.get("raw_path") or "").lower().endswith(".pdf") or document_name.lower().endswith(".pdf")
                    reported_page = hit.get("page") or hit.get("page_num") or self._position_page(hit.get("positions"))
                    try:
                        reported_page_number = int(reported_page) if reported_page is not None else None
                    except (TypeError, ValueError):
                        reported_page_number = None
                    page_number, page_locator = self._resolve_pdf_page(meta, content, reported_page_number) if is_pdf else (None, "not_pdf")
                    source_quality = self._source_quality(meta)
                    source_url = str(hit.get("source_url") or meta.get("source_url") or meta.get("repo_url") or "")
                    relevance = min(1.0, max(0.0, float(hit.get("similarity") or hit.get("score") or 0.0)))
                    if source_url and dataset_key == "code_examples":
                        evidence_ref = f"{source_url}#ragflow-chunk-{chunk_id}"
                    elif page_number is not None:
                        evidence_ref = f"{document_name or meta.get('title')}:p{page_number}:chunk-{chunk_locator_id}"
                    else:
                        evidence_ref = f"{document_name or source_id}#chunk-{chunk_locator_id}"
                    evidence_ref = str(hit.get("evidence_ref") or evidence_ref)
                    rows.append({
                        "doc_id": source_id,
                        "dataset_key": str(meta.get("dataset_key") or dataset_key),
                        "dataset_id": str(meta.get("dataset_id") or self.dataset_ids.get(dataset_key, "")),
                        "title": str(hit.get("title") or meta.get("title") or hit.get("docnm_kwd") or source_id),
                        "source_path": str(hit.get("source_path") or meta.get("source_path") or meta.get("raw_path") or ""),
                        "raw_path": str(meta.get("raw_path") or ""),
                        "ragflow_document_id": str(meta.get("ragflow_document_id") or source_id),
                        "page": page_number,
                        "page_locator": page_locator,
                        "chunk_id": chunk_id,
                        "evidence_ref": evidence_ref,
                        "retrieval_score": relevance,
                        "source_quality": source_quality,
                        "answer_priority": round(0.72 * relevance + 0.28 * source_quality, 4),
                        "source_class": str(meta.get("source_class") or self._source_class(meta)),
                        "source_url": source_url,
                        "repo_name": str(meta.get("repo_name") or ""),
                        "commit": str(meta.get("commit") or ""),
                        "license": str(meta.get("license") or ""),
                        "provenance_verified": bool(meta.get("provenance_verified", bool(source_url))),
                        "admission_score": source_quality,
                        "parse_quality": float(meta.get("parse_quality") or (0.9 if str(meta.get("mineru_status") or "").upper() == "DONE" else 0.65)),
                        "document_type": "pdf" if is_pdf else "code" if dataset_key == "code_examples" else "document",
                        "content": content,
                    })
        rows.sort(key=lambda item: (float(item.get("answer_priority") or 0.0), float(item.get("retrieval_score") or 0.0)), reverse=True)
        return rows[: max(1, top_k)]

    @staticmethod
    def _position_page(positions: Any) -> int | None:
        pages: list[int] = []
        if isinstance(positions, (list, tuple)):
            for position in positions:
                candidate = position[0] if isinstance(position, (list, tuple)) and position else position
                try:
                    page = int(candidate)
                except (TypeError, ValueError):
                    continue
                if page > 0:
                    pages.append(page)
        return min(pages) if pages else None

    def _resolve_pdf_page(
        self, metadata: dict[str, Any], content: str, reported_page: int | None
    ) -> tuple[int | None, str]:
        """Prefer matching a chunk to the local PDF over batch-local positions."""
        raw_path = str(metadata.get("raw_path") or "")
        if raw_path and Path(raw_path).is_file():
            matched_page = self._locate_page(raw_path, content)
            if matched_page is not None:
                return matched_page, "local_pdf_text_match"
            # A RAGFlow position can be relative to a MinerU batch. If local
            # text cannot corroborate it, suppress the page number rather than
            # sending the assistant to a confidently wrong page.
            return None, "local_pdf_match_failed"
        if reported_page is not None and reported_page > 0:
            return reported_page, "ragflow_position_unverified"
        return None, "page_unavailable"

    def _page_texts(self, raw_path: str) -> dict[int, str]:
        if raw_path in self._pages:
            return self._pages[raw_path]
        from pypdf import PdfReader
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            reader = PdfReader(raw_path)
            pages = {index: (page.extract_text() or "") for index, page in enumerate(reader.pages, 1)}
        finally:
            logging.disable(previous_disable)
        self._pages[raw_path] = pages
        return pages

    def _locate_page(self, raw_path: str, content: str) -> int | None:
        terms = _tokens(content)
        if not terms or not raw_path:
            return None
        page_scores: list[tuple[int, int]] = []
        for page, text in self._page_texts(raw_path).items():
            score = len(terms & _tokens(text))
            page_scores.append((score, page))
        if not page_scores:
            return None
        page_scores.sort(reverse=True)
        best_score, best_page = page_scores[0]
        second_score = page_scores[1][0] if len(page_scores) > 1 else 0
        # RAGFlow's PDF parser can restart page positions for every batch.
        # A few shared words are not enough to replace those positions with a
        # local PDF page number: require meaningful token coverage and a clear
        # winner, otherwise suppress the locator instead of inventing one.
        if best_score < 4 or best_score - second_score < 2 or best_score / len(terms) < 0.15:
            return None
        return best_page

    def get_page(self, source_id: str, page: int | None, **_: Any) -> list[dict[str, Any]]:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            return []
        if page_number <= 0:
            return []
        meta = self.metadata.get(str(source_id), {})
        raw_path = str(meta.get("raw_path") or "")
        if not raw_path or not Path(raw_path).is_file():
            return []
        text = self._page_texts(raw_path).get(page_number, "")
        source_quality = self._source_quality(meta)
        return [{
            "doc_id": str(source_id),
            "dataset_key": str(meta.get("dataset_key") or "unknown"),
            "dataset_id": str(meta.get("dataset_id") or ""),
            "title": str(meta.get("title") or source_id),
            "source_path": str(meta.get("source_path") or raw_path),
            "raw_path": raw_path,
            "ragflow_document_id": str(meta.get("ragflow_document_id") or source_id),
            "page": page_number,
            "chunk_id": f"page-{page_number}-full",
            "evidence_ref": f"{source_id}:p{page_number}:full",
            "retrieval_score": 1.0,
            "source_quality": source_quality,
            "answer_priority": source_quality,
            "source_class": str(meta.get("source_class") or self._source_class(meta)),
            "source_url": str(meta.get("source_url") or ""),
            "provenance_verified": bool(meta.get("provenance_verified", bool(meta.get("source_url")))),
            "admission_score": source_quality,
            "parse_quality": float(meta.get("parse_quality") or 0.9),
            "document_type": "pdf",
            "content": text,
        }]


def build_live_retriever(config_path: str | Path, index_path: str | Path) -> RAGFlowRetriever:
    return RAGFlowRetriever(MultiDatasetRAGFlowClient(config_path, index_path))
