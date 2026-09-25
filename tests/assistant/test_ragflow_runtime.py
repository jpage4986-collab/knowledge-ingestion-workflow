import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fdtd_assistant import run_assistant
from fdtd_assistant.adapters import RAGFlowRetriever
from fdtd_assistant.models import Evidence
from fdtd_assistant.query_translation import coverage_alias_groups
from fdtd_assistant.ragflow_runtime import MultiDatasetRAGFlowClient
from fdtd_assistant.query_translation import translate_for_retrieval
from fdtd_assistant.graph import _retrieve


class FakeDatasetClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve(self, query, top_k=5, similarity_threshold=0.1):
        self.calls.append(query)
        return self.rows[:top_k]


def runtime_with_manifests(root):
    runtime = object.__new__(MultiDatasetRAGFlowClient)
    runtime.dataset_ids = {
        "materials": "materials-id",
        "papers": "papers-id",
        "fundamentals": "fundamentals-id",
        "software_manuals": "manuals-id",
        "cases": "cases-id",
        "code_examples": "code-id",
    }
    runtime.clients = {key: FakeDatasetClient([]) for key in runtime.dataset_ids}
    runtime.metadata = {}
    runtime.name_to_metadata = {}
    runtime._pages = {}
    runtime.last_dataset_keys = []
    runtime._load_ingest_manifests(root)
    runtime._load_material_states(root)
    return runtime


class RAGFlowRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        code_dir = self.root / "code_examples" / "audit"
        cases_dir = self.root / "comsol_cases" / "audit"
        evaluation_dir = self.root / "evaluation"
        code_dir.mkdir(parents=True)
        cases_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)
        (code_dir / "ingest_manifest.json").write_text(json.dumps({
            "files": [{
                "document_name": "meep__pml.py.md",
                "ragflow_document_id": "code-doc-1",
                "repo_name": "Meep",
                "repo_url": "https://github.com/NanoComp/meep",
                "source_url": "https://github.com/NanoComp/meep/blob/abc123/python/pml.py",
                "commit": "abc123",
                "source_sha256": "hash-1",
                "license": "BSD-2-Clause",
                "parse_status": "DONE",
            }]
        }), encoding="utf-8")
        (cases_dir / "comsol_case_pdf_manifest.json").write_text(json.dumps({
            "files": [{
                "document_name": "comsol-rf-grating.pdf",
                "file_name": "COMSOL_RF_Grating.pdf",
                "ragflow_document_id": "case-doc-1",
                "title": "Hexagonal Grating (RF)",
                "source_url": "https://www.comsol.com/model/download/example.pdf",
                "file_path": "C:/fixtures/COMSOL_RF_Grating.pdf",
                "page_count": 24,
                "parse_status": "DONE",
            }]
        }), encoding="utf-8")
        (evaluation_dir / "material_registry_ragflow_state.json").write_text(json.dumps({
            "status": "uploaded_and_parsed",
            "dataset_id": "materials-id",
            "document_id": "material-doc-1",
            "source_path": str(self.root / "material_registry_ragflow.md"),
            "rows": 499,
        }), encoding="utf-8")
        (evaluation_dir / "material_optical_ragflow_state.json").write_text(json.dumps({
            "status": "pending_or_failed",
            "dataset_id": "materials-id",
            "document_id": "optical-doc-pending",
        }), encoding="utf-8")
        self.runtime = runtime_with_manifests(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_routes_code_case_and_material_questions_to_separate_datasets(self):
        self.assertEqual(self.runtime.route_dataset_keys("Python FDTD source code example"), ["code_examples"])
        self.assertEqual(self.runtime.route_dataset_keys("Meep Python FDTD code example"), ["code_examples"])
        self.assertEqual(
            self.runtime.route_dataset_keys("Meep Python API reference manual and code example"),
            ["code_examples", "software_manuals"],
        )
        self.assertEqual(self.runtime.route_dataset_keys("COMSOL hexagonal grating simulation case"), ["cases"])
        self.assertEqual(self.runtime.route_dataset_keys("SiO2 dielectric constant and loss tangent"), ["materials"])
        self.assertEqual(self.runtime.route_dataset_keys("gold optical constants n k wavelength"), ["materials"])
        self.assertEqual(self.runtime.route_dataset_keys("refractive index and extinction coefficient vs wavelength"), ["materials"])

    def test_optical_material_queries_gain_english_retrieval_terms(self):
        translated, changed = translate_for_retrieval("SiO2 折射率和消光系数波长范围")
        self.assertTrue(changed)
        for term in ("refractive index n", "extinction coefficient k", "wavelength range"):
            self.assertIn(term, translated)
        self.assertEqual(self.runtime.route_dataset_keys(translated), ["materials"])

    def test_yee_ez_location_queries_gain_specific_aliases_and_translation(self):
        translated, changed = translate_for_retrieval("Yee 网格中 E_z 节点位于哪里？")
        self.assertTrue(changed)
        for term in ("E_z", "Yee grid", "cell center"):
            self.assertIn(term, translated)
        groups = coverage_alias_groups("Where is E_z located on a Yee grid?")
        self.assertEqual(len(groups), 3)
        self.assertTrue(any("Yee's grid" in group for group in groups))
        self.assertTrue(any("E_z" in group for group in groups))
        self.assertTrue(any("cell centers" in group for group in groups))
        plane_wave_groups = coverage_alias_groups("plane-wave 源")
        self.assertTrue(any("planewave" in group for group in plane_wave_groups))
        flux_query, flux_changed = translate_for_retrieval("flux monitor")
        self.assertTrue(flux_changed)
        self.assertIn("add_flux", flux_query)

    def test_graph_discards_chunks_without_specific_yee_location_support(self):
        class MixedRetriever:
            def __init__(self):
                self.rows = [
                    Evidence(
                        "equation-noise", "paper-noise", "Maxwell field equations",
                        "The field E_z appears in Maxwell equations and electromagnetic wave solutions.",
                        "pdf", 2, 0.98, 0.95, "paper-noise:p2",
                    ),
                    Evidence(
                        "yee-location", "paper-yee", "Yee FDTD grid field-node placement",
                        "Yee's FDTD algorithm uses a staggered grid. In the 2D TMz unit cell, "
                        "$E _ { z }$ is stored at cell centers.",
                        "pdf", 3, 0.62, 0.82, "paper-yee:p3",
                    ),
                    Evidence(
                        "unrelated-pml", "paper-pml", "PML solver benchmark",
                        "The PML absorbs outgoing waves in an FDTD simulation.",
                        "pdf", 4, 0.97, 0.9, "paper-pml:p4",
                    ),
                ]

            def search(self, _query, top_k=5):
                return self.rows[:top_k]

            def get_page(self, *_args, **_kwargs):
                return []

        result = run_assistant(
            "Where is E_z located on a Yee grid?",
            retriever=MixedRetriever(), include_full_page=False,
        )
        self.assertEqual([item["evidence_id"] for item in result["selected_evidence"]], ["yee-location"])
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["citations"], ["paper-yee:p3"])
        self.assertEqual(result["evaluation"]["precision"], 1.0)
        self.assertEqual(result["evaluation"]["evidence_coverage"], 1.0)

    def test_graph_runs_two_query_variants_concurrently_with_a_small_bound(self):
        class ConcurrentRetriever:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.calls = 0
                self.lock = threading.Lock()

            def search(self, query, top_k=5):
                with self.lock:
                    self.calls += 1
                    call_id = self.calls
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.03)
                with self.lock:
                    self.active -= 1
                return [Evidence(
                    f"chunk-{call_id}", "source", "Yee grid", query,
                    "document", None, 0.8, 0.8, f"source:{call_id}",
                )]

        state = {
            "question": "Yee grid",
            "subquestions": [{"part_id": "q1", "text": "Yee grid", "required_terms": [], "required_term_groups": []}],
        }
        retriever = ConcurrentRetriever()
        _retrieve(state, retriever)
        self.assertEqual(retriever.calls, 2)
        self.assertEqual(retriever.max_active, 2)

    def test_metamaterial_modeling_question_is_not_misrouted_to_parameter_catalog(self):
        self.assertEqual(
            self.runtime.route_dataset_keys("FDTD 超材料反射透射建模方法"),
            ["fundamentals", "papers"],
        )

    def test_periodic_metasurface_case_routes_to_case_library(self):
        query = "COMSOL 频率选择表面互补开口谐振环周期单元透射带通"
        self.assertEqual(self.runtime.route_dataset_keys(query), ["cases"])
        translated, changed = translate_for_retrieval(query)
        self.assertTrue(changed)
        for term in ("frequency selective surface", "complementary split ring resonator", "periodic unit cell", "transmission bandpass", "Floquet", "S-parameters"):
            self.assertIn(term, translated)

    def test_comsol_metasurface_shape_optimization_terms_are_translated_precisely(self):
        translated, changed = translate_for_retrieval(
            "COMSOL 超表面光束偏转器形状优化的设计变量、边界范围、迭代次数和目标衍射级次"
        )
        self.assertTrue(changed)
        for term in (
            "metasurface beam deflector", "shape optimization", "design variables",
            "moveMax", "lower bound", "upper bound", "scaleMax", "optimization iterations", "target diffraction order",
        ):
            self.assertIn(term, translated)

    def test_thin_wire_case_questions_translate_and_route_to_cases(self):
        query = "薄导线等效半径模型包含哪个数值系数？"
        translated, changed = translate_for_retrieval(query)
        self.assertTrue(changed)
        self.assertIn("thin wire equivalent radius model", translated)
        self.assertIn("numerical coefficient factor", translated)
        self.assertEqual(self.runtime.route_dataset_keys(query), ["cases"])

    def test_each_csrr_term_stays_in_case_and_manual_libraries_after_translation(self):
        for query in ("频率选择表面", "互补开口谐振环", "周期单元", "透射带通"):
            with self.subTest(query=query):
                translated, changed = translate_for_retrieval(query)
                self.assertTrue(changed)
                self.assertEqual(self.runtime.route_dataset_keys(translated), ["cases"])

    def test_explicit_case_documentation_question_also_searches_software_manuals(self):
        self.assertEqual(
            self.runtime.route_dataset_keys("COMSOL 超表面案例 manual menu command"),
            ["cases", "software_manuals"],
        )

    def test_code_and_case_terms_get_english_retrieval_aliases(self):
        translated, changed = translate_for_retrieval("FDTD 代码示例和吸收边界")
        self.assertTrue(changed)
        self.assertIn("code example", translated)
        self.assertIn("perfectly matched layer PML", translated)

        code_query, changed = translate_for_retrieval("包含几何、材料、plane-wave 源、通量监视器和版本来源")
        self.assertTrue(changed)
        for term in ("geometry setup", "material medium", "plane wave source", "flux monitor add_flux", "commit license"):
            self.assertIn(term, code_query)

    def test_fdtd_paper_terms_get_english_aliases_before_ragflow_search(self):
        cases = {
            "Yee 网格 E_z 节点位置": ("Yee grid", "staggered grid"),
            "粗网格和局部细网格的子网格方法": ("coarse mesh", "subgridding mesh"),
            "激励脉冲带宽和高斯脉冲": ("Gaussian pulse", "pulse bandwidth"),
            "移动结构和广义 FDTD": ("moving structures", "generalized FDTD"),
            "移动界面的边界条件": ("moving interface", "boundary conditions"),
            "可穿透介质与超材料": ("penetrable media", "metamaterials"),
            "论文区分运动介质的类型：移动物质和移动扰动": ("moving media", "moving-matter", "moving-perturbation"),
            "移动薄板、移动晶体和移动梯度": ("moving slab", "moving crystal", "moving gradient"),
        }
        for question, expected_terms in cases.items():
            with self.subTest(question=question):
                translated, changed = translate_for_retrieval(question)
                self.assertTrue(changed)
                for term in expected_terms:
                    self.assertIn(term, translated)

    def test_moving_structure_questions_are_fully_translated_before_search(self):
        questions = (
            "移动结构散射采用了什么数值方案？",
            "移动界面的边界条件如何处理？",
            "速度如何进入电磁场变换关系？",
            "该方案是否适用于可穿透介质？",
            "论文区分了哪两类运动介质？",
        )
        for question in questions:
            with self.subTest(question=question):
                translated, changed = translate_for_retrieval(question)
                self.assertTrue(changed)
                self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in translated))

    def test_independent_ingest_manifests_make_new_documents_searchable(self):
        self.assertIn("code-doc-1", self.runtime.metadata)
        self.assertIn("case-doc-1", self.runtime.metadata)
        self.assertIn("material-doc-1", self.runtime.metadata)
        self.assertNotIn("optical-doc-pending", self.runtime.metadata)
        self.assertEqual(self.runtime.metadata["code-doc-1"]["source_class"], "version_pinned_code_repository")
        self.assertTrue(self.runtime.metadata["code-doc-1"]["provenance_verified"])
        self.assertEqual(self.runtime.metadata["case-doc-1"]["source_class"], "official_case")
        self.assertEqual(self.runtime.metadata["case-doc-1"]["source_quality"], 0.97)
        self.assertEqual(self.runtime.metadata["material-doc-1"]["source_class"], "verified_material_catalog")
        self.assertEqual(self.runtime.metadata["material-doc-1"]["registry_rows"], 499)

    def test_material_hits_resolve_to_successfully_parsed_registry_state(self):
        self.runtime.clients["materials"] = FakeDatasetClient([{
            "id": "chunk-material-1",
            "document_id": "material-doc-1",
            "document_keyword": "material_registry_ragflow.md",
            "content": "Silicon dioxide source-indexed optical constants",
            "similarity": 0.8,
        }])
        hit = self.runtime.search("SiO2 dielectric constant", top_k=1)[0]
        self.assertEqual(self.runtime.last_dataset_keys, ["materials"])
        self.assertEqual(hit["doc_id"], "material-doc-1")
        self.assertEqual(hit["source_quality"], 0.9)
        self.assertEqual(hit["dataset_key"], "materials")

    def test_optical_lookup_filters_material_and_wavelength_and_preserves_sources(self):
        table_path = self.root / "optical_curve_reference.md"
        table_path.write_text(
            "## 金（Au） — Johnson\n"
            "- 来源索引：`main/Au/nk/Johnson`。\n"
            "- 样品/测量条件：Room temperature\n"
            "- 来源链接：https://doi.org/10.1103/PhysRevB.6.4370\n"
            "| 材料 / 来源记录 ID | 波长 λ（μm） | 折射率 n | 消光系数 k |\n"
            "| 金（Au） — main/Au/nk/Johnson | 0.5000 | 0.3000 | 2.0000 |\n"
            "| 金（Au） — main/Au/nk/Johnson | 0.6000 | 0.2000 | 3.0000 |\n"
            "\n## 金（Au） — Babar\n"
            "- 来源索引：`main/Au/nk/Babar`。\n"
            "- 样品/测量条件：Original record\n"
            "- 来源链接：https://doi.org/10.1364/AO.54.000477\n"
            "| 材料 / 来源记录 ID | 波长 λ（μm） | 折射率 n | 消光系数 k |\n"
            "| 金（Au） — main/Au/nk/Babar | 0.5000 | 0.3100 | 2.1000 |\n"
            "| 金（Au） — main/Au/nk/Babar | 0.6000 | 0.2200 | 3.1000 |\n"
            "\n## 银（Ag） — Johnson\n"
            "- 来源索引：`main/Ag/nk/Johnson`。\n"
            "- 来源链接：https://doi.org/10.1103/PhysRevB.6.4370\n"
            "| 材料 / 来源记录 ID | 波长 λ（μm） | 折射率 n | 消光系数 k |\n"
            "| 银（Ag） — main/Ag/nk/Johnson | 0.5000 | 1.0000 | 2.0000 |\n"
            "| 银（Ag） — main/Ag/nk/Johnson | 0.6000 | 1.1000 | 3.0000 |\n",
            encoding="utf-8",
        )
        state_path = self.root / "evaluation" / "material_optical_ragflow_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({
            "status": "uploaded_and_parsed", "dataset_id": "materials-id",
            "document_id": "optical-doc-1", "source_path": str(table_path),
            "display_name": "optical_curve_reference.md", "curve_count": 3, "point_count": 6,
        })
        state_path.write_text(json.dumps(state), encoding="utf-8")
        runtime = runtime_with_manifests(self.root)
        result = run_assistant(
            "金在 550 nm 的折射率 n 和消光系数 k 是多少？",
            retriever=RAGFlowRetriever(runtime), include_full_page=False,
        )
        self.assertEqual(runtime.last_dataset_keys, ["materials"])
        self.assertEqual(result["evaluation"]["evidence_coverage"], 1.0)
        self.assertEqual(result["evaluation"]["precision"], 1.0)
        self.assertEqual(result["evaluation"]["units_formula"], 1.0)
        self.assertEqual(len(result["selected_evidence"]), 2)
        contents = [item["content"] for item in result["selected_evidence"]]
        self.assertTrue(all("550 nm = 0.5500 μm" in item for item in contents))
        self.assertTrue(all("不把插值值冒充原始数据" in item for item in contents))
        self.assertTrue(any("main/Au/nk/Johnson" in item for item in contents))
        self.assertTrue(any("main/Au/nk/Babar" in item for item in contents))
        self.assertFalse(any("main/Ag/nk/Johnson" in item for item in contents))
        self.assertTrue(all("optical_curve_reference.md:L" in item for item in result["citations"]))

    def test_original_pdf_and_markdown_ids_map_to_one_local_source(self):
        index_path = self.root / "papers.sqlite"
        connection = sqlite3.connect(index_path)
        try:
            connection.execute(
                "CREATE TABLE papers (doc_id TEXT, title TEXT, status TEXT, "
                "ragflow_document_id TEXT, ragflow_pdf_document_id TEXT, ragflow_dataset_id TEXT)"
            )
            connection.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                ("paper-source", "FDTD paper", "published_ragflow", "markdown-id", "pdf-id", "papers-id"),
            )
            connection.commit()
        finally:
            connection.close()
        runtime = object.__new__(MultiDatasetRAGFlowClient)
        runtime.dataset_ids = {"papers": "papers-id"}
        runtime.metadata = {}
        runtime.name_to_metadata = {}
        runtime._load_metadata(index_path)
        self.assertIn("markdown-id", runtime.metadata)
        self.assertIn("pdf-id", runtime.metadata)
        self.assertEqual(runtime.metadata["pdf-id"]["source_id"], "markdown-id")
        self.assertEqual(runtime.metadata["pdf-id"]["ragflow_document_id"], "pdf-id")

    def test_local_pdf_text_match_overrides_batch_relative_ragflow_page(self):
        local_pdf = self.root / "fixture.pdf"
        local_pdf.write_bytes(b"fixture placeholder")
        self.runtime.metadata["case-doc-1"]["raw_path"] = str(local_pdf)
        self.runtime.clients["cases"] = FakeDatasetClient([{
            "id": "chunk-11",
            "document_id": "case-doc-1",
            "document_keyword": "comsol-rf-grating.pdf",
            "content": "Far-field boundary condition on the periodic structure",
            "similarity": 0.8,
            "positions": [[9, 1, 2, 3, 4]],
        }])
        with mock.patch.object(self.runtime, "_locate_page", return_value=237):
            hit = self.runtime.search("COMSOL hexagonal grating simulation case", top_k=1)[0]
        self.assertEqual(hit["page"], 237)
        self.assertEqual(hit["page_locator"], "local_pdf_text_match")
        self.assertIn(":p237:", hit["evidence_ref"])

    def test_local_pdf_page_match_requires_a_unique_high_confidence_hit(self):
        self.runtime._page_texts = mock.Mock(return_value={
            12: "Maxwell displacement current density in media",
            87: "An unrelated passage about mesh refinement and solver tolerance",
        })
        self.assertEqual(
            self.runtime._locate_page("fixture.pdf", "Maxwell displacement current density in media"),
            12,
        )

        self.runtime._page_texts = mock.Mock(return_value={
            12: "RF wave port boundary condition",
            87: "RF wave port boundary condition",
        })
        self.assertIsNone(
            self.runtime._locate_page("fixture.pdf", "RF wave port boundary condition")
        )

    def test_search_maps_source_quality_page_and_stable_citation(self):
        self.runtime.clients["code_examples"] = FakeDatasetClient([{
            "id": "chunk-7",
            "document_id": "code-doc-1",
            "document_keyword": "meep__pml.py.md",
            "content": "PML boundary setup source",
            "similarity": 0.75,
            "positions": [[1, 0, 0, 20, 40]],
        }])
        code_hit = self.runtime.search("Python source code example", top_k=3)[0]
        self.assertEqual(self.runtime.last_dataset_keys, ["code_examples"])
        self.assertEqual(code_hit["source_quality"], 0.82)
        self.assertEqual(code_hit["source_class"], "version_pinned_code_repository")
        self.assertIn("github.com/NanoComp/meep/blob/abc123", code_hit["evidence_ref"])
        self.assertEqual(code_hit["document_type"], "code")
        self.assertIsNone(code_hit["page"])

        self.runtime.clients["cases"] = FakeDatasetClient([{
            "id": "chunk-9",
            "document_id": "case-doc-1",
            "document_keyword": "comsol-rf-grating.pdf",
            "content": "Periodic boundary condition",
            "similarity": 0.7,
            "positions": [[5, 10, 20, 30, 40]],
        }])
        case_hit = self.runtime.search("COMSOL hexagonal grating simulation case", top_k=3)[0]
        self.assertEqual(self.runtime.last_dataset_keys, ["cases"])
        self.assertEqual(case_hit["page"], 5)
        self.assertEqual(case_hit["source_quality"], 0.97)
        self.assertIn(":p5:chunk-9", case_hit["evidence_ref"])

    def test_get_page_rejects_missing_or_nonpositive_page_numbers(self):
        self.runtime._page_texts = mock.Mock(side_effect=AssertionError("must not load a PDF for an invalid page"))
        self.assertEqual(self.runtime.get_page("unknown-source", None), [])
        self.assertEqual(self.runtime.get_page("unknown-source", 0), [])
        self.runtime._page_texts.assert_not_called()

    def test_answer_graph_prefers_quality_when_relevance_is_close(self):
        high_quality = Evidence("high", "source-high", "Pinned official example", "source examples for FDTD", "code", None, 0.97, 0.78, "high:chunk-1")
        lower_quality = Evidence("lower", "source-low", "Unverified example", "source examples for FDTD", "code", None, 0.45, 0.82, "low:chunk-1")

        class Retriever:
            def search(self, query, top_k=5):
                return [lower_quality, high_quality]

            def get_page(self, source_id, page):
                return []

        result = run_assistant("source examples for FDTD", retriever=Retriever(), max_iterations=0)
        self.assertEqual(result["selected_evidence"][0]["evidence_id"], "high")


if __name__ == "__main__":
    unittest.main()
