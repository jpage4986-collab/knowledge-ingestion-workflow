import unittest
from unittest.mock import patch

from fdtd_assistant import run_assistant
from fdtd_assistant.adapters import MockFixtureRetriever, RAGFlowRetriever
from fdtd_assistant.graph import _decompose, _page_fallback
from fdtd_assistant.models import AssistantState, Evidence, QuestionPart
from fdtd_assistant.evaluation import evaluate_evidence
from fdtd_assistant.knowledge import SimulationSpec
from fdtd_assistant.knowledge import FixtureKnowledgeProvider
from fdtd_assistant.workflow import run_workflow
from fdtd_assistant.prompting import answer_prompt
from fdtd_assistant.api import app, configure_web_searcher
from fdtd_assistant.client import FDTDClient


class FakeHTTPResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b'{"status":"ok"}'


class FakeWebSearcher:
    def search(self, query, top_k=3):
        return [Evidence("web-1", "web-fixture", "Injected web result", "天线远场需要单独验证边界与观测距离。", "web", None, 0.9, 0.9, "web-fixture:1")]


class FailingKnowledgeProvider(FixtureKnowledgeProvider):
    def search_claims(self, query, *, method=None, top_k=8):
        raise RuntimeError("injected provider failure")


class OfflineAssistantTests(unittest.TestCase):
    def test_unknown_question_is_refused_without_unrelated_citations(self):
        result = run_assistant("量子引力火星天气与FDTD无关的冷门问题")
        self.assertTrue(result["refusal"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["page_context"], [])
        self.assertIn("拒绝", result["answer"])

    def test_python_client_serializes_json_without_extra_dependencies(self):
        with patch("fdtd_assistant.client.urlopen", return_value=FakeHTTPResponse()) as opener:
            result = FDTDClient().health()
        self.assertEqual(result, {"status": "ok"})
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/health")

    def test_fdtd_question_has_answer_and_citations(self):
        result = run_assistant("请解释 FDTD 的 Yee 网格、CFL 稳定条件，并说明公式中的单位。")
        self.assertIn("CFL", result["answer"])
        self.assertTrue(result["citations"])
        self.assertFalse(result["evaluation"]["missing_parts"])
        self.assertIn("decompose", result["trace"])
        self.assertIn("assess", result["trace"])
        self.assertEqual(result["route"], "answer")

    def test_page_fallback_is_reachable(self):
        result = run_assistant("FDTD 稳定条件以及天线远场", retriever=MockFixtureRetriever(), max_iterations=2)
        self.assertIn("page_fallback", result["trace"])
        self.assertTrue(result["fallback_pages"])
        self.assertTrue(result["page_context"])
        self.assertTrue(all(item["is_full_page"] for item in result["page_context"]))
        self.assertIn("整页", result["page_context_instruction"])

    def test_page_fallback_skips_evidence_without_a_resolved_page(self):
        evidence = Evidence(
            "chunk-1", "source-pdf", "COMSOL manual", "retrieved chunk without a reliable page",
            "pdf", None, 0.97, 0.75, "source-pdf:chunk-1",
        )

        class Retriever:
            calls = 0

            def get_page(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("unresolved page must not trigger a PDF page read")

        retriever = Retriever()
        state = {"selected_evidence": [evidence.as_dict()], "iteration": 0}
        result = _page_fallback(state, retriever)

        self.assertEqual(retriever.calls, 0)
        self.assertEqual(result["fallback_pages"], [])
        self.assertEqual(result["selected_evidence"][0]["evidence_id"], "chunk-1")

    def test_case_query_coverage_uses_specific_english_phrases_without_loose_token_hits(self):
        state: AssistantState = {
            "question": "COMSOL 超表面光束偏转器的形状优化包含哪些设计变量、边界范围、迭代次数和目标衍射级次？"
        }
        _decompose(state)
        parts = [QuestionPart(**item) for item in state["subquestions"]]
        alias_groups = [group for part in parts for group in part.required_term_groups]
        flattened = {term for group in alias_groups for term in group}
        self.assertIn("metasurface beam deflector", flattened)
        self.assertIn("shape optimization", flattened)
        self.assertIn("design variables", flattened)
        self.assertIn("lower bound", flattened)
        self.assertIn("upper bound", flattened)
        self.assertIn("moveMax", flattened)
        self.assertIn("scaleMax", flattened)
        self.assertIn("optimization iterations", flattened)
        self.assertIn("target diffraction order", flattened)

        evidence = Evidence(
            "case-1", "case-pdf", "COMSOL metasurface beam deflector shape optimization",
            "The metasurface beam deflector uses shape optimization. The design variables are pillar positions and radii. "
            "The maximum translation moveMax is 30 nm; X and Y have lower bound -moveMax "
            "and upper bound moveMax. The maximum scaling scaleMax is 0.2; XY is limited by "
            "1-scaleMax and 1+scaleMax. The maximum number of iterations is 50, and the model maximizes "
            "transmittance of the target diffraction order [1,0]. Results report reflection from 85% to 92%.",
            "pdf", 4, 0.97, 0.92, "case-pdf:p4",
        )
        result = evaluate_evidence(parts, [evidence], 0, 1, False)
        self.assertEqual(result.evidence_coverage, 1.0)
        self.assertEqual(result.conflict, 1.0)
        self.assertTrue(any("指标口径不一致" in note for note in result.conflict_notes))

        unrelated = Evidence(
            "case-2", "other-pdf", "Generic optimization case",
            "A numerical optimization improves a response curve.", "pdf", 1, 0.9, 0.9, "other-pdf:p1",
        )
        rejected = evaluate_evidence(parts, [unrelated], 0, 1, False)
        self.assertLess(rejected.evidence_coverage, 1.0)

    def test_thin_wire_numeric_coefficient_requires_the_cited_value(self):
        state: AssistantState = {"question": "薄导线等效半径模型包含哪个数值系数？"}
        _decompose(state)
        parts = [QuestionPart(**item) for item in state["subquestions"]]
        correct = Evidence(
            "thin-wire-1", "case-pdf", "COMSOL thin wire equivalent radius model",
            "Thin wire equivalent radius is computed with coefficient 0.23.",
            "pdf", 7, 0.97, 0.9, "case-pdf:p7",
        )
        wrong = Evidence(
            "thin-wire-2", "case-pdf", "COMSOL thin wire equivalent radius model",
            "Thin wire equivalent radius is computed with coefficient 0.24.",
            "pdf", 7, 0.97, 0.9, "case-pdf:p7",
        )
        self.assertEqual(evaluate_evidence(parts, [correct], 0, 1, False).evidence_coverage, 1.0)
        self.assertEqual(evaluate_evidence(parts, [wrong], 0, 1, False).evidence_coverage, 0.0)

    def test_code_example_coverage_requires_geometry_sources_boundaries_and_provenance(self):
        state: AssistantState = {
            "question": "给我一个 Meep Python FDTD 基础代码例子，包含几何、材料、plane-wave 源、PML 和 flux monitor，并标注版本来源。"
        }
        _decompose(state)
        parts = [QuestionPart(**item) for item in state["subquestions"]]
        evidence = Evidence(
            "meep-example", "meep", "Meep Python FDTD source code example",
            "geometry = [mp.Block(material=mp.Medium(epsilon=12))]; plane-wave source = mp.Source(mp.GaussianSource(1.0), component=mp.Ez); "
            "boundary_layers = [mp.PML(1.0)]; flux = sim.add_flux(...). Source repository URL and commit 0a4e71f9.",
            "code", None, 0.82, 0.9, "https://github.com/NanoComp/meep/blob/0a4e71f9/python/examples/basic.py",
        )
        result = evaluate_evidence(parts, [evidence], 0, 1, False)
        self.assertEqual(result.evidence_coverage, 1.0)
        self.assertEqual(result.missing_parts, [])

    def test_code_gap_does_not_trigger_pdf_page_fallback(self):
        evidence = Evidence(
            "code-geometry", "meep-source", "Meep FDTD code example",
            "geometry = mp.Block(size=mp.Vector3(1, 1, 0))",
            "code", None, 0.82, 0.9, "https://github.com/NanoComp/meep/blob/abc123/examples/geometry.py",
        )

        class CodeRetriever:
            def search(self, _query, top_k=5):
                return [evidence]

            def get_page(self, *_args, **_kwargs):
                raise AssertionError("code evidence has no PDF page to expand")

        result = run_assistant(
            "FDTD 代码，包含几何和材料",
            retriever=CodeRetriever(), max_iterations=2, include_full_page=False,
        )
        self.assertNotIn("page_fallback", result["trace"])
        self.assertIn("q2", result["evaluation"]["missing_parts"])
        self.assertEqual(result["trace"], ["decompose", "retrieve", "assess", "generate"])

    def test_ragflow_page_adapter_rejects_missing_page_without_io(self):
        class Client:
            calls = 0

            def get_page(self, *_args, **_kwargs):
                self.calls += 1
                return []

        client = Client()
        retriever = RAGFlowRetriever(client)
        self.assertEqual(retriever.get_page("source-pdf", None), [])
        self.assertEqual(retriever.get_page("source-pdf", 0), [])
        self.assertEqual(client.calls, 0)

    def test_full_page_context_can_be_omitted(self):
        result = run_assistant("FDTD 稳定条件以及天线远场", include_full_page=False)
        self.assertNotIn("page_context", result)

    def test_no_external_adapter_claims_connection(self):
        with self.assertRaises(RuntimeError):
            RAGFlowRetriever().search("CFL")

    def test_web_route_is_injectable_and_only_used_when_enabled(self):
        result = run_assistant(
            "FDTD 稳定条件以及天线远场",
            web_searcher=FakeWebSearcher(),
            allow_web=True,
            max_iterations=2,
        )
        self.assertIn("web_retrieve", result["trace"])
        self.assertLess(result["trace"].index("page_fallback"), result["trace"].index("web_retrieve"))
        self.assertEqual(result["web_search_status"], "used")
        self.assertIn("web-fixture:1", result["citations"])

    def test_web_request_reports_not_configured_instead_of_claiming_network_access(self):
        result = run_assistant("FDTD 稳定条件以及天线远场", allow_web=True)
        self.assertEqual(result["web_search_status"], "not_configured")

    def test_api_uses_explicitly_configured_web_fallback(self):
        if app is None:
            self.skipTest("API extra is not installed")
        from fastapi.testclient import TestClient

        configure_web_searcher(FakeWebSearcher())
        try:
            response = TestClient(app).post("/answer", json={"question": "FDTD 稳定条件以及天线远场", "allow_web": True})
        finally:
            configure_web_searcher(None)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["web_search_status"], "used")

    def test_workflow_feeds_simulation_and_drawing_agents(self):
        result = run_workflow(
            "FDTD CFL 网格稳定性和场分布",
            SimulationSpec("FDTD", "meep", 3, 10e9, 1e-3, 1e-3, 1e-3, 1e-12),
        )
        self.assertEqual(result["simulation_plan"]["execution"], "dry_run_only")
        self.assertTrue(result["knowledge"])
        self.assertIn("visualization_task", result)
        self.assertEqual(result["validation_issues"], [])

    def test_workflow_rejects_cfl_violation_and_passes_warning_to_drawing(self):
        result = run_workflow(
            "FDTD CFL 稳定性",
            SimulationSpec("FDTD", "openems", 3, 10e9, 1e-3, 1e-3, 1e-3, 1e-9),
        )
        codes = {issue["code"] for issue in result["validation_issues"]}
        self.assertIn("cfl_violation", codes)
        task_codes = {issue["code"] for issue in result["visualization_task"]["data_contract"]["issues"]}
        self.assertIn("cfl_violation", task_codes)

    def test_workflow_handles_invalid_material_without_crashing(self):
        result = run_workflow(
            "FDTD 材料参数",
            SimulationSpec("FDTD", "meep", 3, 1e9, 1e-4, 1e-4, 1e-4, 1e-13, 0.0),
        )
        codes = {issue["code"] for issue in result["validation_issues"]}
        self.assertIn("material", codes)
        self.assertEqual(result["simulation_plan"]["execution"], "blocked")

    def test_workflow_handles_invalid_3d_spacing_without_dividing_by_zero(self):
        result = run_workflow(
            "FDTD 三维网格",
            SimulationSpec("FDTD", "meep", 3, 1e9, 1e-3, 1e-3, 0.0, 1e-12),
        )
        codes = {issue["code"] for issue in result["validation_issues"]}
        self.assertIn("length_unit", codes)
        self.assertEqual(result["simulation_plan"]["execution"], "blocked")

    def test_unknown_software_is_blocked_instead_of_guessed(self):
        result = run_workflow(
            "FDTD 场分布",
            SimulationSpec("FDTD", "unknown-solver", 3, 1e9, 1e-4, 1e-4, 1e-4, 1e-13),
        )
        self.assertEqual(result["simulation_plan"]["execution"], "blocked")
        self.assertIn("unsupported_software", {issue["code"] for issue in result["validation_issues"]})

    def test_knowledge_provider_failure_is_fail_closed(self):
        result = run_workflow(
            "FDTD CFL 稳定性",
            SimulationSpec("FDTD", "meep", 3, 1e9, 1e-4, 1e-4, 1e-4, 1e-13),
            provider=FailingKnowledgeProvider(),
        )
        self.assertEqual(result["simulation_plan"]["execution"], "blocked")
        self.assertIn("knowledge_unavailable", {issue["code"] for issue in result["validation_issues"]})

    def test_langchain_prompt_seam_formats_without_credentials(self):
        prompt = answer_prompt("CFL 是什么？", "证据：dt 受 CFL 限制")
        self.assertTrue(prompt)

    def test_api_module_is_optional_and_has_app_when_installed(self):
        # The test environment may omit the API extra, but the import must remain safe.
        if app is not None:
            self.assertTrue(hasattr(app, "routes"))


if __name__ == "__main__":
    unittest.main()
