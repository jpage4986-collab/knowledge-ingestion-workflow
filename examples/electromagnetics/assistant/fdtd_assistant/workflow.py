from __future__ import annotations

from typing import Any, TypedDict

from .knowledge import FixtureKnowledgeProvider, KnowledgeProvider, SimulationSpec, SoftwareCapabilityRegistry, make_visualization_task, validate_simulation


class WorkflowState(TypedDict, total=False):
    request: str
    simulation_spec: dict[str, Any]
    knowledge: list[dict[str, Any]]
    validation_issues: list[dict[str, Any]]
    simulation_plan: dict[str, Any]
    visualization_task: dict[str, Any]
    trace: list[str]


def _retrieve_knowledge(state: WorkflowState, provider: KnowledgeProvider) -> WorkflowState:
    spec = SimulationSpec(**state["simulation_spec"])
    try:
        claims = provider.search_claims(state.get("request", spec.method), method=spec.method)
        state["knowledge"] = [claim.as_dict() for claim in claims]
    except Exception as exc:
        state["knowledge"] = []
        state["knowledge_error"] = f"{type(exc).__name__}: knowledge provider unavailable"
    state.setdefault("trace", []).append("retrieve_knowledge")
    return state


def _compile_plan(state: WorkflowState) -> WorkflowState:
    spec = SimulationSpec(**state["simulation_spec"])
    capabilities = SoftwareCapabilityRegistry()
    supported = capabilities.supports(spec.software, spec.method)
    state["simulation_plan"] = {"software": spec.software, "method": spec.method, "supported": supported, "execution": "dry_run_only" if supported else "blocked", "spec": spec.as_dict()}
    state.setdefault("trace", []).append("compile_simulation_plan")
    return state


def _validate(state: WorkflowState, provider: KnowledgeProvider) -> WorkflowState:
    spec = SimulationSpec(**state["simulation_spec"])
    try:
        issues, claims = validate_simulation(spec, provider)
    except Exception as exc:
        from .knowledge import ValidationIssue

        issues, claims = [ValidationIssue("knowledge_unavailable", "error", f"knowledge validation failed closed: {type(exc).__name__}")], []
    state["validation_issues"] = [issue.as_dict() for issue in issues]
    if any(issue.severity == "error" for issue in issues):
        state.setdefault("simulation_plan", {})["execution"] = "blocked"
    if not state.get("knowledge"):
        state["knowledge"] = [claim.as_dict() for claim in claims]
    state.setdefault("trace", []).append("validate_simulation")
    return state


def _visualize(state: WorkflowState) -> WorkflowState:
    from .knowledge import KnowledgeClaim, ValidationIssue

    claims = [KnowledgeClaim(**claim) for claim in state.get("knowledge", [])]
    issues = [ValidationIssue(**issue) for issue in state.get("validation_issues", [])]
    task = make_visualization_task(SimulationSpec(**state["simulation_spec"]), claims, issues)
    state["visualization_task"] = task.as_dict()
    state.setdefault("trace", []).append("make_visualization_task")
    return state


class LocalWorkflow:
    def __init__(self, provider: KnowledgeProvider) -> None:
        self.provider = provider

    def invoke(self, initial: WorkflowState) -> WorkflowState:
        state: WorkflowState = dict(initial)
        state.setdefault("trace", [])
        _retrieve_knowledge(state, self.provider)
        _compile_plan(state)
        _validate(state, self.provider)
        _visualize(state)
        return state


def _langgraph_workflow(provider: KnowledgeProvider):
    from langgraph.graph import END, StateGraph  # type: ignore

    graph = StateGraph(WorkflowState)
    graph.add_node("retrieve_knowledge", lambda s: _retrieve_knowledge(s, provider))
    graph.add_node("compile_simulation_plan", _compile_plan)
    graph.add_node("validate_simulation", lambda s: _validate(s, provider))
    graph.add_node("make_visualization_task", _visualize)
    graph.set_entry_point("retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "compile_simulation_plan")
    graph.add_edge("compile_simulation_plan", "validate_simulation")
    graph.add_edge("validate_simulation", "make_visualization_task")
    graph.add_edge("make_visualization_task", END)
    return graph.compile()


_DEFAULT_WORKFLOW = None


def build_workflow(provider: KnowledgeProvider | None = None):
    global _DEFAULT_WORKFLOW
    if provider is None and _DEFAULT_WORKFLOW is not None:
        return _DEFAULT_WORKFLOW
    provider = provider or FixtureKnowledgeProvider()
    try:
        graph = _langgraph_workflow(provider)
    except ImportError:
        graph = LocalWorkflow(provider)
    if provider.__class__ is FixtureKnowledgeProvider and _DEFAULT_WORKFLOW is None:
        _DEFAULT_WORKFLOW = graph
    return graph


def run_workflow(request: str, simulation_spec: SimulationSpec, *, provider: KnowledgeProvider | None = None) -> WorkflowState:
    return build_workflow(provider).invoke({"request": request, "simulation_spec": simulation_spec.as_dict()})
