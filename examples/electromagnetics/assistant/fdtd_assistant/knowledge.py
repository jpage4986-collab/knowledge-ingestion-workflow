"""Knowledge runtime for the workflow agent.

The upstream literature agent owns acquisition and extraction. This module owns
normalization, retrieval, applicability checks, and simulation-facing contracts.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from .adapters import DOMAIN_TERMS

C0 = 299_792_458.0


@dataclass(frozen=True)
class KnowledgeClaim:
    claim_id: str
    subject: str
    predicate: str
    object: str
    content: str
    domain: str
    method: str | None
    conditions: dict[str, Any]
    formula: str | None
    variables: dict[str, dict[str, str]]
    source_title: str
    source_url: str
    source_locator: str
    source_quality: float
    confidence: float
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationSpec:
    method: str
    software: str
    dimension: int
    max_frequency_hz: float
    dx_m: float
    dy_m: float
    dz_m: float | None = None
    dt_s: float | None = None
    relative_permittivity: float = 1.0
    boundary: str = "open"
    outputs: list[str] = field(default_factory=lambda: ["field", "s_parameters"])

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    expected: str | None = None
    actual: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualizationTask:
    figure_type: str
    title: str
    data_contract: dict[str, Any]
    annotations: list[dict[str, Any]]
    citations: list[dict[str, str]]
    restrictions: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeProvider(Protocol):
    def search_claims(self, query: str, *, method: str | None = None, top_k: int = 8) -> Sequence[KnowledgeClaim]: ...

    def get_claim(self, claim_id: str) -> KnowledgeClaim | None: ...


def _terms(text: str) -> set[str]:
    result = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower()))
    result |= set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    result |= {term.lower() for term in DOMAIN_TERMS if term.lower() in text.lower() or term in text}
    return result


def paper_knowledge_fixtures() -> list[KnowledgeClaim]:
    """Small source-backed templates paraphrased from public abstracts/pages."""
    return [
        KnowledgeClaim(
            "fdtd-yee-grid", "Yee grid", "staggered_representation", "E/H fields",
            "Yee 网格将电场 E 与磁场 H 在空间和时间上交错放置。", "electromagnetics", "FDTD",
            {"dimension": "1D/2D/3D"}, None, {}, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media",
            "https://doi.org/10.1109/TAP.1966.1138693", "fixture:yee1966:abstract", 0.95, 0.95, ["grid", "field", "discretization"],
        ),
        KnowledgeClaim(
            "fdtd-cfl-3d", "FDTD time step", "bounded_by", "CFL stability limit",
            "三维均匀介质中的 FDTD 时间步长必须不超过 CFL 稳定上限。", "electromagnetics", "FDTD",
            {"dimension": 3, "material": "uniform"},
            "dt <= 1/(c*sqrt((1/dx)^2+(1/dy)^2+(1/dz)^2))",
            {"dt": {"unit": "s"}, "dx": {"unit": "m"}, "dy": {"unit": "m"}, "dz": {"unit": "m"}, "c": {"unit": "m/s"}},
            "FDTD stability and accuracy analysis", "https://arxiv.org/abs/1406.7042", "fixture:fdtd-cfl:p1", 0.85, 0.94,
            ["stability", "formula", "unit"],
        ),
        KnowledgeClaim(
            "coding-metamaterial-1bit", "digital coding metamaterial", "uses_states", "0/1 phase coding",
            "1-bit 编码超材料使用两类相位响应约为 0 和 π 的单元；更高 bit 数可提供更多离散相位状态。", "metamaterials", None,
            {"bit_depth": "1 or higher"}, "phase states = {0, pi} for 1-bit",
            {"phase": {"unit": "rad or degree"}}, "Coding metamaterials, digital metamaterials and programmable metamaterials",
            "https://www.nature.com/articles/lsa201499", "fixture:lsa2014:abstract", 0.95, 0.96,
            ["coding", "digital", "programmable", "phase"],
        ),
        KnowledgeClaim(
            "space-time-metasurface", "space-time coding metasurface", "controls", "direction and harmonic power",
            "空间-时间编码超表面可以同时调控电磁波的空间传播方向和谐波功率分布。", "metamaterials", None,
            {"modulation": "time-varying"}, None, {}, "Space-time-coding digital metasurfaces",
            "https://www.nature.com/articles/s41467-018-06802-0", "fixture:ncomms2018:abstract", 0.95, 0.95,
            ["space-time", "harmonic", "beamforming"],
        ),
    ]


class FixtureKnowledgeProvider:
    def __init__(self, claims: Iterable[KnowledgeClaim] | None = None) -> None:
        self.claims = list(claims or paper_knowledge_fixtures())
        self._cache: dict[tuple[str, str | None, int], tuple[KnowledgeClaim, ...]] = {}

    def search_claims(self, query: str, *, method: str | None = None, top_k: int = 8) -> Sequence[KnowledgeClaim]:
        cache_key = (query, method, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]
        q = _terms(query)
        scored: list[tuple[float, KnowledgeClaim]] = []
        for claim in self.claims:
            if method and claim.method and claim.method.lower() != method.lower():
                continue
            text = " ".join((claim.subject, claim.predicate, claim.object, claim.content, " ".join(claim.tags)))
            overlap = len(q & _terms(text))
            if overlap:
                score = min(1.0, 0.25 + 0.12 * overlap + 0.25 * claim.confidence)
                scored.append((score, claim))
        scored.sort(key=lambda pair: (-pair[0], pair[1].claim_id))
        result = tuple(claim for _, claim in scored[:top_k])
        self._cache[cache_key] = result
        return result

    def get_claim(self, claim_id: str) -> KnowledgeClaim | None:
        return next((claim for claim in self.claims if claim.claim_id == claim_id), None)


class SoftwareCapabilityRegistry:
    """Canonical names and execution seams; no software is launched here."""

    SUPPORTED = {
        "cst": {"fdtd", "frequency_domain", "time_domain"},
        "hfss": {"frequency_domain", "fem"},
        "comsol": {"fdtd", "fem", "frequency_domain"},
        "meep": {"fdtd"},
        "openems": {"fdtd"},
        "mock": {"fdtd", "fem", "frequency_domain", "time_domain"},
    }

    def supports(self, software: str, method: str) -> bool:
        return method.lower() in self.SUPPORTED.get(software.lower(), set())


def validate_simulation(spec: SimulationSpec, provider: KnowledgeProvider) -> tuple[list[ValidationIssue], list[KnowledgeClaim]]:
    claims = list(provider.search_claims(f"{spec.method} 网格 CFL 稳定性 公式 单位", method=spec.method))
    issues: list[ValidationIssue] = []
    if not SoftwareCapabilityRegistry().supports(spec.software, spec.method):
        issues.append(ValidationIssue("unsupported_software", "error", f"software '{spec.software}' has no registered support for method '{spec.method}'"))
    if spec.dimension not in (1, 2, 3):
        issues.append(ValidationIssue("dimension", "error", "dimension must be 1, 2, or 3"))
    if spec.max_frequency_hz <= 0:
        issues.append(ValidationIssue("frequency", "error", "max_frequency_hz must be positive"))
    spacings_valid = spec.dx_m > 0 and spec.dy_m > 0 and (spec.dimension != 3 or (spec.dz_m is not None and spec.dz_m > 0))
    if not spacings_valid:
        issues.append(ValidationIssue("length_unit", "error", "mesh spacings must be positive meters; 3D requires dz_m", "m"))
    if spec.relative_permittivity <= 0:
        issues.append(ValidationIssue("material", "error", "relative_permittivity must be positive"))
    if spec.method.lower() == "fdtd" and spec.dimension in (1, 2, 3) and spec.dt_s is None:
        issues.append(ValidationIssue("missing_dt", "error", "FDTD requires dt_s so the CFL limit can be checked", "s"))
    if spec.method.lower() == "fdtd" and spec.dt_s is not None and spec.max_frequency_hz > 0 and spec.relative_permittivity > 0 and spacings_valid:
        spacings = [spec.dx_m, spec.dy_m] + ([spec.dz_m] if spec.dimension == 3 and spec.dz_m is not None else [])
        c_medium = C0 / math.sqrt(spec.relative_permittivity)
        cfl = 1.0 / (c_medium * math.sqrt(sum(1.0 / (d * d) for d in spacings)))
        if spec.dt_s > cfl * (1 + 1e-12):
            issues.append(ValidationIssue("cfl_violation", "error", "dt_s exceeds the CFL stability limit", f"dt <= {cfl:.6e} s", f"dt = {spec.dt_s:.6e} s", ["fdtd-cfl-3d"]))
        wavelength = c_medium / spec.max_frequency_hz
        max_cell = max(spacings)
        if max_cell > wavelength / 10:
            issues.append(ValidationIssue("coarse_mesh", "warning", "mesh may be too coarse for the highest frequency; inspect dispersion/convergence", f"cell <= {wavelength/10:.6e} m", f"cell = {max_cell:.6e} m", ["fdtd-cfl-3d"]))
    return issues, claims


def make_visualization_task(spec: SimulationSpec, claims: Sequence[KnowledgeClaim], issues: Sequence[ValidationIssue]) -> VisualizationTask:
    figure_type = "field_and_mesh" if "field" in spec.outputs else "simulation_summary"
    annotations = [{"label": "method", "value": spec.method}, {"label": "software", "value": spec.software}, {"label": "units", "value": "SI (m, s, Hz)"}]
    restrictions = ["Do not invent field values; use simulation output files only.", "Show warnings from validation beside the figure."]
    citations = [{"claim_id": claim.claim_id, "locator": claim.source_locator, "url": claim.source_url} for claim in claims]
    return VisualizationTask(figure_type, f"{spec.method} simulation: mesh, fields and validation", {"spec": spec.as_dict(), "required_outputs": spec.outputs, "issues": [issue.as_dict() for issue in issues]}, annotations, citations, restrictions)
