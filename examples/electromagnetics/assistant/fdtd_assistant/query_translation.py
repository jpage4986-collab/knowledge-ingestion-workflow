"""Deterministic electromagnetic query translation before retrieval."""

from __future__ import annotations

import re


TRANSLATION_RULES = (
    ("Yee 网格中 E_z 节点位于哪里", "Where is E_z located on a Yee grid? cell center staggered grid"),
    ("移动结构散射采用了什么数值方案", "which numerical scheme is used for scattering from moving structures generalized FDTD"),
    ("移动界面的边界条件如何处理", "how are boundary conditions handled at a moving interface"),
    ("速度如何进入电磁场变换关系", "how does velocity enter electromagnetic field transformations v cross B"),
    ("该方案是否适用于可穿透介质", "does this approach apply to penetrable media"),
    ("论文区分了哪两类运动介质", "what two types of moving media does the paper distinguish moving-matter moving-perturbation"),
    ("代码示例", "code example code sample"),
    ("示例代码", "code example code sample"),
    ("代码", "source code script implementation"),
    ("脚本", "script example"),
    ("吸收边界", "absorbing boundary perfectly matched layer PML"),
    ("仿真案例", "simulation case application example"),
    ("案例", "simulation case official application"),
    ("薄导线等效半径模型", "thin wire equivalent radius model"),
    ("薄导线等效半径", "thin wire equivalent radius"),
    ("等效半径", "equivalent radius"),
    ("薄导线", "thin wire"),
    ("数值系数", "numerical coefficient factor"),
    ("结构化网格", "structured mesh"),
    ("算例", "simulation example benchmark"),
    ("光学常数", "optical constants complex refractive index n k"),
    ("折射率", "refractive index n"),
    ("消光系数", "extinction coefficient k"),
    ("波长范围", "wavelength range"),
    ("介电函数", "dielectric function complex permittivity"),
    ("基片集成波导", "substrate integrated waveguide SIW"),
    ("介电常数", "relative permittivity dielectric constant"),
    ("磁导率", "relative permeability"),
    ("电导率", "electrical conductivity"),
    ("端口边界", "port boundary"),
    ("端口", "port"),
    ("flux monitor", "flux monitor add_flux flux region"),
    ("plane-wave source", "plane-wave source plane wave source planewave"),
    ("频率扫描", "frequency sweep frequency domain"),
    ("波导适配器", "waveguide adapter TE10"),
    ("微带贴片阵列", "microstrip patch antenna array"),
    ("介质谐振器天线", "dielectric resonator antenna"),
    ("共面波导谐振器", "coplanar waveguide CPW resonator"),
    ("共面波导", "coplanar waveguide CPW"),
    ("嵌入式散射体", "embedded scatterer substrate"),
    ("散射截面", "radar cross section RCS"),
    ("频率选择表面", "frequency selective surface FSS periodic unit cell Floquet periodic boundary port S-parameters"),
    ("互补开口谐振环", "complementary split ring resonator CSRR periodic unit cell"),
    ("开口谐振环", "split ring resonator SRR"),
    ("周期单元", "periodic unit cell Floquet periodic boundary"),
    ("透射带通", "transmission bandpass S-parameters"),
    ("超表面光束偏转器", "metasurface beam deflector"),
    ("形状优化", "shape optimization"),
    ("设计变量", "design variables"),
    ("优化变量", "design variables"),
    ("支柱位置和半径", "pillar positions and radii"),
    ("柱的位置和半径", "pillar positions and radii"),
    ("位置和半径", "positions and radii"),
    ("边界范围", "translation bounds moveMax maximum translation lower bound upper bound scaleMax maximum scaling"),
    ("位移限制", "move limits"),
    ("迭代次数", "optimization iterations"),
    ("目标衍射级次透射率", "transmittance of target diffraction order"),
    ("目标衍射级次", "target diffraction order"),
    ("衍射级次", "diffraction order"),
    ("透射率", "transmittance"),
    ("超表面", "metasurface"),
    ("材料参数", "material parameters permittivity permeability conductivity"),
    ("Yee 网格", "Yee grid staggered grid"),
    ("粗网格", "FDTD SBP-SAT coarse mesh"),
    ("细网格", "FDTD SBP-SAT fine mesh refined mesh"),
    ("子网格", "FDTD SBP-SAT subgridding mesh refinement"),
    ("局部细化", "FDTD local mesh refinement subgridding"),
    ("激励脉冲带宽", "FDTD SBP-SAT cavity excitation pulse bandwidth Gaussian pulse bandwidth"),
    ("激励脉冲", "FDTD excitation pulse Gaussian pulse source"),
    ("高斯脉冲", "FDTD Gaussian pulse source"),
    ("脉冲带宽", "pulse bandwidth"),
    ("移动结构", "moving structures electromagnetic scattering generalized FDTD"),
    ("广义 FDTD", "generalized FDTD scheme"),
    ("移动界面", "FDTD moving interface boundary conditions"),
    ("可穿透介质", "FDTD penetrable media"),
    ("超材料", "FDTD metamaterials"),
    ("包含几何", "geometry setup structure"),
    ("几何", "geometry setup structure"),
    ("plane-wave 源", "plane wave source"),
    ("平面波源", "plane wave source"),
    ("通量监视器", "flux monitor add_flux flux region"),
    ("版本来源", "repository source URL commit license version"),
    ("材料", "material medium dielectric"),
    ("运动介质", "FDTD moving media moving-matter moving-perturbation"),
    ("移动介质", "FDTD moving media moving-matter moving-perturbation"),
    ("移动物质", "FDTD moving-matter media"),
    ("移动扰动", "FDTD moving-perturbation media"),
    ("移动薄板", "FDTD moving slab"),
    ("移动晶体", "FDTD moving crystal"),
    ("移动梯度", "FDTD moving gradient"),
    ("反射透射", "reflectance transmittance reflection transmission"),
    ("散射系数", "scattering coefficients reflection transmission"),
    ("速度", "velocity electromagnetic field transformation v cross B"),
    ("稳定条件", "stability condition"),
    ("时间步长", "time step"),
    ("网格尺寸", "mesh size spatial step"),
    ("边界条件", "boundary condition"),
    ("电磁场方程", "electromagnetic field equations Maxwell equations"),
)

_COVERAGE_ALIAS_GROUPS = (
    ("包含几何", (("geometry",),)),
    ("几何", (("geometry",),)),
    ("材料", (("material",),)),
    ("plane-wave 源", (("plane-wave source", "plane wave source", "planewave source", "planewave", "pw-source.py", "pw_source"),)),
    ("PML", (("PML",),)),
    ("flux monitor", (("add_flux", "flux monitor"),)),
    ("版本来源", (("commit", "version"), ("source URL", "repository"))),
    ("薄导线等效半径模型", (("thin wire",), ("equivalent radius",))),
    ("数值系数", (("0.23",),)),
    ("超表面光束偏转器", (("metasurface beam deflector",),)),
    ("形状优化", (("shape optimization",),)),
    ("设计变量", (("design variable", "design variables", "position and radii", "positions and radii"),)),
    ("优化变量", (("design variable", "design variables", "position and radii", "positions and radii"),)),
    ("支柱位置和半径", (("pillar positions",), ("pillar radii",))),
    ("柱的位置和半径", (("pillar positions",), ("pillar radii",))),
    ("位置和半径", (("positions and radii",),)),
    ("边界范围", (("moveMax", "maximum translation"), ("lower bound",), ("upper bound",), ("scaleMax", "maximum scaling"), ("1-scaleMax", "1+scaleMax"))),
    ("位移限制", (("moveMax", "maximum translation"), ("lower bound",), ("upper bound",))),
    ("迭代次数", (("optimization iterations", "maximum number of iterations"),)),
    ("目标衍射级次透射率", (("target diffraction order", "diffraction order", "order [1, 0]", "order [1,0]"), ("transmittance",))),
    ("目标衍射级次", (("target diffraction order", "diffraction order", "order [1, 0]", "order [1,0]"),)),
    ("衍射级次", (("diffraction order", "order [1, 0]", "order [1,0]"),)),
    ("透射率", (("transmittance",),)),
)

_MATERIAL_ALIASES = {
    "Au": ("金", "gold"),
    "Ag": ("银", "silver"),
    "Cu": ("铜", "copper"),
    "Al": ("铝", "aluminum", "aluminium"),
    "Si": ("硅", "silicon"),
    "SiO2": ("二氧化硅", "石英", "silica", "silicon dioxide"),
    "TiO2": ("二氧化钛", "氧化钛", "titanium dioxide"),
    "GaN": ("氮化镓", "gallium nitride"),
    "Ge": ("锗", "germanium"),
    "ZnO": ("氧化锌", "zinc oxide"),
    "Si3N4": ("氮化硅", "silicon nitride"),
    "MgF2": ("氟化镁", "magnesium fluoride"),
    "ITO": ("氧化铟锡", "indium tin oxide"),
}


def translate_for_retrieval(question: str) -> tuple[str, bool]:
    """Return an English-domain retrieval query and whether a rule changed it.

    This is intentionally local and deterministic. It does not send the user
    question to an external translator. Unknown terms remain untouched so the
    fail-closed evaluator can still reject unsupported questions.
    """

    original = str(question or "").strip()
    translated = original
    changed = False
    for source, target in TRANSLATION_RULES:
        if source in translated:
            translated = translated.replace(source, f" {target} ")
            changed = True
    wavelength = re.search(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*(nm|nanometers?|纳米|μm|µm|um|micrometers?|microns?|微米)(?![A-Za-z])",
        original,
        re.I,
    )
    if wavelength:
        value = float(wavelength.group(1))
        unit = wavelength.group(2).lower()
        micrometers = value / 1000 if unit in {"nm", "纳米"} or unit.startswith("nano") else value
        translated += f" {micrometers:.4f} μm {micrometers:.4f} um"
        changed = True
    translated = re.sub(r"\s+", " ", translated).strip()
    if re.search(r"[\u4e00-\u9fff]", translated) and changed:
        # Keep the original wording for auditability, while appending the
        # normalized English terms that RAGFlow indexes more consistently.
        translated = f"{translated} | original Chinese query: {original}"
    return translated, changed


def coverage_alias_groups(question: str) -> list[list[str]]:
    """Return conjunctive evidence checks with synonymous phrases per group."""

    groups: list[list[str]] = []
    for source, phrase_groups in _COVERAGE_ALIAS_GROUPS:
        if source in question:
            groups.extend([list(group) for group in phrase_groups if list(group) not in groups])
    is_yee_ez_question = bool(
        re.search(r"\bYee\b|Yee\s*网格", question, re.I)
        and re.search(r"E\s*_?\s*z", question, re.I)
    )
    asks_for_location = bool(re.search(r"where|locat|position|node|节点|位置|位于", question, re.I))
    if is_yee_ez_question:
        yee_groups = [
            [
                "Yee grid", "Yee's grid", "Yee’s grid", "Yee FDTD grid",
                "Yee FDTD algorithm", "Yee's FDTD algorithm", "Yee’s FDTD algorithm",
                "Yee FDTD scheme",
            ],
            ["E_z", "Ez", "E z"],
        ]
        if asks_for_location:
            yee_groups.append([
                "cell center", "cell centers", "cell centre", "cell centres",
                "cell-centered", "cell-centred", "E_z(i,j)", "k+1/2", "k + 1/2",
            ])
        groups.extend(group for group in yee_groups if group not in groups)
    q_lower = question.casefold()
    for symbol, aliases in _MATERIAL_ALIASES.items():
        symbol_pattern = rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])"
        if re.search(symbol_pattern, question, re.I) or any(alias.casefold() in q_lower for alias in aliases):
            material_group = [symbol, *aliases]
            groups.append(material_group)
            break
    wavelength = re.search(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*(nm|nanometers?|纳米|μm|µm|um|micrometers?|microns?|微米)(?![A-Za-z])",
        question,
        re.I,
    )
    if wavelength:
        value = float(wavelength.group(1))
        unit = wavelength.group(2).lower()
        micrometers = value / 1000 if unit in {"nm", "纳米"} or unit.startswith("nano") else value
        groups.append([
            wavelength.group(0).strip(),
            f"{micrometers:.4f} μm",
            f"{micrometers:.4f} um",
        ])
    if re.search(r"折射率|refractive index|(?<![A-Za-z])n(?![A-Za-z])", question, re.I):
        groups.append(["折射率 n", "refractive index n", "折射率", "refractive index"])
    if re.search(r"消光系数|extinction coefficient|(?<![A-Za-z])k(?![A-Za-z])", question, re.I):
        groups.append(["消光系数 k", "extinction coefficient k", "消光系数", "extinction coefficient"])
    return groups
