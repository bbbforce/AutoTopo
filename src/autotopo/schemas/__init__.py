"""Pydantic 结构化 Schema：定义拓扑优化问题的数据模型。

用于 LLM 结构化输出 (with_structured_output)，
解析后可序列化为 YAML / JSON。

适配 FEniCS (DOLFIN) + dolfin-adjoint 后端。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────── 枚举类型 ────────────────────────────


class BCType(str, Enum):
    """边界条件类型"""
    FIXED = "fixed"
    FIXED_X = "fixed_x"
    FIXED_Y = "fixed_y"
    DISPLACEMENT = "displacement"
    SYMMETRY = "symmetry"
    ROLLER = "roller"


class LoadType(str, Enum):
    """载荷类型"""
    POINT_FORCE = "point_force"
    DISTRIBUTED = "distributed"
    PRESSURE = "pressure"


class ObjectiveType(str, Enum):
    """目标函数类型"""
    MINIMIZE_COMPLIANCE = "minimize_compliance"
    MINIMIZE_VOLUME = "minimize_volume"


class ConstraintType(str, Enum):
    """约束类型"""
    VOLUME_FRACTION = "volume_fraction"
    STRESS = "stress"
    DISPLACEMENT_LIMIT = "displacement_limit"
    CUSTOM = "custom"


class DefectType(str, Enum):
    """拓扑优化缺陷类型"""
    GRAY_ELEMENTS = "gray_elements"
    CHECKERBOARD = "checkerboard"
    ISLAND = "island"
    DISCONNECTION = "disconnection"


class Severity(str, Enum):
    """缺陷严重程度"""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class BenchmarkType(str, Enum):
    """最小研究实验支持的二维基准问题"""
    MBB = "mbb"
    CANTILEVER = "cantilever"
    L_SHAPE = "l_shape"


class BenchmarkMethod(str, Enum):
    """最小 benchmark 对比方法"""
    BASELINE_DIRECT = "baseline_direct"
    BASELINE_NAIVE_RAG = "baseline_naive_rag"
    OURS_CORRECTIVE_RAG = "ours_corrective_rag"


class FailureMode(str, Enum):
    """研究 workflow 的统一失败模式枚举"""
    PYTHON_EXCEPTION = "python_exception"
    MISSING_DEPENDENCY = "missing_dependency"
    SHAPE_MISMATCH = "shape_mismatch"
    SINGULAR_STIFFNESS_MATRIX = "singular_stiffness_matrix"
    INVALID_BOUNDARY_CONDITION = "invalid_boundary_condition"
    NO_SUPPORT = "no_support"
    NO_LOAD = "no_load"
    LOAD_ON_FIXED_DOF = "load_on_fixed_dof"
    RIGID_BODY_MOTION = "rigid_body_motion"
    INVALID_MESH_RESOLUTION = "invalid_mesh_resolution"
    INVALID_VOLUME_FRACTION = "invalid_volume_fraction"
    INVALID_FILTER_RADIUS = "invalid_filter_radius"
    INVALID_PENAL = "invalid_penal"
    NON_CONVERGENCE = "non_convergence"
    COMPLIANCE_NAN_OR_INF = "compliance_nan_or_inf"
    VOLUME_CONSTRAINT_VIOLATION = "volume_constraint_violation"
    DENSITY_COLLAPSE = "density_collapse"
    MMA_OSCILLATION = "mma_oscillation"
    ABNORMAL_OBJECTIVE_INCREASE = "abnormal_objective_increase"
    GRAYNESS_TOO_HIGH = "grayness_too_high"
    CHECKERBOARD = "checkerboard"
    DISCONNECTED_ISLANDS = "disconnected_islands"
    TOO_THIN_MEMBERS = "too_thin_members"
    INVALID_LOAD_PATH = "invalid_load_path"


# ──────────────────────────── 问题定义 ────────────────────────────


class NonDesignRegion(BaseModel):
    """非设计域区域"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    density: float = Field(default=1.0, description="固定密度值 (0=空洞, 1=实体)")


class DomainSpec(BaseModel):
    """设计域规格"""
    width: float = Field(description="设计域宽度")
    height: float = Field(description="设计域高度")
    mesh_resolution: float = Field(
        default=1.0,
        description="Gmsh 网格特征尺寸 (控制网格精细程度，越小越精细)",
    )
    non_design_regions: List[NonDesignRegion] = Field(
        default_factory=list,
        description="非设计域区域列表（从图片中识别）",
    )


class MaterialSpec(BaseModel):
    """材料参数"""
    youngs_modulus: float = Field(default=1.0, description="杨氏模量 E")
    poissons_ratio: float = Field(default=0.3, description="泊松比 ν")


class BoundaryCondition(BaseModel):
    """边界条件"""
    type: BCType
    location: str = Field(description="位置描述, 如 'left_edge', 'bottom_left_corner'")
    value: Optional[List[float]] = Field(
        default=None, description="位移值 [ux, uy]（displacement BC 时使用）"
    )


class LoadSpec(BaseModel):
    """载荷定义"""
    type: LoadType
    location: str = Field(description="施加位置描述")
    magnitude: float = Field(description="载荷大小")
    direction: List[float] = Field(description="方向向量 [fx, fy]")


class ConstraintSpec(BaseModel):
    """约束定义"""
    type: ConstraintType
    value: float = Field(description="约束值，如体积分数 0.5")
    description: Optional[str] = Field(default=None, description="约束的文字说明")


class OptParams(BaseModel):
    """优化参数"""
    penal: float = Field(default=3.0, description="SIMP 罚因子")
    rmin: float = Field(
        default=0.05,
        description="Helmholtz 过滤半径（相对域最大尺寸的比例，如 0.05 表示 5%）",
    )
    max_iter: int = Field(default=200, description="最大优化迭代数")
    tol: float = Field(default=1e-6, description="优化收敛容差")
    optimizer: str = Field(
        default="SLSQP",
        description="优化器名称。当前标准后端默认使用 SLSQP 以支持体积约束",
    )


class OptimizationProblem(BaseModel):
    """完整的拓扑优化问题定义 — LLM 结构化输出的目标 Schema"""
    description: str = Field(description="问题的自然语言描述")
    domain: DomainSpec = Field(description="设计域几何")
    material: MaterialSpec = Field(default_factory=MaterialSpec, description="材料参数")
    boundary_conditions: List[BoundaryCondition] = Field(description="边界条件列表")
    loads: List[LoadSpec] = Field(description="载荷列表")
    objective: ObjectiveType = Field(
        default=ObjectiveType.MINIMIZE_COMPLIANCE,
        description="目标函数",
    )
    constraints: List[ConstraintSpec] = Field(description="约束列表")
    parameters: OptParams = Field(default_factory=OptParams, description="优化参数")


# ──────────────────────────── 评估结果 ────────────────────────────


class ParameterAdjustment(BaseModel):
    """参数调整建议"""
    parameter: str = Field(description="要调整的参数名")
    current_value: float = Field(description="当前值")
    suggested_value: float = Field(description="建议值")
    reason: str = Field(description="调整理由")


class EvaluationResult(BaseModel):
    """视觉评估结果 — 评估 Agent 的结构化输出"""
    has_defects: bool = Field(description="是否存在缺陷")
    defect_types: List[DefectType] = Field(
        default_factory=list, description="检测到的缺陷类型"
    )
    severity: Severity = Field(default=Severity.MINOR, description="缺陷严重程度")
    suggested_fixes: List[ParameterAdjustment] = Field(
        default_factory=list, description="参数调整建议"
    )
    reasoning: str = Field(description="评估推理过程")


# ──────────────────────── 最小研究 workflow Schema ────────────────────────


class CaseSpec(BaseModel):
    """benchmark-aware 的最小拓扑优化 case 描述。"""
    case_id: str
    benchmark_type: BenchmarkType
    natural_language: str = ""
    variant: str = "clear"
    nelx: int = Field(default=30, ge=2, description="x 方向单元数")
    nely: int = Field(default=10, ge=2, description="y 方向单元数")
    volume_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    penal: float = Field(default=3.0, gt=0.0)
    rmin: float = Field(default=1.5, gt=0.0)
    max_iter: int = Field(default=20, ge=1)
    tol: float = Field(default=1e-2, gt=0.0)
    optimizer: str = "MMA"
    material: MaterialSpec = Field(default_factory=MaterialSpec)
    structured_params: Dict[str, Any] = Field(default_factory=dict)
    problem: Dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    """本地 RAG 返回的证据片段。"""
    evidence_id: str
    source: str
    content: str
    score: float = 0.0
    kind: str = "generic"
    parent_id: str = ""
    chunk_id: str = ""
    heading: str = ""
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    rerank_features: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_score_fields(self) -> "RetrievedEvidence":
        """Keep old `score` callers aligned with the richer score breakdown."""

        if self.final_score == 0.0 and self.score != 0.0:
            self.final_score = self.score
        elif self.score == 0.0 and self.final_score != 0.0:
            self.score = self.final_score
        if not self.chunk_id:
            self.chunk_id = self.evidence_id
        if not self.parent_id:
            self.parent_id = self.evidence_id
        return self


class QueryContext(BaseModel):
    """Task-aware retrieval context for local/hybrid RAG."""

    task_type: str = "code_generation"
    benchmark_type: Optional[str] = None
    solver_backend: str = "python_simp_mma"
    optimizer: str = "MMA"
    case_id: str = ""
    natural_language: str = ""
    structured_terms: List[str] = Field(default_factory=list)
    error_text: str = ""
    failure_modes: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Validator 的 fail-closed 检查报告。"""
    case_id: str
    is_valid: bool
    failure_modes: List[FailureMode] = Field(default_factory=list)
    severity: Severity = Severity.MINOR
    messages: List[str] = Field(default_factory=list)
    normalized_problem: Dict[str, Any] = Field(default_factory=dict)


class CodePlan(BaseModel):
    """Planner/Coder 选择的可执行代码任务计划。"""
    case_id: str
    method: BenchmarkMethod
    engine: str = "python_simp_mma"
    template_id: str
    optimizer: str = "MMA"
    allow_generated_code: bool = False
    steps: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExecutionReport(BaseModel):
    """Executor 捕获到的运行结果，workflow 不直接抛异常。"""
    case_id: str
    method: BenchmarkMethod
    success: bool
    output_dir: str
    stdout_path: str = ""
    stderr_path: str = ""
    error_type: Optional[str] = None
    exception: Optional[str] = None
    traceback: Optional[str] = None
    optimizer: str = "MMA"
    optimizer_fallback: Optional[str] = None
    iterations: int = 0
    converged: bool = False
    compliance: Optional[float] = None
    volume_fraction: Optional[float] = None
    files: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class FailureDiagnosis(BaseModel):
    """Reviewer/diagnostics 对失败的结构化诊断。"""
    case_id: str
    has_failure: bool
    failure_modes: List[FailureMode] = Field(default_factory=list)
    severity: Severity = Severity.MINOR
    likely_causes: List[str] = Field(default_factory=list)
    repair_suggestions: List[str] = Field(default_factory=list)
    auto_repair_allowed: bool = False
    evidence_ids: List[str] = Field(default_factory=list)


class RepairPlan(BaseModel):
    """有界自修复计划，只允许修改安全参数。"""
    case_id: str
    should_repair: bool
    repair_iteration: int = 0
    max_repair_rounds: int = 3
    repair_type: str = "fail_closed"
    target: str = ""
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    rationale: str = ""
    parameter_updates: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    failure_modes: List[FailureMode] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    auto_repair_allowed: bool = False
    auto_apply_allowed: bool = False
    risk_level: str = "low"


class EvaluatorReport(BaseModel):
    """执行成功后的拓扑质量与优化有效性评估。"""
    case_id: str
    success: bool
    has_quality_failure: bool = False
    failure_modes: List[FailureMode] = Field(default_factory=list)
    compliance: Optional[float] = None
    volume_error: Optional[float] = None
    grayness_index: Optional[float] = None
    checkerboard_score: Optional[float] = None
    connectivity_score: Optional[float] = None
    objective_nan_or_inf: bool = False
    density_collapse: bool = False
    converged: bool = False
    messages: List[str] = Field(default_factory=list)
    repair_plan: Optional[RepairPlan] = None


class BenchmarkCaseResult(BaseModel):
    """单个 case-method 的汇总行。"""
    case_id: str
    benchmark_type: BenchmarkType
    method: BenchmarkMethod
    first_pass_success: bool = False
    final_success: bool = False
    repair_success: bool = False
    repair_iterations: int = 0
    execution_error_type: Optional[str] = None
    detected_failure_modes: List[FailureMode] = Field(default_factory=list)
    compliance: Optional[float] = None
    volume_error: Optional[float] = None
    grayness_index: Optional[float] = None
    checkerboard_score: Optional[float] = None
    connectivity_score: Optional[float] = None
    converged: bool = False
    output_dir: str = ""
