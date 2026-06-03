"""Pydantic 结构化 Schema：定义拓扑优化问题的数据模型。

用于 LLM 结构化输出 (with_structured_output)，
解析后可序列化为 YAML / JSON。
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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


# ──────────────────────────── 问题定义 ────────────────────────────


class DomainSpec(BaseModel):
    """设计域规格"""
    width: float = Field(description="设计域宽度")
    height: float = Field(description="设计域高度")
    nelx: int = Field(default=60, description="x方向单元数")
    nely: int = Field(default=30, description="y方向单元数")
    non_design_regions: List[NonDesignRegion] = Field(
        default_factory=list,
        description="非设计域区域列表（从图片中识别）",
    )


class NonDesignRegion(BaseModel):
    """非设计域区域"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    density: float = Field(default=1.0, description="固定密度值 (0=空洞, 1=实体)")


class MaterialSpec(BaseModel):
    """材料参数"""
    youngs_modulus: float = Field(default=1.0, description="杨氏模量 E")
    poissons_ratio: float = Field(default=0.3, description="泊松比 ν")


class BoundaryCondition(BaseModel):
    """边界条件"""
    type: BCType
    location: str = Field(description="位置描述, 如 'left_edge', 'bottom_left_corner'")
    node_indices: Optional[List[int]] = Field(
        default=None, description="具体节点索引（解析后填充）"
    )


class LoadSpec(BaseModel):
    """载荷定义"""
    type: LoadType
    location: str = Field(description="施加位置描述")
    magnitude: float = Field(description="载荷大小")
    direction: List[float] = Field(description="方向向量 [fx, fy]")
    node_indices: Optional[List[int]] = Field(
        default=None, description="具体节点索引（解析后填充）"
    )


class ConstraintSpec(BaseModel):
    """约束定义"""
    type: ConstraintType
    value: float = Field(description="约束值，如体积分数 0.5")
    description: Optional[str] = Field(default=None, description="约束的文字说明")


class OptParams(BaseModel):
    """优化参数"""
    penal: float = Field(default=3.0, description="SIMP 罚因子")
    rmin: float = Field(default=1.5, description="过滤半径")
    ft: int = Field(default=1, description="过滤类型: 0=灵敏度过滤, 1=密度过滤, 2=Heaviside投影")
    max_iter: int = Field(default=200, description="最大优化迭代数")
    tol: float = Field(default=0.01, description="收敛容差")
    # Heaviside 投影参数（ft=2 时生效）
    beta: float = Field(default=1.0, description="Heaviside投影初始β值，控制投影锐度，越大边界越清晰")
    beta_max: float = Field(default=32.0, description="Heaviside投影最大β值")
    beta_interval: int = Field(default=40, description="β翻倍间隔（每隔多少迭代步β翻倍）")
    eta: float = Field(default=0.5, description="Heaviside投影阈值η，通常取0.5")


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


# 解决前向引用
DomainSpec.model_rebuild()
