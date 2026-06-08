"""仿真引擎抽象接口。

所有具体引擎（JAX-FEM、FEALPy 等）实现此接口，
使上层节点与具体引擎解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SolveResult:
    """单次仿真求解结果"""
    displacement: np.ndarray             # 位移场
    compliance: float                    # 柔度值
    volume_fraction: float               # 当前体积分数
    converged: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptResult:
    """优化迭代最终结果"""
    densities: Any                       # 最终密度场 (numpy 或 dolfin.Function)
    compliance_history: list[float]      # 柔度收敛历史
    volume_history: list[float]          # 体积分数历史
    iterations: int                      # 实际迭代次数
    converged: bool
    extra: dict[str, Any] = field(default_factory=dict)
    mesh_info: dict[str, Any] = field(default_factory=dict)  # 网格元信息


class TopoEngine(ABC):
    """拓扑优化仿真引擎抽象基类"""

    @abstractmethod
    def setup(self, problem: dict[str, Any]) -> None:
        """根据结构化问题定义初始化引擎。"""
        ...

    @abstractmethod
    def optimize(
        self,
        *,
        max_iter: int = 200,
        tol: float = 0.01,
        penal: Optional[float] = None,
        rmin: Optional[float] = None,
        volfrac: Optional[float] = None,
    ) -> OptResult:
        """执行拓扑优化迭代计算。"""
        ...

    @abstractmethod
    def get_density_field(self) -> np.ndarray:
        """返回当前密度场。"""
        ...

    @abstractmethod
    def export_image(self, path: str, dpi: int = 300) -> str:
        """将密度场导出为图片，返回保存路径。"""
        ...
