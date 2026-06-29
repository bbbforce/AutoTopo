"""结构化四节点单元 FE 辅助函数。"""

from __future__ import annotations

import numpy as np


def element_stiffness(nu: float = 0.3) -> np.ndarray:
    """返回 8-DOF 四节点平面应力单元刚度矩阵。"""

    from autotopo.engines.python_simp_mma_engine import PythonSimpMMAEngine

    return PythonSimpMMAEngine._element_stiffness(nu)


def build_edof_matrix(nelx: int, nely: int) -> np.ndarray:
    """构建单元到 DOF 映射矩阵。"""

    edof_mat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof_mat[el, :] = np.array([
                2 * n1 + 2,
                2 * n1 + 3,
                2 * n2 + 2,
                2 * n2 + 3,
                2 * n2,
                2 * n2 + 1,
                2 * n1,
                2 * n1 + 1,
            ])
    return edof_mat
