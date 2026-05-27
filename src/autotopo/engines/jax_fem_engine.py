"""JAX-FEM 仿真引擎适配层。

实现 2D SIMP 拓扑优化，基于经典的 88 行代码逻辑，
使用 NumPy/SciPy 实现核心计算（后续可切换为 JAX 后端）。

NOTE: 这是一个功能完整的 2D SIMP 实现，
      后续集成真正的 JAX-FEM 库时替换此文件。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from autotopo.engines.base import OptResult, SolveResult, TopoEngine
from autotopo.utils.visualization import plot_density_field


class JaxFemEngine(TopoEngine):
    """2D SIMP 拓扑优化引擎。"""

    def __init__(self) -> None:
        self.nelx: int = 60
        self.nely: int = 30
        self.volfrac: float = 0.5
        self.penal: float = 3.0
        self.rmin: float = 1.5
        self.E0: float = 1.0
        self.Emin: float = 1e-9
        self.nu: float = 0.3
        self.densities: Optional[np.ndarray] = None
        self._ke: Optional[np.ndarray] = None
        self._fixed_dofs: list[int] = []
        self._force: Optional[np.ndarray] = None

    # ────────── 接口实现 ──────────

    def setup(self, problem: dict[str, Any]) -> None:
        domain = problem.get("domain", {})
        material = problem.get("material", {})

        self.nelx = domain.get("nelx", 60)
        self.nely = domain.get("nely", 30)
        self.E0 = material.get("youngs_modulus", 1.0)
        self.nu = material.get("poissons_ratio", 0.3)

        # 从约束中提取体积分数
        for c in problem.get("constraints", []):
            if c.get("type") == "volume_fraction":
                self.volfrac = c.get("value", 0.5)

        # 从参数中提取优化设置
        params = problem.get("parameters", {})
        self.penal = params.get("penal", 3.0)
        self.rmin = params.get("rmin", 1.5)

        # 构造单元刚度矩阵
        self._ke = self._element_stiffness()

        # 设置边界条件和载荷
        self._setup_bc_and_loads(problem)

        # 初始化密度场
        self.densities = np.full((self.nely, self.nelx), self.volfrac)

    def optimize(
        self,
        *,
        max_iter: int = 200,
        tol: float = 0.01,
        penal: Optional[float] = None,
        rmin: Optional[float] = None,
        volfrac: Optional[float] = None,
    ) -> OptResult:
        if penal is not None:
            self.penal = penal
        if rmin is not None:
            self.rmin = rmin
        if volfrac is not None:
            self.volfrac = volfrac

        if self.densities is None:
            raise RuntimeError("请先调用 setup() 初始化引擎")

        x = self.densities.copy()
        compliance_history = []
        volume_history = []
        change = 1.0

        for iteration in range(max_iter):
            if change < tol and iteration > 10:
                break

            # FE 求解
            u = self._fe_solve(x)

            # 目标函数 & 灵敏度
            ce = self._element_compliance(u)
            c = float(np.sum(
                (self.Emin + x ** self.penal * (self.E0 - self.Emin)) * ce
            ))
            dc = -self.penal * x ** (self.penal - 1) * (self.E0 - self.Emin) * ce
            dv = np.ones_like(x)

            # 灵敏度过滤
            dc = self._density_filter(x, dc)

            # OC 更新
            x_new = self._oc_update(x, dc, dv)
            change = float(np.max(np.abs(x_new - x)))
            x = x_new

            compliance_history.append(c)
            volume_history.append(float(np.mean(x)))

        self.densities = x
        return OptResult(
            densities=x,
            compliance_history=compliance_history,
            volume_history=volume_history,
            iterations=len(compliance_history),
            converged=(change < tol),
        )

    def get_density_field(self) -> np.ndarray:
        if self.densities is None:
            raise RuntimeError("尚未执行优化")
        return self.densities

    def export_image(self, path: str, dpi: int = 300) -> str:
        plot_density_field(self.densities, path, dpi=dpi)
        return path

    # ────────── 内部方法 ──────────

    def _element_stiffness(self) -> np.ndarray:
        """8-DOF 四节点平面应力单元刚度矩阵。"""
        E = 1.0
        nu = self.nu
        k = np.array([
            1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
            -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8,
        ])
        KE = E / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ])
        return KE

    def _setup_bc_and_loads(self, problem: dict[str, Any]) -> None:
        """解析边界条件和载荷，生成 DOF 索引。"""
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self._force = np.zeros(ndof)
        self._fixed_dofs = []

        # 解析边界条件
        for bc in problem.get("boundary_conditions", []):
            bc_type = bc.get("type", "fixed")
            location = bc.get("location", "")
            dofs = self._location_to_dofs(location, bc_type)
            self._fixed_dofs.extend(dofs)

        # 解析载荷
        for load in problem.get("loads", []):
            location = load.get("location", "")
            direction = load.get("direction", [0, -1])
            magnitude = load.get("magnitude", 1.0)
            node = self._location_to_node(location)
            if node is not None:
                self._force[2 * node] = direction[0] * magnitude
                self._force[2 * node + 1] = direction[1] * magnitude

        # 如果没有设置边界条件，使用默认悬臂梁设置
        if not self._fixed_dofs:
            # 左端全固定
            for i in range(self.nely + 1):
                self._fixed_dofs.extend([2 * i * (self.nelx + 1), 2 * i * (self.nelx + 1) + 1])

        if np.allclose(self._force, 0):
            # 默认：右端中点向下力
            node = (self.nely + 1) * self.nelx + self.nely // 2
            self._force[2 * node + 1] = -1.0

    def _location_to_node(self, location: str) -> Optional[int]:
        """将位置描述转换为节点索引。"""
        loc = location.lower().replace(" ", "_")
        nely, nelx = self.nely, self.nelx

        mapping = {
            "top_left": 0,
            "top_right": nelx,
            "bottom_left": (nely) * (nelx + 1),
            "bottom_right": (nely + 1) * (nelx + 1) - 1,
            "top_center": nelx // 2,
            "top_mid": nelx // 2,
            "bottom_center": nely * (nelx + 1) + nelx // 2,
            "right_center": (nely // 2) * (nelx + 1) + nelx,
            "right_mid": (nely // 2) * (nelx + 1) + nelx,
            "left_center": (nely // 2) * (nelx + 1),
        }

        for key, node in mapping.items():
            if key in loc:
                return node
        return None

    def _location_to_dofs(self, location: str, bc_type: str) -> list[int]:
        """将位置描述 + 类型转换为 DOF 列表。"""
        loc = location.lower().replace(" ", "_")
        dofs = []
        nely, nelx = self.nely, self.nelx

        if "left" in loc and "edge" in loc:
            for i in range(nely + 1):
                node = i * (nelx + 1)
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)
        elif "right" in loc and "edge" in loc:
            for i in range(nely + 1):
                node = i * (nelx + 1) + nelx
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)
        elif "bottom" in loc and "edge" in loc:
            for j in range(nelx + 1):
                node = nely * (nelx + 1) + j
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)
        elif "top" in loc and "edge" in loc:
            for j in range(nelx + 1):
                node = j
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)
        else:
            # 单点
            node = self._location_to_node(location)
            if node is not None:
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)

        return dofs

    def _fe_solve(self, x: np.ndarray) -> np.ndarray:
        """有限元求解：组装全局刚度矩阵并求解。"""
        nelx, nely = self.nelx, self.nely
        ndof = 2 * (nelx + 1) * (nely + 1)
        KE = self._ke

        # 组装全局刚度矩阵
        edof_mat = self._edof_matrix()
        iK = np.kron(edof_mat, np.ones((8, 1), dtype=int)).flatten()
        jK = np.kron(edof_mat, np.ones((1, 8), dtype=int)).flatten()

        # SIMP 材料插值
        sK = (
            (self.Emin + x.flatten() ** self.penal * (self.E0 - self.Emin))[:, np.newaxis, np.newaxis]
            * KE[np.newaxis, :, :]
        ).flatten()

        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

        # 施加边界条件
        all_dofs = set(range(ndof))
        fixed = set(self._fixed_dofs)
        free_dofs = sorted(all_dofs - fixed)

        f = self._force.copy()
        u = np.zeros(ndof)
        u[free_dofs] = spsolve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])

        return u

    def _edof_matrix(self) -> np.ndarray:
        """构建单元-DOF 映射矩阵。"""
        nelx, nely = self.nelx, self.nely
        edof = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = ely * (nelx + 1) + elx
                n2 = n1 + 1
                n3 = (ely + 1) * (nelx + 1) + elx
                n4 = n3 + 1
                edof[el] = [
                    2*n1, 2*n1+1, 2*n2, 2*n2+1,
                    2*n4, 2*n4+1, 2*n3, 2*n3+1,
                ]
        return edof

    def _element_compliance(self, u: np.ndarray) -> np.ndarray:
        """计算各单元柔度。"""
        edof = self._edof_matrix()
        KE = self._ke
        ce = np.zeros(self.nelx * self.nely)
        for i in range(len(edof)):
            ue = u[edof[i]]
            ce[i] = ue @ KE @ ue
        return ce.reshape(self.nely, self.nelx)

    def _density_filter(self, x: np.ndarray, dc: np.ndarray) -> np.ndarray:
        """灵敏度过滤（密度加权平均）。"""
        rmin = self.rmin
        nely, nelx = self.nely, self.nelx

        # 构建卷积核
        r = int(np.ceil(rmin))
        size = 2 * r + 1
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - r) ** 2 + (j - r) ** 2)
                kernel[i, j] = max(0, rmin - dist)

        dc_filtered = convolve(x * dc, kernel, mode="reflect")
        weight_sum = convolve(x, kernel, mode="reflect")
        weight_sum[weight_sum == 0] = 1e-9
        return dc_filtered / weight_sum

    def _oc_update(
        self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray
    ) -> np.ndarray:
        """OC (Optimality Criteria) 密度更新。"""
        move = 0.2
        l1, l2 = 0.0, 1e9

        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            x_new = np.maximum(
                0.001,
                np.maximum(
                    x - move,
                    np.minimum(
                        1.0,
                        np.minimum(
                            x + move,
                            x * np.sqrt(-dc / dv / lmid),
                        ),
                    ),
                ),
            )
            if np.mean(x_new) > self.volfrac:
                l1 = lmid
            else:
                l2 = lmid

        return x_new
