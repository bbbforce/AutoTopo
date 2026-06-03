"""JAX-FEM 仿真引擎适配层。

基于 Niels Aage & Villads Egede Johansen 的 165 行 Python 拓扑优化代码
(DTU TopOpt Group, 2013) 实现 2D SIMP 最小柔度拓扑优化。

参考文献:
  - Efficient topology optimization in MATLAB using 88 lines of code,
    E. Andreassen, A. Clausen, M. Schevenels, B.S. Lazarov, O. Sigmund,
    Struct Multidisc Optim, 43(1), 1-16, 2011.

核心计算使用 NumPy/SciPy，后续可切换为 JAX 后端。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from autotopo.engines.base import OptResult, SolveResult, TopoEngine
from autotopo.utils.visualization import plot_density_field


class JaxFemEngine(TopoEngine):
    """2D SIMP 拓扑优化引擎（基于 Aage 165 行代码）。"""

    def __init__(self) -> None:
        # 网格参数
        self.nelx: int = 150
        self.nely: int = 50
        self.volfrac: float = 0.5
        self.penal: float = 3.0
        self.rmin: float = 6.0
        self.ft: int = 1  # 0=灵敏度过滤, 1=密度过滤（默认），2=Heaviside投影

        # Heaviside 投影参数（ft=2 时生效）
        # self.beta: float = 1.0          # 初始 β
        # self.beta_max: float = 32.0     # 最大 β
        # self.beta_interval: int = 40    # 每隔多少次迭代翻倍
        # self.eta: float = 0.5           # 投影阈值

        # 材料参数
        self.Emin: float = 1e-9
        self.Emax: float = 1.0
        self.nu: float = 0.3

        # 内部状态（setup 后初始化）
        self.densities: Optional[np.ndarray] = None
        self._ke: Optional[np.ndarray] = None
        self._edof_mat: Optional[np.ndarray] = None
        self._iK: Optional[np.ndarray] = None
        self._jK: Optional[np.ndarray] = None
        self._H: Optional[Any] = None
        self._Hs: Optional[np.ndarray] = None
        self._fixed_dofs: np.ndarray = np.array([], dtype=int)
        self._free_dofs: np.ndarray = np.array([], dtype=int)
        self._force: Optional[np.ndarray] = None

    # ────────── 接口实现 ──────────

    def setup(self, problem: dict[str, Any]) -> None:
        domain = problem.get("domain", {})
        material = problem.get("material", {})

        self.nelx = domain.get("nelx", 60)
        self.nely = domain.get("nely", 20)
        self.Emax = material.get("youngs_modulus", 1.0)
        self.nu = material.get("poissons_ratio", 0.3)

        # 从约束中提取体积分数
        for c in problem.get("constraints", []):
            if c.get("type") == "volume_fraction":
                self.volfrac = c.get("value", 0.5)

        # 从参数中提取优化设置
        params = problem.get("parameters", {})
        self.penal = params.get("penal", 3.0)
        self.rmin = params.get("rmin", 1.5)
        self.ft = params.get("ft", 1)

        # Heaviside 投影参数
        self.beta = params.get("beta", 1.0)
        self.beta_max = params.get("beta_max", 32.0)
        self.beta_interval = params.get("beta_interval", 40)
        self.eta = params.get("eta", 0.5)

        # 构建核心数据结构
        self._ke = self._element_stiffness()
        self._build_edof_matrix()
        self._build_filter_matrix()

        # 设置边界条件和载荷
        self._setup_bc_and_loads(problem)

        # 初始化密度场
        self.densities = np.full(self.nely * self.nelx, self.volfrac)

    def optimize(
        self,
        *,
        max_iter: int = 2000,
        tol: float = 0.01,
        penal: Optional[float] = None,
        rmin: Optional[float] = None,
        volfrac: Optional[float] = None,
        ft: Optional[int] = None,
        beta: Optional[float] = None,
        beta_max: Optional[float] = None,
        beta_interval: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> OptResult:
        if penal is not None:
            self.penal = penal
        if rmin is not None:
            self.rmin = rmin
            self._build_filter_matrix()  # rmin 变化需要重建过滤矩阵
        if volfrac is not None:
            self.volfrac = volfrac
        if ft is not None:
            self.ft = ft
        if beta is not None:
            self.beta = beta
        if beta_max is not None:
            self.beta_max = beta_max
        if beta_interval is not None:
            self.beta_interval = beta_interval
        if eta is not None:
            self.eta = eta

        if self.densities is None:
            raise RuntimeError("请先调用 setup() 初始化引擎")

        nelx, nely = self.nelx, self.nely
        ndof = 2 * (nelx + 1) * (nely + 1)
        KE = self._ke
        edofMat = self._edof_mat
        iK, jK = self._iK, self._jK
        H, Hs = self._H, self._Hs
        free = self._free_dofs

        # 设计变量
        x = self.densities.copy()
        xold = x.copy()
        xPhys = x.copy()
        g = 0  # Nguyen/Paulino OC 累积量

        # Heaviside 投影参数（从实例属性读取，允许 Agent 调整）
        beta = self.beta
        beta_max = self.beta_max
        beta_interval = self.beta_interval
        eta = self.eta

        compliance_history: list[float] = []
        volume_history: list[float] = []
        change = 1.0
        loop = 0

        while change > tol and loop < max_iter:
            loop += 1

            # ── β-continuation (ft=2 时) ──
            if self.ft == 2 and loop > 1 and loop % beta_interval == 0 and beta < beta_max:
                beta = min(2 * beta, beta_max)

            # ── FE 求解 ──
            sK = (
                (KE.flatten()[np.newaxis]).T
                * (self.Emin + xPhys ** self.penal * (self.Emax - self.Emin))
            ).flatten(order='F')
            K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            K = K[free, :][:, free]

            f = self._force.copy()
            u = np.zeros((ndof, 1))
            u[free, 0] = spsolve(K, f[free, 0])

            # ── 目标函数 & 灵敏度 ──
            ce = (
                np.dot(u[edofMat].reshape(nelx * nely, 8), KE)
                * u[edofMat].reshape(nelx * nely, 8)
            ).sum(1)
            obj = float(((self.Emin + xPhys ** self.penal * (self.Emax - self.Emin)) * ce).sum())
            dc = (-self.penal * xPhys ** (self.penal - 1) * (self.Emax - self.Emin)) * ce
            dv = np.ones(nely * nelx)

            # ── 灵敏度/密度过滤 ──
            if self.ft == 0:
                dc[:] = np.asarray(
                    (H * (x * dc))[np.newaxis].T / Hs
                )[:, 0] / np.maximum(0.001, x)
            elif self.ft == 1:
                dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
                dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
            elif self.ft == 2:
                # Heaviside: 先密度过滤，再链式法则修正灵敏度
                xTilde = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
                dxPhys = self._heaviside_derivative(xTilde, beta, eta)
                dc[:] = np.asarray(H * (dc[np.newaxis].T * dxPhys[np.newaxis].T / Hs))[:, 0]
                dv[:] = np.asarray(H * (dv[np.newaxis].T * dxPhys[np.newaxis].T / Hs))[:, 0]

            # ── OC 更新 ──
            xold[:] = x
            x[:], g = self._oc_update(x, dc, dv, g)

            # ── 物理密度更新 ──
            if self.ft == 0:
                xPhys[:] = x
            elif self.ft == 1:
                xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
            elif self.ft == 2:
                xTilde = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
                xPhys[:] = self._heaviside_projection(xTilde, beta, eta)

            # 收敛判断
            change = float(np.linalg.norm(
                x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf
            ))

            compliance_history.append(obj)
            volume_history.append(float(np.mean(xPhys)))

            extra = f", beta: {beta:.0f}" if self.ft == 2 else ""
            print(
                f"it.: {loop:4d}, obj.: {obj:.3f}, "
                f"Vol.: {volume_history[-1]:.3f}, ch.: {change:.3f}{extra}"
            )

        self.densities = xPhys.copy()
        return OptResult(
            densities=xPhys.reshape(nely, nelx, order='F'),
            compliance_history=compliance_history,
            volume_history=volume_history,
            iterations=loop,
            converged=(change <= tol),
        )

    def get_density_field(self) -> np.ndarray:
        if self.densities is None:
            raise RuntimeError("尚未执行优化")
        d = self.densities
        if d.ndim == 1:
            return d.reshape(self.nely, self.nelx, order='F')
        return d

    def export_image(self, path: str, dpi: int = 300) -> str:
        plot_density_field(self.get_density_field(), path, dpi=dpi)
        return path

    # ────────── 内部方法 ──────────

    @staticmethod
    def _element_stiffness(nu: float = 0.3) -> np.ndarray:
        """8-DOF 四节点平面应力单元刚度矩阵（与 Aage lk() 一致）。"""
        E = 1
        k = np.array([
            1/2 - nu/6,   1/8 + nu/8,  -1/4 - nu/12, -1/8 + 3*nu/8,
            -1/4 + nu/12, -1/8 - nu/8,  nu/6,         1/8 - 3*nu/8,
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

    def _build_edof_matrix(self) -> None:
        """构建单元-DOF 映射矩阵（列优先编号，与 Aage 一致）。"""
        nelx, nely = self.nelx, self.nely
        edofMat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array([
                    2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                    2*n2,   2*n2+1, 2*n1,   2*n1+1,
                ])
        self._edof_mat = edofMat
        self._iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        self._jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    def _build_filter_matrix(self) -> None:
        """构建稀疏过滤矩阵 H（与 Aage 一致，替换 convolve）。"""
        nelx, nely = self.nelx, self.nely
        rmin = self.rmin
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc += 1
        self._H = coo_matrix(
            (sH[:cc], (iH[:cc], jH[:cc])),
            shape=(nelx * nely, nelx * nely),
        ).tocsc()
        self._Hs = self._H.sum(1)

    def _setup_bc_and_loads(self, problem: dict[str, Any]) -> None:
        """解析边界条件和载荷，生成 DOF 索引。"""
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self._force = np.zeros((ndof, 1)) 
        fixed_dofs: list[int] = []

        # 解析边界条件
        for bc in problem.get("boundary_conditions", []):
            bc_type = bc.get("type", "fixed")
            location = bc.get("location", "")
            dofs = self._location_to_dofs(location, bc_type)
            fixed_dofs.extend(dofs)

        # 解析载荷
        for load in problem.get("loads", []):
            location = load.get("location", "")
            direction = load.get("direction", [0, -1])
            magnitude = load.get("magnitude", 1.0)
            node = self._location_to_node(location)
            if node is not None:
                self._force[2 * node, 0] = direction[0] * magnitude
                self._force[2 * node + 1, 0] = direction[1] * magnitude

        # 如果没有设置边界条件，使用默认 half-MBB 梁设置
        if not fixed_dofs:
            dofs = np.arange(ndof)
            # 左边 x 方向固定（每隔一个 dof 取 x 分量） + 右下角 y 方向固定
            fixed_dofs = list(np.union1d(
                dofs[0:2*(self.nely+1):2],
                np.array([ndof - 1]),
            ))

        if np.allclose(self._force, 0):
            # 默认：左上角向下力 f[1] = -1（half-MBB 梁）
            self._force[1, 0] = -1.0

        all_dofs = np.arange(ndof)
        self._fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
        self._free_dofs = np.setdiff1d(all_dofs, self._fixed_dofs)

    def _location_to_node(self, location: str) -> Optional[int]:
        """将位置描述转换为节点索引（列优先编号）。"""
        loc = location.lower().replace(" ", "_")
        nely, nelx = self.nely, self.nelx

        # 列优先编号：node = (nely+1)*col + row
        mapping = {
            "top_left": 0,                                          # col=0, row=0
            "top_right": (nely + 1) * nelx,                        # col=nelx, row=0
            "bottom_left": nely,                                    # col=0, row=nely
            "bottom_right": (nely + 1) * (nelx + 1) - 1,          # col=nelx, row=nely
            "top_center": (nely + 1) * (nelx // 2),               # col=nelx/2, row=0
            "top_mid": (nely + 1) * (nelx // 2),
            "bottom_center": (nely + 1) * (nelx // 2) + nely,     # col=nelx/2, row=nely
            "right_center": (nely + 1) * nelx + nely // 2,        # col=nelx, row=nely/2
            "right_mid": (nely + 1) * nelx + nely // 2,
            "left_center": nely // 2,                               # col=0, row=nely/2
        }

        for key, node in mapping.items():
            if key in loc:
                return node
        return None

    def _location_to_dofs(self, location: str, bc_type: str) -> list[int]:
        """将位置描述 + 类型转换为 DOF 列表（列优先编号）。"""
        loc = location.lower().replace(" ", "_")
        dofs: list[int] = []
        nely, nelx = self.nely, self.nelx

        if "left" in loc and "edge" in loc:
            # 左边列 col=0 的所有节点
            for row in range(nely + 1):
                node = row  # (nely+1)*0 + row
                if bc_type in ("fixed", "fixed_x", "symmetry", "roller"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)
        elif "right" in loc and "edge" in loc:
            for row in range(nely + 1):
                node = (nely + 1) * nelx + row
                if bc_type in ("fixed", "fixed_x", "symmetry", "roller"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y"):
                    dofs.append(2 * node + 1)
        elif "bottom" in loc and "edge" in loc:
            for col in range(nelx + 1):
                node = (nely + 1) * col + nely
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y", "symmetry", "roller"):
                    dofs.append(2 * node + 1)
        elif "top" in loc and "edge" in loc:
            for col in range(nelx + 1):
                node = (nely + 1) * col
                if bc_type in ("fixed", "fixed_x"):
                    dofs.append(2 * node)
                if bc_type in ("fixed", "fixed_y", "symmetry", "roller"):
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

    @staticmethod
    def _heaviside_projection(
        xTilde: np.ndarray, beta: float, eta: float = 0.5
    ) -> np.ndarray:
        """Heaviside 平滑投影函数。

        将过滤后的密度场投影到接近 0-1 的分布：
        x̃_bar = (tanh(β·η) + tanh(β·(x̃ - η))) / (tanh(β·η) + tanh(β·(1 - η)))
        """
        return (
            np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))
        ) / (
            np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        )

    @staticmethod
    def _heaviside_derivative(
        xTilde: np.ndarray, beta: float, eta: float = 0.5
    ) -> np.ndarray:
        """Heaviside 投影函数对 xTilde 的导数（链式法则）。

        dx̃_bar/dx̃ = β·(1 - tanh(β·(x̃ - η))²) / (tanh(β·η) + tanh(β·(1 - η)))
        """
        return (
            beta * (1.0 - np.tanh(beta * (xTilde - eta)) ** 2)
        ) / (
            np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        )

    def _oc_update(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        g: float,
    ) -> tuple[np.ndarray, float]:
        """OC (Optimality Criteria) 密度更新（完整 Nguyen/Paulino 方法）。"""
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        xnew = np.zeros_like(x)

        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # 钳位保护：防止 dc 为正或 dv 为零时 sqrt 参数为负
            Be = np.maximum(1e-12, -dc / np.maximum(dv, 1e-12) / lmid)
            xnew[:] = np.maximum(
                0.0,
                np.maximum(
                    x - move,
                    np.minimum(
                        1.0,
                        np.minimum(
                            x + move,
                            x * np.sqrt(Be),
                        ),
                    ),
                ),
            )
            gt = g + np.sum(dv * (xnew - x))
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid

        return xnew, gt
