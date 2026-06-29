"""Python SIMP + MMA 结构化网格后端。

该后端从 old_engine.py 的 NumPy/SciPy SIMP 实现重构而来，保留 FE 装配、
过滤矩阵、边界条件和载荷映射逻辑；设计变量更新改为本地 MMA-style 更新器。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings

from autotopo.engines.base import OptResult, TopoEngine
from autotopo.engines.mma import MMAState, initialize_mma_state, mma_update


class PythonSimpMMAEngine(TopoEngine):
    """2D 结构化 SIMP/MMA 拓扑优化引擎。"""

    def __init__(self) -> None:
        self.nelx = 30
        self.nely = 10
        self.volfrac = 0.5
        self.penal = 3.0
        self.rmin = 1.5
        self.ft = 1
        self.Emin = 1e-9
        self.Emax = 1.0
        self.nu = 0.3

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
        self._passive_void_mask_flat: np.ndarray = np.array([], dtype=bool)
        self._active_mask_flat: np.ndarray = np.array([], dtype=bool)
        self._history_rows: list[dict[str, float | int | bool]] = []
        self._last_result: Optional[OptResult] = None

    def setup(self, problem: dict[str, Any]) -> None:
        """根据结构化问题初始化网格、FE 矩阵、过滤器和载荷。"""

        domain = problem.get("domain", {})
        material = problem.get("material", {})
        self.nelx = int(domain.get("nelx", max(2, round(domain.get("width", 30)))))
        self.nely = int(domain.get("nely", max(2, round(domain.get("height", 10)))))
        self.Emax = float(material.get("youngs_modulus", 1.0))
        self.nu = float(material.get("poissons_ratio", 0.3))

        for constraint in problem.get("constraints", []):
            if constraint.get("type") == "volume_fraction":
                self.volfrac = float(constraint.get("value", 0.5))

        params = problem.get("parameters", {})
        self.penal = float(params.get("penal", 3.0))
        self.rmin = float(params.get("rmin", 1.5))
        self.ft = int(params.get("ft", 1))

        self._passive_void_mask_flat = self._parse_passive_void_mask(domain)
        self._active_mask_flat = ~self._passive_void_mask_flat

        self._ke = self._element_stiffness(self.nu)
        self._build_edof_matrix()
        self._build_filter_matrix()
        self._setup_bc_and_loads(problem)

        x0 = np.full(self.nely * self.nelx, self.volfrac, dtype=float)
        x0[self._passive_void_mask_flat] = 0.0
        self.densities = x0
        self._history_rows = []
        self._last_result = None

    def optimize(
        self,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        penal: Optional[float] = None,
        rmin: Optional[float] = None,
        volfrac: Optional[float] = None,
        ft: Optional[int] = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **_: Any,
    ) -> OptResult:
        """执行 SIMP/MMA 优化。"""

        if self.densities is None:
            raise RuntimeError("请先调用 setup() 初始化引擎")
        if penal is not None:
            self.penal = float(penal)
        if rmin is not None and float(rmin) != self.rmin:
            self.rmin = float(rmin)
            self._build_filter_matrix()
        if volfrac is not None:
            self.volfrac = float(volfrac)
        if ft is not None:
            self.ft = int(ft)

        start = time.perf_counter()
        nelx, nely = self.nelx, self.nely
        ndof = 2 * (nelx + 1) * (nely + 1)
        KE = self._ke
        edof_mat = self._edof_mat
        iK, jK = self._iK, self._jK
        H, Hs = self._H, self._Hs
        if KE is None or edof_mat is None or iK is None or jK is None or H is None or Hs is None:
            raise RuntimeError("引擎内部矩阵尚未初始化")

        x = self.densities.copy()
        xold = x.copy()
        xPhys = self._physical_density(x)
        mma_state: MMAState = initialize_mma_state(x)
        active = self._active_mask_flat
        passive = self._passive_void_mask_flat

        compliance_history: list[float] = []
        volume_history: list[float] = []
        change = 1.0
        loop = 0

        while change > tol and loop < max_iter:
            loop += 1
            sK = (
                (KE.flatten()[np.newaxis]).T
                * (self.Emin + xPhys ** self.penal * (self.Emax - self.Emin))
            ).flatten(order="F")
            K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            Kff = K[self._free_dofs, :][:, self._free_dofs]

            f = self._force.copy() if self._force is not None else np.zeros((ndof, 1))
            u = np.zeros((ndof, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                u[self._free_dofs, 0] = spsolve(Kff, f[self._free_dofs, 0])
            if not np.all(np.isfinite(u)):
                raise RuntimeError("singular stiffness matrix produced non-finite displacement")

            ce = (
                np.dot(u[edof_mat].reshape(nelx * nely, 8), KE)
                * u[edof_mat].reshape(nelx * nely, 8)
            ).sum(1)
            obj = float(((self.Emin + xPhys ** self.penal * (self.Emax - self.Emin)) * ce).sum())
            if not np.isfinite(obj):
                raise RuntimeError("compliance_nan_or_inf")

            dc = (-self.penal * xPhys ** (self.penal - 1) * (self.Emax - self.Emin)) * ce
            dv = np.ones(nely * nelx)
            dc[passive] = 0.0
            dv[passive] = 0.0

            if self.ft == 0:
                dc = (H @ (x * dc) / Hs) / np.maximum(0.001, x)
            else:
                dc = H @ (dc / Hs)
                dv = H @ (dv / Hs)
            dc = np.asarray(dc).reshape(-1)
            dv = np.asarray(dv).reshape(-1)
            dc[passive] = 0.0
            dv[passive] = 0.0

            xold[:] = x
            x, mma_state = mma_update(
                x,
                dc,
                dv,
                volfrac=self.volfrac,
                state=mma_state,
                active_mask=active,
                passive_void_mask=passive,
            )
            xPhys = self._physical_density(x)
            change = float(np.linalg.norm(x - xold, ord=np.inf))
            active_volume = float(np.mean(xPhys[active])) if np.any(active) else 0.0
            compliance_history.append(obj)
            volume_history.append(active_volume)
            self._history_rows.append(
                {
                    "iteration": loop,
                    "compliance": obj,
                    "volume": active_volume,
                    "change": change,
                    "converged": change <= tol,
                }
            )
            print(
                f"it.: {loop:4d}, obj.: {obj:.3f}, "
                f"Vol.: {active_volume:.3f}, ch.: {change:.3f}, opt.: MMA"
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "iteration": loop,
                        "max_iter": max_iter,
                        "compliance": obj,
                        "volume": active_volume,
                        "change": change,
                        "converged": change <= tol,
                        "tol": tol,
                        "optimizer": "MMA",
                    }
                )

        self.densities = xPhys.copy()
        result = OptResult(
            densities=xPhys.reshape(nely, nelx, order="F"),
            compliance_history=compliance_history,
            volume_history=volume_history,
            iterations=loop,
            converged=(change <= tol),
            extra={
                "optimizer": "MMA",
                "optimizer_fallback": None,
                "timings": {"optimization": time.perf_counter() - start},
            },
            mesh_info={
                "nelx": nelx,
                "nely": nely,
                "active_elements": int(np.sum(active)),
                "passive_void_elements": int(np.sum(passive)),
            },
        )
        self._last_result = result
        return result

    def get_density_field(self) -> np.ndarray:
        """返回二维密度场。"""

        if self.densities is None:
            raise RuntimeError("尚未执行优化")
        d = self.densities
        if d.ndim == 1:
            return d.reshape(self.nely, self.nelx, order="F")
        return d

    def export_image(self, path: str, dpi: int = 300) -> str:
        """导出密度图。"""

        from autotopo.utils.visualization import plot_density_field

        plot_density_field(self.get_density_field(), path, dpi=dpi, show_axes=False)
        return path

    def save_outputs(self, output_dir: str | Path, result: OptResult | None = None) -> dict[str, str]:
        """保存后端标准产物。"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result = result or self._last_result
        if result is None:
            raise RuntimeError("没有可保存的优化结果")

        density_npy = output_path / "density.npy"
        density_png = output_path / "density.png"
        history_csv = output_path / "optimization_history.csv"
        history_png = output_path / "optimization_history.png"
        result_json = output_path / "result.json"

        density = self.get_density_field()
        np.save(density_npy, density)
        from autotopo.utils.visualization import plot_density_field

        plot_density_field(
            density,
            str(density_png),
            title="Continuous Density Field",
            dpi=120,
            show_axes=False,
        )

        with history_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["iteration", "compliance", "volume", "change", "converged"])
            writer.writeheader()
            writer.writerows(self._history_rows)

        from autotopo.utils.visualization import plot_convergence_history

        plot_convergence_history(
            result.compliance_history,
            result.volume_history,
            str(history_png),
            title="Optimization History",
            dpi=120,
        )

        payload = {
            "optimizer": result.extra.get("optimizer", "MMA"),
            "optimizer_fallback": result.extra.get("optimizer_fallback"),
            "iterations": result.iterations,
            "converged": result.converged,
            "compliance_history": result.compliance_history,
            "volume_history": result.volume_history,
            "mesh_info": result.mesh_info,
            "files": {
                "density": density_npy.name,
                "density_image": density_png.name,
                "optimization_history": history_csv.name,
                "optimization_history_image": history_png.name,
            },
        }
        result_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "density": str(density_npy),
            "density_image": str(density_png),
            "optimization_history": str(history_csv),
            "optimization_history_image": str(history_png),
            "result_json": str(result_json),
        }

    @staticmethod
    def _element_stiffness(nu: float = 0.3) -> np.ndarray:
        """8-DOF 四节点平面应力单元刚度矩阵。"""

        E = 1.0
        k = np.array([
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ])
        return E / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ])

    def _build_edof_matrix(self) -> None:
        """构建单元到 DOF 的映射矩阵。"""

        edof_mat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
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
        self._edof_mat = edof_mat
        self._iK = np.kron(edof_mat, np.ones((8, 1))).flatten()
        self._jK = np.kron(edof_mat, np.ones((1, 8))).flatten()

    def _build_filter_matrix(self) -> None:
        """构建稀疏密度过滤矩阵。"""

        nfilter = int(self.nelx * self.nely * ((2 * (np.ceil(self.rmin) - 1) + 1) ** 2))
        iH = np.zeros(max(nfilter, 1))
        jH = np.zeros(max(nfilter, 1))
        sH = np.zeros(max(nfilter, 1))
        cc = 0
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - (np.ceil(self.rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(self.rmin), self.nelx))
                ll1 = int(np.maximum(j - (np.ceil(self.rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(self.rmin), self.nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * self.nely + l
                        fac = self.rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc += 1
        self._H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(self.nelx * self.nely, self.nelx * self.nely)).tocsc()
        self._Hs = np.asarray(self._H.sum(1)).reshape(-1)
        self._Hs[self._Hs <= 0] = 1.0

    def _setup_bc_and_loads(self, problem: dict[str, Any]) -> None:
        """解析边界条件和载荷，生成固定/自由 DOF。"""

        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self._force = np.zeros((ndof, 1))
        fixed_dofs: list[int] = []

        for bc in problem.get("boundary_conditions", []):
            dofs = self._location_to_dofs(bc.get("location", ""), bc.get("type", "fixed"))
            if not dofs:
                raise ValueError(f"invalid boundary condition location: {bc.get('location', '')}")
            fixed_dofs.extend(dofs)

        for load in problem.get("loads", []):
            node = self._location_to_node(load.get("location", ""))
            if node is None:
                raise ValueError(f"invalid load location: {load.get('location', '')}")
            direction = load.get("direction", [0, -1])
            magnitude = float(load.get("magnitude", 1.0))
            self._force[2 * node, 0] += float(direction[0]) * magnitude
            self._force[2 * node + 1, 0] += float(direction[1]) * magnitude

        if not fixed_dofs:
            raise ValueError("invalid boundary condition: no support")
        if np.allclose(self._force, 0):
            raise ValueError("invalid load: no nonzero load")

        all_dofs = np.arange(ndof)
        self._fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)
        self._free_dofs = np.setdiff1d(all_dofs, self._fixed_dofs)

    def _location_to_node(self, location: str) -> Optional[int]:
        """将位置描述转换为结构化网格节点索引。"""

        loc = location.lower().replace(" ", "_").replace("_corner", "")
        mapping = {
            "top_left": 0,
            "top_right": (self.nely + 1) * self.nelx,
            "bottom_left": self.nely,
            "bottom_right": (self.nely + 1) * (self.nelx + 1) - 1,
            "top_center": (self.nely + 1) * (self.nelx // 2),
            "top_mid": (self.nely + 1) * (self.nelx // 2),
            "bottom_center": (self.nely + 1) * (self.nelx // 2) + self.nely,
            "right_center": (self.nely + 1) * self.nelx + self.nely // 2,
            "right_mid": (self.nely + 1) * self.nelx + self.nely // 2,
            "left_center": self.nely // 2,
        }
        for key, node in mapping.items():
            if key in loc:
                return node
        return None

    def _location_to_dofs(self, location: str, bc_type: str) -> list[int]:
        """将位置描述和边界类型转换为 DOF 列表。"""

        loc = location.lower().replace(" ", "_").replace("_corner", "")
        dofs: list[int] = []
        if "left" in loc and "edge" in loc:
            nodes = [row for row in range(self.nely + 1)]
        elif "right" in loc and "edge" in loc:
            nodes = [(self.nely + 1) * self.nelx + row for row in range(self.nely + 1)]
        elif "bottom" in loc and "edge" in loc:
            nodes = [(self.nely + 1) * col + self.nely for col in range(self.nelx + 1)]
        elif "top" in loc and "edge" in loc:
            nodes = [(self.nely + 1) * col for col in range(self.nelx + 1)]
        else:
            node = self._location_to_node(location)
            nodes = [] if node is None else [node]

        for node in nodes:
            if bc_type in ("fixed", "fixed_x", "symmetry", "roller"):
                dofs.append(2 * node)
            if bc_type in ("fixed", "fixed_y"):
                dofs.append(2 * node + 1)
            if bc_type == "roller" and ("bottom" in loc or "top" in loc):
                dofs.append(2 * node + 1)
        return dofs

    def _parse_passive_void_mask(self, domain: dict[str, Any]) -> np.ndarray:
        """读取被动空洞单元 mask。"""

        raw = domain.get("passive_void_mask")
        if raw is None:
            return np.zeros(self.nely * self.nelx, dtype=bool)
        mask = np.asarray(raw, dtype=bool)
        if mask.shape != (self.nely, self.nelx):
            raise ValueError(f"passive_void_mask shape mismatch: expected {(self.nely, self.nelx)}, got {mask.shape}")
        return mask.flatten(order="F")

    def _physical_density(self, x: np.ndarray) -> np.ndarray:
        """应用过滤并强制被动空洞为 0。"""

        if self.ft == 0:
            xPhys = x.copy()
        else:
            xPhys = np.asarray(self._H @ x / self._Hs).reshape(-1)
        xPhys[self._passive_void_mask_flat] = 0.0
        return np.clip(xPhys, 0.0, 1.0)
