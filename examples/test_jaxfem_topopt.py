"""jax-fem 2D 悬臂梁拓扑优化验证脚本。

使用 jax-fem 的 Problem + ad_wrapper 进行 FEM 求解与伴随法自动灵敏度计算，
外部实现 SIMP 插值 + 密度过滤 + OC 更新器。

工作流:
  1. 定义 Elasticity(Problem) 子类：
     - get_tensor_map: SIMP 本构 (σ = E(ρ) * ε)
     - get_surface_maps: Neumann BC (面力)
     - set_params: 密度参数化
  2. 用 ad_wrapper 包装为可微求解器 fwd_pred(density) -> sol_list
  3. 预计算外力向量 f_ext = -R(u=0)
  4. 定义柔度目标 J(ρ) = f_ext^T · u(ρ)，用 jax.value_and_grad 自动微分
  5. 密度过滤 + OC 更新迭代
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import Mesh, rectangle_mesh, get_meshio_cell_type


# ──────────── 1. 线弹性 Problem（SIMP 插值 + Neumann BC）────────────

class Elasticity(Problem):
    """2D 平面应力线弹性问题，SIMP 密度参数化 + Neumann BC。"""

    def custom_init(self):
        self.Emax = 1.0
        self.Emin = 1e-9
        self.nu = 0.3
        self.penal = 3.0

    def get_tensor_map(self):
        """本构关系：平面应力 + SIMP 插值。"""
        Emax, Emin, nu, penal = self.Emax, self.Emin, self.nu, self.penal

        def stress_fn(u_grad, theta):
            E = Emin + theta[0] ** penal * (Emax - Emin)
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11, eps22, eps12 = epsilon[0, 0], epsilon[1, 1], epsilon[0, 1]
            coeff = E / (1 - nu * nu)
            sig11 = coeff * (eps11 + nu * eps22)
            sig22 = coeff * (nu * eps11 + eps22)
            sig12 = coeff * (1 - nu) / 2 * eps12 * 2
            return jnp.array([[sig11, sig12], [sig12, sig22]])

        return stress_fn

    def get_surface_maps(self):
        """Neumann BC: 面力 t(u, x) = [0, -1] 在加载面上。

        返回值需与 location_fns 一一对应。
        location_fns[0] 标识加载面 -> surface_maps[0] 给出对应的面力。
        """
        def traction(u, x):
            return jnp.array([0.0, -1.0])

        return [traction]

    def set_params(self, params):
        """params: (num_cells,) -> internal_vars: [(num_cells, num_quads, 1)]"""
        nq = self.fes[0].num_quads
        self.internal_vars = [jnp.repeat(params[:, None, None], nq, axis=1)]


# ──────────── 2. 过滤 + OC 工具 ────────────

def build_filter(nelx, nely, rmin):
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH, jH, sH = np.zeros(nfilter), np.zeros(nfilter), np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k in range(max(int(i - np.ceil(rmin) + 1), 0), min(int(i + np.ceil(rmin)), nelx)):
                for ll in range(max(int(j - np.ceil(rmin) + 1), 0), min(int(j + np.ceil(rmin)), nely)):
                    iH[cc], jH[cc] = row, k * nely + ll
                    sH[cc] = max(0.0, rmin - np.sqrt((i - k) ** 2 + (j - ll) ** 2))
                    cc += 1
    H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(nelx * nely, nelx * nely)).tocsc()
    return H, H.sum(1)


def density_filter(H, Hs, x):
    return np.asarray(H * x[:, None] / Hs)[:, 0]


def oc_update(x, dc, dv, volfrac, move=0.2):
    l1, l2 = 0.0, 1e9
    xnew = np.empty_like(x)
    while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        Be = np.maximum(1e-12, -dc / np.maximum(dv, 1e-12) / lmid)
        xnew[:] = np.clip(x * np.sqrt(Be), x - move, x + move).clip(0, 1)
        l1, l2 = (lmid, l2) if np.sum(dv * (xnew - x)) > 0 else (l1, lmid)
    return xnew


# ──────────── 3. 主流程 ────────────

def main():
    nelx, nely = 60, 30
    volfrac, penal, rmin = 0.5, 3.0, 1.5
    max_iter, tol = 10, 0.01
    Lx, Ly = float(nelx), float(nely)

    print(f"🔧 jax-fem TopOpt 验证: {nelx}×{nely}, vf={volfrac}")

    # 网格
    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nelx, Ny=nely, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    num_cells = len(mesh.cells)
    print(f"   节点: {len(mesh.points)}, 单元: {num_cells}")

    # 边界条件
    # Dirichlet: 左边全固定 (u_x = u_y = 0)
    def left(pt):
        return jnp.isclose(pt[0], 0., atol=1e-5)

    dirichlet_bc_info = [[left, left], [0, 1], [lambda p: 0., lambda p: 0.]]

    # Neumann: 右侧中点区域施加向下力
    # location_fns 标识加载面（扁平列表），get_surface_maps 提供面力值
    def load_loc(pt):
        return jnp.logical_and(
            jnp.isclose(pt[0], Lx, atol=1e-5),
            jnp.isclose(pt[1], Ly / 2, atol=Ly / nely + 1e-5))

    location_fns = [load_loc]

    # 创建 Problem
    problem = Elasticity(mesh=mesh, vec=2, dim=2, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         location_fns=location_fns)
    problem.penal = penal
    print(f"   积分点/单元: {problem.fes[0].num_quads}")

    solver_opts = {'umfpack_solver': {}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_opts, adjoint_solver_options=solver_opts)

    # 预计算外力向量: f_ext = -R(u=0)，用全实体密度
    problem.set_params(jnp.full(num_cells, 1.0))
    zero_sol = [jnp.zeros((len(mesh.points), 2))]
    f_ext = -problem.compute_residual(zero_sol)[0]
    print(f"   |f_ext| = {float(jnp.linalg.norm(f_ext)):.4f}")

    # 目标函数 J(ρ) = f^T u(ρ)
    def J(density):
        sol = fwd_pred(density)
        return jnp.sum(f_ext * sol[0])

    J_and_dJ = jax.value_and_grad(J)

    # 过滤矩阵
    H, Hs = build_filter(nelx, nely, rmin)

    # 优化循环
    x = np.full(num_cells, volfrac)
    comp_hist, vol_hist = [], []
    change = 1.0

    print(f"\n{'='*60}")
    print(f"{'it':>5s} {'compliance':>12s} {'vol':>8s} {'change':>8s}")
    print(f"{'='*60}")
    t0 = time.time()

    for it in range(1, max_iter + 1):
        if change <= tol and it > 1:
            break

        xPhys = density_filter(H, Hs, x)
        obj, dc = J_and_dJ(jnp.array(xPhys))
        obj, dc = float(obj), np.array(dc)
        dv = np.ones(num_cells)

        # 过滤灵敏度
        dc = np.asarray(H * (dc[:, None] / Hs))[:, 0]
        dv = np.asarray(H * (dv[:, None] / Hs))[:, 0]

        xold = x.copy()
        x = oc_update(x, dc, dv, volfrac)
        change = float(np.linalg.norm(x - xold, np.inf))

        comp_hist.append(obj)
        vol_hist.append(float(np.mean(xPhys)))
        print(f"{it:5d} {obj:12.4f} {vol_hist[-1]:8.3f} {change:8.4f}", flush=True)

    print(f"{'='*60}")
    print(f"✅ 完成! 耗时 {time.time()-t0:.1f}s, {it} 次迭代")

    # 保存结果图
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xf = density_filter(H, Hs, x).reshape(nely, nelx)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(xf, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax.set_title(f"jax-fem SIMP ({nelx}x{nely}, vf={volfrac})")
    ax.set_aspect("equal"); plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "jaxfem_cantilever_test.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   结果图: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
