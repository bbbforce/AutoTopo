import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# 启用 x64 精度（JAX-FEM 要求的）
from jax import config
config.update("jax_enable_x64", True)

from jax_fem.generate_mesh import rectangle_mesh, Mesh
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
import jax_fem.mma
from jax_fem.mma import optimize

import scipy.spatial
from jax.experimental.sparse import BCOO

# 🛠️ 猴子补丁 (Monkey Patch)：重写 mma 的过滤矩阵计算，以支持自定义过滤半径 rmin = 4.0
def patched_compute_filter_kd_tree(fe):
    cell_centroids = np.mean(np.take(fe.points, fe.cells, axis=0), axis=1)
    flex_num_cells = len(fe.flex_inds)
    flex_cell_centroids = np.take(cell_centroids, fe.flex_inds, axis=0)
    
    # 设定用户指定的 rmin = 4.0
    rmin = 4.0
    print(f"[Patch] 成功应用猴子补丁，强制设置过滤半径 rmin = {rmin}")
    
    kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
    I = []
    J = []
    V = []
    # rmin=4.0 时，2D 范围内圆面积约 pi * 4^2 ≈ 50，设为 80 邻居以确保完全覆盖
    num_nbs = 80
    for i in range(flex_num_cells):
        dd, ii = kd_tree.query(flex_cell_centroids[i], num_nbs)
        vals = np.where(rmin - dd > 0., rmin - dd, 0.)
        I += [i]*num_nbs
        J += ii.tolist()
        V += vals.tolist()
        
    H_sp = scipy.sparse.csc_array((V, (I, J)), shape=(flex_num_cells, flex_num_cells))
    H = BCOO.from_scipy_sparse(H_sp).sort_indices()
    Hs = H.sum(1).todense()
    return H, Hs

# 替换 mma 模块中的原函数
jax_fem.mma.compute_filter_kd_tree = patched_compute_filter_kd_tree


class TOElasticity(Problem):
    """用于半对称 MBB 梁拓扑优化的线弹性有限元问题类"""
    
    def custom_init(self, Emin, Emax, nu, penal):
        self.Emin = Emin
        self.Emax = Emax
        self.nu = nu
        self.penal = penal

    def get_tensor_map(self):
        # 二维平面应力各向同性弹性本构
        def stress_fn(u_grad, theta):
            E = self.Emin + (self.Emax - self.Emin) * (theta ** self.penal)
            nu = self.nu
            lmbda = E * nu / (1.0 - nu**2)
            mu = E / (2.0 * (1.0 + nu))
            
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(2) + 2.0 * mu * epsilon
            return sigma
        return stress_fn

    def get_surface_maps(self):
        # 左上角承受向下集中力 (施加在最左上角 1.0 宽度内，总合力为 -1.0)
        def surface_map(u, x):
            return jnp.array([0.0, -1.0])
        return [surface_map]

    def set_params(self, params):
        self.internal_vars = [jnp.repeat(params[:, None], self.fes[0].num_quads, axis=1)]


def run_mbb():
    print("================== 开始进行 JAX-FEM 半对称 MBB 梁拓扑优化 ==================")
    
    # 1. 划分网格：150 x 50
    Nx, Ny = 150, 50
    Lx, Ly = 150.0, 50.0
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')
    
    # 2. 定义边界条件 (Dirichlet BC)
    # 对称边界 (x = 0) 限制 u_x = 0
    def symmetry_boundary(point):
        return jnp.isclose(point[0], 0.0)

    # 右下角垂直约束 (x = Lx, y = 0) 限制 u_y = 0
    def right_bottom_corner(point):
        return jnp.isclose(point[0], Lx, atol=1e-3) & jnp.isclose(point[1], 0.0, atol=1e-3)

    dirichlet_bc_info = [
        [symmetry_boundary, right_bottom_corner], # 约束位置
        [0, 1],                                   # 约束自由度 (x, y)
        [lambda p: 0.0, lambda p: 0.0]            # 约束值
    ]

    # 3. 定义 Neumann 力的作用面 (左上角 y = Ly 且 x <= 1.0 处的单元边面)
    def load_location(point):
        return jnp.isclose(point[1], Ly, atol=1e-3) & (point[0] <= 1.0)

    location_fns = [load_location]

    # 4. 创建 Problem 实例
    problem = TOElasticity(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type='QUAD4',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        additional_info=(1e-9, 1.0, 0.3, 3.0) # Emin, Emax, nu, penal
    )

    # 5. 封装可微的前向求解器
    fwd_pred = ad_wrapper(
        problem, 
        solver_options={'umfpack_solver': {}}, 
        adjoint_solver_options={'umfpack_solver': {}}
    )

    # 6. 计算外力向量 (利用 u=0 时的残差)
    volfrac = 0.5
    rho_ini = np.full((problem.fes[0].num_cells, 1), volfrac)
    problem.set_params(rho_ini.flatten())
    
    zero_sol = [jnp.zeros((fe.num_total_nodes, fe.vec)) for fe in problem.fes]
    res_list_0 = problem.compute_residual(zero_sol)
    F_ext = [-res for res in res_list_0]

    # 7. 定义 Compliance 目标函数与 JAX 自动微分
    def compliance_val(theta):
        sol_list = fwd_pred(theta)
        comp = 0.
        for f, u in zip(F_ext, sol_list):
            comp += jnp.sum(f * u)
        return comp

    def obj_and_grad(rho):
        val, grad = jax.value_and_grad(compliance_val)(rho)
        return val, grad

    # 8. 定义目标函数和约束函数句柄
    def objectiveHandle(rho):
        val, grad = obj_and_grad(rho)
        return val, grad

    def consHandle(rho, loop):
        vc = np.array([np.mean(rho) - volfrac])
        dvc = np.ones((1, len(rho), 1)) / len(rho)
        return vc, dvc

    # 9. 配置优化参数并执行 MMA 优化
    num_cells = problem.fes[0].num_cells
    problem.fes[0].flex_inds = np.arange(num_cells)
    
    optimizationParams = {
        'movelimit': 0.2,
        'maxIters': 400  # 用户指定的 max_iter = 400
    }

    print(f"网格单元数: {num_cells}, 初始体积分数: {volfrac}")
    print("启动 MMA 优化器...")
    
    rho_opt = optimize(
        fe=problem.fes[0],
        rho_ini=rho_ini,
        optimizationParams=optimizationParams,
        objectiveHandle=objectiveHandle,
        consHandle=consHandle,
        numConstraints=1
    )
    
    print("\n优化执行成功！")
    print("部分优化后的单元密度 field (前 10 个):")
    print(rho_opt[:10])

    # 10. 保存图片
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from autotopo.utils.visualization import plot_density_field
        rho_2d = rho_opt.reshape((Nx, Ny)).T
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mbb_topopt_result.png')
        plot_density_field(rho_2d, img_path, dpi=300)
        print(f"\n拓扑优化高清图已成功保存至: {img_path}")
    except Exception as e:
        print(f"\n保存高清图失败: {e}")

    # 11. 打印简单的 ASCII 艺术图以直观查看拓扑结构
    print("\n直观拓扑优化密度分布示意图 (用 ASCII 字符表示):")
    rho_2d = rho_opt.reshape((Nx, Ny)).T
    for r in range(Ny - 1, -1, -1):
        row_str = ""
        for c in range(Nx):
            val = rho_2d[r, c]
            if val > 0.7:
                row_str += "█"
            elif val > 0.4:
                row_str += "▒"
            else:
                row_str += " "
        print(row_str)

if __name__ == "__main__":
    run_mbb()
