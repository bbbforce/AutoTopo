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
from jax_fem.mma import optimize

class TOElasticity(Problem):
    """用于拓扑优化的线弹性有限元问题类"""
    
    def custom_init(self):
        # 初始化 SIMP 材料和优化相关参数
        self.Emin = 1e-9
        self.Emax = 1.0
        self.nu = 0.3
        self.penal = 3.0

    def get_tensor_map(self):
        # 定义本构关系（平面应力各向同性弹性）
        def stress_fn(u_grad, theta):
            # theta 是当前高斯积分点上的物理密度
            E = self.Emin + (self.Emax - self.Emin) * (theta ** self.penal)
            nu = self.nu
            lmbda = E * nu / (1.0 - nu**2)
            mu = E / (2.0 * (1.0 + nu))
            
            # 应变 epsilon = 0.5 * (grad u + grad u^T)
            epsilon = 0.5 * (u_grad + u_grad.T)
            # 应力 sigma = lmbda * tr(epsilon) * I + 2 * mu * epsilon
            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(2) + 2.0 * mu * epsilon
            return sigma
        return stress_fn

    def get_surface_maps(self):
        # 右边界的 Neumann 力定义（向下的分布载荷）
        def surface_map(u, x):
            return jnp.array([0.0, -0.1])
        return [surface_map]

    def set_params(self, params):
        # 将单元密度 params (num_cells,) 广播到每个单元的每个高斯积分点 (num_cells, num_quads)
        self.internal_vars = [jnp.repeat(params[:, None], self.fes[0].num_quads, axis=1)]


def run_test():
    print("================== 开始进行 jax-fem 拓扑优化测试 ==================")
    
    # 1. 划分网格：60 x 20 悬臂梁，长度 60，高度 20
    Nx, Ny = 60, 20
    Lx, Ly = 60.0, 20.0
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')
    
    # 2. 定义边界条件 (左端固定)
    def left_boundary(point):
        return jnp.isclose(point[0], 0.0)

    dirichlet_bc_info = [
        [left_boundary, left_boundary], # 约束位置
        [0, 1],                         # 约束自由度 (x, y)
        [lambda p: 0.0, lambda p: 0.0]  # 约束值
    ]

    # 3. 定义 Neumann 力的作用面 (最右侧 x = Lx)
    def right_boundary(point):
        return jnp.isclose(point[0], Lx)

    location_fns = [right_boundary]

    # 4. 创建物理求解 Problem 实例
    problem = TOElasticity(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type='QUAD4',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns
    )

    # 5. 封装可微的前向求解器
    # 使用 umfpack_solver (scipy) 加速线性系统求解
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}})

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
    # 动态附加 flex_inds 以满足 jax-fem 内置 mma 过滤器的计算需求
    problem.fes[0].flex_inds = np.arange(num_cells)
    
    optimizationParams = {
        'movelimit': 0.2,
        'maxIters': 50  # 增加到 50 代
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

    # 将 src 路径添加到 sys.path 以导入可视化工具
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from autotopo.utils.visualization import plot_density_field
        rho_2d = rho_opt.reshape((Nx, Ny)).T
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topopt_result.png')
        plot_density_field(rho_2d, img_path, dpi=300)
        print(f"\n拓扑优化高清图已成功保存至: {img_path}")
    except Exception as e:
        print(f"\n保存高清图失败: {e}")

    # 10. 打印简单的 ASCII 艺术图以直观查看拓扑结构
    print("\n直观拓扑优化密度分布示意图 (用 ASCII 字符表示):")
    rho_2d = rho_opt.reshape((Nx, Ny)).T  # 还原成 2D 形状并转置以供打印
    # 从上往下打印行
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
    run_test()
