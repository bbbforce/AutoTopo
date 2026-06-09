#!/usr/bin/env python3
"""FEniCS + dolfin-adjoint 独立求解脚本。

此脚本运行在 dolfin-adjoint Docker 容器内，不依赖 autotopo 包。
通过命令行接收问题定义 JSON，执行拓扑优化，输出结果。

用法:
    python3 solver_runner.py --input problem.json --output-dir /tmp/result

输入 (problem.json):
    标准 AutoTopo 问题定义 + 优化参数

输出 (output-dir/):
    result.json          优化结果 (收敛历史、迭代次数等)
    density.png          密度场分布图
    convergence.png      收敛历史曲线
    density_grid.npy     密度场 numpy 数组 (规则网格采样)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
import types

import numpy as np

# ─── FEniCS 导入 ───
import dolfin
import dolfin_adjoint as da
import gmsh
import meshio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def patch_function_riesz(func):
    """Add the missing Riesz conversion hook for this container's Function type."""

    def _ad_convert_riesz(self, gradient, riesz_map=None):
        if riesz_map not in (None, "L2", "l2"):
            raise ValueError(f"Unsupported Riesz map for density control: {riesz_map}")

        out = dolfin.Function(self.function_space())
        if hasattr(gradient, "vector"):
            out.vector()[:] = gradient.vector()
        else:
            out.vector()[:] = np.asarray(gradient, dtype=float)
        return out

    func._ad_convert_riesz = types.MethodType(_ad_convert_riesz, func)
    return func


# ════════════════════════════════════════════════════════════════
#  网格生成
# ════════════════════════════════════════════════════════════════

def generate_mesh(width, height, mesh_resolution, non_design_regions=None):
    """使用 Gmsh 生成 2D 三角形网格并转为 DOLFIN Mesh。"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("domain")

    # 矩形设计域
    rect = gmsh.model.occ.addRectangle(0, 0, 0, width, height)

    # 非设计域挖孔
    if non_design_regions:
        holes = []
        for region in non_design_regions:
            x0 = region.get("x_min", 0)
            y0 = region.get("y_min", 0)
            w = region.get("x_max", 0) - x0
            h = region.get("y_max", 0) - y0
            if w > 0 and h > 0:
                hole = gmsh.model.occ.addRectangle(x0, y0, 0, w, h)
                holes.append((2, hole))
        if holes:
            gmsh.model.occ.cut([(2, rect)], holes)

    gmsh.model.occ.synchronize()

    # 网格尺寸控制
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_resolution * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_resolution)

    # 物理组
    surfaces = gmsh.model.getEntities(dim=2)
    surface_tags = [s[1] for s in surfaces]
    gmsh.model.addPhysicalGroup(2, surface_tags, tag=1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    # 生成网格
    gmsh.model.mesh.generate(2)

    # 导出临时 msh 文件
    tmp_dir = tempfile.mkdtemp(prefix="autotopo_mesh_")
    msh_path = os.path.join(tmp_dir, "mesh.msh")
    xml_path = os.path.join(tmp_dir, "mesh.xml")
    gmsh.write(msh_path)
    gmsh.finalize()

    # meshio 转换为 DOLFIN XML
    msh = meshio.read(msh_path)
    triangle_cells = None
    for cell_block in msh.cells:
        if cell_block.type == "triangle":
            triangle_cells = cell_block.data
            break

    if triangle_cells is None:
        raise RuntimeError("Gmsh 未生成三角形单元")

    points_2d = msh.points[:, :2]
    mesh_for_dolfin = meshio.Mesh(
        points=points_2d,
        cells=[("triangle", triangle_cells)],
    )
    meshio.write(xml_path, mesh_for_dolfin, file_format="dolfin-xml")

    # 使用 dolfin-adjoint 重载的网格，确保带注释的求解操作可安全地
    # 在 pyadjoint 记录链中登记网格依赖关系
    mesh = da.Mesh(xml_path)
    print(f"  网格生成完成: {mesh.num_vertices()} 节点, {mesh.num_cells()} 单元")
    return mesh


# ════════════════════════════════════════════════════════════════
#  边界子域
# ════════════════════════════════════════════════════════════════

def make_boundary(location, W, H, tol):
    """创建 dolfin.SubDomain 边界标记。"""
    loc = location.lower().replace(" ", "_")

    if "left" in loc and "edge" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < tol
        return Sub()

    elif "right" in loc and "edge" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0] - W) < tol
        return Sub()

    elif "bottom" in loc and "edge" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < tol
        return Sub()

    elif "top" in loc and "edge" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[1] - H) < tol
        return Sub()

    elif "bottom_left" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < tol and x[1] < tol
        return Sub()

    elif "bottom_right" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0] - W) < tol and x[1] < tol
        return Sub()

    elif "top_left" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < tol and abs(x[1] - H) < tol
        return Sub()

    elif "top_right" in loc:
        class Sub(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0] - W) < tol and abs(x[1] - H) < tol
        return Sub()

    return None


def location_to_point(location, W, H):
    """将位置描述转换为坐标 (x, y)。"""
    loc = location.lower().replace(" ", "_")
    mapping = {
        "top_left": (0.0, H),
        "top_right": (W, H),
        "bottom_left": (0.0, 0.0),
        "bottom_right": (W, 0.0),
        "top_center": (W / 2, H),
        "top_mid": (W / 2, H),
        "bottom_center": (W / 2, 0.0),
        "right_center": (W, H / 2),
        "right_mid": (W, H / 2),
        "left_center": (0.0, H / 2),
    }
    for key, point in mapping.items():
        if key in loc:
            return point
    return None


# ════════════════════════════════════════════════════════════════
#  求解核心
# ════════════════════════════════════════════════════════════════

def solve_topology_optimization(problem, output_dir):
    """执行 SIMP 拓扑优化 (FEniCS + dolfin-adjoint)。"""

    # ── 解析参数 ──
    domain = problem.get("domain", {})
    material = problem.get("material", {})
    params = problem.get("parameters", {})

    width = domain.get("width", 60.0)
    height = domain.get("height", 20.0)
    mesh_resolution = domain.get("mesh_resolution", 1.0)
    non_design_regions = domain.get("non_design_regions", [])

    # 向后兼容 nelx/nely
    if "mesh_resolution" not in domain and "nelx" in domain:
        mesh_resolution = width / domain["nelx"]

    E0 = material.get("youngs_modulus", 1.0)
    Emin = 1e-9
    nu = material.get("poissons_ratio", 0.3)

    penal = params.get("penal", 3.0)
    rmin = params.get("rmin", 0.05)
    max_iter = params.get("max_iter", 200)
    tol = params.get("tol", 1e-6)
    optimizer = params.get("optimizer", "SLSQP")
    if optimizer.upper() == "L-BFGS-B":
        print("  L-BFGS-B 不支持体积约束，自动切换为 SLSQP")
        optimizer = "SLSQP"

    volfrac = params.get("volfrac", 0.5)
    for c in problem.get("constraints", []):
        if c.get("type") == "volume_fraction":
            volfrac = c.get("value", 0.5)

    # Helmholtz 过滤半径 (绝对值)
    R = rmin * max(width, height)

    # ── 生成网格 ──
    mesh = generate_mesh(width, height, mesh_resolution, non_design_regions)

    # ── 边界条件 ──
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    bcs = []
    W_dim = width
    H_dim = height
    bc_tol = 1e-10 * max(W_dim, H_dim)

    for bc_def in problem.get("boundary_conditions", []):
        bc_type = bc_def.get("type", "fixed")
        location = bc_def.get("location", "")
        subdomain = make_boundary(location, W_dim, H_dim, bc_tol)
        if subdomain is None:
            continue

        if bc_type == "fixed":
            bcs.append(da.DirichletBC(V, da.Constant((0.0, 0.0)), subdomain))
        elif bc_type == "fixed_x":
            bcs.append(da.DirichletBC(V.sub(0), da.Constant(0.0), subdomain))
        elif bc_type == "fixed_y":
            bcs.append(da.DirichletBC(V.sub(1), da.Constant(0.0), subdomain))
        elif bc_type in ("symmetry", "roller"):
            if "left" in location.lower() or "right" in location.lower():
                bcs.append(da.DirichletBC(V.sub(0), da.Constant(0.0), subdomain))
            else:
                bcs.append(da.DirichletBC(V.sub(1), da.Constant(0.0), subdomain))

    # 默认 BC: half-MBB
    if not bcs:
        left = make_boundary("left_edge", W_dim, H_dim, bc_tol)
        bcs.append(da.DirichletBC(V.sub(0), da.Constant(0.0), left))
        br = make_boundary("bottom_right", W_dim, H_dim, bc_tol)
        bcs.append(da.DirichletBC(V.sub(1), da.Constant(0.0), br))

    # ── 载荷 ──
    loads = []
    for load_def in problem.get("loads", []):
        location = load_def.get("location", "")
        direction = load_def.get("direction", [0, -1])
        magnitude = load_def.get("magnitude", 1.0)
        point = location_to_point(location, W_dim, H_dim)
        if point is not None:
            loads.append({
                "point": point,
                "force": [direction[0] * magnitude, direction[1] * magnitude],
            })

    if not loads:
        loads = [{"point": (0.0, H_dim), "force": [0.0, -1.0]}]

    # ══════════════════════════════════════════════════════════
    #  FEniCS + dolfin-adjoint 优化
    # ══════════════════════════════════════════════════════════

    # 重置 tape
    da.set_working_tape(da.Tape())

    # 函数空间
    W_space = dolfin.FunctionSpace(mesh, "CG", 1)   # 密度

    # 设计变量
    rho = patch_function_riesz(da.Function(W_space, name="Density"))
    rho.vector()[:] = volfrac

    # Helmholtz 过滤
    rho_f = dolfin.TrialFunction(W_space)
    w = dolfin.TestFunction(W_space)

    a_helm = (R**2 * dolfin.inner(dolfin.grad(rho_f), dolfin.grad(w)) * dolfin.dx
              + rho_f * w * dolfin.dx)
    L_helm = rho * w * dolfin.dx

    rho_tilde = da.Function(W_space, name="FilteredDensity")
    da.solve(a_helm == L_helm, rho_tilde)

    # SIMP 插值
    E = Emin + rho_tilde**penal * (E0 - Emin)

    # 平面应力本构
    lmbda = E * nu / ((1.0 + nu) * (1.0 - nu))
    mu = E / (2.0 * (1.0 + nu))

    def epsilon(u):
        return dolfin.sym(dolfin.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * dolfin.tr(epsilon(u)) * dolfin.Identity(2)

    # 弹性力学变分问题
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    a = dolfin.inner(sigma(u), epsilon(v)) * dolfin.dx

    # 载荷形式 (Gaussian 近似点力)
    L_form = dolfin.dot(da.Constant((0.0, 0.0)), v) * dolfin.dx
    for load in loads:
        px, py = load["point"]
        fx, fy = load["force"]
        sig = mesh_resolution * 1.5
        gauss = da.Expression(
            "exp(-((x[0]-px)*(x[0]-px) + (x[1]-py)*(x[1]-py)) / (2*s*s)) / (2*pi*s*s)",
            px=px, py=py, s=sig, pi=np.pi, degree=2,
        )
        f_vec = dolfin.as_vector([da.Constant(fx), da.Constant(fy)])
        L_form = L_form + gauss * dolfin.dot(f_vec, v) * dolfin.dx

    u_sol = da.Function(V, name="Displacement")
    da.solve(a == L_form, u_sol, bcs)

    # 目标函数 (柔度)
    J = da.assemble(dolfin.inner(sigma(u_sol), epsilon(u_sol)) * dolfin.dx)

    # 体积约束
    total_volume = da.assemble(da.Constant(1.0) * dolfin.dx(mesh))

    class VolumeConstraint(da.InequalityConstraint):
        def __init__(self, vf, vol):
            self.vf = float(vf)
            self.vol = float(vol)

        @staticmethod
        def _values(m):
            if isinstance(m, (list, tuple)):
                m = m[0]
            if hasattr(m, "vector"):
                return m.vector().get_local()
            return np.asarray(m, dtype=float)

        def function(self, m):
            values = self._values(m)
            return [self.vf - float(np.mean(values))]

        def jacobian(self, m):
            values = self._values(m)
            return [-np.ones_like(values, dtype=float) / values.size]

        def output_workspace(self):
            return [0.0]

        def length(self):
            return 1

    # 收敛历史
    compliance_history = []
    volume_history = []
    iter_count = [0]
    latest_density_values = [rho.vector().get_local().copy()]

    def eval_cb(j, rho_vals):
        iter_count[0] += 1
        compliance_history.append(float(j))
        rho_func = rho_vals[0] if isinstance(rho_vals, (list, tuple)) else rho_vals
        if hasattr(rho_func, "vector"):
            values = rho_func.vector().get_local()
        else:
            values = np.asarray(rho_func, dtype=float)
        latest_density_values[0] = values.copy()
        vol_frac = float(np.mean(values))
        volume_history.append(vol_frac)
        if iter_count[0] % 10 == 0 or iter_count[0] == 1:
            print(f"  it.: {iter_count[0]:4d}, obj.: {j:.4f}, vol.: {vol_frac:.4f}")

    # 简化泛函
    control = da.Control(rho, riesz_map="L2")
    Jhat = da.ReducedFunctional(J, control, eval_cb_post=eval_cb)

    # 优化
    print(f"\n  开始优化: optimizer={optimizer}, max_iter={max_iter}")
    converged = True
    try:
        rho_opt = da.minimize(
            Jhat,
            method=optimizer,
            bounds=(0.0, 1.0),
            constraints=VolumeConstraint(volfrac, float(total_volume)),
            options={"maxiter": max_iter, "ftol": tol, "disp": True},
        )
    except Exception as exc:
        if "Iteration limit reached" not in str(exc):
            raise

        converged = False
        print(f"  优化达到迭代上限，导出当前可用设计: {exc}")
        rho_opt = da.Function(W_space, name="DensityIterationLimit")
        rho_opt.vector()[:] = latest_density_values[0]

    # ══════════════════════════════════════════════════════════
    #  输出结果
    # ══════════════════════════════════════════════════════════

    os.makedirs(output_dir, exist_ok=True)

    # 密度场投影到规则网格
    nx = max(int(width / mesh_resolution), 60)
    ny = max(int(height / mesh_resolution), 20)
    density_grid = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * width / nx
            y = height - (j + 0.5) * height / ny
            try:
                density_grid[j, i] = rho_opt(dolfin.Point(x, y))
            except RuntimeError:
                density_grid[j, i] = 0.0

    # 保存密度场 numpy
    np.save(os.path.join(output_dir, "density_grid.npy"), density_grid)

    # 绘制密度场图
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(density_grid, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    ax.set_title("Topology Optimization Result", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "density.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 绘制收敛历史图
    if compliance_history:
        iterations = range(1, len(compliance_history) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color1 = "#2563eb"
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Compliance", color=color1, fontsize=12)
        ax1.plot(iterations, compliance_history, color=color1, linewidth=1.5)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#dc2626"
        ax2.set_ylabel("Volume Fraction", color=color2, fontsize=12)
        ax2.plot(iterations, volume_history, color=color2, linewidth=1.5, linestyle="--")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.suptitle("Convergence History", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "convergence.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 结果 JSON
    result = {
        "iterations": iter_count[0],
        "converged": converged,
        "compliance_history": compliance_history,
        "volume_history": volume_history,
        "mesh_info": {
            "num_cells": mesh.num_cells(),
            "num_vertices": mesh.num_vertices(),
            "width": width,
            "height": height,
        },
        "files": {
            "density_image": "density.png",
            "convergence_image": "convergence.png",
            "density_grid": "density_grid.npy",
        },
    }

    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  结果已保存到: {output_dir}")
    return result


# ════════════════════════════════════════════════════════════════
#  CLI 入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FEniCS + dolfin-adjoint 拓扑优化求解器")
    parser.add_argument("--input", required=True, help="问题定义 JSON 文件路径")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    args = parser.parse_args()

    # 读取问题定义
    with open(args.input, "r") as f:
        problem = json.load(f)

    print("=" * 60)
    print("FEniCS + dolfin-adjoint 拓扑优化求解器")
    print("=" * 60)

    try:
        result = solve_topology_optimization(problem, args.output_dir)
        print("\n✅ 求解完成!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 求解失败: {e}")
        traceback.print_exc()
        # 写入错误信息
        os.makedirs(args.output_dir, exist_ok=True)
        error_result = {"error": str(e), "traceback": traceback.format_exc()}
        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
