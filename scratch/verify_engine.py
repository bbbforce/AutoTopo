"""最小化验证脚本：直接测试 FEniCS + dolfin-adjoint 引擎。

不依赖 LangGraph / LangChain 等 AI 框架，只测试核心引擎逻辑。
"""
import sys
sys.path.insert(0, "/root/shared/AutoTopo/src")

print("=" * 60)
print("AutoTopo FEniCS 引擎验证")
print("=" * 60)

# 测试 1: 导入
print("\n[1/4] 导入引擎模块...")
from autotopo.engines.dolfin_adjoint_engine import DolfinAdjointEngine
print("  ✅ 导入成功")

# 测试 2: 悬臂梁 setup + optimize
print("\n[2/4] 悬臂梁 setup...")
problem = {
    "domain": {"width": 20.0, "height": 10.0, "mesh_resolution": 1.5},
    "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
    "boundary_conditions": [
        {"type": "fixed", "location": "left_edge"},
    ],
    "loads": [
        {"type": "point_force", "location": "right_center",
         "magnitude": 1.0, "direction": [0, -1]},
    ],
    "constraints": [
        {"type": "volume_fraction", "value": 0.5},
    ],
    "parameters": {"penal": 3.0, "rmin": 0.05, "optimizer": "L-BFGS-B"},
}

engine = DolfinAdjointEngine()
engine.setup(problem)
print("  ✅ setup 完成")

# 测试 3: 优化 (少量迭代)
print("\n[3/4] 运行优化 (max_iter=30)...")
result = engine.optimize(max_iter=30)
print(f"  ✅ 优化完成: {result.iterations} 次迭代")
print(f"     收敛: {result.converged}")
if result.compliance_history:
    print(f"     初始柔度: {result.compliance_history[0]:.4f}")
    print(f"     最终柔度: {result.compliance_history[-1]:.4f}")
if result.volume_history:
    print(f"     最终体积分数: {result.volume_history[-1]:.4f}")
print(f"     网格信息: {result.mesh_info}")

# 测试 4: 导出图片
print("\n[4/4] 导出结果图...")
import os
os.makedirs("/root/shared/AutoTopo/output", exist_ok=True)
img_path = engine.export_image("/root/shared/AutoTopo/output/verify_cantilever.png")
print(f"  ✅ 图片已保存: {img_path}")

print("\n" + "=" * 60)
print("🎉 所有验证通过!")
print("=" * 60)
