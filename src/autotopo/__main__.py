"""AutoTopo CLI 入口。

用法:
    python -m autotopo "设计一个悬臂梁..."
    python -m autotopo --engine-only --nelx 60 --nely 20
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autotopo",
        description="AutoTopo: 自主拓扑优化 AI 智能体工作流",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── run: 完整 AI 工作流 ──
    run_parser = subparsers.add_parser("run", help="运行完整的 AI 工作流")
    run_parser.add_argument("prompt", type=str, help="问题描述文本")
    run_parser.add_argument("--images", nargs="*", default=[], help="设计域示意图路径")
    run_parser.add_argument("--output", default="./output", help="输出目录")
    run_parser.add_argument("--max-retries", type=int, default=3, help="视觉评估最大重试次数")
    run_parser.add_argument("--provider", default=None, help="LLM Provider (openai/deepseek/glm/bailian)")

    # ── solve: 纯引擎求解（不依赖 LLM）──
    solve_parser = subparsers.add_parser("solve", help="直接运行 SIMP 引擎（无需 LLM）")
    solve_parser.add_argument("--preset", choices=["cantilever", "mbb", "bridge"], default="cantilever",
                              help="预设问题")
    solve_parser.add_argument("--nelx", type=int, default=60, help="x方向单元数")
    solve_parser.add_argument("--nely", type=int, default=20, help="y方向单元数")
    solve_parser.add_argument("--volfrac", type=float, default=0.5, help="体积分数")
    solve_parser.add_argument("--penal", type=float, default=3.0, help="SIMP 罚因子")
    solve_parser.add_argument("--rmin", type=float, default=1.5, help="过滤半径")
    solve_parser.add_argument("--max-iter", type=int, default=200, help="最大迭代数")
    solve_parser.add_argument("--output", default="./output", help="输出目录")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        _run_workflow(args)
    elif args.command == "solve":
        _run_engine(args)


def _run_workflow(args: argparse.Namespace) -> None:
    """运行完整 AI 工作流。"""
    from autotopo.graph import compile_graph

    # 每次运行保存在以当前时间戳命名的子目录下
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(Path(args.output) / f"run_{timestamp}")

    print("🚀 AutoTopo AI 工作流启动")
    print(f"   问题: {args.prompt[:80]}...")
    print(f"   输出: {output_path}")

    app = compile_graph()
    initial_state = {
        "user_input": args.prompt,
        "image_paths": args.images,
        "max_retries": args.max_retries,
        "output_path": output_path,
        "iteration": 0,
        "history": [],
    }

    result = app.invoke(initial_state)

    print("\n✅ 工作流完成!")
    print(f"   结果图: {result.get('result_image_path', 'N/A')}")
    print(f"   输出目录: {result.get('output_path', args.output)}")

    if result.get("evaluation", {}).get("has_defects"):
        print(f"   ⚠️ 仍存在缺陷: {result['evaluation'].get('defect_types', [])}")
    else:
        print("   质量评估: 合格")


def _run_engine(args: argparse.Namespace) -> None:
    """直接运行引擎求解（不依赖 LLM）。"""
    from autotopo.engines.jax_fem_engine import JaxFemEngine

    presets = {
        "cantilever": {
            "domain": {"nelx": args.nelx, "nely": args.nely},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [{"type": "fixed", "location": "left_edge"}],
            "loads": [{"type": "point_force", "location": "right_center",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": args.volfrac}],
            "parameters": {"penal": args.penal, "rmin": args.rmin},
        },
        "mbb": {
            "domain": {"nelx": args.nelx, "nely": args.nely},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [
                {"type": "fixed_x", "location": "left_edge"},
                {"type": "fixed_y", "location": "bottom_right"},
            ],
            "loads": [{"type": "point_force", "location": "top_left",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": args.volfrac}],
            "parameters": {"penal": args.penal, "rmin": args.rmin},
        },
        "bridge": {
            "domain": {"nelx": args.nelx, "nely": args.nely},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [
                {"type": "fixed", "location": "bottom_left"},
                {"type": "fixed_y", "location": "bottom_right"},
            ],
            "loads": [{"type": "point_force", "location": "top_center",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": args.volfrac}],
            "parameters": {"penal": args.penal, "rmin": args.rmin},
        },
    }

    problem = presets[args.preset]
    print(f"🔧 SIMP 引擎求解: {args.preset}")
    print(f"   网格: {args.nelx}×{args.nely}, volfrac={args.volfrac}, penal={args.penal}")

    engine = JaxFemEngine()
    engine.setup(problem)
    result = engine.optimize(max_iter=args.max_iter, volfrac=args.volfrac,
                             penal=args.penal, rmin=args.rmin)

    output_dir = Path(args.output) / f"solve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(output_dir / f"{args.preset}_result.png")
    engine.export_image(img_path)

    print(f"\n✅ 优化完成!")
    print(f"   迭代次数: {result.iterations}")
    print(f"   收敛: {result.converged}")
    print(f"   最终柔度: {result.compliance_history[-1]:.4f}")
    print(f"   结果图: {img_path}")


if __name__ == "__main__":
    main()
