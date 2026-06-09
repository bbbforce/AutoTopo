"""AutoTopo CLI 入口。

用法:
    python -m autotopo run "设计一个悬臂梁..."
    python -m autotopo solve --preset cantilever --mesh-res 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _load_config() -> dict:
    for p in [Path("config/settings.yaml"), Path(__file__).resolve().parents[2] / "config" / "settings.yaml"]:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    return {}


def _profile_config(name: str) -> dict:
    config = _load_config()
    return dict(config.get("engine", {}).get("profiles", {}).get(name, {}))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autotopo",
        description="AutoTopo: 自主拓扑优化 AI 智能体工作流 (FEniCS + dolfin-adjoint)",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── run: 完整 AI 工作流 ──
    run_parser = subparsers.add_parser("run", help="运行完整的 AI 工作流")
    run_parser.add_argument("prompt", type=str, help="问题描述文本")
    run_parser.add_argument("--images", nargs="*", default=[], help="设计域示意图路径")
    run_parser.add_argument("--output", default="./output", help="输出目录")
    run_parser.add_argument("--max-retries", type=int, default=2, help="预览阶段视觉评估最大重试次数")
    run_parser.add_argument(
        "--solve-profile",
        choices=["preview_refine", "final_only", "preview_only"],
        default="preview_refine",
        help="求解模式：预览后精修、只精修或只预览",
    )
    run_parser.add_argument("--provider", default=None, help="LLM Provider (openai/deepseek/glm/bailian)")

    # ── solve: 纯引擎求解（不依赖 LLM）──
    solve_parser = subparsers.add_parser("solve", help="直接运行 FEniCS 引擎（无需 LLM）")
    solve_parser.add_argument("--preset", choices=["cantilever", "mbb", "bridge"], default="cantilever",
                              help="预设问题")
    solve_parser.add_argument("--profile", choices=["preview", "final"], default="final", help="求解 profile")
    solve_parser.add_argument("--mesh-res", type=float, default=None, help="Gmsh 网格特征尺寸")
    solve_parser.add_argument("--volfrac", type=float, default=0.5, help="体积分数")
    solve_parser.add_argument("--penal", type=float, default=3.0, help="SIMP 罚因子")
    solve_parser.add_argument("--rmin", type=float, default=0.05, help="Helmholtz 过滤半径比例")
    solve_parser.add_argument("--max-iter", type=int, default=None, help="最大迭代数")
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

    print("🚀 AutoTopo AI 工作流启动 (FEniCS + dolfin-adjoint)")
    print(f"   问题: {args.prompt[:80]}...")
    print(f"   求解模式: {args.solve_profile}")
    print(f"   输出: {output_path}")

    initial_stage = "final" if args.solve_profile == "final_only" else "preview"
    app = compile_graph()
    initial_state = {
        "user_input": args.prompt,
        "image_paths": args.images,
        "max_retries": args.max_retries,
        "solve_profile": args.solve_profile,
        "solve_stage": initial_stage,
        "final_refine_done": args.solve_profile != "preview_refine",
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
    """直接运行 FEniCS 引擎求解（不依赖 LLM）。"""
    from autotopo.engines.dolfin_adjoint_engine import DolfinAdjointEngine

    config = _load_config()
    engine_cfg = config.get("engine", {})
    defaults = engine_cfg.get("default_params", {})
    profile = _profile_config(args.profile)
    mesh_res = args.mesh_res if args.mesh_res is not None else profile.get(
        "mesh_resolution", defaults.get("mesh_resolution", 1.0)
    )
    max_iter = args.max_iter if args.max_iter is not None else profile.get(
        "max_iter", defaults.get("max_iter", 200)
    )
    tol = profile.get("tol", defaults.get("tol", 1e-6))
    output_dpi = profile.get("output_dpi", 300)

    presets = {
        "cantilever": {
            "domain": {"width": 60.0, "height": 20.0, "mesh_resolution": mesh_res},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [{"type": "fixed", "location": "left_edge"}],
            "loads": [{"type": "point_force", "location": "right_center",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": args.volfrac}],
            "parameters": {"penal": args.penal, "rmin": args.rmin,
                           "max_iter": max_iter, "tol": tol, "optimizer": "SLSQP",
                           "output_dpi": output_dpi, "solve_stage": args.profile},
        },
        "mbb": {
            "domain": {"width": 60.0, "height": 20.0, "mesh_resolution": mesh_res},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [
                {"type": "fixed_x", "location": "left_edge"},
                {"type": "fixed_y", "location": "bottom_right"},
            ],
            "loads": [{"type": "point_force", "location": "top_left",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": args.volfrac}],
            "parameters": {"penal": args.penal, "rmin": args.rmin,
                           "max_iter": max_iter, "tol": tol, "optimizer": "SLSQP",
                           "output_dpi": output_dpi, "solve_stage": args.profile},
        },
        "bridge": {
            "domain": {"width": 60.0, "height": 20.0, "mesh_resolution": mesh_res},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [
                {"type": "fixed", "location": "bottom_left"},
                {"type": "fixed_y", "location": "bottom_right"},
            ],
            "loads": [{"type": "point_force", "location": "top_center",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": args.volfrac}],
            "parameters": {"penal": args.penal, "rmin": args.rmin,
                           "max_iter": max_iter, "tol": tol, "optimizer": "SLSQP",
                           "output_dpi": output_dpi, "solve_stage": args.profile},
        },
    }

    problem = presets[args.preset]
    early_stop_cfg = engine_cfg.get("early_stop", {})
    if early_stop_cfg:
        problem["early_stop"] = dict(early_stop_cfg)
    problem["solve_stage"] = args.profile
    print(f"🔧 FEniCS + dolfin-adjoint 引擎求解: {args.preset}")
    print(f"   profile={args.profile}, 网格分辨率: {mesh_res}, volfrac={args.volfrac}, penal={args.penal}")

    engine = DolfinAdjointEngine()
    engine.setup(problem)
    result = engine.optimize(max_iter=max_iter, tol=tol, volfrac=args.volfrac,
                             penal=args.penal, rmin=args.rmin)

    output_dir = Path(args.output) / f"solve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(output_dir / f"{args.preset}_result.png")
    convergence_path = str(output_dir / f"{args.preset}_convergence.png")
    result_json_path = output_dir / "result.json"
    engine.export_image(img_path, dpi=output_dpi)
    engine.get_convergence_image(convergence_path)

    result_payload = {
        "solve_stage": result.extra.get("solve_stage", args.profile),
        "iterations": result.iterations,
        "converged": result.converged,
        "early_stopped": result.extra.get("early_stopped", False),
        "timings": result.extra.get("timings", {}),
        "compliance_history": result.compliance_history,
        "volume_history": result.volume_history,
        "mesh_info": result.mesh_info,
        "files": {
            "density_image": Path(img_path).name,
            "convergence_image": Path(convergence_path).name,
        },
    }
    result_json_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✅ 优化完成!")
    print(f"   迭代次数: {result.iterations}")
    print(f"   收敛: {result.converged}")
    if result.compliance_history:
        print(f"   最终柔度: {result.compliance_history[-1]:.4f}")
    print(f"   结果图: {img_path}")
    print(f"   收敛图: {convergence_path}")
    print(f"   结果 JSON: {result_json_path}")


if __name__ == "__main__":
    main()
