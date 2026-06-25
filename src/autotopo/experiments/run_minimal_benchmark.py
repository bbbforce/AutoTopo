"""运行最小多阶段拓扑优化智能体 benchmark。

用法:
    python -m autotopo.experiments.run_minimal_benchmark --quick
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from autotopo.engines.structured_benchmarks import minimal_benchmark_cases
from autotopo.research_graph import run_research_workflow
from autotopo.schemas import BenchmarkCaseResult, BenchmarkMethod


DEFAULT_BENCHMARK_OUTPUT = Path("output") / "minimal_benchmark"

SUMMARY_FIELDS = [
    "case_id",
    "benchmark_type",
    "method",
    "first_pass_success",
    "final_success",
    "repair_success",
    "repair_iterations",
    "execution_error_type",
    "detected_failure_modes",
    "compliance",
    "volume_error",
    "grayness_index",
    "checkerboard_score",
    "connectivity_score",
    "converged",
    "output_dir",
]


def _summary_row(result: BenchmarkCaseResult) -> dict[str, object]:
    data = result.model_dump(mode="json")
    data["detected_failure_modes"] = ",".join(data.get("detected_failure_modes", []))
    data["benchmark_type"] = result.benchmark_type.value
    data["method"] = result.method.value
    return {field: data.get(field, "") for field in SUMMARY_FIELDS}


def write_summary(results: list[BenchmarkCaseResult], output_dir: Path) -> None:
    """保存 summary.csv 和 summary.md。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(_summary_row(result))

    lines = [
        "# Minimal Benchmark Summary",
        "",
        f"- total_runs: {len(results)}",
        f"- final_success: {sum(1 for item in results if item.final_success)}",
        f"- repair_success: {sum(1 for item in results if item.repair_success)}",
        "",
        "| case_id | benchmark_type | method | first_pass_success | final_success | repair_iterations | failure_modes | compliance | volume_error | grayness_index | checkerboard_score | connectivity_score | converged |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        modes = ",".join(mode.value for mode in result.detected_failure_modes)
        lines.append(
            "| "
            + " | ".join([
                result.case_id,
                result.benchmark_type.value,
                result.method.value,
                str(result.first_pass_success),
                str(result.final_success),
                str(result.repair_iterations),
                modes,
                "" if result.compliance is None else f"{result.compliance:.6g}",
                "" if result.volume_error is None else f"{result.volume_error:.6g}",
                "" if result.grayness_index is None else f"{result.grayness_index:.6g}",
                "" if result.checkerboard_score is None else f"{result.checkerboard_score:.6g}",
                "" if result.connectivity_score is None else f"{result.connectivity_score:.6g}",
                str(result.converged),
            ])
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_minimal_benchmark(
    *,
    output: str | Path = DEFAULT_BENCHMARK_OUTPUT,
    quick: bool = False,
    use_llm_agents: bool = False,
    llm_provider: str | None = None,
) -> list[BenchmarkCaseResult]:
    """运行 6 case × 3 method 的最小实验。"""

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    methods = [
        BenchmarkMethod.BASELINE_DIRECT,
        BenchmarkMethod.BASELINE_NAIVE_RAG,
        BenchmarkMethod.OURS_CORRECTIVE_RAG,
    ]
    results: list[BenchmarkCaseResult] = []
    for case_spec in minimal_benchmark_cases(quick=quick):
        for method in methods:
            case_dir = root / f"{case_spec.case_id}__{method.value}"
            result = run_research_workflow(
                case_spec,
                output_dir=case_dir,
                method=method,
                quick=quick,
                max_repair_rounds=3,
                use_llm_agents=use_llm_agents,
                llm_provider=llm_provider,
            )
            results.append(result)
    write_summary(results, root)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行 AutoTopo 最小研究 benchmark")
    parser.add_argument("--quick", action="store_true", help="使用小网格和少量迭代")
    parser.add_argument("--output", default=str(DEFAULT_BENCHMARK_OUTPUT), help="输出目录")
    parser.add_argument("--llm-agents", action="store_true", help="启用 Scientist/Planner/Reviewer LLM 路径")
    parser.add_argument("--provider", default=None, help="LLM provider；为空时使用 config/settings.yaml 默认值")
    args = parser.parse_args(argv)

    results = run_minimal_benchmark(
        output=args.output,
        quick=args.quick,
        use_llm_agents=args.llm_agents,
        llm_provider=args.provider,
    )
    final_success = sum(1 for item in results if item.final_success)
    print(f"minimal benchmark complete: {len(results)} runs, final_success={final_success}")
    print(f"summary: {Path(args.output) / 'summary.csv'}")


if __name__ == "__main__":
    main()
