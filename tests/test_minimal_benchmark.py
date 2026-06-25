"""最小 benchmark 实验入口测试。"""

from __future__ import annotations

import csv
from pathlib import Path

from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.experiments.run_minimal_benchmark import run_minimal_benchmark
from autotopo.research_graph import run_research_workflow
from autotopo.schemas import BenchmarkMethod


def test_run_minimal_benchmark_quick_outputs_summary(tmp_path):
    results = run_minimal_benchmark(output=tmp_path, quick=True)

    assert len(results) == 18
    assert {result.method for result in results} == {
        BenchmarkMethod.BASELINE_DIRECT,
        BenchmarkMethod.BASELINE_NAIVE_RAG,
        BenchmarkMethod.OURS_CORRECTIVE_RAG,
    }
    summary_csv = tmp_path / "summary.csv"
    summary_md = tmp_path / "summary.md"
    assert summary_csv.exists()
    assert summary_md.exists()

    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert len(rows) == 18
    ours = [row for row in rows if row["method"] == BenchmarkMethod.OURS_CORRECTIVE_RAG.value]
    assert len(ours) == 6

    sample_dir = Path(ours[0]["output_dir"])
    for name in [
        "case_spec.json",
        "validation_report.json",
        "retrieved_evidence.json",
        "retrieved_evidence_codegen.json",
        "retrieved_evidence_execution_repair.json",
        "retrieved_evidence_critic_repair.json",
        "retrieved_evidence_validation.json",
        "code_plan.json",
        "execution_report.json",
        "run_stdout.log",
        "run_stderr.log",
        "optimization_history.csv",
        "optimization_history.png",
        "density.npy",
        "density.png",
        "evaluator_report.json",
        "failure_diagnosis.json",
        "repair_plan.json",
        "repair_trace.json",
        "final_summary.md",
    ]:
        assert (sample_dir / name).exists()


def test_research_workflow_default_output_under_project_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = default_case_spec("cantilever", quick=True)

    result = run_research_workflow(case, method=BenchmarkMethod.OURS_CORRECTIVE_RAG, quick=True)

    expected = tmp_path / "output" / "research_graph" / "cantilever_clear__ours_corrective_rag"
    assert tmp_path / Path(result.output_dir) == expected
    assert (expected / "case_spec.json").exists()
