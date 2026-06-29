"""最小 benchmark 实验入口测试。"""

from __future__ import annotations

import csv
from pathlib import Path

from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.experiments.run_minimal_benchmark import run_minimal_benchmark
from autotopo.monitoring import WorkflowTracer, read_jsonl
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


def test_research_workflow_tracer_records_agent_timeline(tmp_path):
    tracer = WorkflowTracer(run_id="research-test", workflow_type="research", output_dir=tmp_path)

    result = run_research_workflow(
        "请做一个快速悬臂梁 benchmark",
        output_dir=tmp_path,
        method=BenchmarkMethod.BASELINE_DIRECT,
        quick=True,
        max_repair_rounds=0,
        tracer=tracer,
    )

    events = read_jsonl(tmp_path / "workflow_events.jsonl")
    completed = [event for event in events if event["status"] == "completed"]
    completed_stages = [event["stage"] for event in completed]
    progress_events = [event for event in events if event["stage"] == "optimization_iteration"]
    assert result.output_dir == str(tmp_path)
    assert "scientist" in completed_stages
    assert "validator" in completed_stages
    assert "planner_coder" in completed_stages
    assert "executor" in completed_stages
    assert "evaluator" in completed_stages
    assert "final_summary" in completed_stages
    assert progress_events
    assert {"iteration", "max_iter", "compliance", "volume", "change"} <= set(progress_events[0]["payload"])
    assert (tmp_path / "case_spec.json").exists()
    assert (tmp_path / "workflow_events.jsonl").exists()
