"""最小 benchmark 实验入口测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from autotopo import research_graph
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.experiments.run_minimal_benchmark import run_minimal_benchmark, write_summary
from autotopo.monitoring import WorkflowTracer, read_jsonl
from autotopo.research_graph import run_research_workflow
from autotopo.schemas import (
    BenchmarkCaseResult,
    BenchmarkMethod,
    BenchmarkType,
    EvaluatorReport,
    ExecutionReport,
    FailureMode,
    RepairPlan,
    RetrievedEvidence,
)


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
        "artifact_index.json",
        "llm_agent_trace.json",
        "00_scientist/case_spec.json",
        "00_scientist/case_spec_causality.json",
        "01_validator/validation_report.json",
        "02_planner_coder/code_plan.json",
        "03_executor/round_00/execution_report.json",
        "03_executor/round_00/run_stdout.log",
        "03_executor/round_00/run_stderr.log",
        "03_executor/round_00/optimization_history.csv",
        "03_executor/round_00/optimization_history.png",
        "03_executor/round_00/density.npy",
        "03_executor/round_00/density.png",
        "05_evaluator/round_00/evaluator_report.json",
        "06_summary/failure_diagnosis.json",
        "06_summary/repair_plan.json",
        "06_summary/repair_trace.json",
        "06_summary/final_summary.md",
        "result/result_index.json",
        "result/round_00_density.png",
        "result/round_00_optimization_history.png",
        "result/latest_density.png",
        "result/latest_optimization_history.png",
    ]:
        assert (sample_dir / name).exists()
    for name in [
        "01_validator/retrieved_evidence_validation.json",
        "02_planner_coder/retrieved_evidence.json",
        "02_planner_coder/retrieved_evidence_codegen.json",
        "05_evaluator/round_00/retrieved_evidence_critic_repair.json",
        "06_summary/retrieved_evidence.json",
        "06_summary/retrieved_evidence_critic_repair.json",
        "06_summary/retrieved_evidence_execution_repair.json",
    ]:
        assert not (sample_dir / name).exists()
    artifact_index = json.loads((sample_dir / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["artifacts"]["case_spec.json"]["path"] == "00_scientist/case_spec.json"
    assert artifact_index["artifacts"]["code_plan.json"]["path"] == "02_planner_coder/code_plan.json"
    assert artifact_index["artifacts"]["execution_report.json"]["path"].startswith("03_executor/round_")
    assert (
        artifact_index["artifacts"]["executor_round_00_execution_report.json"]["path"]
        == "03_executor/round_00/execution_report.json"
    )
    assert "artifact_history" not in artifact_index
    result_index = json.loads((sample_dir / "result" / "result_index.json").read_text(encoding="utf-8"))
    assert result_index["rounds"]
    assert result_index["rounds"][0]["density_image"] == "result/round_00_density.png"
    assert result_index["latest"]["density_image"] == "result/latest_density.png"


def test_research_workflow_default_output_under_project_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = default_case_spec("cantilever", quick=True)

    result = run_research_workflow(case, method=BenchmarkMethod.OURS_CORRECTIVE_RAG, quick=True)

    expected = tmp_path / "output" / "research_graph" / "cantilever_clear__ours_corrective_rag"
    assert tmp_path / Path(result.output_dir) == expected
    assert (expected / "00_scientist" / "case_spec.json").exists()
    assert (expected / "artifact_index.json").exists()
    assert (expected / "result" / "latest_density.png").exists()


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
    assert all(event["artifacts"] == [] for event in progress_events)
    assert (tmp_path / "00_scientist" / "case_spec.json").exists()
    assert tracer.resolve_artifact("00_scientist/case_spec.json").exists()
    assert tracer.resolve_artifact("result/latest_density.png").exists()
    assert (tmp_path / "workflow_events.jsonl").exists()


def test_research_workflow_writes_case_spec_causality_after_repair(tmp_path, monkeypatch):
    case = default_case_spec("cantilever", quick=True)

    def fake_execute(case_spec, code_plan, output_dir, **_kwargs):
        report = ExecutionReport(
            case_id=case_spec.case_id,
            method=code_plan.method,
            success=True,
            output_dir=str(output_dir),
            stdout_path=str(Path(output_dir) / "run_stdout.log"),
            stderr_path=str(Path(output_dir) / "run_stderr.log"),
            compliance=1.0,
            converged=True,
            files={
                "density_image": str(Path(output_dir) / "density.png"),
                "optimization_history_image": str(Path(output_dir) / "optimization_history.png"),
            },
        )
        Path(report.stdout_path).write_text("", encoding="utf-8")
        Path(report.stderr_path).write_text("", encoding="utf-8")
        (Path(output_dir) / "density.png").write_bytes(b"density")
        (Path(output_dir) / "optimization_history.png").write_bytes(b"history")
        (Path(output_dir) / "execution_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
        return report

    round_00_evidence = RetrievedEvidence(
        evidence_id="round_00_quality_rule",
        source="failure_kb.md",
        content="第一轮质量修复证据。",
        score=1.0,
        kind="quality",
    )
    round_01_evidence = RetrievedEvidence(
        evidence_id="round_01_quality_rule",
        source="failure_kb.md",
        content="第二轮质量修复证据。",
        score=1.0,
        kind="quality",
    )

    def fake_evaluate(case_spec, _execution_report, *, repair_iteration=0, max_repair_rounds=3, **_kwargs):
        if repair_iteration == 0:
            repair_plan = RepairPlan(
                case_id=case_spec.case_id,
                should_repair=True,
                repair_iteration=repair_iteration,
                max_repair_rounds=max_repair_rounds,
                repair_type="parameter_patch",
                target="penal",
                old_value={"penal": case_spec.penal},
                new_value={"penal": 4.0},
                parameter_updates={"penal": 4.0},
                reason="测试修复后 CaseSpec artifact 因果链。",
                failure_modes=[FailureMode.GRAYNESS_TOO_HIGH],
                auto_repair_allowed=True,
                auto_apply_allowed=True,
            )
            report = EvaluatorReport(
                case_id=case_spec.case_id,
                success=False,
                has_quality_failure=True,
                local_has_quality_failure=True,
                failure_modes=[FailureMode.GRAYNESS_TOO_HIGH],
                repair_plan=repair_plan,
                evidence_ids=[round_00_evidence.evidence_id],
            )
            return report, repair_plan, [round_00_evidence]
        return (
            EvaluatorReport(
                case_id=case_spec.case_id,
                success=True,
                has_quality_failure=False,
                evidence_ids=[round_01_evidence.evidence_id],
            ),
            None,
            [round_01_evidence],
        )

    monkeypatch.setattr(research_graph, "execute", fake_execute)
    monkeypatch.setattr(research_graph, "evaluate_execution", fake_evaluate)

    result = run_research_workflow(
        case,
        output_dir=tmp_path,
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        quick=True,
        max_repair_rounds=1,
    )

    case_spec = json.loads((tmp_path / "00_scientist" / "case_spec.json").read_text(encoding="utf-8"))
    repaired = json.loads((tmp_path / "00_scientist" / "case_spec_repaired.json").read_text(encoding="utf-8"))
    causality = json.loads((tmp_path / "00_scientist" / "case_spec_causality.json").read_text(encoding="utf-8"))

    assert result.repair_iterations == 1
    assert case_spec["penal"] == 4.0
    assert repaired["penal"] == 4.0
    assert set(causality) >= {"raw", "normalized", "repair"}
    assert causality["raw"]["input_type"] == "case_spec"
    assert "case_spec" not in causality["raw"]
    assert causality["normalized"]["artifact"] == "00_scientist/case_spec.json"
    assert "case_spec" not in causality["normalized"]
    assert causality["repair"]["final_case_spec_artifact"] == "00_scientist/case_spec.json"
    assert causality["repair"]["applications"][0]["repair_plan"]["parameter_updates"] == {"penal": 4.0}
    assert "raw_case_spec" not in causality["repair"]["applications"][0]
    assert "repaired_case_spec" not in causality["repair"]["applications"][0]
    artifact_index = json.loads((tmp_path / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["artifacts"]["case_spec_repaired.json"]["path"] == "00_scientist/case_spec_repaired.json"
    assert not (tmp_path / "05_evaluator" / "round_00" / "retrieved_evidence_critic_repair.json").exists()
    assert not (tmp_path / "05_evaluator" / "round_01" / "retrieved_evidence_critic_repair.json").exists()
    assert not (tmp_path / "06_summary" / "retrieved_evidence_critic_repair.json").exists()
    assert not (tmp_path / "02_planner_coder" / "retrieved_evidence.json").exists()
    assert not (tmp_path / "06_summary" / "retrieved_evidence.json").exists()
    assert (tmp_path / "result" / "round_00_density.png").exists()
    assert (tmp_path / "result" / "round_01_density.png").exists()


def test_artifact_index_history_keeps_unique_registrations(tmp_path):
    from autotopo.research_graph import _ResearchArtifactLayout

    layout = _ResearchArtifactLayout(tmp_path, persist_debug_artifacts=True)
    layout.write_json(tmp_path / "artifact.json", {"value": 1}, stage="summary")
    layout.write_json(tmp_path / "artifact.json", {"value": 2}, stage="summary")
    layout.write_index()

    artifact_index = json.loads((tmp_path / "artifact_index.json").read_text(encoding="utf-8"))
    matching = [item for item in artifact_index["artifact_history"] if item["name"] == "artifact.json"]
    assert len(matching) == 1


def test_research_workflow_debug_artifacts_persist_full_evidence_and_history(tmp_path, monkeypatch):
    case = default_case_spec("cantilever", quick=True)
    debug_evidence = RetrievedEvidence(
        evidence_id="debug_quality_rule",
        source="failure_kb.md",
        content="调试模式保存完整检索证据。",
        score=1.0,
        kind="quality",
    )

    def fake_execute(case_spec, code_plan, output_dir, **_kwargs):
        report = ExecutionReport(
            case_id=case_spec.case_id,
            method=code_plan.method,
            success=True,
            output_dir=str(output_dir),
            stdout_path=str(Path(output_dir) / "run_stdout.log"),
            stderr_path=str(Path(output_dir) / "run_stderr.log"),
            compliance=1.0,
            converged=True,
            files={
                "density_image": str(Path(output_dir) / "density.png"),
                "optimization_history_image": str(Path(output_dir) / "optimization_history.png"),
            },
        )
        Path(report.stdout_path).write_text("", encoding="utf-8")
        Path(report.stderr_path).write_text("", encoding="utf-8")
        (Path(output_dir) / "density.png").write_bytes(b"density")
        (Path(output_dir) / "optimization_history.png").write_bytes(b"history")
        (Path(output_dir) / "execution_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
        return report

    def fake_evaluate(case_spec, _execution_report, **_kwargs):
        report = EvaluatorReport(
            case_id=case_spec.case_id,
            success=True,
            has_quality_failure=False,
            evidence_ids=[debug_evidence.evidence_id],
        )
        return report, None, [debug_evidence]

    monkeypatch.setattr(research_graph, "execute", fake_execute)
    monkeypatch.setattr(research_graph, "evaluate_execution", fake_evaluate)

    run_research_workflow(
        case,
        output_dir=tmp_path,
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        quick=True,
        max_repair_rounds=0,
        persist_debug_artifacts=True,
    )

    assert (tmp_path / "debug" / "evidence" / "05_evaluator" / "round_00_retrieved_evidence_critic_repair.json").exists()
    assert (tmp_path / "debug" / "evidence" / "06_summary" / "retrieved_evidence.json").exists()
    assert (tmp_path / "debug" / "case_spec_causality_full.json").exists()
    assert (tmp_path / "debug" / "artifact_history.jsonl").exists()
    full_causality = json.loads((tmp_path / "debug" / "case_spec_causality_full.json").read_text(encoding="utf-8"))
    assert full_causality["raw"]["case_spec"]["penal"] == 3.0
    artifact_index = json.loads((tmp_path / "artifact_index.json").read_text(encoding="utf-8"))
    assert "artifact_history" in artifact_index


def test_write_summary_splits_execution_and_quality_success(tmp_path):
    result = BenchmarkCaseResult(
        case_id="smoke_case",
        benchmark_type=BenchmarkType.MBB,
        method=BenchmarkMethod.BASELINE_DIRECT,
        first_pass_success=True,
        execution_success=True,
        quality_success=False,
        final_success=False,
        output_dir=str(tmp_path / "smoke_case"),
    )

    write_summary([result], tmp_path)

    row = next(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert row["execution_success"] == "True"
    assert row["quality_success"] == "False"
    assert row["final_success"] == "False"

    summary_md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "- execution_success: 1" in summary_md
    assert "- quality_success: 0" in summary_md
    assert "| case_id | benchmark_type | method | first_pass_success | execution_success | quality_success | final_success |" in summary_md
