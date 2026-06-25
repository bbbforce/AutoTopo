"""最小 TopOptAgents-style 研究 workflow。

该 workflow 独立于现有 `graph.py`，用于 deterministic benchmark 实验：
Scientist → Validator → Planner → Coder → Executor → Reviewer/Evaluator → bounded repair。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotopo.agents.coder import select_or_generate_code
from autotopo.agents.evaluator import evaluate_execution
from autotopo.agents.executor import execute
from autotopo.agents.planner import plan_code
from autotopo.agents.reviewer import review_execution_failure
from autotopo.agents.scientist import build_case_spec
from autotopo.agents.validator import validate
from autotopo.diagnostics.repair_rules import apply_repair_plan
from autotopo.rag.corrective_rag import retrieve_for_codegen, retrieve_for_validation_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import (
    BenchmarkCaseResult,
    BenchmarkMethod,
    CaseSpec,
    CodePlan,
    EvaluatorReport,
    ExecutionReport,
    FailureDiagnosis,
    RepairPlan,
    RetrievedEvidence,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_model_dump(item) for item in obj]
    return obj


def _write_final_summary(path: Path, result: BenchmarkCaseResult, repair_trace: list[RepairPlan]) -> None:
    lines = [
        f"# {result.case_id}",
        "",
        f"- benchmark_type: {result.benchmark_type.value}",
        f"- method: {result.method.value}",
        f"- first_pass_success: {result.first_pass_success}",
        f"- final_success: {result.final_success}",
        f"- repair_success: {result.repair_success}",
        f"- repair_iterations: {result.repair_iterations}",
        f"- execution_error_type: {result.execution_error_type or ''}",
        f"- detected_failure_modes: {', '.join(mode.value for mode in result.detected_failure_modes)}",
        f"- compliance: {result.compliance}",
        f"- volume_error: {result.volume_error}",
        f"- grayness_index: {result.grayness_index}",
        f"- checkerboard_score: {result.checkerboard_score}",
        f"- connectivity_score: {result.connectivity_score}",
        f"- converged: {result.converged}",
        "",
        "## Repair Trace",
    ]
    if repair_trace:
        for item in repair_trace:
            lines.append(f"- iter {item.repair_iteration}: {item.parameter_updates} ({item.reason})")
    else:
        lines.append("- no repair")
    path.write_text("\n".join(lines), encoding="utf-8")


def _empty_diagnosis(case_spec: CaseSpec) -> FailureDiagnosis:
    return FailureDiagnosis(case_id=case_spec.case_id, has_failure=False)


def _empty_evaluator(case_spec: CaseSpec) -> EvaluatorReport:
    return EvaluatorReport(case_id=case_spec.case_id, success=False, has_quality_failure=False)


def _empty_repair_plan(case_spec: CaseSpec, reason: str = "没有执行修复。") -> RepairPlan:
    return RepairPlan(
        case_id=case_spec.case_id,
        should_repair=False,
        repair_type="fail_closed",
        reason=reason,
        rationale=reason,
        auto_repair_allowed=False,
        auto_apply_allowed=False,
        risk_level="low",
    )


def _benchmark_result(
    case_spec: CaseSpec,
    method: BenchmarkMethod,
    output_dir: Path,
    *,
    first_pass_success: bool,
    final_success: bool,
    repair_trace: list[RepairPlan],
    execution_report: ExecutionReport | None,
    evaluator_report: EvaluatorReport | None,
    failure_diagnosis: FailureDiagnosis | None,
) -> BenchmarkCaseResult:
    modes = []
    if failure_diagnosis:
        modes.extend(failure_diagnosis.failure_modes)
    if evaluator_report:
        modes.extend(evaluator_report.failure_modes)
    unique_modes = list(dict.fromkeys(modes))
    return BenchmarkCaseResult(
        case_id=case_spec.case_id,
        benchmark_type=case_spec.benchmark_type,
        method=method,
        first_pass_success=first_pass_success,
        final_success=final_success,
        repair_success=bool(repair_trace) and final_success,
        repair_iterations=len(repair_trace),
        execution_error_type=execution_report.error_type if execution_report else None,
        detected_failure_modes=unique_modes,
        compliance=evaluator_report.compliance if evaluator_report else None,
        volume_error=evaluator_report.volume_error if evaluator_report else None,
        grayness_index=evaluator_report.grayness_index if evaluator_report else None,
        checkerboard_score=evaluator_report.checkerboard_score if evaluator_report else None,
        connectivity_score=evaluator_report.connectivity_score if evaluator_report else None,
        converged=evaluator_report.converged if evaluator_report else False,
        output_dir=str(output_dir),
    )


def run_research_workflow(
    case_or_text: CaseSpec | str,
    *,
    output_dir: str | Path | None = None,
    method: BenchmarkMethod | str = BenchmarkMethod.OURS_CORRECTIVE_RAG,
    structured_params: dict[str, Any] | None = None,
    quick: bool = False,
    max_repair_rounds: int = 3,
    use_llm_agents: bool = False,
    llm_provider: str | None = None,
    llm_overrides: dict[str, Any] | None = None,
    agent_llms: dict[str, Any] | None = None,
) -> BenchmarkCaseResult:
    """运行单个 case-method 的最小研究 workflow。

    LLM agents are opt-in. When enabled, Scientist/Planner/Reviewer try a
    structured LLM call first and fall back to deterministic rules on failure.
    """

    method = BenchmarkMethod(method)
    retriever = LocalRetriever()
    agent_llms = agent_llms or {}
    llm_agent_trace: list[dict[str, Any]] = []

    def _use_llm(agent_name: str) -> bool:
        return bool(use_llm_agents or llm_provider or agent_name in agent_llms)

    case_spec = case_or_text if isinstance(case_or_text, CaseSpec) else build_case_spec(
        case_or_text,
        structured_params=structured_params,
        quick=quick,
        use_llm=_use_llm("scientist"),
        llm_provider=llm_provider,
        llm=agent_llms.get("scientist"),
        llm_overrides=llm_overrides,
        trace=llm_agent_trace,
    )
    if not case_spec.problem:
        from autotopo.engines.structured_benchmarks import case_to_problem

        case_spec = case_spec.model_copy(update={"problem": case_to_problem(case_spec)})

    # 默认输出到项目 output 下的研究 workflow 独立子目录。
    if output_dir is None:
        output_dir = Path("output") / "research_graph" / f"{case_spec.case_id}__{method.value}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_evidence: list[RetrievedEvidence] = []
    codegen_evidence: list[RetrievedEvidence] = []
    execution_repair_evidence: list[RetrievedEvidence] = []
    critic_repair_evidence: list[RetrievedEvidence] = []
    validation_evidence: list[RetrievedEvidence] = []
    repair_trace: list[RepairPlan] = []
    final_repair_plan: RepairPlan = _empty_repair_plan(case_spec)
    final_execution: ExecutionReport | None = None
    final_evaluator: EvaluatorReport | None = None
    final_diagnosis: FailureDiagnosis | None = None
    first_pass_success = False

    _write_json(out / "case_spec.json", case_spec.model_dump(mode="json"))
    _write_json(out / "llm_agent_trace.json", llm_agent_trace)
    validation_report = validate(case_spec)
    _write_json(out / "validation_report.json", validation_report.model_dump(mode="json"))
    if not validation_report.is_valid:
        validation_evidence = retrieve_for_validation_failure(validation_report, case_spec, retriever)
        all_evidence.extend(validation_evidence)
        code_plan = CodePlan(
            case_id=case_spec.case_id,
            method=method,
            template_id=case_spec.benchmark_type.value,
            steps=["Validator fail-closed，未进入求解。"],
        )
        final_repair_plan = _empty_repair_plan(case_spec, reason="Validator fail-closed，未进入求解。")
        final_repair_plan = final_repair_plan.model_copy(
            update={
                "failure_modes": validation_report.failure_modes,
                "evidence_ids": [item.evidence_id for item in validation_evidence],
                "risk_level": "high",
            }
        )
        _write_json(out / "retrieved_evidence_codegen.json", [])
        _write_json(out / "retrieved_evidence_execution_repair.json", [])
        _write_json(out / "retrieved_evidence_critic_repair.json", [])
        _write_json(out / "retrieved_evidence_validation.json", _model_dump(validation_evidence))
        _write_json(out / "retrieved_evidence.json", _model_dump(all_evidence))
        _write_json(out / "code_plan.json", code_plan.model_dump(mode="json"))
        final_diagnosis = FailureDiagnosis(
            case_id=case_spec.case_id,
            has_failure=True,
            failure_modes=validation_report.failure_modes,
            severity=validation_report.severity,
            likely_causes=validation_report.messages,
            repair_suggestions=[],
            auto_repair_allowed=False,
            evidence_ids=[item.evidence_id for item in validation_evidence],
        )
        _write_json(out / "failure_diagnosis.json", final_diagnosis.model_dump(mode="json"))
        _write_json(out / "repair_plan.json", final_repair_plan.model_dump(mode="json"))
        _write_json(out / "repair_trace.json", [])
        _write_json(out / "llm_agent_trace.json", llm_agent_trace)
        final_evaluator = _empty_evaluator(case_spec)
        _write_json(out / "evaluator_report.json", final_evaluator.model_dump(mode="json"))
        result = _benchmark_result(
            case_spec,
            method,
            out,
            first_pass_success=False,
            final_success=False,
            repair_trace=repair_trace,
            execution_report=None,
            evaluator_report=final_evaluator,
            failure_diagnosis=final_diagnosis,
        )
        _write_final_summary(out / "final_summary.md", result, repair_trace)
        return result

    if method in {BenchmarkMethod.BASELINE_NAIVE_RAG, BenchmarkMethod.OURS_CORRECTIVE_RAG}:
        codegen_evidence = retrieve_for_codegen(case_spec, retriever)
        all_evidence.extend(codegen_evidence)
    _write_json(out / "retrieved_evidence_codegen.json", _model_dump(codegen_evidence))
    _write_json(out / "retrieved_evidence_execution_repair.json", [])
    _write_json(out / "retrieved_evidence_critic_repair.json", [])
    _write_json(out / "retrieved_evidence_validation.json", [])
    _write_json(out / "retrieved_evidence.json", _model_dump(all_evidence))
    code_plan = select_or_generate_code(
        plan_code(
            case_spec,
            method,
            all_evidence,
            use_llm=_use_llm("planner"),
            llm_provider=llm_provider,
            llm=agent_llms.get("planner"),
            llm_overrides=llm_overrides,
            trace=llm_agent_trace,
        )
    )
    _write_json(out / "code_plan.json", code_plan.model_dump(mode="json"))
    _write_json(out / "llm_agent_trace.json", llm_agent_trace)

    for repair_iteration in range(max_repair_rounds + 1):
        final_execution = execute(case_spec, code_plan, out)
        if repair_iteration == 0:
            first_pass_success = final_execution.success

        if not final_execution.success:
            final_diagnosis, repair_plan, evidence = review_execution_failure(
                case_spec,
                final_execution,
                repair_iteration=repair_iteration,
                max_repair_rounds=max_repair_rounds,
                retriever=retriever,
                use_llm=_use_llm("reviewer"),
                llm_provider=llm_provider,
                llm=agent_llms.get("reviewer"),
                llm_overrides=llm_overrides,
                trace=llm_agent_trace,
            )
            final_repair_plan = repair_plan
            execution_repair_evidence.extend(evidence)
            all_evidence.extend(evidence)
            _write_json(out / "retrieved_evidence_execution_repair.json", _model_dump(execution_repair_evidence))
            _write_json(out / "repair_plan.json", final_repair_plan.model_dump(mode="json"))
            if method == BenchmarkMethod.OURS_CORRECTIVE_RAG and repair_plan.should_repair:
                repair_trace.append(repair_plan)
                case_spec = apply_repair_plan(case_spec, repair_plan)
                _write_json(out / "repair_trace.json", _model_dump(repair_trace))
                continue
            break

        final_diagnosis = _empty_diagnosis(case_spec)
        final_evaluator, repair_plan, evidence = evaluate_execution(
            case_spec,
            final_execution,
            repair_iteration=repair_iteration,
            max_repair_rounds=max_repair_rounds,
            retriever=retriever,
        )
        if repair_plan is not None:
            final_repair_plan = repair_plan
        critic_repair_evidence.extend(evidence)
        all_evidence.extend(evidence)
        _write_json(out / "retrieved_evidence_critic_repair.json", _model_dump(critic_repair_evidence))
        _write_json(out / "repair_plan.json", final_repair_plan.model_dump(mode="json"))
        if (
            method == BenchmarkMethod.OURS_CORRECTIVE_RAG
            and repair_plan is not None
            and repair_plan.should_repair
        ):
            repair_trace.append(repair_plan)
            case_spec = apply_repair_plan(case_spec, repair_plan)
            _write_json(out / "repair_trace.json", _model_dump(repair_trace))
            continue
        break

    _write_json(out / "retrieved_evidence_codegen.json", _model_dump(codegen_evidence))
    _write_json(out / "retrieved_evidence_execution_repair.json", _model_dump(execution_repair_evidence))
    _write_json(out / "retrieved_evidence_critic_repair.json", _model_dump(critic_repair_evidence))
    _write_json(out / "retrieved_evidence_validation.json", _model_dump(validation_evidence))
    _write_json(out / "retrieved_evidence.json", _model_dump(all_evidence))
    _write_json(out / "failure_diagnosis.json", (final_diagnosis or _empty_diagnosis(case_spec)).model_dump(mode="json"))
    _write_json(out / "repair_plan.json", final_repair_plan.model_dump(mode="json"))
    _write_json(out / "repair_trace.json", _model_dump(repair_trace))
    _write_json(out / "evaluator_report.json", (final_evaluator or _empty_evaluator(case_spec)).model_dump(mode="json"))
    _write_json(out / "llm_agent_trace.json", llm_agent_trace)

    final_success = bool(final_execution and final_execution.success and final_evaluator and final_evaluator.success)
    result = _benchmark_result(
        case_spec,
        method,
        out,
        first_pass_success=first_pass_success,
        final_success=final_success,
        repair_trace=repair_trace,
        execution_report=final_execution,
        evaluator_report=final_evaluator,
        failure_diagnosis=final_diagnosis,
    )
    _write_final_summary(out / "final_summary.md", result, repair_trace)
    return result
