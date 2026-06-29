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
    AgentAuthority,
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
    agent_authority: AgentAuthority | str = AgentAuthority.DETERMINISTIC,
    allow_generated_code: bool = False,
    generated_code_timeout_s: int = 60,
    tracer: Any | None = None,
) -> BenchmarkCaseResult:
    """运行单个 case-method 的最小研究 workflow。

    LLM agents 默认关闭。`use_llm_agents=True` 保持兼容并映射为 llm_assisted；
    只有 llm_primary 才允许 LLM 覆盖本地判断或自动执行生成脚本。
    """

    method = BenchmarkMethod(method)
    retriever = LocalRetriever()
    agent_llms = agent_llms or {}
    authority = AgentAuthority(agent_authority)
    if authority == AgentAuthority.DETERMINISTIC and (use_llm_agents or llm_provider or agent_llms):
        authority = AgentAuthority.LLM_ASSISTED
    allow_llm_override = authority == AgentAuthority.LLM_PRIMARY
    llm_agent_trace: list[dict[str, Any]] = []

    def _start(stage: str, agent: str, summary: str, payload: Any | None = None) -> Any:
        if tracer is None:
            return None
        return tracer.start_stage(stage, agent=agent, summary=summary, payload=payload)

    def _complete(
        token: Any,
        *,
        summary: str,
        payload: Any | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        if tracer is not None:
            tracer.complete_stage(token, summary=summary, payload=payload, artifacts=artifacts)

    def _fail(token: Any, exc: BaseException, payload: Any | None = None) -> None:
        if tracer is not None:
            tracer.fail_stage(token, exc, payload=payload)

    def _use_llm(agent_name: str) -> bool:
        if authority == AgentAuthority.DETERMINISTIC:
            return False
        return bool(use_llm_agents or llm_provider or agent_name in agent_llms)

    def _diagnosis_requests(diagnosis: FailureDiagnosis | None, action: str) -> bool:
        if diagnosis is None:
            return False
        needle = action.lower()
        return any(needle in item.lower().replace("-", "_") for item in diagnosis.repair_suggestions)

    scientist_token = _start(
        "scientist",
        "Scientist",
        "Scientist 构建 CaseSpec",
        {"quick": quick, "llm_enabled": _use_llm("scientist"), "agent_authority": authority.value},
    )
    try:
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
        _write_json(out / "case_spec.json", case_spec.model_dump(mode="json"))
        _write_json(out / "llm_agent_trace.json", llm_agent_trace)
    except Exception as exc:
        _fail(scientist_token, exc)
        raise
    _complete(
        scientist_token,
        summary="Scientist 完成 CaseSpec",
        payload={
            "case_spec": case_spec.model_dump(mode="json"),
            "llm_agent_trace": llm_agent_trace,
        },
    )

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

    validator_token = _start(
        "validator",
        "Validator",
        "Validator 执行 fail-closed 检查",
        {"llm_enabled": _use_llm("validator"), "allow_llm_override": allow_llm_override},
    )
    try:
        validation_report = validate(
            case_spec,
            retriever=retriever,
            use_llm=_use_llm("validator"),
            allow_llm_override=allow_llm_override,
            llm_provider=llm_provider,
            llm=agent_llms.get("validator"),
            llm_overrides=llm_overrides,
            trace=llm_agent_trace,
        )
        if validation_report.local_is_valid is False or validation_report.evidence_ids:
            validation_rag_report = validation_report
            if validation_report.overridden_failure_modes and not validation_report.failure_modes:
                validation_rag_report = validation_report.model_copy(
                    update={
                        "is_valid": False,
                        "failure_modes": validation_report.overridden_failure_modes,
                    }
                )
            validation_evidence = retrieve_for_validation_failure(validation_rag_report, case_spec, retriever)
            all_evidence.extend(validation_evidence)
        _write_json(out / "validation_report.json", validation_report.model_dump(mode="json"))
        _write_json(out / "retrieved_evidence_validation.json", _model_dump(validation_evidence))
        _write_json(out / "llm_agent_trace.json", llm_agent_trace)
    except Exception as exc:
        _fail(validator_token, exc)
        raise
    _complete(
        validator_token,
        summary="Validator 检查完成",
        payload=validation_report.model_dump(mode="json"),
    )
    if not validation_report.is_valid:
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
        summary_token = _start("final_summary", "Reporter", "保存研究 workflow 摘要")
        _complete(summary_token, summary="研究 workflow fail-closed 完成", payload=result.model_dump(mode="json"))
        return result

    planner_token = _start(
        "planner_coder",
        "Planner/Coder",
        "Planner/Coder 选择求解计划",
        {
            "method": method.value,
            "llm_enabled": _use_llm("planner"),
            "coder_llm_enabled": _use_llm("coder"),
            "allow_generated_code": allow_generated_code,
            "agent_authority": authority.value,
        },
    )
    try:
        if method in {BenchmarkMethod.BASELINE_NAIVE_RAG, BenchmarkMethod.OURS_CORRECTIVE_RAG}:
            codegen_evidence = retrieve_for_codegen(case_spec, retriever)
            all_evidence.extend(codegen_evidence)
        _write_json(out / "retrieved_evidence_codegen.json", _model_dump(codegen_evidence))
        _write_json(out / "retrieved_evidence_execution_repair.json", [])
        _write_json(out / "retrieved_evidence_critic_repair.json", [])
        _write_json(out / "retrieved_evidence_validation.json", _model_dump(validation_evidence))
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
            ),
            case_spec=case_spec,
            evidence=all_evidence,
            output_dir=out,
            agent_authority=authority,
            allow_generated_code=allow_generated_code,
            use_llm=_use_llm("coder"),
            llm_provider=llm_provider,
            llm=agent_llms.get("coder"),
            llm_overrides=llm_overrides,
            trace=llm_agent_trace,
        )
        _write_json(out / "code_plan.json", code_plan.model_dump(mode="json"))
        _write_json(out / "llm_agent_trace.json", llm_agent_trace)
    except Exception as exc:
        _fail(planner_token, exc)
        raise
    _complete(
        planner_token,
        summary="Planner/Coder 计划完成",
        payload={
            "code_plan": code_plan.model_dump(mode="json"),
            "evidence_count": len(codegen_evidence),
            "llm_agent_trace": llm_agent_trace,
        },
    )

    for repair_iteration in range(max_repair_rounds + 1):
        executor_token = _start(
            "executor",
            "Executor",
            f"Executor 执行第 {repair_iteration} 轮",
            {"repair_iteration": repair_iteration},
        )

        def _format_progress_value(value: Any) -> str:
            try:
                return f"{float(value):.4g}"
            except (TypeError, ValueError):
                return "?"

        def _progress_event(progress: dict[str, Any]) -> None:
            if tracer is None:
                return
            iteration = progress.get("iteration")
            max_iter = progress.get("max_iter")
            summary = (
                f"优化迭代 {iteration}/{max_iter}: "
                f"compliance={_format_progress_value(progress.get('compliance'))}, "
                f"volume={_format_progress_value(progress.get('volume'))}, "
                f"change={_format_progress_value(progress.get('change'))}"
            )
            tracer.emit(
                stage="optimization_iteration",
                agent="Executor",
                status="running",
                summary=summary,
                payload={
                    **progress,
                    "repair_iteration": repair_iteration,
                    "case_id": case_spec.case_id,
                },
            )

        try:
            final_execution = execute(
                case_spec,
                code_plan,
                out,
                generated_code_timeout_s=generated_code_timeout_s,
                progress_callback=_progress_event,
            )
            if code_plan.execution_mode == "generated_script":
                llm_agent_trace.append(
                    {
                        "agent": "executor",
                        "enabled": True,
                        "used_llm": False,
                        "fallback_reason": "",
                        "execution_mode": "generated_script",
                        "sandbox": final_execution.metrics.get("sandbox", {}),
                        "success": final_execution.success,
                        "error_type": final_execution.error_type or "",
                    }
                )
                _write_json(out / "llm_agent_trace.json", llm_agent_trace)
        except Exception as exc:
            _fail(executor_token, exc)
            raise
        _complete(
            executor_token,
            summary=f"Executor 第 {repair_iteration} 轮完成",
            payload=final_execution.model_dump(mode="json"),
        )
        if repair_iteration == 0:
            first_pass_success = final_execution.success

        if not final_execution.success:
            reviewer_token = _start(
                "reviewer",
                "Reviewer",
                f"Reviewer 诊断第 {repair_iteration} 轮执行失败",
                {"llm_enabled": _use_llm("reviewer")},
            )
            try:
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
            except Exception as exc:
                _fail(reviewer_token, exc)
                raise
            _complete(
                reviewer_token,
                summary="Reviewer 诊断完成",
                payload={
                    "failure_diagnosis": final_diagnosis.model_dump(mode="json"),
                    "repair_plan": repair_plan.model_dump(mode="json"),
                    "evidence_count": len(evidence),
                    "llm_agent_trace": llm_agent_trace,
                },
            )
            final_repair_plan = repair_plan
            execution_repair_evidence.extend(evidence)
            all_evidence.extend(evidence)
            _write_json(out / "retrieved_evidence_execution_repair.json", _model_dump(execution_repair_evidence))
            _write_json(out / "repair_plan.json", final_repair_plan.model_dump(mode="json"))
            if method == BenchmarkMethod.OURS_CORRECTIVE_RAG and repair_plan.should_repair:
                repair_token = _start("repair", "Repair", "应用执行失败修复计划")
                repair_trace.append(repair_plan)
                case_spec = apply_repair_plan(case_spec, repair_plan)
                _write_json(out / "repair_trace.json", _model_dump(repair_trace))
                _complete(
                    repair_token,
                    summary="执行失败修复计划已应用",
                    payload=repair_plan.model_dump(mode="json"),
                )
                continue
            if (
                method == BenchmarkMethod.OURS_CORRECTIVE_RAG
                and code_plan.execution_mode == "generated_script"
                and _diagnosis_requests(final_diagnosis, "fallback_template")
            ):
                repair_token = _start("repair", "Repair", "Reviewer 要求回退模板执行")
                fallback_plan = RepairPlan(
                    case_id=case_spec.case_id,
                    should_repair=True,
                    repair_iteration=repair_iteration,
                    max_repair_rounds=max_repair_rounds,
                    repair_type="fallback_template",
                    rationale="Reviewer 建议从生成脚本回退到模板执行。",
                    reason="Reviewer 建议从生成脚本回退到模板执行。",
                    failure_modes=final_diagnosis.failure_modes,
                    evidence_ids=final_diagnosis.evidence_ids,
                    auto_repair_allowed=True,
                    auto_apply_allowed=True,
                    risk_level="medium",
                )
                final_repair_plan = fallback_plan
                repair_trace.append(fallback_plan)
                code_plan = select_or_generate_code(code_plan, agent_authority=AgentAuthority.DETERMINISTIC)
                _write_json(out / "code_plan.json", code_plan.model_dump(mode="json"))
                _write_json(out / "repair_trace.json", _model_dump(repair_trace))
                _complete(repair_token, summary="已回退到模板执行", payload=fallback_plan.model_dump(mode="json"))
                continue
            if (
                method == BenchmarkMethod.OURS_CORRECTIVE_RAG
                and code_plan.execution_mode == "generated_script"
                and allow_generated_code
                and authority == AgentAuthority.LLM_PRIMARY
                and _diagnosis_requests(final_diagnosis, "regenerate_code")
            ):
                repair_token = _start("repair", "Repair", "Reviewer 要求重新生成代码")
                regenerate_plan = RepairPlan(
                    case_id=case_spec.case_id,
                    should_repair=True,
                    repair_iteration=repair_iteration,
                    max_repair_rounds=max_repair_rounds,
                    repair_type="regenerate_code",
                    rationale="Reviewer 建议重新生成脚本。",
                    reason="Reviewer 建议重新生成脚本。",
                    failure_modes=final_diagnosis.failure_modes,
                    evidence_ids=final_diagnosis.evidence_ids,
                    auto_repair_allowed=True,
                    auto_apply_allowed=True,
                    risk_level="medium",
                )
                final_repair_plan = regenerate_plan
                repair_trace.append(regenerate_plan)
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
                    ),
                    case_spec=case_spec,
                    evidence=all_evidence,
                    output_dir=out,
                    agent_authority=authority,
                    allow_generated_code=allow_generated_code,
                    use_llm=_use_llm("coder"),
                    llm_provider=llm_provider,
                    llm=agent_llms.get("coder"),
                    llm_overrides=llm_overrides,
                    trace=llm_agent_trace,
                )
                _write_json(out / "code_plan.json", code_plan.model_dump(mode="json"))
                _write_json(out / "repair_trace.json", _model_dump(repair_trace))
                _write_json(out / "llm_agent_trace.json", llm_agent_trace)
                _complete(repair_token, summary="已重新生成代码计划", payload=regenerate_plan.model_dump(mode="json"))
                continue
            break

        final_diagnosis = _empty_diagnosis(case_spec)
        evaluator_token = _start(
            "evaluator",
            "Evaluator",
            f"Evaluator 评估第 {repair_iteration} 轮拓扑质量",
        )
        try:
            final_evaluator, repair_plan, evidence = evaluate_execution(
                case_spec,
                final_execution,
                repair_iteration=repair_iteration,
                max_repair_rounds=max_repair_rounds,
                retriever=retriever,
                use_llm=_use_llm("evaluator"),
                allow_llm_override=allow_llm_override,
                llm_provider=llm_provider,
                llm=agent_llms.get("evaluator"),
                llm_overrides=llm_overrides,
                trace=llm_agent_trace,
            )
        except Exception as exc:
            _fail(evaluator_token, exc)
            raise
        _complete(
            evaluator_token,
            summary="Evaluator 评估完成",
            payload={
                "evaluator_report": final_evaluator.model_dump(mode="json"),
                "repair_plan": repair_plan.model_dump(mode="json") if repair_plan is not None else None,
                "evidence_count": len(evidence),
            },
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
            repair_token = _start("repair", "Repair", "应用拓扑质量修复计划")
            repair_trace.append(repair_plan)
            case_spec = apply_repair_plan(case_spec, repair_plan)
            _write_json(out / "repair_trace.json", _model_dump(repair_trace))
            _complete(
                repair_token,
                summary="拓扑质量修复计划已应用",
                payload=repair_plan.model_dump(mode="json"),
            )
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
    summary_token = _start("final_summary", "Reporter", "保存研究 workflow 摘要")
    _complete(summary_token, summary="研究 workflow 完成", payload=result.model_dump(mode="json"))
    return result
