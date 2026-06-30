"""最小 TopOptAgents-style 研究 workflow。

该 workflow 独立于现有 `graph.py`，用于 deterministic benchmark 实验：
Scientist → Validator → Planner → Coder → Executor → Reviewer/Evaluator → bounded repair。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from autotopo.agents.coder import select_or_generate_code
from autotopo.agents.evaluator import evaluate_execution
from autotopo.agents.executor import execute
from autotopo.agents.planner import plan_code
from autotopo.agents.reviewer import review_execution_failure
from autotopo.agents.scientist import build_case_spec_with_causality
from autotopo.agents.validator import validate
from autotopo.diagnostics.repair_rules import apply_repair_plan
from autotopo.engines.structured_benchmarks import case_to_problem
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


def _extend_unique_evidence(target: list[RetrievedEvidence], items: list[RetrievedEvidence]) -> None:
    """按 evidence_id 追加证据，避免跨轮累计文件重复膨胀。"""

    seen = {item.evidence_id for item in target}
    for item in items:
        if item.evidence_id in seen:
            continue
        target.append(item)
        seen.add(item.evidence_id)


class _ResearchArtifactLayout:
    """管理 research workflow 的阶段目录和机器可读 artifact 索引。"""

    STAGE_DIRS = {
        "scientist": "00_scientist",
        "validator": "01_validator",
        "planner_coder": "02_planner_coder",
        "executor": "03_executor",
        "reviewer_repair": "04_reviewer_repair",
        "evaluator": "05_evaluator",
        "summary": "06_summary",
        "result": "result",
        "debug": "debug",
        "global": "",
    }

    def __init__(self, root: Path, *, persist_debug_artifacts: bool = False) -> None:
        self.root = root
        self.persist_debug_artifacts = persist_debug_artifacts
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, Any] = {
            "schema_version": "research_artifact_layout_v1",
            "artifacts": {},
            "stages": {},
        }
        if self.persist_debug_artifacts:
            self.index["artifact_history"] = []
        self._history_seen: set[tuple[str, str, str, int | None]] = set()

    def stage_dir(self, stage: str) -> Path:
        directory = self.root / self.STAGE_DIRS[stage]
        directory.mkdir(parents=True, exist_ok=True)
        self.index["stages"].setdefault(stage, self._relative_path(directory))
        return directory

    def round_dir(self, stage: str, repair_iteration: int) -> Path:
        directory = self.stage_dir(stage) / f"round_{repair_iteration:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def debug_dir(self, *parts: str | Path) -> Path:
        directory = self.stage_dir("debug")
        for part in parts:
            directory = directory / part
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def register(
        self,
        path: Path,
        *,
        logical_name: str | None = None,
        stage: str = "global",
        repair_iteration: int | None = None,
    ) -> None:
        name = logical_name or path.name
        entry: dict[str, Any] = {
            "path": self._relative_path(path),
            "stage": stage,
        }
        if repair_iteration is not None:
            entry["round"] = repair_iteration
        self.index["artifacts"][name] = entry
        history_key = (name, entry["path"], stage, repair_iteration)
        if self.persist_debug_artifacts and history_key not in self._history_seen:
            self._history_seen.add(history_key)
            self.index["artifact_history"].append({"name": name, **entry})

    def write_json(
        self,
        path: Path,
        payload: Any,
        *,
        logical_name: str | None = None,
        stage: str = "global",
        repair_iteration: int | None = None,
    ) -> None:
        _write_json(path, payload)
        self.register(path, logical_name=logical_name, stage=stage, repair_iteration=repair_iteration)

    def write_debug_json(
        self,
        relative_path: str | Path,
        payload: Any,
        *,
        logical_name: str | None = None,
        stage: str = "debug",
        repair_iteration: int | None = None,
    ) -> Path | None:
        if not self.persist_debug_artifacts:
            return None
        path = self.debug_dir() / relative_path
        self.write_json(
            path,
            payload,
            logical_name=logical_name,
            stage=stage,
            repair_iteration=repair_iteration,
        )
        return path

    def write_index(self) -> None:
        if self.persist_debug_artifacts:
            history_path = self.debug_dir() / "artifact_history.jsonl"
            self.register(history_path, stage="debug")
            lines = [
                json.dumps(item, ensure_ascii=False)
                for item in self.index.get("artifact_history", [])
            ]
            history_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        index_path = self.root / "artifact_index.json"
        self.register(index_path, stage="global")
        _write_json(index_path, self.index)


def _case_spec_input_causality_full(case_spec: CaseSpec, *, quick: bool) -> dict[str, Any]:
    spec_dump = case_spec.model_dump(mode="json")
    return {
        "raw": {
            "input_type": "case_spec",
            "case_spec": spec_dump,
            "quick": quick,
            "source": "case_spec_input",
        },
        "normalized": {
            "artifact": "00_scientist/case_spec.json",
            "case_spec": spec_dump,
        },
        "repair": {
            "artifact": "00_scientist/case_spec_repaired.json",
            "applications": [],
            "final_case_spec": spec_dump,
        },
    }


def _case_spec_scalar_summary(case_spec: CaseSpec) -> dict[str, Any]:
    return {
        "case_id": case_spec.case_id,
        "benchmark_type": case_spec.benchmark_type.value,
        "variant": case_spec.variant,
        "nelx": case_spec.nelx,
        "nely": case_spec.nely,
        "volume_fraction": case_spec.volume_fraction,
        "penal": case_spec.penal,
        "rmin": case_spec.rmin,
        "max_iter": case_spec.max_iter,
        "tol": case_spec.tol,
        "optimizer": case_spec.optimizer,
    }


def _compact_case_spec_causality(full_causality: dict[str, Any], case_spec: CaseSpec) -> dict[str, Any]:
    raw_full = full_causality.get("raw", {})
    raw: dict[str, Any] = {
        "input_type": raw_full.get("input_type", "case_spec"),
        "quick": raw_full.get("quick"),
        "source": raw_full.get("source", "case_spec_input"),
        "case_id": case_spec.case_id,
        "benchmark_type": case_spec.benchmark_type.value,
    }
    for key in ("natural_language", "structured_params", "llm_requested"):
        if key in raw_full:
            raw[key] = raw_full[key]
    if raw_full.get("case_spec_draft") is not None:
        raw["case_spec_draft_present"] = True

    compact = {
        "raw": raw,
        "normalized": {
            "artifact": "00_scientist/case_spec.json",
            "parameters": _case_spec_scalar_summary(case_spec),
        },
        "repair": {
            "artifact": "00_scientist/case_spec_repaired.json",
            "applications": [],
            "final_case_spec_artifact": "00_scientist/case_spec.json",
        },
    }
    return compact


def _repair_application_summary(
    raw_case_spec: CaseSpec,
    repaired_case_spec: CaseSpec,
    repair_plan: RepairPlan,
) -> dict[str, Any]:
    parameter_changes = {}
    for key, new_value in repair_plan.parameter_updates.items():
        parameter_changes[key] = {
            "before": getattr(raw_case_spec, key, None),
            "after": new_value,
        }
    return {
        "repair_iteration": repair_plan.repair_iteration,
        "case_spec_before_artifact": "00_scientist/case_spec.json",
        "case_spec_after_artifact": "00_scientist/case_spec_repaired.json",
        "repair_plan": repair_plan.model_dump(mode="json"),
        "parameter_changes": parameter_changes,
    }


def _write_case_spec_artifacts(
    layout: _ResearchArtifactLayout,
    case_spec: CaseSpec,
    causality: dict[str, Any],
    *,
    full_causality: dict[str, Any] | None = None,
    repaired: bool = False,
) -> None:
    output_dir = layout.stage_dir("scientist")
    spec_dump = case_spec.model_dump(mode="json")
    causality.setdefault("normalized", {})["artifact"] = "00_scientist/case_spec.json"
    repair_layer = causality.setdefault("repair", {})
    repair_layer.setdefault("artifact", "00_scientist/case_spec_repaired.json")
    repair_layer.setdefault("applications", [])
    repair_layer["final_case_spec_artifact"] = "00_scientist/case_spec.json"
    if full_causality is not None:
        full_causality.setdefault("normalized", {})["artifact"] = "00_scientist/case_spec.json"
        full_causality.setdefault("normalized", {})["case_spec"] = spec_dump
        full_repair = full_causality.setdefault("repair", {})
        full_repair["artifact"] = "00_scientist/case_spec_repaired.json"
        full_repair.setdefault("applications", [])
        full_repair["final_case_spec"] = spec_dump
    layout.write_json(output_dir / "case_spec.json", spec_dump, stage="scientist")
    if repaired:
        layout.write_json(output_dir / "case_spec_repaired.json", spec_dump, stage="scientist")
    layout.write_json(output_dir / "case_spec_causality.json", causality, stage="scientist")
    if full_causality is not None:
        layout.write_debug_json(
            "case_spec_causality_full.json",
            full_causality,
            logical_name="case_spec_causality_full.json",
            stage="scientist",
        )


def _apply_repair_and_record_case_spec(
    layout: _ResearchArtifactLayout,
    case_spec: CaseSpec,
    repair_plan: RepairPlan,
    causality: dict[str, Any],
    full_causality: dict[str, Any],
) -> CaseSpec:
    raw_case_spec = case_spec
    repaired = apply_repair_plan(case_spec, repair_plan)
    repaired = repaired.model_copy(update={"problem": case_to_problem(repaired)})
    repair_layer = causality.setdefault("repair", {})
    applications = repair_layer.setdefault("applications", [])
    applications.append(_repair_application_summary(raw_case_spec, repaired, repair_plan))
    full_repair_layer = full_causality.setdefault("repair", {})
    full_applications = full_repair_layer.setdefault("applications", [])
    full_applications.append(
        {
            "repair_iteration": repair_plan.repair_iteration,
            "raw_case_spec": raw_case_spec.model_dump(mode="json"),
            "repair_plan": repair_plan.model_dump(mode="json"),
            "repaired_case_spec": repaired.model_dump(mode="json"),
        }
    )
    _write_case_spec_artifacts(layout, repaired, causality, full_causality=full_causality, repaired=True)
    return repaired


def _write_final_summary(path: Path, result: BenchmarkCaseResult, repair_trace: list[RepairPlan]) -> None:
    lines = [
        f"# {result.case_id}",
        "",
        f"- benchmark_type: {result.benchmark_type.value}",
        f"- method: {result.method.value}",
        f"- first_pass_success: {result.first_pass_success}",
        f"- execution_success: {result.execution_success}",
        f"- quality_success: {result.quality_success}",
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


def _path_from_report(value: str, default_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return default_dir / path


def _register_execution_artifacts(
    layout: _ResearchArtifactLayout,
    execution_report: ExecutionReport,
    *,
    repair_iteration: int,
) -> None:
    execution_dir = Path(execution_report.output_dir)
    layout.register(
        execution_dir / "execution_report.json",
        stage="executor",
        repair_iteration=repair_iteration,
    )
    layout.register(
        execution_dir / "execution_report.json",
        logical_name=f"executor_round_{repair_iteration:02d}_execution_report.json",
        stage="executor",
        repair_iteration=repair_iteration,
    )
    if execution_report.stdout_path:
        layout.register(
            _path_from_report(execution_report.stdout_path, execution_dir),
            stage="executor",
            repair_iteration=repair_iteration,
        )
    if execution_report.stderr_path:
        layout.register(
            _path_from_report(execution_report.stderr_path, execution_dir),
            stage="executor",
            repair_iteration=repair_iteration,
        )
    for value in execution_report.files.values():
        if isinstance(value, str) and value:
            layout.register(_path_from_report(value, execution_dir), stage="executor", repair_iteration=repair_iteration)


def _debug_evidence_path(stage: str, filename: str, repair_iteration: int | None = None) -> Path:
    if repair_iteration is None:
        return Path("evidence") / _ResearchArtifactLayout.STAGE_DIRS[stage] / filename
    return Path("evidence") / _ResearchArtifactLayout.STAGE_DIRS[stage] / f"round_{repair_iteration:02d}_{filename}"


def _write_debug_evidence(
    layout: _ResearchArtifactLayout,
    stage: str,
    filename: str,
    evidence: list[RetrievedEvidence],
    *,
    repair_iteration: int | None = None,
    logical_name: str | None = None,
) -> None:
    layout.write_debug_json(
        _debug_evidence_path(stage, filename, repair_iteration),
        _model_dump(evidence),
        logical_name=logical_name or filename,
        stage=stage,
        repair_iteration=repair_iteration,
    )


def _execution_summary_payload(execution_report: ExecutionReport) -> dict[str, Any]:
    return {
        "case_id": execution_report.case_id,
        "method": execution_report.method.value,
        "success": execution_report.success,
        "output_dir": execution_report.output_dir,
        "error_type": execution_report.error_type,
        "iterations": execution_report.iterations,
        "converged": execution_report.converged,
        "compliance": execution_report.compliance,
        "volume_fraction": execution_report.volume_fraction,
        "files": execution_report.files,
    }


def _evaluator_summary_payload(
    evaluator_report: EvaluatorReport,
    repair_plan: RepairPlan | None,
    *,
    evidence_count: int,
) -> dict[str, Any]:
    return {
        "case_id": evaluator_report.case_id,
        "success": evaluator_report.success,
        "has_quality_failure": evaluator_report.has_quality_failure,
        "failure_modes": [mode.value for mode in evaluator_report.failure_modes],
        "compliance": evaluator_report.compliance,
        "volume_error": evaluator_report.volume_error,
        "grayness_index": evaluator_report.grayness_index,
        "checkerboard_score": evaluator_report.checkerboard_score,
        "connectivity_score": evaluator_report.connectivity_score,
        "converged": evaluator_report.converged,
        "evidence_count": evidence_count,
        "evidence_ids": evaluator_report.evidence_ids,
        "repair_plan": None if repair_plan is None else {
            "should_repair": repair_plan.should_repair,
            "repair_iteration": repair_plan.repair_iteration,
            "repair_type": repair_plan.repair_type,
            "parameter_updates": repair_plan.parameter_updates,
            "evidence_ids": repair_plan.evidence_ids,
        },
    }


def _repair_plan_summary_payload(repair_plan: RepairPlan) -> dict[str, Any]:
    return {
        "case_id": repair_plan.case_id,
        "should_repair": repair_plan.should_repair,
        "repair_iteration": repair_plan.repair_iteration,
        "repair_type": repair_plan.repair_type,
        "parameter_updates": repair_plan.parameter_updates,
        "evidence_ids": repair_plan.evidence_ids,
        "risk_level": repair_plan.risk_level,
    }


def _result_source_image(execution_report: ExecutionReport, key: str, default_name: str) -> Path:
    execution_dir = Path(execution_report.output_dir)
    value = execution_report.files.get(key) or str(execution_dir / default_name)
    return _path_from_report(value, execution_dir)


def _copy_result_image(
    layout: _ResearchArtifactLayout,
    source: Path,
    filename: str,
    *,
    repair_iteration: int | None = None,
) -> str | None:
    if not source.exists() or not source.is_file():
        return None
    target = layout.stage_dir("result") / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    layout.register(target, logical_name=filename, stage="result", repair_iteration=repair_iteration)
    return layout._relative_path(target)


def _write_result_gallery_round(
    layout: _ResearchArtifactLayout,
    execution_report: ExecutionReport,
    *,
    repair_iteration: int,
) -> dict[str, Any]:
    density_source = _result_source_image(execution_report, "density_image", "density.png")
    history_source = _result_source_image(execution_report, "optimization_history_image", "optimization_history.png")
    entry: dict[str, Any] = {
        "round": repair_iteration,
        "source_output_dir": layout._relative_path(Path(execution_report.output_dir)),
        "success": execution_report.success,
        "execution_success": execution_report.success,
        "compliance": execution_report.compliance,
        "volume_fraction": execution_report.volume_fraction,
        "converged": execution_report.converged,
        "source_density_image": layout._relative_path(density_source),
        "source_optimization_history": layout._relative_path(history_source),
        "density_image": _copy_result_image(
            layout,
            density_source,
            f"round_{repair_iteration:02d}_density.png",
            repair_iteration=repair_iteration,
        ),
        "optimization_history_image": _copy_result_image(
            layout,
            history_source,
            f"round_{repair_iteration:02d}_optimization_history.png",
            repair_iteration=repair_iteration,
        ),
    }
    return entry


def _update_result_gallery_round(
    result_rounds: list[dict[str, Any]],
    *,
    repair_iteration: int,
    evaluator_report: EvaluatorReport,
) -> None:
    for entry in reversed(result_rounds):
        if entry.get("round") != repair_iteration:
            continue
        entry["quality_success"] = evaluator_report.success
        entry["success"] = bool(entry.get("execution_success")) and evaluator_report.success
        entry["compliance"] = evaluator_report.compliance if evaluator_report.compliance is not None else entry.get("compliance")
        entry["volume_error"] = evaluator_report.volume_error
        entry["grayness_index"] = evaluator_report.grayness_index
        entry["checkerboard_score"] = evaluator_report.checkerboard_score
        entry["connectivity_score"] = evaluator_report.connectivity_score
        entry["converged"] = evaluator_report.converged
        break


def _write_result_gallery_index(layout: _ResearchArtifactLayout, result_rounds: list[dict[str, Any]]) -> None:
    result_dir = layout.stage_dir("result")
    latest: dict[str, Any] = {}
    if result_rounds:
        final_round = result_rounds[-1]
        if final_round.get("density_image"):
            source = layout.root / final_round["density_image"]
            latest_path = _copy_result_image(layout, source, "latest_density.png")
            latest["source_density_image"] = final_round["density_image"]
            latest["density_image"] = latest_path
        if final_round.get("optimization_history_image"):
            source = layout.root / final_round["optimization_history_image"]
            latest_path = _copy_result_image(layout, source, "latest_optimization_history.png")
            latest["source_optimization_history_image"] = final_round["optimization_history_image"]
            latest["optimization_history_image"] = latest_path
    layout.write_json(
        result_dir / "result_index.json",
        {
            "schema_version": "research_result_index_v1",
            "rounds": result_rounds,
            "latest": latest,
        },
        logical_name="result_index.json",
        stage="result",
    )


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
    execution_success = bool(execution_report and execution_report.success)
    quality_success = bool(evaluator_report and evaluator_report.success)
    final_success = execution_success and quality_success
    return BenchmarkCaseResult(
        case_id=case_spec.case_id,
        benchmark_type=case_spec.benchmark_type,
        method=method,
        first_pass_success=first_pass_success,
        execution_success=execution_success,
        quality_success=quality_success,
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
    persist_debug_artifacts: bool = False,
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
        if isinstance(case_or_text, CaseSpec):
            case_spec = case_or_text
            case_spec_causality_full = _case_spec_input_causality_full(case_spec, quick=quick)
        else:
            case_spec, case_spec_causality_full = build_case_spec_with_causality(
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
            case_spec = case_spec.model_copy(update={"problem": case_to_problem(case_spec)})

        # 默认输出到项目 output 下的研究 workflow 独立子目录。
        if output_dir is None:
            output_dir = Path("output") / "research_graph" / f"{case_spec.case_id}__{method.value}"
        out = Path(output_dir)
        layout = _ResearchArtifactLayout(out, persist_debug_artifacts=persist_debug_artifacts)
        case_spec_causality = _compact_case_spec_causality(case_spec_causality_full, case_spec)
        case_spec_causality_full["normalized"] = {
            "artifact": "00_scientist/case_spec.json",
            "case_spec": case_spec.model_dump(mode="json"),
        }
        _write_case_spec_artifacts(
            layout,
            case_spec,
            case_spec_causality,
            full_causality=case_spec_causality_full,
        )
        layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")
    except Exception as exc:
        _fail(scientist_token, exc)
        raise
    _complete(
        scientist_token,
        summary="Scientist 完成 CaseSpec",
        payload={
            "case_id": case_spec.case_id,
            "benchmark_type": case_spec.benchmark_type.value,
            "parameters": _case_spec_scalar_summary(case_spec),
            "llm_trace_count": len(llm_agent_trace),
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
    result_rounds: list[dict[str, Any]] = []

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
            _extend_unique_evidence(all_evidence, validation_evidence)
        validator_dir = layout.stage_dir("validator")
        layout.write_json(
            validator_dir / "validation_report.json",
            validation_report.model_dump(mode="json"),
            stage="validator",
        )
        _write_debug_evidence(
            layout,
            "validator",
            "retrieved_evidence_validation.json",
            validation_evidence,
        )
        layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")
    except Exception as exc:
        _fail(validator_token, exc)
        raise
    _complete(
        validator_token,
        summary="Validator 检查完成",
        payload={
            "case_id": validation_report.case_id,
            "is_valid": validation_report.is_valid,
            "local_is_valid": validation_report.local_is_valid,
            "failure_modes": [mode.value for mode in validation_report.failure_modes],
            "severity": validation_report.severity.value,
            "evidence_count": len(validation_evidence),
            "evidence_ids": validation_report.evidence_ids,
        },
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
        planner_dir = layout.stage_dir("planner_coder")
        reviewer_dir = layout.round_dir("reviewer_repair", 0)
        evaluator_dir = layout.round_dir("evaluator", 0)
        summary_dir = layout.stage_dir("summary")
        _write_debug_evidence(layout, "planner_coder", "retrieved_evidence_codegen.json", [])
        _write_debug_evidence(layout, "planner_coder", "retrieved_evidence.json", all_evidence)
        layout.write_json(planner_dir / "code_plan.json", code_plan.model_dump(mode="json"), stage="planner_coder")
        _write_debug_evidence(
            layout,
            "reviewer_repair",
            "retrieved_evidence_execution_repair.json",
            [],
            repair_iteration=0,
        )
        _write_debug_evidence(
            layout,
            "evaluator",
            "retrieved_evidence_critic_repair.json",
            [],
            repair_iteration=0,
        )
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
        layout.write_json(
            summary_dir / "failure_diagnosis.json",
            final_diagnosis.model_dump(mode="json"),
            stage="summary",
        )
        layout.write_json(
            summary_dir / "repair_plan.json",
            final_repair_plan.model_dump(mode="json"),
            stage="summary",
        )
        layout.write_json(summary_dir / "repair_trace.json", [], stage="summary")
        layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")
        final_evaluator = _empty_evaluator(case_spec)
        layout.write_json(
            summary_dir / "evaluator_report.json",
            final_evaluator.model_dump(mode="json"),
            stage="summary",
        )
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
        final_summary_path = summary_dir / "final_summary.md"
        _write_final_summary(final_summary_path, result, repair_trace)
        layout.register(final_summary_path, stage="summary")
        _write_result_gallery_index(layout, result_rounds)
        layout.write_index()
        summary_token = _start("final_summary", "Reporter", "保存研究 workflow 摘要")
        _complete(
            summary_token,
            summary="研究 workflow fail-closed 完成",
            payload={
                "case_id": result.case_id,
                "method": result.method.value,
                "final_success": result.final_success,
                "execution_success": result.execution_success,
                "quality_success": result.quality_success,
                "repair_iterations": result.repair_iterations,
                "output_dir": result.output_dir,
            },
        )
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
        planner_dir = layout.stage_dir("planner_coder")
        if method in {BenchmarkMethod.BASELINE_NAIVE_RAG, BenchmarkMethod.OURS_CORRECTIVE_RAG}:
            codegen_evidence = retrieve_for_codegen(case_spec, retriever)
            _extend_unique_evidence(all_evidence, codegen_evidence)
        _write_debug_evidence(layout, "planner_coder", "retrieved_evidence_codegen.json", codegen_evidence)
        _write_debug_evidence(layout, "planner_coder", "retrieved_evidence.json", all_evidence)
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
            output_dir=planner_dir,
            agent_authority=authority,
            allow_generated_code=allow_generated_code,
            use_llm=_use_llm("coder"),
            llm_provider=llm_provider,
            llm=agent_llms.get("coder"),
            llm_overrides=llm_overrides,
            trace=llm_agent_trace,
        )
        if code_plan.generated_code_path:
            layout.register(Path(code_plan.generated_code_path), stage="planner_coder")
        if code_plan.generated_code_manifest_path:
            layout.register(Path(code_plan.generated_code_manifest_path), stage="planner_coder")
        layout.write_json(planner_dir / "code_plan.json", code_plan.model_dump(mode="json"), stage="planner_coder")
        layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")
    except Exception as exc:
        _fail(planner_token, exc)
        raise
    _complete(
        planner_token,
        summary="Planner/Coder 计划完成",
        payload={
            "case_id": code_plan.case_id,
            "method": code_plan.method.value,
            "engine": code_plan.engine,
            "template_id": code_plan.template_id,
            "execution_mode": code_plan.execution_mode,
            "generated_code": bool(code_plan.generated_code_path),
            "evidence_ids": code_plan.evidence_ids,
            "evidence_count": len(codegen_evidence),
            "llm_trace_count": len(llm_agent_trace),
        },
    )

    for repair_iteration in range(max_repair_rounds + 1):
        executor_dir = layout.round_dir("executor", repair_iteration)
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
                artifacts=[],
            )

        try:
            layout.write_json(
                executor_dir / "case_spec.json",
                case_spec.model_dump(mode="json"),
                logical_name=f"executor_round_{repair_iteration:02d}_case_spec.json",
                stage="executor",
                repair_iteration=repair_iteration,
            )
            layout.write_json(
                executor_dir / "code_plan.json",
                code_plan.model_dump(mode="json"),
                logical_name=f"executor_round_{repair_iteration:02d}_code_plan.json",
                stage="executor",
                repair_iteration=repair_iteration,
            )
            final_execution = execute(
                case_spec,
                code_plan,
                executor_dir,
                generated_code_timeout_s=generated_code_timeout_s,
                generated_code_sandbox_root=out,
                progress_callback=_progress_event,
            )
            _register_execution_artifacts(layout, final_execution, repair_iteration=repair_iteration)
            result_rounds.append(
                _write_result_gallery_round(
                    layout,
                    final_execution,
                    repair_iteration=repair_iteration,
                )
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
                layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")
        except Exception as exc:
            _fail(executor_token, exc)
            raise
        _complete(
            executor_token,
            summary=f"Executor 第 {repair_iteration} 轮完成",
            payload=_execution_summary_payload(final_execution),
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
                    "case_id": final_diagnosis.case_id,
                    "has_failure": final_diagnosis.has_failure,
                    "failure_modes": [mode.value for mode in final_diagnosis.failure_modes],
                    "severity": final_diagnosis.severity.value,
                    "repair_plan": _repair_plan_summary_payload(repair_plan),
                    "evidence_count": len(evidence),
                    "evidence_ids": final_diagnosis.evidence_ids,
                    "llm_trace_count": len(llm_agent_trace),
                },
            )
            final_repair_plan = repair_plan
            _extend_unique_evidence(execution_repair_evidence, evidence)
            _extend_unique_evidence(all_evidence, evidence)
            reviewer_dir = layout.round_dir("reviewer_repair", repair_iteration)
            layout.write_json(
                reviewer_dir / "failure_diagnosis.json",
                final_diagnosis.model_dump(mode="json"),
                stage="reviewer_repair",
                repair_iteration=repair_iteration,
            )
            _write_debug_evidence(
                layout,
                "reviewer_repair",
                "retrieved_evidence_execution_repair.json",
                evidence,
                logical_name=f"reviewer_round_{repair_iteration:02d}_retrieved_evidence_execution_repair.json",
                repair_iteration=repair_iteration,
            )
            layout.write_json(
                reviewer_dir / "repair_plan.json",
                final_repair_plan.model_dump(mode="json"),
                stage="reviewer_repair",
                repair_iteration=repair_iteration,
            )
            if method == BenchmarkMethod.OURS_CORRECTIVE_RAG and repair_plan.should_repair:
                repair_token = _start("repair", "Repair", "应用执行失败修复计划")
                repair_trace.append(repair_plan)
                case_spec = _apply_repair_and_record_case_spec(
                    layout,
                    case_spec,
                    repair_plan,
                    case_spec_causality,
                    case_spec_causality_full,
                )
                layout.write_json(
                    reviewer_dir / "repair_trace.json",
                    _model_dump(repair_trace),
                    logical_name=f"reviewer_round_{repair_iteration:02d}_repair_trace.json",
                    stage="reviewer_repair",
                    repair_iteration=repair_iteration,
                )
                _complete(
                    repair_token,
                    summary="执行失败修复计划已应用",
                    payload=_repair_plan_summary_payload(repair_plan),
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
                layout.write_json(
                    planner_dir / "code_plan.json",
                    code_plan.model_dump(mode="json"),
                    stage="planner_coder",
                )
                layout.write_json(
                    reviewer_dir / "repair_trace.json",
                    _model_dump(repair_trace),
                    logical_name=f"reviewer_round_{repair_iteration:02d}_repair_trace.json",
                    stage="reviewer_repair",
                    repair_iteration=repair_iteration,
                )
                _complete(repair_token, summary="已回退到模板执行", payload=_repair_plan_summary_payload(fallback_plan))
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
                    output_dir=planner_dir,
                    agent_authority=authority,
                    allow_generated_code=allow_generated_code,
                    use_llm=_use_llm("coder"),
                    llm_provider=llm_provider,
                    llm=agent_llms.get("coder"),
                    llm_overrides=llm_overrides,
                    trace=llm_agent_trace,
                )
                if code_plan.generated_code_path:
                    layout.register(Path(code_plan.generated_code_path), stage="planner_coder")
                if code_plan.generated_code_manifest_path:
                    layout.register(Path(code_plan.generated_code_manifest_path), stage="planner_coder")
                layout.write_json(
                    planner_dir / "code_plan.json",
                    code_plan.model_dump(mode="json"),
                    stage="planner_coder",
                )
                layout.write_json(
                    reviewer_dir / "repair_trace.json",
                    _model_dump(repair_trace),
                    logical_name=f"reviewer_round_{repair_iteration:02d}_repair_trace.json",
                    stage="reviewer_repair",
                    repair_iteration=repair_iteration,
                )
                layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")
                _complete(repair_token, summary="已重新生成代码计划", payload=_repair_plan_summary_payload(regenerate_plan))
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
            payload=_evaluator_summary_payload(final_evaluator, repair_plan, evidence_count=len(evidence)),
        )
        _update_result_gallery_round(result_rounds, repair_iteration=repair_iteration, evaluator_report=final_evaluator)
        if repair_plan is not None:
            final_repair_plan = repair_plan
        _extend_unique_evidence(critic_repair_evidence, evidence)
        _extend_unique_evidence(all_evidence, evidence)
        evaluator_dir = layout.round_dir("evaluator", repair_iteration)
        layout.write_json(
            evaluator_dir / "evaluator_report.json",
            final_evaluator.model_dump(mode="json"),
            stage="evaluator",
            repair_iteration=repair_iteration,
        )
        _write_debug_evidence(
            layout,
            "evaluator",
            "retrieved_evidence_critic_repair.json",
            evidence,
            logical_name=f"evaluator_round_{repair_iteration:02d}_retrieved_evidence_critic_repair.json",
            repair_iteration=repair_iteration,
        )
        layout.write_json(
            evaluator_dir / "repair_plan.json",
            final_repair_plan.model_dump(mode="json"),
            stage="evaluator",
            repair_iteration=repair_iteration,
        )
        if (
            method == BenchmarkMethod.OURS_CORRECTIVE_RAG
            and repair_plan is not None
            and repair_plan.should_repair
        ):
            repair_token = _start("repair", "Repair", "应用拓扑质量修复计划")
            repair_trace.append(repair_plan)
            case_spec = _apply_repair_and_record_case_spec(
                layout,
                case_spec,
                repair_plan,
                case_spec_causality,
                case_spec_causality_full,
            )
            layout.write_json(
                evaluator_dir / "repair_trace.json",
                _model_dump(repair_trace),
                logical_name=f"evaluator_round_{repair_iteration:02d}_repair_trace.json",
                stage="evaluator",
                repair_iteration=repair_iteration,
            )
            _complete(
                repair_token,
                summary="拓扑质量修复计划已应用",
                payload=_repair_plan_summary_payload(repair_plan),
            )
            continue
        break

    summary_dir = layout.stage_dir("summary")
    _write_debug_evidence(layout, "planner_coder", "retrieved_evidence_codegen.json", codegen_evidence)
    _write_debug_evidence(
        layout,
        "summary",
        "retrieved_evidence.json",
        all_evidence,
        logical_name="retrieved_evidence_all.json",
    )
    _write_debug_evidence(layout, "summary", "retrieved_evidence_execution_repair.json", execution_repair_evidence)
    _write_debug_evidence(layout, "summary", "retrieved_evidence_critic_repair.json", critic_repair_evidence)
    layout.write_json(
        summary_dir / "failure_diagnosis.json",
        (final_diagnosis or _empty_diagnosis(case_spec)).model_dump(mode="json"),
        stage="summary",
    )
    layout.write_json(summary_dir / "repair_plan.json", final_repair_plan.model_dump(mode="json"), stage="summary")
    layout.write_json(summary_dir / "repair_trace.json", _model_dump(repair_trace), stage="summary")
    layout.write_json(
        summary_dir / "evaluator_report.json",
        (final_evaluator or _empty_evaluator(case_spec)).model_dump(mode="json"),
        stage="summary",
    )
    layout.write_json(out / "llm_agent_trace.json", llm_agent_trace, stage="global")

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
    final_summary_path = summary_dir / "final_summary.md"
    _write_final_summary(final_summary_path, result, repair_trace)
    layout.register(final_summary_path, stage="summary")
    _write_result_gallery_index(layout, result_rounds)
    layout.write_index()
    summary_token = _start("final_summary", "Reporter", "保存研究 workflow 摘要")
    _complete(
        summary_token,
        summary="研究 workflow 完成",
        payload={
            "case_id": result.case_id,
            "method": result.method.value,
            "final_success": result.final_success,
            "execution_success": result.execution_success,
            "quality_success": result.quality_success,
            "repair_iterations": result.repair_iterations,
            "output_dir": result.output_dir,
        },
    )
    return result
