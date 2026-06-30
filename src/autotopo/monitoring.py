"""AutoTopo 运行时事件记录与 artifact 安全访问。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    """返回 ISO-8601 UTC 时间戳。"""

    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any, *, max_string: int = 8000, max_items: int = 80) -> Any:
    """把任意 Python 对象转换为适合写入 JSON 的小型表示。"""

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > max_string:
            return value[:max_string] + f"... [truncated {len(value) - max_string} chars]"
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}

    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"), max_string=max_string, max_items=max_items)

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        return {"type": "ndarray", "shape": list(shape or []), "dtype": str(dtype)}

    if isinstance(value, dict):
        items = list(value.items())
        payload = {
            str(key): jsonable(item, max_string=max_string, max_items=max_items)
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            payload["_truncated_items"] = len(items) - max_items
        return payload

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        payload = [jsonable(item, max_string=max_string, max_items=max_items) for item in seq[:max_items]]
        if len(seq) > max_items:
            payload.append({"_truncated_items": len(seq) - max_items})
        return payload

    return repr(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 事件文件，忽略空行。"""

    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


@dataclass(frozen=True)
class StageToken:
    """阶段计时 token。"""

    stage: str
    agent: str | None
    started_at: float


class WorkflowTracer:
    """记录 workflow timeline，并持久化到 run 输出目录。"""

    def __init__(
        self,
        *,
        run_id: str,
        workflow_type: str,
        output_dir: str | Path,
        request: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.workflow_type = workflow_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "workflow_events.jsonl"
        self.record_path = self.output_dir / "run_record.json"
        self._lock = RLock()
        self._seq = 0
        self._events: list[dict[str, Any]] = []
        self._record: dict[str, Any] = {
            "run_id": run_id,
            "workflow_type": workflow_type,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "output_dir": str(self.output_dir),
            "request": jsonable(request or {}),
            "error": None,
            "result": None,
        }
        self._write_record()

    @property
    def record(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._record)

    def events_after(self, seq: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in self._events if event["seq"] > seq]

    def set_status(
        self,
        status: str,
        *,
        error: str | None = None,
        result: Any | None = None,
    ) -> None:
        """更新 run 级状态。"""

        with self._lock:
            self._record["status"] = status
            self._record["updated_at"] = utc_now()
            if error is not None:
                self._record["error"] = error
            if result is not None:
                self._record["result"] = jsonable(result)
            self._write_record()

    def start_stage(
        self,
        stage: str,
        *,
        agent: str | None = None,
        summary: str = "",
        payload: Any | None = None,
    ) -> StageToken:
        self.emit(
            stage=stage,
            agent=agent,
            status="running",
            summary=summary or f"{stage} 开始",
            payload=payload,
        )
        return StageToken(stage=stage, agent=agent, started_at=time.perf_counter())

    def complete_stage(
        self,
        token: StageToken | None,
        *,
        stage: str | None = None,
        agent: str | None = None,
        summary: str = "",
        payload: Any | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        duration_ms = None
        if token is not None:
            duration_ms = int((time.perf_counter() - token.started_at) * 1000)
            stage = stage or token.stage
            agent = agent or token.agent
        self.emit(
            stage=stage or "",
            agent=agent,
            status="completed",
            summary=summary or f"{stage} 完成",
            payload=payload,
            artifacts=artifacts,
            duration_ms=duration_ms,
        )

    def fail_stage(
        self,
        token: StageToken | None,
        exc: BaseException,
        *,
        stage: str | None = None,
        agent: str | None = None,
        payload: Any | None = None,
    ) -> None:
        duration_ms = None
        if token is not None:
            duration_ms = int((time.perf_counter() - token.started_at) * 1000)
            stage = stage or token.stage
            agent = agent or token.agent
        self.emit(
            stage=stage or "",
            agent=agent,
            status="failed",
            summary=f"{stage} 失败",
            payload=payload,
            duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {exc}",
        )

    def emit(
        self,
        *,
        stage: str,
        status: str,
        agent: str | None = None,
        summary: str = "",
        payload: Any | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """追加一个 timeline 事件。"""

        with self._lock:
            self._seq += 1
            event = {
                "run_id": self.run_id,
                "workflow_type": self.workflow_type,
                "seq": self._seq,
                "timestamp": utc_now(),
                "stage": stage,
                "agent": agent,
                "status": status,
                "summary": summary,
                "payload": jsonable(payload),
                "artifacts": artifacts if artifacts is not None else self.collect_artifacts(payload),
                "duration_ms": duration_ms,
                "error": error,
            }
            self._events.append(event)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(self._console_line(event), flush=True)
            return event

    def _console_line(self, event: dict[str, Any]) -> str:
        """生成后端终端中的一行实时事件日志。"""

        agent = f" · {event['agent']}" if event.get("agent") else ""
        error = f" | {event['error']}" if event.get("error") else ""
        return (
            f"[{event['run_id']} #{event['seq']}] "
            f"{event['status']} {event['stage']}{agent}: {event.get('summary', '')}{error}"
        )

    def artifact_for_path(self, value: str | Path, *, label: str | None = None) -> dict[str, Any] | None:
        """把输出目录内的文件转换为前端可访问 artifact。"""

        raw = Path(value)
        path = raw if raw.is_absolute() else self.output_dir / raw
        try:
            resolved = path.resolve()
            root = self.output_dir.resolve()
            rel = resolved.relative_to(root)
        except (OSError, ValueError):
            return None

        if not resolved.exists() or not resolved.is_file():
            return None

        suffix = resolved.suffix.lower()
        kind = "file"
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            kind = "image"
        elif suffix in {".json", ".jsonl", ".yaml", ".yml", ".csv", ".md", ".log", ".txt"}:
            kind = "text"

        return {
            "name": rel.as_posix(),
            "label": label or rel.name,
            "kind": kind,
            "size": resolved.stat().st_size,
        }

    def collect_artifacts(self, payload: Any | None) -> list[dict[str, Any]]:
        """从常见 path/files 字段里提取 artifact。"""

        found: dict[str, dict[str, Any]] = {}

        def add(value: Any, label: str | None = None) -> None:
            if not isinstance(value, (str, Path)):
                return
            artifact = self.artifact_for_path(value, label=label)
            if artifact is not None:
                found[artifact["name"]] = artifact

        def walk(value: Any, key: str | None = None) -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    child_key_str = str(child_key)
                    if (
                        child_key_str.endswith("_path")
                        or child_key_str in {"path", "density", "density_image", "result_json"}
                        or child_key_str.endswith("_image")
                        or child_key_str.endswith("_history")
                    ):
                        add(child_value, label=child_key_str)
                    walk(child_value, child_key_str)
            elif isinstance(value, list):
                for item in value:
                    walk(item, key)

        walk(payload)
        for name in [
            "problem_definition.yaml",
            "evaluation_history.json",
            "report.md",
            "final_summary.md",
            "case_spec.json",
            "validation_report.json",
            "code_plan.json",
            "llm_agent_trace.json",
            "execution_report.json",
            "failure_diagnosis.json",
            "repair_plan.json",
            "repair_trace.json",
            "evaluator_report.json",
            "density.png",
            "optimization_history.png",
            "result.json",
            "artifact_index.json",
            "00_scientist/case_spec.json",
            "00_scientist/case_spec_causality.json",
            "01_validator/validation_report.json",
            "02_planner_coder/code_plan.json",
            "02_planner_coder/retrieved_evidence.json",
            "03_executor/round_00/execution_report.json",
            "03_executor/round_00/density.png",
            "03_executor/round_00/optimization_history.png",
            "05_evaluator/round_00/evaluator_report.json",
            "06_summary/failure_diagnosis.json",
            "06_summary/repair_plan.json",
            "06_summary/repair_trace.json",
            "06_summary/final_summary.md",
            "result/result_index.json",
            "result/latest_density.png",
            "result/latest_optimization_history.png",
        ]:
            add(name)
        return list(found.values())

    def resolve_artifact(self, name: str) -> Path:
        """解析 artifact 路径，禁止逃逸出 run 输出目录。"""

        path = self.output_dir / name
        resolved = path.resolve()
        root = self.output_dir.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise FileNotFoundError(name) from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(name)
        return resolved

    def _write_record(self) -> None:
        self.record_path.write_text(
            json.dumps(self._record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
