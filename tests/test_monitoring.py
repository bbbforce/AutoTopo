from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotopo.monitoring import WorkflowTracer, read_jsonl


def test_workflow_tracer_persists_events_and_record(tmp_path):
    tracer = WorkflowTracer(
        run_id="run123",
        workflow_type="research",
        output_dir=tmp_path,
        request={"prompt": "测试"},
    )
    token = tracer.start_stage("scientist", agent="Scientist", summary="开始")
    tracer.complete_stage(token, summary="完成", payload={"answer": 1})
    tracer.set_status("completed", result={"ok": True})

    events = read_jsonl(tmp_path / "workflow_events.jsonl")
    assert [event["status"] for event in events] == ["running", "completed"]
    assert events[0]["seq"] == 1
    assert events[1]["payload"] == {"answer": 1}

    record = json.loads((tmp_path / "run_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["result"] == {"ok": True}


def test_workflow_tracer_artifact_safety(tmp_path):
    tracer = WorkflowTracer(run_id="run123", workflow_type="main", output_dir=tmp_path)
    artifact_file = tmp_path / "report.md"
    artifact_file.write_text("# report", encoding="utf-8")

    artifact = tracer.artifact_for_path(artifact_file)
    assert artifact is not None
    assert artifact["name"] == "report.md"
    assert tracer.resolve_artifact("report.md") == artifact_file.resolve()

    with pytest.raises(FileNotFoundError):
        tracer.resolve_artifact("../outside.txt")


def test_workflow_tracer_respects_explicit_empty_artifacts(tmp_path):
    tracer = WorkflowTracer(run_id="run123", workflow_type="research", output_dir=tmp_path)
    artifact_file = tmp_path / "artifact_index.json"
    artifact_file.write_text("{}", encoding="utf-8")

    event = tracer.emit(
        stage="optimization_iteration",
        status="running",
        payload={"iteration": 1, "case_id": "mbb_clear"},
        artifacts=[],
    )

    events = read_jsonl(tmp_path / "workflow_events.jsonl")
    assert event["artifacts"] == []
    assert events[0]["artifacts"] == []
