from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from autotopo.ui import HTML_PAGE, RunCreateRequest, RunManager, create_app, serve_ui


def test_ui_inline_script_is_valid_javascript(tmp_path):
    node = shutil.which("node")
    if node is None:
        return

    start = HTML_PAGE.index("<script>") + len("<script>")
    end = HTML_PAGE.index("</script>")
    script_path = tmp_path / "ui.js"
    script_path.write_text(HTML_PAGE[start:end], encoding="utf-8")
    result = subprocess.run([node, "--check", str(script_path)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_run_manager_starts_run_records_events_and_resolves_artifact(tmp_path, monkeypatch):
    def fake_research(self, request, tracer):
        artifact = tracer.output_dir / "artifact.txt"
        artifact.write_text("ok", encoding="utf-8")
        token = tracer.start_stage("fake_agent", agent="Fake", summary="fake 开始")
        tracer.complete_stage(token, summary="fake 完成", payload={"artifact_path": str(artifact)})
        return {"ok": True}

    monkeypatch.setattr(RunManager, "_run_research_workflow", fake_research)

    manager = RunManager(tmp_path)
    record = manager.create_run(
        RunCreateRequest(
            workflow_type="research",
            prompt="测试",
            quick=True,
            method="baseline_direct",
        )
    )
    run_id = record["run_id"]

    for _ in range(40):
        record = manager.get_run(run_id)
        if record["status"] == "completed":
            break
        time.sleep(0.05)
    assert record["status"] == "completed"

    events = manager.events_after(run_id)
    assert any(event["stage"] == "fake_agent" for event in events)
    assert any(event["status"] == "completed" for event in events)

    artifact = manager.resolve_artifact(run_id, "artifact.txt")
    assert artifact.read_text(encoding="utf-8") == "ok"

    with pytest.raises(FileNotFoundError):
        manager.resolve_artifact(run_id, "../run_record.json")


def test_create_app_registers_ui_routes(tmp_path):
    app = create_app(tmp_path)
    paths = {route.path for route in app.routes}

    assert app.state.manager.output_root == tmp_path
    assert "/" in paths
    assert "/api/runs" in paths
    assert "/api/runs/{run_id}/events" in paths
    assert "/api/runs/{run_id}/artifacts/{name:path}" in paths


def test_serve_ui_disables_uvicorn_access_log(monkeypatch, tmp_path):
    captured = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    serve_ui(host="127.0.0.1", port=8766, output=tmp_path)

    assert captured["access_log"] is False
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8766


def test_ui_rejects_second_active_run(tmp_path, monkeypatch):
    def slow_research(self, request, tracer):
        time.sleep(0.4)
        return {"ok": True}

    monkeypatch.setattr(RunManager, "_run_research_workflow", slow_research)
    manager = RunManager(tmp_path)

    first = manager.create_run(
        RunCreateRequest(workflow_type="research", prompt="第一个", method="baseline_direct")
    )
    assert first["run_id"]

    with pytest.raises(RuntimeError):
        manager.create_run(
            RunCreateRequest(workflow_type="research", prompt="第二个", method="baseline_direct")
        )
