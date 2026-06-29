"""本地 FastAPI 监控前端。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from autotopo.monitoring import TERMINAL_STATUSES, WorkflowTracer, jsonable
from autotopo.schemas import BenchmarkMethod


class RunCreateRequest(BaseModel):
    """前端启动 run 的请求体。"""

    workflow_type: Literal["main", "research"] = "main"
    prompt: str = ""
    images: list[str] = Field(default_factory=list)
    provider: str | None = None
    max_retries: int = Field(default=2, ge=0, le=20)
    solve_profile: Literal["preview_refine", "final_only", "preview_only"] = "preview_refine"
    quick: bool = True
    llm_agents: bool = False
    method: BenchmarkMethod = BenchmarkMethod.OURS_CORRECTIVE_RAG
    max_repair_rounds: int = Field(default=3, ge=0, le=10)
    structured_params: dict[str, Any] = Field(default_factory=dict)


class RunManager:
    """管理 UI server 当前进程中的 run。"""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tracers: dict[str, WorkflowTracer] = {}
        self._threads: dict[str, threading.Thread] = {}

    def create_run(self, request: RunCreateRequest) -> dict[str, Any]:
        with self._lock:
            active = self._active_run_id_locked()
            if active is not None:
                raise RuntimeError(f"当前已有运行中的任务: {active}")

            run_id = uuid.uuid4().hex[:12]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.output_root / f"{stamp}_{request.workflow_type}_{run_id}"
            tracer = WorkflowTracer(
                run_id=run_id,
                workflow_type=request.workflow_type,
                output_dir=run_dir,
                request=request.model_dump(mode="json"),
            )
            self._tracers[run_id] = tracer

            thread = threading.Thread(
                target=self._run_target,
                args=(request, tracer),
                name=f"autotopo-ui-{run_id}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()
            return tracer.record

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            records = [tracer.record for tracer in self._tracers.values()]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get_tracer(run_id).record

    def events_after(self, run_id: str, seq: int = 0) -> list[dict[str, Any]]:
        return self._get_tracer(run_id).events_after(seq)

    def resolve_artifact(self, run_id: str, name: str) -> Path:
        return self._get_tracer(run_id).resolve_artifact(name)

    def _get_tracer(self, run_id: str) -> WorkflowTracer:
        with self._lock:
            tracer = self._tracers.get(run_id)
        if tracer is None:
            raise KeyError(run_id)
        return tracer

    def _active_run_id_locked(self) -> str | None:
        for run_id, tracer in self._tracers.items():
            if tracer.record["status"] not in TERMINAL_STATUSES:
                return run_id
        return None

    def _run_target(self, request: RunCreateRequest, tracer: WorkflowTracer) -> None:
        tracer.set_status("running")
        tracer.emit(
            stage="run",
            agent="UI",
            status="running",
            summary="运行已启动",
            payload=request.model_dump(mode="json"),
        )
        try:
            if request.workflow_type == "main":
                result = self._run_main_workflow(request, tracer)
            else:
                result = self._run_research_workflow(request, tracer)
        except Exception as exc:  # noqa: BLE001 - UI 需要把后台异常写入事件流
            tracer.emit(
                stage="run",
                agent="UI",
                status="failed",
                summary="运行失败",
                error=f"{type(exc).__name__}: {exc}",
            )
            tracer.set_status("failed", error=f"{type(exc).__name__}: {exc}")
            return

        tracer.emit(
            stage="run",
            agent="UI",
            status="completed",
            summary="运行完成",
            payload=result,
        )
        tracer.set_status("completed", result=result)

    def _run_main_workflow(self, request: RunCreateRequest, tracer: WorkflowTracer) -> dict[str, Any]:
        from autotopo.graph import compile_graph

        if not request.prompt.strip():
            raise ValueError("主 workflow 需要 prompt。")

        initial_stage = "final" if request.solve_profile == "final_only" else "preview"
        initial_state = {
            "user_input": request.prompt,
            "image_paths": request.images,
            "llm_provider": request.provider,
            "max_retries": request.max_retries,
            "solve_profile": request.solve_profile,
            "solve_stage": initial_stage,
            "final_refine_done": request.solve_profile != "preview_refine",
            "output_path": str(tracer.output_dir),
            "iteration": 0,
            "history": [],
        }
        result = compile_graph(tracer=tracer).invoke(initial_state)
        return jsonable(result)

    def _run_research_workflow(self, request: RunCreateRequest, tracer: WorkflowTracer) -> dict[str, Any]:
        from autotopo.research_graph import run_research_workflow

        prompt = request.prompt.strip() or "运行一个快速悬臂梁 benchmark"
        result = run_research_workflow(
            prompt,
            output_dir=tracer.output_dir,
            method=request.method,
            structured_params=request.structured_params or None,
            quick=request.quick,
            max_repair_rounds=request.max_repair_rounds,
            use_llm_agents=request.llm_agents,
            llm_provider=request.provider if request.llm_agents else None,
            tracer=tracer,
        )
        return result.model_dump(mode="json")


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoTopo 运行监控</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f9;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #627084;
      --line: #d9e0e8;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --danger: #b42318;
      --ok: #15803d;
      --warn: #a16207;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 680;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 56px);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .controls {
      padding: 14px;
      align-self: start;
    }
    .workspace {
      display: grid;
      grid-template-rows: minmax(280px, 42vh) minmax(320px, 1fr);
      gap: 16px;
      background: transparent;
      border: 0;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 620;
      margin: 12px 0 6px;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea {
      min-height: 132px;
      resize: vertical;
      line-height: 1.45;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .segmented {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 10px;
    }
    .segmented button {
      border: 0;
      background: #fff;
      border-radius: 0;
      padding: 10px;
      cursor: pointer;
      color: var(--muted);
      font-weight: 650;
    }
    .segmented button.active {
      background: #e7f5f3;
      color: var(--accent);
    }
    .checkline {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .checkline input {
      width: 16px;
      height: 16px;
    }
    .primary {
      width: 100%;
      margin-top: 14px;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: #fff;
      padding: 11px 12px;
      font-weight: 720;
      cursor: pointer;
    }
    .primary:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 12px;
      overflow-wrap: anywhere;
    }
    .timeline {
      overflow: auto;
      padding: 10px;
    }
    .event {
      display: grid;
      grid-template-columns: 92px 1fr auto;
      gap: 10px;
      align-items: start;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
    }
    .event:hover, .event.selected {
      background: #f0f7f6;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 76px;
      height: 24px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      color: #fff;
    }
    .running { background: var(--accent-2); }
    .completed { background: var(--ok); }
    .failed { background: var(--danger); }
    .queued { background: var(--warn); }
    .event-title {
      font-weight: 700;
      font-size: 14px;
    }
    .event-summary {
      color: var(--muted);
      font-size: 13px;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }
    .time {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .detail {
      overflow: auto;
      padding: 14px;
    }
    .detail h2 {
      margin: 0 0 8px;
      font-size: 16px;
    }
    .artifact-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
      margin: 12px 0;
    }
    .artifact {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      min-width: 0;
      background: #fff;
    }
    .artifact img {
      width: 100%;
      max-height: 180px;
      object-fit: contain;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 7px;
      background: #fafafa;
    }
    .artifact a {
      color: var(--accent-2);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #0f172a;
      color: #e5e7eb;
      overflow: auto;
      font-size: 12px;
      line-height: 1.5;
      min-height: 140px;
    }
    .payload-details {
      margin: 12px 0;
    }
    .payload-details summary {
      cursor: pointer;
      color: var(--accent-2);
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
    }
    .payload-box {
      min-height: 80px;
      max-height: 180px;
    }
    .console-box {
      min-height: 180px;
      max-height: 260px;
      white-space: pre-wrap;
    }
    .hidden { display: none; }
    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }
      .workspace {
        grid-template-rows: auto auto;
      }
      .event {
        grid-template-columns: 82px 1fr;
      }
      .time {
        grid-column: 2;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>AutoTopo 运行监控</h1>
    <div id="headerStatus" class="status">未启动</div>
  </header>
  <main>
    <section class="controls">
      <div class="segmented" role="tablist" aria-label="workflow">
        <button id="mainTab" class="active" type="button">主 workflow</button>
        <button id="researchTab" type="button">研究 workflow</button>
      </div>
      <form id="runForm">
        <label for="prompt">输入</label>
        <textarea id="prompt" name="prompt">标准半对称 MBB 梁拓扑优化问题。设计域尺寸为 60x20，目标是最小化柔度，体积分数约束为 0.5。</textarea>
        <label for="provider">Provider</label>
        <input id="provider" name="provider" placeholder="留空使用 settings.yaml">
        <div id="mainOptions">
          <label for="images">图片路径</label>
          <input id="images" name="images" placeholder="每行或逗号分隔">
          <div class="row">
            <div>
              <label for="solveProfile">求解模式</label>
              <select id="solveProfile">
                <option value="preview_refine">preview_refine</option>
                <option value="final_only">final_only</option>
                <option value="preview_only">preview_only</option>
              </select>
            </div>
            <div>
              <label for="maxRetries">最大重试</label>
              <input id="maxRetries" type="number" min="0" max="20" value="2">
            </div>
          </div>
        </div>
        <div id="researchOptions" class="hidden">
          <div class="row">
            <div>
              <label for="method">方法</label>
              <select id="method">
                <option value="ours_corrective_rag">ours_corrective_rag</option>
                <option value="baseline_naive_rag">baseline_naive_rag</option>
                <option value="baseline_direct">baseline_direct</option>
              </select>
            </div>
            <div>
              <label for="maxRepairRounds">修复轮数</label>
              <input id="maxRepairRounds" type="number" min="0" max="10" value="3">
            </div>
          </div>
          <label class="checkline"><input id="quick" type="checkbox" checked> quick</label>
          <label class="checkline"><input id="llmAgents" type="checkbox"> LLM agents</label>
        </div>
        <button class="primary" id="startButton" type="submit">启动</button>
      </form>
      <div id="runStatus" class="status">等待启动。</div>
    </section>
    <section class="workspace">
      <section class="timeline" id="timeline"></section>
      <section class="detail" id="detail">
        <h2>事件详情</h2>
        <div class="status">选择 timeline 中的事件查看输出。</div>
        <h2 style="margin-top: 16px;">优化迭代输出</h2>
        <pre id="progressConsole" class="console-box">等待优化迭代...</pre>
      </section>
    </section>
  </main>
  <script>
    const state = {
      workflowType: "main",
      runId: null,
      events: [],
      progressEvents: [],
      selectedSeq: null,
      payloadOpen: false,
      progressFollow: true,
      progressScrollTop: 0,
      source: null
    };
    const $ = (id) => document.getElementById(id);
    const statusNames = { queued: "排队", running: "运行中", completed: "完成", failed: "失败" };

    function setWorkflow(type) {
      state.workflowType = type;
      $("mainTab").classList.toggle("active", type === "main");
      $("researchTab").classList.toggle("active", type === "research");
      $("mainOptions").classList.toggle("hidden", type !== "main");
      $("researchOptions").classList.toggle("hidden", type !== "research");
    }

    function artifactUrl(name) {
      return `/api/runs/${state.runId}/artifacts/${name.split("/").map(encodeURIComponent).join("/")}`;
    }

    function renderTimeline() {
      const root = $("timeline");
      if (!state.events.length) {
        root.innerHTML = '<div class="status">暂无事件。</div>';
        return;
      }
      root.innerHTML = state.events.map((event) => {
        const selected = event.seq === state.selectedSeq ? " selected" : "";
        const status = statusNames[event.status] || event.status;
        const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "";
        const agent = event.agent ? ` · ${event.agent}` : "";
        return `<div class="event${selected}" data-seq="${event.seq}">
          <span class="badge ${event.status}">${status}</span>
          <div>
            <div class="event-title">${event.stage || "run"}${agent}</div>
            <div class="event-summary">${event.summary || ""}</div>
            ${event.error ? `<div class="event-summary" style="color: var(--danger)">${event.error}</div>` : ""}
          </div>
          <div class="time">${time}${event.duration_ms ? ` · ${event.duration_ms}ms` : ""}</div>
        </div>`;
      }).join("");
      root.querySelectorAll(".event").forEach((node) => {
        node.addEventListener("click", () => selectEvent(Number(node.dataset.seq)));
      });
    }

    function formatMetric(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "?";
      return Number(number.toPrecision(4)).toString();
    }

    function progressLine(event) {
      const payload = event.payload || {};
      return `优化迭代 ${payload.iteration}/${payload.max_iter}: compliance=${formatMetric(payload.compliance)}, volume=${formatMetric(payload.volume)}, change=${formatMetric(payload.change)}`;
    }

    function progressText() {
      if (!state.progressEvents.length) return "等待优化迭代...";
      return state.progressEvents.map(progressLine).join("\\n");
    }

    function renderProgressLog() {
      const consoleNode = $("progressConsole");
      if (!consoleNode) return;
      const wasFollowing = state.progressFollow || isProgressNearBottom(consoleNode);
      consoleNode.textContent = progressText();
      if (wasFollowing) {
        consoleNode.scrollTop = consoleNode.scrollHeight;
        state.progressFollow = true;
      } else {
        consoleNode.scrollTop = Math.min(state.progressScrollTop, consoleNode.scrollHeight);
      }
    }

    function isProgressNearBottom(node) {
      return node.scrollHeight - node.scrollTop - node.clientHeight <= 12;
    }

    function bindProgressConsole() {
      const consoleNode = $("progressConsole");
      if (!consoleNode) return;
      consoleNode.scrollTop = Math.min(state.progressScrollTop, consoleNode.scrollHeight);
      if (state.progressFollow) consoleNode.scrollTop = consoleNode.scrollHeight;
      consoleNode.addEventListener("scroll", () => {
        state.progressScrollTop = consoleNode.scrollTop;
        state.progressFollow = isProgressNearBottom(consoleNode);
      });
    }

    function rememberProgress(event) {
      const idx = state.progressEvents.findIndex((item) => item.seq === event.seq);
      if (idx >= 0) {
        state.progressEvents[idx] = event;
      } else {
        state.progressEvents.push(event);
      }
      state.progressEvents.sort((a, b) => a.seq - b.seq);
    }

    function progressBoxHtml() {
      return `<h2 style="margin-top: 16px;">优化迭代输出</h2><pre id="progressConsole" class="console-box">${escapeHtml(progressText())}</pre>`;
    }

    function renderArtifacts(event) {
      const artifacts = event.artifacts || [];
      if (!artifacts.length) return "";
      return `<div class="artifact-grid">${artifacts.map((item) => {
        const url = artifactUrl(item.name);
        if (item.kind === "image") {
          return `<div class="artifact"><img src="${url}" alt="${item.label || item.name}"><a href="${url}" target="_blank">${item.label || item.name}</a></div>`;
        }
        return `<div class="artifact"><a href="${url}" target="_blank">${item.label || item.name}</a><div class="status">${item.kind} · ${item.size || 0} bytes</div></div>`;
      }).join("")}</div>`;
    }

    function selectEvent(seq) {
      const event = state.events.find((item) => item.seq === seq);
      if (!event) return;
      const previousSeq = state.selectedSeq;
      if (previousSeq !== seq) state.payloadOpen = false;
      state.selectedSeq = seq;
      renderTimeline();
      $("detail").innerHTML = `<h2>${event.stage || "run"}${event.agent ? ` · ${event.agent}` : ""}</h2>
        <div class="status">${event.summary || ""}</div>
        ${event.error ? `<div class="status" style="color: var(--danger)">${event.error}</div>` : ""}
        ${renderArtifacts(event)}
        <details class="payload-details" ${state.payloadOpen ? "open" : ""}>
          <summary>结构化 payload</summary>
          <pre class="payload-box">${escapeHtml(JSON.stringify(event.payload ?? {}, null, 2))}</pre>
        </details>
        ${progressBoxHtml()}`;
      const details = $("detail").querySelector(".payload-details");
      if (details) {
        details.addEventListener("toggle", () => {
          state.payloadOpen = details.open;
        });
      }
      bindProgressConsole();
      renderProgressLog();
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    function connectEvents(runId) {
      if (state.source) state.source.close();
      state.source = new EventSource(`/api/runs/${runId}/events`);
      state.source.addEventListener("workflow", (message) => {
        const event = JSON.parse(message.data);
        if (event.stage === "optimization_iteration") {
          rememberProgress(event);
          renderProgressLog();
          $("headerStatus").textContent = `${runId} · 优化迭代 ${event.payload?.iteration || ""}/${event.payload?.max_iter || ""}`;
          return;
        }
        const idx = state.events.findIndex((item) => item.seq === event.seq);
        if (idx >= 0) state.events[idx] = event; else state.events.push(event);
        state.events.sort((a, b) => a.seq - b.seq);
        const shouldSelectFirstEvent = state.selectedSeq === null;
        if (shouldSelectFirstEvent) state.selectedSeq = event.seq;
        renderTimeline();
        if (shouldSelectFirstEvent) selectEvent(state.selectedSeq);
        $("headerStatus").textContent = `${runId} · ${statusNames[event.status] || event.status}`;
        if (event.stage === "run" && ["completed", "failed", "cancelled"].includes(event.status)) {
          $("startButton").disabled = false;
        }
      });
      state.source.onerror = () => {
        $("runStatus").textContent = "事件流已关闭或等待重连。";
      };
    }

    async function startRun(event) {
      event.preventDefault();
      $("startButton").disabled = true;
      state.events = [];
      state.progressEvents = [];
      state.selectedSeq = null;
      state.payloadOpen = false;
      state.progressFollow = true;
      state.progressScrollTop = 0;
      renderTimeline();
      $("detail").innerHTML = `<h2>事件详情</h2>
        <div class="status">选择 timeline 中的事件查看输出。</div>
        ${progressBoxHtml()}`;
      bindProgressConsole();
      const images = $("images").value.split(new RegExp("[,\\\\n]")).map((item) => item.trim()).filter(Boolean);
      const payload = {
        workflow_type: state.workflowType,
        prompt: $("prompt").value,
        provider: $("provider").value.trim() || null,
        images,
        solve_profile: $("solveProfile").value,
        max_retries: Number($("maxRetries").value || 2),
        method: $("method").value,
        quick: $("quick").checked,
        llm_agents: $("llmAgents").checked,
        max_repair_rounds: Number($("maxRepairRounds").value || 3)
      };
      try {
        const response = await fetch("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error(await response.text());
        const record = await response.json();
        state.runId = record.run_id;
        $("runStatus").textContent = `已启动：${record.run_id}`;
        connectEvents(record.run_id);
      } catch (err) {
        $("runStatus").textContent = `启动失败：${err.message}`;
        $("startButton").disabled = false;
      }
    }

    $("mainTab").addEventListener("click", () => setWorkflow("main"));
    $("researchTab").addEventListener("click", () => setWorkflow("research"));
    $("runForm").addEventListener("submit", startRun);
    setWorkflow("main");
  </script>
</body>
</html>
"""


def _wsl_access_urls(host: str, port: int) -> list[str]:
    """生成 Windows 浏览器可尝试访问的 WSL 地址。"""

    if host not in {"0.0.0.0", "::"}:
        return []
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except Exception:
        return []

    urls = []
    for item in result.stdout.split():
        if "." in item and not item.startswith("127."):
            urls.append(f"http://{item}:{port}")
    return urls


def create_app(output_root: str | Path = "./output/ui_runs"):
    """创建 FastAPI app。"""

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

    manager = RunManager(output_root)
    app = FastAPI(title="AutoTopo UI")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML_PAGE

    @app.post("/api/runs")
    def create_run(request: RunCreateRequest) -> dict[str, Any]:
        try:
            return manager.create_run(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return manager.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return manager.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run 不存在") from exc

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str):
        try:
            manager.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run 不存在") from exc

        async def generate():
            last_seq = 0
            idle_after_terminal = 0
            while True:
                events = manager.events_after(run_id, last_seq)
                for event in events:
                    last_seq = event["seq"]
                    data = json.dumps(event, ensure_ascii=False)
                    yield f"id: {event['seq']}\nevent: workflow\ndata: {data}\n\n"

                record = manager.get_run(run_id)
                if record["status"] in TERMINAL_STATUSES:
                    idle_after_terminal += 1
                    if idle_after_terminal >= 2:
                        break
                await asyncio.sleep(0.35)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/artifacts/{name:path}")
    def get_artifact(run_id: str, name: str):
        try:
            path = manager.resolve_artifact(run_id, name)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="artifact 不存在") from exc
        return FileResponse(path)

    return app


def serve_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output: str | Path = "./output/ui_runs",
    open_browser: bool = False,
) -> None:
    """启动本地 UI server。"""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给 CLI 清晰错误
        raise RuntimeError("缺少 uvicorn，请在 AT-env 中安装项目依赖后再运行 UI。") from exc

    app = create_app(output)
    url = f"http://{host}:{port}"
    print(f"AutoTopo UI: {url}", flush=True)
    for access_url in _wsl_access_urls(host, port):
        print(f"Windows 浏览器可尝试: {access_url}", flush=True)
    print(f"输出目录: {Path(output)}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, access_log=False)
