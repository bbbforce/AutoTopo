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
  return state.progressEvents.map(progressLine).join("\n");
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
  const images = $("images").value.split(new RegExp("[,\\n]")).map((item) => item.trim()).filter(Boolean);
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
    persist_debug_artifacts: $("persistDebugArtifacts").checked,
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
