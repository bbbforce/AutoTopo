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
    from fastapi.staticfiles import StaticFiles

    manager = RunManager(output_root)
    app = FastAPI(title="AutoTopo UI")
    app.state.manager = manager

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

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
