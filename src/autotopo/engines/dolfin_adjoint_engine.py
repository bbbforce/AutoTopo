"""FEniCS (DOLFIN) + dolfin-adjoint Docker 远程调用代理。

主控逻辑运行在本地 AT-env，底层优化通过 docker exec 发送到
常驻的 dolfin-adjoint 容器执行。

数据流：
    1. 序列化问题定义为 JSON
    2. docker cp → 容器 /tmp/autotopo/
    3. docker exec → solver_runner.py
    4. docker cp ← 结果文件 (result.json + PNG + npy)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from autotopo.engines.base import OptResult, TopoEngine


# Docker 容器名 (固定)
CONTAINER_NAME = "dolfin-adjoint"

# 容器内的工作目录
CONTAINER_WORK_DIR = "/tmp/autotopo"

# solver_runner.py 在宿主机上的路径
_SOLVER_RUNNER_PATH = Path(__file__).parent / "solver_runner.py"


class DolfinAdjointEngine(TopoEngine):
    """FEniCS + dolfin-adjoint 引擎 (Docker 远程调用代理)。

    本类不直接调用 FEniCS，而是通过 docker exec 将计算任务
    发送给 dolfin-adjoint 容器中的 solver_runner.py 脚本。
    """

    def __init__(self, container: str = CONTAINER_NAME) -> None:
        self.container = container
        self._problem: dict[str, Any] = {}
        self._params_override: dict[str, Any] = {}
        self._result: Optional[dict[str, Any]] = None
        self._local_output_dir: Optional[str] = None
        self._density_grid: Optional[np.ndarray] = None

    # ────────── 接口实现 ──────────

    def setup(self, problem: dict[str, Any]) -> None:
        """存储问题定义 (实际计算在 optimize 时发送给容器)。"""
        self._problem = dict(problem)

        # 从约束中提取 volfrac 到 parameters 中
        params = dict(problem.get("parameters", {}))
        for c in problem.get("constraints", []):
            if c.get("type") == "volume_fraction":
                params.setdefault("volfrac", c.get("value", 0.5))
        self._problem["parameters"] = params

        # 向后兼容：nelx/nely → mesh_resolution
        domain = self._problem.get("domain", {})
        if "mesh_resolution" not in domain and "nelx" in domain:
            domain["mesh_resolution"] = domain.get("width", 60.0) / domain["nelx"]
            self._problem["domain"] = domain

        # 确保 solver_runner.py 已部署到容器
        self._deploy_solver()

    def optimize(
        self,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        penal: Optional[float] = None,
        rmin: Optional[float] = None,
        volfrac: Optional[float] = None,
    ) -> OptResult:
        """通过 Docker 容器执行拓扑优化。"""

        # 合并参数覆盖
        params = dict(self._problem.get("parameters", {}))
        if penal is not None:
            params["penal"] = penal
        if rmin is not None:
            params["rmin"] = rmin
        if volfrac is not None:
            params["volfrac"] = volfrac
        params["max_iter"] = max_iter
        params["tol"] = tol

        problem = dict(self._problem)
        problem["parameters"] = params

        # 创建本地临时目录
        local_tmp = tempfile.mkdtemp(prefix="autotopo_run_")
        self._local_output_dir = local_tmp

        # 写入问题定义 JSON
        problem_json_path = os.path.join(local_tmp, "problem.json")
        with open(problem_json_path, "w") as f:
            json.dump(problem, f, indent=2, ensure_ascii=False)

        # 容器内路径
        container_input = f"{CONTAINER_WORK_DIR}/problem.json"
        container_output = f"{CONTAINER_WORK_DIR}/output"

        # ── Step 1: 准备容器工作目录 ──
        self._docker_exec(f"mkdir -p {CONTAINER_WORK_DIR}")

        # ── Step 2: 复制问题定义到容器 ──
        self._docker_cp_to(problem_json_path, container_input)

        # ── Step 3: 在容器内运行求解器 ──
        print(f"🔧 发送优化任务到 Docker 容器 [{self.container}]...")
        print(f"   penal={params.get('penal', 3.0)}, rmin={params.get('rmin', 0.05)}, "
              f"volfrac={params.get('volfrac', 0.5)}, max_iter={max_iter}")

        cmd = (
            f"python3 {CONTAINER_WORK_DIR}/solver_runner.py "
            f"--input {container_input} "
            f"--output-dir {container_output}"
        )
        returncode, stdout, stderr = self._docker_exec(cmd, capture=True)

        if stdout:
            print(stdout)
        if returncode != 0:
            error_msg = f"容器内求解失败 (exit code {returncode})"
            if stderr:
                error_msg += f"\n{stderr}"
            raise RuntimeError(error_msg)

        # ── Step 4: 从容器复制结果文件 ──
        local_output = os.path.join(local_tmp, "output")
        os.makedirs(local_output, exist_ok=True)

        for fname in ["result.json", "density.png", "convergence.png", "density_grid.npy"]:
            try:
                self._docker_cp_from(f"{container_output}/{fname}",
                                     os.path.join(local_output, fname))
            except subprocess.CalledProcessError:
                pass  # 某些文件可能不存在

        # ── Step 5: 解析结果 ──
        result_json_path = os.path.join(local_output, "result.json")
        if not os.path.exists(result_json_path):
            raise RuntimeError("容器未生成 result.json")

        with open(result_json_path, "r") as f:
            self._result = json.load(f)

        if "error" in self._result:
            raise RuntimeError(f"容器内求解错误: {self._result['error']}")

        # 加载密度场
        npy_path = os.path.join(local_output, "density_grid.npy")
        if os.path.exists(npy_path):
            self._density_grid = np.load(npy_path)

        return OptResult(
            densities=self._density_grid if self._density_grid is not None else np.array([]),
            compliance_history=self._result.get("compliance_history", []),
            volume_history=self._result.get("volume_history", []),
            iterations=self._result.get("iterations", 0),
            converged=self._result.get("converged", False),
            mesh_info=self._result.get("mesh_info", {}),
        )

    def get_density_field(self) -> np.ndarray:
        """返回密度场 numpy 数组。"""
        if self._density_grid is None:
            raise RuntimeError("尚未执行优化")
        return self._density_grid

    def export_image(self, path: str, dpi: int = 300) -> str:
        """将容器生成的密度场图复制到指定路径。"""
        if self._local_output_dir is None:
            raise RuntimeError("尚未执行优化")

        src = os.path.join(self._local_output_dir, "output", "density.png")
        if os.path.exists(src):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, path)
        else:
            # 回退：使用 density_grid 重新绘制
            from autotopo.utils.visualization import plot_density_field
            plot_density_field(self.get_density_field(), path, dpi=dpi)

        return path

    def get_convergence_image(self, path: str) -> str:
        """将容器生成的收敛图复制到指定路径。"""
        if self._local_output_dir is None:
            raise RuntimeError("尚未执行优化")

        src = os.path.join(self._local_output_dir, "output", "convergence.png")
        if os.path.exists(src):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, path)
        else:
            # 回退：使用收敛历史重新绘制
            if self._result:
                from autotopo.utils.visualization import plot_convergence_history
                plot_convergence_history(
                    self._result.get("compliance_history", []),
                    self._result.get("volume_history", []),
                    path,
                )
        return path

    # ────────── Docker 操作 ──────────

    def _deploy_solver(self) -> None:
        """将 solver_runner.py 部署到容器内。"""
        container_solver = f"{CONTAINER_WORK_DIR}/solver_runner.py"
        try:
            self._docker_exec(f"mkdir -p {CONTAINER_WORK_DIR}")
            self._docker_cp_to(str(_SOLVER_RUNNER_PATH), container_solver)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"无法将 solver_runner.py 部署到容器 [{self.container}]。"
                f"请确保容器正在运行: docker ps | grep {self.container}"
            ) from e

    def _docker_exec(self, cmd: str, *, capture: bool = False):
        """在 Docker 容器内执行命令。"""
        full_cmd = ["docker", "exec", self.container, "bash", "-c", cmd]

        if capture:
            proc = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=3600,
            )
            return proc.returncode, proc.stdout, proc.stderr
        else:
            subprocess.run(full_cmd, check=True, timeout=60)
            return 0, "", ""

    def _docker_cp_to(self, local_path: str, container_path: str) -> None:
        """从宿主机复制文件到容器。"""
        subprocess.run(
            ["docker", "cp", local_path, f"{self.container}:{container_path}"],
            check=True, timeout=30,
        )

    def _docker_cp_from(self, container_path: str, local_path: str) -> None:
        """从容器复制文件到宿主机。"""
        subprocess.run(
            ["docker", "cp", f"{self.container}:{container_path}", local_path],
            check=True, timeout=30,
        )
