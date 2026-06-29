# AutoTopo

> 基于 LangGraph 与 FEniCS + dolfin-adjoint 的自主拓扑优化 AI 智能体工作流

## 简介

AutoTopo 是一个面向拓扑优化问题的 AI 智能体工作流。项目主流程运行在 WSL 中的 Python 虚拟环境，负责自然语言/图像输入解析、问题路由、理论推导、代码生成、仿真调度、视觉评估和结果归档；底层数值优化通过 Docker 容器 `dolfin-adjoint` 调用 FEniCS (DOLFIN 2019) + dolfin-adjoint 完成。

当前实现重点：

- LangGraph 编排端到端工作流，支持条件分支和视觉评估后的反馈重试。
- 默认采用“预览再精修”求解策略：反馈闭环使用粗网格快速预览，最终结果再高精度求解一次。
- 输入解析节点将自然语言和可选设计域示意图转换为结构化问题定义。
- 路由节点区分标准拓扑优化问题和复杂约束问题，复杂路径会进入理论推导与代码生成节点。
- 仿真节点通过 `DolfinAdjointEngine` 将问题 JSON 发送到 Docker 容器执行。
- 容器内求解器使用 Gmsh 生成二维网格，使用 SIMP、Helmholtz 过滤和 SLSQP 执行拓扑优化。
- 视觉评估节点检查灰度单元、棋盘格、孤岛等缺陷，并可调整参数后重新求解。
- 支持 OpenAI、DeepSeek、GLM、阿里云百炼兼容接口和 Ollama 等 LLM provider 配置。
- 新增独立的最小研究 workflow：`Scientist → Validator → Planner → Coder → Executor → Reviewer/Evaluator → bounded repair`。
- 新增 `PythonSimpMMAEngine`，支持 MBB 梁、悬臂梁和 L 型梁的结构化小型 benchmark，使用 Python SIMP + MMA。
- 新增本地关键词 RAG、失败模式诊断、拓扑质量指标和最小 benchmark 输出，用于验证 CaseSpec-aware RAG 与 corrective repair 机制。

## 系统架构

```
用户输入
  ↓
parse_input
  ↓
route_problem
  ├─ standard_path → run_simulation → evaluate_result ── accept → save_output
  │                                      ↑
  │                                      └─ retry ← apply_fixes
  └─ complex_path → theory_derivation → code_generation → run_simulation

run_simulation:
  WSL / AT-env 主流程
    ↓ docker cp / docker exec
  Docker 容器 dolfin-adjoint
    ↓
  FEniCS (DOLFIN) + dolfin-adjoint 求解器
```

最小研究 workflow 独立于上面的 `graph.py` 主流程，入口为 `src/autotopo/research_graph.py`：

```
自然语言/CaseSpec
  ↓
Scientist          # 可选 LLM 解析；默认规则解析，结构化参数优先
  ↓
Validator          # fail-closed 物理检查
  ↓
Planner / Coder    # 可选 LLM 规划；锁定已有 benchmark 模板和 PythonSimpMMAEngine
  ↓
Executor           # 执行 Python SIMP + MMA，不把异常抛崩 workflow
  ↓
Reviewer/Evaluator # 可选 LLM 失败诊断、质量评估、bounded repair
  ↓
输出 case artifact 与 summary
```

研究 workflow 默认保持 deterministic/rule-based，以保证最小实验可复现；如需真实 LLM，可显式启用 Scientist、Planner、Reviewer 的结构化 LLM 路径。LLM 调用失败、输出无法通过 schema 校验或未显式启用时，会自动回退到 deterministic agent。

## 快速开始

### 1. 准备 Python 环境

项目推荐在 WSL 中的 Conda 虚拟环境 `AT-env` 运行主流程：

```bash
cd ~/AutoTopo
conda activate AT-env
pip install -e ".[dev]"
```

`pyproject.toml` 中只声明主流程所需的 Python 依赖。FEniCS (DOLFIN 2019) + dolfin-adjoint 由 Docker 容器提供，不通过 pip 安装。

### 2. 准备 Docker 求解器容器

当前数值后端默认使用容器名为 `dolfin-adjoint` 的 Docker 容器。运行求解前请确认容器已启动，并且容器内可使用 `python3` 导入 `dolfin`、`dolfin_adjoint`、`gmsh`、`meshio`、`numpy` 和 `matplotlib`。

```bash
docker ps
```

主流程会在运行时把 `src/autotopo/engines/solver_runner.py` 复制到容器的 `/tmp/autotopo/`，然后通过 `docker exec` 调用容器内求解器。

### 3. 配置 API Key

如果需要运行完整 AI 工作流，在项目根目录创建 `.env` 文件，填入对应 provider 的 API Key：

```env
# OpenAI / DeepSeek / GLM / 阿里云百炼
OPENAI_API_KEY=sk-your-openai-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
GLM_API_KEY=your-glm-key
DASHSCOPE_API_KEY=your-dashscope-key
```

纯数值求解命令 `python -m autotopo solve ...` 不依赖 LLM，也不需要 API Key。

### 4. 选择 LLM Provider

编辑 `config/settings.yaml`：

```yaml
llm:
  default_provider: "bailian"
  vision_provider: "bailian"
```

可选 provider 包括 `openai`、`deepseek`、`glm`、`bailian` 和 `ollama`。视觉评估使用 `vision_provider`，应选择支持图像输入的模型。

### 5. 运行完整 AI 工作流

```bash
python -m autotopo run "标准半对称 MBB 梁拓扑优化问题。设计域尺寸为 60x20，左侧为对称边界，右下角施加竖向约束，左上角承受向下集中力。目标为最小化柔度，体积分数约束为 0.5，penal=3，rmin=0.05，max_iter=200。" --max-retries 5
```

常用参数：

- `--images`: 可选设计域示意图路径，支持多张图片。
- `--output`: 输出目录，默认 `./output`。
- `--max-retries`: 预览阶段视觉评估反馈最大重试次数，默认 `2`。
- `--solve-profile`: 求解模式，默认 `preview_refine`；可选 `preview_refine`、`final_only`、`preview_only`。
- `--provider`: 指定本次运行使用的 LLM provider。

### 6. 启动实时监控前端

本地前端用于在浏览器里启动主 workflow 或最小研究 workflow，并实时查看阶段进度、agent 输出、错误信息、图片和 JSON 产物。建议在 `AT-env` 中运行：

```bash
conda run --no-capture-output -n AT-env python -m autotopo ui --host 0.0.0.0 --port 8766
```

在 Windows 浏览器中优先打开启动日志里的 `Windows 浏览器可尝试` 地址，例如 `http://172.18.8.129:8766/`。如果你的 WSL localhost 转发正常，也可以打开 `http://127.0.0.1:8766/`。

- 主 workflow：展示 `parse_input`、`route_problem`、`run_simulation`、`evaluate_result`、`apply_fixes`、`save_output` 等阶段。
- 研究 workflow：展示 `Scientist`、`Validator`、`Planner/Coder`、`Executor`、`Reviewer`、`Evaluator`、`Repair` 等 agent 阶段。

UI 默认输出到 `output/ui_runs/`。每个 run 会保存：

- `workflow_events.jsonl`: 实时 timeline 事件，可用于刷新后回放。
- `run_record.json`: run 状态、请求参数和最终结果。
- workflow 原有结果文件，例如 `problem_definition.yaml`、`report.md`、`case_spec.json`、`code_plan.json`、`density.png`。

### 7. 直接运行数值求解器

绕过 LLM 和 LangGraph，直接调用 FEniCS + dolfin-adjoint 后端：

```bash
python -m autotopo solve --preset cantilever --mesh-res 1.0 --volfrac 0.4 --penal 3 --rmin 0.05 --max-iter 200
```

内置预设：

- `cantilever`: 悬臂梁。
- `mbb`: 半 MBB 梁。
- `bridge`: 桥式受载结构。

可使用 `--profile preview` 进行快速预览求解，例如：

```bash
python -m autotopo solve --preset mbb --profile preview --max-iter 20
```

### 8. 运行最小研究 benchmark

最小研究 benchmark 不依赖 Docker，也不依赖 LLM API Key。它使用本地 `PythonSimpMMAEngine`、本地关键词 RAG 和 rule-based agents，默认输出到项目内的 `output/minimal_benchmark/`。

```bash
python -m autotopo.experiments.run_minimal_benchmark
```

默认会运行 6 个 case × 3 个方法：

- case：MBB 梁 clear/fuzzy、悬臂梁 clear/fuzzy、L 型梁 clear/fuzzy。
- method：`baseline_direct`、`baseline_naive_rag`、`ours_corrective_rag`。

默认非 quick 算例用于查看拓扑优化结果形状：

- MBB / 悬臂梁：`60 × 20` 单元，`max_iter=100`。
- L 型梁：`40 × 40` 单元，`max_iter=100`。

快速冒烟测试可以使用：

```bash
python -m autotopo.experiments.run_minimal_benchmark --quick
```

`--quick` 会使用很小的网格和很少迭代，只用于验证流程是否跑通，不适合查看最终拓扑形状。

如需在最小研究 workflow 中启用真实 LLM agent：

```bash
python -m autotopo.experiments.run_minimal_benchmark --quick --llm-agents --provider bailian
```

`--llm-agents` 会让 Scientist、Planner、Reviewer 先尝试结构化 LLM 输出；任一 agent 调用失败都会回退到本地 deterministic 逻辑，且每个 case-method 会写出 `llm_agent_trace.json`。

也可以指定输出目录：

```bash
python -m autotopo.experiments.run_minimal_benchmark --output output/minimal_benchmark
```

单个 case 可直接调用 `research_graph`：

```python
from autotopo.research_graph import run_research_workflow

result = run_research_workflow(
    "做一个悬臂梁拓扑优化，左边固定，右端向下受力",
    method="ours_corrective_rag",
    use_llm_agents=True,
    llm_provider="bailian",
    structured_params={
        "benchmark_type": "cantilever",
        "volume_fraction": 0.4,
        "max_iter": 100,
    },
)

print(result)
```

## 项目结构

```
AutoTopo/
├── .env                         # 本地 API Key，已被 .gitignore 忽略
├── config/
│   └── settings.yaml            # LLM、引擎、评估和输出配置
├── examples/
│   ├── ai_workflow.py           # AI 工作流示例
│   ├── cantilever_beam.py       # 悬臂梁示例
│   └── mbb_beam.py              # MBB 梁示例
├── src/autotopo/
│   ├── __main__.py              # CLI 入口：run / solve
│   ├── graph.py                 # LangGraph 工作流定义
│   ├── research_graph.py        # 最小研究 workflow
│   ├── state.py                 # 全局状态类型
│   ├── llm_factory.py           # 多 LLM provider 工厂
│   ├── agents/                  # rule-based Scientist / Validator / Planner / Coder / Executor / Reviewer / Evaluator
│   ├── rag/                     # 本地关键词 RAG 与 corrective RAG
│   ├── diagnostics/             # 失败模式、拓扑指标和修复规则
│   ├── experiments/             # 最小 benchmark 入口
│   ├── schemas/                 # Pydantic 问题定义模型
│   ├── engines/
│   │   ├── base.py              # 拓扑优化引擎抽象接口
│   │   ├── dolfin_adjoint_engine.py
│   │   │                          # Docker 调用代理
│   │   ├── python_simp_mma_engine.py
│   │   │                          # Python SIMP + MMA 研究后端
│   │   ├── structured_benchmarks.py
│   │   │                          # MBB / 悬臂梁 / L 型梁结构化 benchmark
│   │   └── solver_runner.py     # 容器内 FEniCS 求解脚本
│   ├── nodes/
│   │   ├── input_parser.py      # 输入解析
│   │   ├── router.py            # 问题路由
│   │   ├── theory_agent.py      # 理论推导
│   │   ├── codegen_agent.py     # 代码生成
│   │   ├── simulator.py         # 仿真调度
│   │   └── evaluator.py         # 视觉评估与参数修正
│   ├── library/                 # 标准目标函数与约束占位库
│   └── utils/                   # I/O 与可视化工具
├── knowledge_base/              # 本地 RAG 知识库
├── tests/
├── output/                      # 默认运行输出目录
└── pyproject.toml               # 包元数据与 Python 依赖
```

## 配置说明

| 配置项 | 位置 | 说明 |
|--------|------|------|
| API Key | `.env` 或 `config/settings.yaml` | 各 LLM provider 的密钥，配置文件为空时回退到环境变量 |
| LLM provider | `config/settings.yaml.llm` | 默认 provider、视觉 provider、模型名、base_url、温度等 |
| 仿真后端 | `config/settings.yaml.engine.backend` | 当前默认 `dolfin_adjoint` |
| 求解 profile | `config/settings.yaml.engine.profiles` | `preview` 用于反馈闭环快速预览，`final` 用于最终精修 |
| 早停设置 | `config/settings.yaml.engine.early_stop` | 控制最小迭代数、窗口长度和目标函数相对变化阈值 |
| 默认优化参数 | `config/settings.yaml.engine.default_params` | `mesh_resolution`、`volfrac`、`penal`、`rmin`、`max_iter`、`tol`、`optimizer` |
| 视觉评估 | `config/settings.yaml.evaluation` | 最大重试次数和缺陷阈值 |
| 输出设置 | `config/settings.yaml.output` | 输出目录、图片格式、DPI 和是否保存中间结果 |

## 输出文件

完整工作流每次运行会在输出目录下创建时间戳子目录，例如 `output/run_YYYYMMDD_HHMMSS/`。主要产物包括：

- `problem_definition.yaml`: 解析后的结构化问题定义。
- `result_iter_*.png`: 每轮优化密度场结果图。
- `convergence_iter_*.png`: 每轮收敛历史图。
- `convergence_history.png`: 最终汇总收敛图。
- `evaluation_history.json`: 视觉评估与参数修正历史。
- `report.md`: 自动生成的优化报告。

直接求解命令会创建 `output/solve_YYYYMMDD_HHMMSS/`，并导出预设问题的结果图。

最小研究 benchmark 默认写入 `output/minimal_benchmark/`：

- `summary.csv`: 全部 case-method 的机器可读汇总。
- `summary.md`: 全部 case-method 的 Markdown 汇总。
- `{case_id}__{method}/case_spec.json`: 本轮 CaseSpec。
- `{case_id}__{method}/validation_report.json`: Validator 检查结果。
- `{case_id}__{method}/retrieved_evidence.json`: 本地 RAG 聚合检索证据。
- `{case_id}__{method}/retrieved_evidence_codegen.json`: 代码生成/模板选择阶段证据。
- `{case_id}__{method}/retrieved_evidence_execution_repair.json`: 执行失败修复阶段证据。
- `{case_id}__{method}/retrieved_evidence_critic_repair.json`: 拓扑质量修复阶段证据。
- `{case_id}__{method}/retrieved_evidence_validation.json`: Validator fail-closed 阶段证据。
- `{case_id}__{method}/code_plan.json`: Planner/Coder 选择的执行计划。
- `{case_id}__{method}/llm_agent_trace.json`: 可选 LLM agent 使用情况与 fallback 原因。
- `{case_id}__{method}/execution_report.json`: Executor 执行报告。
- `{case_id}__{method}/failure_diagnosis.json`: 失败诊断结果。
- `{case_id}__{method}/repair_plan.json`: 最近一次有界修复建议。
- `{case_id}__{method}/repair_trace.json`: bounded repair 轨迹。
- `{case_id}__{method}/evaluator_report.json`: 拓扑质量与优化有效性评估。
- `{case_id}__{method}/density.npy`: 连续密度场数组。
- `{case_id}__{method}/density.png`: 连续密度场图片，不做二值化阈值过滤。
- `{case_id}__{method}/optimization_history.csv`: 每轮 compliance、volume、change。
- `{case_id}__{method}/optimization_history.png`: 优化历史曲线图。
- `{case_id}__{method}/final_summary.md`: 单个 case-method 的最终摘要。

当前本地 RAG 默认不需要安装 embedding 模型。`src/autotopo/rag/retriever.py` 提供 lexical、可选 dense 和 hybrid 检索：无 embedding 后端时自动退回关键词检索；注入 embedding model 或安装 `autotopo[rag]` 并显式配置本地模型后，可启用 dense 融合。对外仍返回统一的 `RetrievedEvidence`，其中保留旧字段并增加 `parent_id`、`chunk_id`、score breakdown 和 rerank features 以便审计。

## License

MIT

## 后续规划

当前仓库已经包含两条互不破坏的路径：

- 面向完整 AI 工作流的 `graph.py` 路径：使用 LangGraph、LLM 结构化解析、FEniCS + dolfin-adjoint Docker 后端和视觉评估反馈。
- 面向最小研究验证的 `research_graph.py` 路径：使用 rule-based agents、本地 RAG、Python SIMP + MMA 后端、失败诊断和 bounded repair。

下一步可逐步增强：

1. 用更标准的 GCMMA/MMA 实现替换当前本地 MMA-style 更新器。
2. 扩展 benchmark 到更多二维/三维算例，并加入更严格的连通性、载荷路径和制造约束指标。
3. 将研究 workflow 的结果与完整 LangGraph 主流程打通，形成可复现的 preview → repair → final refine 实验链路。
