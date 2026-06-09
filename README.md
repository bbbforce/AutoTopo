# AutoTopo

> 基于 LangGraph 与 FEniCS + dolfin-adjoint 的自主拓扑优化 AI 智能体工作流

## 简介

AutoTopo 是一个面向拓扑优化问题的 AI 智能体工作流。项目主流程运行在 WSL 中的 Python 虚拟环境，负责自然语言/图像输入解析、问题路由、理论推导、代码生成、仿真调度、视觉评估和结果归档；底层数值优化通过 Docker 容器 `dolfin-adjoint` 调用 FEniCS (DOLFIN 2019) + dolfin-adjoint 完成。

当前实现重点：

- LangGraph 编排端到端工作流，支持条件分支和视觉评估后的反馈重试。
- 输入解析节点将自然语言和可选设计域示意图转换为结构化问题定义。
- 路由节点区分标准拓扑优化问题和复杂约束问题，复杂路径会进入理论推导与代码生成节点。
- 仿真节点通过 `DolfinAdjointEngine` 将问题 JSON 发送到 Docker 容器执行。
- 容器内求解器使用 Gmsh 生成二维网格，使用 SIMP、Helmholtz 过滤和 SLSQP 执行拓扑优化。
- 视觉评估节点检查灰度单元、棋盘格、孤岛等缺陷，并可调整参数后重新求解。
- 支持 OpenAI、DeepSeek、GLM、阿里云百炼兼容接口和 Ollama 等 LLM provider 配置。

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
- `--max-retries`: 视觉评估反馈最大重试次数。
- `--provider`: 指定本次运行使用的 LLM provider。

### 6. 直接运行数值求解器

绕过 LLM 和 LangGraph，直接调用 FEniCS + dolfin-adjoint 后端：

```bash
python -m autotopo solve --preset cantilever --mesh-res 1.0 --volfrac 0.4 --penal 3 --rmin 0.05 --max-iter 200
```

内置预设：

- `cantilever`: 悬臂梁。
- `mbb`: 半 MBB 梁。
- `bridge`: 桥式受载结构。

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
│   ├── state.py                 # 全局状态类型
│   ├── llm_factory.py           # 多 LLM provider 工厂
│   ├── schemas/                 # Pydantic 问题定义模型
│   ├── engines/
│   │   ├── base.py              # 拓扑优化引擎抽象接口
│   │   ├── dolfin_adjoint_engine.py
│   │   │                          # Docker 调用代理
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

## License

MIT

## 规划
我要开发一个可以自主进行拓扑优化的 AI 智能体工作流
1.LangGraph框架，可以完美定义带有条件分支和循环反馈
2.支持多模态输入解析，并输出结构化的数据格式（如  YAML 格式，json格式）
3.支持问题规划与路由，分析解析出的结构化数据，判断当前优化库中是否已包含所需的约束或目标函数。如果只是常规的简单问题，直接调用标准库；如果是复杂的约束（如应力约束等等），则将任务路由给“代码生成模块”。
4.理论推导与代码生成模块，让一个agent进行力学公式推导，另一个agent基于公式生成符合底层仿真器接口规范的代码。
5.仿真求解与自动微分引擎，执行拓扑优化迭代计算。
6.视觉评估与反馈修正模块，利用支持视觉的LLM对单次优化完成后的结果图进行视觉检查，识别是否存在“灰度单元（中间密度）”、“棋盘格”或“孤岛”等力学缺陷，并自主提出修正方案，输出调整建议，将新参数注入仿真器重新运行，形成系统自闭环。
7.最后输出拓扑优化结果图储存至指定位置


项目主流程放在wsl中单独虚拟环境AT-env运行，求解器备选jax-fea或FEniCS (DOLFIN) + dolfin-adjoint，其中FEniCS (DOLFIN) + dolfin-adjoint已安装在docker容器dolfin-adjoint中
