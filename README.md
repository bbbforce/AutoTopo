# AutoTopo

> 基于 LangGraph 的自主拓扑优化 AI 智能体工作流

## 简介

AutoTopo 是一个端到端的 AI 驱动拓扑优化系统，支持从自然语言问题描述到优化结果输出的全自动流程。

**核心特性：**

- 🔄 **LangGraph 工作流编排** — 条件分支 + 循环反馈闭环
- 🖼️ **多模态输入解析** — 文本 + 设计域示意图 → 结构化 YAML/JSON
- 🧠 **智能问题路由** — 简单问题走标准库，复杂约束自动推导
- ⚙️ **理论推导 + 代码生成** — 双 Agent 协作：力学公式推导 → 引擎代码
- 🚀 **2D SIMP 仿真引擎** — 内置完整求解器（后续支持 JAX-FEM / FEALPy）
- 👁️ **视觉评估反馈** — VLM 自动检测灰度单元/棋盘格/孤岛，闭环修正
- 🔌 **多 LLM 支持** — 支持 OpenAI、DeepSeek、GLM 以及阿里云百炼（Qwen）等主流大模型

## 系统架构

```
用户输入 → 多模态解析 → 问题路由 →┬→ 仿真求解 → 视觉评估 →┬→ 输出结果
                                   │                        │
                                   └→ 理论推导 → 代码生成 ──┘→ 参数修正 ↺
```

## 快速开始

### 1. 安装

```bash
pip install -e ".[dev]"
```

### 2. 配置 API Key

在项目根目录下创建 `.env` 文件，填入你的 API 密钥：

```env
# OpenAI / DeepSeek / GLM / 百炼 API Keys
OPENAI_API_KEY=sk-your-openai-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
GLM_API_KEY=your-glm-key
DASHSCOPE_API_KEY=your-dashscope-key
```

### 3. 切换 LLM Provider

编辑 `config/settings.yaml` 中的 `default_provider` 和 `vision_provider` 字段：

```yaml
llm:
  default_provider: "bailian"   # openai / deepseek / glm / bailian / ollama
  vision_provider: "bailian"
```

### 4. 运行方式

#### A. 完整 AI 智能体工作流 (通过 CLI)

运行端到端工作流：
```bash
python -m autotopo run "标准半对称MBB梁拓扑优化问题。设计域尺寸为150x50，采用150x50网格划分。左侧为对称边界，右下角施加垂直约束，左上角承受向下集中力。目标为最小化柔度，体积分数约束为0.5。penal=3, rmin=4.0, max_iter=400。" --max-retries 5
```

#### B. 纯数值仿真求解 (无需 LLM)

直接运行内置的 2D SIMP 求解器（内置 cantilever / mbb / bridge 预设）：
```bash
python -m autotopo solve --preset cantilever --nelx 80 --nely 40 --volfrac 0.4
```

## 项目结构

```
AutoTopo/
├── .env                        # API 密钥（已被 .gitignore 忽略）
├── config/settings.yaml        # 全局模型与算法参数配置
├── examples/                   # 包含 cantilever, mbb 以及 AI 流程的测试用例
├── src/autotopo/
│   ├── graph.py                # LangGraph 主工作流
│   ├── state.py                # 全局状态定义
│   ├── llm_factory.py          # 多 LLM Provider 工厂
│   ├── schemas/                # Pydantic 数据模型
│   ├── engines/                # 仿真引擎（SIMP / JAX-FEM）
│   ├── nodes/                  # 工作流节点
│   ├── library/                # 标准约束/目标函数库
│   └── utils/                  # 可视化 & I/O 工具
├── tests/
└── output/                     # 优化结果输出
```

## 配置说明

| 配置项 | 位置 | 说明 |
|--------|------|------|
| API 密钥 | `.env` | 各 LLM Provider 的 API Key |
| LLM 选择 | `config/settings.yaml` | 默认 Provider、模型名、温度 |
| 仿真参数 | `config/settings.yaml` | 网格尺寸、罚因子、过滤半径等 |
| 视觉评估 | `config/settings.yaml` | 最大重试次数（默认 3） |

## License

MIT
