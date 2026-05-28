# AutoTopo

> 基于 LangGraph 的自主拓扑优化 AI 智能体工作流

## 简介

AutoTopo 是一个端到端的 AI 驱动拓扑优化系统，支持从自然语言问题描述到优化结果输出的全自动流程。

**核心特性：**

- 🔄 **LangGraph 工作流编排** — 条件分支 + 循环反馈闭环
- 🖼️ **多模态输入解析** — 文本 + 设计域示意图 → 结构化 YAML/JSON
- 🧠 **智能问题路由** — 简单问题走标准库，复杂约束自动推导
- ⚙️ **理论推导 + 代码生成** — 双 Agent 协作：力学公式推导 → 引擎代码
- 🚀 **2D SIMP 仿真引擎** — 内置优化好的完整求解器（支持灰度单元和棋盘格自动过滤）
- 👁️ **视觉评估反馈** — VLM 自动检测缺陷，基于“只增不减 + 步进限幅”策略进行智能参数修正，闭环迭代
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
python -m autotopo run "设计一个 120x40 的悬臂梁，左端固定，右端中点施加向下单位力，体积分数 0.4，最小化柔度。" --max-retries 5
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
│   ├── __main__.py             # 命令行工具入口
│   ├── graph.py                # LangGraph 核心状态机与工作流拓扑
│   ├── state.py                # 全局状态字典
│   ├── llm_factory.py          # 统一的多 Provider LLM 适配器
│   ├── schemas/                # Pydantic 结构化数据定义
│   ├── engines/                # 仿真引擎（内置高效 SIMP 求解器）
│   ├── nodes/                  # 工作流各智能体节点实现
│   └── utils/                  # 可视化绘图与 IO 模块
├── tests/                      # 完善的单元测试与 Mock 集成测试
└── output/                     # 优化密度图与历史评估记录输出
```

## 闭环自适应机制

评估模块中内置了**防震荡优化机制**：
1. **参数只增不减**：为防止 VLM 给出相反的优化方向导致参数来回震荡，对于惩罚因子 (`penal`) 和过滤半径 (`rmin`) 采取只增不减的安全更新策略。
2. **步进限幅**：单次参数调整最大不超过当前值的 `50%`，避免调整过激引起拓扑畸变。

## License

MIT
