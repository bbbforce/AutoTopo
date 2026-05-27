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
- 🔌 **多 LLM 支持** — OpenAI / DeepSeek / GLM / Ollama 一键切换

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

复制 `.env` 文件并填入你的 API 密钥：

```bash
cp .env .env.local   # 可选：保留一份本地副本
```

编辑 `.env`：

```env
OPENAI_API_KEY=sk-your-key
DEEPSEEK_API_KEY=sk-your-key
GLM_API_KEY=your-key
```

### 3. 切换 LLM Provider

编辑 `config/settings.yaml` 中的 `default_provider` 字段：

```yaml
llm:
  default_provider: "deepseek"   # openai / deepseek / glm / ollama
```

### 4. 运行

```python
from autotopo.graph import compile_graph

app = compile_graph()
result = app.invoke({
    "user_input": "设计一个60x20的悬臂梁，左端固定，右端中点施加向下单位力，体积分数0.5",
    "image_paths": [],
    "max_retries": 3,
    "output_path": "./output",
})
```

## 项目结构

```
AutoTopo/
├── .env                        # API 密钥（不提交到 Git）
├── config/settings.yaml        # 全局配置
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
