"""示例：完整 AI 工作流运行（需要配置 LLM API Key）。

演示从自然语言到优化结果的端到端流程。
"""

from autotopo.graph import compile_graph


def run_simple_example():
    """简单问题：直接走标准路径。"""
    app = compile_graph()

    result = app.invoke({
        "user_input": (
            "请设计一个二维悬臂梁的拓扑优化。"
            "设计域尺寸为宽60高20，使用60x20的网格。"
            "左端完全固定，右端中点施加竖直向下的单位集中力。"
            "材料杨氏模量E=1，泊松比ν=0.3。"
            "目标是最小化柔度，约束体积分数不超过50%。"
            "使用SIMP方法，罚因子p=3，过滤半径比例rmin=0.05。"
        ),
        "image_paths": [],
        "max_retries": 3,
        "output_path": "./output/ai_cantilever",
        "iteration": 0,
        "history": [],
    })

    print(f"✅ 完成! 结果: {result.get('result_image_path')}")
    return result


def run_complex_example():
    """复杂问题：需要理论推导 + 代码生成。"""
    app = compile_graph()

    result = app.invoke({
        "user_input": (
            "请设计一个带应力约束的悬臂梁拓扑优化。"
            "设计域60x20，左端固定，右端中点向下力。"
            "目标：最小化柔度。"
            "约束：体积分数≤0.4，von Mises应力不超过许用应力σ_allow=100MPa。"
        ),
        "image_paths": [],
        "max_retries": 2,
        "output_path": "./output/ai_stress_constrained",
        "iteration": 0,
        "history": [],
    })

    print(f"✅ 完成! 结果: {result.get('result_image_path')}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("示例1：简单悬臂梁（标准路径）")
    print("=" * 60)
    run_simple_example()
