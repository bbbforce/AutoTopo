"""路由节点测试。"""

from autotopo.nodes.router import route_decision, route_problem


def _make_state(constraints, objective="minimize_compliance"):
    return {
        "problem_definition": {
            "constraints": constraints,
            "objective": objective,
        },
    }


class TestRouter:

    def test_standard_path_compliance(self):
        """最小柔度 + 体积分数约束 → 标准路径。"""
        state = _make_state(
            constraints=[{"type": "volume_fraction", "value": 0.5}],
            objective="minimize_compliance",
        )
        result = route_problem(state)
        assert result["route"] == "standard_path"
        assert result["unknown_constraints"] == []

    def test_complex_path_stress(self):
        """应力约束 → 复杂路径。"""
        state = _make_state(
            constraints=[
                {"type": "volume_fraction", "value": 0.5},
                {"type": "stress", "value": 100.0},
            ],
        )
        result = route_problem(state)
        assert result["route"] == "complex_path"
        assert "stress" in result["unknown_constraints"]

    def test_complex_path_custom(self):
        """自定义约束 → 复杂路径。"""
        state = _make_state(
            constraints=[{"type": "custom", "value": 0}],
        )
        result = route_problem(state)
        assert result["route"] == "complex_path"

    def test_complex_path_unknown_objective(self):
        """未知目标函数 → 复杂路径。"""
        state = _make_state(
            constraints=[{"type": "volume_fraction", "value": 0.5}],
            objective="minimize_max_stress",
        )
        result = route_problem(state)
        assert result["route"] == "complex_path"

    def test_route_decision_function(self):
        """route_decision 应从 state 中读取 route 字段。"""
        assert route_decision({"route": "standard_path"}) == "standard_path"
        assert route_decision({"route": "complex_path"}) == "complex_path"

    def test_empty_constraints(self):
        """无约束 → 标准路径。"""
        state = _make_state(constraints=[])
        result = route_problem(state)
        assert result["route"] == "standard_path"
