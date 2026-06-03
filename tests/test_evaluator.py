"""评估节点逻辑测试（不涉及 LLM 调用）。"""

from autotopo.nodes.evaluator import apply_fixes, should_retry


class TestShouldRetry:

    def test_no_defects_accept(self):
        state = {"evaluation": {"has_defects": False}, "iteration": 0, "max_retries": 3}
        assert should_retry(state) == "accept"

    def test_has_defects_retry(self):
        state = {"evaluation": {"has_defects": True}, "iteration": 1, "max_retries": 3}
        assert should_retry(state) == "retry"

    def test_max_retries_exceeded(self):
        state = {"evaluation": {"has_defects": True}, "iteration": 3, "max_retries": 3}
        assert should_retry(state) == "accept"

    def test_custom_max_retries(self):
        state = {"evaluation": {"has_defects": True}, "iteration": 4, "max_retries": 5}
        assert should_retry(state) == "retry"

        state["iteration"] = 5
        assert should_retry(state) == "accept"

    def test_default_max_retries(self):
        """未设置 max_retries 时默认 3。"""
        state = {"evaluation": {"has_defects": True}, "iteration": 2}
        assert should_retry(state) == "retry"

        state["iteration"] = 3
        assert should_retry(state) == "accept"


class TestApplyFixes:

    def test_apply_penal_increase(self):
        """penal 建议值在步进范围内（+50%=4.5），直接采纳。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "penal", "current_value": 3.0,
                     "suggested_value": 4.0, "reason": "test"},
                ],
            },
            "current_params": {"penal": 3.0, "rmin": 1.5},
        }
        result = apply_fixes(state)
        assert result["current_params"]["penal"] == 4.0

    def test_penal_step_clamp(self):
        """penal 建议跳跃过大（3→8），应被限幅为 3+3*0.5=4.5。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "penal", "current_value": 3.0,
                     "suggested_value": 8.0, "reason": "test"},
                ],
            },
            "current_params": {"penal": 3.0},
        }
        result = apply_fixes(state)
        assert result["current_params"]["penal"] == 4.5

    def test_penal_no_decrease(self):
        """penal 只增不减：LLM 建议降低时维持当前值。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "penal", "current_value": 5.0,
                     "suggested_value": 3.0, "reason": "test"},
                ],
            },
            "current_params": {"penal": 5.0},
        }
        result = apply_fixes(state)
        assert result["current_params"]["penal"] == 5.0

    def test_apply_rmin_increase(self):
        """rmin 建议值在步进范围内（+50%=2.25），直接采纳。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "rmin", "current_value": 1.5,
                     "suggested_value": 2.0, "reason": "test"},
                ],
            },
            "current_params": {"penal": 3.0, "rmin": 1.5},
        }
        result = apply_fixes(state)
        assert result["current_params"]["rmin"] == 2.0

    def test_rmin_step_clamp(self):
        """rmin 跳跃过大（1.5→4.0），应被限幅为 1.5+0.75=2.25。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "rmin", "current_value": 1.5,
                     "suggested_value": 4.0, "reason": "test"},
                ],
            },
            "current_params": {"rmin": 1.5},
        }
        result = apply_fixes(state)
        assert result["current_params"]["rmin"] == 2.25

    def test_safety_bounds_penal(self):
        """penal 上限 10.0：当前8.0 + 50% = 12，应限制为 10.0。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "penal", "current_value": 8.0,
                     "suggested_value": 15.0, "reason": "test"},
                ],
            },
            "current_params": {"penal": 8.0},
        }
        result = apply_fixes(state)
        assert result["current_params"]["penal"] == 10.0

    def test_safety_bounds_volfrac(self):
        """体积分数应被限制在 [0.1, 0.9]。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "volfrac", "current_value": 0.5,
                     "suggested_value": 0.01, "reason": "test"},
                ],
            },
            "current_params": {"volfrac": 0.5},
        }
        result = apply_fixes(state)
        assert result["current_params"]["volfrac"] == 0.1

    def test_multiple_fixes(self):
        """同时调整 penal 和 rmin，均在步进范围内。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "penal", "current_value": 3.0,
                     "suggested_value": 4.0, "reason": ""},
                    {"parameter": "rmin", "current_value": 1.5,
                     "suggested_value": 2.0, "reason": ""},
                ],
            },
            "current_params": {"penal": 3.0, "rmin": 1.5},
        }
        result = apply_fixes(state)
        assert result["current_params"]["penal"] == 4.0
        assert result["current_params"]["rmin"] == 2.0

    def test_no_fixes(self):
        state = {
            "evaluation": {"suggested_fixes": []},
            "current_params": {"penal": 3.0},
        }
        result = apply_fixes(state)
        assert result["current_params"]["penal"] == 3.0

    def test_apply_ft_fix(self):
        """ft 建议值（浮点数）应被转为整型采纳。"""
        state = {
            "evaluation": {
                "suggested_fixes": [
                    {"parameter": "ft", "current_value": 1.0,
                     "suggested_value": 2.0, "reason": "test"},
                ],
            },
            "current_params": {"ft": 1},
        }
        result = apply_fixes(state)
        assert result["current_params"]["ft"] == 2
        assert isinstance(result["current_params"]["ft"], int)
