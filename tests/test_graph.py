"""LangGraph 图结构 + 端到端 Mock 测试。"""

from unittest.mock import MagicMock, patch

import numpy as np

from autotopo.graph import build_graph, compile_graph


class TestGraphStructure:
    """图结构验证（不需要 LLM）。"""

    def test_compile_success(self):
        app = compile_graph()
        assert app is not None

    def test_node_count(self):
        graph = build_graph()
        compiled = graph.compile()
        nodes = list(compiled.get_graph().nodes.keys())
        # __start__, __end__ + 8 个业务节点
        assert len(nodes) == 10

    def test_expected_nodes(self):
        graph = build_graph()
        compiled = graph.compile()
        nodes = set(compiled.get_graph().nodes.keys())
        expected = {
            "__start__", "__end__",
            "parse_input", "route_problem",
            "theory_derivation", "code_generation",
            "run_simulation", "evaluate_result",
            "apply_fixes", "save_output",
        }
        assert nodes == expected


class TestEndToEndMock:
    """端到端 Mock 测试：用 Mock 替换 LLM 调用，验证完整流程。"""

    def _mock_problem_dict(self):
        return {
            "description": "悬臂梁",
            "domain": {"width": 60, "height": 20, "nelx": 30, "nely": 10,
                        "non_design_regions": []},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [
                {"type": "fixed", "location": "left_edge", "node_indices": None},
            ],
            "loads": [
                {"type": "point_force", "location": "right_center",
                 "magnitude": 1.0, "direction": [0, -1], "node_indices": None},
            ],
            "objective": "minimize_compliance",
            "constraints": [
                {"type": "volume_fraction", "value": 0.5, "description": None},
            ],
            "parameters": {"penal": 3.0, "rmin": 1.5, "max_iter": 20, "tol": 0.01},
        }

    def _mock_eval_pass(self):
        return {
            "has_defects": False,
            "defect_types": [],
            "severity": "minor",
            "suggested_fixes": [],
            "reasoning": "结果清晰。",
        }

    @patch("autotopo.nodes.input_parser.get_llm")
    @patch("autotopo.nodes.evaluator.get_llm")
    def test_standard_path_e2e(self, mock_eval_llm, mock_parser_llm, tmp_path):
        """标准路径端到端：解析 → 路由(标准) → 仿真 → 评估(通过) → 保存。"""
        import yaml

        problem_dict = self._mock_problem_dict()

        # Mock 解析 LLM
        mock_problem_obj = MagicMock()
        mock_problem_obj.model_dump.return_value = problem_dict
        mock_parser_model = MagicMock()
        mock_parser_model.invoke.return_value = mock_problem_obj
        mock_parser_llm.return_value = mock_parser_model

        # Mock 评估 LLM
        mock_eval_obj = MagicMock()
        mock_eval_obj.model_dump.return_value = self._mock_eval_pass()
        mock_eval_model = MagicMock()
        mock_eval_model.invoke.return_value = mock_eval_obj
        mock_eval_llm.return_value = mock_eval_model

        app = compile_graph()
        result = app.invoke({
            "user_input": "悬臂梁测试",
            "image_paths": [],
            "max_retries": 3,
            "output_path": str(tmp_path),
            "iteration": 0,
            "history": [],
        })

        # 验证流程完成
        assert result.get("route") == "standard_path"
        assert result.get("result_image_path") is not None
        assert result["evaluation"]["has_defects"] is False

    @patch("autotopo.nodes.input_parser.get_llm")
    @patch("autotopo.nodes.evaluator.get_llm")
    def test_retry_loop(self, mock_eval_llm, mock_parser_llm, tmp_path):
        """测试反馈闭环：第一次评估有缺陷 → 修正 → 第二次通过。"""
        problem_dict = self._mock_problem_dict()

        # Mock 解析
        mock_problem_obj = MagicMock()
        mock_problem_obj.model_dump.return_value = problem_dict
        mock_parser_model = MagicMock()
        mock_parser_model.invoke.return_value = mock_problem_obj
        mock_parser_llm.return_value = mock_parser_model

        # Mock 评估：第一次有缺陷，第二次通过
        eval_fail = {
            "has_defects": True,
            "defect_types": ["gray_elements"],
            "severity": "moderate",
            "suggested_fixes": [
                {"parameter": "penal", "current_value": 3.0,
                 "suggested_value": 5.0, "reason": "增大罚因子"},
            ],
            "reasoning": "存在灰度单元。",
        }

        call_count = [0]

        def eval_side_effect(*args, **kwargs):
            mock_obj = MagicMock()
            call_count[0] += 1
            if call_count[0] <= 1:
                mock_obj.model_dump.return_value = eval_fail
            else:
                mock_obj.model_dump.return_value = self._mock_eval_pass()
            mock_model = MagicMock()
            mock_model.invoke.return_value = mock_obj
            return mock_model

        mock_eval_llm.side_effect = eval_side_effect

        app = compile_graph()
        result = app.invoke({
            "user_input": "悬臂梁测试",
            "image_paths": [],
            "max_retries": 3,
            "output_path": str(tmp_path),
            "iteration": 0,
            "history": [],
        })

        # 第二次评估应通过
        assert result["evaluation"]["has_defects"] is False
        # 应经过至少2次迭代
        assert result["iteration"] >= 2
