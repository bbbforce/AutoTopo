"""LangGraph 图结构 + 端到端 Mock 测试。"""

from unittest.mock import MagicMock, patch

import numpy as np

from autotopo.engines.base import OptResult
from autotopo.graph import build_graph, compile_graph
from autotopo.monitoring import WorkflowTracer, read_jsonl


class FakeEngine:
    def setup(self, problem):
        self.problem = problem

    def optimize(self, *, max_iter=200, tol=1e-6, penal=None, rmin=None, volfrac=None):
        return OptResult(
            densities=np.full((2, 3), volfrac if volfrac is not None else 0.5),
            compliance_history=[10.0, 5.0],
            volume_history=[volfrac if volfrac is not None else 0.5] * 2,
            iterations=2,
            converged=True,
            mesh_info={"num_cells": 6},
        )

    def export_image(self, path, dpi=300):
        from pathlib import Path
        Path(path).write_bytes(b"fake density")
        return path

    def get_convergence_image(self, path):
        from pathlib import Path
        Path(path).write_bytes(b"fake convergence")
        return path


class TestGraphStructure:
    """图结构验证（不需要 LLM）。"""

    def test_compile_success(self):
        app = compile_graph()
        assert app is not None

    def test_node_count(self):
        graph = build_graph()
        compiled = graph.compile()
        nodes = list(compiled.get_graph().nodes.keys())
        # __start__, __end__ + 9 个业务节点
        assert len(nodes) == 11

    def test_expected_nodes(self):
        graph = build_graph()
        compiled = graph.compile()
        nodes = set(compiled.get_graph().nodes.keys())
        expected = {
            "__start__", "__end__",
            "parse_input", "route_problem",
            "theory_derivation", "code_generation",
            "run_simulation", "evaluate_result",
            "apply_fixes", "prepare_final_refine", "save_output",
        }
        assert nodes == expected


class TestEndToEndMock:
    """端到端 Mock 测试：用 Mock 替换 LLM 调用，验证完整流程。"""

    def _mock_problem_dict(self):
        return {
            "description": "悬臂梁",
            "domain": {"width": 60, "height": 20, "mesh_resolution": 1.0,
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
            "parameters": {
                "penal": 3.0,
                "rmin": 0.05,
                "max_iter": 20,
                "tol": 1e-6,
                "optimizer": "SLSQP",
            },
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
    @patch("autotopo.nodes.simulator._get_engine", return_value=FakeEngine())
    def test_standard_path_e2e(self, mock_get_engine, mock_eval_llm, mock_parser_llm, tmp_path):
        """标准路径端到端：解析 → 路由(标准) → 仿真 → 评估(通过) → 保存。"""
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
            "solve_profile": "preview_refine",
            "solve_stage": "preview",
            "final_refine_done": False,
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
    @patch("autotopo.nodes.simulator._get_engine", return_value=FakeEngine())
    def test_retry_loop(self, mock_get_engine, mock_eval_llm, mock_parser_llm, tmp_path):
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
            "solve_profile": "preview_refine",
            "solve_stage": "preview",
            "final_refine_done": False,
            "output_path": str(tmp_path),
            "iteration": 0,
            "history": [],
        })

        # 最终精修评估应通过
        assert result["evaluation"]["has_defects"] is False
        assert result["solve_stage"] == "final"
        # 应经过预览失败、预览通过、最终精修通过
        assert result["iteration"] >= 3

    @patch("autotopo.nodes.input_parser.get_llm")
    @patch("autotopo.nodes.evaluator.get_llm")
    @patch("autotopo.nodes.simulator._get_engine", return_value=FakeEngine())
    def test_instrumented_graph_writes_timeline(self, mock_get_engine, mock_eval_llm, mock_parser_llm, tmp_path):
        """带 tracer 的主图应写出阶段事件，且不改变 workflow 结果。"""
        problem_dict = self._mock_problem_dict()

        mock_problem_obj = MagicMock()
        mock_problem_obj.model_dump.return_value = problem_dict
        mock_parser_model = MagicMock()
        mock_parser_model.invoke.return_value = mock_problem_obj
        mock_parser_llm.return_value = mock_parser_model

        mock_eval_obj = MagicMock()
        mock_eval_obj.model_dump.return_value = self._mock_eval_pass()
        mock_eval_model = MagicMock()
        mock_eval_model.invoke.return_value = mock_eval_obj
        mock_eval_llm.return_value = mock_eval_model

        tracer = WorkflowTracer(run_id="graph-test", workflow_type="main", output_dir=tmp_path)
        app = compile_graph(tracer=tracer)
        result = app.invoke({
            "user_input": "悬臂梁测试",
            "image_paths": [],
            "max_retries": 1,
            "solve_profile": "preview_only",
            "solve_stage": "preview",
            "final_refine_done": True,
            "output_path": str(tmp_path),
            "iteration": 0,
            "history": [],
        })

        events = read_jsonl(tmp_path / "workflow_events.jsonl")
        completed_stages = [event["stage"] for event in events if event["status"] == "completed"]
        assert result["route"] == "standard_path"
        assert "parse_input" in completed_stages
        assert "route_problem" in completed_stages
        assert "run_simulation" in completed_stages
        assert "evaluate_result" in completed_stages
        assert "save_output" in completed_stages
