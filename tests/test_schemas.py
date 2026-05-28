"""Schema 数据模型测试。"""

import json

import yaml

from autotopo.schemas import (
    BCType,
    BoundaryCondition,
    ConstraintSpec,
    ConstraintType,
    DefectType,
    DomainSpec,
    EvaluationResult,
    LoadSpec,
    LoadType,
    MaterialSpec,
    ObjectiveType,
    OptimizationProblem,
    OptParams,
    ParameterAdjustment,
    Severity,
)


class TestOptimizationProblem:
    """OptimizationProblem Schema 测试。"""

    def _make_problem(self, **overrides) -> OptimizationProblem:
        defaults = {
            "description": "悬臂梁拓扑优化",
            "domain": DomainSpec(width=60, height=20, nelx=60, nely=20),
            "material": MaterialSpec(youngs_modulus=1.0, poissons_ratio=0.3),
            "boundary_conditions": [
                BoundaryCondition(type=BCType.FIXED, location="left_edge"),
            ],
            "loads": [
                LoadSpec(type=LoadType.POINT_FORCE, location="right_center",
                         magnitude=1.0, direction=[0, -1]),
            ],
            "objective": ObjectiveType.MINIMIZE_COMPLIANCE,
            "constraints": [
                ConstraintSpec(type=ConstraintType.VOLUME_FRACTION, value=0.5),
            ],
            "parameters": OptParams(penal=3.0, rmin=1.5),
        }
        defaults.update(overrides)
        return OptimizationProblem(**defaults)

    def test_basic_creation(self):
        problem = self._make_problem()
        assert problem.description == "悬臂梁拓扑优化"
        assert problem.domain.nelx == 60
        assert problem.material.youngs_modulus == 1.0
        assert len(problem.boundary_conditions) == 1
        assert len(problem.loads) == 1

    def test_serialize_to_dict(self):
        problem = self._make_problem()
        d = problem.model_dump()
        assert isinstance(d, dict)
        assert d["objective"] == "minimize_compliance"
        assert d["constraints"][0]["type"] == "volume_fraction"

    def test_serialize_to_json(self):
        problem = self._make_problem()
        j = problem.model_dump_json()
        parsed = json.loads(j)
        assert parsed["domain"]["nelx"] == 60

    def test_serialize_to_yaml(self):
        problem = self._make_problem()
        y = yaml.dump(problem.model_dump(mode="json"), allow_unicode=True)
        parsed = yaml.safe_load(y)
        assert parsed["material"]["poissons_ratio"] == 0.3

    def test_defaults(self):
        """未指定的参数应有合理默认值。"""
        problem = self._make_problem()
        assert problem.parameters.penal == 3.0
        assert problem.parameters.max_iter == 200
        assert problem.parameters.tol == 0.01

    def test_multiple_constraints(self):
        problem = self._make_problem(constraints=[
            ConstraintSpec(type=ConstraintType.VOLUME_FRACTION, value=0.5),
            ConstraintSpec(type=ConstraintType.STRESS, value=100.0,
                           description="von Mises 应力约束"),
        ])
        assert len(problem.constraints) == 2
        types = {c.type for c in problem.constraints}
        assert ConstraintType.STRESS in types


class TestEvaluationResult:
    """EvaluationResult Schema 测试。"""

    def test_no_defects(self):
        result = EvaluationResult(
            has_defects=False,
            defect_types=[],
            severity=Severity.MINOR,
            suggested_fixes=[],
            reasoning="结果清晰，黑白分明，无明显缺陷。",
        )
        assert not result.has_defects
        assert result.suggested_fixes == []

    def test_with_defects(self):
        result = EvaluationResult(
            has_defects=True,
            defect_types=[DefectType.GRAY_ELEMENTS, DefectType.CHECKERBOARD],
            severity=Severity.MODERATE,
            suggested_fixes=[
                ParameterAdjustment(
                    parameter="penal", current_value=3.0,
                    suggested_value=4.0, reason="增大罚因子减少灰度单元",
                ),
                ParameterAdjustment(
                    parameter="rmin", current_value=1.5,
                    suggested_value=2.5, reason="增大过滤半径消除棋盘格",
                ),
            ],
            reasoning="存在明显灰度区域和棋盘格图案。",
        )
        assert result.has_defects
        assert len(result.suggested_fixes) == 2
        assert result.suggested_fixes[0].parameter == "penal"

    def test_serialize_roundtrip(self):
        result = EvaluationResult(
            has_defects=True,
            defect_types=[DefectType.ISLAND],
            severity=Severity.SEVERE,
            suggested_fixes=[],
            reasoning="存在孤岛。",
        )
        d = result.model_dump()
        restored = EvaluationResult(**d)
        assert restored.defect_types == [DefectType.ISLAND]
