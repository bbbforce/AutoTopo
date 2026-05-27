"""文件 I/O 工具。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def save_yaml(data: dict, path: str) -> None:
    """保存字典为 YAML 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def load_yaml(path: str) -> dict:
    """从 YAML 文件加载字典。"""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict, path: str) -> None:
    """保存字典为 JSON 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: str) -> dict:
    """从 JSON 文件加载字典。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
