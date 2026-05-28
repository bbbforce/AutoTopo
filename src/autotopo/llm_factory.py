"""多 LLM Provider 工厂。

通过统一接口切换 OpenAI / DeepSeek / GLM 等模型，
只需修改 config/settings.yaml 中的 provider 字段即可。
API Key 从项目根目录 .env 文件中自动加载。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）
_ENV_CANDIDATES = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[2] / ".env",
]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p)
        break

import yaml
from langchain_core.language_models import BaseChatModel


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _load_config() -> dict:
    """加载全局配置文件"""
    # 先尝试项目根目录，再回退到包内路径
    for candidate in [Path("config/settings.yaml"), _CONFIG_PATH]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError("找不到 config/settings.yaml")


def get_llm(
    provider: Optional[str] = None,
    *,
    vision: bool = False,
    structured_output: Optional[type] = None,
    **overrides: Any,
) -> BaseChatModel:
    """获取 LLM 实例。

    Parameters
    ----------
    provider : 指定 provider 名称，None 则使用配置文件默认值。
    vision : 如果为 True，使用视觉专用 provider。
    structured_output : 如果提供 Pydantic 类，返回 with_structured_output 绑定后的模型。
    **overrides : 覆盖配置中的任意参数（如 temperature, model 等）。
    """
    config = _load_config()
    llm_cfg = config["llm"]

    if provider is None:
        provider = llm_cfg["vision_provider"] if vision else llm_cfg["default_provider"]

    provider_cfg: dict = {**llm_cfg["providers"].get(provider, {}), **overrides}
    model_name = provider_cfg.pop("model", "gpt-4o")
    base_url = provider_cfg.pop("base_url", None)
    temperature = provider_cfg.pop("temperature", 0.1)

    llm = _create_llm(provider, model_name, base_url, temperature)

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    return llm


def _create_llm(
    provider: str,
    model: str,
    base_url: Optional[str],
    temperature: float,
) -> BaseChatModel:
    """根据 provider 类型实例化对应的 LangChain ChatModel。"""

    if provider in ("openai", "deepseek"):
        # DeepSeek 使用 OpenAI 兼容 API
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        if provider == "deepseek":
            kwargs["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
        return ChatOpenAI(**kwargs)

    if provider == "glm":
        # 智谱 GLM — 也走 OpenAI 兼容接口
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
            api_key=os.environ.get("GLM_API_KEY", ""),
            temperature=temperature,
        )

    if provider == "bailian":
        # 阿里云百炼平台 — 走 OpenAI 兼容接口
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=model, temperature=temperature)

    raise ValueError(f"不支持的 LLM provider: {provider}")
