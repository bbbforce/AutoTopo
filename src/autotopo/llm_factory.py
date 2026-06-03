"""多 LLM Provider 工厂。

通过统一接口切换 OpenAI / DeepSeek / GLM 等模型，
只需修改 config/settings.yaml 中的 provider 字段即可。
API Key 从项目根目录 .env 文件中自动加载。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

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


def clean_json_string(text: str) -> str:
    """提取并清洗大模型返回的 JSON 字符串。

    能够处理：
    1. 首尾空白字符与多余的提示词符号（例如前导的 '>'）
    2. API 端由于未正常截断返回的 stop tokens（如 </|im_assistant|>, </|im_end|>, <|im_end|>）
    3. Markdown 围栏（如 ```json ... ``` 或者是 ``` ... ```）
    """
    text = text.strip()

    # 移除开头的 '>' 或者其他异常提示引导词
    if text.startswith(">"):
        text = text[1:].strip()

    # 移除未正确截断的 stop tokens
    text = re.sub(r'</\|im_assistant\|>$', '', text).strip()
    text = re.sub(r'<\|im_end\|>$', '', text).strip()
    text = re.sub(r'</\|im_end\|>$', '', text).strip()

    # 提取 Markdown 代码块中的 JSON 字符串
    json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_block_match:
        text = json_block_match.group(1).strip()

    return text


def _create_structured_output_runnable(llm: BaseChatModel, pydantic_model: type[BaseModel]):
    """通过 RunnableLambda 构建带有高度容错的 JSON 清洗和解析的结构化输出组件。"""
    def _invoke_and_parse(input_data: Any, config: Any = None, **kwargs: Any) -> BaseModel:
        res = llm.invoke(input_data, config=config, **kwargs)
        raw_text = res.content
        cleaned_json = clean_json_string(raw_text)
        try:
            return pydantic_model.model_validate_json(cleaned_json)
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON into {pydantic_model.__name__}.\n"
                f"Error: {e}\n"
                f"Original output: {raw_text!r}\n"
                f"Cleaned output: {cleaned_json!r}"
            ) from e

    return RunnableLambda(_invoke_and_parse)


# 仅用于元数据 / 逻辑分支的字段，不应透传给底层 ChatModel 构造函数
_CAPABILITY_KEYS = {"vision", "function_calling", "json_output"}

# provider → 环境变量名映射
_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "GLM_API_KEY",
    "bailian": "DASHSCOPE_API_KEY",
}

# provider → 默认 base_url 映射（None 表示使用 SDK 默认）
_DEFAULT_BASE_URL: dict[str, Optional[str]] = {
    "openai": None,
    "deepseek": "https://api.deepseek.com/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def get_llm(
    provider: Optional[str] = None,
    *,
    vision: bool = False,
    structured_output: Optional[type[BaseModel]] = None,
    **overrides: Any,
) -> Any:
    """获取 LLM 实例。

    Parameters
    ----------
    provider : 指定 provider 名称，None 则使用配置文件默认值。
    vision : 如果为 True，使用视觉专用 provider。
    structured_output : 如果提供 Pydantic 类，返回包含清洗与验证功能的解析模型。
    **overrides : 覆盖配置中的任意参数（如 temperature, model, api_key 等）。
    """
    config = _load_config()
    llm_cfg = config["llm"]

    if provider is None:
        provider = llm_cfg["vision_provider"] if vision else llm_cfg["default_provider"]

    # 合并：配置文件 ← overrides（调用方覆盖优先）
    provider_cfg: dict = {**llm_cfg["providers"].get(provider, {}), **overrides}

    # 提取核心参数
    model_name = provider_cfg.pop("model", "gpt-4o")
    base_url = provider_cfg.pop("base_url", None)
    temperature = provider_cfg.pop("temperature", 0.1)
    api_key = provider_cfg.pop("api_key", None)
    top_p = provider_cfg.pop("top_p", None)
    max_tokens = provider_cfg.pop("max_tokens", None)
    reasoning_effort = provider_cfg.pop("reasoning_effort", None)
    thinking = provider_cfg.pop("thinking", True)

    # 剥离能力声明字段（不传给底层 API）
    for cap_key in _CAPABILITY_KEYS:
        provider_cfg.pop(cap_key, None)

    # 剩余的 provider_cfg 将作为 extra_kwargs 透传
    llm = _create_llm(
        provider,
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        api_key=api_key,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        **provider_cfg,
    )

    if structured_output is not None:
        return _create_structured_output_runnable(llm, structured_output)
    return llm


def _create_llm(
    provider: str,
    *,
    model: str,
    base_url: Optional[str],
    temperature: float,
    api_key: Optional[str] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    thinking: bool = True,
    **extra_kwargs: Any,
) -> BaseChatModel:
    """根据 provider 类型实例化对应的 LangChain ChatModel。

    Parameters
    ----------
    api_key : 显式传入的 API 密钥，优先于环境变量。
    top_p : 核采样参数，None 时不传（使用 API 默认值）。
    max_tokens : 最大输出 token 数，None 时不传。
    reasoning_effort : 思考强度 'low'|'medium'|'high'|'max'，None 时不传。
    thinking : 是否启用深度思考模式，通过 extra_body 传递。
    """

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if top_p is not None:
            kwargs["top_p"] = top_p
        kwargs.update(extra_kwargs)
        return ChatOllama(**kwargs)

    # ── 所有 OpenAI 兼容 provider 统一走此路径 ──
    from langchain_openai import ChatOpenAI

    # 解析 api_key：配置文件 > 环境变量
    resolved_key = api_key or os.environ.get(_ENV_KEY_MAP.get(provider, ""), "")

    # 解析 base_url：配置文件 > 默认值
    resolved_url = base_url or _DEFAULT_BASE_URL.get(provider)

    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": resolved_key,
    }
    if resolved_url:
        kwargs["base_url"] = resolved_url
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    model_kw: dict[str, Any] = extra_kwargs.pop("model_kwargs", {})
    if thinking:
        model_kw["thinking"] = {"type": "enabled"}
    else:
        model_kw["thinking"] = {"type": "disabled"}
    if model_kw:
        kwargs["model_kwargs"] = model_kw

    kwargs.update(extra_kwargs)
    return ChatOpenAI(**kwargs)
