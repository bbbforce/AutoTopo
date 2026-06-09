"""测试大语言模型工厂。"""

import os
from unittest.mock import patch
from autotopo.llm_factory import get_llm


def _assert_chat_kwargs(mock_chat_openai, **expected):
    kwargs = mock_chat_openai.call_args.kwargs
    for key, value in expected.items():
        assert kwargs.get(key) == value


@patch("autotopo.llm_factory._load_config")
@patch("langchain_openai.ChatOpenAI")
def test_get_llm_bailian(mock_chat_openai, mock_load_config):
    """测试实例化阿里云百炼平台提供商。"""
    mock_load_config.return_value = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "bailian": {
                    "model": "qwen-plus",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "temperature": 0.1
                }
            }
        }
    }

    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-dashscope-key"}, clear=True):
        get_llm(provider="bailian")

        mock_chat_openai.assert_called_once()
        _assert_chat_kwargs(
            mock_chat_openai,
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="test-dashscope-key",
            temperature=0.1,
        )
        assert mock_chat_openai.call_args.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


@patch("autotopo.llm_factory._load_config")
@patch("langchain_openai.ChatOpenAI")
def test_get_llm_openai(mock_chat_openai, mock_load_config):
    """测试实例化 OpenAI 提供商。"""
    mock_load_config.return_value = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "model": "gpt-4o",
                    "temperature": 0.2
                }
            }
        }
    }

    with patch.dict(os.environ, {}, clear=True):
        get_llm(provider="openai")

    mock_chat_openai.assert_called_once()
    _assert_chat_kwargs(mock_chat_openai, model="gpt-4o", temperature=0.2)
    assert mock_chat_openai.call_args.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


@patch("autotopo.llm_factory._load_config")
@patch("langchain_openai.ChatOpenAI")
def test_get_llm_deepseek(mock_chat_openai, mock_load_config):
    """测试实例化 DeepSeek 提供商。"""
    mock_load_config.return_value = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "deepseek": {
                    "model": "deepseek-chat",
                    "base_url": "https://api.deepseek.com/v1",
                    "temperature": 0.15
                }
            }
        }
    }

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-deepseek-key"}, clear=True):
        get_llm(provider="deepseek")
        mock_chat_openai.assert_called_once()
        _assert_chat_kwargs(
            mock_chat_openai,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="test-deepseek-key",
            temperature=0.15,
        )
        assert mock_chat_openai.call_args.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


@patch("autotopo.llm_factory._load_config")
@patch("langchain_openai.ChatOpenAI")
def test_get_llm_glm(mock_chat_openai, mock_load_config):
    """测试实例化智谱 GLM 提供商。"""
    mock_load_config.return_value = {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "glm": {
                    "model": "glm-4v",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "temperature": 0.1
                }
            }
        }
    }

    with patch.dict(os.environ, {"GLM_API_KEY": "test-glm-key"}, clear=True):
        get_llm(provider="glm")
        mock_chat_openai.assert_called_once()
        _assert_chat_kwargs(
            mock_chat_openai,
            model="glm-4v",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="test-glm-key",
            temperature=0.1,
        )
        assert mock_chat_openai.call_args.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
