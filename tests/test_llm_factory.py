"""测试大语言模型工厂。"""

import os
from unittest.mock import patch, MagicMock
import pytest
from autotopo.llm_factory import get_llm


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

    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-dashscope-key"}):
        get_llm(provider="bailian")

        mock_chat_openai.assert_called_once_with(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="test-dashscope-key",
            temperature=0.1
        )


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

    get_llm(provider="openai")
    mock_chat_openai.assert_called_once_with(
        model="gpt-4o",
        temperature=0.2
    )


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

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-deepseek-key"}):
        get_llm(provider="deepseek")
        mock_chat_openai.assert_called_once_with(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="test-deepseek-key",
            temperature=0.15
        )


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

    with patch.dict(os.environ, {"GLM_API_KEY": "test-glm-key"}):
        get_llm(provider="glm")
        mock_chat_openai.assert_called_once_with(
            model="glm-4v",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="test-glm-key",
            temperature=0.1
        )
