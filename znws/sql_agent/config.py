"""简化版 SQL Agent 的模型与全局配置（BIRD 数据集）。

数据库相关的路径解析与 schema 元数据读取已统一收敛到 znws/utils/bird.py，
本模块只保留模型工厂、全局配置说明与运行上限。
"""
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

# 工程根目录（znws/sql_agent/config.py -> 上溯两级）
BASE_DIR = Path(__file__).resolve().parents[2]
# 数据库类型与时间格式的全局配置说明（随 BIRD 数据集更新）
DB_CONFIG_PATH = BASE_DIR / "znws" / "agent" / "prompts" / "db_config.md"

# agent 单轮运行的最大步数，防止模型陷入无限循环
DEFAULT_RECURSION_LIMIT = 30


def load_db_config() -> str:
    """读取数据库全局配置说明。"""
    if DB_CONFIG_PATH.exists():
        return DB_CONFIG_PATH.read_text(encoding="utf-8").strip()
    return ""


def _deepseek_extra_body(base_url: str | None) -> dict | None:
    """DeepSeek 官方 API 的思考模式配置。

    DeepSeek V4 默认开启思考模式，而思考模式不支持强制 tool_choice；
    本工程的 ReAct 工具调用与结构化输出都会强制 tool_choice，因此默认关闭。
    显式设置 DEEPSEEK_THINKING=enabled/disabled 时对代理地址同样生效。
    """
    mode = os.getenv("DEEPSEEK_THINKING", "").strip().lower()
    if not mode:
        if not base_url or "api.deepseek.com" not in base_url.lower():
            return None
        mode = "disabled"
    thinking = "enabled" if mode in {"enabled", "on", "true", "1"} else "disabled"
    return {"thinking": {"type": thinking}}


def build_model(model_name: str | None = None, base_url: str | None = None,
                temperature: float = 0.0) -> ChatOpenAI:
    """按环境变量构造对话模型；显式入参优先。

    网络鲁棒性参数（可用环境变量覆盖）：
    - OPENAI_API_TIMEOUT：单次请求超时秒数，默认 90；
    - OPENAI_API_MAX_RETRIES：SDK 级瞬时错误重试次数，默认 2。

    DeepSeek 官方 API 默认关闭思考模式（DEEPSEEK_THINKING=disabled），
    否则其思考模式会拒绝强制 tool_choice，导致结构化输出与工具调用失败。
    """
    timeout = float(os.getenv("OPENAI_API_TIMEOUT", "90"))
    max_retries = int(os.getenv("OPENAI_API_MAX_RETRIES", "2"))
    resolved_base = base_url or os.getenv("OPENAI_API_BASE") or None
    return ChatOpenAI(
        model=model_name or os.getenv("OPENAI_API_MODEL", "qwen-plus"),
        base_url=resolved_base,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        extra_body=_deepseek_extra_body(resolved_base),
    )
