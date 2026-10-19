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


def build_model(model_name: str | None = None, base_url: str | None = None,
                temperature: float = 0.0) -> ChatOpenAI:
    """按环境变量构造对话模型；显式入参优先。"""
    return ChatOpenAI(
        model=model_name or os.getenv("OPENAI_API_MODEL", "qwen-plus"),
        base_url=base_url or os.getenv("OPENAI_API_BASE") or None,
        temperature=temperature,
    )
