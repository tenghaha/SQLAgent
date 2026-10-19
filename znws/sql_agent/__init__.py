"""简化版 SQL Agent：去除了中断/人工介入、可视化与记忆模块的 Text-to-SQL 实验框架。

一次运行 = 一个独立 query，流程为：
    Router（依据数据库简述选定唯一库） -> DataLoader（探查表与字段）
    -> Worker（生成并验证 SQL） -> 复算结果
全过程记录到 logs/sql_agent/ 下的单个日志文件。
"""
from .db_router import select_database
from .dataloader import SchemaPlan
from .pipeline import SQLAgentResult, execute_sql_preview, run_sql_agent
from .tracing import RunRecorder
from .worker import SqlAnswer

__all__ = [
    "run_sql_agent",
    "execute_sql_preview",
    "select_database",
    "SQLAgentResult",
    "SchemaPlan",
    "SqlAnswer",
    "RunRecorder",
]
