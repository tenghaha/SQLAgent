"""简化版 SQL Agent：去除了中断/人工介入、可视化与记忆模块的 Text-to-SQL 实验框架。

一次运行 = 一个独立 query，流程为：
    QueryProcessor（NLU 前处理） -> Worker（探查表与字段、生成并验证 SQL） -> 复算结果
全过程记录到 logs/sql_agent/ 下的单个日志文件。
"""
from .pipeline import SQLAgentResult, execute_sql_preview, run_sql_agent
from .query_processor import QueryEntity, QueryProcessResult, run_query_processor
from .tracing import RunRecorder
from .worker import SqlAnswer

__all__ = [
    "run_sql_agent",
    "execute_sql_preview",
    "run_query_processor",
    "SQLAgentResult",
    "QueryProcessResult",
    "QueryEntity",
    "SqlAnswer",
    "RunRecorder",
]
