"""简化版 SQL Agent 的工具集。

这里的工具全部复用工程内已有实现，避免研究代码与线上代码出现两套口径：
- SQL 工具：znws/agent/tools/sql_tools.py
- 向量库（schema 抽样库）工具：znws/agent/tools/schema_retriever.py
"""
from znws.agent.tools.evidence_tools import load_bird_evidence
from znws.agent.tools.schema_retriever import (
    get_table_directory,
    search_schema_by_field,
    search_schema_by_value,
)
from znws.agent.tools.sql_tools import execute_sql, get_table_schema

# Worker 在单一智能体内完成探查、Schema 定位与 SQL 生成验证，因此同时具备
# 数据库探查工具（目录 / 结构 / 抽样库检索）与 SQL 执行验证工具
WORKER_TOOLS = [
    get_table_directory,
    get_table_schema,
    execute_sql,
    search_schema_by_field,
    search_schema_by_value,
]

__all__ = [
    "WORKER_TOOLS",
    "get_table_directory",
    "get_table_schema",
    "execute_sql",
    "search_schema_by_field",
    "search_schema_by_value",
    "load_bird_evidence",
]
