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

# DataLoader 用于“探查数据库 -> 确定表与字段”：SQL 工具 + 向量库工具
DATA_LOADER_TOOLS = [
    get_table_directory,
    get_table_schema,
    execute_sql,
    search_schema_by_field,
    search_schema_by_value,
]

# Worker 用于“生成 -> 执行验证 -> 修正”：以执行工具为主，辅以表结构核对
WORKER_TOOLS = [
    execute_sql,
    get_table_schema,
]

__all__ = [
    "DATA_LOADER_TOOLS",
    "WORKER_TOOLS",
    "get_table_directory",
    "get_table_schema",
    "execute_sql",
    "search_schema_by_field",
    "search_schema_by_value",
    "load_bird_evidence",
]
