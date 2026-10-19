"""SQL 执行与表结构读取工具（适配 BIRD-like 数据集）。

执行目标不再写死为单个库，而是由运行期设置的“活动数据库”决定，
即配置中登记的 <databases>/<db_id>/<db_id>.sqlite。
"""
import sqlite3

import pandas as pd
from langchain_core.tools import tool

from znws.utils.dataset import (
    active_database_path,
    get_active_database,
    table_column_descriptions,
)


@tool
def execute_sql(sql_query: str) -> str:
    """
    Execute a SQL query against the current BIRD SQLite database and return the results.
    Use this to test your SQL queries and verify the structure or data.
    If the query fails, it returns the error message.
    """
    try:
        conn = sqlite3.connect(str(active_database_path()))
        # We use pandas to easily format the output as a string (e.g., Markdown or CSV-like)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        # If the result is too large, truncate it to avoid overflowing context window
        if len(df) > 50:
            return f"Query succeeded. First 50 rows of {len(df)} total:\n{df.head(50).to_string()}"
        return f"Query succeeded. Result:\n{df.to_string()}"
    except Exception as e:
        return f"Query failed with error: {e}"


@tool
def get_table_schema(table_name: str) -> str:
    """
    Get the schema (CREATE TABLE statement) for a specific table in the current BIRD
    SQLite database, together with BIRD's per-column descriptions
    (database_description/*.csv). Use this to understand the exact columns and their
    business meaning before writing SQL.
    """
    try:
        conn = sqlite3.connect(str(active_database_path()))
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        res = cursor.fetchone()
        conn.close()
        if not res:
            return f"Table '{table_name}' not found in the database."
        return _with_bird_descriptions(res[0], table_name)
    except Exception as e:
        return f"Error retrieving schema: {e}"


def _with_bird_descriptions(ddl: str, table_name: str) -> str:
    """在建表语句后追加 BIRD 提供的列文字说明。"""
    db_id, split = get_active_database()
    if not db_id:
        return ddl
    descriptions = table_column_descriptions(db_id, table_name, split)
    notes = [f"- {field}: {text}" for field, text in descriptions.items() if text]
    if not notes:
        return ddl
    return ddl + "\n\n-- BIRD 列说明:\n" + "\n".join(notes)
