"""基于 SQLite authorizer 的表 / 字段提取与 P/R 计算。

SQLite 在 prepare 阶段会对 SQL 实际读取的每一列触发 ``SQLITE_READ`` 回调。
这里用 ``EXPLAIN`` 只做 prepare、不执行查询，因此：
- 提取的是数据库引擎真实解析出的表与列，覆盖别名、子查询、CTE、``SELECT *``；
- 成本极低，不需要扫描数据，也不受查询执行时间影响。

字段统一表示为 ``"表名.列名"``（小写），表名同样小写。
"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


def extract_sql_schema(sql: str, db_path) -> Dict[str, Any]:
    """提取 SQL 读取的表与字段。

    返回 ``{"status": "ok"/"empty"/"error", "tables": [...], "fields": [...], "error": ...}``。
    ``status=error`` 时表/字段为空，调用方可据此决定是否计入指标。
    """
    sql = (sql or "").strip()
    if not sql:
        return {"status": "empty", "tables": [], "fields": [], "error": None}

    db_file = Path(db_path)
    if not db_file.exists():
        return {"status": "error", "tables": [], "fields": [], "error": "数据库文件不存在"}

    tables = set()
    fields = set()

    def _authorizer(action, arg1, arg2, dbname, trigger):
        if action == sqlite3.SQLITE_READ and arg1:
            table = str(arg1).lower()
            if table.startswith("sqlite_"):
                return sqlite3.SQLITE_OK
            tables.add(table)
            column = (arg2 or "").strip().lower()
            if column:
                fields.add(f"{table}.{column}")
        return sqlite3.SQLITE_OK

    conn = sqlite3.connect(str(db_file))
    try:
        conn.set_authorizer(_authorizer)
        cursor = conn.cursor()
        cursor.execute("EXPLAIN " + sql)
    except Exception as exc:
        return {
            "status": "error",
            "tables": [],
            "fields": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        conn.close()

    return {
        "status": "ok",
        "tables": sorted(tables),
        "fields": sorted(fields),
        "error": None,
    }


def schema_prf(pred_items: Iterable[str], gt_items: Iterable[str]) -> Dict[str, Any]:
    """计算单个题目的表/字段 Precision、Recall 与 TP/FP/FN。"""
    pred = set(pred_items or [])
    gt = set(gt_items or [])
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not gt else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": len(pred),
        "gt_count": len(gt),
    }


def aggregate_schema_prf(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把多题的 TP/FP/FN 汇总为 micro Precision/Recall；无题目时返回 None。"""
    if not items:
        return {
            "precision": None,
            "recall": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "count": 0,
        }
    tp = sum(item["tp"] for item in items)
    fp = sum(item["fp"] for item in items)
    fn = sum(item["fn"] for item in items)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "count": len(items),
    }


def average_metric(items: List[Dict[str, Any]], key: str) -> float | None:
    """计算多题某个指标的宏平均。"""
    values = [item[key] for item in items if item.get(key) is not None]
    return sum(values) / len(values) if values else None


__all__ = [
    "extract_sql_schema",
    "schema_prf",
    "aggregate_schema_prf",
    "average_metric",
]
