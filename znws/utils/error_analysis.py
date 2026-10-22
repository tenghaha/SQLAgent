"""错例分析：对已有评测报告中的 EX=0 题目做结构化对比与归因。

分析流程（可重复执行）：
1. 读取评测明细（eval_*.json）与预测文件（preds_*.json）；
2. 复用评测的 EX 标记，筛出全部错例；
3. 重新执行 GT / Pred SQL，记录结果规模、列信息与样例；
4. 用轻量 SQL 特征（表集合、嵌套、聚合、JOIN、过滤字面量等）做规则归因；
5. 输出汇总统计、抽样案例、全量错例 JSON/CSV 供人工复核。

本模块只读取已有文件，不修改评测与推理流程。
"""
import csv
import json
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from znws.utils.dataset import database_path, load_questions

LEVELS = ["simple", "moderate", "challenging"]
STRUCTURE_KEYS = [
    "single_table",
    "multi_table",
    "nested",
    "non_nested",
    "cte",
    "union",
]
MAX_SCAN_ROWS = 50000


# ------------------------------------------------------------------ 执行与工具


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execute_sql(
    sql: str, db_path: str, timeout: float, max_rows: int
) -> Dict[str, Any]:
    """执行 SQL 并返回结果摘要；超时通过 sqlite progress handler 中断。"""
    if not sql.strip():
        return {
            "status": "empty",
            "columns": [],
            "row_count": 0,
            "distinct_row_count": 0,
            "sample_rows": [],
            "truncated": False,
            "error": None,
            "elapsed_ms": 0.0,
        }

    started = time.perf_counter()
    conn = sqlite3.connect(db_path)
    try:
        deadline = started + timeout

        def _on_progress() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_on_progress, 10000)
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            fetched = cursor.fetchmany(MAX_SCAN_ROWS + 1)
            truncated = len(fetched) > MAX_SCAN_ROWS
            if truncated:
                fetched = fetched[:MAX_SCAN_ROWS]
        except sqlite3.OperationalError as exc:
            if time.perf_counter() >= deadline:
                return {
                    "status": "timeout",
                    "columns": [],
                    "row_count": 0,
                    "distinct_row_count": 0,
                    "sample_rows": [],
                    "truncated": False,
                    "error": f"执行超过 {timeout}s 被中断",
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            return {
                "status": "error",
                "columns": [],
                "row_count": 0,
                "distinct_row_count": 0,
                "sample_rows": [],
                "truncated": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as exc:
            return {
                "status": "error",
                "columns": [],
                "row_count": 0,
                "distinct_row_count": 0,
                "sample_rows": [],
                "truncated": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }

        safe_rows = [tuple(_json_safe(value) for value in row) for row in fetched]
        return {
            "status": "ok",
            "columns": columns,
            "row_count": len(safe_rows),
            "distinct_row_count": len(set(safe_rows)),
            "sample_rows": [list(row) for row in safe_rows[:max_rows]],
            "truncated": truncated,
            "error": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
    finally:
        conn.close()


# ------------------------------------------------------------------ SQL 特征


def _strip_comments(sql: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", text)


def _extract_tables(sql: str) -> List[str]:
    text = re.sub(r"'(?:''|[^'])*'", "''", _strip_comments(sql))
    matches = re.findall(
        r"\b(?:from|join)\s+([`\"\[]?[A-Za-z_][A-Za-z0-9_]*[`\"\]]?)",
        text,
        flags=re.IGNORECASE,
    )
    return sorted({match.strip("`\"[]").lower() for match in matches})


def _extract_cte_names(sql: str) -> List[str]:
    text = _strip_comments(sql)
    matches = re.findall(
        r"(?:\bwith\b|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(",
        text,
        flags=re.IGNORECASE,
    )
    return sorted({match.lower() for match in matches})


def _extract_filter_literals(sql: str) -> List[str]:
    text = _strip_comments(sql)
    literals = {f"'{match}'" for match in re.findall(r"'((?:''|[^'])*)'", text)}
    clauses = re.findall(
        r"\b(?:where|having)\b(.*?)(?=\bgroup\s+by\b|\border\s+by\b|\blimit\b|\bunion\b|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    numbers = re.findall(
        r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", " ".join(clauses)
    )
    literals.update(numbers)
    return sorted(literals)


def sql_features(sql: str) -> Dict[str, Any]:
    """提取用于错例归因的轻量 SQL 结构特征。"""
    text = _strip_comments(sql or "")
    lowered = text.lower()
    tables = _extract_tables(text)
    ctes = set(_extract_cte_names(text))
    physical_tables = sorted(table for table in tables if table not in ctes)
    select_count = len(re.findall(r"\bselect\b", lowered))
    aggregates = sorted(
        set(re.findall(r"\b(count|sum|avg|min|max|total)\s*\(", lowered))
    )
    return {
        "tables": physical_tables,
        "table_count": len(physical_tables),
        "is_multi_table": len(physical_tables) > 1,
        "select_count": select_count,
        "nested": bool(re.search(r"\(\s*select\b", lowered)),
        "has_cte": bool(re.match(r"\s*with\b", lowered)),
        "has_union": bool(re.search(r"\bunion\b", lowered)),
        "has_distinct": bool(re.search(r"\bselect\s+distinct\b", lowered)),
        "aggregates": aggregates,
        "joins": len(re.findall(r"\bjoin\b", lowered)),
        "where_count": len(re.findall(r"\bwhere\b", lowered)),
        "has_group_by": bool(re.search(r"\bgroup\s+by\b", lowered)),
        "has_having": bool(re.search(r"\bhaving\b", lowered)),
        "has_order_by": bool(re.search(r"\border\s+by\b", lowered)),
        "has_limit": bool(re.search(r"\blimit\b", lowered)),
        "filter_literals": _extract_filter_literals(text),
    }


# ------------------------------------------------------------------ 归因


def _signal(key: str, detail: str) -> Dict[str, str]:
    return {"key": key, "detail": detail}


def _classify(
    status: str, gt_features: Dict[str, Any], pred_features: Dict[str, Any],
    gt_exec: Dict[str, Any], pred_exec: Dict[str, Any],
) -> Tuple[str, List[Dict[str, str]]]:
    """按结构差异给出主归因与全部可疑信号。"""
    if status == "empty_sql":
        return "empty_sql", [_signal("empty_sql", "预测未产出 SQL")]
    if status == "pred_timeout":
        return "pred_not_executable", [_signal("pred_timeout", "预测 SQL 执行超时")]
    if status == "pred_error":
        return "pred_not_executable", [_signal("pred_error", "预测 SQL 不可执行")]
    if status == "gt_timeout":
        return "gt_not_executable", [_signal("gt_timeout", "金标 SQL 执行超时")]
    if status == "gt_error":
        return "gt_not_executable", [_signal("gt_error", "金标 SQL 不可执行")]
    if status in {"db_not_found", "missing_db_id", "missing_gt", "missing_gt_sql"}:
        return status, [_signal(status, f"评测状态为 {status}")]

    signals: List[Dict[str, str]] = []

    if gt_exec.get("status") != "ok" or pred_exec.get("status") != "ok":
        if gt_exec.get("status") != "ok":
            signals.append(_signal("gt_execution", f"金标执行状态: {gt_exec.get('status')}"))
        if pred_exec.get("status") != "ok":
            signals.append(_signal("pred_execution", f"预测执行状态: {pred_exec.get('status')}"))
        return "execution_failed", signals

    gt_tables = set(gt_features["tables"])
    pred_tables = set(pred_features["tables"])
    if gt_tables != pred_tables:
        missing = sorted(gt_tables - pred_tables)
        extra = sorted(pred_tables - gt_tables)
        detail = (
            f"表集合不一致: GT={sorted(gt_tables)} / Pred={sorted(pred_tables)}"
            + (f"; 缺少 {missing}" if missing else "")
            + (f"; 多余 {extra}" if extra else "")
        )
        signals.append(_signal("table_mismatch", detail))

    if gt_features["nested"] and not pred_features["nested"]:
        signals.append(_signal("nested_query_missing", "GT 使用子查询/嵌套，Pred 未使用"))
    elif pred_features["nested"] and not gt_features["nested"]:
        signals.append(_signal("nested_query_extra", "Pred 使用子查询/嵌套，GT 未使用"))

    if gt_features["aggregates"] != pred_features["aggregates"]:
        signals.append(_signal(
            "aggregation_mismatch",
            f"聚合函数不一致: GT={gt_features['aggregates'] or '无'} / "
            f"Pred={pred_features['aggregates'] or '无'}",
        ))

    gt_grouping = (gt_features["has_group_by"], gt_features["has_having"])
    pred_grouping = (pred_features["has_group_by"], pred_features["has_having"])
    if gt_grouping != pred_grouping:
        signals.append(_signal(
            "grouping_mismatch",
            f"GROUP BY/HAVING 不一致: GT={gt_grouping} / Pred={pred_grouping}",
        ))

    if gt_features["joins"] != pred_features["joins"]:
        signals.append(_signal(
            "join_mismatch",
            f"JOIN 数不一致: GT={gt_features['joins']} / Pred={pred_features['joins']}",
        ))

    gt_order_limit = (gt_features["has_order_by"], gt_features["has_limit"])
    pred_order_limit = (pred_features["has_order_by"], pred_features["has_limit"])
    if gt_order_limit != pred_order_limit:
        signals.append(_signal(
            "order_limit_mismatch",
            f"ORDER BY/LIMIT 不一致: GT={gt_order_limit} / Pred={pred_order_limit}",
        ))

    if gt_features["filter_literals"] != pred_features["filter_literals"]:
        signals.append(_signal(
            "filter_mismatch",
            f"过滤字面量不一致: GT={gt_features['filter_literals']} / "
            f"Pred={pred_features['filter_literals']}",
        ))

    gt_col_count = len(gt_exec.get("columns") or [])
    pred_col_count = len(pred_exec.get("columns") or [])
    if gt_col_count != pred_col_count:
        signals.append(_signal(
            "projection_mismatch",
            f"输出列数不一致: GT={gt_col_count} / Pred={pred_col_count}",
        ))

    if gt_exec.get("row_count") != pred_exec.get("row_count"):
        signals.append(_signal(
            "row_count_mismatch",
            f"结果行数不一致: GT={gt_exec.get('row_count')} / "
            f"Pred={pred_exec.get('row_count')}",
        ))

    if gt_exec.get("distinct_row_count") != pred_exec.get("distinct_row_count"):
        signals.append(_signal(
            "distinct_row_mismatch",
            f"去重后行数不一致: GT={gt_exec.get('distinct_row_count')} / "
            f"Pred={pred_exec.get('distinct_row_count')}",
        ))

    if gt_tables != pred_tables:
        return "table_mismatch", signals
    if gt_features["nested"] and not pred_features["nested"]:
        return "nested_query_missing", signals
    if gt_features["aggregates"] != pred_features["aggregates"]:
        return "aggregation_mismatch", signals
    if gt_grouping != pred_grouping:
        return "grouping_mismatch", signals
    if gt_features["joins"] != pred_features["joins"]:
        return "join_mismatch", signals
    if gt_order_limit != pred_order_limit:
        return "order_limit_mismatch", signals
    if gt_features["filter_literals"] != pred_features["filter_literals"]:
        return "filter_mismatch", signals
    if gt_col_count != pred_col_count:
        return "projection_mismatch", signals
    if gt_exec.get("row_count") != pred_exec.get("row_count"):
        return "row_count_mismatch", signals
    if gt_exec.get("distinct_row_count") != pred_exec.get("distinct_row_count"):
        return "distinct_row_mismatch", signals
    return "value_mismatch", signals or [_signal("value_mismatch", "结构一致但结果值不同")]


# ------------------------------------------------------------------ 统计

def _percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _structure_flags(features: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "single_table": not features["is_multi_table"],
        "multi_table": features["is_multi_table"],
        "nested": features["nested"],
        "non_nested": not features["nested"],
        "cte": features["has_cte"],
        "union": features["has_union"],
    }


def _structure_stats(
    all_features: Dict[object, Dict[str, Any]],
    error_ids: set,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for key in STRUCTURE_KEYS:
        total = sum(
            1 for features in all_features.values() if _structure_flags(features)[key]
        )
        errors = sum(
            1 for qid, features in all_features.items()
            if qid in error_ids and _structure_flags(features)[key]
        )
        stats[key] = {
            "total": total,
            "errors": errors,
            "error_rate": _percent(errors, total),
        }
    return stats


# ------------------------------------------------------------------ 主流程


def analyze_errors(
    *,
    eval_report_path,
    preds_path,
    dataset: str,
    split: str,
    sample_size: int = 20,
    seed: int = 42,
    max_rows: int = 3,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """执行错例分析，返回可直接序列化为 JSON 的结果。"""
    report = _load_json(eval_report_path)
    preds = _load_json(preds_path)
    questions = load_questions(split, dataset)

    details = report.get("details") or []
    if not details:
        raise ValueError(f"评测明细为空: {eval_report_path}")

    gt_map = {
        item.get("question_id", index): item
        for index, item in enumerate(questions)
    }
    pred_map = {
        item.get("question_id"): item for item in preds if "question_id" in item
    }

    all_features: Dict[object, Dict[str, Any]] = {}
    for detail in details:
        qid = detail.get("question_id")
        gt = gt_map.get(qid) or {}
        all_features[qid] = sql_features(gt.get("SQL") or "")

    error_details = [detail for detail in details if detail.get("ex") != 1]
    error_ids = {detail.get("question_id") for detail in error_details}

    errors: List[Dict[str, Any]] = []
    for detail in error_details:
        qid = detail.get("question_id")
        gt = gt_map.get(qid) or {}
        pred = pred_map.get(qid) or {}
        status = detail.get("status") or "ok"
        gt_sql = (gt.get("SQL") or "").strip()
        pred_sql = (pred.get("SQL") or "").strip()
        db_id = detail.get("db_id") or gt.get("db_id") or pred.get("db_id")

        gt_features = sql_features(gt_sql)
        pred_features = sql_features(pred_sql)

        db_file = None
        if db_id:
            try:
                candidate = database_path(db_id, split, dataset)
                if candidate.exists():
                    db_file = str(candidate)
            except Exception:
                db_file = None

        if db_file is None:
            gt_exec = {
                "status": "db_missing", "columns": [], "row_count": 0,
                "distinct_row_count": 0, "sample_rows": [], "truncated": False,
                "error": "数据库文件不存在", "elapsed_ms": 0.0,
            }
            pred_exec = dict(gt_exec)
        else:
            if status == "gt_timeout":
                gt_exec = {
                    "status": "timeout", "columns": [], "row_count": 0,
                    "distinct_row_count": 0, "sample_rows": [], "truncated": False,
                    "error": detail.get("error") or "金标 SQL 超时", "elapsed_ms": 0.0,
                }
            elif status == "gt_error":
                gt_exec = {
                    "status": "error", "columns": [], "row_count": 0,
                    "distinct_row_count": 0, "sample_rows": [], "truncated": False,
                    "error": detail.get("error") or "金标 SQL 不可执行", "elapsed_ms": 0.0,
                }
            else:
                gt_exec = _execute_sql(gt_sql, db_file, timeout, max_rows)

            if status == "empty_sql":
                pred_exec = {
                    "status": "empty", "columns": [], "row_count": 0,
                    "distinct_row_count": 0, "sample_rows": [], "truncated": False,
                    "error": None, "elapsed_ms": 0.0,
                }
            elif status == "pred_timeout":
                pred_exec = {
                    "status": "timeout", "columns": [], "row_count": 0,
                    "distinct_row_count": 0, "sample_rows": [], "truncated": False,
                    "error": detail.get("error") or "预测 SQL 超时", "elapsed_ms": 0.0,
                }
            elif status == "pred_error":
                pred_exec = {
                    "status": "error", "columns": [], "row_count": 0,
                    "distinct_row_count": 0, "sample_rows": [], "truncated": False,
                    "error": detail.get("error") or "预测 SQL 不可执行", "elapsed_ms": 0.0,
                }
            else:
                pred_exec = _execute_sql(pred_sql, db_file, timeout, max_rows)

        cause, signals = _classify(
            status, gt_features, pred_features, gt_exec, pred_exec
        )

        errors.append({
            "question_id": qid,
            "db_id": db_id,
            "difficulty": detail.get("difficulty") or "unknown",
            "status": status,
            "ex": detail.get("ex"),
            "ves": detail.get("ves"),
            "pred_error": detail.get("error"),
            "question": gt.get("question") or "",
            "evidence": gt.get("evidence") or "",
            "gt_sql": gt_sql,
            "pred_sql": pred_sql,
            "gt_features": gt_features,
            "pred_features": pred_features,
            "gt_exec": gt_exec,
            "pred_exec": pred_exec,
            "cause": cause,
            "signals": signals,
        })

    difficulty_stats: List[Dict[str, Any]] = []
    for level in LEVELS + ["unknown"]:
        total = sum(
            1 for detail in details
            if (detail.get("difficulty") or "unknown") == level
        )
        error_count = sum(
            1 for detail in error_details
            if (detail.get("difficulty") or "unknown") == level
        )
        if total:
            difficulty_stats.append({
                "level": level,
                "total": total,
                "errors": error_count,
                "error_rate": _percent(error_count, total),
            })

    structure_stats = _structure_stats(all_features, error_ids)

    cross_stats: Dict[str, Dict[str, int]] = {}
    for level in LEVELS + ["unknown"]:
        level_errors = [
            error for error in errors
            if (error.get("difficulty") or "unknown") == level
        ]
        if not level_errors:
            continue
        cross_stats[level] = {
            "errors": len(level_errors),
            "single_table": sum(
                1 for error in level_errors
                if not error["gt_features"]["is_multi_table"]
            ),
            "multi_table": sum(
                1 for error in level_errors
                if error["gt_features"]["is_multi_table"]
            ),
            "nested": sum(1 for error in level_errors if error["gt_features"]["nested"]),
            "non_nested": sum(
                1 for error in level_errors if not error["gt_features"]["nested"]
            ),
        }

    cause_counts = Counter(error["cause"] for error in errors)
    signal_counts = Counter(
        signal["key"] for error in errors for signal in error["signals"]
    )

    db_totals: Counter = Counter()
    db_errors: Counter = Counter()
    for detail in details:
        db_id = detail.get("db_id") or "unknown"
        db_totals[db_id] += 1
        if detail.get("ex") != 1:
            db_errors[db_id] += 1
    db_stats = [
        {
            "db_id": db_id,
            "total": total,
            "errors": db_errors[db_id],
            "error_rate": _percent(db_errors[db_id], total),
        }
        for db_id, total in db_totals.most_common()
    ]
    db_stats.sort(key=lambda item: (-item["errors"], item["db_id"]))

    status_counts_all = Counter(
        detail.get("status") or "ok" for detail in details
    )
    status_counts_errors = Counter(
        detail.get("status") or "ok" for detail in error_details
    )

    total = len(details)
    correct = total - len(error_details)
    randomizer = random.Random(seed)
    sample_count = min(max(0, int(sample_size)), len(errors))
    sample_entries = randomizer.sample(errors, sample_count) if sample_count else []
    sample_ids = [entry["question_id"] for entry in sample_entries]

    return {
        "dataset": dataset,
        "split": split,
        "preds_path": str(preds_path),
        "eval_report_path": str(eval_report_path),
        "sample_size": sample_count,
        "seed": seed,
        "timeout": timeout,
        "max_rows": max_rows,
        "summary": {
            "total": total,
            "ex_correct": correct,
            "errors": len(error_details),
            "ex": _percent(correct, total),
            "status_counts_all": dict(status_counts_all),
            "status_counts_errors": dict(status_counts_errors),
        },
        "difficulty_stats": difficulty_stats,
        "structure_stats": structure_stats,
        "error_structure": {
            key: sum(
                1 for error in errors
                if _structure_flags(error["gt_features"])[key]
            )
            for key in STRUCTURE_KEYS
        },
        "cross_stats": cross_stats,
        "cause_counts": dict(cause_counts.most_common()),
        "signal_counts": dict(signal_counts.most_common()),
        "db_stats": db_stats,
        "sample_ids": sample_ids,
        "errors": errors,
    }


# ------------------------------------------------------------------ 报告输出


def _format_list(values: List[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else "无"


def _format_exec(exec_info: Dict[str, Any]) -> str:
    if exec_info.get("status") != "ok":
        message = f"{exec_info.get('status')}"
        if exec_info.get("error"):
            message += f"（{exec_info['error']}）"
        return message
    column_count = len(exec_info.get("columns") or [])
    truncated = "，已截断" if exec_info.get("truncated") else ""
    return (
        f"成功，{exec_info.get('row_count')} 行 × {column_count} 列"
        f"（去重后 {exec_info.get('distinct_row_count')} 行{truncated}）"
    )


def _sample_rows_text(exec_info: Dict[str, Any]) -> str:
    rows = exec_info.get("sample_rows") or []
    if not rows:
        return "（无样例）"
    return json.dumps(rows, ensure_ascii=False)


def _case_markdown(entry: Dict[str, Any], index: int) -> str:
    gt_exec = entry["gt_exec"]
    pred_exec = entry["pred_exec"]
    signals = "；".join(signal["detail"] for signal in entry["signals"]) or "无"
    evidence = entry.get("evidence") or "（无）"
    lines = [
        f"### 案例 {index}：question_id={entry['question_id']}",
        "",
        f"- 数据库：`{entry['db_id']}`",
        f"- 难度：{entry['difficulty']}",
        f"- 评测状态：`{entry['status']}`",
        f"- 主归因：**{entry['cause']}**",
        f"- 归因依据：{signals}",
        "",
        f"**问题**：{entry['question']}",
        "",
        "**BIRD evidence**：",
        "",
        f"> {evidence}",
        "",
        "**GT SQL**：",
        "",
        "```sql",
        entry["gt_sql"] or "-- 空",
        "```",
        "",
        "**Pred SQL**：",
        "",
        "```sql",
        entry["pred_sql"] or "-- 空",
        "```",
        "",
        "**结果对比**：",
        "",
        f"- GT：{_format_exec(gt_exec)}",
        f"- Pred：{_format_exec(pred_exec)}",
        "",
        "样例结果：",
        "",
        "```text",
        f"GT   : {_sample_rows_text(gt_exec)}",
        f"Pred : {_sample_rows_text(pred_exec)}",
        "```",
        "",
        "**结构差异**：",
        "",
        f"- 表：GT={entry['gt_features']['tables']} / Pred={entry['pred_features']['tables']}",
        f"- 嵌套：GT={entry['gt_features']['nested']} / Pred={entry['pred_features']['nested']}",
        f"- 聚合：GT={_format_list(entry['gt_features']['aggregates'])} / "
        f"Pred={_format_list(entry['pred_features']['aggregates'])}",
        f"- JOIN 数：GT={entry['gt_features']['joins']} / Pred={entry['pred_features']['joins']}",
        f"- GROUP BY/HAVING：GT={entry['gt_features']['has_group_by']}/"
        f"{entry['gt_features']['has_having']} / "
        f"Pred={entry['pred_features']['has_group_by']}/"
        f"{entry['pred_features']['has_having']}",
        f"- 过滤字面量：GT={entry['gt_features']['filter_literals']} / "
        f"Pred={entry['pred_features']['filter_literals']}",
        "",
    ]
    return "\n".join(lines)


def render_markdown(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    errors = result["errors"]
    sample_ids = set(result["sample_ids"])
    sample_entries = sorted(
        [error for error in errors if error["question_id"] in sample_ids],
        key=lambda item: item["question_id"],
    )

    lines = [
        "# 错例分析报告",
        "",
        f"- 数据集：{result['dataset']} / {result['split']}",
        f"- 预测文件：{result['preds_path']}",
        f"- 评测明细：{result['eval_report_path']}",
        f"- 总体 EX：{summary['ex']:.2f}%（{summary['ex_correct']}/{summary['total']}）",
        f"- 错例数：{summary['errors']}",
        f"- 抽样数：{result['sample_size']}（seed={result['seed']}）",
        "",
        "## 1. 错例状态与难度分布",
        "",
        "### 1.1 错例状态",
        "",
        "| 状态 | 题数 |",
        "| --- | ---: |",
    ]
    for status, count in sorted(
        summary["status_counts_errors"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {status} | {count} |")

    lines += [
        "",
        "### 1.2 难度分布",
        "",
        "| 难度 | 总题数 | 错例数 | 错误率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in result["difficulty_stats"]:
        lines.append(
            f"| {item['level']} | {item['total']} | {item['errors']} | "
            f"{item['error_rate']:.2f}% |"
        )

    lines += [
        "",
        "## 2. 查询结构分布",
        "",
        "### 2.1 错例结构占比",
        "",
        "| 结构 | 错例数 | 占错例比例 |",
        "| --- | ---: | ---: |",
    ]
    error_structure = result["error_structure"]
    total_errors = summary["errors"] or 1
    for key in STRUCTURE_KEYS:
        count = error_structure.get(key, 0)
        lines.append(f"| {key} | {count} | {count / total_errors * 100:.2f}% |")

    lines += [
        "",
        "### 2.2 不同结构的错误率（全量）",
        "",
        "| 结构 | 总题数 | 错例数 | 错误率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in STRUCTURE_KEYS:
        item = result["structure_stats"].get(key, {})
        lines.append(
            f"| {key} | {item.get('total', 0)} | {item.get('errors', 0)} | "
            f"{item.get('error_rate', 0.0):.2f}% |"
        )

    lines += [
        "",
        "### 2.3 难度 × 查询结构（仅错例）",
        "",
        "| 难度 | 错例数 | 单表 | 多表 | 嵌套 | 无嵌套 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level, item in result["cross_stats"].items():
        lines.append(
            f"| {level} | {item['errors']} | {item['single_table']} | "
            f"{item['multi_table']} | {item['nested']} | {item['non_nested']} |"
        )

    lines += [
        "",
        "## 3. 疑似原因统计",
        "",
        "### 3.1 主归因",
        "",
        "| 疑似原因 | 题数 | 占错例比例 |",
        "| --- | ---: | ---: |",
    ]
    for cause, count in result["cause_counts"].items():
        lines.append(f"| {cause} | {count} | {count / total_errors * 100:.2f}% |")

    lines += [
        "",
        "### 3.2 归因信号（可多选）",
        "",
        "| 信号 | 出现次数 |",
        "| --- | ---: |",
    ]
    for signal_key, count in result["signal_counts"].items():
        lines.append(f"| {signal_key} | {count} |")

    lines += [
        "",
        "## 4. 数据库错误分布（前 15）",
        "",
        "| 数据库 | 总题数 | 错例数 | 错误率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in result["db_stats"][:15]:
        lines.append(
            f"| {item['db_id']} | {item['total']} | {item['errors']} | "
            f"{item['error_rate']:.2f}% |"
        )

    lines += [
        "",
        f"## 5. 错例抽样对比（{len(sample_entries)} 例）",
        "",
    ]
    for index, entry in enumerate(sample_entries, 1):
        lines.append(_case_markdown(entry, index))

    lines += [
        "",
        "## 6. 全量错例清单",
        "",
        "| question_id | 数据库 | 难度 | 状态 | 主归因 | GT 行×列 | Pred 行×列 |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for error in sorted(errors, key=lambda item: item["question_id"]):
        gt_exec = error["gt_exec"]
        pred_exec = error["pred_exec"]
        gt_shape = (
            f"{gt_exec.get('row_count')}×{len(gt_exec.get('columns') or [])}"
            if gt_exec.get("status") == "ok" else gt_exec.get("status")
        )
        pred_shape = (
            f"{pred_exec.get('row_count')}×{len(pred_exec.get('columns') or [])}"
            if pred_exec.get("status") == "ok" else pred_exec.get("status")
        )
        lines.append(
            f"| {error['question_id']} | {error['db_id']} | {error['difficulty']} | "
            f"{error['status']} | {error['cause']} | {gt_shape} | {pred_shape} |"
        )

    lines += [
        "",
        "## 7. 方法与口径",
        "",
        "- 错例集合直接采用评测明细中 `ex=0` 的题目，与官方 EX（执行结果集相等）保持一致。",
        "- 查询结构特征从 SQL 文本提取：GT 表集合、是否含子查询、聚合函数、JOIN 数、"
        "GROUP BY/HAVING、ORDER BY/LIMIT、过滤字面量等。",
        "- 主归因是规则优先级结果，`signals` 保留全部可疑信号；规则无法区分时标为 "
        "`value_mismatch`，需要人工结合结果样例判断。",
        "- 结果摘要通过重新执行 SQL 获得，单条执行超时可配置；评测中已知的 GT 超时/预测超时"
        "不会重复执行。",
        "- 全量错例的 SQL、结果样例与结构特征保存在同名 JSON，CSV 便于表格化人工复核。",
        "",
    ]
    return "\n".join(lines)


def write_analysis(result: Dict[str, Any], output_dir) -> Tuple[Path, Path, Path]:
    """写出 Markdown / JSON / CSV 三种报告，返回路径。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"error_analysis_{result['dataset']}_{result['split']}"
    md_path = output_path / f"{stem}.md"
    json_path = output_path / f"{stem}.json"
    csv_path = output_path / f"{stem}.csv"

    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "question_id", "db_id", "difficulty", "status", "ex", "ves",
        "cause", "signals", "question", "evidence",
        "gt_sql", "pred_sql",
        "gt_status", "gt_columns", "gt_row_count", "gt_distinct_row_count",
        "pred_status", "pred_columns", "pred_row_count", "pred_distinct_row_count",
        "gt_tables", "pred_tables", "gt_nested", "pred_nested",
        "gt_aggregates", "pred_aggregates", "gt_joins", "pred_joins",
        "gt_filter_literals", "pred_filter_literals", "pred_error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for error in result["errors"]:
            gt_exec = error["gt_exec"]
            pred_exec = error["pred_exec"]
            writer.writerow({
                "question_id": error["question_id"],
                "db_id": error["db_id"],
                "difficulty": error["difficulty"],
                "status": error["status"],
                "ex": error["ex"],
                "ves": error["ves"],
                "cause": error["cause"],
                "signals": " | ".join(
                    signal["detail"] for signal in error["signals"]
                ),
                "question": error["question"],
                "evidence": error["evidence"],
                "gt_sql": error["gt_sql"],
                "pred_sql": error["pred_sql"],
                "gt_status": gt_exec.get("status"),
                "gt_columns": " | ".join(gt_exec.get("columns") or []),
                "gt_row_count": gt_exec.get("row_count"),
                "gt_distinct_row_count": gt_exec.get("distinct_row_count"),
                "pred_status": pred_exec.get("status"),
                "pred_columns": " | ".join(pred_exec.get("columns") or []),
                "pred_row_count": pred_exec.get("row_count"),
                "pred_distinct_row_count": pred_exec.get("distinct_row_count"),
                "gt_tables": " | ".join(error["gt_features"]["tables"]),
                "pred_tables": " | ".join(error["pred_features"]["tables"]),
                "gt_nested": error["gt_features"]["nested"],
                "pred_nested": error["pred_features"]["nested"],
                "gt_aggregates": " | ".join(error["gt_features"]["aggregates"]),
                "pred_aggregates": " | ".join(error["pred_features"]["aggregates"]),
                "gt_joins": error["gt_features"]["joins"],
                "pred_joins": error["pred_features"]["joins"],
                "gt_filter_literals": " | ".join(error["gt_features"]["filter_literals"]),
                "pred_filter_literals": " | ".join(error["pred_features"]["filter_literals"]),
                "pred_error": error.get("pred_error"),
            })

    return md_path, json_path, csv_path
