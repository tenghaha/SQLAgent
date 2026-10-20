"""BIRD 官方口径评测：EX、VES 与 SQL 实际执行耗时。

指标实现参照官方仓库 ``D:/DAMO-ConvAI/bird``：

- EX  : ``llm/src/evaluation.py`` —— 预测结果集与金标结果集完全相等（set 比较）。
- VES : ``llm/src/evaluation_ves.py`` —— 仅对结果正确的题目重复执行 N 次，
        取 ``ratio = gt_time / pred_time``，经官方 3σ 规则过滤后求均值，
        ``VES = sqrt(ratio) * 100``；错误、超时与未命中的题目计 0。
- 耗时: 不含 SQL 生成时间，只统计 sqlite 执行。

与官方的唯一差异在计时范围：官方只计 ``cursor.execute`` 的耗时，但 Python
sqlite3 对 SELECT 是惰性执行，实际计算多发生在 ``fetchall``；为反映真实执行
成本，这里把取数一并计时。明细中同时保留 ``exec_ms``（官方口径）与
``elapsed_ms``（execute + fetch，报告采用后者）。
"""
import json
import math
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from znws.utils.dataset import (
    database_path,
    default_dataset_name,
    default_split,
    load_questions,
)

LEVELS = ["simple", "moderate", "challenging"]
PROGRESS_HANDLER_INTERVAL = 10000


class SqlExecutionTimeout(Exception):
    """SQL 执行超过给定时间。"""


def _execute_sql(sql: str, db_path: str, timeout: float) -> Tuple[list, float, float]:
    """执行 SQL，返回 (结果行, execute 耗时, execute+fetch 耗时)。

    超时通过 sqlite3 的 progress handler 在语句执行期间中断，避免挂死。
    计时口径与官方一致（仅 SQL 执行），生成阶段不计入。
    """
    conn = sqlite3.connect(db_path)
    try:
        deadline = time.perf_counter() + timeout

        def _on_progress() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_on_progress, PROGRESS_HANDLER_INTERVAL)
        cursor = conn.cursor()
        started = time.perf_counter()
        try:
            cursor.execute(sql)
            exec_seconds = time.perf_counter() - started
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            if time.perf_counter() >= deadline:
                raise SqlExecutionTimeout(f"执行超过 {timeout}s 被中断") from exc
            raise
        elapsed_seconds = time.perf_counter() - started
        return rows, exec_seconds, elapsed_seconds
    finally:
        conn.close()


def _clean_abnormal(values: List[float]) -> List[float]:
    """复刻官方 clean_abnormal：按总体标准差做 3σ 过滤（严格不等号）。"""
    if not values:
        return []
    count = len(values)
    mean = sum(values) / count
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / count)
    return [value for value in values if mean - 3 * std < value < mean + 3 * std]


def _evaluate_one(task: dict) -> dict:
    """评测单题（可在子进程中运行）。"""
    result: Dict[str, Any] = {
        "order": task["order"],
        "question_id": task["question_id"],
        "db_id": task["db_id"],
        "difficulty": task["difficulty"],
        "ex": 0,
        "ves": 0.0,
        "time_ratio": 0.0,
        "exec_ms": None,
        "elapsed_ms": None,
        "status": "ok",
        "error": None,
    }

    try:
        pred_rows, pred_exec, pred_elapsed = _execute_sql(
            task["pred_sql"], task["db_path"], task["timeout"]
        )
    except SqlExecutionTimeout as exc:
        result.update(status="pred_timeout", error=str(exc))
        return result
    except Exception as exc:  # 预测 SQL 本身不可执行
        result.update(status="pred_error", error=f"{type(exc).__name__}: {exc}")
        return result

    result["exec_ms"] = pred_exec * 1000
    result["elapsed_ms"] = pred_elapsed * 1000

    try:
        gt_rows, _, _ = _execute_sql(task["gt_sql"], task["db_path"], task["timeout"])
    except SqlExecutionTimeout as exc:
        result.update(status="gt_timeout", error=str(exc))
        return result
    except Exception as exc:
        result.update(status="gt_error", error=f"{type(exc).__name__}: {exc}")
        return result

    result["ex"] = int(set(pred_rows) == set(gt_rows))
    if not result["ex"]:
        return result

    # 官方 VES：仅对结果正确的题目多次执行取时间比
    ratios: List[float] = []
    try:
        for _ in range(task["ves_iterations"]):
            _, _, pred_time = _execute_sql(task["pred_sql"], task["db_path"], task["timeout"])
            _, _, gt_time = _execute_sql(task["gt_sql"], task["db_path"], task["timeout"])
            ratios.append(gt_time / pred_time)
    except SqlExecutionTimeout:
        ratios = []
    except Exception:
        ratios = []

    cleaned = _clean_abnormal(ratios)
    if cleaned:
        result["time_ratio"] = sum(cleaned) / len(cleaned)
        result["ves"] = math.sqrt(result["time_ratio"]) * 100
    return result


def _build_result(
    *,
    dataset: str,
    split: str,
    preds_path: Path,
    records: List[dict],
    pre_results: List[dict],
    task_results: List[dict],
    ves_iterations: int,
    timeout: float,
    workers: int,
) -> dict:
    details = sorted(pre_results + task_results, key=lambda item: item["order"])
    for item in details:
        item.pop("order", None)

    executed = [item for item in details if item["elapsed_ms"] is not None]
    status_counts: Dict[str, int] = {}
    for item in details:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    difficulties = list(LEVELS)
    if any(item["difficulty"] not in LEVELS for item in details):
        difficulties.append("unknown")

    metrics: Dict[str, dict] = {}
    for level in difficulties + ["total"]:
        group = details if level == "total" else [item for item in details if item["difficulty"] == level]
        if not group:
            continue
        correct = sum(item["ex"] for item in group)
        timings = [item["elapsed_ms"] for item in group if item["elapsed_ms"] is not None]
        metrics[level] = {
            "count": len(group),
            "ex_count": correct,
            "ex": correct / len(group) * 100,
            "ves": sum(item["ves"] for item in group) / len(group),
            "exec_count": len(timings),
            "avg_exec_ms": sum(timings) / len(timings) if timings else None,
        }

    return {
        "dataset": dataset,
        "split": split,
        "preds_path": str(preds_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ves_iterations": ves_iterations,
            "timeout": timeout,
            "workers": workers,
        },
        "counts": {
            "preds": len(records),
            "evaluated": len(details),
            "executed": len(executed),
            "ex_correct": sum(item["ex"] for item in details),
        },
        "metrics": metrics,
        "status_counts": status_counts,
        "details": details,
    }


def evaluate_preds(
    preds_path,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    *,
    ves_iterations: int = 100,
    timeout: float = 30.0,
    workers: Optional[int] = None,
    progress: bool = True,
) -> dict:
    """对 preds.json 执行 EX / VES / 执行耗时评测，返回报告数据。"""
    dataset_name = dataset or default_dataset_name()
    split_name = split or default_split(dataset_name)
    preds_file = Path(preds_path)
    if not preds_file.exists():
        raise FileNotFoundError(f"未找到预测文件 {preds_file}")

    records = json.loads(preds_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"预测文件 {preds_file} 应为记录列表（BIRD preds 格式）")

    questions = load_questions(split_name, dataset_name)
    gt_map: Dict[object, dict] = {}
    for index, item in enumerate(questions):
        gt_map[item.get("question_id", index)] = item

    ves_iterations = max(1, int(ves_iterations))
    timeout = float(timeout)

    tasks: List[dict] = []
    pre_results: List[dict] = []

    for order, record in enumerate(records):
        question_id = record.get("question_id")
        gt = gt_map.get(question_id)
        base = {
            "order": order,
            "question_id": question_id,
            "db_id": record.get("db_id"),
            "difficulty": (gt or {}).get("difficulty") or "unknown",
            "ex": 0,
            "ves": 0.0,
            "time_ratio": 0.0,
            "exec_ms": None,
            "elapsed_ms": None,
            "error": None,
        }

        if gt is None:
            pre_results.append({**base, "status": "missing_gt"})
            continue

        pred_sql = (record.get("SQL") or "").strip()
        if not pred_sql:
            pre_results.append({**base, "status": "empty_sql"})
            continue

        gt_sql = (gt.get("SQL") or "").strip()
        if not gt_sql:
            pre_results.append({**base, "status": "missing_gt_sql"})
            continue

        db_id = record.get("db_id") or gt.get("db_id")
        db_file = database_path(db_id, split_name, dataset_name) if db_id else None
        if not db_file or not db_file.exists():
            pre_results.append({**base, "status": "db_not_found"})
            continue

        tasks.append({
            "order": order,
            "question_id": question_id,
            "db_id": db_id,
            "difficulty": base["difficulty"],
            "pred_sql": pred_sql,
            "gt_sql": gt_sql,
            "db_path": str(db_file),
            "ves_iterations": ves_iterations,
            "timeout": timeout,
        })

    if records and not tasks and pre_results and all(
        item["status"] == "db_not_found" for item in pre_results
    ):
        raise ValueError(
            f"预测中的 db_id 在切分 {split_name!r} 下均不存在，--split 可能与 preds 文件不匹配"
            f"（preds 文件名通常为 preds_<dataset>_<split>.json，可据此推断切分）。"
        )

    if workers is None:
        workers = min(os.cpu_count() or 1, 16)
    workers = max(1, min(int(workers), len(tasks) or 1))

    task_results: List[dict] = []
    if tasks:
        if workers == 1:
            for index, task in enumerate(tasks, 1):
                task_results.append(_evaluate_one(task))
                _report_progress(index, len(tasks), progress)
        else:
            chunksize = max(1, len(tasks) // (workers * 4))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for index, item in enumerate(
                    pool.map(_evaluate_one, tasks, chunksize=chunksize), 1
                ):
                    task_results.append(item)
                    _report_progress(index, len(tasks), progress)

    return _build_result(
        dataset=dataset_name,
        split=split_name,
        preds_path=preds_file,
        records=records,
        pre_results=pre_results,
        task_results=task_results,
        ves_iterations=ves_iterations,
        timeout=timeout,
        workers=workers,
    )


def _report_progress(done: int, total: int, enabled: bool) -> None:
    if not enabled or total < 20:
        return
    step = max(1, total // 10)
    if done % step == 0 or done == total:
        print(f">>> 评测进度 {done}/{total}")


def format_summary(result: dict) -> str:
    """生成终端摘要。"""
    total = result["metrics"].get("total")
    if not total:
        return "评测完成：没有可评分的题目。"
    counts = result["counts"]
    return (
        f"EX: {total['ex']:.2f}% ({total['ex_count']}/{total['count']}) | "
        f"VES: {total['ves']:.2f} | "
        f"平均 SQL 执行耗时: {_format_ms(total['avg_exec_ms'])}"
        f"（n={total['exec_count']}，不含生成时间）\n"
        f"预测 {counts['preds']} 条，可执行 {counts['executed']} 条。"
    )


def _format_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _format_rate(value: float) -> str:
    return f"{value:.2f}"


def render_markdown(result: dict) -> str:
    """将评测结果渲染为 Markdown 报告。"""
    config = result["config"]
    counts = result["counts"]
    total = result["metrics"].get("total") or {}

    lines = [
        "# SQL Agent 评测报告",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 数据集 | {result['dataset']} / {result['split']} |",
        f"| 预测文件 | {result['preds_path']} |",
        f"| 生成时间 | {result['generated_at']} |",
        f"| VES 迭代次数 | {config['ves_iterations']} |",
        f"| 单次执行超时 | {config['timeout']:.1f} s |",
        f"| 评测并行度 | {config['workers']} |",
        f"| 预测题数 | {counts['preds']} |",
        "",
        "## 指标汇总",
        "",
        f"- **EX（Execution Accuracy）**: {_format_rate(total.get('ex', 0.0))}%"
        f"（{counts['ex_correct']}/{counts['evaluated']}）",
        f"- **VES（Valid Efficiency Score）**: {_format_rate(total.get('ves', 0.0))}",
        f"- **平均 SQL 执行耗时**: {_format_ms(total.get('avg_exec_ms'))}"
        f"（n={counts['executed']}，仅 SQL 执行，不含生成时间）",
        "",
        "## 分难度指标",
        "",
        "| 难度 | 题数 | EX (%) | EX 题数 | VES | 平均执行耗时 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for level in LEVELS + ["unknown", "total"]:
        metric = result["metrics"].get(level)
        if not metric:
            continue
        lines.append(
            f"| {level} | {metric['count']} | {metric['ex']:.2f} | "
            f"{metric['ex_count']} | {metric['ves']:.2f} | "
            f"{_format_ms(metric['avg_exec_ms'])} |"
        )

    lines += [
        "",
        "## 状态统计",
        "",
        "| 状态 | 题数 |",
        "| --- | ---: |",
    ]
    for status, count in sorted(result["status_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {status} | {count} |")

    abnormal = [item for item in result["details"] if item["status"] != "ok"]
    if abnormal:
        lines += [
            "",
            "## 异常题目（前 20 条）",
            "",
            "| question_id | db_id | 状态 | 错误 |",
            "| ---: | --- | --- | --- |",
        ]
        for item in abnormal[:20]:
            error = (item.get("error") or "").replace("|", "\\|")
            lines.append(
                f"| {item['question_id']} | {item['db_id']} | {item['status']} | {error} |"
            )

    lines += [
        "",
        "## 指标说明",
        "",
        "- **EX**：按官方 `evaluation.py`，预测 SQL 与金标 SQL 的执行结果集完全相等即记为 1。",
        "- **VES**：按官方 `evaluation_ves.py`，仅对 EX 正确的题目重复执行，"
        "取 `gt_time / pred_time` 的均值后计算 `sqrt(ratio) * 100`；"
        "3σ 异常值过滤规则与官方 `clean_abnormal` 一致；错误/超时/未命中计 0。",
        "- **执行耗时**：仅统计 sqlite 中 SQL 的实际执行（含结果读取），"
        "不含 LLM 生成与网络等待时间；官方 `execute` 计时值保留在明细 JSON 的 `exec_ms` 字段中。",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict, report_path) -> Tuple[Path, Path]:
    """写出 Markdown 报告与同名 JSON 明细，返回 (报告路径, 明细路径)。"""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
    details_path = path.with_suffix(".json")
    if details_path == path:  # 报告路径本身是 .json 时避免覆盖
        details_path = path.with_name(path.stem + "_details.json")
    details_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path, details_path
