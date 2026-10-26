"""错例分析 CLI：读取已有评测报告与 preds，产出可复用的错例分析报告。

分析内容包括：错例抽样对比、难度/结构统计、表/字段 P/R、规则归因与
全量错例明细。表/字段集合优先复用评测明细，旧报告则通过 SQLite
authorizer + EXPLAIN 从 GT/Pred SQL 中提取，不执行查询。

用法:
    python analyze_errors.py --dataset bird --split mini_dev
    python analyze_errors.py --eval-report outputs/reports/eval_bird_mini_dev.json \
        --preds outputs/preds/preds_bird_mini_dev.json --sample-size 20
    python analyze_errors.py            # 自动选择最新的 eval_*.json

输出（默认 outputs/reports/）:
    error_analysis_<dataset>_<split>.md     # 汇总统计 + 抽样案例
    error_analysis_<dataset>_<split>.json   # 全量错例对比
    error_analysis_<dataset>_<split>.csv    # 全量错例表格，便于人工复核
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from znws.utils.dataset import default_dataset_name, default_split
from znws.utils.error_analysis import analyze_errors, write_analysis


def _latest_eval_report() -> Optional[Path]:
    candidates = sorted(
        Path("outputs").glob("**/eval_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_eval_report(args) -> Optional[Path]:
    if args.eval_report:
        return Path(args.eval_report)

    if args.dataset and args.split:
        for candidate in (
            Path("outputs") / "reports" / f"eval_{args.dataset}_{args.split}.json",
            Path("outputs") / args.split / "reports" / f"eval_{args.dataset}_{args.split}.json",
        ):
            if candidate.exists():
                return candidate
        return None

    return _latest_eval_report()


def _resolve_preds(args, report: dict, dataset: str, split: str) -> Optional[Path]:
    if args.preds:
        return Path(args.preds)

    candidates = []
    report_preds = report.get("preds_path")
    if report_preds:
        candidates.append(Path(report_preds))
    candidates.extend([
        Path("outputs") / "preds" / f"preds_{dataset}_{split}.json",
        Path("outputs") / split / "preds" / f"preds_{dataset}_{split}.json",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="错例分析：EX=0 题目的 SQL 与结果对比")
    parser.add_argument("--eval-report", default=None, help="评测明细 JSON（缺省自动选择最新）")
    parser.add_argument("--preds", default=None, help="预测文件（缺省从评测报告或默认目录查找）")
    parser.add_argument("--dataset", default=None, help="数据集名（缺省从评测报告读取）")
    parser.add_argument("--split", default=None, help="切分名（缺省从评测报告读取）")
    parser.add_argument("--output-dir", default="outputs/reports", help="报告输出目录")
    parser.add_argument("--sample-size", type=int, default=20, help="Markdown 中详细展示的案例数")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--max-rows", type=int, default=3, help="每个结果集显示的样例行数")
    parser.add_argument("--timeout", type=float, default=30.0, help="单条 SQL 重放超时秒数")
    args = parser.parse_args()

    eval_path = _resolve_eval_report(args)
    if eval_path is None or not eval_path.exists():
        print("未找到评测报告，请先运行 evaluate.py，或用 --eval-report 指定路径。")
        return 1

    try:
        report = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"读取评测报告失败: {exc}")
        return 1

    dataset = args.dataset or report.get("dataset") or default_dataset_name()
    split = args.split or report.get("split") or default_split(dataset)
    preds_path = _resolve_preds(args, report, dataset, split)
    if preds_path is None or not preds_path.exists():
        print("未找到预测文件，请用 --preds 指定路径。")
        return 1

    print(f">>> 数据集 {dataset} / {split}")
    print(f">>> 评测明细: {eval_path}")
    print(f">>> 预测文件: {preds_path}")
    print(">>> 正在重放错例 SQL 并生成分析报告（可能耗时数分钟）...")

    try:
        result = analyze_errors(
            eval_report_path=eval_path,
            preds_path=preds_path,
            dataset=dataset,
            split=split,
            sample_size=args.sample_size,
            seed=args.seed,
            max_rows=args.max_rows,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"错例分析失败: {type(exc).__name__}: {exc}")
        return 1

    md_path, json_path, csv_path = write_analysis(result, args.output_dir)
    summary = result["summary"]
    print(
        f"\n完成：错例 {summary['errors']} 条，"
        f"总体 EX {summary['ex']:.2f}%（{summary['ex_correct']}/{summary['total']}）"
    )
    print(f"报告: {md_path}")
    print(f"明细: {json_path}")
    print(f"表格: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
