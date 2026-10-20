"""独立评测：读取 preds.json 与金标，产出 EX / VES / SQL 执行耗时报告。

指标口径遵循 BIRD 官方仓库（D:/DAMO-ConvAI/bird）的 evaluation.py 与
evaluation_ves.py；报告同时写出 Markdown 与同名 JSON 明细。

用法:
    python evaluate.py --split dev_20240627
    python evaluate.py --preds outputs/preds/preds_bird_dev_20240627.json --ves-iterations 10
    python evaluate.py --split dev_20240627 --report outputs/reports/custom.md
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from znws.utils.dataset import default_dataset_name, default_split, load_config
from znws.utils.evaluation import evaluate_preds, format_summary, write_report


def _infer_from_preds(preds_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """从 preds_<dataset>_<split>.json 文件名推断数据集与切分。"""
    stem = preds_path.stem
    matches = []
    for dataset_name, entry in load_config()["datasets"].items():
        for split_name in (entry.get("splits") or {}):
            if stem == f"preds_{dataset_name}_{split_name}":
                matches.append((dataset_name, split_name))
    return matches[0] if len(matches) == 1 else (None, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="评测 BIRD 格式 preds.json（EX / VES / 执行耗时）")
    parser.add_argument("--dataset", default=None, help="数据集名（缺省用 datasets.yaml 的 default_dataset）")
    parser.add_argument("--split", default=None, help="切分名（缺省从 preds 文件名推断，否则用默认切分）")
    parser.add_argument("--preds", default=None, help="preds.json 路径（缺省为 outputs/preds/preds_<dataset>_<split>.json）")
    parser.add_argument("--report", default=None, help="报告路径（缺省为 outputs/reports/eval_<dataset>_<split>.md）")
    parser.add_argument("--ves-iterations", type=int, default=100, help="VES 每题的重复执行次数（官方默认 100）")
    parser.add_argument("--timeout", type=float, default=30.0, help="单条 SQL 执行超时秒数")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（缺省按 CPU 自动，最多 16）")
    args = parser.parse_args()

    preds_path = Path(args.preds) if args.preds else None
    inferred_dataset, inferred_split = (None, None)
    if preds_path is not None:
        inferred_dataset, inferred_split = _infer_from_preds(preds_path)

    dataset = args.dataset or inferred_dataset or default_dataset_name()
    split = args.split or (inferred_split if inferred_dataset == dataset else None) or default_split(dataset)
    if preds_path is None:
        preds_path = Path("outputs") / "preds" / f"preds_{dataset}_{split}.json"
    report_path = Path(args.report) if args.report else Path("outputs") / "reports" / f"eval_{dataset}_{split}.md"

    if inferred_split and not args.split:
        print(f">>> 切分由预测文件名推断: {inferred_split}")
    elif inferred_split and args.split and inferred_split != args.split:
        print(f">>> [warn] preds 文件名指向切分 {inferred_split}，但显式指定了 --split {args.split}")
    print(f">>> 数据集 {dataset} / {split}")
    print(f">>> 预测: {preds_path}")
    print(f">>> 报告: {report_path}")

    try:
        result = evaluate_preds(
            preds_path,
            dataset=dataset,
            split=split,
            ves_iterations=args.ves_iterations,
            timeout=args.timeout,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"评测失败: {type(exc).__name__}: {exc}")
        return 1

    report_file, details_file = write_report(result, report_path)
    print("\n" + format_summary(result))
    print(f"\n报告已写入: {report_file}")
    print(f"明细已写入: {details_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
