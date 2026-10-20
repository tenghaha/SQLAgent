"""单题运行：按 question_id 指定一条问题运行 SQL Agent，其余配置与 batch_eval 一致。

默认会把该题结果合并写入与 batch_eval 相同的 preds.json；加 --no-save 则只打印。

用法:
    python single_eval.py --question-id 5 --split dev_20240627
    python single_eval.py --question-id 0 --split dev_20240627 --model qwen3.6-flash
    python single_eval.py --question-id 5 --split dev_20240627 --no-save
"""
import argparse
import sys
from pathlib import Path

from znws.sql_agent import run_sql_agent
from znws.sql_agent.tracing import DEFAULT_LOG_DIR
from znws.utils.dataset import default_dataset_name, default_split, load_questions
from znws.utils.preds import clean_sql, load_existing, write_preds


def _print_rows(columns, rows, limit: int = 20) -> None:
    """以等宽文本打印查询结果。"""
    if not columns:
        print("  （无列信息）")
        return
    widths = [len(str(col)) for col in columns]
    for row in rows[:limit]:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    print("  " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows[:limit]:
        print("  " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    if len(rows) > limit:
        print(f"  ...（共 {len(rows)} 行，仅显示前 {limit} 行）")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 question_id 运行单条问题")
    parser.add_argument("--question-id", type=int, required=True,
                        help="题目编号（dev 为 question_id；train 无该字段，使用记录下标）")
    parser.add_argument("--dataset", default=None, help="数据集名（缺省用 datasets.yaml 的 default_dataset）")
    parser.add_argument("--split", default=None, help="切分名（缺省用该数据集的默认切分）")
    parser.add_argument("--output", default=None, help="preds.json 输出路径（缺省与 batch_eval 相同）")
    parser.add_argument("--no-save", action="store_true", help="只打印结果，不写入 preds.json")
    parser.add_argument("--model", default=None, help="覆盖 OPENAI_API_MODEL")
    parser.add_argument("--base-url", default=None, help="覆盖 OPENAI_API_BASE")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="单题运行日志目录")
    parser.add_argument("--max-rows", type=int, default=50, help="结果预览行数上限")
    args = parser.parse_args()

    dataset = args.dataset or default_dataset_name()
    split = args.split or default_split(dataset)

    try:
        questions = load_questions(split, dataset)
    except Exception as exc:
        print(f"读取问答文件失败: {exc}")
        return 1

    target = None
    for index, item in enumerate(questions):
        if item.get("question_id", index) == args.question_id:
            target = item
            break
    if target is None:
        print(f"未找到 question_id={args.question_id}（数据集 {dataset} / {split}，共 {len(questions)} 题）")
        return 1

    query = target.get("question", "")
    gt_db_id = target.get("db_id")
    if not gt_db_id:
        print(f"题目 {args.question_id} 缺少 db_id，无法运行。")
        return 1

    print(f">>> 数据集 {dataset} / {split} | question_id={args.question_id} | db_id={gt_db_id}")
    print(f">>> 问题: {query}")

    result = run_sql_agent(
        query,
        db_id=gt_db_id,
        dataset=dataset,
        split=split,
        model_name=args.model,
        base_url=args.base_url,
        log_dir=args.log_dir,
        max_rows=args.max_rows,
    )

    plan = result.schema_plan
    print("\n[Schema 方案]")
    if plan:
        print(f"  表        : {', '.join(plan.get('tables') or [])}")
        print(f"  字段      : {', '.join(plan.get('fields') or [])}")
        print(f"  统计字段  : {', '.join(plan.get('statistics_fields') or []) or '无'}")
        print(f"  外键关系  : {', '.join(plan.get('relationships') or []) or '无'}")
        print(f"  理由      : {plan.get('reasoning')}")
    else:
        print("  未获得结构化 Schema 方案")

    print("\n[最终 SQL]")
    print(result.final_sql or "  （未产出 SQL）")

    execution = result.execution
    print("\n[执行结果]")
    if not execution:
        print("  未执行")
    elif execution.get("status") == "error":
        print(f"  执行失败: {execution.get('error')}")
    else:
        truncated = "（已截断）" if execution.get("truncated") else ""
        print(f"  返回 {execution.get('returned_rows')} 行 {truncated}")
        _print_rows(execution.get("columns") or [], execution.get("rows") or [])

    print("\n" + "-" * 60)
    print(f"用时 : {result.elapsed_seconds:.1f}s")
    print(f"日志 : {Path(result.log_path).name}")
    print(f"路径 : {result.log_path}")
    if result.error:
        print(f"错误 : {result.error}")

    if not args.no_save:
        output_path = Path(args.output) if args.output else Path("outputs") / "preds" / f"preds_{dataset}_{split}.json"
        records = load_existing(output_path)
        records[args.question_id] = {
            "question_id": args.question_id,
            "db_id": result.db_id or gt_db_id,
            "SQL": clean_sql(result.final_sql),
        }
        write_preds(output_path, list(records.values()))
        print(f"preds: {output_path}")

    return 0 if result.error is None and (result.final_sql or "").strip() else 1


if __name__ == "__main__":
    sys.exit(main())
