"""简化版 SQL Agent 的最小运行入口（多数据集，由 datasets.yaml 控制）。

每次执行只处理一个 query，形成一次完全独立的运行（无多轮反馈、无中断）。

用法:
    python sql_agent_client.py --db-id california_schools "How many schools ..."
    python sql_agent_client.py --db-id financial --split dev_20240627 --verbose
    python sql_agent_client.py --db-id card_games "..." --json       # 以 JSON 输出完整结果
"""
import argparse
import json
import os
import sys

# 工具内部使用工程根目录推导的绝对路径，这里统一以工程根目录为工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from znws.sql_agent import run_sql_agent  # noqa: E402
from znws.sql_agent.tracing import DEFAULT_LOG_DIR  # noqa: E402


def print_rows(columns, rows, limit=20):
    """以等宽文本打印查询结果。"""
    if not columns:
        print("  （无列信息）")
        return
    widths = [len(str(col)) for col in columns]
    for row in rows[:limit]:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    header = "  " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
    print(header)
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows[:limit]:
        print("  " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    if len(rows) > limit:
        print(f"  ...（共 {len(rows)} 行，仅显示前 {limit} 行）")


def main() -> int:
    parser = argparse.ArgumentParser(description="简化版 SQL Agent CLI（单次独立运行）")
    parser.add_argument("query", nargs="?", help="业务问题；缺省时从标准输入读取一条")
    parser.add_argument("--db-id", required=True, help="数据库标识（db_id）")
    parser.add_argument("--dataset", default=None, help="datasets.yaml 中登记的数据集名（缺省用配置的 default_dataset）")
    parser.add_argument("--split", default=None, help="切分名（缺省用该数据集的默认切分）")
    parser.add_argument("--model", default=None, help="覆盖环境变量 OPENAI_API_MODEL")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="运行日志目录")
    parser.add_argument("--max-rows", type=int, default=50, help="结果预览的最大行数")
    parser.add_argument("--verbose", action="store_true", help="把 agent 过程同步打印到控制台")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    args = parser.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Query> ").strip()
        except EOFError:
            query = ""
    if not query:
        print("未提供 query，退出。")
        return 1

    print(f">>> 开始运行: {query}")
    result = run_sql_agent(
        query,
        db_id=args.db_id,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model,
        log_dir=args.log_dir,
        echo=args.verbose,
        max_rows=args.max_rows,
    )

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.error is None else 1

    print("\n" + "=" * 70)
    print(f"run_id : {result.run_id}")
    print(f"数据集 : {result.dataset} / {result.split}")
    print(f"数据库 : {result.db_id}")
    print(f"模型   : {result.model}")
    print(f"用时   : {result.elapsed_seconds:.1f}s")
    print(f"日志   : {result.log_path}")
    print("=" * 70)

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
        print_rows(execution.get("columns") or [], execution.get("rows") or [])

    if result.error:
        print(f"\n[错误] {result.error}")

    return 0 if result.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
