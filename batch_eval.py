"""批量预测：按 datasets.yaml 指定的数据集/切分运行 SQL Agent，生成官方格式 preds.json。

预测完成后默认调用官方口径评测（EX / VES / SQL 执行耗时）并产出报告；
加 ``--no-eval`` 则只推理、不评测。预测记录采用 BIRD 官方字段：
``question_id`` / ``db_id`` / ``SQL``。

用法:
    python batch_eval.py --split mini_dev --no-eval             # 只推理
    python batch_eval.py --split mini_dev --resume             # 续跑并重试失败题
    python batch_eval.py --split mini_dev --limit 20           # 前 20 题
    python batch_eval.py --split mini_dev                      # 推理 + 评测

长跑保护:
- 每道题在独立子进程中执行，超过 ``--timeout``（默认 300s）会被强制终止，
  不会让单题拖垮整批；
- LLM 请求级超时与瞬时错误重试由 ``OPENAI_API_TIMEOUT`` /
  ``OPENAI_API_MAX_RETRIES`` 控制（见 znws/sql_agent/config.py）；
- ``--resume`` 只跳过已有非空 SQL 的记录，失败题会在下一次续跑时自动重试；
- 需要单次运行内自动重试时可显式设置 ``--retries``（会额外消耗 token）。

说明:
- 使用题目自带的 ``db_id``；缺少 db_id 的题目记为失败并跳过。
- 逐题顺序执行（单 worker 子进程）：``set_active_database`` 是进程级状态，
  不能在同一进程内并发。需要并行请按 ``--limit`` 分片、开多个进程。
- 评测也可事后用 ``python evaluate.py --split <split>`` 单独运行。
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from znws.sql_agent import run_sql_agent
from znws.sql_agent.config import DEFAULT_RECURSION_LIMIT
from znws.sql_agent.tracing import DEFAULT_LOG_DIR
from znws.utils.dataset import default_dataset_name, default_split, load_questions
from znws.utils.evaluation import evaluate_preds, format_summary, write_report
from znws.utils.preds import clean_sql, load_existing, write_preds


def _run_one(payload: dict) -> dict:
    """子进程入口：运行单题推理，返回可序列化结果。"""
    try:
        result = run_sql_agent(
            payload["query"],
            db_id=payload["db_id"],
            question_id=payload["question_id"],
            dataset=payload["dataset"],
            split=payload["split"],
            model_name=payload["model_name"],
            base_url=payload["base_url"],
            log_dir=payload["log_dir"],
            recursion_limit=payload["recursion_limit"],
            max_rows=payload["max_rows"],
        )
        return {
            "sql": clean_sql(result.final_sql),
            "db_id": result.db_id or payload["db_id"],
            "log_path": result.log_path or "",
            "error": result.error,
        }
    except Exception as exc:  # 子进程内兜底，保证父进程总能拿到结构化结果
        return {
            "sql": "",
            "db_id": payload["db_id"],
            "log_path": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="批量推理并评测（EX / VES / SQL 执行耗时）")
    parser.add_argument("--dataset", default=None, help="数据集名（缺省用 datasets.yaml 的 default_dataset）")
    parser.add_argument("--split", default=None, help="切分名（缺省用该数据集的默认切分）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0 表示全量")
    parser.add_argument("--output", default=None, help="preds.json 输出路径")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已有非空 SQL 的记录；失败/空 SQL 记录会重试")
    parser.add_argument("--model", default=None, help="覆盖 OPENAI_API_MODEL")
    parser.add_argument("--base-url", default=None, help="覆盖 OPENAI_API_BASE")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="单题运行日志目录")
    parser.add_argument("--max-rows", type=int, default=50, help="单题结果预览行数上限")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="单题硬超时秒数；超时强制终止子进程，0 表示不限制")
    parser.add_argument("--retries", type=int, default=0,
                        help="单题失败后的额外重试次数（默认 0，靠 --resume 跨批次重试）")
    parser.add_argument("--retry-backoff", type=float, default=5.0, help="重试前等待秒数")
    parser.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT,
                        help="Worker agent 的最大步数")
    parser.add_argument("--no-eval", action="store_true", help="只推理，不执行评测")
    parser.add_argument("--report", default=None, help="评测报告路径（缺省 outputs/reports/eval_<dataset>_<split>.md）")
    parser.add_argument("--ves-iterations", type=int, default=100,
                        help="VES 每题的重复执行次数（官方默认 100）")
    parser.add_argument("--eval-timeout", type=float, default=30.0, help="评测时单条 SQL 执行超时秒数")
    parser.add_argument("--eval-workers", type=int, default=None,
                        help="评测并行进程数（缺省按 CPU 自动，最多 16）")
    args = parser.parse_args()

    dataset = args.dataset or default_dataset_name()
    split = args.split or default_split(dataset)

    try:
        questions = load_questions(split, dataset)
    except Exception as exc:
        print(f"读取问答文件失败: {exc}")
        return 1

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    output_path = Path(args.output) if args.output else Path("outputs") / "preds" / f"preds_{dataset}_{split}.json"
    results: Dict[object, dict] = load_existing(output_path) if args.resume else {}

    print(f">>> 数据集 {dataset} / {split}：本次共 {len(questions)} 题")
    print(f">>> 输出: {output_path}")
    if args.timeout > 0:
        print(f">>> 单题硬超时: {args.timeout:.0f}s（子进程隔离）")
    if args.retries > 0:
        print(f">>> 单题自动重试: {args.retries} 次（会额外消耗 token）")
    else:
        print(">>> 单题自动重试: 关闭（失败题由 --resume 在下一批重试）")
    if args.resume and results:
        done = sum(1 for record in results.values() if (record.get("SQL") or "").strip())
        print(f">>> 断点续跑：已有 {len(results)} 条"
              f"（{done} 条有 SQL 将跳过，{len(results) - done} 条待重试）")

    success = 0
    failed = 0
    skipped = 0
    max_retries = max(0, args.retries)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)
    try:
        for index, item in enumerate(questions):
            question_id = item.get("question_id", index)
            gt_db_id = item.get("db_id")
            query = item.get("question", "")

            if args.resume and question_id in results:
                if (results[question_id].get("SQL") or "").strip():
                    skipped += 1
                    continue

            if not gt_db_id:
                results[question_id] = {"question_id": question_id, "db_id": None, "SQL": ""}
                write_preds(output_path, list(results.values()))
                failed += 1
                print(f"[{index + 1}/{len(questions)}] qid={question_id} status=failed log=- "
                      f"reason=题目缺少 db_id")
                continue

            payload = {
                "query": query,
                "question_id": question_id,
                "db_id": gt_db_id,
                "dataset": dataset,
                "split": split,
                "model_name": args.model,
                "base_url": args.base_url,
                "log_dir": args.log_dir,
                "recursion_limit": args.recursion_limit,
                "max_rows": args.max_rows,
            }

            started = time.perf_counter()
            outcome: Optional[dict] = None
            error: Optional[str] = None
            attempts = 0

            while attempts <= max_retries:
                attempts += 1
                timed_out = False
                try:
                    async_result = pool.apply_async(_run_one, (payload,))
                    outcome = async_result.get(
                        timeout=args.timeout if args.timeout > 0 else None
                    )
                    error = outcome.get("error")
                    if error is None and outcome.get("sql"):
                        break
                    error = error or "未产出 SQL"
                except mp.TimeoutError:
                    timed_out = True
                    error = f"单题运行超过 {args.timeout:.0f}s，已终止"
                    outcome = None
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    outcome = None

                if timed_out or outcome is None:
                    # worker 已被终止或状态不可信，重建进程池
                    pool.terminate()
                    pool.join()
                    pool = ctx.Pool(processes=1)

                if attempts <= max_retries:
                    print(f"    qid={question_id} 第 {attempts} 次尝试失败（{error}），"
                          f"{args.retry_backoff:.0f}s 后重试")
                    time.sleep(args.retry_backoff)

            elapsed = time.perf_counter() - started
            sql = (outcome or {}).get("sql", "")
            used_db_id = (outcome or {}).get("db_id") or gt_db_id
            log_path = (outcome or {}).get("log_path") or ""
            results[question_id] = {
                "question_id": question_id,
                "db_id": used_db_id,
                "SQL": sql,
            }
            write_preds(output_path, list(results.values()))

            log_name = Path(log_path).name if log_path else "-"
            if error is None and sql:
                success += 1
                print(f"[{index + 1}/{len(questions)}] qid={question_id} db={used_db_id} "
                      f"status=ok attempts={attempts} elapsed={elapsed:.1f}s log={log_name}")
            else:
                failed += 1
                print(f"[{index + 1}/{len(questions)}] qid={question_id} db={used_db_id} "
                      f"status=failed attempts={attempts} elapsed={elapsed:.1f}s "
                      f"log={log_name} reason={error or '未产出 SQL'}")
    finally:
        pool.terminate()
        pool.join()

    print(f"\n完成：输出 {len(results)} 条（本次成功 {success}，失败 {failed}，跳过 {skipped}）")
    print(f"preds.json: {output_path}")

    if args.no_eval:
        print(">>> 已跳过评测（--no-eval）")
        return 0

    report_path = Path(args.report) if args.report else Path("outputs") / "reports" / f"eval_{dataset}_{split}.md"
    print(f"\n>>> 开始评测（VES 迭代 {args.ves_iterations} 次）")
    try:
        eval_result = evaluate_preds(
            output_path,
            dataset=dataset,
            split=split,
            ves_iterations=args.ves_iterations,
            timeout=args.eval_timeout,
            workers=args.eval_workers,
        )
    except Exception as exc:
        print(f"评测失败: {type(exc).__name__}: {exc}")
        return 1

    report_file, details_file = write_report(eval_result, report_path)
    print("\n" + format_summary(eval_result))
    print(f"\n评测报告: {report_file}")
    print(f"评测明细: {details_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
