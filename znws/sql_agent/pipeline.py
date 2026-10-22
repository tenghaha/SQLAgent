"""编排层：把 QueryProcessor、DataLoader 与 Worker 串成一次独立运行，并产出可复现的结果对象。"""
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

from znws.utils.dataset import (
    active_database_path,
    default_dataset_name,
    default_split,
    ensure_database,
    set_active_database,
)

from .config import DEFAULT_RECURSION_LIMIT, build_model
from .dataloader import SchemaPlan, run_data_loader
from .query_processor import QueryProcessResult, run_query_processor
from .tracing import DEFAULT_LOG_DIR, RunRecorder
from .worker import SqlAnswer, run_worker

DEFAULT_MAX_ROWS = 50


@dataclass
class SQLAgentResult:
    """一次独立运行的完整结果，供后续研究分析直接消费。"""

    run_id: str
    query: str
    dataset: Optional[str]
    db_id: Optional[str]
    split: Optional[str]
    log_path: str
    model: str
    schema_plan: Optional[dict]
    final_sql: Optional[str]
    explanation: Optional[str]
    execution: Optional[dict]
    query_processing: Optional[dict] = None
    steps: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def execute_sql_preview(sql: str, max_rows: int = DEFAULT_MAX_ROWS, db_path=None) -> dict:
    """独立复算最终 SQL，产出确定性的执行结果（不依赖模型自述）。

    db_path 缺省时使用当前活动数据库（BIRD 的 <db_id>.sqlite）。
    """
    try:
        conn = sqlite3.connect(str(db_path or active_database_path()))
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            fetched = cursor.fetchmany(max_rows + 1)
        finally:
            conn.close()
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    truncated = len(fetched) > max_rows
    rows = [[_json_safe(value) for value in row] for row in fetched[:max_rows]]
    return {
        "status": "success",
        "columns": columns,
        "rows": rows,
        "returned_rows": len(rows),
        "truncated": truncated,
        "error": None,
    }


def run_sql_agent(
    query: str,
    *,
    db_id: str,
    question_id: Optional[int] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    echo: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> SQLAgentResult:
    """执行一次完整的 Text-to-SQL 运行（针对指定的 db_id）。

    question_id 用于让 QueryProcessor 通过 load_bird_evidence 读取外部知识；
    不传时跳过 evidence，仅使用问题文本。dataset 缺省为 datasets.yaml 中的
    默认数据集；split 缺省为该数据集的默认切分。
    每次调用都是独立运行：不依赖任何历史状态，也不接收多轮反馈。
    """
    load_dotenv()
    dataset = dataset or default_dataset_name()
    split = split or default_split(dataset)

    set_active_database(db_id, split=split, dataset=dataset)
    ensure_database()

    model = build_model(model_name=model_name, base_url=base_url)
    model_label = getattr(model, "model_name", "") or (model_name or "")

    recorder = RunRecorder(query=query, log_dir=log_dir, echo=echo, model=model_label)

    plan: Optional[SchemaPlan] = None
    answer: Optional[SqlAnswer] = None
    processed: Optional[QueryProcessResult] = None
    execution: Optional[dict] = None
    error: Optional[str] = None

    try:
        processed = run_query_processor(
            query=query,
            model=model,
            recorder=recorder,
            question_id=question_id,
            dataset=dataset,
            split=split,
        )
        plan, _ = run_data_loader(
            query=processed.rewritten_query,
            entities=processed.entities,
            sub_queries=processed.sub_queries,
            evidence=processed.evidence,
            model=model,
            recorder=recorder,
            recursion_limit=recursion_limit,
        )
        answer, _ = run_worker(
            query=query,
            plan=plan,
            model=model,
            recorder=recorder,
            evidence=processed.evidence,
            recursion_limit=recursion_limit,
        )
        if answer and answer.sql.strip():
            execution = execute_sql_preview(answer.sql, max_rows=max_rows)
            recorder.note(
                "Pipeline",
                f"最终 SQL 复算结果: status={execution.get('status')}, "
                f"returned_rows={execution.get('returned_rows')}",
            )
        else:
            recorder.note("Pipeline", "Worker 未产出可执行的 SQL")
    except Exception as exc:  # 保留已记录的过程，同时把异常交给调用方
        error = f"{type(exc).__name__}: {exc}"
        recorder.note("Pipeline", f"运行异常: {error}")

    recorder.finish({
        "数据集": dataset,
        "数据库": f"{db_id} ({split})",
        "模型": model_label,
        "Query 处理": processed.to_dict() if processed else None,
        "Schema 方案": plan.model_dump() if plan else None,
        "最终 SQL": answer.sql if answer else None,
        "执行结果": execution,
        "错误": error,
        "日志文件": str(recorder.log_path),
    })

    return SQLAgentResult(
        run_id=recorder.run_id,
        query=query,
        dataset=dataset,
        db_id=db_id,
        split=split,
        log_path=str(recorder.log_path),
        model=model_label,
        schema_plan=plan.model_dump() if plan else None,
        final_sql=answer.sql if answer else None,
        explanation=answer.explanation if answer else None,
        execution=execution,
        query_processing=processed.to_dict() if processed else None,
        steps=recorder.steps,
        elapsed_seconds=recorder.elapsed_seconds,
        error=error,
    )
