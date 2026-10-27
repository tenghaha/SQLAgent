"""FinalAlign：在 Worker 结束之后，用一次独立 LLM 调用审查最终 SQL 是否包含冗余列。

该调用不在 Worker 的 ReAct 运行范围内，只执行一次：
1. 输入原始问题与 Worker 产出的最终 SQL；
2. 询问模型该 SQL 是否会产生与问题无关或未显式提及的冗余列；
3. 模型给出对齐后的 SQL，由编排层执行验证；
4. 验证失败时由编排层回退到调用前的 SQL 版本。
"""
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .tracing import RunRecorder

AGENT_NAME = "FinalAlign"


class FinalAlignedSql(BaseModel):
    """最终对齐调用的结构化产出。"""

    has_redundant_columns: bool = Field(
        description="最终 SQL 是否会产生与问题无关或问题没有显式提及的冗余列"
    )
    sql: str = Field(description="对齐后的 SQL；若不存在冗余列则与原 SQL 完全一致")
    explanation: str = Field(description="判断依据，以及删除了哪些冗余列")


SYSTEM_PROMPT = """你是 SQL 结果审校员。你会收到一个业务问题与一条已通过执行验证的最终 SQL，请回答并处理：
最终 SQL 是否会生成与问题无关或没有显式提及的冗余列？生成的 SQL 应当严格与问题意图对应。

【审校要求】
1. 只判断与处理冗余列：问题没有要求输出的列不应出现在 SELECT 结果中；仅用于过滤、排序、连接的列不必输出。
2. 若不存在冗余列，保持原 SQL 完全不变。
3. 若存在冗余列，只删除这些多余列；不得改变筛选条件、聚合口径、分组、排序、LIMIT 与结果行数。
4. 输出必须是可直接执行的 SQLite 查询语句，禁止使用原文中不存在的表或列。
5. 必须调用结构化输出工具（FinalAlignedSql）提交结论：
   - has_redundant_columns：是否存在冗余列
   - sql：对齐后的 SQL
   - explanation：判断依据与改动说明"""


def run_final_align(
    *,
    query: str,
    sql: str,
    model: BaseChatModel,
    recorder: RunRecorder,
) -> Optional[FinalAlignedSql]:
    """执行一次最终对齐调用，返回结构化结果；调用失败时返回 None。"""
    recorder.agent_start(AGENT_NAME, "审查最终 SQL 是否包含与问题无关的冗余列")

    try:
        chain = model.with_structured_output(
            FinalAlignedSql, include_raw=True, method="function_calling"
        )
        result = chain.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"【业务问题】\n{query}\n\n【最终 SQL】\n{sql}"),
        ])
    except Exception as exc:
        recorder.note(AGENT_NAME, f"最终对齐调用失败: {type(exc).__name__}: {exc}")
        recorder.agent_end(AGENT_NAME, "调用失败，沿用对齐前的 SQL")
        return None

    if isinstance(result, dict) and "raw" in result:
        raw = result.get("raw")
        if raw is not None:
            recorder.record_messages(AGENT_NAME, [raw])
        parsing_error = result.get("parsing_error")
        if parsing_error:
            recorder.note(AGENT_NAME, f"结构化输出解析失败: {parsing_error}")
        parsed = result.get("parsed")
    else:
        parsed = result

    aligned = parsed if isinstance(parsed, FinalAlignedSql) else None
    if aligned is None:
        recorder.agent_end(AGENT_NAME, "未获得结构化结果，沿用对齐前的 SQL")
        return None

    summary = (
        f"存在冗余列: {aligned.has_redundant_columns}\n"
        f"对齐后 SQL:\n{aligned.sql}\n"
        f"说明: {aligned.explanation}"
    )
    recorder.agent_end(AGENT_NAME, summary)
    return aligned
