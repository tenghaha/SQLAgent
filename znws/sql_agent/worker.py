"""Worker Agent：在单一智能体内完成数据库探查、Schema 定位、SQL 生成与执行验证。"""
import re
from typing import Optional, Tuple

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .config import DEFAULT_RECURSION_LIMIT, load_db_config
from .runner import run_react_agent
from .tools import WORKER_TOOLS
from .tracing import RunRecorder

AGENT_NAME = "Worker"


class SqlAnswer(BaseModel):
    """Worker 的结构化产出：最终可直接执行的 SQL 及其说明。"""

    sql: str = Field(description="最终可直接执行的 SQLite 查询语句，必须是通过工具验证成功的那一条")
    explanation: str = Field(description="该 SQL 的查询逻辑与统计口径说明")


# 回退解析：兼容 ```sql 代码块，以及 “sql: xxx / **sql**: `xxx`” 两种常见写法
_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_FIELD_RE = re.compile(r"(?im)^\s*[-*]?\s*\**\s*sql\s*\**\s*[:：]\s*(.+?)\s*$")
_EXPLANATION_FIELD_RE = re.compile(r"(?im)^\s*[-*]?\s*\**\s*explanation\s*\**\s*[:：]\s*(.+?)\s*$")


def _extract_sql_answer(text: str) -> Optional[SqlAnswer]:
    """结构化输出缺失时，从最终自然语言输出中回退解析 SQL。"""
    if not text:
        return None

    sql = ""
    match = _SQL_FENCE_RE.search(text)
    if match:
        sql = match.group(1).strip()
    if not sql:
        match = _SQL_FIELD_RE.search(text)
        if match:
            sql = match.group(1).strip().strip("`").strip()
    if not sql:
        return None

    explanation = ""
    match = _EXPLANATION_FIELD_RE.search(text)
    if match:
        explanation = match.group(1).strip()
    return SqlAnswer(sql=sql, explanation=explanation)


SYSTEM_PROMPT_TEMPLATE = """你是资深的 SQLite 数据分析智能体。你需要独立完成从理解业务问题到产出可用 SQL 的全过程：先自主探查数据库、定位所需的表与字段，再编写 SQL 并通过实际执行验证，必要时反复修正。

【数据库与系统全局配置】
{db_config}

【可用工具】
- get_table_directory：列出数据库中有哪些表及其业务对象
- search_schema_by_field：按业务术语（如“销量”“产品”）检索 schema 抽样库
- search_schema_by_value：按具体实体值（如公司名、产品名）检索 schema 抽样库
- get_table_schema：查看指定表的建表语句（含外键与 BIRD 列说明）
- execute_sql：执行只读 SQL 做数据探查或验证，返回结果或错误信息

【工作要求】
1. 只使用真实存在的表和字段，严禁编造；不确定时先用工具确认。
2. 问题中的每个限定条件（时间、公司、产品、区域等）都要落到具体字段上，统计指标要选对聚合方式。
3. 同一含义可能存在多个近似字段（例如“公司 / 销售组织 / 客户”都含公司意味），结合字段说明与实际取值判断哪一个才是问题所指。
4. 涉及多张表时，依据建表语句中的外键确定 JOIN 条件，不要凭列名相似性猜测。
5. 必须实际执行验证：报错时阅读错误信息定位原因并修正；执行成功但结果为空或不符合业务预期时，重新检查筛选条件与统计口径。
6. 【外部知识】只用于澄清业务术语与统计口径，不得直接当作表名、列名或新的筛选条件。

【输出】
验证通过后，必须调用结构化输出工具（SqlAnswer）提交最终结果，不要只用自然语言作答：
- sql：与验证成功的那次执行完全一致的 SQL
- explanation：查询逻辑与口径说明
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(db_config=load_db_config())


def _format_entities(entities: Optional[list[dict]]) -> str:
    if not entities:
        return "（未识别出实体）"
    return "\n".join(
        f"- {item.get('text', '')}（{item.get('category', '其他')}）"
        for item in entities
    )


def build_user_input(
    query: str,
    rewritten_query: Optional[str] = None,
    entities: Optional[list[dict]] = None,
    sub_queries: Optional[list[str]] = None,
    evidence: Optional[str] = None,
) -> str:
    """组装 Worker 输入：原始问题、规范化描述、实体清单、子句分解与外部知识。"""
    parts = [f"【用户问题】\n{query}"]
    rewritten = (rewritten_query or "").strip()
    if rewritten and rewritten != query.strip():
        parts.append(f"【规范化描述】\n{rewritten}")
    parts.append(f"【实体清单】\n{_format_entities(entities)}")
    if sub_queries:
        parts.append(
            "【子句分解】\n"
            + "\n".join(f"{index}. {item}" for index, item in enumerate(sub_queries, 1))
        )
    evidence = (evidence or "").strip()
    if evidence:
        parts.append(f"【外部知识】\n{evidence}")
    parts.append(
        "请先探查数据库确认所需的表与字段，再生成 SQL 并用 execute_sql 验证；"
        "结果符合业务预期后提交最终答案。"
    )
    return "\n\n".join(parts)


def run_worker(
    *,
    query: str,
    model: BaseChatModel,
    recorder: RunRecorder,
    rewritten_query: Optional[str] = None,
    entities: Optional[list[dict]] = None,
    sub_queries: Optional[list[str]] = None,
    evidence: Optional[str] = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> Tuple[Optional[SqlAnswer], str]:
    """运行 Worker agent，返回 (SqlAnswer, 最终文本)。"""
    recorder.agent_start(AGENT_NAME, "探查数据库、定位表与字段、生成并验证 SQL")
    recorder.note(
        AGENT_NAME,
        f"接收 QueryProcessor 结果：{len(entities or [])} 个实体，"
        f"{len(sub_queries or [])} 个子句，外部知识 {len((evidence or '').strip())} 字符",
    )

    structured, final_text, _ = run_react_agent(
        agent_name=AGENT_NAME,
        model=model,
        tools=WORKER_TOOLS,
        system_prompt=build_system_prompt(),
        user_input=build_user_input(
            query, rewritten_query, entities, sub_queries, evidence
        ),
        recorder=recorder,
        response_format=SqlAnswer,
        recursion_limit=recursion_limit,
    )

    answer = structured if isinstance(structured, SqlAnswer) else None
    if answer is None and final_text:
        answer = _extract_sql_answer(final_text)
        if answer is not None:
            recorder.note(AGENT_NAME, "未获得结构化输出，已从最终文本回退解析出 SQL")

    if answer:
        summary = f"最终 SQL:\n{answer.sql}\n\n说明: {answer.explanation}"
    else:
        summary = f"未获得结构化结果，模型最终输出:\n{final_text}"
    recorder.agent_end(AGENT_NAME, summary)
    return answer, final_text
