"""Worker Agent：生成 SQL，并通过工具实际执行验证，反复修正直到正确。"""
import re
from typing import Optional, Tuple

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .config import load_db_config
from .dataloader import SchemaPlan
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


SYSTEM_PROMPT_TEMPLATE = """你是资深的 SQLite 查询工程师。你会收到用户的业务问题，以及已经确定好的表和字段，请生成可直接执行的 SQL。

【数据库与系统全局配置】
{db_config}

【可用工具】
- execute_sql：执行 SQL 并返回结果或错误信息，用于验证你写的语句
- get_table_schema：查看表结构，用于核对列名

【工作要求】
1. 只能使用给定范围内的、真实存在的列，禁止编造列名；确需核对时用 get_table_schema 查看。
2. 必须实际执行验证，形成“生成 → 执行 → 根据反馈修正 → 再执行”的闭环：
   - 若执行报错，阅读错误信息定位原因（列名、语法、类型等），修正后重新执行；
   - 若执行成功但结果为空或明显异常，检查过滤条件与统计口径是否正确，必要时修正后重试。
3. 只有在 SQL 成功执行、且结果符合业务预期之后，才输出最终答案。
4. 用户问题中的所有限定条件（时间、公司、产品等）都必须体现在 WHERE 中，不得遗漏。
5. 需要 Top-N 时使用 ORDER BY + LIMIT，并按业务含义选择聚合方式。
6. 涉及多张表时，必须依据【表间外键关系】确定 JOIN 条件（左为外键列、右为被引用列），不要凭列名相似性猜测；关系缺失时先用 get_table_schema 核对再连接。
7. 若提供【外部知识】，用它确认统计口径、计算公式或单位；但仍只能使用给定范围内的真实表和字段，不得把外部知识中的文字直接当作 SQL 或新条件。

【输出】
验证通过后，必须调用结构化输出工具（SqlAnswer）提交最终结果，不要只用自然语言作答：
- sql：与验证成功的那次执行完全一致的 SQL 语句
- explanation：查询逻辑与口径说明
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(db_config=load_db_config())


def build_user_input(
    query: str, plan: Optional[SchemaPlan], evidence: Optional[str] = None
) -> str:
    """把用户问题、Schema 方案与外部知识一起交给 Worker。"""
    evidence_block = ""
    if (evidence or "").strip():
        evidence_block = f"\n\n【外部知识】\n{evidence.strip()}"
    if not plan:
        return (
            f"【用户问题】\n{query}{evidence_block}\n\n"
            "【说明】未获得确定的 Schema 方案，请先用工具自行了解表结构再生成 SQL。"
        )
    relationships = "\n".join(plan.relationships) if plan.relationships else "无（单表查询或未发现外键关系）"
    return (
        f"【用户问题】\n{query}\n\n"
        f"【已确定的表】\n{', '.join(plan.tables)}\n\n"
        f"【可用字段】\n{', '.join(plan.fields)}\n\n"
        f"【统计字段】\n{', '.join(plan.statistics_fields) if plan.statistics_fields else '无'}\n\n"
        f"【表间外键关系】（左为外键列，右为被引用列）\n{relationships}\n\n"
        f"【Schema 定位理由】\n{plan.reasoning}"
        f"{evidence_block}\n\n"
        "请先生成 SQL，再用 execute_sql 验证；若报错或结果异常，修正后重试，直到验证通过，最后输出最终 SQL。"
        "涉及多表时，必须依据【表间外键关系】确定 JOIN 条件，不要凭列名相似性猜测；"
        "若所需关系未给出，先用 get_table_schema 核对后再连接。"
    )


def run_worker(
    *,
    query: str,
    plan: Optional[SchemaPlan],
    model: BaseChatModel,
    recorder: RunRecorder,
    evidence: Optional[str] = None,
    recursion_limit: int = 30,
) -> Tuple[Optional[SqlAnswer], str]:
    """运行 Worker agent，返回 (SqlAnswer, 最终文本)。"""
    recorder.agent_start(AGENT_NAME, "生成 SQL 并用工具执行验证，修正至正确")
    recorder.note(AGENT_NAME, f"外部知识 {len((evidence or '').strip())} 字符")

    structured, final_text, _ = run_react_agent(
        agent_name=AGENT_NAME,
        model=model,
        tools=WORKER_TOOLS,
        system_prompt=build_system_prompt(),
        user_input=build_user_input(query, plan, evidence),
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
