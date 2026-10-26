"""DataLoader Agent：自主探查数据库，确定回答用户问题所需的表与字段。"""
from typing import Optional, Tuple

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from znws.utils.dataset import db_relationships, get_active_database, get_active_dataset

from .config import load_db_config
from .runner import run_react_agent
from .tools import DATA_LOADER_TOOLS
from .tracing import RunRecorder

AGENT_NAME = "DataLoader"


class SchemaPlan(BaseModel):
    """DataLoader 的结构化产出：回答该问题需要用到哪些表与字段。"""

    tables: list[str] = Field(description="需要使用的物理表名，可包含多张表")
    fields: list[str] = Field(description="需要用到的全部字段，格式为 '表名.字段名'，包含筛选字段与分组维度")
    statistics_fields: list[str] = Field(description="参与聚合统计的字段，格式为 '表名.字段名'")
    reasoning: str = Field(description="选择这些表和字段的理由，以及业务口径的判断依据")
    relationships: list[str] = Field(
        default_factory=list,
        description="选定表之间的外键关系，由系统根据元数据自动注入，模型无需填写",
    )


SYSTEM_PROMPT_TEMPLATE = """你是数据探查与 Schema 定位专家。用户会给出一个业务问题，你要通过调用工具自主探查数据库，最终确定回答该问题所需的表和字段。

【数据库与系统全局配置】
{db_config}

【可用工具】
- get_table_directory：列出数据库中有哪些表及其业务对象
- get_table_schema：查看指定表的建表语句
- execute_sql：执行只读 SQL 做数据探查（确认字段是否存在、取值是否符合业务含义）

【输入说明】
- 用户问题已由 QueryProcessor 改写为规范化描述，可能包含多步子句分解；
- 实体清单来自原始问题，包含时间、地点、组织机构、数值以及“统计指标”，是定位筛选字段与聚合字段的重要线索；
- 【外部知识】是 BIRD 提供的业务口径提示（如计算公式、术语说明），只用于澄清语义，不能替代真实 schema。

【工作要求】
1. 先摸清数据库有哪些表，再逐表核对字段，缩小到候选表和候选字段。
2. 允许并鼓励选择多张表：只要问题涉及多个实体或维度，就应把所有必要的表都选出来，不要人为限制为单表。
3. 对确定的每一张表，逐一核对其字段含义是否与原始问题中的表述对应（例如“学校”“成绩”分别落在哪个表的哪个字段），确认字段确实能承载问题语义。
4. 关注业务口径：同一含义可能存在多个字段（例如“公司 / 销售组织 / 会计主体 / 客户”都含公司意味），必须结合真实数据确认哪一个才是用户指的对象。必要时用 execute_sql 查看字段的实际取值来佐证。
5. 用户问题中的每一个限定条件（时间、公司、产品、区域等）都必须落到具体字段上，不允许遗漏筛选字段。
6. 只允许使用真实存在的表和字段，严禁编造；字段一律写成 “表名.字段名”。
7. 本阶段只负责确认“涉及哪些表与字段”，不要思考、也不要给出 JOIN 条件或连接方式，也不要在 reasoning 中展开表间连接；表间关联由系统在后续自动注入。
8. 实体清单中的每一项都必须落到具体字段：时间、地点、组织机构、数值对应筛选字段；“统计指标”对应参与聚合的字段。
9. 若给出【子句分解】，需要覆盖每个子句所需的表和字段，保证多步查询的中间结果与最终统计都能被承载。
10. 若提供【外部知识】，用它确认业务术语和统计口径对应的字段；不得据此引入与问题无关的新条件。

【输出】
探查完成后，必须调用结构化输出工具（SchemaPlan）提交结论，不要只用自然语言作答：
- tables：需要使用的物理表（可多张）
- fields：需要用到的全部字段（含筛选字段与分组维度）
- statistics_fields：参与聚合统计的字段（如销量、金额、数量）
- reasoning：选择理由与口径判断依据，需说明关键字段是如何确认的
（relationships 由系统自动注入，无需填写）
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
    entities: Optional[list[dict]] = None,
    sub_queries: Optional[list[str]] = None,
    evidence: Optional[str] = None,
) -> str:
    """组装 DataLoader 输入：规范化 Query、实体清单、子句分解与外部知识。"""
    parts = [
        f"【用户问题（已规范化）】\n{query}",
        f"【实体清单】\n{_format_entities(entities)}",
    ]
    if sub_queries:
        parts.append(
            "【子句分解】\n"
            + "\n".join(f"{index}. {item}" for index, item in enumerate(sub_queries, 1))
        )
    evidence = (evidence or "").strip()
    if evidence:
        parts.append(f"【外部知识】\n{evidence}")
    parts.append("请基于以上信息探查数据库，确定回答该问题所需的表与字段。")
    return "\n\n".join(parts)


def run_data_loader(
    *,
    query: str,
    model: BaseChatModel,
    recorder: RunRecorder,
    entities: Optional[list[dict]] = None,
    sub_queries: Optional[list[str]] = None,
    evidence: Optional[str] = None,
    recursion_limit: int = 30,
) -> Tuple[Optional[SchemaPlan], str]:
    """运行 DataLoader agent，返回 (SchemaPlan, 最终文本)。"""
    recorder.agent_start(AGENT_NAME, "探查数据库并确定所需的表与字段")
    recorder.note(
        AGENT_NAME,
        f"接收 QueryProcessor 结果：{len(entities or [])} 个实体，"
        f"{len(sub_queries or [])} 个子句，外部知识 {len((evidence or '').strip())} 字符",
    )

    structured, final_text, _ = run_react_agent(
        agent_name=AGENT_NAME,
        model=model,
        tools=DATA_LOADER_TOOLS,
        system_prompt=build_system_prompt(),
        user_input=build_user_input(query, entities, sub_queries, evidence),
        recorder=recorder,
        response_format=SchemaPlan,
        recursion_limit=recursion_limit,
    )

    plan = structured if isinstance(structured, SchemaPlan) else None
    if plan:
        # 表间关联由系统注入，不交给模型推断
        db_id, split = get_active_database()
        if db_id:
            relations = db_relationships(
                db_id, tables=plan.tables, split=split, dataset=get_active_dataset(),
            )
            plan.relationships = [
                f"{rel['child_table']}.{rel['child_field']} -> "
                f"{rel['parent_table']}.{rel['parent_field']}"
                for rel in relations
            ]
        summary = (
            f"表: {plan.tables}\n"
            f"字段: {plan.fields}\n"
            f"统计字段: {plan.statistics_fields}\n"
            f"外键关系: {plan.relationships or '无'}\n"
            f"理由: {plan.reasoning}"
        )
    else:
        summary = f"未获得结构化结果，模型最终输出:\n{final_text}"
    recorder.agent_end(AGENT_NAME, summary)
    return plan, final_text
