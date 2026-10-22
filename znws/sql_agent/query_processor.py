"""QueryProcessor：Schema 链接之前的 NLU 前处理节点。

固定流程：
1. 通过 ``load_bird_evidence`` 读取题目对应的 BIRD 外部知识（evidence）；
2. 第一次 LLM 调用抽取实体并标注类别（标准 NER 类别 + “统计指标”）；
3. 第二次 LLM 调用结合实体与外部知识改写问题，并做多步子句分解；
4. 把改写结果与 evidence 一起交付 DataLoader 与 Worker 参考。

为便于后续扩展，各步骤封装为独立函数，统一通过 QueryProcessResult 交付结果；
以后可在 run_query_processor 中插入更多 NLU 步骤，或把本模块直接包装成可
单独运行的智能体。
"""
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from znws.agent.tools.evidence_tools import load_bird_evidence

from .tracing import RunRecorder

AGENT_NAME = "QueryProcessor"

EntityCategory = Literal["时间", "地点", "人物", "组织机构", "数值", "统计指标", "其他"]

CATEGORY_GUIDE = """- 时间：年份、日期、时间段、相对时间（如“2020年”“上个月”“近三年”）
- 地点：国家、城市、省份、地区、具体地址
- 人物：人名
- 组织机构：公司、学校、部门、政府机构等
- 数值：问题中出现的具体数量、金额、百分比或阈值（如“超过 500 人”“100 万元”）
- 统计指标：需要被统计或聚合的业务度量名称（不是具体数值），如“数量总计”“销量”“销售额”“金额”“平均成绩”“最高分”“占比”“人数”
- 其他：产品、赛事、专业、类别等无法归入上述类别的业务实体"""


class QueryEntity(BaseModel):
    """一个自然语言实体及其类别。"""

    text: str = Field(description="实体在原始问题中的原文片段，必须逐字一致")
    category: EntityCategory = Field(description="实体类别")


class EntityExtraction(BaseModel):
    """第一次 LLM 调用的结构化产出。"""

    entities: list[QueryEntity] = Field(
        default_factory=list,
        description="按原文出现顺序排列的实体列表；没有实体时为空列表",
    )


class QueryRewrite(BaseModel):
    """第二次 LLM 调用的结构化产出。"""

    rewritten_query: str = Field(description="改写后的正式查询描述，使用与原始问题相同的语言")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="多步查询的子句分解，按执行依赖排序；单步查询为空列表",
    )
    notes: str = Field(default="", description="改写中的关键口径澄清或省略说明")


@dataclass
class QueryProcessResult:
    """QueryProcessor 交付给 DataLoader / Worker 的完整 NLU 结果。"""

    original_query: str
    rewritten_query: str
    entities: list[dict] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    notes: str = ""
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


ENTITY_SYSTEM_PROMPT = f"""你是 Text-to-SQL 流程中的自然语言实体抽取器。你的任务是从用户问题中抽取所有对数据库查询有意义的实体，并标注类别。

【实体类别】
{CATEGORY_GUIDE}

【抽取要求】
1. text 必须是原文中真实出现的片段，逐字一致，不得改写、翻译或补全。
2. 按实体在原文中出现的顺序输出；同一实体只输出一次。
3. “统计指标”只标记业务度量名称本身，不要把限定条件中的具体数值标成统计指标。
4. 不要抽取 SQL 关键字、表名或列名；不要臆造原文中不存在的实体。
5. 没有可用实体时返回空列表。
6. 如果用户输入包含【外部知识】，它只用于理解术语与业务口径；实体仍必须从原始问题中抽取，不得把外部知识中的词当作实体。
7. 必须调用结构化输出工具（EntityExtraction）提交结果，不要只输出自然语言。"""


REWRITE_SYSTEM_PROMPT = """你是 Text-to-SQL 流程中的查询规范化专家。你会收到原始用户问题、已抽取的实体清单，以及可能为空的【外部知识】，需要把问题改写成逻辑清晰、步骤合理的正式查询描述，供后续 Schema 链接与 SQL 生成使用。

【改写要求】
1. 保留原问题的全部语义与限定条件（时间、地点、组织、数值阈值、排序、Top-N、分组、比较等），不得遗漏，也不得新增事实。
2. 将口语化、模糊、省略的表达改写为正式、明确的查询描述；只做语义澄清，不生成 SQL，不猜测表名或列名。
3. 实体名称必须原样保留（尤其是筛选值），不要在改写中翻译或替换。
4. 涉及多步查询时（如先筛选再聚合、先分组再排序、子查询/嵌套查询、多个统计口径），拆分为有序的子查询 sub_queries；每个子查询应完整表达一步，并保持原有依赖顺序。单步问题返回空列表。
5. rewritten_query 应是一个完整、连贯的正式问题描述，并与 sub_queries 的语义一致。
6. 输出语言必须与原始问题一致（原文为英文则用英文改写）。
7. 如果提供【外部知识】，用它澄清业务定义、计算口径或单位；不得引入与问题无关的新条件，也不要把外部知识改写成新的筛选条件。
8. 必须调用结构化输出工具（QueryRewrite）提交结果：
   - rewritten_query：改写后的正式查询描述
   - sub_queries：多步子句分解
   - notes：简要说明关键口径澄清（可选）"""


def _invoke_structured(
    *,
    model: BaseChatModel,
    schema: type[BaseModel],
    system_prompt: str,
    user_input: str,
    recorder: RunRecorder,
) -> Optional[BaseModel]:
    """执行一次带结构化输出的 LLM 调用，并把原始消息写入运行记录。"""
    chain = model.with_structured_output(
        schema, include_raw=True, method="function_calling"
    )
    result = chain.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])

    if isinstance(result, dict) and "raw" in result:
        raw = result.get("raw")
        if raw is not None:
            recorder.record_messages(AGENT_NAME, [raw])
        parsing_error = result.get("parsing_error")
        if parsing_error:
            recorder.note(AGENT_NAME, f"结构化输出解析失败: {parsing_error}")
        return result.get("parsed")

    return result


def _evidence_block(evidence: str) -> str:
    evidence = (evidence or "").strip()
    if not evidence:
        return ""
    return f"\n\n【外部知识】\n{evidence}"


def extract_entities(
    *, query: str, model: BaseChatModel, recorder: RunRecorder, evidence: str = ""
) -> Optional[EntityExtraction]:
    """步骤 1：抽取实体并标注类别。"""
    recorder.note(AGENT_NAME, "步骤 1/2：抽取自然语言实体并标注类别")
    parsed = _invoke_structured(
        model=model,
        schema=EntityExtraction,
        system_prompt=ENTITY_SYSTEM_PROMPT,
        user_input=f"【用户问题】\n{query}{_evidence_block(evidence)}",
        recorder=recorder,
    )
    return parsed if isinstance(parsed, EntityExtraction) else None


def rewrite_query(
    *,
    query: str,
    entities: list[QueryEntity],
    model: BaseChatModel,
    recorder: RunRecorder,
    evidence: str = "",
) -> Optional[QueryRewrite]:
    """步骤 2：结合实体清单与外部知识改写问题并分解多步子句。"""
    recorder.note(AGENT_NAME, "步骤 2/2：将问题改写为正式查询并分解子句")
    entity_block = "\n".join(
        f"- {entity.text}（{entity.category}）" for entity in entities
    ) or "（未识别出实体）"
    user_input = (
        f"【原始问题】\n{query}\n\n"
        f"【实体清单】\n{entity_block}"
        f"{_evidence_block(evidence)}"
    )
    parsed = _invoke_structured(
        model=model,
        schema=QueryRewrite,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_input=user_input,
        recorder=recorder,
    )
    return parsed if isinstance(parsed, QueryRewrite) else None


def run_query_processor(
    *,
    query: str,
    model: BaseChatModel,
    recorder: RunRecorder,
    question_id: Optional[int] = None,
    evidence: Optional[str] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
) -> QueryProcessResult:
    """运行 QueryProcessor，返回供 DataLoader / Worker 消费的规范化结果。

    evidence 缺省时通过 load_bird_evidence 按 question_id 读取；
    也可由调用方直接传入 evidence 文本，便于独立运行与测试。
    """
    recorder.agent_start(
        AGENT_NAME, "读取外部知识、实体抽取与查询规范化"
    )

    if evidence is None and question_id is not None:
        try:
            evidence = load_bird_evidence.invoke({
                "question_id": question_id,
                "dataset": dataset,
                "split": split,
            })
        except Exception as exc:
            recorder.note(
                AGENT_NAME, f"读取 BIRD evidence 失败: {type(exc).__name__}: {exc}"
            )
            evidence = ""
    evidence = (evidence or "").strip()
    recorder.note(
        AGENT_NAME,
        f"外部知识: {'已加载' if evidence else '无'}（{len(evidence)} 字符）",
    )

    extraction = extract_entities(
        query=query, model=model, recorder=recorder, evidence=evidence
    )
    entities = list(extraction.entities) if extraction else []
    if extraction is None:
        recorder.note(AGENT_NAME, "实体抽取未获得结构化结果，按空实体清单继续")

    rewrite = rewrite_query(
        query=query,
        entities=entities,
        model=model,
        recorder=recorder,
        evidence=evidence,
    )

    if rewrite is None:
        recorder.note(AGENT_NAME, "查询改写未获得结构化结果，回退为原始问题")
        result = QueryProcessResult(
            original_query=query,
            rewritten_query=query,
            entities=[entity.model_dump() for entity in entities],
            evidence=evidence,
        )
    else:
        rewritten = (rewrite.rewritten_query or "").strip() or query
        result = QueryProcessResult(
            original_query=query,
            rewritten_query=rewritten,
            entities=[entity.model_dump() for entity in entities],
            sub_queries=[
                item.strip() for item in rewrite.sub_queries if item and item.strip()
            ],
            notes=(rewrite.notes or "").strip(),
            evidence=evidence,
        )

    summary_lines = ["实体清单:"]
    if result.entities:
        summary_lines.extend(
            f"- {entity['text']}（{entity['category']}）" for entity in result.entities
        )
    else:
        summary_lines.append("- 无")
    summary_lines += ["", f"改写后 Query:\n{result.rewritten_query}"]
    if result.sub_queries:
        summary_lines += ["", "子句分解:"]
        summary_lines.extend(
            f"{index}. {item}" for index, item in enumerate(result.sub_queries, 1)
        )
    if result.notes:
        summary_lines += ["", f"备注: {result.notes}"]
    if result.evidence:
        summary_lines += ["", f"外部知识:\n{result.evidence}"]

    recorder.agent_end(AGENT_NAME, "\n".join(summary_lines))
    return result
