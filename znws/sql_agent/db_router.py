"""数据库路由：在 schema 检索之前，依据数据库简述为问题选定唯一数据库。

候选范围限定在当前数据集/切分的 tables.json；描述来自
<persist_dir>/database_descriptions.json。选定后由调用方设置活动数据库，
后续的表目录与按字段/按取值检索都只在该库内进行。
"""
import logging
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from znws.utils.dataset import list_db_ids, load_database_descriptions

logger = logging.getLogger(__name__)


class DatabaseChoice(BaseModel):
    """数据库路由的结构化输出。"""

    db_id: str = Field(description="最有可能回答该问题的唯一数据库标识")
    reasoning: str = Field(default="", description="选择该数据库的简要理由")


SYSTEM_PROMPT = (
    "你是数据库路由专家。下面给出候选数据库及其业务简述，"
    "请根据用户问题判断最需要查询的 1 个数据库。"
    "只能从候选 db_id 中选择，且必须恰好选择一个。"
)


def _format_candidates(candidates: List[Tuple[str, str]]) -> str:
    return "\n".join(f"- {db_id}: {desc or '（暂无简述）'}" for db_id, desc in candidates)


def select_database(
    query: str,
    *,
    model: BaseChatModel,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
) -> Tuple[str, str]:
    """为问题选择唯一数据库，返回 (db_id, reasoning)。

    - 候选范围限定在当前数据集/切分的 db_id；
    - 只有一个候选时不调用模型，直接返回；
    - 简述文件缺失时抛错，提示先运行 --descriptions-only。
    """
    descriptions = load_database_descriptions(dataset)
    if not descriptions:
        raise RuntimeError(
            "未找到数据库简述 database_descriptions.json；"
            "请先运行 python build_schema_sample.py --descriptions-only 生成。"
        )

    db_ids = list_db_ids(split, dataset)
    if not db_ids:
        raise RuntimeError("当前数据集/切分下没有可用的数据库。")

    # 候选限定在当前切分；缺简述的库以占位文本参与，避免被直接排除
    candidates = [(db_id, descriptions.get(db_id, "")) for db_id in db_ids]
    if len(candidates) == 1:
        db_id, _ = candidates[0]
        logger.info("仅有一个候选数据库，直接选用: %s", db_id)
        return db_id, "仅有一个候选数据库"

    # 注意：DashScope 兼容接口在使用 response_format=json_object 时，要求消息中
    # 必须出现 "json" 字样，因此这里显式给出 JSON 输出格式说明。
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"【候选数据库】\n{_format_candidates(candidates)}\n\n"
        f"【用户问题】\n{query}\n\n"
        "请以 JSON 输出，且只输出 JSON，格式为："
        '{"db_id": "<候选中的一个 db_id>", "reasoning": "<简要理由>"}。'
    )

    choice = model.with_structured_output(DatabaseChoice).invoke(prompt)
    db_id = (choice.db_id or "").strip()

    valid = {candidate for candidate, _ in candidates}
    if db_id not in valid:
        raise ValueError(f"数据库路由返回了无效的 db_id: {db_id!r}；候选为: {sorted(valid)}")

    logger.info("数据库路由: %s（%s）", db_id, choice.reasoning or "无理由")
    return db_id, choice.reasoning or ""
