"""BIRD 外部知识（evidence）读取工具。

QueryProcessor 在固定流程中直接调用本工具，把题目对应的 evidence 注入
实体抽取、查询改写、Schema 链接与 SQL 生成的上下文。
"""
from functools import lru_cache
from typing import Dict

from langchain_core.tools import tool

from znws.utils.dataset import get_active_database, get_active_dataset, load_questions


@lru_cache(maxsize=None)
def _evidence_map(dataset: str, split: str) -> Dict[object, str]:
    """按 (dataset, split) 缓存 question_id -> evidence 映射。"""
    questions = load_questions(split, dataset)
    return {
        item.get("question_id", index): (item.get("evidence") or "").strip()
        for index, item in enumerate(questions)
    }


@tool
def load_bird_evidence(
    question_id: int,
    dataset: str | None = None,
    split: str | None = None,
) -> str:
    """
    Load BIRD's external evidence for a question (business definitions, formulas,
    or domain hints). Returns an empty string when the question has no evidence.
    The dataset and split default to the currently active run context.
    """
    dataset_name = dataset or get_active_dataset()
    split_name = split or get_active_database()[1]
    try:
        return _evidence_map(dataset_name, split_name).get(question_id, "")
    except Exception:
        # 外部知识是可选增强，读取失败时不阻断主流程
        return ""


__all__ = ["load_bird_evidence"]
