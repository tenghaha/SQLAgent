"""BIRD 官方预测文件（preds.json）的公共读写工具。

官方字段：``question_id`` / ``db_id`` / ``SQL``；写入前统一压缩空白（去掉换行）。
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def clean_sql(sql: Optional[str]) -> str:
    """写入 preds 前压缩空白：去掉换行/制表符，并合并连续空格。"""
    return " ".join((sql or "").split())


def load_existing(path: Path) -> Dict[object, dict]:
    """读取已有 preds，返回 {question_id: record}；文件缺失或损坏时返回空字典。"""
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return {record["question_id"]: record for record in records if "question_id" in record}
    except Exception as exc:
        print(f"[warn] 读取已有 preds 失败: {exc}")
        return {}


def write_preds(path: Path, records: List[dict]) -> None:
    """原子写入 preds.json，避免中途崩溃损坏文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
