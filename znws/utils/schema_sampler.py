"""基于 BIRD-like 数据集离线构建 schema 抽样向量库（一库一集合）。

与原实现（单一物理库 + 内置元数据库）的差异：
- 数据源由 datasets.yaml 决定：schema 取自 <tables_json>，取值取自每个 <db_id>.sqlite，
  列说明取自配置登记的列描述 CSV；
- 每个 db_id 单独建一个 Chroma 集合（<base>__<db_id>），避免单一集合体量过大；
- 同时用 LLM（输入仅表名与字段名）为每个数据库生成一句话简述，汇总写入
  <persist_dir>/database_descriptions.json；
- 可按 dataset / split / db_ids 选择性构建，并按库分批写入。

产物位置、集合名前缀与嵌入模型取自 datasets.yaml 的 vector_store / embedder_model_path。
"""
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chromadb

from znws.utils.dataset import (
    BASE_DIR,
    DESCRIPTIONS_FILENAME,
    column_description,
    database_collection_name,
    database_path,
    db_columns,
    db_tables,
    default_dataset_name,
    default_split,
    embedder_model_path,
    list_db_ids,
    vector_store_config,
)

logger = logging.getLogger(__name__)

# str-like 字段每个字段枚举的不同取值数量：只抽样单个取值会让大量真实取值无法被检索命中
STR_ENUM_VALUES = 5
# 枚举取值时的扫描行数上限，避免大表全表扫描拖慢构建
STR_ENUM_SCAN_LIMIT = 20000
# 声明类型中出现这些关键字时视为数值字段
_NUMERIC_TYPE_KEYS = ("INT", "REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL")
# Chroma 单次 add 的文档批量
_ADD_BATCH_SIZE = 1000


def _looks_numeric(value) -> bool:
    """判断一个取值是否是数字。"""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_numeric_field(declared_type: str, sample_value, max_val) -> bool:
    """
    判断字段是否为数值类型。

    除声明类型外，还会看真实取值：有些数值列在 SQLite 里被声明为 TEXT，
    这类字段同样按数值处理，不做多值枚举。
    """
    declared = (declared_type or "").upper()
    if any(key in declared for key in _NUMERIC_TYPE_KEYS):
        return True
    return _looks_numeric(sample_value) and _looks_numeric(max_val)


def _sample_distinct_values(cursor, table: str, field: str, cn_name: str, limit: int = STR_ENUM_VALUES) -> List[str]:
    """
    枚举字段的若干个不同真实取值。

    为了让用户提到的具体实体值有机会通过向量检索命中对应字段，需要把字段的
    多个典型取值分别建立索引；只抽样单个取值会导致绝大多数取值无法被召回。
    取值按出现频次排序，优先索引更有代表性的典型值。
    """
    sql = (
        f'SELECT "{field}" FROM ('
        f'SELECT "{field}" FROM "{table}" '
        f'WHERE "{field}" IS NOT NULL AND TRIM(CAST("{field}" AS TEXT)) != \'\' '
        f'AND CAST("{field}" AS TEXT) != ? '
        f'LIMIT {STR_ENUM_SCAN_LIMIT}'
        f') GROUP BY "{field}" ORDER BY COUNT(*) DESC LIMIT {limit}'
    )
    try:
        cursor.execute(sql, (cn_name,))
        return [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
    except Exception as e:
        logger.debug(f"枚举字段取值失败 {table}.{field}: {e}")
        return []


def _encode(embedder, documents: List[str]) -> List[list]:
    """向量化；部分模型不接受 task 参数时回退。"""
    try:
        return embedder.encode(documents, task="retrieval").tolist()
    except Exception as e:
        logger.warning(f"向量化时出错，尝试不使用 task 参数: {e}")
        return embedder.encode(documents).tolist()


def _collect_db_documents(
    db_id: str, split: str, dataset: str, db_file: Path
) -> Tuple[List[str], List[Dict], List[str]]:
    """抽取单个数据库的抽样文档（Key 文档 + Value 文档）。"""
    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    try:
        # 缓存每张表的字段声明类型与总行数，避免重复扫描
        table_column_types: Dict[str, Dict[str, str]] = {}
        table_totals: Dict[str, int] = {}

        for table, field, meta_type in db_columns(db_id, split, dataset):
            try:
                safe_table = f'"{table}"'
                safe_field = f'"{field}"'

                if table not in table_column_types:
                    cursor.execute(f"PRAGMA table_info({safe_table})")
                    table_column_types[table] = {row[1]: (row[2] or "") for row in cursor.fetchall()}
                declared_type = table_column_types[table].get(field, "") or meta_type

                if table not in table_totals:
                    cursor.execute(f"SELECT COUNT(*) FROM {safe_table}")
                    table_totals[table] = cursor.fetchone()[0]
                total_rows = table_totals[table]
                if total_rows == 0:
                    continue

                cursor.execute(f"""
                    SELECT
                        COUNT(*),
                        COUNT(DISTINCT {safe_field}),
                        MIN({safe_field}),
                        MAX({safe_field})
                    FROM {safe_table}
                    WHERE {safe_field} IS NOT NULL
                      AND TRIM(CAST({safe_field} AS TEXT)) != ''
                      AND CAST({safe_field} AS TEXT) != 'null'
                      AND CAST({safe_field} AS TEXT) != 'None'
                """)
                non_empty_count, distinct_count, min_val, max_val = cursor.fetchone()

                if not non_empty_count:
                    continue

                missing_rate = (total_rows - non_empty_count) / total_rows
                if missing_rate >= 0.9:
                    logger.debug("字段 %s.%s.%s 缺失率 %.1f%% >= 90%%，跳过。", db_id, table, field, missing_rate * 100)
                    continue

                cn_name = column_description(db_id, table, field, split, dataset) or field

                # 获取一个真实取值（BIRD 的 SQLite 中不含表头行，无需额外跳过）
                cursor.execute(
                    f'SELECT {safe_field} FROM {safe_table} WHERE {safe_field} IS NOT NULL '
                    f'AND TRIM(CAST({safe_field} AS TEXT)) != \'\' LIMIT 1'
                )
                sample_res = cursor.fetchone()
                sample_value = str(sample_res[0]) if sample_res and sample_res[0] is not None else str(max_val)

                base_meta = {
                    "db_id": db_id,
                    "table": table,
                    "field": field,
                    "cn_name": cn_name,
                    "sample_value": sample_value,
                    "missing_rate": missing_rate,
                    "distinct_count": distinct_count,
                    "min_value": str(min_val) if min_val is not None else "",
                    "max_value": str(max_val) if max_val is not None else "",
                }

                # --- Key 文档：供业务术语检索 ---
                documents.append(f"{cn_name} ({field})")
                metadatas.append({**base_meta, "type": "key"})
                ids.append(f"key_{db_id}_{table}_{field}")

                # --- Value 文档：供实体取值检索 ---
                if _is_numeric_field(declared_type, sample_value, max_val):
                    value_docs = [sample_value]
                else:
                    value_docs = _sample_distinct_values(cursor, table, field, cn_name) or [sample_value]

                for value_index, value_doc in enumerate(value_docs):
                    documents.append(value_doc)
                    metadatas.append({**base_meta, "type": "value", "sample_value": value_doc})
                    ids.append(
                        f"val_{db_id}_{table}_{field}" if value_index == 0
                        else f"val_{db_id}_{table}_{field}_{value_index}"
                    )
            except Exception as e:
                logger.debug("抽样 %s.%s.%s 失败: %s", db_id, table, field, e)
                continue
    finally:
        conn.close()

    return documents, metadatas, ids


def _schema_outline(db_id: str, split: str, dataset: str) -> str:
    """构造仅含表名与字段名的紧凑 schema 描述，作为生成数据库简述的输入。"""
    tables = db_tables(db_id, split, dataset)
    fields_by_table: Dict[str, List[str]] = {name: [] for name, _ in tables}
    for table, field, _ in db_columns(db_id, split, dataset):
        fields_by_table.setdefault(table, []).append(field)

    lines = []
    for name, human in tables:
        label = f"{name} ({human})" if human and human != name else name
        lines.append(f"- {label}: {', '.join(fields_by_table.get(name, []))}")
    return "\n".join(lines)


def _build_description_model():
    """构造生成数据库简述用的对话模型（沿用工程的 OPENAI_* 环境变量）。

    显式加载工程根目录下的 .env，保证无论从哪个入口调用都能读到 API 配置。
    """
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv(BASE_DIR / ".env")
    return ChatOpenAI(
        model=os.getenv("OPENAI_API_MODEL", "qwen-plus"),
        base_url=os.getenv("OPENAI_API_BASE") or None,
        temperature=0.0,
    )


def _describe_database(model, db_id: str, split: str, dataset: str) -> str:
    """让 LLM 依据表名/字段名生成一句话的数据库简述。"""
    outline = _schema_outline(db_id, split, dataset)
    prompt = (
        "下面是一个关系型数据库的表结构（仅列出表名与字段名）：\n\n"
        f"{outline}\n\n"
        "请用一句简洁、信息密集的中文概括这个数据库存储的业务数据，"
        "说明它属于什么领域、主要记录哪些实体。只输出这一句话，不要任何解释。"
    )
    response = model.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()


def _generate_database_descriptions(
    db_ids: Iterable[str], split: str, dataset: str
) -> Dict[str, str]:
    """逐库生成简述；模型或单个库失败只记录告警，不中断整体构建。"""
    try:
        model = _build_description_model()
    except Exception as e:
        logger.warning(
            "初始化数据库简述模型失败，已跳过简述生成（请检查工程根目录 .env 中的 "
            "OPENAI_API_KEY / OPENAI_API_BASE）: %s",
            e,
        )
        return {}

    descriptions: Dict[str, str] = {}
    for db_id in db_ids:
        try:
            descriptions[db_id] = _describe_database(model, db_id, split, dataset)
            logger.info("已生成数据库简述: %s", db_id)
        except Exception as e:
            logger.warning("生成数据库简述失败 %s: %s", db_id, e)
    return descriptions


def _write_descriptions(path: Path, descriptions: Dict[str, str]) -> None:
    """写入简述 JSON；与已有文件合并，只覆盖非空结果，保留历史成功条目。"""
    existing: Dict[str, str] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception as e:
            logger.warning("读取已有数据库简述失败，将重新生成: %s", e)
    existing.update({db_id: text for db_id, text in descriptions.items() if text})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def build_database_descriptions(
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    db_ids: Optional[Iterable[str]] = None,
    chroma_persist_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """仅生成/合并数据库简述 JSON，不触碰已建好的向量集合。

    适用于向量库已构建、但简述因缺少 API 配置而失败的场景。
    返回 (json 路径, {db_id: 简述})。
    """
    dataset = dataset or default_dataset_name()
    split = split or default_split(dataset)
    default_dir, _ = vector_store_config(dataset)
    persist_dir = chroma_persist_dir or default_dir
    targets = list(db_ids) if db_ids else list_db_ids(split, dataset)

    descriptions = _generate_database_descriptions(targets, split, dataset)
    path = Path(persist_dir) / DESCRIPTIONS_FILENAME
    if descriptions:
        _write_descriptions(path, descriptions)
    return str(path), descriptions


def build_schema_sample(
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    db_ids: Optional[Iterable[str]] = None,
    rebuild: bool = True,
    chroma_persist_dir: Optional[str] = None,
    collection_prefix: Optional[str] = None,
    embedder_path: Optional[str] = None,
    generate_descriptions: bool = True,
) -> dict:
    """离线构建 schema 抽样向量库（一库一集合）与数据库简述，返回构建摘要。

    dataset: datasets.yaml 中登记的数据集名；缺省为配置的默认数据集。
    split: 切分名；缺省为该数据集的默认切分。
    db_ids: 限定构建哪些库；缺省为该切分下全部数据库。
    rebuild: True 表示先删除各库对应的同名集合再重建。
    generate_descriptions: 是否调用 LLM 生成 database_descriptions.json。
    其余参数缺省取自 datasets.yaml 的 vector_store 与 embedder_model_path。
    """
    from sentence_transformers import SentenceTransformer

    dataset = dataset or default_dataset_name()
    split = split or default_split(dataset)
    default_dir, default_prefix = vector_store_config(dataset)
    persist_dir = chroma_persist_dir or default_dir
    prefix = collection_prefix or default_prefix
    model_path = embedder_path or embedder_model_path()

    targets = list(db_ids) if db_ids else list_db_ids(split, dataset)
    logger.info(
        "开始为数据集 %s（切分 %s）的 %d 个数据库构建 schema 抽样库...",
        dataset, split, len(targets),
    )

    logger.info("正在加载嵌入模型: %s", model_path)
    embedder = SentenceTransformer(model_path, trust_remote_code=True)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    collections: Dict[str, object] = {}
    total_docs = 0
    for db_id in targets:
        db_file = database_path(db_id, split, dataset)
        if not db_file.exists():
            logger.warning("跳过 %s：未找到数据库文件 %s", db_id, db_file)
            continue

        documents, metadatas, ids = _collect_db_documents(db_id, split, dataset, db_file)
        if not documents:
            logger.warning("跳过 %s：未抽取到任何有效的字段值。", db_id)
            continue

        # 一库一集合：<prefix>__<db_id>
        collection_name = database_collection_name(prefix, db_id)
        if rebuild:
            try:
                chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass  # 集合不存在则忽略
        collection = chroma_client.create_collection(name=collection_name)

        vectors = _encode(embedder, documents)
        for start in range(0, len(documents), _ADD_BATCH_SIZE):
            end = start + _ADD_BATCH_SIZE
            collection.add(
                documents=documents[start:end],
                embeddings=vectors[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

        collections[db_id] = collection
        total_docs += len(documents)
        logger.info("完成 %s：集合 %s，新增 %d 条文档。", db_id, collection_name, len(documents))

    descriptions_path = Path(persist_dir) / DESCRIPTIONS_FILENAME
    descriptions: Dict[str, str] = {}
    if generate_descriptions:
        descriptions = _generate_database_descriptions(targets, split, dataset)
        if descriptions:
            _write_descriptions(descriptions_path, descriptions)
            logger.info("数据库简述已写入 %s（%d 条）。", descriptions_path, len(descriptions))
        else:
            logger.warning(
                "未生成任何数据库简述；向量库已构建，配置 .env 后可执行 "
                "--descriptions-only 补齐。"
            )

    logger.info(
        "schema 抽样库构建完成：%d 个数据库集合，共 %d 条文档。",
        len(collections), total_docs,
    )
    return {
        "dataset": dataset,
        "split": split,
        "persist_dir": persist_dir,
        "collection_prefix": prefix,
        "collections": collections,
        "documents": total_docs,
        "descriptions": descriptions,
        "descriptions_path": str(descriptions_path),
    }
