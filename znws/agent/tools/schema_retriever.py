"""Schema 检索工具（适配 BIRD-like 数据集）。

对外三类能力保持不变：
- get_table_directory：列出当前数据库的表目录
- search_schema_by_field：按业务术语检索 schema 抽样库
- search_schema_by_value：按实体取值检索 schema 抽样库

适配要点：
- 表目录不再读工程内的元数据库，而是读配置中登记的 <tables_json>；
- 抽样库中每条文档带有 db_id，检索时按“活动数据库”过滤；
- 列的文字说明来自配置中登记的列描述 CSV。

注：嵌入模型与向量库位置来自 datasets.yaml（当前活动数据集）；集合对象在首次
调用时解析并缓存，因此请在进程内固定数据集（研究场景即一次运行一个数据集）。
"""
import logging
from typing import List, Tuple

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from znws.utils.dataset import (
    column_description,
    database_collection_name,
    db_columns,
    db_tables,
    embedder_model_path,
    get_active_database,
    vector_store_config,
)

logger = logging.getLogger(__name__)

# 值检索候选数：str 类字段会为多个不同取值分别建索引，同一实体的文档rank 会靠后，
# 需要更大的候选额度才能在按字段去重后仍召回正确的字段。
FILTER_VALUE_TOP_K = 50
SCHEMA_FIELD_TOP_K = 5

_shared_embedder = None
# 显式注入的集合（供 comprehensive_schema_retrieval 使用），优先于按 db_id 解析
_shared_schema_collection = None
# 一库一集合：集合名 -> Collection
_db_schema_collections: dict = {}


def get_shared_embedder():
    global _shared_embedder
    if _shared_embedder is None:
        from sentence_transformers import SentenceTransformer
        _shared_embedder = SentenceTransformer(embedder_model_path(), trust_remote_code=True)
    return _shared_embedder


def get_shared_schema_collection():
    """按当前活动数据库取对应的向量集合（一库一集合）。"""
    if _shared_schema_collection is not None:
        return _shared_schema_collection

    db_id, _ = get_active_database()
    if not db_id:
        return None

    persist_dir, base_collection = vector_store_config()
    collection_name = database_collection_name(base_collection, db_id)
    if collection_name not in _db_schema_collections:
        chroma_client_schema = chromadb.PersistentClient(path=persist_dir)
        try:
            _db_schema_collections[collection_name] = chroma_client_schema.get_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"获取 {collection_name} 失败: {e}")
            _db_schema_collections[collection_name] = None
    return _db_schema_collections.get(collection_name)


def _schema_where(doc_type: str) -> dict:
    """构造检索过滤条件：按当前活动数据库 + 文档类型（key / value）过滤。

    注意：chromadb 的 where 只接受单运算符，多条件必须显式用 $and 组合，
    不能写成 {"type": ..., "db_id": ...}（会抛 "Expected where to have exactly
    one operator"）。
    """
    db_id, _ = get_active_database()
    if db_id:
        return {"$and": [{"type": doc_type}, {"db_id": db_id}]}
    return {"type": doc_type}


def _format_meta(meta: dict) -> str:
    missing_rate_str = f"{meta.get('missing_rate', 0):.1%}"
    return (
        f"表名: {meta['table']}, 字段: {meta['field']}, 中文: {meta['cn_name']}, "
        f"典型值: {meta['sample_value']}, 缺失率: {missing_rate_str}, "
        f"唯一值数: {meta.get('distinct_count', 0)}, "
        f"Min: {meta.get('min_value', '')}, Max: {meta.get('max_value', '')}"
    )


class SearchSchemaByValueArgs(BaseModel):
    value: str = Field(description="需要检索的实体实例或具体值")


@tool(args_schema=SearchSchemaByValueArgs)
def search_schema_by_value(value: str) -> str:
    """
    通过具体的实体实例/具体值（如人名、地名、时间等）检索 Schema 向量库，获取相关的物理表、字段及其典型值统计信息。
    """
    embedder = get_shared_embedder()
    collection = get_shared_schema_collection()
    if not collection:
        return "Schema 向量库未初始化或无法连接。"

    try:
        query_embedding = embedder.encode(value, task="retrieval").tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=FILTER_VALUE_TOP_K,
            where=_schema_where("value"),
        )
        if results and results.get("metadatas") and results["metadatas"][0]:
            docs_info = []
            seen_fields = set()
            for meta in results["metadatas"][0]:
                # 同一个字段可能有多个取值文档同时命中，按字段去重，避免占满候选额度
                field_key = (meta.get("table"), meta.get("field"))
                if field_key in seen_fields:
                    continue
                seen_fields.add(field_key)
                docs_info.append(_format_meta(meta))
            return f"【值匹配 - {value}】检索结果:\n" + "\n".join(docs_info)
        return f"未找到与值 '{value}' 相关的字段信息。"
    except Exception as e:
        return f"基于值检索时出错: {e}"


class SearchSchemaByFieldArgs(BaseModel):
    field_name: str = Field(description="需要检索的字段业务术语或口径名称")


@tool(args_schema=SearchSchemaByFieldArgs)
def search_schema_by_field(field_name: str) -> str:
    """
    通过字段的业务术语或口径（如“销量”、“产品”等名词）检索 Schema 向量库，获取相关的物理表、字段及其统计信息。
    """
    embedder = get_shared_embedder()
    collection = get_shared_schema_collection()
    if not collection:
        return "Schema 向量库未初始化或无法连接。"

    try:
        query_embedding = embedder.encode(field_name, task="retrieval").tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=SCHEMA_FIELD_TOP_K,
            where=_schema_where("key"),
        )
        if results and results.get("metadatas") and results["metadatas"][0]:
            docs_info = [_format_meta(meta) for meta in results["metadatas"][0]]
            return f"【字段匹配 - {field_name}】检索结果:\n" + "\n".join(docs_info)
        return f"未找到与字段名词 '{field_name}' 相关的字段信息。"
    except Exception as e:
        return f"基于字段名检索时出错: {e}"


class GetTableDirectoryArgs(BaseModel):
    pass


@tool(args_schema=GetTableDirectoryArgs)
def get_table_directory() -> str:
    """
    检索当前数据库的表目录，返回所有存在的物理表名及其人类可读名称。
    """
    db_id, split = get_active_database()
    if not db_id:
        return "尚未设置活动数据库（db_id），无法读取表目录。"
    try:
        rows = []
        for name, human in db_tables(db_id, split):
            rows.append(f"{name} ({human})" if human and human != name else name)
        if rows:
            return f"数据库表目录 [{db_id}]:\n" + "\n".join(rows)
        return "数据库表目录为空。"
    except Exception as e:
        return f"读取表目录失败: {e}"


class SchemaDecision(BaseModel):
    tables: List[str] = Field(description="模型判断的物理表列表（'表名'的形式，跨表查询可返回多个）")
    fields: List[str] = Field(description="模型判断的字段列表（'table.field'的形式）")
    statistics_fields: List[str] = Field(description="模型判断的关键统计量字段列表（'table.field'的形式）")


def comprehensive_schema_retrieval(
    final_query: str,
    schema_fields: dict,
    filter_values: dict,
    embedder,
    schema_collection,
    model_name: str,
    base_url: str = None,
    db_id: str = None,
) -> Tuple[SchemaDecision, List[str], dict]:
    """
    综合检索工具（在单个 BIRD 数据库范围内）：
    1. 用提取出的 filter_values 对 schema 抽样库进行基于 Value 的向量检索
    2. 用提取出的 schema_fields 对 schema 抽样库进行基于 Key 的向量检索
    3. 从 *_tables.json 读取当前数据库的表目录
    4. 组装 prompt 并通过 LLM 进行结构化决策
    5. 用 tables.json + database_description 过滤幻觉字段并补齐列说明
    """
    db_id = db_id or get_active_database()[0]
    split = get_active_database()[1]
    schema_results = []

    if schema_collection:
        # 1. 实体值匹配检索 (Filter Values -> Value 文档)
        values_to_search = []
        if filter_values.get("start_time"):
            values_to_search.extend(filter_values["start_time"])
        if filter_values.get("end_time"):
            values_to_search.extend(filter_values["end_time"])
        if filter_values.get("company"):
            values_to_search.append(filter_values["company"])
        if filter_values.get("other"):
            values_to_search.extend(filter_values["other"])

        # 设置全局共享缓存以供内部工具使用
        global _shared_embedder, _shared_schema_collection
        _shared_embedder = embedder
        _shared_schema_collection = schema_collection

        for val in set(values_to_search):
            res = search_schema_by_value.invoke({"value": val})
            if not res.startswith("未找到") and not res.startswith("Schema 向量库未") and not res.startswith("基于值检索时出错"):
                schema_results.append(res)

        # 2. 字段映射匹配检索 (Schema Fields -> Key 文档)
        fields_to_search = []
        if schema_fields.get("dimension"):
            fields_to_search.extend(schema_fields["dimension"])
        if schema_fields.get("metric"):
            fields_to_search.extend(schema_fields["metric"])

        for field in set(fields_to_search):
            res = search_schema_by_field.invoke({"field_name": field})
            if not res.startswith("未找到") and not res.startswith("Schema 向量库未") and not res.startswith("基于字段名检索时出错"):
                schema_results.append(res)

    # 3. 读取当前数据库的表目录（BIRD: *_tables.json）
    table_directory_str = get_table_directory.invoke({})

    # 4. 构建 LLM Prompt
    prompt_template = """你是一个专业的数据库 Schema 映射专家。请根据以下信息，判断需要使用哪些表、哪些字段。

【之前的 Query 及上下文】
{final_query}

【Schema 向量检索结果】
{schema_results}

【数据库表目录】
{table_directory}

【任务要求】
1. 从该数据库中选出最符合用户查询意图的表；跨表查询时可选择多个表。
2. 从所选表中选出需要使用的字段。你只能从【Schema 向量检索结果】中提供的字段中进行选择。严禁编造任何不存在的字段！如果在检索结果中找不到合适的字段，请不要强行编造，可以留空或仅返回明确存在的字段。
3. 选出关键统计量所使用的字段。同样，严禁编造！
4. 如果查询中存在具体的实体过滤条件（如人名、地名、类别等），必须在 fields 中选出承载该实体的过滤字段，不得因为它是筛选条件而遗漏。

请严格以 JSON 格式输出决策，并严格使用以下 JSON 结构：
{{
    "tables": ["表名1"],
    "fields": ["table1.field1", "table1.field2"],
    "statistics_fields": ["table1.stat_field1"]
}}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(SchemaDecision)
    chain = prompt | structured_llm

    decision = chain.invoke({
        "final_query": final_query,
        "schema_results": "\n\n".join(schema_results) if schema_results else "无",
        "table_directory": table_directory_str,
    })

    # 5. 补充字段的文字说明并过滤幻觉字段（BIRD: tables.json + database_description）
    valid_columns = {f"{table}.{field}" for table, field, _ in db_columns(db_id, split)} if db_id else set()
    cn_names_mapping: List[str] = []
    pure_cn_name_dict: dict = {}
    valid_fields: List[str] = []
    valid_stat_fields: List[str] = []

    def get_field_info(table_field: str):
        if "." not in table_field:
            return None
        table, field = table_field.split(".", 1)
        if valid_columns and table_field not in valid_columns:
            return None

        cn_name = column_description(db_id, table, field, split) if db_id else ""
        info_str = f"{table}.{field}: 中文名[{cn_name}]"

        if schema_collection and db_id:
            try:
                get_res = schema_collection.get(ids=[f"key_{db_id}_{table}_{field}"])
                if get_res and get_res.get("metadatas") and get_res["metadatas"]:
                    meta = get_res["metadatas"][0]
                    missing_rate_str = f"{meta.get('missing_rate', 0):.1%}"
                    info_str += (
                        f", 典型值: {meta.get('sample_value', '')}, 缺失率: {missing_rate_str}, "
                        f"唯一值数: {meta.get('distinct_count', 0)}, "
                        f"Min: {meta.get('min_value', '')}, Max: {meta.get('max_value', '')}"
                    )
            except Exception as e:
                logger.warning(f"获取 schema sample 信息失败 {table}.{field}: {e}")
        return info_str, cn_name

    # 验证 decision.fields
    for table_field in decision.fields:
        result = get_field_info(table_field)
        if result:
            info_str, cn_name = result
            valid_fields.append(table_field)
            cn_names_mapping.append(info_str)
            if cn_name:
                pure_cn_name_dict[table_field] = cn_name
        else:
            logger.warning(f"过滤掉不存在的幻觉字段: {table_field}")

    # 验证 decision.statistics_fields
    for table_field in decision.statistics_fields:
        result = get_field_info(table_field)
        if result:
            info_str, cn_name = result
            valid_stat_fields.append(table_field)
            if info_str not in cn_names_mapping:
                cn_names_mapping.append(info_str)
            if cn_name and table_field not in pure_cn_name_dict:
                pure_cn_name_dict[table_field] = cn_name
        else:
            logger.warning(f"过滤掉不存在的幻觉统计字段: {table_field}")

    decision.fields = valid_fields
    decision.statistics_fields = valid_stat_fields

    return decision, cn_names_mapping, pure_cn_name_dict
