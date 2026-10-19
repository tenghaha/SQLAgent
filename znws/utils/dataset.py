"""数据集注册与访问层（配置化，支持任意 BIRD-like 数据集）。

数据集的选择与布局由工程根目录下的 datasets.yaml 控制（可用环境变量
SQLAGENT_DATASET_CONFIG 覆盖路径），因此只要把其他数据集整理成 BIRD-like 结构
并登记到配置中，即可在不改代码的前提下接入。

BIRD-like 的最小约定：
    <root>/<split>/<databases>/<db_id>/<db_id>.sqlite          # 物理库
    <root>/<split>/<databases>/<db_id>/<desc_dir>/<table>.csv  # 列描述（可选）
    <root>/<split>/<tables_json>                               # Spiders 风格 schema 元数据

本模块提供：配置加载、路径解析、schema 元数据读取、列描述读取、活动数据集/数据库
状态，以及向量库与嵌入模型位置。
"""
import csv
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 工程根目录（znws/utils/dataset.py -> 上溯两级）
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_ENV_VAR = "SQLAGENT_DATASET_CONFIG"
DEFAULT_CONFIG_PATH = BASE_DIR / "datasets.yaml"

# BIRD-like 的默认命名/字段约定，可在 YAML 中逐项覆盖
DEFAULT_DATABASE_DIR_TEMPLATE = "{db_id}"
DEFAULT_DATABASE_FILE_TEMPLATE = "{db_id}.sqlite"
DEFAULT_DESC_DIR = "database_description"
DEFAULT_DESC_FILE_TEMPLATE = "{table}.csv"
DEFAULT_DESC_ENCODING = "utf-8-sig"
DEFAULT_DESC_COLUMNS = {
    "original_name": "original_column_name",
    "name": "column_name",
    "description": "column_description",
}
DEFAULT_SCHEMA_FIELDS = {
    "db_id": "db_id",
    "tables_original": "table_names_original",
    "tables_readable": "table_names",
    "columns_original": "column_names_original",
    "column_types": "column_types",
    "foreign_keys": "foreign_keys",
}
DEFAULT_EMBEDDER_MODEL_PATH = "znws/memory/models/jinaai/jina-embeddings-v5-text-small"
DEFAULT_VECTOR_STORE_DIR = "data/schema_sample"


def _resolve(path_like) -> Path:
    """相对路径按工程根目录解析；绝对路径原样返回。"""
    path = Path(path_like)
    return path if path.is_absolute() else (BASE_DIR / path)


# ------------------------------------------------------------------ 配置加载

def config_path() -> Path:
    override = os.getenv(CONFIG_ENV_VAR)
    return Path(override) if override else DEFAULT_CONFIG_PATH


@lru_cache(maxsize=None)
def _load_config_cached(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据集配置文件 {path}。请创建 datasets.yaml，"
            f"或用环境变量 {CONFIG_ENV_VAR} 指定配置路径。"
        )
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not data.get("datasets"):
        raise ValueError(f"数据集配置 {path} 缺少 datasets 段。")
    return data


def load_config() -> dict:
    return _load_config_cached(str(config_path()))


def list_datasets() -> List[str]:
    return list(load_config()["datasets"].keys())


def default_dataset_name() -> str:
    cfg = load_config()
    return cfg.get("default_dataset") or next(iter(cfg["datasets"]))


def _resolve_dataset(dataset: Optional[str]) -> str:
    return dataset or _active_dataset or default_dataset_name()


def get_dataset_config(dataset: Optional[str] = None) -> dict:
    cfg = load_config()
    name = _resolve_dataset(dataset)
    entry = cfg["datasets"].get(name)
    if entry is None:
        raise KeyError(f"数据集 {name!r} 未在配置中登记（可用：{', '.join(cfg['datasets'])}）")
    return {"name": name, **entry}


def embedder_model_path() -> str:
    cfg = load_config()
    return str(_resolve(cfg.get("embedder_model_path") or DEFAULT_EMBEDDER_MODEL_PATH))


def vector_store_config(dataset: Optional[str] = None) -> Tuple[str, str]:
    """返回 (persist_dir, collection_prefix)。

    集合名前缀缺省为该数据集专属（schema_samples_<dataset>），避免不同数据集互相污染；
    实际集合按“一库一集合”再拼上 db_id，见 database_collection_name。
    """
    entry = get_dataset_config(dataset)
    store = entry.get("vector_store") or {}
    persist_dir = store.get("persist_dir") or DEFAULT_VECTOR_STORE_DIR
    collection = store.get("collection") or f"schema_samples_{entry['name']}"
    return str(_resolve(persist_dir)), collection


def database_collection_name(base_collection: str, db_id: str) -> str:
    """一个数据库对应一个向量集合：<base_collection>__<db_id>。

    Chroma 的集合名允许字母/数字/下划线/连字符，且需以字母或数字开头结尾；
    BIRD-like 的 db_id 满足该约束。
    """
    return f"{base_collection}__{db_id}"


DESCRIPTIONS_FILENAME = "database_descriptions.json"


def descriptions_path(dataset: Optional[str] = None) -> Path:
    """数据库简述 JSON 的位置：<persist_dir>/database_descriptions.json。"""
    persist_dir, _ = vector_store_config(dataset)
    return Path(persist_dir) / DESCRIPTIONS_FILENAME


def load_database_descriptions(dataset: Optional[str] = None) -> Dict[str, str]:
    """读取数据库简述，返回 {db_id: 描述}；文件缺失或损坏时返回空字典。"""
    path = descriptions_path(dataset)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("读取数据库简述失败 %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("数据库简述格式异常（应为 db_id -> 描述 的字典）: %s", path)
        return {}
    return {str(db_id): str(text) for db_id, text in data.items() if text}


# ------------------------------------------------------------------ 路径解析

def dataset_root(dataset: Optional[str] = None) -> Path:
    return _resolve(get_dataset_config(dataset)["root"])


def default_split(dataset: Optional[str] = None) -> str:
    entry = get_dataset_config(dataset)
    splits = entry.get("splits") or {}
    if not splits:
        raise ValueError(f"数据集 {entry['name']!r} 未配置任何切分（splits）。")
    return entry.get("default_split") or next(iter(splits))


def _resolve_split(dataset: str, split: Optional[str]) -> str:
    if split:
        return split
    # 访问的是当前活动数据集时，沿用运行期已解析的切分
    if dataset == _active_dataset and _active_split:
        return _active_split
    return default_split(dataset)


def _split_config(dataset: Optional[str], split: Optional[str]) -> Tuple[str, str, dict]:
    """返回 (dataset_name, split_name, split 配置)。"""
    name = _resolve_dataset(dataset)
    splits = get_dataset_config(name).get("splits") or {}
    split_name = _resolve_split(name, split)
    entry = splits.get(split_name)
    if entry is None:
        raise KeyError(f"数据集 {name!r} 未登记切分 {split_name!r}（可用：{', '.join(splits)}）")
    return name, split_name, entry


def split_dir(split: Optional[str] = None, dataset: Optional[str] = None) -> Path:
    name, split_name, entry = _split_config(dataset, split)
    return dataset_root(name) / entry.get("path", split_name)


def tables_json_path(split: Optional[str] = None, dataset: Optional[str] = None) -> Path:
    return split_dir(split, dataset) / _split_config(dataset, split)[2]["tables"]


def databases_dir(split: Optional[str] = None, dataset: Optional[str] = None) -> Path:
    return split_dir(split, dataset) / _split_config(dataset, split)[2]["databases"]


def _database_dir(db_id: str, split: Optional[str], dataset: Optional[str]) -> Path:
    entry = get_dataset_config(dataset)
    template = (entry.get("database") or {}).get("dir_template", DEFAULT_DATABASE_DIR_TEMPLATE)
    return databases_dir(split, dataset) / template.format(db_id=db_id)


def database_path(db_id: str, split: Optional[str] = None, dataset: Optional[str] = None) -> Path:
    """物理库路径：<databases>/<dir_template(db_id)>/<file_template(db_id)>"""
    entry = get_dataset_config(dataset)
    template = (entry.get("database") or {}).get("file_template", DEFAULT_DATABASE_FILE_TEMPLATE)
    return _database_dir(db_id, split, dataset) / template.format(db_id=db_id)


def description_dir(db_id: str, split: Optional[str] = None, dataset: Optional[str] = None) -> Path:
    """列描述目录：<库目录>/<description.dir>"""
    entry = get_dataset_config(dataset)
    dir_name = (entry.get("description") or {}).get("dir", DEFAULT_DESC_DIR)
    return _database_dir(db_id, split, dataset) / dir_name


# --------------------------------------------------------- 活动数据集/数据库状态

_active_dataset: Optional[str] = None
_active_db_id: Optional[str] = None
_active_split: Optional[str] = None


def set_active_dataset(dataset: Optional[str] = None) -> None:
    """切换活动数据集；切分清空，回落到该数据集的默认切分。"""
    global _active_dataset, _active_split
    _active_dataset = dataset or default_dataset_name()
    _active_split = None


def set_active_database(
    db_id: str, split: Optional[str] = None, dataset: Optional[str] = None
) -> None:
    """设置本次运行所针对的数据集/切分/数据库，供各工具在执行期读取。"""
    global _active_dataset, _active_db_id, _active_split
    _active_dataset = dataset or _active_dataset or default_dataset_name()
    _active_db_id = db_id
    _active_split = split or default_split(_active_dataset)


def get_active_dataset() -> str:
    return _active_dataset or default_dataset_name()


def get_active_database() -> Tuple[Optional[str], str]:
    """返回 (db_id, split)；split 始终为可用的具体切分名。"""
    return _active_db_id, (_active_split or default_split(get_active_dataset()))


def active_database_path() -> Path:
    db_id, split = get_active_database()
    if not db_id:
        raise RuntimeError("尚未设置活动数据库，请先调用 set_active_database(db_id, split, dataset)。")
    return database_path(db_id, split, get_active_dataset())


def ensure_database(
    db_id: Optional[str] = None, split: Optional[str] = None, dataset: Optional[str] = None
) -> Path:
    """确认目标数据库存在，避免因 dataset / split / db_id 错误而静默失败。"""
    dataset_name = _resolve_dataset(dataset)
    db_id = db_id or _active_db_id
    if not db_id:
        raise RuntimeError("未指定 db_id，无法定位数据库。")
    split_name = _resolve_split(dataset_name, split)
    path = database_path(db_id, split_name, dataset_name)
    if not path.exists():
        raise RuntimeError(
            f"未找到数据库 {path}。请确认数据集({dataset_name})、切分({split_name})"
            f"与 db_id({db_id}) 正确，且数据集已解压。"
        )
    return path


# ----------------------------------------------------------- schema 元数据读取

def _schema_fields(dataset: Optional[str]) -> dict:
    entry = get_dataset_config(dataset)
    return {**DEFAULT_SCHEMA_FIELDS, **(entry.get("schema_fields") or {})}


@lru_cache(maxsize=None)
def _load_tables_meta_cached(dataset: str, split: str) -> Dict[str, dict]:
    path = tables_json_path(split, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 schema 元数据 {path}。请确认数据集({dataset})、切分({split})正确且已解压。"
        )
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    key_field = _schema_fields(dataset)["db_id"]
    return {record[key_field]: record for record in records}


def load_tables_meta(split: Optional[str] = None, dataset: Optional[str] = None) -> Dict[str, dict]:
    """读取切分的 schema 元数据文件，返回 {db_id: 记录}。"""
    name, split_name, _ = _split_config(dataset, split)
    return _load_tables_meta_cached(name, split_name)


def list_db_ids(split: Optional[str] = None, dataset: Optional[str] = None) -> List[str]:
    """列出切分下全部 db_id（保持元数据文件中的顺序）。"""
    return list(load_tables_meta(split, dataset).keys())


def get_db_meta(db_id: str, split: Optional[str] = None, dataset: Optional[str] = None) -> dict:
    meta = load_tables_meta(split, dataset).get(db_id)
    if meta is None:
        name, split_name, _ = _split_config(dataset, split)
        raise KeyError(f"数据集 {name!r} 切分 {split_name!r} 中不存在数据库 {db_id!r}")
    return meta


def db_tables(db_id: str, split: Optional[str] = None, dataset: Optional[str] = None) -> List[Tuple[str, str]]:
    """返回 [(原始表名, 人类可读表名), ...]。"""
    fields = _schema_fields(dataset)
    meta = get_db_meta(db_id, split, dataset)
    original = meta.get(fields["tables_original"]) or []
    readable = meta.get(fields["tables_readable"]) or []
    tables = []
    for idx, name in enumerate(original):
        human = readable[idx] if idx < len(readable) else name
        tables.append((name, human))
    return tables


def db_columns(db_id: str, split: Optional[str] = None, dataset: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """返回 [(表名, 字段名, 声明类型), ...]。

    列名采用 [表下标, 列名] 二元组；表下标 -1 表示全局通配列 '*'，不是真实列，需跳过。
    """
    fields = _schema_fields(dataset)
    meta = get_db_meta(db_id, split, dataset)
    table_names = meta.get(fields["tables_original"]) or []
    columns = meta.get(fields["columns_original"]) or []
    column_types = meta.get(fields["column_types"]) or []
    result = []
    for idx, entry in enumerate(columns):
        if not entry:
            continue
        table_idx, field = entry[0], entry[1]
        if table_idx is None or table_idx < 0 or table_idx >= len(table_names):
            continue
        col_type = column_types[idx] if idx < len(column_types) else ""
        result.append((table_names[table_idx], field, col_type or ""))
    return result


def db_relationships(
    db_id: str,
    tables: Optional[Iterable[str]] = None,
    split: Optional[str] = None,
    dataset: Optional[str] = None,
) -> List[Dict[str, str]]:
    """返回该库的外键关系，可限定在给定表集合内。

    每条记录形如 {"child_table", "child_field", "parent_table", "parent_field"}。
    BIRD 的 foreign_keys 为 [子列下标, 父列下标]，下标相对 column_names_original
    （含表下标 -1 的通配列，不能按表内序号理解）。
    """
    fields = _schema_fields(dataset)
    meta = get_db_meta(db_id, split, dataset)
    table_names = meta.get(fields["tables_original"]) or []
    columns = meta.get(fields["columns_original"]) or []
    foreign_keys = meta.get(fields["foreign_keys"]) or []

    def column_at(index) -> Optional[Tuple[str, str]]:
        if not isinstance(index, int) or index < 0 or index >= len(columns):
            return None
        entry = columns[index]
        if not entry:
            return None
        table_idx, field = entry[0], entry[1]
        if table_idx is None or table_idx < 0 or table_idx >= len(table_names):
            return None
        return table_names[table_idx], field

    allowed = set(tables) if tables is not None else None
    relationships: List[Dict[str, str]] = []
    seen = set()

    for pair in foreign_keys:
        if not pair or len(pair) < 2:
            continue
        child = column_at(pair[0])
        parent = column_at(pair[1])
        if child is None or parent is None:
            continue
        child_table, child_field = child
        parent_table, parent_field = parent
        if allowed is not None and (child_table not in allowed or parent_table not in allowed):
            continue
        key = (child_table, child_field, parent_table, parent_field)
        if key in seen:
            continue
        seen.add(key)
        relationships.append({
            "child_table": child_table,
            "child_field": child_field,
            "parent_table": parent_table,
            "parent_field": parent_field,
        })
    return relationships


# --------------------------------------------------------------- 列描述读取

def _description_config(dataset: Optional[str]) -> dict:
    entry = get_dataset_config(dataset)
    desc = entry.get("description") or {}
    return {
        "dir": desc.get("dir", DEFAULT_DESC_DIR),
        "file_template": desc.get("file_template", DEFAULT_DESC_FILE_TEMPLATE),
        "encoding": desc.get("encoding", DEFAULT_DESC_ENCODING),
        "columns": {**DEFAULT_DESC_COLUMNS, **(desc.get("columns") or {})},
    }


def _match_description_file(desc_dir: Path, table: str, file_template: str) -> Optional[Path]:
    """定位某张表的描述文件：先精确匹配，再退化为大小写不敏感匹配。"""
    exact = desc_dir / file_template.format(table=table)
    if exact.exists():
        return exact
    target = table.strip().lower()
    for candidate in desc_dir.glob("*"):
        if candidate.is_file() and candidate.stem.strip().lower() == target:
            return candidate
    return None


@lru_cache(maxsize=None)
def _table_column_descriptions_cached(
    dataset: str, split: str, db_id: str, table: str
) -> Dict[str, str]:
    cfg = _description_config(dataset)
    desc_dir = description_dir(db_id, split, dataset)
    if not desc_dir.exists():
        return {}
    csv_path = _match_description_file(desc_dir, table, cfg["file_template"])
    if csv_path is None:
        return {}

    columns = cfg["columns"]
    mapping: Dict[str, str] = {}
    try:
        with csv_path.open("r", encoding=cfg["encoding"], newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row.get(columns["original_name"]) or "").strip()
                if not key:
                    continue
                text = (row.get(columns["description"]) or "").strip() or (row.get(columns["name"]) or "").strip()
                mapping[key] = text
    except Exception as exc:  # 描述缺失不应中断主流程
        logger.debug("读取列描述失败 %s.%s: %s", db_id, table, exc)
    return mapping


def table_column_descriptions(
    db_id: str, table: str, split: Optional[str] = None, dataset: Optional[str] = None
) -> Dict[str, str]:
    """读取某张表的描述文件，返回 {原始列名: 文字描述}。"""
    name, split_name, _ = _split_config(dataset, split)
    return _table_column_descriptions_cached(name, split_name, db_id, table)


def column_description(
    db_id: str, table: str, field: str, split: Optional[str] = None, dataset: Optional[str] = None
) -> str:
    """返回某列的文字描述，列名按去空格、大小写不敏感方式匹配。"""
    mapping = table_column_descriptions(db_id, table, split, dataset)
    if not mapping:
        return ""
    if field in mapping:
        return mapping[field]
    target = field.strip().lower()
    for name, text in mapping.items():
        if name.strip().lower() == target:
            return text
    return ""
