"""离线构建 BIRD-like 数据集的 schema 抽样向量库与数据库简述。

每个 db_id 单独建一个向量集合（<前缀>__<db_id>），并把 LLM 生成的一句话简述
汇总写入 <persist_dir>/database_descriptions.json。

API 配置从工程根目录的 .env 读取（OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_API_MODEL）。

用法:
    python build_schema_sample.py                       # 默认数据集 + 其默认切分 + 全部库
    python build_schema_sample.py --split dev_20240627
    python build_schema_sample.py --dataset bird --split dev_20240627 --db-ids california_schools financial
    python build_schema_sample.py --skip-descriptions   # 只建向量库，不调用 LLM、不消耗 token
    python build_schema_sample.py --descriptions-only   # 只补数据库简述，不重建向量库
"""
import argparse
import logging

from znws.utils.schema_sampler import build_database_descriptions, build_schema_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="离线构建 schema 抽样向量库与数据库简述")
    parser.add_argument("--dataset", default=None, help="datasets.yaml 中登记的数据集名（缺省用配置的 default_dataset）")
    parser.add_argument("--split", default=None, help="切分名（缺省用该数据集的默认切分）")
    parser.add_argument("--db-ids", nargs="*", default=None, help="仅构建这些 db_id；缺省为该切分下全部")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-descriptions", action="store_true", help="跳过 LLM 数据库简述生成（不消耗 token）")
    mode.add_argument("--descriptions-only", action="store_true", help="只生成/更新数据库简述，不重建向量集合")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        if args.descriptions_only:
            path, descriptions = build_database_descriptions(
                dataset=args.dataset, split=args.split, db_ids=args.db_ids,
            )
            print(f"数据库简述: {path}（生成 {len(descriptions)} 条）")
            return 0 if descriptions else 1

        result = build_schema_sample(
            dataset=args.dataset,
            split=args.split,
            db_ids=args.db_ids,
            generate_descriptions=not args.skip_descriptions,
        )
        print(
            f"构建完成：{result['dataset']}/{result['split']}，"
            f"{len(result['collections'])} 个数据库集合，共 {result['documents']} 条文档。"
        )
        for db_id, collection in result["collections"].items():
            print(f"  - {db_id}: {collection.count()} 条")
        if not args.skip_descriptions:
            print(f"数据库简述: {result['descriptions_path']}")
        return 0
    except Exception as e:
        print(f"构建 Schema Sample 时发生错误: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
