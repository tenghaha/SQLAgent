# 数据库与系统全局配置（BIRD 数据集）

1. **数据库类型 (Database Type)**: SQLite
2. **数据集 (Dataset)**: BIRD（每道题对应一个独立的 `db_id` 数据库；表结构见 `<split>_tables.json`，列说明见对应库的 `database_description/*.csv`）
3. **时间日期格式 (Datetime Format)**: 各列格式不统一，具体以 `database_description/*.csv` 的 `data_format` 列为准，请使用 SQLite 的 `STRFTIME` / `DATE` 等函数处理，不要假定固定格式。

**注意**：所有 Agent 在进行实体提取、SQL 生成和 SQL 审查时，必须严格遵循以上约束。
