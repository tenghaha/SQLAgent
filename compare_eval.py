"""评测报告对比 CLI：把多份 evaluate.py 产出的 eval_*.json 汇总为 Markdown 对比表。

只做表格汇总，不做额外分析；兼容缺少 Schema P/R 字段的旧版报告（显示 N/A）。

用法:
    python compare_eval.py outputs/mini_dev outputs/mini_dev_without_ss
    python compare_eval.py baseline=outputs/mini_dev/reports/eval_bird_mini_dev.json \
        no_ss=outputs/mini_dev_without_ss/reports/eval_bird_mini_dev.json
    python compare_eval.py outputs/mini_dev outputs/mini_dev_without_ss \
        --output outputs/reports/compare.md

参数既可以是评测报告 JSON 路径，也可以是实验输出目录（自动取目录下最新的
reports/eval_*.json）；用 name=path 可为实验自定义名称。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

DIFFICULTY_ORDER = ["simple", "moderate", "challenging", "unknown", "total"]
DEFAULT_OUTPUT = Path("outputs") / "reports" / "compare_eval.md"


def _latest_eval_file(directory: Path) -> Path:
    candidates = sorted(
        directory.glob("**/reports/eval_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            directory.glob("**/eval_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"目录中没有 eval_*.json: {directory}")
    return candidates[0]


def _default_name(path: Path) -> str:
    parent = path.parent
    if parent.name == "reports":
        candidate = parent.parent.name
        if candidate and candidate != "outputs":
            return candidate
    return path.stem


def _resolve_report(spec: str) -> Tuple[str, Path]:
    name, separator, path_text = spec.partition("=")
    if separator:
        label = name.strip() or None
        path = Path(path_text.strip())
    else:
        label = None
        path = Path(spec)

    if path.is_dir():
        path = _latest_eval_file(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到评测报告: {path}")
    return label or _default_name(path), path


def _load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"{path} 不是 evaluate.py 产出的评测报告")
    return report


def _fmt(value, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_ratio_pct(value) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _metric(report: dict, level: str) -> dict:
    return (report.get("metrics") or {}).get(level) or {}


def _levels(experiments: List[dict]) -> List[str]:
    available = {level for item in experiments for level in item["report"]["metrics"]}
    ordered = [level for level in DIFFICULTY_ORDER if level in available]
    return ordered + sorted(available - set(DIFFICULTY_ORDER))


def render_tables(experiments: List[dict]) -> str:
    """生成 Markdown 对比表；experiments 每项含 name / path / report。"""
    names = [item["name"] for item in experiments]
    lines: List[str] = ["# 评测结果对比", ""]

    # 实验来源
    lines += [
        "## 实验来源",
        "",
        "| 实验 | 报告文件 | 生成时间 | VES 迭代 | 执行超时 (s) | 评测并行度 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in experiments:
        report = item["report"]
        config = report.get("config") or {}
        lines.append(
            f"| {item['name']} | {item['path']} | {report.get('generated_at', 'N/A')} | "
            f"{config.get('ves_iterations', 'N/A')} | {_fmt(config.get('timeout'), 1)} | "
            f"{config.get('workers', 'N/A')} |"
        )

    # 总体指标
    lines += [
        "",
        "## 总体指标",
        "",
        "| 实验 | 数据集 / 切分 | 题数 | EX (%) | EX 题数 | VES | 平均执行耗时 (ms) | 可执行题数 | Schema 统计题数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in experiments:
        report = item["report"]
        total = _metric(report, "total")
        counts = report.get("counts") or {}
        pair = f"{report.get('dataset', 'N/A')} / {report.get('split', 'N/A')}"
        schema_count = total.get("schema_count", counts.get("schema_evaluated"))
        lines.append(
            f"| {item['name']} | {pair} | {_fmt(total.get('count'), 0)} | "
            f"{_fmt(total.get('ex'))} | {_fmt(total.get('ex_count'), 0)} | "
            f"{_fmt(total.get('ves'))} | {_fmt(total.get('avg_exec_ms'))} | "
            f"{_fmt(counts.get('executed'), 0)} | {_fmt(schema_count, 0)} |"
        )

    # 分难度指标
    lines += ["", "## 分难度指标", ""]
    header = ["| 难度 |"]
    divider = ["| --- |"]
    for name in names:
        header.append(f" {name} 题数 | {name} EX (%) | {name} VES | {name} 耗时 (ms) |")
        divider.append(" ---: | ---: | ---: | ---: |")
    lines.append("".join(header))
    lines.append("".join(divider))
    for level in _levels(experiments):
        cells = [f"| {level} |"]
        for item in experiments:
            metric = _metric(item["report"], level)
            cells.append(
                f" {_fmt(metric.get('count'), 0)} | {_fmt(metric.get('ex'))} | "
                f"{_fmt(metric.get('ves'))} | {_fmt(metric.get('avg_exec_ms'))} |"
            )
        lines.append("".join(cells))

    # Schema Linking
    lines += [
        "",
        "## Schema Linking（表 / 字段）",
        "",
        "| 实验 | Table P (micro) | Table R (micro) | Field P (micro) | Field R (micro) | "
        "Table P (macro) | Table R (macro) | Field P (macro) | Field R (macro) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in experiments:
        total = _metric(item["report"], "total")
        lines.append(
            f"| {item['name']} | {_fmt_ratio_pct(total.get('table_precision'))} | "
            f"{_fmt_ratio_pct(total.get('table_recall'))} | "
            f"{_fmt_ratio_pct(total.get('field_precision'))} | "
            f"{_fmt_ratio_pct(total.get('field_recall'))} | "
            f"{_fmt_ratio_pct(total.get('table_precision_macro'))} | "
            f"{_fmt_ratio_pct(total.get('table_recall_macro'))} | "
            f"{_fmt_ratio_pct(total.get('field_precision_macro'))} | "
            f"{_fmt_ratio_pct(total.get('field_recall_macro'))} |"
        )

    # 状态分布
    statuses = sorted(
        {status for item in experiments for status in (item["report"].get("status_counts") or {})}
    )
    if statuses:
        statuses.sort(
            key=lambda status: (
                -sum((item["report"].get("status_counts") or {}).get(status, 0) for item in experiments),
                status,
            )
        )
        lines += ["", "## 执行状态分布", ""]
        lines.append("| 状态 | " + " | ".join(names) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in names) + " |")
        for status in statuses:
            cells = [str((item["report"].get("status_counts") or {}).get(status, 0)) for item in experiments]
            lines.append(f"| {status} | " + " | ".join(cells) + " |")
        totals = [
            str(sum((item["report"].get("status_counts") or {}).values())) for item in experiments
        ]
        lines.append("| 合计 | " + " | ".join(totals) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比多份 evaluate.py 评测报告，产出 Markdown 表格")
    parser.add_argument("reports", nargs="+", help="报告路径或实验目录；可用 name=path 指定实验名")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="对比表输出路径")
    args = parser.parse_args()

    if len(args.reports) < 2:
        print("至少需要两个实验报告才能对比。")
        return 1

    experiments: List[dict] = []
    used_names: List[str] = []
    try:
        for spec in args.reports:
            name, path = _resolve_report(spec)
            if name in used_names:
                name = f"{name}-{len(used_names) + 1}"
            used_names.append(name)
            experiments.append({"name": name, "path": path, "report": _load_report(path)})
    except Exception as exc:
        print(f"读取评测报告失败: {type(exc).__name__}: {exc}")
        return 1

    pairs = {(item["report"].get("dataset"), item["report"].get("split")) for item in experiments}
    if len(pairs) > 1:
        print(f">>> [warn] 各实验的数据集/切分不一致: {sorted(pairs, key=str)}")

    markdown = render_tables(experiments)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\n对比表已写入: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
