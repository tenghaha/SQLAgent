"""运行记录：为每一次独立运行生成一份完整的思考过程与工具调用日志。

研究用途下需要可复现、可回溯的过程数据，因此这里同时提供：
1. 每run 一个人类可读的日志文件；
2. 结构化的步骤列表（供 Python API 直接消费，无需解析日志文本）。
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_LOG_DIR = "logs/sql_agent"

# 单条记录写入日志文件时保留的最大字符数，避免超长工具返回把日志撑爆
MAX_TEXT_CHARS = 20000


def flatten_content(content: Any) -> str:
    """把 LangChain 消息的 content 归一化成纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "reasoning":
                    continue
                text = block.get("text")
                parts.append(str(text) if text is not None else json.dumps(block, ensure_ascii=False, default=str))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False, default=str)


def extract_reasoning(message: Any) -> str:
    """提取模型的思考内容（思维链），兼容常见字段位置。"""
    kwargs = getattr(message, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return value

    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                text = block.get("reasoning") or block.get("text")
                if text:
                    parts.append(str(text))
        joined = "\n".join(parts)
        if joined.strip():
            return joined
    return ""


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return f"{text[:MAX_TEXT_CHARS]}\n...（已截断，原文 {len(text)} 字符）"


def _indent(text: str, prefix: str = "    ") -> str:
    if not text:
        return f"{prefix}(空)"
    return "\n".join(prefix + line for line in text.splitlines())


class RunRecorder:
    """记录一次独立运行的全部过程，并落盘为单个日志文件。"""

    def __init__(self, query: str, log_dir: str = DEFAULT_LOG_DIR, echo: bool = False, model: str = ""):
        self.run_id = uuid.uuid4().hex[:12]
        self.query = query
        self.model = model
        self.echo = echo
        self.started_at = datetime.now()
        self._t0 = time.perf_counter()
        self.steps: list[dict] = []

        self.log_path = Path(log_dir) / f"{self.started_at:%Y%m%d_%H%M%S}_{self.run_id}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", encoding="utf-8")
        self._write_header()

    # ------------------------------------------------------------------ 内部

    def _write(self, text: str = "") -> None:
        self._fh.write(text + "\n")
        self._fh.flush()
        if self.echo:
            print(text, flush=True)

    def _write_header(self) -> None:
        self._write("#" * 78)
        self._write("# ZNWS 简化版 SQL Agent · 运行记录")
        self._write(f"# run_id : {self.run_id}")
        self._write(f"# 开始时间: {self.started_at:%Y-%m-%d %H:%M:%S}")
        if self.model:
            self._write(f"# 模型   : {self.model}")
        self._write(f"# 查询   : {self.query}")
        self._write("#" * 78)
        self._write()

    def _append_step(self, step: dict) -> None:
        step.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        self.steps.append(step)

    # -------------------------------------------------------------- 对外接口

    def agent_start(self, agent_name: str, purpose: str = "") -> None:
        self._write("-" * 78)
        self._write(f"--- [{agent_name}] 开始 ---")
        if purpose:
            self._write(f"目的: {purpose}")
        self._write()

    def agent_end(self, agent_name: str, summary: str = "") -> None:
        self._write()
        self._write(f"--- [{agent_name}] 结束 ---")
        if summary:
            self._write(summary)
        self._write("-" * 78)
        self._write()

    def note(self, agent_name: str, text: str, echo_console: bool = True) -> None:
        """记录一条非模型产生的事件（如工具调用前的说明）。"""
        self._append_step({"agent": agent_name, "type": "note", "content": text})
        self._write(f"[{agent_name}] {text}")

    def record_messages(self, agent_name: str, messages: Iterable[Any]) -> None:
        """记录一批新产生的消息（模型输出 / 工具返回）。"""
        for message in messages or []:
            msg_type = getattr(message, "type", None) or message.__class__.__name__
            index = sum(1 for step in self.steps if step.get("agent") == agent_name) + 1

            if msg_type == "ai":
                tool_calls = []
                for call in getattr(message, "tool_calls", None) or []:
                    tool_calls.append({
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "args": call.get("args"),
                    })
                text = flatten_content(getattr(message, "content", ""))
                reasoning = extract_reasoning(message)
                self._append_step({
                    "agent": agent_name,
                    "type": "ai",
                    "index": index,
                    "reasoning": reasoning,
                    "content": text,
                    "tool_calls": tool_calls,
                })
                self._write(f"[{agent_name}][步骤 {index}] 模型输出")
                if reasoning:
                    self._write("  思考:")
                    self._write(_indent(_truncate(reasoning)))
                if text:
                    self._write("  内容:")
                    self._write(_indent(_truncate(text)))
                if tool_calls:
                    self._write("  工具调用:")
                    for call in tool_calls:
                        args = json.dumps(call.get("args"), ensure_ascii=False, default=str)
                        self._write(f"    - {call.get('name')}({args})")
                self._write()

            elif msg_type == "tool":
                tool_name = getattr(message, "name", "") or "unknown"
                result = flatten_content(getattr(message, "content", ""))
                self._append_step({
                    "agent": agent_name,
                    "type": "tool",
                    "index": index,
                    "tool_name": tool_name,
                    "tool_result": result,
                    "status": getattr(message, "status", None),
                })
                self._write(f"[{agent_name}][步骤 {index}] 工具返回 · {tool_name}")
                self._write("  结果:")
                self._write(_indent(_truncate(result)))
                self._write()

            else:
                text = flatten_content(getattr(message, "content", ""))
                self._append_step({
                    "agent": agent_name,
                    "type": msg_type,
                    "index": index,
                    "content": text,
                })
                self._write(f"[{agent_name}][步骤 {index}] {msg_type} 消息")
                self._write(_indent(_truncate(text)))
                self._write()

    def finish(self, payload: Optional[dict] = None) -> None:
        """写入运行结论并关闭日志。"""
        elapsed = time.perf_counter() - self._t0
        self._write()
        self._write("#" * 78)
        self._write("# 运行结束")
        self._write(f"# run_id  : {self.run_id}")
        self._write(f"# 总用时  : {elapsed:.1f}s")
        self._write(f"# 步骤总数: {len(self.steps)}")
        if payload:
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                self._write(f"# {key}: {value}")
        self._write("#" * 78)
        self._fh.close()

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._t0
