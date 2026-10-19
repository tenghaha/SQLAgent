"""两个 ReAct agent 共用的执行封装：流式运行 + 过程记录 + 结构化输出。"""
from typing import Any, Optional, Tuple, Type

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .tracing import RunRecorder, flatten_content


def run_react_agent(
    *,
    agent_name: str,
    model: BaseChatModel,
    tools: list,
    system_prompt: str,
    user_input: str,
    recorder: RunRecorder,
    response_format: Optional[Type[BaseModel]] = None,
    recursion_limit: int = 30,
) -> Tuple[Optional[BaseModel], str, dict]:
    """运行一个 ReAct agent 并完整记录过程。

    返回 (结构化结果, 最终文本, 最终状态)。
    structured_response 由 create_agent 的 response_format 机制产出，
    模型在结束前会调用结构化输出工具，因此无需解析 Markdown。
    """
    agent = create_agent(
        model,
        tools,
        system_prompt=system_prompt,
        response_format=response_format,
    )

    config = {"recursion_limit": recursion_limit}
    final_state: dict[str, Any] = {}
    recorded = 0

    # 使用 values 模式：每步都能拿到完整状态，便于按增量记录消息差异
    for state in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    ):
        final_state = state
        messages = state.get("messages") or []
        recorder.record_messages(agent_name, messages[recorded:])
        recorded = len(messages)

    structured = final_state.get("structured_response") if response_format else None

    final_text = ""
    for message in reversed(final_state.get("messages") or []):
        if getattr(message, "type", "") == "ai":
            final_text = flatten_content(getattr(message, "content", ""))
            break

    return structured, final_text, final_state
