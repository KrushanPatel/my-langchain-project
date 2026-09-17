"""The demo registry, in the order it is worth reading them.

Each module is standalone: open one, read it top to bottom, run it, change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

DOCS = "https://docs.langchain.com/oss/python/langchain"


@dataclass(frozen=True)
class Demo:
    """One core component, one runnable module."""

    name: str
    module: str
    blurb: str
    doc: str

    def load(self) -> ModuleType:
        return import_module(f"{__name__}.{self.module}")

    def run(self) -> None:
        self.load().run()


DEMOS: tuple[Demo, ...] = (
    Demo(
        "models",
        "models",
        "invoke / batch / stream / bind_tools / with_structured_output",
        f"{DOCS}/models",
    ),
    Demo(
        "messages",
        "messages",
        "the four roles, content blocks, tool calls, usage, trimming",
        f"{DOCS}/messages",
    ),
    Demo(
        "tools",
        "tools",
        "@tool, args schemas, ToolRuntime, ToolException, BaseTool",
        f"{DOCS}/tools",
    ),
    Demo(
        "agents",
        "agents",
        "create_agent, the tool loop, custom state and context",
        f"{DOCS}/agents",
    ),
    Demo(
        "middleware",
        "middleware",
        "hooks around every step of the loop, plus the built-ins",
        f"{DOCS}/middleware/overview",
    ),
    Demo(
        "memory",
        "short_term_memory",
        "checkpointers, thread_id, inspecting and summarising history",
        f"{DOCS}/short-term-memory",
    ),
    Demo(
        "structured-output",
        "structured_output",
        "response_format, ToolStrategy, validated Pydantic results",
        f"{DOCS}/structured-output",
    ),
    Demo(
        "streaming",
        "streaming",
        "stream_mode: updates / messages / values / custom",
        f"{DOCS}/streaming",
    ),
    Demo(
        "event-streaming",
        "event_streaming",
        "astream_events: every model, tool and chain event",
        f"{DOCS}/event-streaming",
    ),
)

BY_NAME = {demo.name: demo for demo in DEMOS}
