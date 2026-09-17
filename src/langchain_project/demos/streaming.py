"""Streaming - watching an agent work instead of waiting for it.

https://docs.langchain.com/oss/python/langchain/streaming

`.invoke()` blocks until the whole run is done. `.stream()` hands you pieces as
they happen. Which pieces depends on `stream_mode`:

    "updates"  what each node changed, as it finishes     <- progress feed
    "messages" model tokens as they are generated         <- the typing effect
    "values"   the entire state after every step          <- debugging
    "custom"   whatever your own tools choose to emit     <- progress from tools

You can ask for several at once by passing a list, in which case each item
arrives as `(mode, payload)`.

Note the two different granularities: "updates" fires once per *node*, while
"messages" fires once per *token*. Most chat UIs use both - "messages" for the
answer, "updates" for the "searching..." indicator.
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk
from langchain.tools import ToolRuntime, tool

from langchain_project import config
from langchain_project.console import kv, note, section, title
from langchain_project.shared_tools import get_weather


@tool
def slow_lookup(topic: str, runtime: ToolRuntime) -> str:
    """Look something up, reporting progress as it goes."""
    for step in ("connecting", "querying", "formatting"):
        # Anything written here surfaces under stream_mode="custom". This is
        # how a long-running tool tells the UI it is still alive.
        runtime.stream_writer({"tool": "slow_lookup", "status": step})
    return f"Everything worth knowing about {topic}."


def _build_agent():
    """A fresh agent per demo section - the script is consumed as it runs."""
    return create_agent(
        model=config.get_model(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "s1"}],
                ),
                AIMessage(content="Tokyo is 22C and clear today - a good day to be outside."),
            ]
        ),
        tools=[get_weather],
        system_prompt="You are a weather assistant.",
    )


def run() -> None:
    title("STREAMING")
    note(config.describe_provider())

    question = {"messages": [("user", "What's the weather in Tokyo?")]}

    # -----------------------------------------------------------------
    section('stream_mode="updates" - one event per node, as it finishes')
    # -----------------------------------------------------------------
    # Each chunk is {node_name: state_update}. This is the progress feed.
    for chunk in _build_agent().stream(question, stream_mode="updates"):
        for node, update in chunk.items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for message in messages:
                summary = message.text or f"tool call -> {message.tool_calls[0]['name']}"
                kv(node, summary[:64])

    # -----------------------------------------------------------------
    section('stream_mode="messages" - token by token')
    # -----------------------------------------------------------------
    # Each item is (chunk, metadata). The metadata tells you which node the
    # token came from, which matters once you have several models in one graph.
    print("   ", end="")
    final: AIMessageChunk | None = None
    for chunk, metadata in _build_agent().stream(question, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            print(chunk.text, end="", flush=True)
            final = chunk if final is None else final + chunk
    print()
    kv("node", metadata.get("langgraph_node"))
    kv("reassembled", repr(final.text))

    # -----------------------------------------------------------------
    section('stream_mode="values" - the whole state, every step')
    # -----------------------------------------------------------------
    # Verbose, but unbeatable when you want to see exactly how state evolves.
    for step, state in enumerate(_build_agent().stream(question, stream_mode="values")):
        kv(f"step {step}", f"{len(state['messages'])} message(s) in state")

    # -----------------------------------------------------------------
    section('stream_mode="custom" - progress from inside a tool')
    # -----------------------------------------------------------------
    custom_agent = create_agent(
        model=config.get_model(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "slow_lookup", "args": {"topic": "LangChain"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="Here's what I found."),
            ]
        ),
        tools=[slow_lookup],
    )
    for payload in custom_agent.stream(
        {"messages": [("user", "Tell me about LangChain.")]}, stream_mode="custom"
    ):
        kv("custom event", payload)

    # -----------------------------------------------------------------
    section("Several modes at once")
    # -----------------------------------------------------------------
    # With a list of modes each item arrives tagged: (mode, payload). This is
    # what a real chat endpoint streams - status updates and tokens together.
    for mode, payload in _build_agent().stream(question, stream_mode=["updates", "messages"]):
        if mode == "updates":
            kv("updates", list(payload.keys()))
        else:
            chunk, _metadata = payload
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                kv("token", repr(chunk.text))

    # -----------------------------------------------------------------
    section("The async version")
    # -----------------------------------------------------------------
    # `astream` is the same API with `async for`. Use it inside any async
    # server - see server.py for the FastAPI version.
    async def stream_async() -> str:
        pieces: list[str] = []
        async for chunk, _metadata in _build_agent().astream(question, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                pieces.append(chunk.text)
        return "".join(pieces)

    kv("astream result", repr(asyncio.run(stream_async())))
