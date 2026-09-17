"""Event streaming - the fully detailed, introspectable view of a run.

https://docs.langchain.com/oss/python/langchain/event-streaming

`.stream()` tells you about state. `.astream_events()` tells you about
*everything*: every model call, every tool call, every nested runnable, each
with a start event, stream events and an end event.

Every event is a dict with the same shape:

    {"event": "on_tool_start", "name": "get_weather", "run_id": ..., "data": {...},
     "tags": [...], "metadata": {...}, "parent_ids": [...]}

The names follow one pattern: `on_<type>_<start|stream|end>`, where type is
`chat_model`, `tool`, `chain`, `retriever` or `prompt`.

Reach for this when `.stream()` is not enough: tracing, debugging why a tool was
called, building a rich UI that shows tool activity as well as tokens, or
measuring where a slow run spent its time. It is async-only.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.callbacks import adispatch_custom_event

from langchain_project import config
from langchain_project.console import kv, note, section, title
from langchain_project.shared_tools import get_weather


@tool
async def check_flights(city: str, runtime: ToolRuntime) -> str:
    """Check whether flights to a city are delayed."""
    # A custom event surfaces as {"event": "on_custom_event", "name": "flight_progress"}.
    # Unlike stream_mode="custom", it carries the run tree with it, so you know
    # exactly which tool call produced it.
    await adispatch_custom_event("flight_progress", {"city": city, "stage": "querying airline"})
    return f"Flights to {city} are on time."


def _build_agent():
    return create_agent(
        model=config.get_model(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_weather", "args": {"city": "Tokyo"}, "id": "e1"},
                        {"name": "check_flights", "args": {"city": "Tokyo"}, "id": "e2"},
                    ],
                ),
                AIMessage(content="Tokyo is 22C and clear, and flights are on time."),
            ]
        ),
        tools=[get_weather, check_flights],
        system_prompt="You are a travel assistant.",
        name="travel_agent",
    )


async def _tour_of_events() -> None:
    section("Every event, as it happens")

    agent = _build_agent()
    question = {"messages": [("user", "Weather and flights for Tokyo?")]}

    seen: Counter[str] = Counter()
    tokens: list[str] = []

    async for event in agent.astream_events(question, version="v2"):
        kind = event["event"]
        seen[kind] += 1
        name = event["name"]

        # A chat UI generally cares about exactly these four.
        if kind == "on_chat_model_start":
            kv("model start", name)
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.text:
                tokens.append(chunk.text)
        elif kind == "on_tool_start":
            kv("tool start", f"{name}({event['data']['input']})")
        elif kind == "on_tool_end":
            kv("tool end", f"{name} -> {event['data']['output'].content}")
        elif kind == "on_custom_event":
            kv("custom event", f"{name}: {event['data']}")

    print()
    kv("streamed text", repr("".join(tokens)))

    section("What the run was made of")
    for kind, count in sorted(seen.items()):
        kv(kind, count)
    note("start/stream/end for every runnable in the tree, agent included")


async def _filtered() -> None:
    section("Filtering the firehose")

    # A real run emits hundreds of events. Filter server-side rather than
    # if-ing your way through them: include_types, include_names, include_tags
    # (and the exclude_* equivalents).
    agent = _build_agent()
    question = {"messages": [("user", "Weather and flights for Tokyo?")]}

    async for event in agent.astream_events(
        question, version="v2", include_types=["tool"]
    ):
        kv(event["event"], event["name"])

    note("include_types=['tool'] dropped every model and chain event")


async def _timing() -> None:
    section("Using events to find the slow part")

    import time

    agent = _build_agent()
    question = {"messages": [("user", "Weather and flights for Tokyo?")]}

    started: dict[str, float] = {}
    durations: dict[str, float] = {}

    async for event in agent.astream_events(question, version="v2"):
        # run_id is unique per runnable invocation, which is what makes pairing
        # a start event with its end event reliable.
        run_id = event["run_id"]
        if event["event"].endswith("_start"):
            started[run_id] = time.perf_counter()
        elif event["event"].endswith("_end") and run_id in started:
            label = f"{event['name']} ({event['event'].removesuffix('_end').removeprefix('on_')})"
            durations[label] = (time.perf_counter() - started[run_id]) * 1000

    for label, ms in sorted(durations.items(), key=lambda item: -item[1])[:6]:
        kv(label, f"{ms:.2f}ms")
    note("the same pairing is what tracing tools like LangSmith do for you")


def run() -> None:
    title("EVENT STREAMING")
    note(config.describe_provider())
    note("astream_events is async-only, so each section runs inside asyncio.run()")

    asyncio.run(_tour_of_events())
    asyncio.run(_filtered())
    asyncio.run(_timing())
