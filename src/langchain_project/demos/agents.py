"""Agents - the loop that keeps calling the model until it stops asking for tools.

https://docs.langchain.com/oss/python/langchain/agents

`create_agent` builds a small graph:

    START -> model -> (tool calls?) -> tools -> model -> ... -> END

That is genuinely the whole idea. The model either answers (loop ends) or asks
for tools (they run, results go back as `ToolMessage`s, model is called again).

`create_agent` returns a compiled LangGraph graph, so it is a Runnable: it has
`.invoke()`, `.stream()`, `.astream_events()` and accepts a `config` like
anything else in LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.messages import AIMessage, SystemMessage
from langchain.tools import ToolRuntime, tool

from langchain_project import config
from langchain_project.console import kv, note, section, title
from langchain_project.shared_tools import add, get_weather, multiply


@dataclass
class TripContext:
    """Static, per-run data. Not part of the conversation, not model-writable."""

    traveller: str
    home_city: str


class TripState(AgentState):
    """The agent's state, widened with a field of our own.

    `AgentState` already carries `messages` (and `structured_response`).
    Subclass it to track anything else the run needs. Fields are `NotRequired`
    unless the caller must always supply them.
    """

    cities_checked: NotRequired[list[str]]


@tool
def remember_city(city: str, runtime: ToolRuntime[TripContext, TripState]) -> str:
    """Record that the traveller is interested in a city."""
    # Returning a dict from a tool is not how you write state - tools write
    # state by returning a Command, or (as here) you keep it simple and let
    # middleware do the writing. This tool just reads.
    already = runtime.state.get("cities_checked", [])
    return f"Noted {city}. Cities checked so far: {', '.join(already) or 'none'}."


def run() -> None:
    title("AGENTS")
    note(config.describe_provider())

    # -----------------------------------------------------------------
    section("The smallest useful agent")
    # -----------------------------------------------------------------
    model = config.get_model(
        script=[
            # Turn 1: the model wants a tool. Note it can ask for several at
            # once - the agent runs them in parallel.
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Tokyo"}, "id": "w1"},
                    {"name": "get_weather", "args": {"city": "London"}, "id": "w2"},
                ],
            ),
            # Turn 2: it has what it needs, so it answers and the loop ends.
            AIMessage(content="Tokyo is 22C and clear; London is 13C and raining."),
        ]
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, add, multiply],
        system_prompt="You are a travel assistant. Use tools rather than guessing.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Compare the weather in Tokyo and London."}]}
    )

    note("the full transcript the loop produced:")
    print()
    for message in result["messages"]:
        message.pretty_print()

    note(f"{len(result['messages'])} messages: 1 human, 1 AI w/ tool calls, 2 tool, 1 AI answer")
    kv("final answer", result["messages"][-1].text)

    # -----------------------------------------------------------------
    section("What invoke() returns")
    # -----------------------------------------------------------------
    # The return value is the agent's *state*, not just the reply. `messages`
    # holds the whole transcript; that is what you persist between turns.
    kv("state keys", list(result.keys()))

    # -----------------------------------------------------------------
    section("A system prompt can be a SystemMessage too")
    # -----------------------------------------------------------------
    # Useful when you want to attach metadata or reuse a prepared message.
    # For a prompt that changes per run, use @dynamic_prompt middleware - see
    # the middleware demo.
    typed_prompt_agent = create_agent(
        model=config.get_model(script=[AIMessage(content="Understood.")]),
        tools=[],
        system_prompt=SystemMessage("You answer in exactly one word."),
    )
    kv("reply", typed_prompt_agent.invoke({"messages": [("user", "Ready?")]})["messages"][-1].text)

    # -----------------------------------------------------------------
    section("Custom state and runtime context")
    # -----------------------------------------------------------------
    # state_schema   - data that evolves during the run and is checkpointed
    # context_schema - data fixed for the run: user id, tenant, permissions
    context_model = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "remember_city", "args": {"city": "Kyoto"}, "id": "r1"}],
            ),
            AIMessage(content="I've noted Kyoto for you."),
        ]
    )
    context_agent = create_agent(
        model=context_model,
        tools=[remember_city],
        state_schema=TripState,
        context_schema=TripContext,
    )
    context_result = context_agent.invoke(
        {
            "messages": [{"role": "user", "content": "Add Kyoto to my list."}],
            "cities_checked": ["Tokyo", "London"],
        },
        context=TripContext(traveller="Krushan", home_city="Ahmedabad"),
    )
    kv("tool saw state", context_result["messages"][-2].content)
    kv("cities_checked", context_result["cities_checked"])

    # -----------------------------------------------------------------
    section("Stopping a runaway loop")
    # -----------------------------------------------------------------
    # A model that keeps requesting tools would loop forever. Two guards:
    #   - ModelCallLimitMiddleware: a budget on model calls
    #   - recursion_limit in config: LangGraph's hard backstop on graph steps
    looping_model = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": i, "b": 1}, "id": f"a{i}"}],
            )
            for i in range(10)
        ]
    )
    guarded = create_agent(
        model=looping_model,
        tools=[add],
        middleware=[ModelCallLimitMiddleware(thread_limit=3, exit_behavior="end")],
    )
    guarded_result = guarded.invoke({"messages": [("user", "Keep adding one.")]})
    tool_rounds = sum(
        1 for m in guarded_result["messages"] if isinstance(m, AIMessage) and m.tool_calls
    )
    kv("tool-calling rounds", f"{tool_rounds} (thread_limit=3)")
    kv("how it ended", guarded_result["messages"][-1].text)
    note("without a cap this script would have looped ten times")
