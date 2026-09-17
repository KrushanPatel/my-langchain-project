"""Tools - giving the model something it can actually do.

https://docs.langchain.com/oss/python/langchain/tools

A tool is a Python function plus a schema the model can read. The model never
executes anything: it emits a request ("call `get_weather` with city=Tokyo"),
and the agent runs the function and feeds the result back as a `ToolMessage`.

That means the *description and argument names are prompt text*. Vague
docstrings produce a model that picks the wrong tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain.agents import AgentState
from langchain.tools import BaseTool, ToolException, ToolRuntime, tool
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from langchain_project import config
from langchain_project.console import kv, note, section, title

# =====================================================================
# 1. The decorator: the way you will write 95% of your tools
# =====================================================================


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city, e.g. "London".
    """
    return {"london": "13C and raining", "tokyo": "22C and clear"}.get(
        city.strip().lower(), f"No forecast on file for {city}."
    )


# =====================================================================
# 2. An explicit Pydantic schema, for when types alone are not enough
# =====================================================================


class SearchInput(BaseModel):
    """Arguments for the note search tool."""

    query: str = Field(description="Words to search for, case-insensitive")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum results to return")


_NOTES = [
    "LangChain v1 centres on create_agent.",
    "Middleware hooks wrap every step of the agent loop.",
    "Checkpointers give an agent short-term memory.",
    "Tools are functions plus a schema the model can read.",
]


@tool("search_notes", args_schema=SearchInput)
def search_notes(query: str, limit: int = 3) -> str:
    """Search the user's saved notes for a phrase."""
    hits = [note for note in _NOTES if query.lower() in note.lower()][:limit]
    return "\n".join(hits) if hits else "No matching notes."


# =====================================================================
# 3. ToolRuntime: reaching the surrounding execution from inside a tool
# =====================================================================


@dataclass
class UserContext:
    """Per-run data the agent carries but never shows the model."""

    user_name: str = "friend"


@tool
def count_conversation(runtime: ToolRuntime[UserContext, AgentState]) -> str:
    """Report how many messages this conversation contains so far."""
    # Any parameter annotated `ToolRuntime` is injected by the agent and hidden
    # from the model - it never appears in the schema the model sees.
    total = len(runtime.state["messages"])
    user = runtime.context.user_name if runtime.context else "friend"
    return f"{user}, this conversation holds {total} messages (tool call {runtime.tool_call_id})."


# =====================================================================
# 4. Failure: ToolException is the polite way for a tool to give up
# =====================================================================


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        # Raising ToolException turns into a ToolMessage the model can read and
        # recover from, instead of an exception that kills the run.
        raise ToolException("Cannot divide by zero - ask the user for a different divisor.")
    return a / b


# =====================================================================
# 5. BaseTool: the class form, for tools that need state or setup
# =====================================================================


class CounterTool(BaseTool):
    """A tool that remembers how many times it has been called.

    Subclass `BaseTool` when a plain function will not do - when the tool owns a
    connection, a cache, or (as here) mutable state.
    """

    name: str = "call_counter"
    description: str = "Return how many times this tool has been called in this process."
    call_count: int = 0

    def _run(self, *args, **kwargs) -> str:
        self.call_count += 1
        return f"This tool has now been called {self.call_count} time(s)."


def run() -> None:
    title("TOOLS")

    # -----------------------------------------------------------------
    section("What @tool actually builds")
    # -----------------------------------------------------------------
    # The decorator turns the function into a `BaseTool`, taking the name from
    # the function and the description from the docstring.
    kv("name", get_weather.name)
    kv("description", get_weather.description.splitlines()[0])
    kv("args", get_weather.args)
    note("this schema is what gets sent to the model - write it for a reader")

    # -----------------------------------------------------------------
    section("Calling a tool directly")
    # -----------------------------------------------------------------
    # A tool is a Runnable. Invoked with a plain dict you get the raw return
    # value; invoked with a tool-call dict you get a ready-made ToolMessage.
    kv("plain invoke", get_weather.invoke({"city": "Tokyo"}))

    as_tool_message = get_weather.invoke(
        {"name": "get_weather", "args": {"city": "London"}, "id": "call_abc", "type": "tool_call"}
    )
    kv("returned", type(as_tool_message).__name__)
    kv("content", as_tool_message.content)
    kv("tool_call_id", as_tool_message.tool_call_id)

    # -----------------------------------------------------------------
    section("An explicit args_schema")
    # -----------------------------------------------------------------
    # Use one when you want defaults, constraints or better descriptions than
    # type hints can express. Validation happens before your function runs.
    schema = search_notes.args_schema.model_json_schema()
    kv("json schema", json.dumps(schema["properties"], indent=None))
    kv("search result", search_notes.invoke({"query": "middleware"}))

    try:
        search_notes.invoke({"query": "agents", "limit": 99})  # limit is capped at 10
    except Exception as exc:  # noqa: BLE001 - showing the validation error is the point
        note(f"limit=99 rejected before the function ran: {type(exc).__name__}")

    # -----------------------------------------------------------------
    section("A stateful BaseTool")
    # -----------------------------------------------------------------
    counter = CounterTool()
    kv("first call", counter.invoke({}))
    kv("second call", counter.invoke({}))

    # -----------------------------------------------------------------
    section("ToolException, inside a real agent loop")
    # -----------------------------------------------------------------
    # By default a tool exception propagates and kills the run - LangChain will
    # not quietly show internal errors to the model. Opt in with
    # `ToolErrorMiddleware`: it converts the exception into an error
    # `ToolMessage` and lets the model try again.
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolErrorMiddleware

    model = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "divide", "args": {"a": 10, "b": 0}, "id": "d1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "divide", "args": {"a": 10, "b": 2}, "id": "d2"}],
            ),
            AIMessage(content="10 divided by 2 is 5. Dividing by zero is undefined."),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[divide],
        middleware=[
            ToolErrorMiddleware(
                # (exception, request) -> text the model sees, or None to re-raise
                on_error=lambda exc, request: f"{request.tool_call['name']} failed: {exc}"
            )
        ],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "What is 10 / 0?"}]})
    for message in result["messages"]:
        message.pretty_print()

    # -----------------------------------------------------------------
    section("ToolRuntime, inside a real agent loop")
    # -----------------------------------------------------------------
    runtime_model = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "count_conversation", "args": {}, "id": "c1"}],
            ),
            AIMessage(content="I checked - the details are above."),
        ]
    )
    # `context_schema` declares the shape of per-run data that is *not* part of
    # the conversation - who the user is, which tenant, which feature flags.
    runtime_agent = create_agent(
        model=runtime_model,
        tools=[count_conversation],
        context_schema=UserContext,
    )
    runtime_result = runtime_agent.invoke(
        {"messages": [{"role": "user", "content": "How long is this conversation?"}]},
        context=UserContext(user_name="Krushan"),
    )
    kv("tool said", runtime_result["messages"][-2].content)
    note("state, context and tool_call_id were injected - the model never saw them")
