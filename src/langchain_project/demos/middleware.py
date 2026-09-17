"""Middleware - hooks around every step of the agent loop.

https://docs.langchain.com/oss/python/langchain/middleware/overview

`create_agent` gives you the loop. Middleware is how you change what happens at
each point in it without forking the loop itself:

    before_agent      once, before the run starts
      before_model      before each model call - edit state
        wrap_model_call   around each model call - swap model/prompt/tools, retry
      after_model       after each model call - inspect or rewrite the reply
        wrap_tool_call    around each tool call - retry, cache, veto
    after_agent       once, when the run finishes

Middleware nests like an onion: the first item in the list is the outermost
layer, so it sees the request first and the response last.

Two ways to write one:
  * a decorator on a function - for a single hook (most of the time)
  * a subclass of `AgentMiddleware` - when one concern needs several hooks or
    its own state
"""

from __future__ import annotations

import time
from typing import Any, Callable

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from langchain_project import config
from langchain_project.console import kv, note, section, title
from langchain_project.shared_tools import get_weather

# =====================================================================
# Decorator form - one function, one hook
# =====================================================================


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Runs before every model call.

    Return a dict to update state, or None to change nothing.
    """
    print(f"      [before_model] about to call the model with {len(state['messages'])} messages")
    return None


@dynamic_prompt
def prompt_for_time_of_day(request: ModelRequest) -> str:
    """Builds the system prompt fresh on every model call.

    Use this instead of a static `system_prompt=` whenever the instructions
    depend on state, the user, or the world.
    """
    hour = time.localtime().tm_hour
    greeting = "Good morning" if hour < 12 else "Good afternoon"
    return f"{greeting}. You are a concise weather assistant. Never invent a forecast."


@wrap_model_call
def time_the_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Wraps the model call, so it sees both sides.

    `handler(request)` is the rest of the stack plus the model itself. Anything
    before that line happens on the way in, anything after on the way out.
    You can also call `handler` more than once - that is how retry middleware
    is built.
    """
    started = time.perf_counter()
    response = handler(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"      [wrap_model_call] model answered in {elapsed_ms:.1f}ms")
    return response


@wrap_tool_call
def veto_unknown_cities(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Wraps each tool call - the place to add caching, retries, or a veto.

    Returning a `ToolMessage` without calling `handler` short-circuits the tool:
    it never runs, but the model still gets an answer.
    """
    city = request.tool_call["args"].get("city", "")
    if city.lower() in {"atlantis", "narnia"}:
        print(f"      [wrap_tool_call] blocked a lookup for {city!r}")
        return ToolMessage(
            content=f"{city} is not a real place; ask the user to name a real city.",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    return handler(request)


# =====================================================================
# Class form - one concern, several hooks, its own state
# =====================================================================


class RunSummaryMiddleware(AgentMiddleware):
    """Counts model and tool calls across a run and prints a summary at the end.

    A subclass is the right shape here because the two hooks share state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_calls = 0
        self.tool_calls = 0

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self.model_calls = 0
        self.tool_calls = 0
        print("      [before_agent] run starting")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self.model_calls += 1
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            self.tool_calls += len(last.tool_calls)
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(
            f"      [after_agent] run finished: {self.model_calls} model call(s), "
            f"{self.tool_calls} tool call(s)"
        )
        return None


def run() -> None:
    title("MIDDLEWARE")
    note(config.describe_provider())

    # -----------------------------------------------------------------
    section("All of the above, stacked on one agent")
    # -----------------------------------------------------------------
    model = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Atlantis"}, "id": "m1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "m2"}],
            ),
            AIMessage(content="Atlantis isn't real, but Tokyo is 22C and clear."),
        ]
    )

    agent = create_agent(
        model=model,
        tools=[get_weather],
        # Order matters: first in the list is the outermost layer.
        middleware=[
            RunSummaryMiddleware(),
            log_before_model,
            prompt_for_time_of_day,
            time_the_model,
            veto_unknown_cities,
        ],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Weather in Atlantis and Tokyo?"}]}
    )
    print()
    kv("final answer", result["messages"][-1].text)
    note("the vetoed tool call never reached get_weather, but the model still got a reply")

    # -----------------------------------------------------------------
    section("Built-in middleware worth knowing")
    # -----------------------------------------------------------------
    # You rarely need to write these yourself:
    builtins = {
        "SummarizationMiddleware": "compresses old messages once history grows too long",
        "ModelCallLimitMiddleware": "caps model calls per run or per thread",
        "ToolCallLimitMiddleware": "caps tool calls, overall or per tool",
        "ToolErrorMiddleware": "turns chosen tool exceptions into error ToolMessages",
        "ToolRetryMiddleware": "retries a flaky tool with backoff",
        "ModelRetryMiddleware": "retries the model call itself",
        "ModelFallbackMiddleware": "falls back to another model when the first fails",
        "PIIMiddleware": "redacts emails, cards and the like before they reach the model",
        "HumanInTheLoopMiddleware": "pauses for approval before sensitive tool calls",
        "TodoListMiddleware": "gives the agent a scratchpad for multi-step plans",
        "LLMToolSelectorMiddleware": "narrows a large toolset before each model call",
        "ContextEditingMiddleware": "prunes old tool output to reclaim context",
    }
    for name, purpose in builtins.items():
        kv(name, purpose)

    # -----------------------------------------------------------------
    section("Configuring a few of them")
    # -----------------------------------------------------------------
    production_agent = create_agent(
        model=config.get_model(
            script=[AIMessage(content="My email is [REDACTED_EMAIL] - here to help.")]
        ),
        tools=[get_weather],
        middleware=[
            # Redact anything that looks like an email before the model sees it.
            PIIMiddleware("email", strategy="redact"),
            # Hard budgets, so a bad loop costs you three calls, not three hundred.
            ModelCallLimitMiddleware(thread_limit=10, run_limit=5),
            ToolCallLimitMiddleware(thread_limit=10),
            # Keep the transcript inside the context window automatically.
            SummarizationMiddleware(
                model=config.get_model(script=[AIMessage(content="Earlier: user asked about X.")]),
                trigger=("messages", 20),
                keep=("messages", 6),
            ),
        ],
    )
    pii_result = production_agent.invoke(
        {"messages": [{"role": "user", "content": "Mail me at krushan@example.com"}]}
    )
    kv("what the model saw", pii_result["messages"][0].text)
    note("PIIMiddleware rewrote the human message in place before the model call")
