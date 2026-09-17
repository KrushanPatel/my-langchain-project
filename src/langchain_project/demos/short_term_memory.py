"""Short-term memory - how an agent remembers the previous turn.

https://docs.langchain.com/oss/python/langchain/short-term-memory

A model is stateless. Every call sees only the messages you send it. "Memory",
at this level, means: save the transcript after each turn and replay it on the
next one.

A **checkpointer** does that for you. Attach one to the agent, pass a
`thread_id` in the config, and each `invoke` picks up where that thread left
off. One thread = one conversation.

`InMemorySaver` keeps it in a dict and forgets everything when the process
exits - perfect for learning, useless in production. Swap in the SQLite or
Postgres saver (`langgraph-checkpoint-sqlite` / `-postgres`) and nothing else in
your code changes.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_project import config
from langchain_project.console import kv, note, section, title
from langchain_project.shared_tools import get_weather


def run() -> None:
    title("SHORT-TERM MEMORY")
    note(config.describe_provider())

    # -----------------------------------------------------------------
    section("Without a checkpointer: every call starts from nothing")
    # -----------------------------------------------------------------
    forgetful = create_agent(
        model=config.get_model(
            script=[
                AIMessage(content="Nice to meet you, Krushan."),
                AIMessage(content="I'm afraid I don't know your name."),
            ]
        ),
        tools=[],
    )
    forgetful.invoke({"messages": [("user", "Hi, my name is Krushan.")]})
    second = forgetful.invoke({"messages": [("user", "What's my name?")]})
    kv("turn 2 reply", second["messages"][-1].text)
    kv("messages held", len(second["messages"]))
    note("only the one question was sent - the first turn was never stored")

    # -----------------------------------------------------------------
    section("With a checkpointer: the thread carries the history")
    # -----------------------------------------------------------------
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=config.get_model(
            script=[
                AIMessage(content="Nice to meet you, Krushan."),
                AIMessage(content="Your name is Krushan."),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "t1"}],
                ),
                AIMessage(content="Krushan, Tokyo is 22C and clear."),
                # 5th call: the new thread below, which starts with no history.
                AIMessage(content="I don't think you've told me your name yet."),
            ]
        ),
        tools=[get_weather],
        system_prompt="You are a friendly assistant with a good memory.",
        checkpointer=checkpointer,
    )

    # The thread_id is the conversation's identity. Same id, same history.
    thread = {"configurable": {"thread_id": "krushan-session-1"}}

    for turn in (
        "Hi, my name is Krushan.",
        "What's my name?",
        "And what's the weather in Tokyo?",
    ):
        reply = agent.invoke({"messages": [("user", turn)]}, config=thread)
        kv("user", turn)
        kv("agent", reply["messages"][-1].text)
        print()

    note("each invoke sent only the new message - the checkpointer replayed the rest")

    # -----------------------------------------------------------------
    section("Inspecting what was stored")
    # -----------------------------------------------------------------
    # `get_state` reads the thread back without running the agent. This is what
    # you call to render a chat history in a UI.
    snapshot = agent.get_state(thread)
    kv("messages on thread", len(snapshot.values["messages"]))
    for message in snapshot.values["messages"]:
        label = type(message).__name__.replace("Message", "")
        kv(label, (message.text or "<tool call>")[:60])

    kv("checkpoints saved", sum(1 for _ in agent.get_state_history(thread)))
    note("every step is a checkpoint, so you can rewind or branch a conversation")

    # -----------------------------------------------------------------
    section("A different thread_id is a different conversation")
    # -----------------------------------------------------------------
    other_thread = {"configurable": {"thread_id": "someone-else"}}
    other = agent.invoke({"messages": [("user", "What's my name?")]}, config=other_thread)
    kv("reply on new thread", other["messages"][-1].text)
    kv("messages on new thread", len(other["messages"]))
    note("isolation is per thread_id - that is how one agent serves many users")

    # -----------------------------------------------------------------
    section("Keeping a long thread inside the context window")
    # -----------------------------------------------------------------
    # Memory grows without limit; context windows do not. SummarizationMiddleware
    # watches the transcript and, past the trigger, replaces the older messages
    # with a model-written summary while keeping the most recent ones verbatim.
    summarising_agent = create_agent(
        model=config.get_model(script=[AIMessage(content="Got it.")]),
        tools=[],
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=config.get_model(
                    script=[AIMessage(content="Summary: the user counted from 1 to 12.")]
                ),
                trigger=("messages", 8),   # summarise once the thread passes 8 messages
                keep=("messages", 4),      # always keep the last 4 verbatim
            )
        ],
    )
    long_thread = {"configurable": {"thread_id": "long-one"}}
    for i in range(1, 13):
        summarising_agent.invoke({"messages": [("user", f"Message {i}")]}, config=long_thread)

    final = summarising_agent.get_state(long_thread)
    kv("turns sent", 12)
    kv("messages retained", len(final.values["messages"]))
    note("older turns were folded into a summary instead of being replayed in full")
