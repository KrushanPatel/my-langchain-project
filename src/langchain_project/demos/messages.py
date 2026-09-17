"""Messages - the currency every model call is denominated in.

https://docs.langchain.com/oss/python/langchain/messages

A conversation is a list of messages. Four roles do almost all the work:

    SystemMessage  standing instructions for the model
    HumanMessage   what the user said
    AIMessage      what the model said - including any tool calls it wants
    ToolMessage    the result of running a tool, fed back to the model

The important habit to build: a tool call is not a side channel. The model's
request lands in an `AIMessage.tool_calls`, and the answer goes back as a
`ToolMessage` carrying the same `tool_call_id`. It is all just messages.
"""

from __future__ import annotations

from langchain.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from langchain_project.console import kv, note, section, title


def run() -> None:
    title("MESSAGES")

    # -----------------------------------------------------------------
    section("The four roles")
    # -----------------------------------------------------------------
    conversation = [
        SystemMessage("You are a helpful weather assistant."),
        HumanMessage("What's the weather in Tokyo?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_001"}
            ],
        ),
        ToolMessage(content="22C and clear", tool_call_id="call_001", name="get_weather"),
        AIMessage(
            content="It's 22C and clear in Tokyo.",
            usage_metadata={"input_tokens": 52, "output_tokens": 9, "total_tokens": 61},
        ),
    ]

    for message in conversation:
        kv(type(message).__name__, repr(message.text)[:70])

    note("pretty_print() gives the readable view you want while debugging:")
    print()
    for message in conversation:
        message.pretty_print()

    # -----------------------------------------------------------------
    section(".text vs .content vs .content_blocks")
    # -----------------------------------------------------------------
    # `.content` is the raw provider payload: sometimes a plain string,
    # sometimes a list of blocks.
    # `.text` always gives you the text, flattened - reach for this one.
    # `.content_blocks` always gives you a normalised list of typed blocks
    # (text, reasoning, images, citations) no matter which provider produced
    # the message.
    answer = conversation[-1]
    kv(".content", repr(answer.content))
    kv(".text", repr(answer.text))
    kv(".content_blocks", answer.content_blocks)

    # -----------------------------------------------------------------
    section("Reading a tool call off an AIMessage")
    # -----------------------------------------------------------------
    request = conversation[2]
    for call in request.tool_calls:
        kv("tool name", call["name"])
        kv("arguments", call["args"])
        kv("id", call["id"])
    note("the matching ToolMessage quotes that id back, pairing result to request")

    # -----------------------------------------------------------------
    section("Token usage")
    # -----------------------------------------------------------------
    # Providers that report usage attach it to the AIMessage. This is how you
    # measure cost without wrapping every call yourself.
    kv("usage_metadata", answer.usage_metadata)

    # -----------------------------------------------------------------
    section("Chunks add up into a message")
    # -----------------------------------------------------------------
    # Streaming produces `AIMessageChunk`s. They implement `+`, and summing an
    # entire stream reconstructs exactly the message `.invoke()` would return.
    chunks = [
        AIMessageChunk(content="It's "),
        AIMessageChunk(content="22C "),
        AIMessageChunk(content="in Tokyo."),
    ]
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    kv("summed chunks", repr(total.text))
    kv("result type", type(total).__name__)

    # -----------------------------------------------------------------
    section("Multimodal input is just another content block")
    # -----------------------------------------------------------------
    # Text is the common case, but a HumanMessage's content can be a list of
    # typed blocks. LangChain translates these into each provider's own format.
    image_question = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image", "url": "https://example.com/cat.png"},
        ]
    )
    kv("blocks", len(image_question.content_blocks))
    kv("text part", repr(image_question.text))

    # -----------------------------------------------------------------
    section("trim_messages - keeping a history under a budget")
    # -----------------------------------------------------------------
    # Long conversations eventually exceed the context window. `trim_messages`
    # drops from the *start* while keeping the structure valid: the system
    # message is preserved and the history still begins on a human turn.
    long_history = [SystemMessage("You are a helpful assistant.")]
    for i in range(1, 6):
        long_history.append(HumanMessage(f"Question {i}"))
        long_history.append(AIMessage(f"Answer {i}"))

    trimmed = trim_messages(
        long_history,
        strategy="last",           # keep the most recent messages
        token_counter=len,         # count messages, not tokens, to keep it obvious
        max_tokens=5,              # ...so this means "at most 5 messages"
        start_on="human",          # a valid history starts on a human turn
        include_system=True,       # never drop the standing instructions
    )
    kv("before", f"{len(long_history)} messages")
    kv("after", f"{len(trimmed)} messages")
    for message in trimmed:
        kv(type(message).__name__, message.text)
