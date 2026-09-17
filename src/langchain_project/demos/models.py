"""Models - the thing that turns messages into a message.

https://docs.langchain.com/oss/python/langchain/models

A chat model in LangChain is a `BaseChatModel`. Whichever provider is behind it,
it always speaks the same language:

    messages in  ->  AIMessage out

Everything else in LangChain - agents, tools, structured output, streaming - is
built on top of that one contract, which is why swapping providers is a
one-line change.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langchain_project import config
from langchain_project.console import kv, note, section, title


class Rating(BaseModel):
    """How much someone liked something."""

    score: int = Field(description="A rating from 1 to 5")
    reason: str = Field(description="One sentence justifying the score")


def run() -> None:
    title("MODELS")
    note(config.describe_provider())

    # -----------------------------------------------------------------
    section("Building a model")
    # -----------------------------------------------------------------
    # `init_chat_model(model=..., model_provider=...)` is the generic way to
    # build one - see config.get_model(), which wraps it. Passing
    # `temperature` here is passed on to the provider; the offline model
    # ignores it.
    model = config.get_model(
        temperature=0,
        script=[
            AIMessage(
                content="LangChain gives every model provider one shared interface.",
                usage_metadata={"input_tokens": 24, "output_tokens": 11, "total_tokens": 35},
            ),
            AIMessage(content="Paris."),
            AIMessage(content="Tokyo."),
            AIMessage(content="Streaming sends the answer out piece by piece."),
        ],
    )
    kv("type", type(model).__name__)

    # -----------------------------------------------------------------
    section("invoke - one call, one message back")
    # -----------------------------------------------------------------
    # The input can be a list of messages, a list of (role, text) tuples, or
    # even a bare string. All three are normalised to messages internally.
    reply = model.invoke(
        [
            SystemMessage("You are a concise teaching assistant."),
            HumanMessage("In one sentence: what problem does LangChain solve?"),
        ]
    )
    kv("returned", type(reply).__name__)
    kv("reply.text", reply.text)

    # Token accounting rides along on the message when the provider reports it.
    if reply.usage_metadata:
        kv("usage_metadata", reply.usage_metadata)

    # -----------------------------------------------------------------
    section("batch - many independent calls at once")
    # -----------------------------------------------------------------
    # `batch` runs the inputs concurrently rather than one after another.
    # These are *separate* conversations, not one conversation of three turns.
    answers = model.batch(
        [
            [HumanMessage("Capital of France? One word.")],
            [HumanMessage("Capital of Japan? One word.")],
        ]
    )
    for question, answer in zip(("France", "Japan"), answers):
        kv(question, answer.text)

    # -----------------------------------------------------------------
    section("stream - the same call, delivered incrementally")
    # -----------------------------------------------------------------
    # `.stream()` yields `AIMessageChunk`s. Chunks add together with `+`, and
    # the sum is equivalent to what `.invoke()` would have returned.
    print("   ", end="")
    aggregate: AIMessage | None = None
    for chunk in model.stream([HumanMessage("Explain streaming in one sentence.")]):
        print(chunk.text, end="", flush=True)
        aggregate = chunk if aggregate is None else aggregate + chunk
    print()
    note(f"rebuilt from chunks: {aggregate.text!r}")

    # -----------------------------------------------------------------
    section("bind_tools - letting the model ask for a tool")
    # -----------------------------------------------------------------
    # `bind_tools` returns a *new* runnable with the tool schemas attached.
    # The original model is untouched. The model still only produces a
    # message - it never runs anything; it just requests a call. Something
    # else (an agent, or your own code) does the running.
    from langchain_project.shared_tools import get_weather

    tool_model = model.bind_tools([get_weather])
    kv("bound type", type(tool_model).__name__)
    note("the model now sees get_weather's name, description and argument schema")

    # -----------------------------------------------------------------
    section("with_structured_output - a model that returns objects")
    # -----------------------------------------------------------------
    # This wraps bind_tools + parsing, so `.invoke()` hands back a `Rating`
    # instance instead of a message. See the structured-output demo for the
    # agent-level version.
    structured_model = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Rating",
                        "args": {"score": 5, "reason": "One interface for every provider."},
                        "id": "call_rating",
                    }
                ],
            )
        ]
    ).with_structured_output(Rating)

    rating: Rating = structured_model.invoke([HumanMessage("Rate LangChain's model interface.")])
    kv("returned", type(rating).__name__)
    kv("score", rating.score)
    kv("reason", rating.reason)
