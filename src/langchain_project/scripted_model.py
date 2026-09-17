"""A deterministic, offline stand-in for a real chat model.

Why this exists
---------------
Every demo in this project needs *a model*. Waiting on a real LLM makes the
examples slow, non-deterministic, and impossible to run without a server. So by
default the demos run against `ScriptedChatModel`: you hand it the exact
`AIMessage` objects you want it to "generate", and it returns them one per call.

Crucially, this is a *real* `BaseChatModel`. LangChain does not know or care
that it is fake — `create_agent`, the tool-calling loop, streaming, structured
output and middleware all run their genuine code paths against it. You are
seeing real LangChain behaviour with a predictable model attached.

Point `LC_PROVIDER=ollama` at a real model whenever you want (see `config.py`);
none of the demo code changes.

Implementing a chat model comes down to four things, all visible below:
  * `_llm_type`   - an identifier used in logging and tracing
  * `_generate`   - the blocking call: messages in, a `ChatResult` out
  * `_stream`     - the incremental call: yields `ChatGenerationChunk`s
  * `bind_tools`  - attaches tool schemas so the model can request tool calls
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written `AIMessage`s, one per model call.

    Args:
        script: The replies to hand back, in order. An entry carrying
            `tool_calls` makes the agent run those tools and call the model
            again - which is how the multi-step demos work.
        fallback: Text used once the script runs out. Always tool-call free, so
            an agent loop can never spin forever.
        model_name: Cosmetic; shows up in streaming metadata and traces.
    """

    script: list[AIMessage] = Field(default_factory=list)
    fallback: str = "That is everything I have for you."
    model_name: str = "scripted-model"

    # Pydantic private attrs are per-instance mutable state, not part of the
    # model's schema. This one tracks how far through the script we are.
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "script_length": len(self.script)}

    def reset(self) -> None:
        """Rewind to the start of the script so the model can be reused."""
        self._cursor = 0

    def _next_message(self) -> AIMessage:
        """Pop the next scripted reply, or fall back once the script is spent."""
        if self._cursor < len(self.script):
            message = self.script[self._cursor]
            self._cursor += 1
            # Copy: the agent may attach ids/metadata to what it receives, and
            # we do not want that leaking back into the script.
            return message.model_copy(deep=True)
        return AIMessage(content=self.fallback, response_metadata={"scripted": "fallback"})

    # -- the blocking path -------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """What `.invoke()` ultimately calls."""
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    # -- the streaming path ------------------------------------------------

    def _chunks(self) -> list[ChatGenerationChunk]:
        """Chop the next scripted reply into chunks, the way a real model would.

        A real model emits tokens. We emit words, which is close enough to show
        how chunk streaming and chunk aggregation behave.
        """
        message = self._next_message()

        if message.tool_calls:
            # Tool calls arrive as `tool_call_chunks` in a real stream. Sending
            # one complete chunk keeps the example readable.
            return [
                ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_calls=message.tool_calls)
                )
            ]

        pieces = re.findall(r"\S+\s*", message.text) or [message.text]
        chunks = [ChatGenerationChunk(message=AIMessageChunk(content=piece)) for piece in pieces]
        if message.usage_metadata:
            # Usage normally lands on the final chunk of the stream.
            chunks[-1] = ChatGenerationChunk(
                message=AIMessageChunk(content=pieces[-1], usage_metadata=message.usage_metadata)
            )
        return chunks

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._chunks():
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._chunks():
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    # -- tool calling ------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Attach tools to the model.

        A real integration converts each tool to the provider's JSON schema and
        sends it with every request. We only need to accept them, because our
        "decision" to call a tool was made in advance by the script.
        """
        return self.bind(tools=list(tools), **kwargs)
