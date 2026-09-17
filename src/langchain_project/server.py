"""A FastAPI endpoint that streams an agent's answer back over HTTP.

Everything the demos show in a terminal, wired into the shape you would
actually ship: one endpoint, one agent, tokens streamed as they arrive, and a
`thread_id` so the conversation has memory.

    uv run uvicorn langchain_project.server:app --reload

    curl -N -X POST localhost:8000/chat \
         -H 'content-type: application/json' \
         -d '{"message": "What is the weather in Tokyo?", "thread_id": "demo"}'
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from langchain_project import config
from langchain_project.shared_tools import add, get_weather, multiply

app = FastAPI(title="LangChain core components")

# One checkpointer for the process. In production this would be the SQLite or
# Postgres saver so conversations survive a restart.
_checkpointer = InMemorySaver()

_agent = create_agent(
    model=config.get_model(
        # Only used by the offline provider; a real model ignores this entirely.
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "http1"}],
            ),
            AIMessage(content="Tokyo is 22C and clear right now."),
        ],
        fallback="I'm the offline stand-in model - set LC_PROVIDER=ollama for real answers.",
    ),
    tools=[get_weather, add, multiply],
    system_prompt="You are a concise assistant. Prefer tools over guessing.",
    checkpointer=_checkpointer,
)


class ChatRequest(BaseModel):
    """What a client posts to /chat."""

    message: str
    thread_id: str = Field(default="default", description="Conversation to continue")


@app.get("/")
def index() -> dict[str, str]:
    """Health check, and a reminder of which model is behind the endpoint."""
    return {"status": "ok", "provider": config.describe_provider()}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream the agent's reply as plain text, token by token."""

    async def token_stream() -> AsyncIterator[str]:
        async for chunk, _metadata in _agent.astream(
            {"messages": [("user", request.message)]},
            config={"configurable": {"thread_id": request.thread_id}},
            stream_mode="messages",
        ):
            # Tool-call chunks have no text; only forward what a user can read.
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                yield chunk.text

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.get("/history/{thread_id}")
def history(thread_id: str) -> dict[str, object]:
    """Read a conversation back out of the checkpointer."""
    snapshot = _agent.get_state({"configurable": {"thread_id": thread_id}})
    return {
        "thread_id": thread_id,
        "messages": [
            {"role": type(message).__name__, "text": message.text}
            for message in snapshot.values.get("messages", [])
        ],
    }
