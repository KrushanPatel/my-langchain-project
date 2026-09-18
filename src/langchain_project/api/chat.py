from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.messages import AIMessageChunk, HumanMessage, SystemMessage

from langchain_project.context import AgentContext

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"


@router.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    agent = http_request.app.state.agent
    settings = http_request.app.state.settings

    messages = [
        SystemMessage(content=settings.chat_system_prompt),
        HumanMessage(content=request.message),
    ]

    async def generate():
        response = agent.astream(
            {"messages": messages},
            context=AgentContext(user_id=request.user_id),
            stream_mode="messages",
        )

        async for message, _metadata in response:
            if isinstance(message, AIMessageChunk) and isinstance(message.content, str) and message.content:
                yield message.content

    return StreamingResponse(generate(), media_type="text/plain")
