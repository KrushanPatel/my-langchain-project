from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware,ModelCallLimitMiddleware, ToolCallLimitMiddleware, AgentMiddleware
from langchain.agents.middleware import before_agent
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.messages import AIMessage, AIMessageChunk, SystemMessage, HumanMessage, ToolMessage
from langchain.tools import BaseTool, ToolException
from langchain_ollama import ChatOllama
from pprint import pprint
from typing_extensions import NotRequired, Required
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.runtime import Runtime
from typing import Any, Annotated, TypedDict

def create_model() -> BaseChatModel:
    return init_chat_model(
        model="gpt-oss:120b-cloud",
        model_provider="ollama",
        temperature=0,
    )

class MyAgentState(AgentState):

    user_id: Required[Annotated[str | None, "user identifier"]]

class CalculatorTool(BaseTool):

    """
    A simple calculator implemented directly with BaseTool.

    This is intentionally implemented as a class so that you can
    understand what BaseTool looks like internally.
    """
    name: str = "calculator"
    description: str = ( "" \
                "This is tool that performs arthmetic operation" \
                "Use this tool when the user asks for a mathematical calculation. \"" 
    )

    def _run(self, expression: str) -> str:
        """
        Execute the tool synchronously.

        The agent supplies the expression.
        """

        try:
            # This is intentionally simple for learning.
            # Do NOT use eval() like this in production.
            allowed_characters = set(
                "0123456789+-*/(). "
            )

            if not set(expression) <= allowed_characters:
                raise ToolException(
                    "Expression contains unsupported characters."
                )

            result = eval(expression)
            print("Tool Called by LLM")
            return str(result)

        except Exception as exc:
            raise ToolException(
                f"Could not calculate expression: {expression}"
            ) from exc

class UserInfoTool(BaseTool):
    """
    Demonstrates a second BaseTool implementation.
    """

    name: str = "get_user_info"

    description: str = (
        "Get information about the current user. "
        "Use this when the user asks what their name is."
    )

    def _run(self, user_name: str) -> str:
        if not user_name:
            raise ToolException("User name is required.")

        return f"The user's name is {user_name}."

calculator_tool = CalculatorTool(
    handle_tool_error=True
)

user_info_tool = UserInfoTool(
    handle_tool_error=True
)

class LogChatStart(AgentMiddleware):
    """Runs once, Only before the agent loop starts processing a new invocation."""
    def before_agent(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        print(f"Chat starting with {len(state['messages'])} message(s)")
        return super().before_agent(state, runtime)

    def after_agent(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        print(f"Chat ends with {len(state['messages'])} message(s)")
        return super().after_agent(state, runtime)
    
middleware = [
    LogChatStart(),
    SummarizationMiddleware(
        model=ChatOllama(
            model="gpt-oss:120b-cloud",
            temperature=0,
        ),
        trigger=("tokens", 4000),
        keep=("messages", 20),
    ),
    ModelCallLimitMiddleware(
        thread_limit=5
    ),
    ToolCallLimitMiddleware(
        thread_limit=5
    ),
]

tools = [calculator_tool,user_info_tool]

model = create_model()

agent = create_agent(
    model= model,
    tools = tools,
    system_prompt= "This is an AI Agent that perfrom task based on its capacity",
    middleware=middleware,
    state_schema=MyAgentState
)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

@app.post("/chat")
async def chat(request: ChatRequest):

    messages = [
        SystemMessage(
            content="You are helping the user learn LangChain."
        ),

        HumanMessage(
            content=request.message
        )
    ]

    async def generate():

        response = agent.astream(
            {
                "messages": messages,
            },
            stream_mode="messages",
        )
        
        async for message, metadata in response:
            print(message)
            if isinstance(message, AIMessageChunk):

                if (
                    message.content
                    and isinstance(message.content, str)
                ):
                    yield message.content

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )



    