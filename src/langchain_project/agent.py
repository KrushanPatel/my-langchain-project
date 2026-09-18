from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.chat_models import BaseChatModel, init_chat_model

from langchain_project.config import Settings
from langchain_project.context import AgentContext
from langchain_project.middleware import LogChatStart
from langchain_project.tools import tools


def create_model(settings: Settings) -> BaseChatModel:
    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider,
        temperature=settings.model_temperature,
    )


def build_agent(settings: Settings):
    model = create_model(settings)

    middleware = [
        LogChatStart(),
        SummarizationMiddleware(
            model=create_model(settings),
            trigger=("tokens", settings.summarization_trigger_tokens),
            keep=("messages", settings.summarization_keep_messages),
        ),
        ModelCallLimitMiddleware(thread_limit=settings.model_call_thread_limit),
        ToolCallLimitMiddleware(thread_limit=settings.tool_call_thread_limit),
    ]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=settings.system_prompt,
        middleware=middleware,
        context_schema=AgentContext,
    )
