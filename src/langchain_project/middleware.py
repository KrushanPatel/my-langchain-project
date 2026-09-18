import logging
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class LogChatStart(AgentMiddleware):
    """Runs once, only before/after the agent loop processes an invocation."""

    def before_agent(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        logger.info("Chat starting with %d message(s)", len(state["messages"]))
        return super().before_agent(state, runtime)

    def after_agent(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        logger.info("Chat ending with %d message(s)", len(state["messages"]))
        return super().after_agent(state, runtime)
