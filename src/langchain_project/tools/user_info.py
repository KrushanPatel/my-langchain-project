from langchain.tools import ToolException, ToolRuntime, tool

from langchain_project.context import AgentContext


@tool(
    "get_user_info",
    description=(
        "Get information about the current user. "
        "Use this when the user asks what their name is."
    ),
)
def user_info_tool(runtime: ToolRuntime[AgentContext]) -> str:
    """Look up the current user's identity from the authenticated request
    context — never from an LLM-supplied argument, since that can't be
    trusted to reflect who's actually asking."""

    user_id = runtime.context.user_id

    if not user_id:
        raise ToolException("No authenticated user for this request.")

    return f"The current user's id is {user_id}."
