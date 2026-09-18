from dataclasses import dataclass


@dataclass
class AgentContext:
    """Per-invocation identity/config. Immutable and not persisted with
    thread state — distinct from AgentState, which is checkpointed
    conversation data."""

    user_id: str
