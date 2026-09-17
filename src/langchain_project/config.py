"""One place that decides which chat model the demos talk to.

Two providers are wired up:

  LC_PROVIDER=scripted   (default) - `ScriptedChatModel`, no server needed.
  LC_PROVIDER=ollama               - a real local/cloud model through Ollama.

Anything else is passed straight to `init_chat_model`, so
`LC_PROVIDER=anthropic LC_MODEL=claude-sonnet-4-5` works too if you install
that integration package and set its API key.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_project.scripted_model import ScriptedChatModel

DEFAULT_MODELS = {
    "ollama": "gpt-oss:120b-cloud",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1",
}


def provider() -> str:
    """The provider currently selected by the environment."""
    return os.getenv("LC_PROVIDER", "scripted").strip().lower()


def is_scripted() -> bool:
    """True when demos are running against the offline stand-in model."""
    return provider() == "scripted"


def get_model(
    *,
    script: list[AIMessage] | None = None,
    fallback: str | None = None,
    **model_kwargs: Any,
) -> BaseChatModel:
    """Build a chat model for a demo.

    Args:
        script: Replies for the offline model. Ignored by real providers -
            which is the point: the same demo code runs either way.
        fallback: Offline reply used once the script is exhausted.
        **model_kwargs: Passed to the real model (`temperature`, `max_tokens`,
            `timeout`, ...). The offline model ignores them.
    """
    name = provider()

    if name == "scripted":
        model = ScriptedChatModel(script=script or [])
        if fallback is not None:
            model.fallback = fallback
        return model

    # `init_chat_model` is the provider-agnostic constructor: give it a model
    # name and a provider and it imports and configures the right integration.
    return init_chat_model(
        model=os.getenv("LC_MODEL", DEFAULT_MODELS.get(name, "")),
        model_provider=name,
        **model_kwargs,
    )


def describe_provider() -> str:
    """A one-line banner so you always know what you are talking to."""
    name = provider()
    if name == "scripted":
        return "provider=scripted (offline stand-in model - set LC_PROVIDER=ollama for a real one)"
    return f"provider={name} model={os.getenv('LC_MODEL', DEFAULT_MODELS.get(name, '?'))}"
