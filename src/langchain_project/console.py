"""Tiny printing helpers. Nothing LangChain-specific lives here."""

from __future__ import annotations

import shutil

_WIDTH = min(shutil.get_terminal_size((88, 24)).columns, 88)


def title(text: str) -> None:
    """A heavy banner marking the start of a demo."""
    print()
    print("=" * _WIDTH)
    print(f" {text}")
    print("=" * _WIDTH)


def section(text: str) -> None:
    """A light banner marking one idea inside a demo."""
    print()
    print(f"-- {text} ".ljust(_WIDTH, "-"))


def note(text: str) -> None:
    """An aside explaining what just happened."""
    print(f"   . {text}")


def kv(key: str, value: object) -> None:
    """A labelled value."""
    print(f"   {key:<22} {value}")
