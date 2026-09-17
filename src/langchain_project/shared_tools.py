"""Tools reused across several demos.

The tools demo builds more elaborate ones; these are the plain, boring versions
that the agent/memory/streaming demos need in order to have something to call.
"""

from __future__ import annotations

from langchain.tools import tool

# A stand-in for whatever real data source you would reach for.
_FORECASTS = {
    "london": "13C and raining",
    "tokyo": "22C and clear",
    "paris": "18C and overcast",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city, e.g. "London".
    """
    return _FORECASTS.get(city.strip().lower(), f"No forecast on file for {city}.")


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b
