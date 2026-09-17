"""Structured output - getting an object back instead of prose.

https://docs.langchain.com/oss/python/langchain/structured-output

Free text is fine for a chat window and useless for a database. Structured
output constrains the model to a schema you define, and hands you a validated
Python object.

Two levels:

  model.with_structured_output(Schema)   - one call, object out. No agent.
  create_agent(..., response_format=...) - the agent may use tools first, then
                                           produce the object as its final act.

Under the hood there are two strategies:

  ToolStrategy      - the schema is offered to the model as a tool it must call.
                      Works with any tool-calling model. LangChain parses and
                      validates the arguments for you.
  ProviderStrategy  - the provider enforces the schema itself (OpenAI's
                      structured outputs, Anthropic's, ...). Stronger guarantee,
                      only where supported.

Pass a bare schema and LangChain picks the strategy your model supports.
"""

from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import AIMessage
from pydantic import BaseModel, Field

from langchain_project import config
from langchain_project.console import kv, note, section, title
from langchain_project.shared_tools import get_weather


class ContactDetails(BaseModel):
    """A person pulled out of a block of text."""

    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    company: str | None = Field(default=None, description="Employer, if mentioned")


class WeatherReport(BaseModel):
    """A structured weather answer."""

    city: str = Field(description="City the report is about")
    temperature_c: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Short description, e.g. 'clear'")
    advice: str = Field(description="One sentence of practical advice")


class Triage(BaseModel):
    """Classification of an incoming support message."""

    category: Literal["bug", "billing", "feature_request", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="One-line summary of the request")


def run() -> None:
    title("STRUCTURED OUTPUT")
    note(config.describe_provider())

    # -----------------------------------------------------------------
    section("Level 1: with_structured_output on a bare model")
    # -----------------------------------------------------------------
    # The field *descriptions* are sent to the model - they are instructions,
    # so write them as such.
    extractor = config.get_model(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ContactDetails",
                        "args": {
                            "name": "Ada Lovelace",
                            "email": "ada@analytical.io",
                            "company": "Analytical Engines Ltd",
                        },
                        "id": "x1",
                    }
                ],
            )
        ]
    ).with_structured_output(ContactDetails)

    contact = extractor.invoke(
        [("user", "Reach Ada Lovelace at ada@analytical.io; she's at Analytical Engines Ltd.")]
    )
    kv("type", type(contact).__name__)
    kv("name", contact.name)
    kv("email", contact.email)
    kv("company", contact.company)
    note("a real Pydantic model - validated, typed, ready to store")

    # -----------------------------------------------------------------
    section("Level 2: an agent that uses tools, then returns an object")
    # -----------------------------------------------------------------
    # `response_format` does not replace the tool loop; it runs after it. The
    # agent looks up the weather, then packages the answer into the schema.
    agent = create_agent(
        model=config.get_model(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "London"}, "id": "g1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "WeatherReport",
                            "args": {
                                "city": "London",
                                "temperature_c": 13.0,
                                "conditions": "raining",
                                "advice": "Take an umbrella.",
                            },
                            "id": "g2",
                        }
                    ],
                ),
            ]
        ),
        tools=[get_weather],
        response_format=WeatherReport,
    )

    result = agent.invoke({"messages": [("user", "What's it like in London?")]})

    # The object lands on state under `structured_response`; `messages` still
    # holds the full transcript that produced it.
    report: WeatherReport = result["structured_response"]
    kv("state keys", list(result.keys()))
    kv("city", report.city)
    kv("temperature_c", report.temperature_c)
    kv("conditions", report.conditions)
    kv("advice", report.advice)
    note(f"it ran {len(result['messages'])} messages' worth of tool work first")

    # -----------------------------------------------------------------
    section("Choosing a strategy explicitly")
    # -----------------------------------------------------------------
    # ToolStrategy works everywhere and lets you decide what happens when the
    # model returns arguments that fail validation: `handle_errors=True` feeds
    # the error back so the model can correct itself.
    triage_agent = create_agent(
        model=config.get_model(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "Triage",
                            "args": {
                                "category": "bug",
                                "urgency": "high",
                                "summary": "Checkout fails with a 500 on card payment.",
                            },
                            "id": "t1",
                        }
                    ],
                )
            ]
        ),
        tools=[],
        response_format=ToolStrategy(Triage, handle_errors=True),
    )

    triaged: Triage = triage_agent.invoke(
        {"messages": [("user", "Checkout has been 500ing on card payments since this morning!")]}
    )["structured_response"]

    kv("category", triaged.category)
    kv("urgency", triaged.urgency)
    kv("summary", triaged.summary)
    note("Literal fields become enums in the schema - the model cannot invent a category")
    note("for providers with native support, swap ToolStrategy for ProviderStrategy")
