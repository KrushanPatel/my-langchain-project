# LangChain core components — a runnable tour

Nine small programs, one per core component, each written to be **read** as much
as run. Every file is standalone and heavily commented; open the one you care
about and work top to bottom.

```
uv sync
uv run demo            # list the demos
uv run demo all        # run every one, in order
uv run demo agents     # or just one
```

No API key and no model server are needed to run any of this — see
[Which model am I talking to?](#which-model-am-i-talking-to) below.

## The tour

Read them in this order; each builds on the last.

| # | `uv run demo …` | File | What it covers |
|---|---|---|---|
| 1 | `models` | [`demos/models.py`](src/langchain_project/demos/models.py) | `init_chat_model`, `invoke` / `batch` / `stream`, `bind_tools`, `with_structured_output`, token usage |
| 2 | `messages` | [`demos/messages.py`](src/langchain_project/demos/messages.py) | the four roles, `.text` vs `.content` vs `.content_blocks`, tool calls, chunk arithmetic, `trim_messages` |
| 3 | `tools` | [`demos/tools.py`](src/langchain_project/demos/tools.py) | `@tool`, explicit `args_schema`, `ToolRuntime`, `ToolException` + `ToolErrorMiddleware`, `BaseTool` |
| 4 | `agents` | [`demos/agents.py`](src/langchain_project/demos/agents.py) | `create_agent`, the model↔tool loop, custom `state_schema` and `context_schema`, loop guards |
| 5 | `middleware` | [`demos/middleware.py`](src/langchain_project/demos/middleware.py) | every hook, decorator and class form, execution order, the built-ins |
| 6 | `memory` | [`demos/short_term_memory.py`](src/langchain_project/demos/short_term_memory.py) | checkpointers, `thread_id`, `get_state`, summarising a long thread |
| 7 | `structured-output` | [`demos/structured_output.py`](src/langchain_project/demos/structured_output.py) | `response_format`, `ToolStrategy` vs `ProviderStrategy`, validated Pydantic results |
| 8 | `streaming` | [`demos/streaming.py`](src/langchain_project/demos/streaming.py) | `stream_mode` = `updates` / `messages` / `values` / `custom`, several at once, `astream` |
| 9 | `event-streaming` | [`demos/event_streaming.py`](src/langchain_project/demos/event_streaming.py) | `astream_events`, event shapes, filtering, custom events, timing a run |

Docs for each: <https://docs.langchain.com/oss/python/langchain/agents> and its
siblings — every demo's docstring links to its own page.

## The one idea worth carrying away

Everything above is one loop with hooks on it:

```
START ─► model ─► tool calls? ─┬─ no ──► END
                              └─ yes ─► tools ─► model ─► …
```

- **Messages** are the wire format in and out of the model.
- **Tools** are functions plus a schema; the model *requests* them, the agent *runs* them.
- **Middleware** wraps each step of that loop — that is where prompt shaping,
  limits, redaction, retries and approval live.
- **Checkpointers** persist the message list between turns, which is what
  "memory" means here.
- **Structured output** constrains the final message to a schema.
- **Streaming** and **event streaming** are two resolutions of the same run:
  state changes vs. every individual runnable.

## Which model am I talking to?

Demos get their model from [`config.get_model()`](src/langchain_project/config.py),
which reads two environment variables:

| | |
|---|---|
| *(default)* | `ScriptedChatModel` — an offline stand-in, no server required |
| `LC_PROVIDER=ollama LC_MODEL=llama3.1` | a real model through Ollama |
| `LC_PROVIDER=anthropic LC_MODEL=…` | any other provider `init_chat_model` supports |

The default deserves a word, because it is unusual and it is the reason this
repo runs anywhere.

[`ScriptedChatModel`](src/langchain_project/scripted_model.py) is a real
`BaseChatModel` that returns `AIMessage`s you wrote in advance, one per call.
LangChain cannot tell it apart from a hosted model: `create_agent`, the tool
loop, streaming, structured output and middleware all run their genuine code
paths against it. What you lose is the model's judgement; what you gain is a
tour that is instant, free and identical every time — so when output changes,
it changed because *you* changed something.

Each demo passes its `script=[…]` to `get_model()`. Real providers ignore that
argument entirely, so **the demo code is the same either way**:

```bash
ollama pull llama3.1
LC_PROVIDER=ollama LC_MODEL=llama3.1 uv run demo agents
```

Expect different wording, occasional wrong tool choices, and the odd refusal to
use a tool at all — that is the real thing, and worth seeing once you know what
the loop is supposed to look like.

## The same thing as an HTTP service

[`server.py`](src/langchain_project/server.py) wires one agent into FastAPI:
token streaming, plus a `thread_id` so conversations have memory.

```bash
uv run uvicorn langchain_project.server:app --reload

curl -N -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -d '{"message": "What is the weather in Tokyo?", "thread_id": "demo"}'

curl localhost:8000/history/demo
```

## Layout

```
src/langchain_project/
  demos/              the nine components, one module each
  scripted_model.py   the offline BaseChatModel implementation
  config.py           which provider the demos use
  shared_tools.py     a few plain tools the demos share
  server.py           the FastAPI version
  console.py          printing helpers, nothing LangChain-specific
  __main__.py         the `uv run demo` CLI
```

## Notes

- Requires Python 3.13+. (3.14 works once pydantic supports your build — the
  3.14.0rc2 toolchain this was set up on could not import pydantic at all.)
- Pinned to `langchain>=1.4`, whose API — `create_agent`, middleware,
  `ToolRuntime`, content blocks — differs substantially from 0.x tutorials you
  will find online. When something does not match, check the version first.
