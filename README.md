# langchain-project

A LangChain agent served over FastAPI, with a calculator tool and a
user-info tool, streaming responses over `/chat`.

## Layout

```
src/langchain_project/
  app.py            # FastAPI app factory + lifespan (builds the agent once at startup)
  main.py           # ASGI entrypoint: `uvicorn langchain_project.main:app`
  config.py         # Settings (env-driven, prefix APP_)
  agent.py          # Model + middleware + agent construction
  context.py        # AgentContext passed into tool calls
  middleware.py      # Custom agent middleware (chat start/end logging)
  tools/            # Agent tools (calculator, user info)
  api/              # FastAPI routers
tests/
```

## Setup

```
cp .env.example .env   # adjust values as needed
uv sync
```

## Run

```
uv run langchain-project
# or
uv run uvicorn langchain_project.main:app --reload
```

## Test

```
uv run pytest
```

## Configuration

All settings are read from the environment (or `.env`) with an `APP_`
prefix — see `.env.example` for the full list (model name/provider,
temperature, call limits, summarization thresholds, host/port).
