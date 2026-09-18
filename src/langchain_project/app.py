import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from langchain_project.agent import build_agent
from langchain_project.api.chat import router as chat_router
from langchain_project.config import settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.agent = build_agent(settings)
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(chat_router)
    return app
