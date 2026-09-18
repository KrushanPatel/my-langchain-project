import uvicorn

from langchain_project.config import settings


def main() -> None:
    uvicorn.run(
        "langchain_project.main:app",
        host=settings.host,
        port=settings.port,
    )
