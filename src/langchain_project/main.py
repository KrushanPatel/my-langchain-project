"""ASGI entrypoint — run with `uvicorn langchain_project.main:app`."""

from langchain_project.app import create_app

app = create_app()
