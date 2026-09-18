from fastapi.testclient import TestClient

from langchain_project.app import create_app


def test_create_app_registers_chat_route():
    app = create_app()
    client = TestClient(app)
    openapi = client.get("/openapi.json").json()
    assert "/chat" in openapi["paths"]
