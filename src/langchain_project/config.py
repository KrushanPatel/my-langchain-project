from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized app configuration, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    model_name: str = "gpt-oss:120b-cloud"
    model_provider: str = "ollama"
    model_temperature: float = 0

    model_call_thread_limit: int = 5
    tool_call_thread_limit: int = 5

    summarization_trigger_tokens: int = 4000
    summarization_keep_messages: int = 20

    system_prompt: str = "This is an AI Agent that performs tasks based on its capacity"
    chat_system_prompt: str = "You are helping the user learn LangChain."

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
