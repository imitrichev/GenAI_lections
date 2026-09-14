import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class UserConfig(BaseModel):
    model: str = Field(os.getenv("MODEL", "tngtech/deepseek-r1t2-chimera"))
    base_url: str = Field(os.getenv("BASE_URL", "http://localhost:11434"))
    use_local_model: bool = Field(
        os.getenv("USE_LOCAL_MODEL", "false").lower() in ("true", "1", "yes")
    )
    ollama_model: str = Field(os.getenv("OLLAMA_MODEL", "qwen3:0.6b"))


@dataclass
class ProviderConfig:
    """
    Конфигурация LLM-провайдера (OpenRouter или Ollama).
    Вся логика выбора URL, модели и заголовков - здесь.
    """

    url: str
    model: str
    api_key: str | None = None
    stream: bool = False

    @classmethod
    def openrouter(cls, model: str, api_key: str | None = None) -> "ProviderConfig":
        return cls(
            url="https://openrouter.ai/api/v1/chat/completions",
            model=model,
            api_key=api_key,
            stream=False,
        )

    @classmethod
    def ollama(cls, base_url: str, model: str) -> "ProviderConfig":
        return cls(
            url=f"{base_url}/v1/chat/completions",
            model=model,
            api_key=None,
            stream=True,
        )

    @property
    def is_local(self) -> bool:
        return self.api_key is None
