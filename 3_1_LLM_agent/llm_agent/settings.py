import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class UserConfig(BaseModel):
    model: str = Field(os.getenv("MODEL", "tngtech/deepseek-r1t2-chimera"))
    base_url: str = Field(os.getenv("BASE_URL", "http://localhost:11434"))
    use_local_model: bool = Field(os.getenv("USE_LOCAL_MODEL", "false").lower() in ("true", "1", "yes"))
    ollama_model: str = Field(os.getenv("OLLAMA_MODEL", "qwen3:0.6b"))
