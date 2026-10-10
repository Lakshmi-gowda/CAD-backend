from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ARTIFACTS_DIR: str = "backend/app/ml/artifacts"
    CORS_ORIGINS: str = "http://localhost:5173"
    MAX_REQUEST_SIZE_BYTES: int = Field(default=1_048_576, gt=0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def resolved_artifacts_dir(self) -> str:
        artifacts_dir = Path(self.ARTIFACTS_DIR)
        if not artifacts_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            artifacts_dir = project_root / artifacts_dir
        return str(artifacts_dir.resolve())

@lru_cache()
def get_settings() -> Settings:
    return Settings()
