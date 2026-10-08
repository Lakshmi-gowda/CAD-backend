import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ARTIFACTS_DIR: str = "backend/app/ml/artifacts"
    CORS_ORIGINS: str = "http://localhost:5173"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS:
            return ["http://localhost:5173"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def resolved_artifacts_dir(self) -> str:
        # Check explicit path
        candidates = [
            self.ARTIFACTS_DIR,
            os.path.join(os.getcwd(), self.ARTIFACTS_DIR),
            os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts"),
            os.path.join(os.getcwd(), "MODEL"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "MODEL"),
        ]
        for c in candidates:
            if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "CAD_XGBoost_Metadata.pkl")):
                return os.path.abspath(c)
        return os.path.abspath(self.ARTIFACTS_DIR)

@lru_cache()
def get_settings() -> Settings:
    return Settings()
