import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load secrets from real `.env` files only — not from `.env.example` (that file is a template).
# Search order: repo-root `.env` first, then `backend/.env` (backend overrides root).
#
# We use python-dotenv before constructing Settings because pydantic-settings treats OS env vars
# as higher priority than `env_file`; an *empty* `GROQ_API_KEY` in the shell would override a
# valid value in `.env` and look like "not set". Loading with override=True applies file values;
# we then drop empty-string entries so a blank export does not wipe keys.
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent


def _bootstrap_dotenv() -> None:
    paths = (_PROJECT_ROOT / ".env", _BACKEND_DIR / ".env")
    for path in paths:
        if path.is_file():
            load_dotenv(path, override=True)
    # Remove empty API key placeholders so they do not block reload / confuse validation
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(key, None) == "":
            os.environ.pop(key, None)


_bootstrap_dotenv()


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    POSTGRES_URL: str = "postgresql+asyncpg://llm_user:password@localhost:5432/llmoptimizer"
    REDIS_URL: str = "redis://localhost:6379"
    ROUTER_MODEL_PATH: str = "models/router_model.pkl"
    FEATURE_NAMES_PATH: str = "models/feature_names.json"
    LABEL_MAP_PATH: str = "models/label_map.json"
    FAISS_INDEX_PATH: str = "models/faiss_index"
    CACHE_SIMILARITY_THRESHOLD: float = 0.85
    CACHE_TTL_SECONDS: int = 86400
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.60
    DAILY_BUDGET_USD: float = 5.00
    RETRAIN_EVERY_N_QUERIES: int = 500
    BAD_RESPONSE_RATE_THRESHOLD: float = 0.20
    MAX_TOKENS: int = 1000
    LOG_LEVEL: str = "INFO"

    GROQ_LLAMA3_COST: float = 0.00005
    GROQ_MIXTRAL_COST: float = 0.00027
    GEMINI_PRO_COST: float = 0.00350

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
    )


settings = Settings()
