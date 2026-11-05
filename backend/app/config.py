"""Application configuration loaded from .env via Pydantic Settings."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> project root
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"

SQL_SCRIPT_PATH = PROJECT_ROOT / "school_management_postgres92.sql"
MANUAL_PDF_PATH = PROJECT_ROOT / "School Management Database - User Manual.pdf"
MANUAL_TXT_PATH = DATA_DIR / "manual_extracted.txt"


class Settings(BaseSettings):
    """All settings come from environment variables / the .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database -----------------------------------------------------
    database_url: str = "postgresql://postgres:postgres@localhost:5432/school_db"
    db_schema: str = "school"
    statement_timeout_ms: int = 5000
    db_pool_min: int = 1
    db_pool_max: int = 8

    # --- LLM ----------------------------------------------------------
    # local | openai | anthropic | gemini | ollama | custom
    llm_provider: str = "local"
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    agent_max_tool_rounds: int = 8

    # --- Server ---------------------------------------------------------
    backend_port: int = 8000
    frontend_port: int = 5173
    cors_origins: str = "*"


settings = Settings()

# Keys used per provider (LLM_API_KEY overrides everything when present).
_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": "",  # No API key needed for a local Ollama server.
    "custom": "LLM_API_KEY",
}

_PROVIDER_DEFAULTS = {
    "local": {
        "base_url": None,
        "model": "scikit-learn-ml-agent",
        "env_key": "",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",  # OpenAI-compatible endpoint
        "model": "claude-sonnet-4-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "env_key": "",
    },
    "custom": {
        "base_url": None,
        "model": "gpt-4o",
        "env_key": "LLM_API_KEY",
    },
}


def resolve_llm() -> dict:
    """Resolve provider/model/base_url/api_key into one coherent config."""
    provider = (settings.llm_provider or "local").strip().lower()
    if provider not in _PROVIDER_DEFAULTS:
        provider = "local"
    defaults = _PROVIDER_DEFAULTS[provider]

    model = settings.llm_model or defaults["model"]
    base_url = settings.llm_base_url or defaults["base_url"]

    api_key = settings.llm_api_key
    if not api_key and defaults["env_key"]:
        import os

        api_key = os.environ.get(defaults["env_key"]) or None

    # If an external provider was requested but no API key is available, automatically
    # fall back to the built-in local ML agent so the chatbot works immediately with 0 keys!
    if defaults["env_key"] and not api_key:
        provider = "local"
        defaults = _PROVIDER_DEFAULTS["local"]
        model = defaults["model"]
        base_url = None

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "requires_key": bool(defaults["env_key"]),
    }


def llm_is_ready() -> bool:
    cfg = resolve_llm()
    return (not cfg["requires_key"]) or bool(cfg["api_key"])
