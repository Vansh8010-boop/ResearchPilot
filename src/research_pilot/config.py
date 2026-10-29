"""Runtime settings, read from environment variables (and .env if present)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    max_steps: int
    max_tokens: int
    tavily_api_key: str | None
    cache_dir: Path
    trace_dir: Path
    use_cache: bool = True


def load_settings(**overrides) -> Settings:
    load_dotenv(ROOT / ".env")
    settings = Settings(
        model=os.getenv("RP_MODEL", "claude-opus-5-5"),
        effort=os.getenv("RP_EFFORT", "medium"),
        max_steps=int(os.getenv("RP_MAX_STEPS", "8")),
        max_tokens=int(os.getenv("RP_MAX_TOKENS", "16000")),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        cache_dir=Path(os.getenv("RP_CACHE_DIR", ROOT / ".cache")),
        trace_dir=Path(os.getenv("RP_TRACE_DIR", ROOT / "traces")),
    )
    return Settings(**{**settings.__dict__, **overrides})
