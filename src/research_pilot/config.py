"""Runtime settings, read from environment variables (and .env if present)."""

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# provider -> (base URL, env var holding the API key, default model)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),
    "anthropic": (None, "ANTHROPIC_API_KEY", "claude-opus-5-5"),
}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    effort: str
    max_steps: int
    max_plan_steps: int
    max_step_turns: int
    max_tool_turns: int
    max_tokens: int
    max_run_tokens: int
    defenses: bool
    base_url: str | None
    api_key: str | None
    tavily_api_key: str | None
    cache_dir: Path
    trace_dir: Path
    use_cache: bool = True


def load_settings(**overrides) -> Settings:
    load_dotenv(ROOT / ".env")
    provider = overrides.pop("provider", None) or os.getenv("RP_PROVIDER", "groq")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}', expected one of {sorted(PROVIDERS)}")
    base_url, key_env, default_model = PROVIDERS[provider]

    settings = Settings(
        provider=provider,
        model=os.getenv("RP_MODEL", default_model),
        effort=os.getenv("RP_EFFORT", "medium"),
        max_steps=int(os.getenv("RP_MAX_STEPS", "8")),
        max_plan_steps=int(os.getenv("RP_MAX_PLAN_STEPS", "6")),
        max_step_turns=int(os.getenv("RP_MAX_STEP_TURNS", "4")),
        max_tool_turns=int(os.getenv("RP_MAX_TOOL_TURNS", "16")),
        max_tokens=int(os.getenv("RP_MAX_TOKENS", "16000")),
        max_run_tokens=int(os.getenv("RP_MAX_RUN_TOKENS", "60000")),
        defenses=os.getenv("RP_DEFENSES", "on").lower() not in ("off", "0", "false", "no"),
        base_url=os.getenv("RP_BASE_URL", base_url),
        api_key=os.getenv(key_env) if key_env else None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        cache_dir=Path(os.getenv("RP_CACHE_DIR", ROOT / ".cache")),
        trace_dir=Path(os.getenv("RP_TRACE_DIR", ROOT / "traces")),
    )
    return replace(settings, **overrides)
