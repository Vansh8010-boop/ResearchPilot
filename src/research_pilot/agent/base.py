from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

EventCallback = Callable[[str, dict], None]


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: str  # answered | max_steps | refusal | max_tokens | api_error
    steps: int  # number of model calls
    trace_path: Path
    plan: list[str] = field(default_factory=list)
