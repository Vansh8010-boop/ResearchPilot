from dataclasses import dataclass


@dataclass
class ToolResult:
    """What every tool returns. `ok=False` results go back to the model with is_error=True."""

    ok: bool
    content: str
    cached: bool = False
    retryable: bool = False  # transient failure (timeout, 429, 5xx) worth retrying
    blocked: bool = False  # refused by a policy before running (URL policy, duplicate call)


def error(message: str, retryable: bool = False) -> ToolResult:
    return ToolResult(ok=False, content=message, retryable=retryable)
