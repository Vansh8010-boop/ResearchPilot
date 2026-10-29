from dataclasses import dataclass


@dataclass
class ToolResult:
    """What every tool returns. `ok=False` results go back to the model with is_error=True."""

    ok: bool
    content: str
    cached: bool = False


def error(message: str) -> ToolResult:
    return ToolResult(ok=False, content=message)
