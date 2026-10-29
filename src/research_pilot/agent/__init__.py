from .base import AgentResult
from .plan_execute import PlanExecuteAgent
from .react import ReActAgent

AGENTS = {"plan": PlanExecuteAgent, "react": ReActAgent}

__all__ = ["AGENTS", "AgentResult", "PlanExecuteAgent", "ReActAgent"]
