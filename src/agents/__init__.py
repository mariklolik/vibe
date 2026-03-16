"""Agent infrastructure for the autonomous research pipeline.

Each agent is a focused skill that makes fresh-context calls to Claude
via the local proxy at localhost:3456. Supports effort levels, budget caps,
fallback models, and session pooling.
"""

from src.agents.base import BaseAgent
from src.agents.client import (
    AgentResponse,
    create_client,
    call_agent,
    call_agent_structured,
    check_proxy_health,
    check_proxy_metrics,
    get_token_usage,
    DEFAULT_MODEL,
    EFFORT_LOW,
    EFFORT_MEDIUM,
    EFFORT_HIGH,
    EFFORT_MAX,
)
from src.agents.research import ResearchAgent
from src.agents.experiment import ExperimentAgent
from src.agents.writer import WriterAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.orchestrator import Orchestrator

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "create_client",
    "call_agent",
    "call_agent_structured",
    "check_proxy_health",
    "check_proxy_metrics",
    "get_token_usage",
    "DEFAULT_MODEL",
    "EFFORT_LOW",
    "EFFORT_MEDIUM",
    "EFFORT_HIGH",
    "EFFORT_MAX",
    "ResearchAgent",
    "ExperimentAgent",
    "WriterAgent",
    "ReviewerAgent",
    "Orchestrator",
]
