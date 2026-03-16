"""Base agent class implementing the fresh-context loop pattern.

Each agent call is independent — no accumulated conversation history.
State persists via progress.txt and project files only.

Supports proxy features: effort levels, budget caps, fallback models.
"""

import logging
from typing import Optional

import anthropic

from src.agents.client import (
    AgentResponse,
    call_agent,
    call_agent_structured,
    DEFAULT_MODEL,
    EFFORT_MEDIUM,
)
from src.state.progress import read_progress, append_progress

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all research pipeline agents.

    Each agent has:
    - A system prompt defining its role (like skills in research_claude_agents)
    - Access to a project directory for state
    - A fresh context per call (rom4ik pattern)
    - Configurable effort/budget/fallback for proxy optimization
    """

    name: str = "base"
    system_prompt: str = "You are a helpful research assistant."

    # Default proxy settings per agent (subclasses override)
    default_effort: Optional[str] = EFFORT_MEDIUM
    default_fallback_model: Optional[str] = None
    default_max_budget_usd: Optional[float] = None

    def __init__(
        self,
        client: anthropic.Anthropic,
        project_dir: str,
        model: str = DEFAULT_MODEL,
    ):
        self.client = client
        self.project_dir = project_dir
        self.model = model

    def _build_context(self, task: str, extra_context: str = "") -> str:
        """Build the user message with progress state and task.

        This is the key pattern: every call gets fresh context
        from progress.txt, not from conversation history.
        """
        progress = read_progress(self.project_dir)

        parts = []
        if progress:
            parts.append(f"## Current Progress\n{progress}")
        if extra_context:
            parts.append(f"## Additional Context\n{extra_context}")
        parts.append(f"## Task\n{task}")

        return "\n\n".join(parts)

    def call(
        self,
        task: str,
        extra_context: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        effort: Optional[str] = ...,
        max_budget_usd: Optional[float] = ...,
        fallback_model: Optional[str] = ...,
        tools: Optional[list] = None,
    ) -> AgentResponse:
        """Make a single fresh-context call to Claude.

        Args:
            task: What to do this iteration
            extra_context: Additional context (e.g., paper abstracts, results)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            effort: Proxy effort level (None=default, "low"/"medium"/"high"/"max")
            max_budget_usd: Budget cap for this call
            fallback_model: Fallback if primary model unavailable
            tools: Dynamic tool definitions (Anthropic format)
        """
        # Use agent defaults for sentinel values
        if effort is ...:
            effort = self.default_effort
        if max_budget_usd is ...:
            max_budget_usd = self.default_max_budget_usd
        if fallback_model is ...:
            fallback_model = self.default_fallback_model

        user_message = self._build_context(task, extra_context)

        logger.info(f"[{self.name}] Calling (model={self.model}, effort={effort}, task={task[:80]}...)")

        response = call_agent(
            client=self.client,
            system_prompt=self.system_prompt,
            user_message=user_message,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
            max_budget_usd=max_budget_usd,
            fallback_model=fallback_model,
            tools=tools,
        )

        logger.info(
            f"[{self.name}] Response: {len(response.text)} chars, "
            f"model={response.model}, stop={response.stop_reason}, "
            f"tokens={response.usage}"
        )
        return response

    def call_structured(
        self,
        task: str,
        extra_context: str = "",
        output_tag: str = "json",
        max_tokens: int = 8192,
        temperature: float = 0.3,
        effort: Optional[str] = ...,
        max_budget_usd: Optional[float] = ...,
        fallback_model: Optional[str] = ...,
    ) -> tuple[Optional[dict], str]:
        """Call Claude and extract structured JSON output.

        Returns:
            Tuple of (parsed_dict_or_None, raw_text)
        """
        # Use agent defaults for sentinel values
        if effort is ...:
            effort = self.default_effort
        if max_budget_usd is ...:
            max_budget_usd = self.default_max_budget_usd
        if fallback_model is ...:
            fallback_model = self.default_fallback_model

        user_message = self._build_context(task, extra_context)

        return call_agent_structured(
            client=self.client,
            system_prompt=self.system_prompt,
            user_message=user_message,
            output_tag=output_tag,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
            max_budget_usd=max_budget_usd,
            fallback_model=fallback_model,
        )

    def log_progress(self, entry: str):
        """Append to progress.txt with agent name as stage tag."""
        append_progress(self.project_dir, entry, stage=self.name)
