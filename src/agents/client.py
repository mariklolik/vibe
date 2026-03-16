"""Anthropic SDK client wrapper for the research pipeline.

Connects to the local Claude Max proxy at 127.0.0.1:3456.
Exploits all proxy features: session pool, effort levels, budget caps,
dynamic tools, fallback models, rate limit detection, real token usage.
"""

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import anthropic

logger = logging.getLogger(__name__)

# Default proxy configuration
DEFAULT_BASE_URL = "http://127.0.0.1:3456"
DEFAULT_MODEL = "sonnet"  # Proxy normalizes: sonnet, opus, haiku

# Effort levels: low (fast/cheap) → max (opus-quality)
EFFORT_LOW = "low"
EFFORT_MEDIUM = "medium"
EFFORT_HIGH = "high"
EFFORT_MAX = "max"

# Cumulative token usage tracker
_token_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}


@dataclass
class AgentResponse:
    """Structured response from an agent call."""
    text: str
    model: str
    stop_reason: str
    usage: dict = field(default_factory=dict)
    raw_content: list = field(default_factory=list)

    def parse_json_block(self, tag: str = "json") -> Optional[dict]:
        """Extract a JSON block from the response text.

        Uses multiple strategies, including bracket-counting for large
        responses with embedded code blocks that break simple regex.
        """
        import re

        # Strategy 1: Find ```json ... ``` using bracket-counting
        # (regex fails when JSON values contain ``` themselves)
        marker = f"```{tag}"
        idx = self.text.find(marker)
        if idx >= 0:
            # Find the opening { after the marker
            brace_start = self.text.find("{", idx + len(marker))
            if brace_start >= 0:
                result = self._extract_json_by_brackets(self.text[brace_start:])
                if result is not None:
                    return result

        # Strategy 2: Try XML-style tag
        pattern = rf'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, self.text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find the outermost {...} in the entire response
        brace_start = self.text.find("{")
        if brace_start >= 0:
            result = self._extract_json_by_brackets(self.text[brace_start:])
            if result is not None:
                return result

        # Strategy 4: Try parsing entire response as JSON
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _extract_json_by_brackets(text: str) -> Optional[dict]:
        """Extract JSON object using bracket-counting (handles nested code).

        Tracks brace depth while respecting JSON string escaping,
        so embedded Python code with braces doesn't break extraction.
        """
        if not text or text[0] != "{":
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                if in_string:
                    escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try repairing common JSON issues
                        repaired = AgentResponse._repair_json(candidate)
                        if repaired is not None:
                            return repaired
                        return None
        return None

    @staticmethod
    def _repair_json(text: str) -> Optional[dict]:
        """Try to repair common JSON issues from LLM output."""
        import re
        # Fix unescaped newlines inside string values
        # Replace literal newlines inside JSON strings with \\n
        fixed = re.sub(
            r'"([^"]*?)(?:\n)([^"]*?)"',
            lambda m: '"' + m.group(0)[1:-1].replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
            text,
        )
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # More aggressive: extract just the key fields we need
        # Try to find "content" or "files" fields
        try:
            # Find "filename" and "content" fields
            fn_match = re.search(r'"filename"\s*:\s*"([^"]+)"', text)
            # Find content between "content": " and the next top-level key or closing brace
            ct_match = re.search(r'"content"\s*:\s*"', text)
            if fn_match and ct_match:
                start = ct_match.end()
                # Find the end of the content string using bracket counting
                depth = 0
                esc = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if esc:
                        esc = False
                        continue
                    if ch == '\\':
                        esc = True
                        continue
                    if ch == '"':
                        # This is the closing quote of the content string
                        content_raw = text[start:i]
                        # Unescape JSON string
                        try:
                            content = json.loads(f'"{content_raw}"')
                        except json.JSONDecodeError:
                            content = content_raw.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                        return {
                            "filename": fn_match.group(1),
                            "content": content,
                            "description": "extracted via repair",
                        }
        except Exception:
            pass

        return None


def get_token_usage() -> dict:
    """Get cumulative token usage across all calls."""
    return dict(_token_usage)


def _make_local_request(url: str, timeout: float = 5) -> Optional[dict]:
    """Make an HTTP GET request bypassing system proxy for localhost."""
    old_http = os.environ.pop("HTTP_PROXY", None)
    old_https = os.environ.pop("HTTPS_PROXY", None)
    old_no = os.environ.get("NO_PROXY", "")
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    try:
        resp = httpx.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Request to {url} failed: {e}")
    finally:
        if old_http:
            os.environ["HTTP_PROXY"] = old_http
        if old_https:
            os.environ["HTTPS_PROXY"] = old_https
        if old_no:
            os.environ["NO_PROXY"] = old_no
    return None


def check_proxy_health(base_url: str = DEFAULT_BASE_URL) -> dict:
    """Check proxy health, rate limits, and session pool status."""
    result = _make_local_request(f"{base_url}/health")
    return result if result else {"status": "unreachable"}


def check_proxy_metrics(base_url: str = DEFAULT_BASE_URL) -> dict:
    """Get detailed proxy metrics."""
    result = _make_local_request(f"{base_url}/metrics")
    return result if result else {}


def create_client(
    base_url: Optional[str] = None,
    api_key: str = "x",
) -> anthropic.Anthropic:
    """Create an Anthropic client connected to the local proxy."""
    url = base_url or os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)

    # Bypass system HTTP proxy for local connections
    old_http = os.environ.pop("HTTP_PROXY", None)
    old_https = os.environ.pop("HTTPS_PROXY", None)
    old_no = os.environ.get("NO_PROXY", "")
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))

    if old_http:
        os.environ["HTTP_PROXY"] = old_http
    if old_https:
        os.environ["HTTPS_PROXY"] = old_https
    if old_no:
        os.environ["NO_PROXY"] = old_no

    return anthropic.Anthropic(base_url=url, api_key=api_key, http_client=http_client)


def call_agent(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    effort: Optional[str] = None,
    max_budget_usd: Optional[float] = None,
    fallback_model: Optional[str] = None,
    tools: Optional[list] = None,
) -> AgentResponse:
    """Make a single fresh-context call to Claude via proxy.

    Supports all proxy features: effort levels, budget caps,
    fallback models, dynamic tools, session pooling.

    Args:
        client: Anthropic client instance
        system_prompt: Agent role/instructions
        user_message: Task + context
        model: Model name (sonnet, opus, haiku)
        max_tokens: Maximum response tokens
        temperature: Sampling temperature
        effort: Effort level (low/medium/high/max)
        max_budget_usd: Maximum spend for this call
        fallback_model: Fallback if primary model fails
        tools: Dynamic tool definitions (Anthropic format)
    """
    logger.info(
        f"Calling agent (model={model}, max_tokens={max_tokens}"
        f"{f', effort={effort}' if effort else ''}"
        f"{f', budget=${max_budget_usd}' if max_budget_usd else ''})"
    )

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        text_parts = []
        try:
            # Build kwargs — use extra_body for proxy-specific features
            # (effort, budget, fallback) so we keep the SDK's robust streaming
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            # Proxy features via extra_body (SDK passes them through)
            extra = {}
            if effort:
                extra["effort"] = effort
            if max_budget_usd:
                extra["max_budget_usd"] = max_budget_usd
            if fallback_model:
                extra["fallback_model"] = fallback_model
            if extra:
                kwargs["extra_body"] = extra

            response_model = model
            stop_reason = "end_turn"
            usage = {}

            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    text_parts.append(text)

                final = stream.get_final_message()
                response_model = final.model
                stop_reason = final.stop_reason
                if hasattr(final, "usage") and final.usage:
                    usage = {
                        "input_tokens": getattr(final.usage, "input_tokens", 0),
                        "output_tokens": getattr(final.usage, "output_tokens", 0),
                        "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0),
                    }

            _update_usage(usage)

            full_text = "".join(text_parts)
            logger.info(f"Response: {len(full_text)} chars, model={response_model}, stop={stop_reason}")

            return AgentResponse(
                text=full_text,
                model=response_model,
                stop_reason=stop_reason,
                usage=usage,
                raw_content=[],
            )

        except Exception as e:
            last_error = e
            if text_parts:
                partial = "".join(text_parts)
                logger.warning(
                    f"Stream interrupted (attempt {attempt+1}/{max_retries}), "
                    f"returning partial ({len(partial)} chars): {e}"
                )
                return AgentResponse(
                    text=partial, model=model, stop_reason="partial",
                    raw_content=[],
                )

            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"Attempt {attempt+1} failed, retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries} attempts failed: {e}")

    raise last_error


def call_agent_structured(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    output_tag: str = "json",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.3,
    effort: Optional[str] = None,
    max_budget_usd: Optional[float] = None,
    fallback_model: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    """Call Claude and extract structured JSON output."""
    response = call_agent(
        client=client,
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        effort=effort,
        max_budget_usd=max_budget_usd,
        fallback_model=fallback_model,
    )

    parsed = response.parse_json_block(output_tag)
    return parsed, response.text


def _update_usage(usage: dict):
    """Update cumulative token usage tracker."""
    _token_usage["input"] += usage.get("input_tokens", 0)
    _token_usage["output"] += usage.get("output_tokens", 0)
    _token_usage["cache_read"] += usage.get("cache_read_input_tokens", 0)
    _token_usage["cache_creation"] += usage.get("cache_creation_input_tokens", 0)
