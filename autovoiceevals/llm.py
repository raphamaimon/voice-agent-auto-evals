"""LLM client for Claude API calls.

Thin wrapper: handles retries, timeouts, and JSON extraction.
No domain-specific prompts live here — see evaluator.py for those.
"""

from __future__ import annotations

import json
import time

import anthropic
import httpx


def _parse_retry_after(err: anthropic.RateLimitError) -> float:
    """Extract retry-after seconds from rate limit response headers."""
    try:
        headers = err.response.headers if hasattr(err, "response") else {}
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except (ValueError, AttributeError, TypeError):
        pass
    return 0.0


class LLMClient:
    """Claude API client with retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 120,
        max_retries: int = 8,
    ):
        self.model = model
        self.max_retries = max_retries

        http = httpx.Client(
            timeout=httpx.Timeout(float(timeout), connect=30.0),
            transport=httpx.HTTPTransport(retries=max_retries),
        )
        self._client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=max_retries,
            timeout=float(timeout),
            http_client=http,
        )

    def call(self, system: str, user: str, max_tokens: int = 2048, model: str | None = None) -> str:
        """Make a Claude API call with exponential backoff retries.

        Args:
            model: Override the default model for this call (e.g. Opus for judge).
        """
        use_model = model or self.model
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.messages.create(
                    model=use_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return r.content[0].text
            except anthropic.RateLimitError as e:
                if attempt < self.max_retries:
                    retry_after = _parse_retry_after(e)
                    wait = retry_after if retry_after else min(2 * (attempt + 1), 15)
                    print(
                        f"      (rate limited, waiting {wait:.0f}s — "
                        f"retry {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIStatusError as e:
                if e.status_code == 529:  # overloaded
                    if attempt < self.max_retries:
                        wait = min(5 * (attempt + 1), 30)
                        print(
                            f"      (API overloaded, waiting {wait}s — "
                            f"retry {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(wait)
                    else:
                        raise
                elif attempt < self.max_retries:
                    wait = min(2 ** attempt, 10)
                    print(
                        f"      (retry {attempt + 1}/{self.max_retries} "
                        f"after {wait}s: {e.status_code})"
                    )
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 10)
                    print(
                        f"      (retry {attempt + 1}/{self.max_retries} "
                        f"after {wait}s: {type(e).__name__})"
                    )
                    time.sleep(wait)
                else:
                    raise
        return ""  # unreachable, but keeps type checkers happy

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 500,
    ) -> str:
        """Multi-turn conversation. Used for simulated agent conversations.

        Args:
            system: System prompt (the voice agent's prompt).
            messages: Conversation history as [{"role": "user/assistant", "content": "..."}].
            max_tokens: Max response tokens.
        """
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                return r.content[0].text
            except anthropic.RateLimitError as e:
                if attempt < self.max_retries:
                    retry_after = _parse_retry_after(e)
                    wait = retry_after if retry_after else min(2 * (attempt + 1), 15)
                    print(f"      (rate limited, waiting {wait:.0f}s)")
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < self.max_retries:
                    wait = min(5 * (attempt + 1), 30)
                    print(f"      (API overloaded, waiting {wait}s)")
                    time.sleep(wait)
                elif attempt < self.max_retries:
                    wait = min(2 ** attempt, 10)
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 10)
                    time.sleep(wait)
                else:
                    raise
        return ""

    def call_json(self, system: str, user: str, max_tokens: int = 2048, model: str | None = None):
        """Call Claude and parse the response as JSON.

        Args:
            model: Override the default model for this call (e.g. Opus for judge).
        """
        raw = self.call(system, user, max_tokens, model=model)
        return parse_json(raw)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def parse_json(raw: str):
    """Best-effort JSON extraction from LLM output.

    Handles: bare JSON, ```json fences, and embedded objects/arrays.
    Returns None if parsing fails entirely.
    """
    # Strip code fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting the first complete JSON structure
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char) + 1
        if s >= 0 and e > s:
            try:
                return json.loads(raw[s:e])
            except json.JSONDecodeError:
                pass

    return None
