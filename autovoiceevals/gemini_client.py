"""Gemini client for calling the production model.

Wraps the google-genai SDK to call Gemini 2.5 Flash (or any Gemini model)
for single-turn evaluation. Each call sends a system prompt + conversation
context and returns the model's single response.
"""

from __future__ import annotations

import time
import warnings

# Suppress Python version warnings from google-auth
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

from google import genai
from google.genai import types


class GeminiClient:
    """Client for calling Gemini models via Google AI Studio."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 500,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        conversation_context: list[dict],
    ) -> str:
        """Send system prompt + conversation context to Gemini, return ONE response.

        Args:
            system_prompt: The voice agent's system prompt.
            conversation_context: List of message dicts with "role" and "content".
                Roles should be "user" or "assistant".

        Returns:
            The model's text response.
        """
        # Convert conversation context to Gemini Content objects
        # Gemini uses "model" instead of "assistant"
        contents = []
        for msg in conversation_context:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str
                if attempt < self.max_retries:
                    if is_rate_limit:
                        wait = min(30 * (2 ** attempt), 300)
                        print(
                            f"      (Gemini rate limited, waiting {wait}s — "
                            f"retry {attempt + 1}/{self.max_retries})"
                        )
                    else:
                        wait = min(2 ** attempt, 30)
                        print(
                            f"      (Gemini retry {attempt + 1}/{self.max_retries} "
                            f"after {wait}s: {type(e).__name__})"
                        )
                    time.sleep(wait)
                else:
                    raise

        return ""  # unreachable
