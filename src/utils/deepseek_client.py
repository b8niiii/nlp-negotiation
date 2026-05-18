"""
DeepSeek API client wrapper.

Handles all communication with the DeepSeek API, separating the
reasoning (chain-of-thought) content from the final response content.
Uses the OpenAI-compatible API interface: this is useful because it allows us to use
all the client libraries and tools developed for the OpenAI API, which has become a
de facto standard in the field.

Retry strategy: transient errors (HTTP 429 rate limit, 500/503 server errors)
are retried with exponential backoff + random jitter. This is essential when
running many sessions in parallel, as concurrent requests may occasionally
exceed the API rate limit even with a semaphore in place.
"""

import os
import time
import random
from dataclasses import dataclass # decorator that automatically generates
                                # __init__ , __repr__ and other common methods
from typing import Optional

from openai import OpenAI # OpenAI-compatible API interface
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DeepSeekResponse:
    """Structured response from DeepSeek-R1, separating CoT from final content."""
    content: str                        # The final response (sent to other agent)
    reasoning_content: Optional[str]    # The private chain-of-thought (logged only)
    model: str                          # Stores the model name
    prompt_tokens: int                  # Stores the number of prompt tokens
    completion_tokens: int              # Stores the number of completion tokens


class DeepSeekClient:
    """
    Thin wrapper around the DeepSeek API.

    Automatically extracts reasoning_content (private CoT) from the response
    and returns it separately from the visible content. This enables the
    private Chain-of-Thought mechanism: reasoning is logged offline but
    never forwarded to the opposing agent.

    The chat() method includes automatic retry with exponential backoff for
    transient API errors (rate limits, temporary server errors). This makes
    the client safe to use from multiple threads simultaneously.
    """

    # HTTP status codes that are transient and worth retrying
    RETRYABLE_STATUS_CODES = {"429", "500", "502", "503", "504"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url) # OpenAI-compatible API interface

    def _is_retryable(self, error: Exception) -> bool:
        """
        Return True if the error is transient and the request should be retried.

        Checks for rate-limit (429) and server-side errors (5xx) by inspecting
        the string representation of the exception, which always contains the
        HTTP status code when raised by the openai library.
        """
        error_str = str(error)
        return any(code in error_str for code in self.RETRYABLE_STATUS_CODES)

    def chat(
        self,
        messages: list[dict], # example: [{"role": "user", "content": "Ciao"}]
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> DeepSeekResponse:
        """
        Send a chat completion request and return a structured response
        with reasoning and content separated.

        Retries automatically on transient errors using exponential backoff
        with random jitter: wait = 2^attempt + uniform(0, 1) seconds.
        This spreads out retries from concurrent threads and avoids the
        "thundering herd" problem where many threads retry simultaneously.

        Args:
            messages:   List of {"role": ..., "content": ...} dicts.
                        Should NOT include the private CoT of the other agent.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens for the completion.

        Returns:
            DeepSeekResponse with .content (visible) and .reasoning_content (private).

        Raises:
            RuntimeError: if all retry attempts are exhausted.
            Exception:    immediately for non-retryable errors.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                choice = response.choices[0] # Chooses the first response
                message = choice.message      # Extract the object containing message.role, message.content and message.reasoning_content

                # DeepSeek-R1 returns reasoning in a dedicated field
                reasoning = getattr(message, "reasoning_content", None)

                return DeepSeekResponse(
                    content=message.content,
                    reasoning_content=reasoning,
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

            except Exception as e:
                if not self._is_retryable(e):
                    # Non-transient error (e.g. bad API key, malformed request) — raise immediately
                    raise

                last_error = e
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s (+ random jitter up to 1s)
                # Jitter prevents multiple threads from retrying at the exact same moment
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[DeepSeekClient] Retryable error on attempt {attempt + 1}/{self.max_retries}: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

        raise RuntimeError(
            f"[DeepSeekClient] All {self.max_retries} retry attempts exhausted. Last error: {last_error}"
        )
