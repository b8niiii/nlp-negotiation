"""
DeepSeek API client wrapper.

Handles all communication with the DeepSeek API, separating the
reasoning (chain-of-thought) content from the final response content.
Uses the OpenAI-compatible API interface.
"""

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DeepSeekResponse:
    """Structured response from DeepSeek-R1, separating CoT from final content."""
    content: str                        # The final response (sent to other agent)
    reasoning_content: Optional[str]    # The private chain-of-thought (logged only)
    model: str
    prompt_tokens: int
    completion_tokens: int


class DeepSeekClient:
    """
    Thin wrapper around the DeepSeek API.

    Automatically extracts reasoning_content (private CoT) from the response
    and returns it separately from the visible content. This enables the
    private Chain-of-Thought mechanism: reasoning is logged offline but
    never forwarded to the opposing agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set. Check your .env file.")

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> DeepSeekResponse:
        """
        Send a chat completion request and return a structured response
        with reasoning and content separated.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
                      Should NOT include the private CoT of the other agent.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for the completion.

        Returns:
            DeepSeekResponse with .content (visible) and .reasoning_content (private).
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        message = choice.message

        # DeepSeek-R1 returns reasoning in a dedicated field
        reasoning = getattr(message, "reasoning_content", None)

        return DeepSeekResponse(
            content=message.content or "",
            reasoning_content=reasoning,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
