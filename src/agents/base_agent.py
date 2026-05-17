"""
Base LLM agent.

Defines the common interface and behaviour for all agents in the simulation.
Subclasses implement specific roles (negotiator, observer, etc.).
"""

from abc import ABC, abstractmethod # to create an abstract class which the subclasses have to implement 
from dataclasses import dataclass, field
from typing import Optional

from src.utils.deepseek_client import DeepSeekClient, DeepSeekResponse


@dataclass
class Turn:
    """A single turn in a dialogue, including the private CoT."""
    role: str               # "seller" | "buyer" | "observer"
    content: str            # The visible message sent to the other agent
    reasoning: Optional[str] = None   # Private CoT (never shared)
    turn_index: int = 0


class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents.

    Maintains a message history for context, handles API calls via
    DeepSeekClient, and separates private reasoning from visible output.
    """

    def __init__(
        self,
        role: str,
        system_prompt: str,
        client: DeepSeekClient,
        temperature: float = 0.7,
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.temperature = temperature
        self._history: list[dict] = []   # Only visible content (no CoT)

    def reset(self) -> None:
        """Clear conversation history for a new negotiation session."""
        self._history = []

    def _build_messages(self, new_user_message: Optional[str] = None) -> list[dict]:
        """
        Build the messages list for the API call.
        System prompt + history + optional new incoming message. This is an helper method
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._history)
        if new_user_message:
            messages.append({"role": "user", "content": new_user_message})
        return messages

    def respond(self, incoming_message: Optional[str] = None) -> Turn:
        """
        Generate a response to an incoming message (or open the dialogue if None).

        The incoming message is added to history as a "user" turn.
        The response content is added to history as an "assistant" turn.
        The reasoning_content is returned in the Turn but NOT stored in history.

        Returns:
            Turn object with visible content and private reasoning.
        """
        messages = self._build_messages(incoming_message)
        response: DeepSeekResponse = self.client.chat(
            messages=messages,
            temperature=self.temperature,
        )

        # Update visible history only
        if incoming_message:
            self._history.append({"role": "user", "content": incoming_message})
        self._history.append({"role": "assistant", "content": response.content})

        return Turn(
            role=self.role,
            content=response.content,
            reasoning=response.reasoning_content,
            turn_index=len(self._history) // 2,
        )

    @abstractmethod
    def build_system_prompt(self, **kwargs) -> str:
        """Subclasses define how to construct their system prompt."""
        ...
