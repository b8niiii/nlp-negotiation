"""
Prompt updater — stores and serves learned tactics for Phase 2.

Saves extracted tactics to disk so they can be loaded and injected
into Phase 2 agent system prompts without re-running the Observer.
"""

import json # to serialize and deserialize the tactics
import os


class PromptUpdater:
    """
    Persists and retrieves learned tactics for Phase 2 prompt injection.
    """

    def __init__(self, tactics_path: str = "data/processed/learned_tactics.json"):
        self.tactics_path = tactics_path

    def save_tactics(self, tactics: dict) -> None:
        """
        Save extracted tactics to disk.

        Args:
            tactics: Dict with keys "seller" and "buyer", each with
                     {"description": str, "example": str}.
        """
        os.makedirs(os.path.dirname(self.tactics_path), exist_ok=True)
        with open(self.tactics_path, "w", encoding="utf-8") as f:
            json.dump(tactics, f, indent=2, ensure_ascii=False)
        print(f"Tactics saved to {self.tactics_path}")

    def load_tactics(self) -> dict:
        """
        Load previously saved tactics.

        Returns:
            Dict with keys "seller" and "buyer".
        """
        if not os.path.exists(self.tactics_path):
            raise FileNotFoundError(
                f"No tactics file found at {self.tactics_path}. "
                "Run Phase 2 tactic extraction first."
            )
        with open(self.tactics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # returns true if the tactics file exists - useful to check if we already have learned tactics
    def tactics_exist(self) -> bool:
        return os.path.exists(self.tactics_path)
