"""
Qualitative evaluation.

Three components:
1. LLM-judge: per-turn appropriateness scoring (pragmatic adaptation signal)
2. Cosine similarity: imitation test (scripted imitation signal)
3. CoT analysis: reasoning structure detection (genuine reasoning signal)
"""

import json
import re
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.utils.deepseek_client import DeepSeekClient


# ─── LLM Judge ───────────────────────────────────────────────────────────────

LLM_JUDGE_SYSTEM = """You are an expert negotiation analyst evaluating dialogue quality.
For each agent turn, assess whether the move is strategically appropriate given
the dialogue history so far. Be concise and structured."""

LLM_JUDGE_PROMPT = """Rate the following negotiation turn on a scale of 1-3:
1 = Inappropriate or rigid (ignores context, repeats scripted language)
2 = Adequate (acceptable move but not particularly adaptive)
3 = Appropriate and adaptive (tailored to the specific dialogue state)

--- DIALOGUE SO FAR ---
{history}
--- TURN TO EVALUATE ---
{role}: {content}

Respond in JSON: {{"score": <1|2|3>, "rationale": "<one sentence>"}}"""


class LLMJudge:
    """Uses DeepSeek to score the appropriateness of each negotiation turn."""

    def __init__(self, client: DeepSeekClient, temperature: float = 0.2):
        self.client = client
        self.temperature = temperature

    def score_turn(self, history: list[dict], role: str, content: str) -> dict:
        """
        Score a single turn given the preceding dialogue history.

        Returns:
            Dict with 'score' (1-3) and 'rationale'.
        """
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in history
        )
        prompt = LLM_JUDGE_PROMPT.format(
            history=history_text or "(start of negotiation)",
            role=role.upper(),
            content=content,
        )
        messages = [
            {"role": "system", "content": LLM_JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat(messages=messages, temperature=self.temperature)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"score": None, "rationale": response.content} #if the model doesn't return a json, it returns the raw response without a score

    def score_transcript(self, transcript: list[dict]) -> list[dict]:
        """
        Score all turns in a transcript.

        Returns:
            List of dicts with turn index, role, score, and rationale.
        """
        results = []
        for i, turn in enumerate(transcript):
            history = transcript[:i]
            score_data = self.score_turn(history, turn["role"], turn["content"])
            results.append({
                "turn": turn["turn"],
                "role": turn["role"],
                **score_data, # unpacks score and rationale into the results dict
            })
        return results


# ─── Cosine Similarity (Imitation Test) ──────────────────────────────────────

def compute_imitation_similarity(
    transcript: list[dict],
    few_shot_example: str,
) -> float:
    """
    Compute cosine similarity between a negotiation transcript and
    the few-shot tactic example used in Phase 2 prompts.

    High similarity -> scripted imitation signal.

    Args:
        transcript: List of turn dicts from a session.
        few_shot_example: The example text injected into the system prompt.

    Returns:
        Cosine similarity score (0.0-1.0).
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2") # this is a pre-trained model that computes sentence embeddings
    transcript_text = " ".join(t["content"] for t in transcript)
    embeddings = model.encode([transcript_text, few_shot_example], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity, 4)


# ─── CoT Reasoning Analysis ───────────────────────────────────────────────────

REASONING_INDICATORS = {
    "conditional": [r"\bif\b.{0,80}\bthen\b", r"\bif\b.{0,80}(?:,|;)", r"\bwould\b.{0,60}\bif\b"], # checks for conditional statements in the CoT log
    "history_reference": [r"\byou (said|mentioned|asked|offered)\b", r"\bprevious(ly)?\b", r"\bearlier\b"], # checks for references to previous statements in the CoT log
    "strategic_planning": [r"\bbetter to\b", r"\bstrategy\b", r"\brisk\b.{0,40}\bbenefit\b", r"\btrade.?off\b"], # checks for strategic planning in the CoT log
    "goal_awareness": [r"\bmy (goal|objective|aim|target|priority)\b", r"\bi (need|want|must|should)\b"], # checks for goal awareness in the CoT log
}


def analyse_cot(cot_log: list[dict]) -> dict:
    """
    Analyse the private CoT log for reasoning structure indicators.

    Returns:
        Dict with indicator counts and a 'reasoning_score' (0-4).
    """
    all_cot_text = " ".join(
        (entry.get("reasoning") or "") for entry in cot_log
    ).lower()

    indicator_counts = {}
    for category, patterns in REASONING_INDICATORS.items():
        count = sum(
            len(re.findall(p, all_cot_text, re.IGNORECASE))
            for p in patterns
        )
        indicator_counts[category] = count

    # Reasoning score: how many categories have at least one match
    reasoning_score = sum(1 for v in indicator_counts.values() if v > 0)

    return {
        "indicator_counts": indicator_counts,
        "reasoning_score": reasoning_score,   # 0–4, higher = more reasoning signals
        "cot_total_chars": len(all_cot_text),
    }
