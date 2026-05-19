"""
Observer / Arbitrator agent.

Used in Phase 2 to evaluate completed negotiations, score each agent's
performance, select the best runs, and extract winning tactics.
"""

from src.agents.base_agent import BaseAgent
from src.utils.deepseek_client import DeepSeekClient


OBSERVER_SYSTEM_PROMPT = """You are an expert negotiation analyst. Your role is to:
1. Read negotiation transcripts between a Seller and a Buyer.
2. Score each agent's performance objectively.
3. Identify and articulate the strategies that led to successful outcomes.
4. Extract reusable tactic descriptions and representative dialogue examples.

Be precise, structured, and analytical. Always justify your scores with
specific references to the dialogue.
"""

SCORING_PROMPT_TEMPLATE = """Analyse the following negotiation transcript and provide scores.

SCENARIO: {scenario_description}
SELLER PERSONA: {seller_persona}
BUYER PERSONA: {buyer_persona}

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Provide your analysis in the following JSON format:
{{
  "outcome": "deal" | "no_deal",
  "final_price": <number or null>,
  "bug_disclosed": true | false,
  "bug_discovered_by_buyer": true | false,
  "turns": <number>,
  "seller_score": <1-5>,
  "seller_score_rationale": "<explanation>",
  "buyer_score": <1-5>,
  "buyer_score_rationale": "<explanation>",
  "overall_quality": "high" | "medium" | "low",
  "notable_tactics": ["<tactic 1>", "<tactic 2>"]
}}
"""

OUTCOME_VERIFICATION_PROMPT = """Read the following negotiation transcript and determine the outcome.

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Answer ONLY with valid JSON, no other text:
{{
  "outcome": "deal" | "no_deal",
  "final_price": <number or null>
}}

Rules:
- "deal" only if both parties explicitly agreed on a specific price.
- "no_deal" if any party walked away, refused, or the conversation ended without a clear mutual agreement.
- final_price must be the agreed number, or null if no deal.
"""

TACTIC_EXTRACTION_PROMPT = """Based on the following successful negotiations for the {role} role,
extract a reusable strategy description and a representative example snippet.

{successful_transcripts}

Respond in JSON format:
{{
  "role": "{role}",
  "tactic_description": "<3-5 sentence abstract strategy description>",
  "tactic_example": "<a short representative dialogue excerpt, 4-6 turns>"
}}
"""


class ObserverAgent(BaseAgent):
    """
    Observer/Arbitrator that evaluates negotiations and extracts tactics.
    Used exclusively in Phase 2 (social learning pipeline).
    """

    def __init__(self, client: DeepSeekClient, temperature: float = 0.3):
        # Lower temperature for more consistent, analytical outputs
        system_prompt = self.build_system_prompt()
        super().__init__(
            role="observer",
            system_prompt=system_prompt,
            client=client,
            temperature=temperature,
        )

    def build_system_prompt(self, **kwargs) -> str:
        return OBSERVER_SYSTEM_PROMPT

    def score_negotiation(
        self,
        transcript: str,
        scenario_description: str,
        seller_persona: str,
        buyer_persona: str,
    ) -> str:
        """
        Score a single negotiation transcript.

        Returns:
            Raw JSON string from the model (parse downstream).
        """
        self.reset()
        prompt = SCORING_PROMPT_TEMPLATE.format(
            transcript=transcript,
            scenario_description=scenario_description,
            seller_persona=seller_persona,
            buyer_persona=buyer_persona,
        )
        turn = self.respond(prompt)
        return turn.content

    def verify_outcome(self, transcript_text: str) -> dict:
        """
        Lightweight outcome verification using a base (non-reasoning) model.

        Reads the completed transcript and returns the ground-truth outcome
        and final price. Designed to be called with a deepseek-chat client
        (not deepseek-reasoner) — this is a simple classification task that
        does not require chain-of-thought reasoning.

        Args:
            transcript_text: Plain-text transcript (role: content, one per line).

        Returns:
            Dict with keys:
              "outcome"     -> "deal" | "no_deal"
              "final_price" -> float | None
            On parse failure, also includes "raw" with the unparsed model output,
            and "outcome" / "final_price" are both None (caller keeps keyword result).
        """
        import json
        self.reset()
        prompt = OUTCOME_VERIFICATION_PROMPT.format(transcript=transcript_text)
        turn = self.respond(prompt)
        try:
            return json.loads(turn.content)
        except json.JSONDecodeError:
            # Model returned non-JSON — keep keyword-detected outcome, log raw for inspection
            return {"outcome": None, "final_price": None, "raw": turn.content}

    def extract_tactics(self, role: str, successful_transcripts: str) -> str:
        """
        Extract reusable tactic description from the best negotiations for a role.

        Args:
            role: "seller" or "buyer"
            successful_transcripts: concatenated text of top-scoring transcripts

        Returns:
            Raw JSON string with tactic description and example.
        """
        self.reset()
        prompt = TACTIC_EXTRACTION_PROMPT.format(
            role=role,
            successful_transcripts=successful_transcripts,
        )
        turn = self.respond(prompt)
        return turn.content
