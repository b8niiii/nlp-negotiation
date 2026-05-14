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
