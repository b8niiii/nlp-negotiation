"""
Negotiating agent (Buyer or Seller).

Instantiated with a role, a scenario config, and a persona config.
Optionally accepts a learned tactic snippet for Phase 2 (social learning).
"""

from src.agents.base_agent import BaseAgent
from src.utils.deepseek_client import DeepSeekClient


SELLER_PROMPT_TEMPLATE = """You are a Seller in a negotiation for the following product:
Product: {product}
Asking price: {asking_price} {currency}
Minimum acceptable price: {minimum_acceptable_price} {currency}

Private information (do NOT disclose unless strategically necessary):
{private_info}

Persona: {persona_description}
{persona_addon}

{tactic_section}

Instructions:
- Negotiate in natural language, one message at a time.
- Do not reveal your minimum acceptable price or private information spontaneously.
- Aim to reach an agreement above your minimum price.
- If no deal is possible, you may end negotiations politely.
- Keep responses concise (2-5 sentences per turn).

Outcome tags (mandatory):
- When you formally accept a price and close the deal, append on a new line at the very end of your message: [DEAL: <price>]
- When you decide to walk away and end negotiations, append on a new line at the very end of your message: [NO DEAL]
- Use these tags ONLY for firm, final decisions. Do not use them speculatively or in hypothetical examples.
"""

# persona addon is a series of behavioral specifics and tactics of the agent
BUYER_PROMPT_TEMPLATE = """You are a Buyer in a negotiation for the following product:
Product: {product}
Your maximum budget: {budget} {currency}
Your target price: {target_price} {currency}

Private information:
{private_info}

Persona: {persona_description}
{persona_addon} 

{tactic_section}

Instructions:
- Negotiate in natural language, one message at a time.
- Do not reveal your maximum budget or target price.
- Probe for hidden issues, bugs, or risks in the product.
- Use any discovered issues as leverage to negotiate a lower price.
- If the price remains above your budget or risks are too high, walk away.
- Keep responses concise (2-5 sentences per turn).

Outcome tags (mandatory):
- When you formally accept a price and close the deal, append on a new line at the very end of your message: [DEAL: <price>]
- When you decide to walk away and end negotiations, append on a new line at the very end of your message: [NO DEAL]
- Use these tags ONLY for firm, final decisions. Do not use them speculatively or in hypothetical examples.
"""

TACTIC_SECTION_TEMPLATE = """
--- Learned Strategy (from successful past negotiations) ---
Previous successful {role}s used the following approach:
{tactic_description}

Example exchange from a successful negotiation:
{tactic_example}

Apply this strategy as a starting point, adapting it to the current context.
---"""


class NegotiatingAgent(BaseAgent): # this class inherits from BaseAgent
    """
    A Buyer or Seller agent with a defined persona and optional learned tactic.
    """

    def __init__(
        self,
        role: str,           # "seller" or "buyer"
        scenario: dict,      # from scenarios.yaml
        persona: dict,       # from personas.yaml
        client: DeepSeekClient,
        learned_tactic: dict | None = None,   # Phase 2: tactis is a dictionary with the structure {"description": ..., "example": ...}
        temperature: float = 0.7,
    ):
        self.scenario = scenario
        self.persona = persona
        self.learned_tactic = learned_tactic

        system_prompt = self.build_system_prompt(
            role=role,
            scenario=scenario,
            persona=persona,
            learned_tactic=learned_tactic,
        )
        super().__init__(role=role, system_prompt=system_prompt, client=client, temperature=temperature) # to define attributes from the parent class BaseAgent, such as role, system_prompt, client, temperature

    def build_system_prompt(self, role: str, scenario: dict, persona: dict, learned_tactic=None) -> str:
        tactic_section = ""
        if learned_tactic:
            tactic_section = TACTIC_SECTION_TEMPLATE.format(
                role=role,
                tactic_description=learned_tactic.get("description"),
                tactic_example=learned_tactic.get("example"),
            )

        currency = scenario.get("currency", "EUR")

        if role == "seller":
            return SELLER_PROMPT_TEMPLATE.format(
                product=scenario["seller"]["product"],
                asking_price=scenario["seller"]["asking_price"],
                minimum_acceptable_price=scenario["seller"]["minimum_acceptable_price"],
                currency=currency,
                private_info=scenario["seller"]["private_info"],
                persona_description=persona["description"],
                persona_addon=persona["system_prompt_addon"],
                tactic_section=tactic_section,
            )
        elif role == "buyer":
            return BUYER_PROMPT_TEMPLATE.format(
                product=scenario["seller"]["product"],
                budget=scenario["buyer"]["budget"],
                target_price=scenario["buyer"]["target_price"],
                currency=currency,
                private_info=scenario["buyer"]["private_info"],
                persona_description=persona["description"],
                persona_addon=persona["system_prompt_addon"],
                tactic_section=tactic_section,
            )
        else:
            raise ValueError(f"Unknown role: {role}. Expected 'seller' or 'buyer'.")
