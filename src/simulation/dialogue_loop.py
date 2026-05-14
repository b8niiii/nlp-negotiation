"""
Dialogue loop — batch experiment runner.

Runs multiple NegotiationSessions across configurations and phases,
collecting all outcomes into a list for downstream analysis.
"""

import uuid
from tqdm import tqdm

import yaml

from src.agents.negotiating_agent import NegotiatingAgent
from src.simulation.session import NegotiationSession, SessionOutcome
from src.utils.deepseek_client import DeepSeekClient


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(
    scenario_path: str = "config/scenarios.yaml",
    personas_path: str = "config/personas.yaml",
    configs: list[str] | None = None,       # e.g. ["A", "B"] — None = all
    runs_per_config: int = 20,
    phase: int = 1,
    learned_tactics: dict | None = None,    # Phase 2: {"seller": {...}, "buyer": {...}}
    client: DeepSeekClient | None = None,
) -> list[SessionOutcome]:
    """
    Run a batch of negotiations across the specified configurations.

    Args:
        scenario_path: Path to scenarios.yaml
        personas_path: Path to personas.yaml
        configs: List of config keys to run (default: all)
        runs_per_config: Number of negotiations per configuration
        phase: 1 for baseline, 2 for social learning
        learned_tactics: Optional tactic dict for Phase 2 prompt injection
        client: DeepSeekClient instance (created from env if None)

    Returns:
        List of SessionOutcome objects, one per negotiation run.
    """
    if client is None:
        client = DeepSeekClient()

    scenarios = load_config(scenario_path)
    personas_config = load_config(personas_path)

    scenario = scenarios["software_sale"]
    all_configs = personas_config["configurations"]
    persona_defs = {k: v for k, v in personas_config.items() if k != "configurations"}

    if configs is None:
        configs = list(all_configs.keys())

    outcomes: list[SessionOutcome] = []

    for config_key in configs:
        config = all_configs[config_key]
        seller_persona = persona_defs[config["seller"]]
        buyer_persona = persona_defs[config["buyer"]]

        print(f"\n[Config {config_key}] {config['description']}")
        print(f"  Seller: {seller_persona['name']} | Buyer: {buyer_persona['name']}")

        for i in tqdm(range(runs_per_config), desc=f"Config {config_key}"):
            seller = NegotiatingAgent(
                role="seller",
                scenario=scenario,
                persona=seller_persona,
                client=client,
                learned_tactic=learned_tactics.get("seller") if learned_tactics else None,
            )
            buyer = NegotiatingAgent(
                role="buyer",
                scenario=scenario,
                persona=buyer_persona,
                client=client,
                learned_tactic=learned_tactics.get("buyer") if learned_tactics else None,
            )

            session = NegotiationSession(
                session_id=str(uuid.uuid4()),
                config_name=config_key,
                seller=seller,
                buyer=buyer,
                max_turns=scenario.get("max_turns", 15),
                phase=phase,
            )

            outcome = session.run()
            outcomes.append(outcome)

    return outcomes
