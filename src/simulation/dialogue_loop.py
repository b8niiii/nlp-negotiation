"""
Dialogue loop — batch experiment runner.

Runs multiple NegotiationSessions across configurations and phases,
collecting all outcomes into a list for downstream analysis.

Parallelisation strategy: sessions are I/O-bound (each turn is a DeepSeek API
call), so a ThreadPoolExecutor with thread-level concurrency is appropriate.
The GIL is released during network I/O, allowing true parallel execution.
A threading.Semaphore caps concurrent API calls to avoid rate-limiting (HTTP 429).
"""

import uuid
import threading
import concurrent.futures
from tqdm import tqdm

import yaml

from src.agents.negotiating_agent import NegotiatingAgent
from src.agents.observer_agent import ObserverAgent
from src.simulation.session import NegotiationSession, SessionOutcome
from src.utils.deepseek_client import DeepSeekClient


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _run_single_session(
    config_key: str,
    seller_persona: dict,
    buyer_persona: dict,
    scenario: dict,
    client: DeepSeekClient,
    learned_tactics: dict | None,
    phase: int,
    semaphore: threading.Semaphore,
    outcome_verifier: ObserverAgent | None = None,
) -> SessionOutcome:
    """
    Create and run a single negotiation session, respecting the concurrency semaphore.

    The semaphore is acquired before any API call is made and released when the
    session completes. This caps the number of concurrent sessions regardless of
    how many threads the executor has spawned.

    If outcome_verifier is provided, the session will call it after the keyword
    loop to confirm the outcome via LLM (deepseek-chat). The verifier is shared
    across sessions — ObserverAgent is stateless between calls (reset() is called
    inside verify_outcome()), so sharing it across threads is safe.
    """
    with semaphore:
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
            outcome_verifier=outcome_verifier,
        )

        return session.run()


def run_experiment(
    scenario_path: str = "config/scenarios.yaml",
    personas_path: str = "config/personas.yaml",
    configs: list[str] | None = None,       # e.g. ["A", "B"] — None = all
    runs_per_config: int = 20,
    phase: int = 1,                         # 1 for baseline, 2 for social learning
    learned_tactics: dict | None = None,    # Phase 2: {"seller": {...}, "buyer": {...}}, None in fase 1
    client: DeepSeekClient | None = None,   # shared across all sessions (thread-safe for I/O)
    max_workers: int = 10,                  # max concurrent sessions / API calls
    verify_outcomes: bool = True,           # use LLM judge (deepseek-chat) to verify outcome after keyword loop
) -> list[SessionOutcome]:
    """
    Run a batch of negotiations across the specified configurations in parallel.

    Sessions are independent of each other and can be executed concurrently.
    Within a single session, turns remain sequential (each turn depends on the
    previous one). The thread pool therefore parallelises across sessions, not
    within them.

    Args:
        scenario_path:   Path to scenarios.yaml
        personas_path:   Path to personas.yaml
        configs:         List of config keys to run (default: all)
        runs_per_config: Number of negotiations per configuration
        phase:           1 for baseline, 2 for social learning
        learned_tactics: Optional tactic dict for Phase 2 prompt injection
        client:          DeepSeekClient instance (created from env if None)
        max_workers:     Max concurrent sessions — tune this against the
                         DeepSeek rate limit for your API tier
        verify_outcomes: If True, an ObserverAgent backed by deepseek-chat
                         verifies the keyword-detected outcome after each session.
                         Eliminates false positives from negated deal keywords
                         (e.g. "I can't accept", "no deal without guarantees").
                         Adds one cheap API call per session (~$0.001 each).

    Returns:
        List of SessionOutcome objects, one per negotiation run.
        Order is not guaranteed (futures complete in arbitrary order).
    """
    if client is None:
        client = DeepSeekClient()

    # Build outcome verifier — uses deepseek-chat (base model), not deepseek-reasoner.
    # One shared instance is safe across threads: verify_outcome() calls reset()
    # internally before each use, so there is no cross-session state leakage.
    outcome_verifier = None
    if verify_outcomes:
        verifier_client = DeepSeekClient(model="deepseek-chat")
        outcome_verifier = ObserverAgent(client=verifier_client)
        print("Outcome verifier: enabled (deepseek-chat)")
    else:
        print("Outcome verifier: disabled (keyword-only detection)")

    # Load the scenario and personas as dictionaries
    scenarios = load_config(scenario_path)
    personas_config = load_config(personas_path)

    # Get the software sale scenario and persona definitions
    scenario = scenarios["software_sale"]
    all_configs = personas_config["configurations"]
    # extract persona definitions from the personas_config dictionary, by filtering out the "configurations" key
    persona_defs = {k: v for k, v in personas_config.items() if k != "configurations"}

    if configs is None:
        configs = list(all_configs.keys())

    # Build a flat list of all (config_key, seller_persona, buyer_persona) jobs
    jobs = []
    for config_key in configs:
        config = all_configs[config_key]
        seller_persona = persona_defs[config["seller"]]
        buyer_persona = persona_defs[config["buyer"]]

        print(f"\n[Config {config_key}] {config['description']}")
        print(f"  Seller: {seller_persona['name']} | Buyer: {buyer_persona['name']}")

        for _ in range(runs_per_config):
            jobs.append((config_key, seller_persona, buyer_persona))

    # Semaphore caps concurrent API calls regardless of thread pool size.
    # Even if the executor has spawned max_workers threads, only max_workers
    # of them can be inside _run_single_session at the same time.
    semaphore = threading.Semaphore(max_workers)

    outcomes: list[SessionOutcome] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs immediately — the semaphore will pace actual execution
        futures = {
            executor.submit(
                _run_single_session,
                config_key, seller_persona, buyer_persona,
                scenario, client, learned_tactics, phase, semaphore,
                outcome_verifier,
            ): config_key
            for config_key, seller_persona, buyer_persona in jobs
        }

        # as_completed() yields each Future as it finishes (not submission order).
        # tqdm updates the progress bar on every completed negotiation.
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Running negotiations",
        ):
            outcomes.append(future.result())

    return outcomes
