"""
Phase 2 experiment runner — In-Context Social Learning.

Steps:
1. Load Phase 1 run data
2. Run Observer to score negotiations and extract tactics
3. Save tactics to data/processed/learned_tactics.json
4. Run negotiations with updated prompts
5. Save Phase 2 transcripts and summary
"""

import argparse
from src.utils.deepseek_client import DeepSeekClient
from src.agents.observer_agent import ObserverAgent
from src.simulation.dialogue_loop import run_experiment
from src.logging.transcript_logger import TranscriptLogger
from src.social_learning.tactic_extractor import TacticExtractor
from src.social_learning.prompt_updater import PromptUpdater


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 social learning negotiations.")
    parser.add_argument("--phase1-run-id", type=str, required=True, help="Run ID from Phase 1")
    parser.add_argument("--runs", type=int, default=20, help="Negotiations per config (default: 20)")
    parser.add_argument("--configs", nargs="+", default=None, help="Config keys to run (default: all)")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID for output files")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip tactic extraction if learned_tactics.json already exists")
    args = parser.parse_args()

    client = DeepSeekClient()
    updater = PromptUpdater()

    # Step 1: Tactic extraction
    if args.skip_extraction and updater.tactics_exist():
        print("Loading existing tactics from data/processed/learned_tactics.json ...")
        tactics = updater.load_tactics()
    else:
        print("=== Phase 2 Step 1: Tactic Extraction ===")
        run_dir = f"data/raw/{args.phase1_run_id}"
        observer = ObserverAgent(client=client)
        logger = TranscriptLogger()
        extractor = TacticExtractor(observer=observer, logger=logger)

        tactics = extractor.extract_tactics(run_dir=run_dir)
        updater.save_tactics(tactics)

        print("\nExtracted tactics:")
        for role, tactic in tactics.items():
            print(f"\n[{role.upper()}] {tactic['description'][:200]}...")

    # Step 2: Run Phase 2 negotiations with updated prompts
    print("\n=== Phase 2 Step 2: Running Negotiations with Learned Tactics ===")
    print(f"Runs per config: {args.runs}")

    outcomes = run_experiment(
        configs=args.configs,
        runs_per_config=args.runs,
        phase=2,
        learned_tactics=tactics,
        client=client,
    )

    logger = TranscriptLogger()
    run_id = logger.save_batch(outcomes, run_id=args.run_id)
    print(f"\nPhase 2 complete. Run ID: {run_id}")
    print(f"  Transcripts: data/raw/{run_id}/")
    print(f"  Summary:     data/results/{run_id}_summary.csv")


if __name__ == "__main__":
    main()
