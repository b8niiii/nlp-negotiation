"""
Phase 1 experiment runner — Baseline Zero-Shot Negotiation.

Runs all 4 persona configurations x N negotiations each.
Saves all transcripts + CoT logs to data/raw/<run_id>/
and a summary CSV to data/results/<run_id>_summary.csv
"""

import argparse # parse command-line arguments
from src.utils.deepseek_client import DeepSeekClient
from src.simulation.dialogue_loop import run_experiment
from src.logging.transcript_logger import TranscriptLogger


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline negotiations.")
    parser.add_argument("--runs", type=int, default=20, help="Negotiations per config (default: 20)")
    parser.add_argument("--configs", nargs="+", default=None, help="Config keys to run (default: all)")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID for output files")
    args = parser.parse_args()

    print("=== Phase 1: Baseline Zero-Shot Negotiation ===")
    print(f"Runs per config: {args.runs}")
    print(f"Configurations: {args.configs or 'all (A, B, C, D)'}")

    client = DeepSeekClient()
    outcomes = run_experiment(
        configs=args.configs,
        runs_per_config=args.runs,
        phase=1,
        learned_tactics=None,
        client=client,
    )

    logger = TranscriptLogger()
    run_id = logger.save_batch(outcomes, run_id=args.run_id)
    print(f"\nPhase 1 complete. Run ID: {run_id}")
    print(f"  Transcripts: data/raw/{run_id}/")
    print(f"  Summary:     data/results/{run_id}_summary.csv")


if __name__ == "__main__":
    main()
