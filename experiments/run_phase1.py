"""
Phase 1 experiment runner — Baseline Zero-Shot Negotiation.

Runs all 4 persona configurations x N negotiations each, in parallel.
Saves all transcripts + CoT logs to data/raw/<run_id>/
and a summary CSV to data/results/<run_id>_summary.csv
"""

import argparse
from src.utils.deepseek_client import DeepSeekClient
from src.simulation.dialogue_loop import run_experiment
from src.logging.transcript_logger import TranscriptLogger


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline negotiations.")
    parser.add_argument("--runs", type=int, default=20, help="Negotiations per config (default: 20)")
    parser.add_argument("--configs", nargs="+", default=None, help="Config keys to run (default: all)")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID for output files")
    parser.add_argument("--workers", type=int, default=10,
                        help="Max concurrent negotiations / API calls (default: 10). "
                             "Lower this if you hit rate limits.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable LLM outcome verification (use keyword detection only). "
                             "Faster and cheaper, but may produce incorrect outcome labels.")
    args = parser.parse_args()

    print("=== Phase 1: Baseline Zero-Shot Negotiation ===")
    print(f"Runs per config:   {args.runs}")
    print(f"Configurations:    {args.configs or 'all (A, B, C, D)'}")
    print(f"Parallel workers:  {args.workers}")
    print(f"Outcome verify:    {'disabled' if args.no_verify else 'enabled (deepseek-chat)'}")

    client = DeepSeekClient()
    outcomes = run_experiment(
        configs=args.configs,
        runs_per_config=args.runs,
        phase=1,
        learned_tactics=None,
        client=client,
        max_workers=args.workers,
        verify_outcomes=not args.no_verify,
    )

    logger = TranscriptLogger()
    run_id = logger.save_batch(outcomes, run_id=args.run_id)
    print(f"\nPhase 1 complete. Run ID: {run_id}")
    print(f"  Transcripts: data/raw/{run_id}/")
    print(f"  Summary:     data/results/{run_id}_summary.csv")


if __name__ == "__main__":
    main()
