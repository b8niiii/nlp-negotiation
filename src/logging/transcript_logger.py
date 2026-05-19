"""
Transcript logger.

Saves negotiation outcomes (visible transcripts + private CoT logs)
to structured JSON files and a summary CSV for easy analysis.
"""

import json
import csv
import os
from datetime import datetime

from src.simulation.session import SessionOutcome


class TranscriptLogger:
    """
    Persists negotiation outcomes to disk.

    Directory layout:
        data/raw/<run_id>/
            <session_id>.json      ← full transcript + CoT
        data/results/
            <run_id>_summary.csv   ← one row per session, key metrics
    """

    def __init__(self, raw_dir: str = "data/raw", results_dir: str = "data/results"):
        self.raw_dir = raw_dir
        self.results_dir = results_dir

    def _ensure_dirs(self, run_id: str) -> str:
        run_dir = os.path.join(self.raw_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        return run_dir

    def save_session(self, outcome: SessionOutcome, run_id: str) -> str:
        """
        Save a single session's full data (transcript + CoT) to JSON.

        Returns:
            Path to the saved file.
        """
        run_dir = self._ensure_dirs(run_id)
        file_path = os.path.join(run_dir, f"{outcome.session_id}.json")

        data = {
            "session_id": outcome.session_id,
            "run_id": run_id,
            "config": outcome.config_name,
            "seller_persona": outcome.seller_persona,
            "buyer_persona": outcome.buyer_persona,
            "phase": outcome.phase,
            "outcome": outcome.outcome,
            "outcome_verified": outcome.outcome_verified,
            "final_price": outcome.final_price,
            "bug_disclosed": outcome.bug_disclosed,
            "bug_discovered": outcome.bug_discovered,
            "turns": outcome.turns,
            "transcript": outcome.transcript,
            "cot_log": outcome.cot_log,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False) # serializes the dictionary into json in order to save it in a file

        return file_path

    def save_batch(self, outcomes: list[SessionOutcome], run_id: str | None = None) -> str:
        """
        Save a full batch of sessions and write a summary CSV.

        Args:
            outcomes: List of SessionOutcome objects.
            run_id: Identifier for this experiment run (auto-generated if None).

        Returns:
            The run_id used.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Saving {len(outcomes)} sessions to run '{run_id}'...")
        for outcome in outcomes:
            self.save_session(outcome, run_id)

        self._write_summary_csv(outcomes, run_id)
        print(f"Done. Data saved to data/raw/{run_id}/ and data/results/{run_id}_summary.csv")
        return run_id

    def _write_summary_csv(self, outcomes: list[SessionOutcome], run_id: str) -> None:
        csv_path = os.path.join(self.results_dir, f"{run_id}_summary.csv")
        fieldnames = [
            "session_id", "config", "seller_persona", "buyer_persona",
            "phase", "outcome", "outcome_verified", "final_price", "bug_disclosed",
            "bug_discovered", "turns",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            # creates a writer CSV that accepts dictionaries, the advantage is that 
            # the order of the columns is guaranteed to match the order of the keys in the dictionary
            writer = csv.DictWriter(f, fieldnames=fieldnames) 

            writer.writeheader() # writes the header row
            
            for o in outcomes: # iterate through all the outcomes
                writer.writerow({
                    "session_id": o.session_id,
                    "config": o.config_name,
                    "seller_persona": o.seller_persona,
                    "buyer_persona": o.buyer_persona,
                    "phase": o.phase,
                    "outcome": o.outcome,
                    "outcome_verified": o.outcome_verified,
                    "final_price": o.final_price,
                    "bug_disclosed": o.bug_disclosed,
                    "bug_discovered": o.bug_discovered,
                    "turns": o.turns,
                })
    # in phase 2 we use the following two methods to load the data back and to format them in a human (chatbot) readable way
    # so that the observer agent can use them as context
    # these methods are used in the TacticExtractor module
    @staticmethod
    def load_session(path: str) -> dict: # deserialize the JSON file into a Python dictionary
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def format_transcript(session_data: dict) -> str:
        """Return a human-readable transcript string for a session."""
        lines = [f"=== Session {session_data['session_id']} ==="]
        lines.append(f"Config: {session_data['config']} | Phase: {session_data['phase']}")
        lines.append(f"Seller: {session_data['seller_persona']} | Buyer: {session_data['buyer_persona']}")
        lines.append(f"Outcome: {session_data['outcome']} | Price: {session_data['final_price']}")
        lines.append(f"Bug disclosed: {session_data['bug_disclosed']} | Bug discovered: {session_data['bug_discovered']}")
        lines.append("")
        for turn in session_data["transcript"]:
            lines.append(f"[Turn {turn['turn']}] {turn['role'].upper()}")
            lines.append(turn["content"])
            lines.append("")
        return "\n".join(lines)
