"""
Tactic extractor — Phase 2 social learning pipeline.

Uses the ObserverAgent to:
1. Score all Phase 1 negotiations
2. Select the top-performing runs per role
3. Extract a reusable tactic description + example snippet
"""

import json
from pathlib import Path

from src.agents.observer_agent import ObserverAgent
from src.logging.transcript_logger import TranscriptLogger


class TacticExtractor:
    """
    Orchestrates the Observer to score negotiations and extract winning tactics.
    """

    def __init__(self, observer: ObserverAgent):
        self.observer = observer

    def score_all_sessions(
        self,
        run_dir: str, # path to the directory containing the raw data of phase 1 as json files
        scenario_description: str = "Software sale with hidden critical bug",
    ) -> list[dict]:
        """
        Score all sessions in a run directory using the Observer.

        Returns:
            List of dicts: session metadata + Observer scores.
        """
        scored = []
        for path in sorted(Path(run_dir).glob("*.json")): 
            # load session data from JSON file and formats it into a transcript - useful for the observer agent
            session_data = TranscriptLogger.load_session(str(path))
            transcript_text = TranscriptLogger.format_transcript(session_data) 

            score_json = self.observer.score_negotiation(
                transcript=transcript_text,
                scenario_description=scenario_description,
                seller_persona=session_data.get("seller_persona", ""),
                buyer_persona=session_data.get("buyer_persona", ""),
            )

            try:
                scores = json.loads(score_json) # convert the JSON string into a python dictionary
            except json.JSONDecodeError:
                scores = {"raw": score_json}    # fallback to raw string if parsing fails (though this shouldn't happen)

            scored.append({
                "session_id": session_data["session_id"],
                "config": session_data["config"],
                "seller_persona": session_data["seller_persona"],
                "buyer_persona": session_data["buyer_persona"],
                "path": str(path),
                **scores, # unpack the scores dictionary and add its key-value pairs to the current (scored) dictionary
            })

        return scored

    def select_top_sessions(
        self,
        scored_sessions: list[dict],
        role: str,
        top_k: int = 2,
    ) -> list[dict]:
        """
        Select the top-k sessions for a given role based on Observer scores.

        Args:
            scored_sessions: Output of score_all_sessions.
            role: "seller" or "buyer"
            top_k: Number of top sessions to select.

        Returns:
            Top-k session dicts sorted by role score descending.
        """
        score_key = f"{role}_score" # seller score or buyer score - it is the score key
        valid = [s for s in scored_sessions if isinstance(s.get(score_key), (int, float))] # with isinstance() we check if the score is a number 
        sorted_sessions = sorted(valid, key=lambda s: s[score_key], reverse=True)
        return sorted_sessions[:top_k] # returns the top-k sessions as a list of dictionaries

    def extract_tactics(self, run_dir: str, top_k: int = 2) -> dict:
        """
        Full pipeline: score → select → extract tactics for both roles.

        Args:
            run_dir: Path to Phase 1 raw data directory.
            top_k: Number of top sessions to use per role.

        Returns:
            Dict with keys "seller" and "buyer", each containing
            {"description": str, "example": str}.
        """
        print("Scoring all sessions with Observer...")
        scored = self.score_all_sessions(run_dir)

        tactics = {}
        for role in ["seller", "buyer"]:
            print(f"Selecting top {top_k} sessions for role: {role}")
            top = self.select_top_sessions(scored, role=role, top_k=top_k)

            # Build combined transcript text for the top sessions
            combined = ""
            for i, session_meta in enumerate(top, 1): # i is the index, session_meta is the dictionary
                session_data = TranscriptLogger.load_session(session_meta["path"])
                combined += f"\n--- Example {i} (score: {session_meta.get(f'{role}_score')}) ---\n"
                combined += TranscriptLogger.format_transcript(session_data)

            # here we call the method from the observer class, not the extract_tactic form the tactic_extractor class
            print(f"Extracting {role} tactic from top sessions...")
            tactic_json = self.observer.extract_tactics(role=role, successful_transcripts=combined) 

            try:
                tactic = json.loads(tactic_json)
            except json.JSONDecodeError:
                tactic = {"description": tactic_json, "example": ""}
            # we store the tactic_description and tactic_example in the tactics dictionary
            tactics[role] = {
                "description": tactic.get("tactic_description", ""),
                "example": tactic.get("tactic_example", ""),
            }

        return tactics

        # Returns the tactics dict with this format: 
        # {
        #    "seller": {"description": "Seller user...", "example": "Turn 3: ..."},
        #    "buyer":  {"description": "Buyer used...", "example": "Turn 2: ..."},
        # }
