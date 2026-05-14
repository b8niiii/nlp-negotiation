"""
Quantitative evaluation.

Computes numerical metrics from session data:
agreement rate, price distributions, disclosure rates,
turn counts, and basic linguistic complexity measures.
"""

import json
import os
import re
from pathlib import Path

from src.evaluation.metrics import SessionMetrics, ConfigSummary, aggregate_sessions


def compute_session_metrics(session_data: dict) -> SessionMetrics:
    """
    Compute quantitative metrics for a single session loaded from JSON.

    Args:
        session_data: Dict loaded from a session JSON file.

    Returns:
        SessionMetrics dataclass.
    """
    transcript = session_data.get("transcript", [])
    messages = [t["content"] for t in transcript]
    words = [len(m.split()) for m in messages]

    avg_msg_length = sum(words) / len(words) if words else 0.0
    all_words = " ".join(messages).lower().split()
    vocabulary_size = len(set(all_words))

    # Count concessions: price mentions that move in the expected direction
    seller_prices = []
    buyer_prices = []
    for turn in transcript:
        prices = _extract_prices(turn["content"])
        if prices:
            if turn["role"] == "seller":
                seller_prices.extend(prices)
            elif turn["role"] == "buyer":
                buyer_prices.extend(prices)

    seller_concessions = _count_decreasing_sequence(seller_prices)
    buyer_concessions = _count_increasing_sequence(buyer_prices)

    return SessionMetrics(
        session_id=session_data["session_id"],
        config=session_data["config"],
        phase=session_data["phase"],
        outcome=session_data["outcome"],
        final_price=session_data.get("final_price"),
        bug_disclosed=session_data.get("bug_disclosed", False),
        bug_discovered=session_data.get("bug_discovered", False),
        turns=session_data.get("turns", len(transcript)),
        avg_message_length=avg_msg_length,
        vocabulary_size=vocabulary_size,
        seller_concession_count=seller_concessions,
        buyer_concession_count=buyer_concessions,
    )


def evaluate_run(run_dir: str) -> list[SessionMetrics]:
    """
    Load and evaluate all sessions from a run directory.

    Args:
        run_dir: Path to data/raw/<run_id>/

    Returns:
        List of SessionMetrics.
    """
    metrics = []
    for path in Path(run_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics.append(compute_session_metrics(data))
    return metrics


def summarise_run(run_dir: str) -> list[ConfigSummary]:
    """Load, evaluate, and aggregate a full run into per-config summaries."""
    metrics = evaluate_run(run_dir)
    return aggregate_sessions(metrics)


# --- helpers ---

def _extract_prices(text: str) -> list[float]:
    pattern = r'(\d[\d,\.]*)\s*(?:EUR|€|euro|euros)?'
    results = []
    for match in re.findall(pattern, text, re.IGNORECASE):
        try:
            price = float(match.replace(',', ''))
            if 100 < price < 1_000_000:
                results.append(price)
        except ValueError:
            continue
    return results


def _count_decreasing_sequence(values: list[float]) -> int:
    """Count how many consecutive pairs decrease (seller concessions)."""
    return sum(1 for a, b in zip(values, values[1:]) if b < a)


def _count_increasing_sequence(values: list[float]) -> int:
    """Count how many consecutive pairs increase (buyer concessions)."""
    return sum(1 for a, b in zip(values, values[1:]) if b > a)
