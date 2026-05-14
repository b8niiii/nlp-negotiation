"""
Metric dataclasses and aggregation utilities.

Defines the data structures used across quantitative and qualitative evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics


@dataclass
class SessionMetrics:
    """Metrics for a single negotiation session."""
    session_id: str
    config: str
    phase: int
    outcome: str                         # "deal" | "no_deal" | "max_turns_reached"
    final_price: Optional[float]
    bug_disclosed: bool
    bug_discovered: bool
    turns: int
    avg_message_length: float            # average words per message
    vocabulary_size: int                 # unique words across transcript
    seller_concession_count: int         # number of price reductions by seller
    buyer_concession_count: int          # number of price increases by buyer
    llm_judge_scores: list[float] = field(default_factory=list)   # per-turn appropriateness
    imitation_similarity: Optional[float] = None   # cosine sim to few-shot examples (Phase 2)


@dataclass
class ConfigSummary:
    """Aggregated metrics across all sessions for one configuration."""
    config: str
    phase: int
    n_sessions: int
    agreement_rate: float                # % ending in "deal"
    mean_price: Optional[float]
    std_price: Optional[float]
    bug_disclosure_rate: float           # % seller disclosed bug
    bug_discovery_rate: float            # % buyer discovered bug
    mean_turns: float
    std_turns: float
    mean_msg_length: float


def aggregate_sessions(sessions: list[SessionMetrics]) -> list[ConfigSummary]:
    """
    Aggregate SessionMetrics into per-config summaries.

    Args:
        sessions: List of SessionMetrics from evaluation.

    Returns:
        List of ConfigSummary, one per unique (config, phase) combination.
    """
    from itertools import groupby

    summaries = []
    key_fn = lambda s: (s.config, s.phase)
    for (config, phase), group in groupby(sorted(sessions, key=key_fn), key=key_fn):
        group = list(group)
        n = len(group)

        deals = [s for s in group if s.outcome == "deal"]
        prices = [s.final_price for s in deals if s.final_price is not None]
        turns = [s.turns for s in group]
        msg_lengths = [s.avg_message_length for s in group]

        summaries.append(ConfigSummary(
            config=config,
            phase=phase,
            n_sessions=n,
            agreement_rate=len(deals) / n,
            mean_price=statistics.mean(prices) if prices else None,
            std_price=statistics.stdev(prices) if len(prices) > 1 else None,
            bug_disclosure_rate=sum(s.bug_disclosed for s in group) / n,
            bug_discovery_rate=sum(s.bug_discovered for s in group) / n,
            mean_turns=statistics.mean(turns),
            std_turns=statistics.stdev(turns) if len(turns) > 1 else 0.0,
            mean_msg_length=statistics.mean(msg_lengths),
        ))

    return summaries
