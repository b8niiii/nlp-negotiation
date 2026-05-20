"""
Qualitative evaluation.

Three components:
1. LLM-judge: per-turn appropriateness scoring (pragmatic adaptation signal)
2. Cosine similarity: imitation test (scripted imitation signal)
3. CoT analysis: per-turn reasoning-structure detection (genuine reasoning signal)

The CoT analysis is performed per turn (not on a single concatenated blob),
so we can measure whether the reasoning at turn N is grounded in the dialogue
that was visible up to turn N-1.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.utils.deepseek_client import DeepSeekClient


# ─── LLM Judge ───────────────────────────────────────────────────────────────

LLM_JUDGE_SYSTEM = """You are an expert negotiation analyst evaluating dialogue quality.
For each agent turn, assess whether the move is strategically appropriate given
the dialogue history so far. Be concise and structured."""

LLM_JUDGE_PROMPT = """Rate the following negotiation turn on a scale of 1-3:
1 = Inappropriate or rigid (ignores context, repeats scripted language)
2 = Adequate (acceptable move but not particularly adaptive)
3 = Appropriate and adaptive (tailored to the specific dialogue state)

--- DIALOGUE SO FAR ---
{history}
--- TURN TO EVALUATE ---
{role}: {content}

Respond in JSON: {{"score": <1|2|3>, "rationale": "<one sentence>"}}"""


class LLMJudge:
    """Uses DeepSeek to score the appropriateness of each negotiation turn."""

    def __init__(self, client: DeepSeekClient, temperature: float = 0.2):
        self.client = client
        self.temperature = temperature

    def score_turn(self, history: list[dict], role: str, content: str) -> dict:
        """
        Score a single turn given the preceding dialogue history.

        Returns:
            Dict with 'score' (1-3) and 'rationale'.
        """
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in history
        )
        prompt = LLM_JUDGE_PROMPT.format(
            history=history_text or "(start of negotiation)",
            role=role.upper(),
            content=content,
        )
        messages = [
            {"role": "system", "content": LLM_JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat(messages=messages, temperature=self.temperature)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"score": None, "rationale": response.content} #if the model doesn't return a json, it returns the raw response without a score

    def score_transcript(self, transcript: list[dict]) -> list[dict]:
        """
        Score all turns in a transcript.

        Returns:
            List of dicts with turn index, role, score, and rationale.
        """
        results = []
        for i, turn in enumerate(transcript):
            history = transcript[:i]
            score_data = self.score_turn(history, turn["role"], turn["content"])
            results.append({
                "turn": turn["turn"],
                "role": turn["role"],
                **score_data, # unpacks score and rationale into the results dict
            })
        return results


# ─── Cosine Similarity (Imitation Test) ──────────────────────────────────────

# Module-level singleton: the model is loaded once on first call and reused for
# every subsequent call. Avoids re-loading 90 MB of weights 160× (once per
# session × role in the Phase 2 loop).
_SENTENCE_MODEL = None


def _get_sentence_model():
    """Lazy-load and cache the SentenceTransformer model at module level."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer loaded ✓  (cached for all subsequent calls)")
    return _SENTENCE_MODEL


def compute_imitation_similarity(
    transcript: list[dict],
    few_shot_example: str,
) -> float:
    """
    Compute cosine similarity between a negotiation transcript and
    the few-shot tactic example used in Phase 2 prompts.

    High similarity -> scripted imitation signal.

    Args:
        transcript: List of turn dicts from a session.
        few_shot_example: The example text injected into the system prompt.

    Returns:
        Cosine similarity score (0.0-1.0).
    """
    from sentence_transformers import util

    model = _get_sentence_model()  # reuses the cached instance, no re-loading
    transcript_text = " ".join(t["content"] for t in transcript)
    embeddings = model.encode([transcript_text, few_shot_example], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity, 4)


# ─── CoT Reasoning Analysis (Level 1, regex + dialogue grounding) ────────────

# Refined reasoning indicators. We deliberately drop the broad "I need/want/must/
# should" pattern from goal_awareness because it fires on essentially every CoT
# (every turn opens with "We need to respond as the buyer ..."), drowning the
# signal in boilerplate noise. Conditionals are kept because they at least
# reflect contingent reasoning; their *specificity* (whether they contain an
# anchor from the dialogue) is what we actually plot.
REASONING_INDICATORS = {
    # Contingent reasoning markers — "if X then Y", "X would happen if Y", etc.
    "conditional": [
        r"\bif\b.{0,80}\bthen\b",
        r"\bif\b.{0,80}(?:,|;)",
        r"\bwould\b.{0,60}\bif\b",
        r"\botherwise\b",
        r"\b(?:in case|should they)\b",
    ],
    # References to what has already happened in the dialogue
    "history_reference": [
        r"\b(?:you|they|the (?:buyer|seller))\s+(?:said|mentioned|asked|offered|claimed|stated|proposed|countered|rejected|accepted)\b",
        r"\bprevious(?:ly)?\b",
        r"\bearlier\b",
        r"\blast (?:turn|message|offer)\b",
        r"\b(?:just|already)\s+(?:said|named|offered|countered)\b",
    ],
    # Forward-looking strategic deliberation
    "strategic_planning": [
        r"\bbetter to\b",
        r"\bstrateg(?:y|ically|ic)\b",
        r"\brisk\b.{0,40}\bbenefit\b",
        r"\btrade.?off\b",
        r"\b(?:next move|next step|follow up|leverage|anchor)\b",
    ],
    # Tightly-scoped goal awareness — we drop "I need/want/must/should" because
    # they fire on every turn and produce no discriminative signal.
    "goal_awareness": [
        r"\bmy (?:goal|objective|aim|target|priority|minimum|maximum|budget|reserve)\b",
        r"\b(?:must not exceed|cannot go (?:below|above)|walk away)\b",
    ],
}

# Stop-words excluded when extracting anchor tokens from the dialogue. We want
# content-bearing terms (numbers, product/tactic words, named concepts) — not
# pronouns, articles, or auxiliaries.
_ANCHOR_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "this", "that", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they", "them", "us",
    "my", "your", "his", "her", "our", "their", "me", "him", "us",
    "so", "not", "no", "yes", "ok", "okay", "well", "just", "very", "really",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "than", "then", "there", "here", "what", "which", "who", "when", "where",
    "why", "how", "any", "some", "more", "less", "much", "many", "all", "none",
    "about", "into", "out", "up", "down", "over", "under", "again", "also",
    "only", "such", "like", "still", "even", "because", "however", "though",
    "while", "after", "before", "during", "between", "through", "above", "below",
    "off", "per", "etc", "eg", "ie", "vs",
    # very high-frequency negotiation filler that adds no anchoring signal
    "let", "lets", "us", "please", "thanks", "thank", "hello", "hi",
}

_PRICE_RE = re.compile(
    r"(?:€|\$|£)?\s*(\d{1,3}(?:[,\.]\d{3})+|\d{4,7})(?:\s*(?:eur|usd|gbp|euro|euros|dollars?))?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\b(\d{1,3})\s?%")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")  # ≥4-letter words only


def _normalise_number(raw: str) -> str:
    """Strip thousands separators and return a canonical decimal string."""
    s = raw.replace(",", "").replace(".", "") if raw.count(",") + raw.count(".") <= 1 else raw.replace(",", "")
    # Keep only the digit portion (we tolerate either thousands separator style)
    digits = re.sub(r"\D", "", s)
    return digits


def extract_anchor_tokens(transcript_slice: list[dict]) -> set[str]:
    """
    Extract the set of anchor tokens that appear in the dialogue visible up to
    (but not including) the current turn.

    An anchor token is something specific to *this* conversation that a genuinely
    reasoning agent would refer back to: a price, a percentage, or a content-
    bearing word (≥4 letters, not a stop-word). Tokens are lower-cased and
    de-duplicated.

    Args:
        transcript_slice: list of turn dicts with at least a 'content' field.
                          Pass [] for turn 0 (no history yet).

    Returns:
        Set of normalised anchor strings.
    """
    anchors: set[str] = set()
    for turn in transcript_slice:
        text = turn.get("content", "") or ""

        # Numeric anchors — prices and percentages are the most diagnostic
        for match in _PRICE_RE.finditer(text):
            digits = _normalise_number(match.group(1))
            if digits and 100 <= int(digits) < 1_000_000:
                anchors.add(digits)
        for match in _PERCENT_RE.finditer(text):
            anchors.add(f"{match.group(1)}%")

        # Lexical anchors — content words ≥4 chars, not in stopword list
        for word in _WORD_RE.findall(text.lower()):
            if word in _ANCHOR_STOPWORDS:
                continue
            anchors.add(word)

    return anchors


def _count_anchors_in_text(text: str, anchors: set[str]) -> tuple[int, set[str]]:
    """
    Count how many distinct anchors from `anchors` appear in `text`.

    Returns (count, matched_set). The match is substring-based for numbers and
    word-boundary based for lexical anchors, so "7500" matches "$7,500" / "7500"
    in the CoT but "deal" does NOT match "dealing".
    """
    if not anchors or not text:
        return 0, set()
    text_lower = text.lower()
    matched: set[str] = set()
    for anchor in anchors:
        if anchor.endswith("%"):
            # percentage anchors — match the raw "NN%" substring
            if anchor in text_lower:
                matched.add(anchor)
        elif anchor.isdigit():
            # numeric anchor — accept either raw digits or with thousands sep
            patterns = [anchor]
            if len(anchor) > 3:
                patterns.append(f"{anchor[:-3]},{anchor[-3:]}")
                patterns.append(f"{anchor[:-3]}.{anchor[-3:]}")
            if any(p in text_lower for p in patterns):
                matched.add(anchor)
        else:
            # lexical anchor — word-boundary match to avoid "dealing" → "deal"
            if re.search(rf"\b{re.escape(anchor)}\b", text_lower):
                matched.add(anchor)
    return len(matched), matched


def _conditional_specificity(cot_text: str, anchors: set[str]) -> tuple[int, int]:
    """
    Count conditional clauses in the CoT and how many of them contain at least
    one anchor token from the visible dialogue.

    A "conditional clause" is a window of up to ~120 characters starting at an
    "if" / "would ... if" / "otherwise" / "in case" trigger. We then check
    whether that window contains any anchor.

    Returns:
        (conditional_count, anchored_count)
    """
    triggers = list(re.finditer(
        r"\b(?:if|otherwise|in case|should they|would\s+\w+\s+if)\b",
        cot_text,
        flags=re.IGNORECASE,
    ))
    total = len(triggers)
    if total == 0:
        return 0, 0

    text_lower = cot_text.lower()
    anchored = 0
    for m in triggers:
        start = m.start()
        window = text_lower[start : start + 140]
        hit, _ = _count_anchors_in_text(window, anchors)
        if hit > 0:
            anchored += 1
    return total, anchored


def _count_indicators(text: str) -> dict[str, int]:
    """Apply REASONING_INDICATORS to a single CoT string."""
    counts: dict[str, int] = {}
    for category, patterns in REASONING_INDICATORS.items():
        counts[category] = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in patterns
        )
    return counts


def analyse_cot(cot_log: list[dict], transcript: list[dict]) -> list[dict]:
    """
    Per-turn structural analysis of the private CoT log.

    For each turn N, we measure whether the reasoning at turn N is *grounded*
    in the dialogue that was visible up to turn N-1. Concretely, we extract
    anchor tokens (prices, percentages, content words) from the prior turns
    and count how many of them surface inside the turn-N CoT. This is far
    more discriminative than flat regex counts over a concatenated blob:
    a generic reasoning ("we need to maximise price") scores near zero, while
    an adaptive one ("they just named 6500, which undercuts my 7000 floor")
    scores high.

    Args:
        cot_log:    List of turn dicts with 'turn', 'role', 'reasoning'.
        transcript: List of turn dicts with 'turn', 'role', 'content'.
                    Same ordering / length as cot_log.

    Returns:
        List of per-turn dicts. Each dict contains:
            - turn, role
            - cot_length: number of characters in the CoT for this turn
            - anchors_available: number of unique anchor tokens visible up to
              turn N-1
            - anchors_found: number of those anchors that appear in the CoT
            - grounding_score: anchors_found / anchors_available (0 if none
              available, e.g. turn 0)
            - conditional_count: number of conditional clauses in the CoT
            - conditional_anchored: how many of those conditionals contain an
              anchor token
            - conditional_specificity: conditional_anchored / conditional_count
              (0 if no conditionals)
            - indicator_counts: dict per REASONING_INDICATORS category
    """
    results: list[dict] = []

    # Build a turn → content lookup once. Transcript and cot_log share the
    # 'turn' field (assigned in NegotiationSession._log_turn), so we use it as
    # the join key rather than positional order — that way we are robust to
    # any future asymmetry between the two lists.
    transcript_by_turn = {t["turn"]: t for t in transcript}

    for entry in cot_log:
        turn_id = entry["turn"]
        reasoning = entry.get("reasoning") or ""
        cot_lower = reasoning.lower()

        # The "history visible to this turn" is the transcript up to (but not
        # including) the current turn. The seller's opening turn (turn 0) sees
        # no history, so anchors_available == 0 there.
        history = [t for t in transcript if t["turn"] < turn_id]
        anchors = extract_anchor_tokens(history)

        anchors_found, _matched = _count_anchors_in_text(cot_lower, anchors)
        grounding = anchors_found / len(anchors) if anchors else 0.0

        cond_total, cond_anchored = _conditional_specificity(cot_lower, anchors)
        cond_specificity = cond_anchored / cond_total if cond_total else 0.0

        indicator_counts = _count_indicators(cot_lower)

        results.append({
            "turn": turn_id,
            "role": entry.get("role"),
            "cot_length": len(reasoning),
            "anchors_available": len(anchors),
            "anchors_found": anchors_found,
            "grounding_score": round(grounding, 4),
            "conditional_count": cond_total,
            "conditional_anchored": cond_anchored,
            "conditional_specificity": round(cond_specificity, 4),
            "indicator_counts": indicator_counts,
        })

    return results


# ─── CoT Reasoning Analysis (Level 2, LLM classifier with caching) ───────────

COT_CLASSIFIER_SYSTEM = """You are an expert evaluator of LLM agent reasoning.
You will be shown the dialogue an agent could see at a given turn, plus the
agent's PRIVATE chain-of-thought for that turn (the reasoning the agent kept
to itself before producing its visible message).

Your job is to rate the chain-of-thought on two dimensions, on a 1–3 scale.
Be strict: prefer lower scores when the reasoning could be applied verbatim
to a different negotiation."""


COT_CLASSIFIER_PROMPT = """Rate the following private chain-of-thought.

context_specificity (1–3): how closely the reasoning is tied to THIS dialogue
  1 = fully generic — could be the CoT of any negotiation on any topic
  2 = uses some context superficially (mentions the role, the product, etc.)
  3 = references specific events, numbers, or positions that emerged in
      this dialogue (e.g. "they offered 6500 last turn", "the bug we discussed")

planning_depth (1–3): how forward-looking the reasoning is
  1 = no contingencies, just an immediate action
  2 = considers one alternative or a single future scenario
  3 = explicit multi-branch reasoning ("if X then Y, otherwise Z"),
      contingencies tied to specific anchors in the dialogue

--- DIALOGUE VISIBLE TO THE AGENT (up to and including the current turn for
    the speaker; the speaker is the agent whose CoT is below) ---
{dialogue}
--- PRIVATE CoT TO EVALUATE (speaker: {role}, turn {turn}) ---
{cot}

Respond ONLY with a JSON object on a single line, no markdown:
{{"context_specificity": <1|2|3>, "planning_depth": <1|2|3>, "evidence": "<one-sentence justification>"}}"""


class CoTClassifier:
    """
    LLM-based per-turn evaluator of private chain-of-thought.

    Uses `deepseek-chat` (the base model, not the reasoner) because this is a
    rating task — chain-of-thought reasoning at the classifier level adds
    latency and cost without quality benefits.

    Results are cached to disk per `run_id` so this is paid for at most once.
    Cache schema mirrors the existing `data/processed/<run_id>_judge.json`
    convention used by `LLMJudge`.
    """

    DEFAULT_CACHE_DIR = "data/processed"

    def __init__(
        self,
        client: Optional[DeepSeekClient] = None,
        temperature: float = 0.0,
        cache_dir: Optional[str] = None,
    ):
        # Default to deepseek-chat: this is a classification task, not a
        # reasoning one. If the user has the env pointed at deepseek-reasoner
        # they can override by passing their own client.
        self.client = client or DeepSeekClient(model="deepseek-chat")
        self.temperature = temperature
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)

    # ── single-call primitive ────────────────────────────────────────────────
    def classify_turn(self, cot_text: str, dialogue_context: str,
                       role: str, turn_id: int) -> dict:
        """
        Send one CoT + dialogue snippet to the classifier and parse the JSON.

        Returns a dict with `context_specificity`, `planning_depth`, `evidence`.
        On parsing failure both numeric fields are set to None and the raw
        response is stored under `evidence` so the call is still informative.
        """
        prompt = COT_CLASSIFIER_PROMPT.format(
            dialogue=dialogue_context or "(no prior dialogue — opening turn)",
            cot=cot_text or "(empty CoT)",
            role=role,
            turn=turn_id,
        )
        messages = [
            {"role": "system", "content": COT_CLASSIFIER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat(messages=messages, temperature=self.temperature)
        try:
            parsed = json.loads(response.content)
            return {
                "context_specificity": parsed.get("context_specificity"),
                "planning_depth": parsed.get("planning_depth"),
                "evidence": parsed.get("evidence", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            return {
                "context_specificity": None,
                "planning_depth": None,
                "evidence": response.content,
            }

    # ── session-level loop ───────────────────────────────────────────────────
    def classify_session(
        self,
        cot_log: list[dict],
        transcript: list[dict],
    ) -> list[dict]:
        """
        Classify every turn in a session. The dialogue context passed to the
        classifier for turn N is the transcript up to and including turn N for
        the speaker — i.e. the visible dialogue that the speaker had just
        produced. This matches the framing of the prompt ("the agent's
        CoT for this turn").
        """
        results: list[dict] = []
        for entry in cot_log:
            turn_id = entry["turn"]
            role = entry.get("role", "")
            cot_text = entry.get("reasoning") or ""

            # Visible context = everything up to AND including this turn's
            # message. This lets the classifier see what the agent produced
            # and judge whether the CoT actually shaped that move.
            visible = [t for t in transcript if t["turn"] <= turn_id]
            dialogue_context = "\n".join(
                f"{t['role'].upper()} (turn {t['turn']}): {t['content']}"
                for t in visible
            )

            scored = self.classify_turn(cot_text, dialogue_context, role, turn_id)
            results.append({
                "turn": turn_id,
                "role": role,
                **scored,
            })
        return results

    # ── full-run driver with on-disk caching ─────────────────────────────────
    def classify_run(
        self,
        run_id: str,
        raw_dir: str = "data/raw",
        sample_size: Optional[int] = None,
        force: bool = False,
    ):
        """
        Classify every (or a sample of) session in a run, with on-disk caching.

        Args:
            run_id:       e.g. "full_phase1" — used both to locate the raw
                          sessions and to name the cache file.
            raw_dir:      Root of the raw-session directory layout.
            sample_size:  If set, take up to N random sessions per config to
                          keep cost predictable. Mirrors the LLMJudge API.
            force:        If True, ignore any existing cache and re-classify.

        Returns:
            A `pandas.DataFrame` of per-turn classifier results, with columns
            (session_id, config, phase, turn, role, context_specificity,
             planning_depth, evidence).
        """
        # Imported lazily so the module remains import-cheap (pandas pulls in
        # numpy + a few MB of C extensions).
        import pandas as pd  # noqa: WPS433 (local import is intentional)
        import numpy as np   # noqa: WPS433

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{run_id}_cot_llm.json"

        if cache_path.exists() and not force:
            return pd.DataFrame(json.loads(cache_path.read_text()))

        run_dir = Path(raw_dir) / run_id
        paths = sorted(run_dir.glob("*.json"))

        # Optional stratified sub-sampling — same shape as LLMJudge.judge_run.
        if sample_size is not None:
            by_cfg: dict[str, list[Path]] = {}
            for p in paths:
                cfg = json.loads(p.read_text())["config"]
                by_cfg.setdefault(cfg, []).append(p)
            rng = np.random.default_rng(42)
            paths = [
                p
                for cfg in by_cfg
                for p in rng.choice(
                    by_cfg[cfg], min(sample_size, len(by_cfg[cfg])), replace=False
                )
            ]

        rows: list[dict] = []
        for p in paths:
            data = json.loads(p.read_text())
            session_id = data["session_id"]
            config = data["config"]
            phase = data["phase"]

            scored = self.classify_session(data["cot_log"], data["transcript"])
            for s in scored:
                rows.append({
                    "session_id": session_id,
                    "config": config,
                    "phase": phase,
                    **s,
                })

        cache_path.write_text(json.dumps(rows, indent=2))
        print(f"Cached -> {cache_path}")
        return pd.DataFrame(rows)
