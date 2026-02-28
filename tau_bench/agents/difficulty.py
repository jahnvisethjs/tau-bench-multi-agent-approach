# tau_bench/agents/difficulty.py
#
# Shared source of truth for difficulty estimation across ALL agents.
# Every agent (ABF, Beam-PG, Refine-CA, MCTS-RE, MetaController)
# imports from here — no one duplicates this logic.
#
# HOW TO USE:
#   from tau_bench.agents.difficulty import DifficultyEstimator, ABF_BUDGET_TIERS
#
# HOW TO EXTEND (for other team members):
#   Add your own tier dict below (e.g. BEAM_TIERS) and import it in your agent file.

import re
from typing import Dict, Any

# ── Difficulty tier type ───────────────────────────────────────────────────────
# One of these four strings — used consistently across all agents and MetaController
DifficultyTier = str  # Literal["easy", "medium", "hard", "very_hard"]

ALL_TIERS = ["easy", "medium", "hard", "very_hard"]

# ── Keyword lists used by DifficultyEstimator ─────────────────────────────────
POLICY_KEYWORDS = [
    "basic economy",
    "cancel",
    "cancellation",
    "non-refundable",
    "change fee",
    "upgrade",
    "downgrade",
    "voucher",
    "credit",
]

MODIFICATION_TYPES = [
    "cancel",
    "modify",
    "change",
    "update",
    "upgrade",
    "downgrade",
    "rebook",
    "transfer",
]

VAGUE_TERMS = [
    "somehow",
    "maybe",
    "if possible",
    "not sure",
    "whichever",
]

# ── Per-agent budget tier definitions ─────────────────────────────────────────
# Person 1 (ABF): maps difficulty → (num_ignore, max_tokens_thinking)
ABF_BUDGET_TIERS: Dict[DifficultyTier, Dict[str, Any]] = {
    "easy": {
        "num_ignore": 0,
        "max_tokens_thinking": 2000,
        "description": "No forcing — simple lookup/status tasks",
    },
    "medium": {
        "num_ignore": 1,
        "max_tokens_thinking": 4000,
        "description": "One reconsideration — standard booking or modification",
    },
    "hard": {
        "num_ignore": 2,
        "max_tokens_thinking": 6000,
        "description": "Two reconsiderations — policy-constrained tasks",
    },
    "very_hard": {
        "num_ignore": 3,
        "max_tokens_thinking": 8000,
        "description": "Full forcing — multi-step complex tasks",
    },
}

# Person 2 (Beam-PG): add BEAM_TIERS here when ready
# BEAM_TIERS: Dict[DifficultyTier, Dict[str, Any]] = { ... }

# Person 3 (Refine-CA): add REFINE_TIERS here when ready
# REFINE_TIERS: Dict[DifficultyTier, Dict[str, Any]] = { ... }

# Person 4 (MCTS-RE): add MCTS_TIERS here when ready
# MCTS_TIERS: Dict[DifficultyTier, Dict[str, Any]] = { ... }


# ── Single shared estimator ────────────────────────────────────────────────────
class DifficultyEstimator:
    """
    Estimates task difficulty from a natural-language instruction string.

    Returns one of: "easy" | "medium" | "hard" | "very_hard"

    Scoring:
      +2  if instruction implies > 3 actions (multi-item, bulk tasks)
      +2  if policy-constrained keywords detected
      +1  if modification-type task (cancel/modify/change/...)
      +1  if ambiguous / vague language detected

    Thresholds:
      0-1  → easy
      2-3  → medium
      4-5  → hard
      6+   → very_hard
    """

    def estimate(self, instruction: str) -> DifficultyTier:
        text = instruction.lower()
        score = 0

        # Signal 1: multi-action task?
        if self._estimate_action_count(text) > 3:
            score += 2

        # Signal 2: policy-constrained?
        if any(kw in text for kw in POLICY_KEYWORDS):
            score += 2

        # Signal 3: modification type?
        if any(kw in text for kw in MODIFICATION_TYPES):
            score += 1

        # Signal 4: vague / ambiguous?
        if any(t in text for t in VAGUE_TERMS):
            score += 1

        if score <= 1:
            return "easy"
        elif score <= 3:
            return "medium"
        elif score <= 5:
            return "hard"
        else:
            return "very_hard"

    def _estimate_action_count(self, instruction: str) -> int:
        """Rough heuristic: count numeric/plural words that imply many actions."""
        numbers = re.findall(
            r"\b(all|every|each|\d+|two|three|four|five)\b", instruction
        )
        if numbers:
            return max(3, len(numbers) * 2)
        multi_step_clues = ["and then", "also", "additionally", "as well", "plus"]
        return 1 + sum(1 for clue in multi_step_clues if clue in instruction)
