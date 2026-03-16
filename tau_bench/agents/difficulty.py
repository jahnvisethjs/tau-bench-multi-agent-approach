# tau_bench/agents/difficulty.py
#
# Shared source of truth for difficulty estimation across ALL agents.
# Every agent (ABF, PolicyGuard, PACE, ReactReflection, BestOfN, MetaController)
# imports from here — no one duplicates this logic.
#
# HOW TO USE:
#   from tau_bench.agents.difficulty import DifficultyEstimator, ABF_BUDGET_TIERS
#
# HOW TO EXTEND:
#   Add your own tier dict below and import it in your agent file.

import re
from typing import Dict, Any, Optional

# ── Difficulty tier type ───────────────────────────────────────────────────────
DifficultyTier = str  # Literal["very_easy", "easy", "medium", "hard", "very_hard"]

ALL_TIERS = ["very_easy", "easy", "medium", "hard", "very_hard"]

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
    # NOTE: "cancel" removed — already in POLICY_KEYWORDS with higher weight (+2).
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

# ABF: maps difficulty → (num_ignore, max_tokens_thinking)
ABF_BUDGET_TIERS: Dict[DifficultyTier, Dict[str, Any]] = {
    "very_easy": {
        "num_ignore": 0,
        "max_tokens_thinking": 1500,
        "description": "No forcing — trivial lookup tasks",
    },
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

# ReactReflection agent tier config
REFLECTION_TIERS: Dict[DifficultyTier, Dict[str, Any]] = {
    "very_easy": {
        "reflection_interval": 10,
        "description": "Very infrequent reflection — trivial tasks",
    },
    "easy": {
        "reflection_interval": 8,
        "description": "Infrequent reflection — simple tasks rarely need it",
    },
    "medium": {
        "reflection_interval": 5,
        "description": "Moderate reflection frequency",
    },
    "hard": {
        "reflection_interval": 4,
        "description": "Frequent reflection — complex tasks need regular checkpoints",
    },
    "very_hard": {
        "reflection_interval": 3,
        "description": "Very frequent reflection — maximum self-monitoring",
    },
}

# Best-of-N: maps difficulty → number of trajectories to sample
BON_TIERS: Dict[DifficultyTier, Dict[str, Any]] = {
    "very_easy": {
        "n": 1,
        "description": "Single attempt — trivial tasks",
    },
    "easy": {
        "n": 2,
        "description": "Two trajectories — simple tasks rarely need more",
    },
    "medium": {
        "n": 3,
        "description": "Three trajectories — moderate tasks benefit from a few samples",
    },
    "hard": {
        "n": 4,
        "description": "Four trajectories — policy-constrained tasks need more attempts",
    },
    "very_hard": {
        "n": 5,
        "description": "Five trajectories — complex multi-step tasks get maximum coverage",
    },
}

# PolicyGuard: maps difficulty → max retries for critic
POLICY_GUARD_TIERS: Dict[DifficultyTier, Dict[str, Any]] = {
    "very_easy": {
        "max_retries": 1,
        "description": "Minimal policy checking",
    },
    "easy": {
        "max_retries": 2,
        "description": "Standard policy checking",
    },
    "medium": {
        "max_retries": 2,
        "description": "Standard policy checking",
    },
    "hard": {
        "max_retries": 2,
        "description": "Standard policy checking",
    },
    "very_hard": {
        "max_retries": 3,
        "description": "Extra policy retries for complex tasks",
    },
}


# ── Single shared estimator ────────────────────────────────────────────────────
class DifficultyEstimator:
    """
    Estimates task difficulty from a natural-language instruction string.

    Returns one of: "very_easy" | "easy" | "medium" | "hard" | "very_hard"

    Scoring:
      +2  if instruction implies > 3 actions (multi-item, bulk tasks)
      +2  if policy-constrained keywords detected
      +1  if modification-type task (cancel/modify/change/...)
      +1  if ambiguous / vague language detected

    Thresholds:
      0-1  -> very_easy
      2    -> easy
      3-4  -> medium
      5-6  -> hard
      7+   -> very_hard
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
            return "very_easy"
        elif score == 2:
            return "easy"
        elif score <= 4:
            return "medium"
        elif score <= 6:
            return "hard"
        else:
            return "very_hard"

    def estimate_with_llm(
        self,
        instruction: str,
        model: str,
        base_url: str = "http://localhost:8005/v1",
    ) -> DifficultyTier:
        """
        LLM-based difficulty estimation. Uses a single short LLM call
        to classify task difficulty more accurately than keyword heuristics.

        Falls back to keyword-based estimate() on any failure.
        """
        try:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key="EMPTY")
            prompt = (
                "Rate the difficulty of this customer service task. Consider:\n"
                "- How many tool calls are needed?\n"
                "- Are there complex policy rules involved (cancellation eligibility, "
                "payment constraints, refund conditions)?\n"
                "- Does the user have conflicting or ambiguous requirements?\n"
                "- Are there multi-step dependencies (search -> select -> book -> confirm)?\n\n"
                "Difficulty levels:\n"
                "1 = VERY_EASY: Trivial lookup (check status, get details)\n"
                "2 = EASY: Single action with clear policy (simple return, basic change)\n"
                "3 = MEDIUM: Standard 2-3 step task (booking, modification with one constraint)\n"
                "4 = HARD: Multi-step with policy constraints (cancel with conditions, "
                "multi-item exchange, split payments)\n"
                "5 = VERY_HARD: Complex multi-step with conflicting constraints, "
                "multiple policy checks, or ambiguous user intent\n\n"
                f'Task: "{instruction[:500]}"\n\n'
                "Reply with ONLY the number (1, 2, 3, 4, or 5):"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r"[1-5]", text)
            if match:
                tier_map = {
                    1: "very_easy",
                    2: "easy",
                    3: "medium",
                    4: "hard",
                    5: "very_hard",
                }
                return tier_map[int(match.group())]
        except Exception as e:
            print(f"[DifficultyEstimator] LLM estimation failed ({e}), using keyword fallback")

        # Fallback to keyword-based estimation
        return self.estimate(instruction)

    def _estimate_action_count(self, instruction: str) -> int:
        """Rough heuristic: count numeric/plural words that imply many actions."""
        numbers = re.findall(
            r"\b(all|every|each|\d+|two|three|four|five)\b", instruction
        )
        if numbers:
            return max(3, len(numbers) * 2)
        multi_step_clues = ["and then", "also", "additionally", "as well", "plus"]
        return 1 + sum(1 for clue in multi_step_clues if clue in instruction)
