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
from typing import Dict, Any, Optional

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
    # NOTE: "cancel" removed — already in POLICY_KEYWORDS with higher weight (+2).
    # Having it in both lists inflated difficulty scores for simple cancellations.
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

# ReactReflection agent tier config
REFLECTION_TIERS: Dict[DifficultyTier, Dict[str, Any]] = {
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

# Person 4 (very_hard agent): add your tier config here when ready
# VERY_HARD_TIERS: Dict[DifficultyTier, Dict[str, Any]] = { ... }


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

        Args:
            instruction: the task instruction text
            model: vLLM model name/path
            base_url: vLLM endpoint URL
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
                "1 = EASY: Simple lookup or single action (check status, get details)\n"
                "2 = MEDIUM: Standard 2-3 step task (simple booking, basic return)\n"
                "3 = HARD: Multi-step with policy constraints (cancel with conditions, "
                "multi-item exchange, split payments)\n"
                "4 = VERY_HARD: Complex multi-step with conflicting constraints, "
                "multiple policy checks, or ambiguous user intent\n\n"
                f'Task: "{instruction[:500]}"\n\n'
                "Reply with ONLY the number (1, 2, 3, or 4):"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r"[1-4]", text)
            if match:
                tier_map = {1: "easy", 2: "medium", 3: "hard", 4: "very_hard"}
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
