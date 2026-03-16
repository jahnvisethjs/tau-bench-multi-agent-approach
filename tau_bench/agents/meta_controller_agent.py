# tau_bench/agents/meta_controller_agent.py
#
# HA-TTS: Hybrid Adaptive Test-Time Scaling
#
# The MetaControllerAgent is a wrapper agent that:
#   1. Reads the task instruction (without touching the env/user simulator)
#   2. Estimates difficulty using the shared DifficultyEstimator (LLM-based or keyword fallback)
#   3. Routes to the appropriate TTS sub-agent based on 5 difficulty tiers
#
# Routing table (5-tier):
#   very_easy -> ABF (minimal budget)             (~0% overhead)
#   easy      -> PolicyGuard (policy critic)       (~10-20% overhead)
#   medium    -> PACE (constraint register)        (~20-30% overhead)
#   hard      -> ReactReflection (periodic review) (~30-50% overhead)
#   very_hard -> Best-of-N (N=2, retry)           (~2x cost)

from typing import List, Dict, Any, Optional

from tau_bench.agents.base import Agent
from tau_bench.agents.difficulty import DifficultyEstimator, DifficultyTier
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult


# ── Graceful imports ──────────────────────────────────────────────────────────

try:
    from tau_bench.agents.adaptive_budget_agent import AdaptiveBudgetForcingAgent
    _ABF_AVAILABLE = True
except ImportError:
    _ABF_AVAILABLE = False

try:
    from tau_bench.agents.policy_guard_agent import PolicyGuardAgent
    _POLICY_GUARD_AVAILABLE = True
except ImportError:
    _POLICY_GUARD_AVAILABLE = False

try:
    from tau_bench.agents.pace_agent import PaceAgent
    _PACE_AVAILABLE = True
except ImportError:
    _PACE_AVAILABLE = False

try:
    from tau_bench.agents.react_reflection_agent import ReactReflectionAgent
    _REFLECTION_AVAILABLE = True
except ImportError:
    _REFLECTION_AVAILABLE = False

try:
    from tau_bench.agents.best_of_n_agent import BestOfNAgent
    _BON_AVAILABLE = True
except ImportError:
    _BON_AVAILABLE = False


# ── MetaControllerAgent ────────────────────────────────────────────────────────

class MetaControllerAgent(Agent):
    """
    HA-TTS: Hybrid Adaptive Test-Time Scaling agent.

    Estimates task difficulty from the instruction and routes each task
    to the cheapest TTS strategy that can handle it reliably.

    5-Tier Routing:
        very_easy -> ABF (minimal budget, ~0% overhead)
        easy      -> PolicyGuard (policy critic, ~10-20% overhead)
        medium    -> PACE (constraint register, ~20-30% overhead)
        hard      -> ReactReflection (periodic review, ~30-50% overhead)
        very_hard -> Best-of-N (N=2 retry, ~2x cost)
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

        # Shared difficulty estimator
        self.estimator = DifficultyEstimator()

        # ── very_easy: ABF with minimal budget ─────────────────────────────
        if _ABF_AVAILABLE:
            self.very_easy_agent = AdaptiveBudgetForcingAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        else:
            self.very_easy_agent = ToolCallingAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )

        # ── easy: PolicyGuard agent ────────────────────────────────────────
        if _POLICY_GUARD_AVAILABLE:
            self.easy_agent = PolicyGuardAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
                max_retries=2,
            )
        else:
            self.easy_agent = self.very_easy_agent  # fallback

        # ── medium: PACE agent ─────────────────────────────────────────────
        if _PACE_AVAILABLE:
            self.medium_agent = PaceAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        else:
            self.medium_agent = self.very_easy_agent  # fallback

        # ── hard: ReactReflection agent ────────────────────────────────────
        if _REFLECTION_AVAILABLE:
            self.hard_agent = ReactReflectionAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
                reflection_interval=4,
            )
        else:
            self.hard_agent = self.very_easy_agent  # fallback

        # ── very_hard: Best-of-N agent (N=2) ──────────────────────────────
        if _BON_AVAILABLE:
            self.very_hard_agent = BestOfNAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
                max_n=2,
                difficulty_override="very_hard",
            )
        else:
            self.very_hard_agent = self.hard_agent  # fallback

    # ── solve ──────────────────────────────────────────────────────────────────

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        """
        Route this task to the appropriate TTS agent based on difficulty.

        Uses LLM-based difficulty estimation (with keyword fallback).
        Reads env.tasks[task_index].instruction directly — no env.reset() call.
        """

        # ── Step 1: Read instruction without touching the env ────────────────
        if task_index is not None and task_index < len(env.tasks):
            instruction = env.tasks[task_index].instruction
        else:
            instruction = env.task.instruction

        # ── Step 2: Estimate difficulty (LLM-based with keyword fallback) ────
        try:
            difficulty: DifficultyTier = self.estimator.estimate_with_llm(
                instruction, self.model
            )
        except Exception:
            difficulty: DifficultyTier = self.estimator.estimate(instruction)

        self._log_routing(task_index, difficulty, instruction)

        # ── Step 3: Route to the right sub-agent ────────────────────────────
        routing = {
            "very_easy": self.very_easy_agent,
            "easy": self.easy_agent,
            "medium": self.medium_agent,
            "hard": self.hard_agent,
            "very_hard": self.very_hard_agent,
        }
        agent = routing.get(difficulty, self.very_easy_agent)
        return agent.solve(env, task_index, max_num_steps)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _log_routing(
        self,
        task_index: Optional[int],
        difficulty: DifficultyTier,
        instruction: str,
    ) -> None:
        """Print a compact routing summary for each task."""
        strategy_map = {
            "very_easy": (
                "ABF (minimal)" if _ABF_AVAILABLE else "ToolCalling (fallback)",
                "~0%",
            ),
            "easy": (
                "PolicyGuard" if _POLICY_GUARD_AVAILABLE else "ABF (fallback)",
                "~10-20%",
            ),
            "medium": (
                "PACE" if _PACE_AVAILABLE else "ABF (fallback)",
                "~20-30%",
            ),
            "hard": (
                "ReactReflection" if _REFLECTION_AVAILABLE else "ABF (fallback)",
                "~30-50%",
            ),
            "very_hard": (
                "BestOfN (N=2)" if _BON_AVAILABLE else "ReactReflection (fallback)",
                "~2x",
            ),
        }
        strategy_name, cost = strategy_map.get(difficulty, ("unknown", "?"))
        preview = instruction[:72] + "..." if len(instruction) > 72 else instruction

        print(
            f"\n[MetaController] task={task_index} | difficulty={difficulty} | "
            f"strategy={strategy_name} | overhead={cost}"
        )
        print(f'[MetaController] instruction: "{preview}"')
