# tau_bench/agents/meta_controller_agent.py
#
# HA-TTS: Hybrid Adaptive Test-Time Scaling
#
# The MetaControllerAgent is a wrapper agent that:
#   1. Reads the task instruction (without touching the env/user simulator)
#   2. Estimates difficulty using the shared DifficultyEstimator (LLM-based or keyword fallback)
#   3. Routes to the appropriate TTS sub-agent based on difficulty
#
# Routing table:
#   easy      -> baseline ToolCallingAgent          (1x cost)
#   medium    -> Adaptive Budget Forcing agent      (~1.5-2x cost)
#   hard      -> ReactReflection agent              (~1.2-1.5x cost)
#   very_hard -> [stub -> falls back to ABF]        (teammate will implement)
#
# TEAM INTEGRATION:
#   Person 4 (very_hard): replace the stub import below with your agent.

from typing import List, Dict, Any, Optional

from tau_bench.agents.base import Agent
from tau_bench.agents.difficulty import DifficultyEstimator, DifficultyTier
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.agents.chat_react_agent import ChatReActAgent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult


# ── Graceful imports ──────────────────────────────────────────────────────────

try:
    from tau_bench.agents.adaptive_budget_agent import AdaptiveBudgetForcingAgent
    _ABF_AVAILABLE = True
except ImportError:
    _ABF_AVAILABLE = False

try:
    from tau_bench.agents.react_reflection_agent import ReactReflectionAgent
    _REFLECTION_AVAILABLE = True
except ImportError:
    _REFLECTION_AVAILABLE = False

# Stub for very_hard tier — teammate will implement
# Replace this block with your agent import when ready:
#   from tau_bench.agents.your_agent import YourVeryHardAgent
#   _VERY_HARD_AVAILABLE = True
_VERY_HARD_AVAILABLE = False


# ── MetaControllerAgent ────────────────────────────────────────────────────────

class MetaControllerAgent(Agent):
    """
    HA-TTS: Hybrid Adaptive Test-Time Scaling agent.

    Estimates task difficulty from the instruction and routes each task
    to the cheapest TTS strategy that can handle it reliably.

    Routing:
        easy      -> ToolCallingAgent (baseline, 1x cost)
        medium    -> AdaptiveBudgetForcingAgent (~1.5-2x cost)
        hard      -> ReactReflectionAgent (~1.2-1.5x cost)
        very_hard -> [stub, falls back to ABF] (teammate will implement)
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

        # ── easy: baseline ToolCallingAgent ───────────────────────────────────
        self.baseline_agent = ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )

        # ── medium: ABF agent ────────────────────────────────────────────────
        if _ABF_AVAILABLE:
            self.abf_agent = AdaptiveBudgetForcingAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        else:
            self.abf_agent = ChatReActAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                use_reasoning=True,
                temperature=temperature,
            )

        # ── hard: ReactReflection agent ──────────────────────────────────────
        if _REFLECTION_AVAILABLE:
            self.hard_agent = ReactReflectionAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
                reflection_interval=4,  # reflect every 4 tool calls
            )
        else:
            # Fallback to ABF if reflection agent not available
            self.hard_agent = self.abf_agent

        # ── very_hard: stub (teammate will implement) ────────────────────────
        # When your agent is ready, replace this with:
        #   self.very_hard_agent = YourVeryHardAgent(tools_info, wiki, model, ...)
        self.very_hard_agent = self.abf_agent  # Temporary fallback

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
        if difficulty == "easy":
            return self.baseline_agent.solve(env, task_index, max_num_steps)
        elif difficulty == "medium":
            return self.abf_agent.solve(env, task_index, max_num_steps)
        elif difficulty == "hard":
            return self.hard_agent.solve(env, task_index, max_num_steps)
        else:  # very_hard
            return self.very_hard_agent.solve(env, task_index, max_num_steps)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _log_routing(
        self,
        task_index: Optional[int],
        difficulty: DifficultyTier,
        instruction: str,
    ) -> None:
        """Print a compact routing summary for each task."""
        strategy_map = {
            "easy": (
                "baseline (ToolCalling)",
                "1x",
            ),
            "medium": (
                "AdaptiveBudgetForcing" if _ABF_AVAILABLE else "ABF-stub (ReAct)",
                "~1.5-2x",
            ),
            "hard": (
                "ReactReflection" if _REFLECTION_AVAILABLE else "reflection-stub (ABF fallback)",
                "~1.2-1.5x",
            ),
            "very_hard": (
                "very_hard-stub (ABF fallback)" if not _VERY_HARD_AVAILABLE else "VeryHardAgent",
                "~2-3x" if not _VERY_HARD_AVAILABLE else "custom",
            ),
        }
        strategy_name, cost = strategy_map[difficulty]
        preview = instruction[:72] + "..." if len(instruction) > 72 else instruction

        print(
            f"\n[MetaController] task={task_index} | difficulty={difficulty} | "
            f"strategy={strategy_name} | cost={cost}"
        )
        print(f'[MetaController] instruction: "{preview}"')
