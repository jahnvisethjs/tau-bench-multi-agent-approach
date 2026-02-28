# tau_bench/agents/meta_controller_agent.py
#
# HA-TTS: Hybrid Adaptive Test-Time Scaling
#
# The MetaControllerAgent is a wrapper agent that:
#   1. Reads the task instruction (without touching the env/user simulator)
#   2. Estimates difficulty using the shared DifficultyEstimator
#   3. Routes to the appropriate TTS sub-agent based on difficulty
#
# Routing table:
#   easy      → baseline ToolCallingAgent          (1× cost)
#   medium    → Adaptive Budget Forcing agent      (~1.5–2× cost)
#   hard      → Beam Search + Policy Guard         (9× cost)   [stub → fallback]
#   very_hard → MCTS + Rollout Evaluation          (50× cost)  [stub → fallback]
#
# TEAM INTEGRATION:
#   As teammates complete their agents, replace the stub imports below with
#   the real ones. The routing logic and interface stay exactly the same.
#
#   Person 1 (ABF)    → replace abf stub with:    from tau_bench.agents.adaptive_budget_agent import AdaptiveBudgetForcingAgent
#   Person 2 (Beam)   → replace beam stub with:   from tau_bench.agents.beam_pg_agent import BeamPGAgent
#   Person 3 (Refine) → (optional layer, no routing slot yet)
#   Person 4 (MCTS)   → replace mcts stub with:   from tau_bench.agents.mcts_re_agent import MCTSAgent

from typing import List, Dict, Any, Optional

from tau_bench.agents.base import Agent
from tau_bench.agents.difficulty import DifficultyEstimator, DifficultyTier
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.agents.chat_react_agent import ChatReActAgent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult


# ── Graceful stub imports ──────────────────────────────────────────────────────
# Each block tries to import a real agent; falls back to None if not built yet.
# MetaControllerAgent.solve() uses the fallback when a real agent isn't available.

try:
    from tau_bench.agents.adaptive_budget_agent import AdaptiveBudgetForcingAgent
    _ABF_AVAILABLE = True
except ImportError:
    _ABF_AVAILABLE = False

try:
    from tau_bench.agents.beam_pg_agent import BeamPGAgent
    _BEAM_AVAILABLE = True
except ImportError:
    _BEAM_AVAILABLE = False

try:
    from tau_bench.agents.mcts_re_agent import MCTSAgent
    _MCTS_AVAILABLE = True
except ImportError:
    _MCTS_AVAILABLE = False


# ── MetaControllerAgent ────────────────────────────────────────────────────────

class MetaControllerAgent(Agent):
    """
    HA-TTS: Hybrid Adaptive Test-Time Scaling agent.

    Estimates task difficulty from the instruction and routes each task
    to the cheapest TTS strategy that can handle it reliably.

    Args:
        tools_info : list of tool definitions passed from the environment
        wiki       : domain policy text passed from the environment
        model      : LLM model identifier (e.g. "Qwen/Qwen3-4B")
        provider   : LiteLLM provider string (e.g. "openai")
        temperature: sampling temperature (default 0.0)
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.tools_info  = tools_info
        self.wiki        = wiki
        self.model       = model
        self.provider    = provider
        self.temperature = temperature

        # Shared difficulty estimator (single source of truth from difficulty.py)
        self.estimator = DifficultyEstimator()

        # ── Baseline agent — always available, used for easy tasks and fallback ──
        self.baseline_agent = ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )

        # ── ABF agent (medium tasks) ───────────────────────────────────────────
        if _ABF_AVAILABLE:
            self.abf_agent = AdaptiveBudgetForcingAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        else:
            # Stub: ChatReActAgent as placeholder until Person 1's file is merged
            self.abf_agent = ChatReActAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                use_reasoning=True,
                temperature=temperature,
            )

        # ── Beam-PG agent (hard tasks) ─────────────────────────────────────────
        if _BEAM_AVAILABLE:
            self.beam_agent = BeamPGAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        else:
            # Stub: fall back to ABF (or baseline) until Person 2's file is merged
            self.beam_agent = self.abf_agent

        # ── MCTS agent (very_hard tasks) ───────────────────────────────────────
        if _MCTS_AVAILABLE:
            self.mcts_agent = MCTSAgent(
                tools_info=tools_info,
                wiki=wiki,
                model=model,
                provider=provider,
                temperature=temperature,
            )
        else:
            # Stub: fall back to ABF until Person 4's file is merged
            self.mcts_agent = self.abf_agent

    # ── solve ──────────────────────────────────────────────────────────────────

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        """
        Route this task to the appropriate TTS agent based on difficulty.

        Reads env.tasks[task_index].instruction directly — no env.reset() call,
        no side effects on the user simulator. The chosen sub-agent handles reset.
        """

        # ── Step 1: Read instruction without touching the env ──────────────────
        if task_index is not None and task_index < len(env.tasks):
            instruction = env.tasks[task_index].instruction
        else:
            # Fallback: use whatever task is currently loaded
            instruction = env.task.instruction

        # ── Step 2: Estimate difficulty ────────────────────────────────────────
        difficulty: DifficultyTier = self.estimator.estimate(instruction)

        self._log_routing(task_index, difficulty, instruction)

        # ── Step 3: Route to the right sub-agent ──────────────────────────────
        if difficulty == "easy":
            return self.baseline_agent.solve(env, task_index, max_num_steps)

        elif difficulty == "medium":
            return self.abf_agent.solve(env, task_index, max_num_steps)

        elif difficulty == "hard":
            return self.beam_agent.solve(env, task_index, max_num_steps)

        else:  # very_hard
            return self.mcts_agent.solve(env, task_index, max_num_steps)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _log_routing(
        self,
        task_index: Optional[int],
        difficulty: DifficultyTier,
        instruction: str,
    ) -> None:
        """Print a compact routing summary for each task."""
        strategy_map = {
            "easy":      ("baseline (ToolCalling)", "1×"),
            "medium":    (
                "AdaptiveBudgetForcing" if _ABF_AVAILABLE else "ABF-stub (ReAct)",
                "~1.5–2×",
            ),
            "hard":      (
                "BeamPG" if _BEAM_AVAILABLE else "beam-stub (ABF fallback)",
                "~9×",
            ),
            "very_hard": (
                "MCTS" if _MCTS_AVAILABLE else "mcts-stub (ABF fallback)",
                "~50×",
            ),
        }
        strategy_name, cost = strategy_map[difficulty]
        preview = instruction[:72] + "..." if len(instruction) > 72 else instruction

        print(
            f"\n[MetaController] task={task_index} | difficulty={difficulty} | "
            f"strategy={strategy_name} | cost={cost}"
        )
        print(f"[MetaController] instruction: \"{preview}\"")
