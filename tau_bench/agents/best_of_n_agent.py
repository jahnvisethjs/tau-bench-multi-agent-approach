# tau_bench/agents/best_of_n_agent.py
#
# Best-of-N (BoN) agent.
#
# Runs up to N independent trajectories using ToolCallingAgent and
# returns the best one (highest reward), with early stopping on
# reward == 1.0. N is determined dynamically based on task difficulty
# via BON_TIERS, or can be capped by max_n.
#
# DIFFICULTY → N MAPPING (defined in difficulty.py):
#   easy      → N=2
#   medium    → N=3
#   hard      → N=4
#   very_hard → N=5
#
# DIFFICULTY SOURCE (in priority order):
#   1. difficulty_override passed at construction (set by MetaController)
#   2. Self-estimated from task instruction (for standalone --agent-strategy bon runs)

from typing import Optional, List, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.agents.difficulty import DifficultyEstimator, BON_TIERS, DifficultyTier
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult


class BestOfNAgent(Agent):
    """
    Best-of-N sampling agent.

    Runs up to N ToolCallingAgent trajectories with temperature=0.8 for
    diversity, returning the first perfect trajectory (reward == 1.0) or
    the best one if none succeed.

    When called by MetaControllerAgent:
        difficulty_override is set at construction → skips self-estimation.

    When run standalone (--agent-strategy bon):
        difficulty_override=None → estimates difficulty from instruction.
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        max_n: int = 5,
        difficulty_override: Optional[DifficultyTier] = None,
    ) -> None:
        """
        Args:
            tools_info          : list of tool definitions from the environment
            wiki                : domain policy text from the environment
            model               : LLM model name (passed through to ToolCallingAgent)
            provider            : LLM provider name (passed through to ToolCallingAgent)
            temperature         : base temperature (overridden to 0.8 for inner runs)
            max_n               : hard cap on number of trajectories regardless of difficulty
            difficulty_override : if set by MetaController, skips self-estimation
        """
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_n = max_n
        self.difficulty_override = difficulty_override

        self.estimator = DifficultyEstimator()

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        """
        Solve a task with Best-of-N sampling.

        Determines difficulty once, looks up N from BON_TIERS, then runs
        up to N sequential trajectories. Returns immediately on a perfect
        score, otherwise returns the trajectory with the highest reward.
        """

        # ── Step 1: Resolve difficulty ───────────────────────────────────────
        if self.difficulty_override is not None:
            difficulty = self.difficulty_override
        else:
            if task_index is not None and task_index < len(env.tasks):
                instruction = env.tasks[task_index].instruction
            else:
                instruction = env.task.instruction
            difficulty = self.estimator.estimate(instruction)

        # ── Step 2: Determine N from tier, capped by max_n ──────────────────
        tier_n = BON_TIERS[difficulty]["n"]
        n = min(tier_n, self.max_n)

        print(f"\n[BoN] task={task_index} | difficulty={difficulty} | n={n}")

        # ── Step 3: Run up to N trajectories sequentially ───────────────────
        best_result: Optional[SolveResult] = None
        accumulated_cost = 0.0

        for i in range(n):
            inner_agent = ToolCallingAgent(
                tools_info=self.tools_info,
                wiki=self.wiki,
                model=self.model,
                provider=self.provider,
                temperature=0.8,
            )

            result = inner_agent.solve(
                env=env,
                task_index=task_index,
                max_num_steps=max_num_steps,
            )

            accumulated_cost += result.total_cost
            print(f"[BoN] trial {i+1}/{n} | reward={result.reward}")

            if best_result is None or result.reward > best_result.reward:
                best_result = result

            if result.reward == 1.0:
                print(f"[BoN] early stop at trial {i+1}")
                break

        # ── Step 4: Return best trajectory with accumulated cost ────────────
        return SolveResult(
            reward=best_result.reward,
            info=best_result.info,
            messages=best_result.messages,
            total_cost=accumulated_cost,
        )
