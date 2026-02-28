# tau_bench/agents/adaptive_budget_agent.py
#
# Adaptive Budget Forcing (ABF) agent.
#
# Extends the S1-style ChatBudgetForcingAgent by making the two key
# budget parameters — num_ignore and max_tokens_thinking — dynamic
# rather than fixed. They are determined once per task based on difficulty,
# then applied uniformly across all steps of that task.
#
# DIFFICULTY → BUDGET MAPPING (defined in difficulty.py):
#   easy      → num_ignore=0, max_tokens=2000   (no forcing)
#   medium    → num_ignore=1, max_tokens=4000   (one reconsideration)
#   hard      → num_ignore=2, max_tokens=6000   (two reconsiderations)
#   very_hard → num_ignore=3, max_tokens=8000   (full forcing)
#
# DIFFICULTY SOURCE (in priority order):
#   1. difficulty_override passed at construction (set by MetaController)
#   2. Self-estimated from task instruction (for standalone --agent-strategy abf runs)
#
# WHAT IS UNCHANGED FROM S1 BASE:
#   - generate_next_step_with_budget_forcing() — core budget forcing logic
#   - vLLM / OpenAI-compatible API call structure
#   - solve() conversation loop
#   - REACT_INSTRUCTION / ACT_INSTRUCTION prompts

import json
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI

from tau_bench.agents.base import Agent
from tau_bench.agents.difficulty import DifficultyEstimator, ABF_BUDGET_TIERS, DifficultyTier
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


class AdaptiveBudgetForcingAgent(Agent):
    """
    ReAct-style agent with Adaptive Budget Forcing.

    Builds on S1-style budget forcing by scaling num_ignore and
    max_tokens_thinking to the difficulty of each task, rather than
    using fixed values for every task.

    When called by MetaControllerAgent:
        difficulty_override is set at construction → skips self-estimation.

    When run standalone (--agent-strategy abf):
        difficulty_override=None → estimates difficulty from instruction.
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,                                     # model name / path
        provider: str,                                  # kept for interface compat (unused — vLLM is direct)
        temperature: float = 0.0,
        vllm_base_url: str = "http://localhost:8005/v1",
        use_reasoning: bool = True,
        difficulty_override: Optional[DifficultyTier] = None,
    ) -> None:
        """
        Args:
            tools_info         : list of tool definitions from the environment
            wiki               : domain policy text from the environment
            model              : vLLM model name/path (e.g. "Qwen/Qwen3-4B-Instruct-2507")
            provider           : kept for MetaController interface compat; not used by vLLM path
            temperature        : sampling temperature
            vllm_base_url      : vLLM OpenAI-compatible endpoint
            use_reasoning      : True → ReAct format, False → Act format
            difficulty_override: if set by MetaController, skips self-estimation
        """
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )

        # vLLM via OpenAI-compatible API (same as original)
        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",
        )
        self.model_name  = model
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info  = tools_info

        # Shared difficulty estimator — used when no override is provided
        self.estimator = DifficultyEstimator()

        # If MetaController already decided difficulty, store it; else None = self-estimate
        self.difficulty_override = difficulty_override

        # num_ignore and max_tokens_thinking are NOT set here.
        # They are resolved per-task at the start of solve(), then read by
        # generate_next_step_with_budget_forcing() via self.num_ignore / self.max_tokens_thinking.
        self.num_ignore          = 0
        self.max_tokens_thinking = 2000

    # ── Core budget forcing logic — unchanged from S1 base ────────────────────

    def generate_next_step_with_budget_forcing(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """
        Generate next step with budget forcing.

        Reads self.num_ignore and self.max_tokens_thinking which are set
        once per task by solve() before this method is ever called.

        S1-style phases:
          Phase 1 — initial thinking (up to max_tokens_thinking tokens)
          Phase 2 — append "Wait,\\n" num_ignore times to force reconsideration
          Phase 3 — parse final action from accumulated content
        """
        # Calculate available tokens to prevent context overflow
        model_max_context = 32768
        input_text = "".join(msg.get("content", "") for msg in messages)
        estimated_input_tokens = len(input_text) // 4
        available_tokens = model_max_context - estimated_input_tokens - 500
        actual_max_tokens = min(self.max_tokens_thinking, max(100, available_tokens))

        # PHASE 1: Initial thinking
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=actual_max_tokens,
        )

        content      = response.choices[0].message.content
        tokens_used  = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason

        # Track first action for comparison logging
        try:
            first_action_str    = content.split("Action:")[-1].strip()
            first_action_parsed = json.loads(first_action_str)
            first_action_name   = first_action_parsed.get("name", "unknown")
        except Exception:
            first_action_name = "unknown"

        print(f"\n{'='*60}")
        print(f"[ABF] Phase 1 complete")
        print(f"  num_ignore={self.num_ignore} | max_tokens={actual_max_tokens}")
        print(f"  Tokens used: {tokens_used}/{actual_max_tokens}")
        print(f"  First action: {first_action_name}")
        print(f"{'='*60}")

        # PHASE 2: Adaptive budget forcing
        # num_ignore=0 (easy tasks) → loop body never runs → zero overhead
        # num_ignore>0 (harder tasks) → append "Wait,\n" and force reconsideration
        remaining_budget = actual_max_tokens - tokens_used

        if self.num_ignore > 0 and remaining_budget > 10:
            print(f"\n[ABF] Phase 2: forcing reconsideration x{self.num_ignore}")
            print(f"  Remaining budget: {remaining_budget} tokens")

            for i in range(self.num_ignore):
                if remaining_budget <= 10:
                    break

                print(f"\n  --- Forcing iteration {i+1}/{self.num_ignore} ---")

                messages_with_wait = messages + [
                    {"role": "assistant", "content": content + "Wait,\n"}
                ]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_with_wait,
                    temperature=self.temperature,
                    max_tokens=remaining_budget,
                    stop=["Wait,", "Wait"],
                )

                new_content = response.choices[0].message.content
                content     = content + "Wait,\n" + new_content

                tokens_used      += response.usage.completion_tokens
                remaining_budget  = actual_max_tokens - tokens_used
                finish_reason     = response.choices[0].finish_reason

                print(f"  New content: {len(new_content)} chars | "
                      f"total tokens: {tokens_used}/{actual_max_tokens} | "
                      f"remaining: {remaining_budget}")

        elif self.num_ignore == 0:
            print(f"\n[ABF] Phase 2: skipped (easy task, num_ignore=0)")
        else:
            print(f"\n[ABF] Phase 2: skipped (insufficient budget remaining)")

        # Track final action
        try:
            final_action_str    = content.split("Action:")[-1].strip()
            final_action_parsed = json.loads(final_action_str)
            final_action_name   = final_action_parsed.get("name", "unknown")
        except Exception:
            final_action_name = "unknown"

        # Log whether forcing changed the decision
        if self.num_ignore > 0:
            print(f"\n[ABF] Phase 2 complete")
            if first_action_name != final_action_name:
                print(f"  ACTION CHANGED: {first_action_name} → {final_action_name}")
            else:
                print(f"  Action unchanged: {first_action_name}")

        print(f"{'='*60}\n")

        # PHASE 3: Parse action (unchanged from S1 base)
        action_str = content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            print(f"[ABF WARNING] Failed to parse action JSON, using respond fallback")
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }

        if "name" not in action_parsed or "arguments" not in action_parsed:
            print(f"[ABF WARNING] Malformed action, using respond fallback")
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: str(action_parsed)},
            }

        action  = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        message = {"role": "assistant", "content": content}
        cost    = 0.0  # vLLM is local; keep for interface compat

        return message, action, cost

    # ── solve ──────────────────────────────────────────────────────────────────

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        """
        Solve a task with adaptive budget forcing.

        Determines difficulty once at the start, sets num_ignore and
        max_tokens_thinking accordingly, then runs the standard conversation
        loop — generate_next_step_with_budget_forcing reads those values
        at every step.
        """

        # ── Step 1: Resolve difficulty ─────────────────────────────────────────
        if self.difficulty_override is not None:
            # MetaController already classified this task — trust it
            difficulty = self.difficulty_override
        else:
            # Standalone run: self-estimate from the instruction
            if task_index is not None and task_index < len(env.tasks):
                instruction = env.tasks[task_index].instruction
            else:
                instruction = env.task.instruction
            difficulty = self.estimator.estimate(instruction)

        # ── Step 2: Look up budget params for this difficulty ──────────────────
        tier = ABF_BUDGET_TIERS[difficulty]
        self.num_ignore          = tier["num_ignore"]
        self.max_tokens_thinking = tier["max_tokens_thinking"]

        print(f"\n[ABF] task={task_index} | difficulty={difficulty} | "
              f"num_ignore={self.num_ignore} | max_tokens={self.max_tokens_thinking}")

        # ── Step 3: Standard conversation loop (unchanged from S1 base) ────────
        response = env.reset(task_index=task_index)
        reward   = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user",   "content": response.observation},
        ]
        total_cost = 0.0
        info       = {}

        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step_with_budget_forcing(messages)
            response = env.step(action)
            obs      = response.observation
            reward   = response.reward
            info     = {**info, **response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs

            messages.extend([
                message,
                {"role": "user", "content": obs},
            ])
            total_cost += cost

            if response.done:
                break

        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )


# ── Prompts (identical to chat_react_agent.py) ────────────────────────────────

REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""
