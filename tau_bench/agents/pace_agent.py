# tau_bench/agents/pace_agent.py

import json
from litellm import completion
from typing import Optional, List, Dict, Any, Tuple

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from tau_bench.agents.pace.register import ConstraintRegister
from tau_bench.agents.pace.executor import execute_register_tool, REGISTER_TOOL_NAMES
from tau_bench.agents.pace.prompts import build_system_prompt


# Write operations that require pre_action_check before env.step()
# Add any environment-specific write tools here
WRITE_TOOLS = {
    # retail
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    # airline
    "book_reservation",
    "cancel_reservation",
    "update_reservation",
    "update_flight",
    "update_baggage",
}


class PaceAgent(Agent):
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

    def _call_llm(
        self, messages: List[Dict[str, Any]], system_prompt: str
    ) -> Tuple[Dict[str, Any], str, float]:
        """
        Call the LLM. Returns (message_dict, raw_content, cost).
        Follows the same pattern as ChatReActAgent.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=full_messages,
            api_base="http://localhost:8000/v1",   # 32B port
            api_key="EMPTY",
            temperature=self.temperature,
        )
        message = res.choices[0].message
        content = message.content or ""
        cost = res._hidden_params.get("response_cost") or 0.0
        return message.model_dump(), content, cost

    def _parse_action(self, content: str) -> Action:
        """
        Parse Action: JSON block from LLM output.
        Same logic as ChatReActAgent.generate_next_step.
        """
        action_str = content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # Fallback: treat raw text as a respond action
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }

        assert "name" in action_parsed, f"No 'name' in parsed action: {action_parsed}"
        assert "arguments" in action_parsed, f"No 'arguments' in parsed action: {action_parsed}"

        return Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:

        register = ConstraintRegister()

        # Reset env and get first observation
        response = env.reset(task_index=task_index)
        reward = 0.0
        total_cost = 0.0
        info = {}

        # Message history (no system prompt here — injected fresh each turn)
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": response.observation}
        ]

        for step in range(max_num_steps):

            # Rebuild system prompt with current register state each turn
            system_prompt = build_system_prompt(
                wiki=self.wiki,
                tools_info=self.tools_info,
                register=register,
            )

            # LLM call
            message_dict, content, cost = self._call_llm(messages, system_prompt)
            total_cost += cost
            messages.append(message_dict)

            # Parse action from LLM output
            action = self._parse_action(content)

            # ── REGISTER TOOLS ──────────────────────────────────────────────
            # These never go to env.step(). Handle internally and loop.
            if action.name in REGISTER_TOOL_NAMES:
                result = execute_register_tool(action.name, action.kwargs, register)
                messages.append({"role": "user", "content": f"Register: {result}"})
                continue

            # ── RESPOND ACTION ───────────────────────────────────────────────
            if action.name == RESPOND_ACTION_NAME:
                response = env.step(action)
                obs = response.observation
                reward = response.reward
                info = {**info, **response.info.model_dump()}
                total_cost += cost

                if response.done:
                    # Final check: are all intents complete?
                    incomplete = register.incomplete_intents()
                    if incomplete and reward == 0.0:
                        # Log for debugging but don't force continuation —
                        # env has already ended the episode
                        info["pace_incomplete_intents"] = incomplete
                    break

                messages.append({"role": "user", "content": obs})
                continue

            # ── WRITE TOOLS — hard gate ──────────────────────────────────────
            if action.name in WRITE_TOOLS:
                can_proceed, reason = register.pre_action_check(action.name)
                if not can_proceed:
                    # Block the action and tell the agent why
                    block_msg = (
                        f"BLOCKED by PACE register: {reason}\n"
                        f"Resolve this condition before retrying '{action.name}'."
                    )
                    messages.append({"role": "user", "content": block_msg})
                    continue

                # Gate passed — execute in env
                response = env.step(action)
                obs = "API output: " + response.observation
                reward = response.reward
                info = {**info, **response.info.model_dump()}
                register.completed_actions.append(action.name)

                messages.append({"role": "user", "content": obs})

                if response.done:
                    break
                continue

            # ── READ / LOOKUP TOOLS — pass straight through ──────────────────
            response = env.step(action)
            obs = "API output: " + response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}

            messages.append({"role": "user", "content": obs})

            if response.done:
                break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
        )