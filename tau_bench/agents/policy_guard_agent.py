# tau_bench/agents/policy_guard_agent.py
#
# PolicyGuard Agent — lightweight policy compliance checker.
#
# Runs a standard ToolCallingAgent-style conversation loop, but before
# executing any "respond" action, runs a critic LLM call that checks
# the proposed response against domain policy. If the critic flags a
# violation, the agent is asked to revise (up to max_retries times).
#
# Also detects premature surrender/escalation and pushes back.
#
# Overhead: ~1 extra LLM call per respond action (short, 100 max_tokens).
# Used by MetaControllerAgent for "easy" tier tasks.

import json
from typing import Optional, List, Dict, Any

from openai import OpenAI
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.agents.prompts import ENHANCED_GUIDELINES
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
)


# ── Prompts ──────────────────────────────────────────────────────────────────

CRITIC_PROMPT = """You are a strict policy compliance auditor for a customer service agent.

=== DOMAIN POLICY ===
{policy}

=== PROPOSED AGENT RESPONSE ===
{response}

=== CONVERSATION CONTEXT (last few turns) ===
{history_summary}

Your job: check the proposed response against the policy rules above.
Also check: is the agent giving up or escalating prematurely without trying available tools?

Respond with exactly one of:
  PASS
  FAIL: <one sentence explaining what rule is violated and what to do instead>

Do not add anything else."""

SURRENDER_PHRASES = [
    "transfer to", "human agent", "cannot help", "unable to assist",
    "please contact", "I apologize, I cannot", "beyond my capabilities",
    "I'm unable to", "escalate this",
]


# ── Agent ────────────────────────────────────────────────────────────────────

class PolicyGuardAgent(Agent):
    """
    ToolCallingAgent with a pre-response policy compliance critic.

    For each proposed "respond" action, a lightweight LLM call checks
    the response against domain policy. If it fails, the agent retries
    with the critic's feedback injected.
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        vllm_base_url: str = "http://localhost:8005/v1",
        max_retries: int = 2,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki + "\n" + ENHANCED_GUIDELINES
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.critic_client = OpenAI(base_url=vllm_base_url, api_key="EMPTY")

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        recent_actions = []  # Loop detection

        for _ in range(max_num_steps):
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = _message_to_action(next_message)

            # --- Loop detection (same as ToolCallingAgent) ---
            if action.name != RESPOND_ACTION_NAME:
                action_key = (action.name, json.dumps(action.kwargs, sort_keys=True))
                recent_actions.append(action_key)
                repeat_count = recent_actions.count(action_key)

                if repeat_count >= 3:
                    print(f"[LoopDetector] Forced break: {action.name} repeated 3x")
                    action = Action(name=RESPOND_ACTION_NAME, kwargs={
                        "content": "Let me try a different approach to assist you."
                    })
                    env_response = env.step(action)
                    reward = env_response.reward
                    info = {**info, **env_response.info.model_dump()}
                    messages.extend([
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ])
                    if env_response.done:
                        break
                    continue
                elif repeat_count >= 2:
                    print(f"[LoopDetector] Warning: {action.name} repeated 2x")
                    messages.append(next_message)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM] You have already called {action.name} with these exact "
                            f"arguments and received the same result. Do NOT repeat this call. "
                            f"Try different arguments, use a different tool, or respond to the user."
                        ),
                    })
                    continue

                if len(recent_actions) > 10:
                    recent_actions = recent_actions[-10:]
            # --- End loop detection ---

            # --- Policy Guard: critic check before respond actions ---
            if action.name == RESPOND_ACTION_NAME:
                candidate_response = action.kwargs.get("content", "")

                # Check for premature surrender
                if self._is_surrender(candidate_response):
                    messages.append(next_message)
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM] Do NOT give up or transfer to a human yet. "
                            "Review the available tools and try a different approach. "
                            "You must attempt to solve this using the tools provided."
                        ),
                    })
                    continue

                # Run critic check
                passed, reason = self._critic_check(candidate_response, messages)
                if not passed:
                    # Retry: inject critic feedback and re-generate
                    retries_done = 0
                    while not passed and retries_done < self.max_retries:
                        messages.append(next_message)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[POLICY VIOLATION] {reason} "
                                f"Please revise your response to comply with policy. "
                                f"Attempt {retries_done + 1}/{self.max_retries}."
                            ),
                        })
                        # Re-generate
                        res = completion(
                            messages=messages,
                            model=self.model,
                            custom_llm_provider=self.provider,
                            tools=self.tools_info,
                            temperature=self.temperature,
                        )
                        next_message = res.choices[0].message.model_dump()
                        total_cost += res._hidden_params["response_cost"] or 0
                        action = _message_to_action(next_message)

                        if action.name != RESPOND_ACTION_NAME:
                            break  # Agent decided to use a tool instead, let it
                        candidate_response = action.kwargs.get("content", "")
                        passed, reason = self._critic_check(candidate_response, messages)
                        retries_done += 1
            # --- End policy guard ---

            # Execute action
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    # ── Critic ───────────────────────────────────────────────────────────────

    def _critic_check(
        self, candidate_response: str, messages: List[Dict[str, Any]]
    ) -> tuple:
        """
        Run a lightweight LLM critic to check policy compliance.
        Returns (passed: bool, reason: str).
        """
        # Build a short history summary from recent messages
        history_lines = []
        for msg in messages[-6:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content[:150]}")
        history_summary = "\n".join(history_lines)

        prompt = CRITIC_PROMPT.format(
            policy=self.wiki[:2000],  # Truncate policy to avoid context overflow
            response=candidate_response,
            history_summary=history_summary,
        )

        try:
            response = self.critic_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            verdict = response.choices[0].message.content.strip()

            if verdict.upper().startswith("PASS"):
                return True, ""
            elif "FAIL" in verdict.upper():
                reason = verdict.replace("FAIL:", "").replace("FAIL", "").strip()
                return False, reason
            else:
                return True, ""  # Default to pass if unparseable
        except Exception as e:
            print(f"[PolicyGuard] Critic call failed ({e}), defaulting to PASS")
            return True, ""

    def _is_surrender(self, response: str) -> bool:
        lower = response.lower()
        return any(phrase in lower for phrase in SURRENDER_PHRASES)


# ── Helper ───────────────────────────────────────────────────────────────────

def _message_to_action(message: Dict[str, Any]) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
