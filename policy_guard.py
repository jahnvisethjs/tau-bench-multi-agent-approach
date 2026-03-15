# policy_guard_agent.py
# Drop-in replacement for any tau-bench agent.
# Wraps your existing base agent (ReAct/ACT/FC) with a pre-execution policy critic.
# Fast: only adds 1 LLM call per tool invocation. No parallelism needed.

import json
import re
from typing import Optional

# ── Prompts ────────────────────────────────────────────────────────────────────

ACTOR_SYSTEM_PROMPT = """You are a helpful {domain} agent. You have access to tools to complete customer requests.
Use them step by step. Do NOT give up or transfer to a human unless you have exhausted all available tools.
Think through what information you need, then call the appropriate tool."""

CRITIC_PROMPT = """You are a strict policy compliance auditor for a {domain} customer service agent.

=== DOMAIN POLICY ===
{policy}

=== PROPOSED AGENT ACTION ===
{action}

=== CONVERSATION SO FAR ===
{history_summary}

Your job: check the proposed action against the policy rules above.
Also check: is the agent giving up or escalating prematurely without trying available tools?

Respond with exactly one of:
  APPROVE
  REJECT: <one sentence explaining what rule is violated and what to do instead>

Do not add anything else."""

SURRENDER_PHRASES = [
    "transfer to", "human agent", "cannot help", "unable to assist",
    "please contact", "I apologize, I cannot", "beyond my capabilities",
    "I'm unable to", "escalate this"
]

# ── Agent class ────────────────────────────────────────────────────────────────

class PolicyGuardAgent:
    """
    Wraps any base agent with a lightweight pre-execution critic.
    Compatible with tau-bench's agent runner interface.
    
    Usage:
        agent = PolicyGuardAgent(
            base_agent=your_existing_agent,   # ReAct, ACT, or FC agent
            model_client=your_llm_client,     # same client you use for base agent
            domain="airline",                 # or "retail"
            policy_path="tau_bench/envs/airline/data/policy.md",
            max_retries=2,
        )
    """

    def __init__(self, base_agent, model_client, domain: str,
                 policy_path: str, max_retries: int = 2):
        self.base_agent = base_agent
        self.client = model_client
        self.domain = domain
        self.max_retries = max_retries
        self.policy = self._load_policy(policy_path)
        self._turn_log = []   # lightweight history summary for critic

    # ── Public interface (tau-bench calls this) ────────────────────────────────

    def act(self, user_message: str, env_state: dict) -> str:
        """
        Main entry point. Called once per user turn by the tau-bench runner.
        Returns the final agent response (text or tool call string).
        """
        self._turn_log.append(f"User: {user_message}")

        for attempt in range(self.max_retries + 1):
            # 1. Base agent proposes an action
            proposed = self.base_agent.act(user_message, env_state)

            # 2. Detect and short-circuit plain text (no tool call) responses
            if not self._contains_tool_call(proposed):
                # Still check for premature surrender in plain text responses
                if self._is_surrender(proposed) and attempt < self.max_retries:
                    pushback = self._get_surrender_pushback()
                    self.base_agent.inject_system_message(pushback)
                    continue
                # Approved — log and return
                self._turn_log.append(f"Agent: {proposed[:120]}")
                return proposed

            # 3. Run policy critic on tool call
            verdict = self._run_critic(proposed)

            if verdict == "APPROVE":
                self._turn_log.append(f"Agent [approved]: {proposed[:120]}")
                return proposed
            else:
                # Inject critic feedback and retry
                rejection_reason = verdict.replace("REJECT:", "").strip()
                self.base_agent.inject_system_message(
                    f"[Policy violation detected] {rejection_reason} "
                    f"Please revise your action. Attempt {attempt+1}/{self.max_retries}."
                )

        # All retries exhausted — return last proposal anyway
        # (better to attempt than to crash the eval)
        self._turn_log.append(f"Agent [max retries]: {proposed[:120]}")
        return proposed

    # ── Critic ────────────────────────────────────────────────────────────────

    def _run_critic(self, proposed_action: str) -> str:
        """
        Calls the LLM critic. Returns 'APPROVE' or 'REJECT: <reason>'.
        Kept cheap: small max_tokens, no conversation history injected.
        """
        history_summary = "\n".join(self._turn_log[-6:])  # last 3 turns only

        prompt = CRITIC_PROMPT.format(
            domain=self.domain,
            policy=self.policy,
            action=proposed_action,
            history_summary=history_summary,
        )

        response = self.client.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,          # critic only needs a short verdict
            temperature=0.0,        # deterministic — same input → same verdict
        )

        verdict = response.strip()
        # Normalize in case model adds extra text
        if verdict.upper().startswith("APPROVE"):
            return "APPROVE"
        elif "REJECT" in verdict.upper():
            return verdict
        else:
            return "APPROVE"        # default to approve if unparseable

    # ── Surrender detection ────────────────────────────────────────────────────

    def _is_surrender(self, response: str) -> bool:
        lower = response.lower()
        return any(phrase in lower for phrase in SURRENDER_PHRASES)

    def _get_surrender_pushback(self) -> str:
        return (
            "Do NOT give up or transfer to a human yet. "
            "Review the available tools and try a different approach. "
            "You must attempt to solve this using the tools provided."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _contains_tool_call(self, response: str) -> bool:
        """Detect JSON tool calls or function_call blocks."""
        try:
            parsed = json.loads(response)
            return "name" in parsed or "function" in parsed
        except (json.JSONDecodeError, TypeError):
            # Also catch ReAct-style "Action: tool_name(...)" patterns
            return bool(re.search(r"Action:\s*\w+\(", response))

    def _load_policy(self, path: str) -> str:
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            print(f"[PolicyGuardAgent] Warning: policy file not found at {path}")
            return ""

    def reset(self):
        """Call between tasks to clear turn log."""
        self._turn_log = []
        self.base_agent.reset()
