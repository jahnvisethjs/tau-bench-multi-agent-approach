# tau_bench/agents/react_reflection_agent.py
#
# ReAct + Reflection Agent
#
# Standard ReAct execution with periodic reflection checkpoints.
# Every N tool calls, the agent is forced to review its progress,
# check remaining steps, and verify it is meeting user constraints.
#
# This targets multiple error categories simultaneously:
#   - Looping & Inefficient Reasoning (reflection asks "Are you stuck?")
#   - Constraint & Preference Misinterpretation (reflection asks "Meeting all constraints?")
#   - Incomplete Multi-Step Execution (reflection asks "What steps remain?")
#   - Policy & Confirmation Violations (reflection asks "Following all policies?")
#
# Used by MetaControllerAgent for "hard" tasks.
# Can also be run standalone via: --agent-strategy react-reflection

import json
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI

from tau_bench.agents.base import Agent
from tau_bench.agents.prompts import ENHANCED_GUIDELINES, REFLECTION_PROMPT
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


class ReactReflectionAgent(Agent):
    """
    ReAct agent with periodic reflection checkpoints.

    Every `reflection_interval` tool calls, injects a reflection prompt
    that forces the agent to review progress, check constraints, and
    plan remaining steps before continuing execution.

    Args:
        tools_info          : list of tool definitions from the environment
        wiki                : domain policy text from the environment
        model               : vLLM model name/path
        provider            : kept for interface compat
        temperature         : sampling temperature
        vllm_base_url       : vLLM OpenAI-compatible endpoint
        use_reasoning       : True -> ReAct format, False -> Act format
        reflection_interval : inject reflection every N tool calls (default: 4)
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        vllm_base_url: str = "http://localhost:8005/v1",
        use_reasoning: bool = True,
        reflection_interval: int = 4,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki
            + "\n#Available tools\n"
            + json.dumps(tools_info)
            + instruction
            + "\n"
            + ENHANCED_GUIDELINES
        )

        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",
        )
        self.model_name = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.reflection_interval = reflection_interval

    def _generate(self, messages: List[Dict[str, Any]], max_tokens: int = 2048) -> str:
        """Single LLM generation call."""
        # Calculate available context to prevent overflow
        model_max_context = 32768
        input_text = "".join(msg.get("content", "") or "" for msg in messages)
        estimated_input_tokens = len(input_text) // 4
        available_tokens = model_max_context - estimated_input_tokens - 500
        actual_max_tokens = min(max_tokens, max(100, available_tokens))

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=actual_max_tokens,
        )
        return response.choices[0].message.content

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """Generate next action using the ReAct format."""
        content = self._generate(messages)

        # Parse action from content
        action_str = content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }

        if "name" not in action_parsed or "arguments" not in action_parsed:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: str(action_parsed)},
            }

        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        message = {"role": "assistant", "content": content}
        cost = 0.0  # vLLM is local

        return message, action, cost

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        """
        Solve a task with periodic reflection checkpoints.

        Flow:
          1. Reset environment, get first user message
          2. Run ReAct loop with loop detection
          3. Every reflection_interval tool calls, inject a reflection checkpoint
          4. Agent reviews progress, then continues execution
        """
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}

        # Loop detection state
        recent_actions = []

        # Reflection state
        tool_call_count = 0

        for step in range(max_num_steps):

            # --- Reflection checkpoint ---
            if (
                tool_call_count > 0
                and tool_call_count % self.reflection_interval == 0
                and tool_call_count > 0
            ):
                print(
                    f"\n[ReactReflection] Reflection checkpoint at step {step} "
                    f"(after {tool_call_count} tool calls)"
                )
                messages.append({"role": "user", "content": REFLECTION_PROMPT})

                # Get reflection response (agent reviews its progress)
                reflection_content = self._generate(messages, max_tokens=512)
                messages.append({"role": "assistant", "content": reflection_content})

                print(f"[ReactReflection] Reflection: {reflection_content[:200]}...")

                # The reflection itself isn't an action step — we continue to
                # the next iteration where the agent will produce an actual action.
                # Add a prompt to continue execution after reflection.
                messages.append({
                    "role": "user",
                    "content": "Good. Now continue with your next action based on your reflection above."
                })

            # --- Normal ReAct step ---
            message, action, cost = self.generate_next_step(messages)

            # --- Loop detection ---
            if action.name != RESPOND_ACTION_NAME:
                action_key = (action.name, json.dumps(action.kwargs, sort_keys=True))
                recent_actions.append(action_key)

                repeat_count = recent_actions.count(action_key)

                if repeat_count >= 3:
                    # Force break the loop
                    print(f"[ReactReflection] Loop detected (3x): forcing respond")
                    action = Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={"content": "Let me try a different approach to assist you."},
                    )
                    message = {
                        "role": "assistant",
                        "content": f"Thought:\nI have been repeating the same action. Let me try a different approach.\nAction:\n"
                        + json.dumps({"name": action.name, "arguments": action.kwargs}),
                    }
                elif repeat_count >= 2:
                    # Inject warning, skip this duplicate, re-generate
                    print(f"[ReactReflection] Loop detected (2x): injecting warning")
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM] You have already called {action.name} with these exact "
                            f"arguments and received the same result. Do NOT repeat this call. "
                            f"Try different arguments, use a different tool, or respond to the user."
                        ),
                    })
                    continue  # Re-enter generation loop

                # Keep only last 10 actions
                if len(recent_actions) > 10:
                    recent_actions = recent_actions[-10:]

            # --- Execute action ---
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
                tool_call_count += 1

            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost

            if response.done:
                break

        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )


# ── Prompts (same format as chat_react_agent.py) ───────────────────────────────

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
