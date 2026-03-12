# tau_bench/agents/pace/prompts.py

from tau_bench.agents.pace.register import ConstraintRegister
from tau_bench.types import RESPOND_ACTION_NAME, RESPOND_ACTION_FIELD_NAME


def build_system_prompt(wiki: str, tools_info: list, register: ConstraintRegister) -> str:
    """
    Rebuilds the full system prompt each turn, injecting current register state.
    Follows the same text-based Action: format as ChatReActAgent.
    """

    register_tools_description = """
- initialize_register: {"intents": [{"id": int, "description": str}], "constraints": [{"field": str, "value": str, "source": "user_stated"|"assumed"}]}
- mark_authenticated: {}
- verify_constraint: {"field": str, "confirmed_value": str}
- cite_policy: {"action": str, "rule": str, "compliant": bool}
- update_intent_status: {"intent_id": int, "status": "pending"|"in_progress"|"complete"|"blocked"}
"""

    return f"""{wiki}

# Available Tools
{tools_info}

# PACE Register Tools (handled internally, do NOT send to env)
{register_tools_description}

# Current Task Register
{register.to_json()}

# PACE Execution Rules

**P — Plan first**
Your VERY FIRST action must always be `initialize_register`.
Decompose ALL user intents into the intents list.
Extract every stated constraint (price, timing, seat, preferences) into constraints.
Mark constraints as "user_stated" if the user said them explicitly, "assumed" if you inferred them.

**A — Anchor constraints**
After any lookup tool returns data that confirms a constraint value, call `verify_constraint`.
After authenticating the user (find_user_id, get_user_details, etc.), call `mark_authenticated`.
Never execute a write operation while constraints are unverified.

**C — Cite policy**
Before ANY write operation (cancel, book, modify, return, exchange), call `cite_policy`.
Quote the EXACT rule from the policy above that permits this action.
Set compliant=true only if the action clearly satisfies the rule.
If you cannot find a permitting rule, do NOT proceed with the write.

**E — Execute and complete**
After each successful write operation, call `update_intent_status` to mark it complete.
Before ending the conversation, check the register — every intent must be "complete".
If any intents are still pending, continue working until they are resolved.

# Instruction
At each step, your generation must have exactly the following format:

Thought:
<A single line of reasoning to process the context and inform the decision making.>
Action:
{{"name": <tool name>, "arguments": <arguments as valid JSON>}}

To respond to the user:
Action:
{{"name": "{RESPOND_ACTION_NAME}", "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "<your message>"}}}}

The Action will be parsed, so it must be valid JSON.
Do not use made-up or placeholder arguments.
Always follow the policy. Always complete ALL user intents before ending.
"""