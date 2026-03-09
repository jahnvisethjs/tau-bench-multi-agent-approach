# tau_bench/agents/prompts.py
#
# Shared prompt constants used by all agents.
# Import and append to system prompts to improve agent behavior.

ENHANCED_GUIDELINES = """
CRITICAL BEHAVIORAL RULES YOU MUST FOLLOW:
1. NEVER transfer to a human agent or end the conversation prematurely. You MUST attempt to solve the task using ALL available tools before considering escalation. Only transfer if the request is genuinely outside your capabilities after multiple attempts.
2. ALWAYS authenticate the user FIRST before any other action. For retail: use find_user_id_by_email or find_user_id_by_name_zip. For airline: use get_user_details with the provided user_id.
3. ALWAYS get explicit confirmation from the user (wait for them to say "yes" or confirm) before executing any action that modifies data (booking, cancellation, modification, return, exchange).
4. READ the user's requirements carefully. Track ALL constraints (price, timing, class, preferences). Before making a modifying tool call, mentally verify your arguments match EVERY stated constraint.
5. Complete ALL steps of the task. Do not stop after partial progress. If the task requires search, select, book, and confirm — do all four steps.
6. If a tool call returns an error, READ the error message carefully and FIX your arguments. Never retry with identical arguments.
7. NEVER fabricate information. If you need data, use a tool to look it up.
8. When the user provides multiple requirements, address ALL of them, not just the first one.
"""

REFLECTION_PROMPT = """[REFLECTION CHECKPOINT] Before your next action, review your progress:
1. What steps have you completed so far?
2. What steps remain to fully resolve the user's request?
3. Are you meeting ALL of the user's stated constraints and preferences?
4. Have you followed all required policies (confirmation, authentication, etc.)?
5. Are you stuck in a loop or making progress?
After reflecting, continue with your next Thought and Action."""
