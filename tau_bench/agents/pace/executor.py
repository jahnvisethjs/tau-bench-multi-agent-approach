# tau_bench/agents/pace/executor.py

from tau_bench.agents.pace.register import (
    ConstraintRegister,
    Intent,
    Constraint,
    PolicyCitation,
)


REGISTER_TOOL_NAMES = {
    "initialize_register",
    "update_intent_status",
    "verify_constraint",
    "cite_policy",
    "mark_authenticated",
}


def execute_register_tool(
    name: str, arguments: dict, register: ConstraintRegister
) -> str:
    """
    Handle all register tool calls internally.
    Returns a result string that goes back into the message history.
    Never touches env.step().
    """

    if name == "initialize_register":
        intents = arguments.get("intents", [])
        constraints = arguments.get("constraints", [])

        register.intents = [
            Intent(id=i["id"], description=i["description"])
            for i in intents
        ]
        register.constraints = [
            Constraint(
                field=c["field"],
                value=c["value"],
                source=c["source"],
            )
            for c in constraints
        ]
        register.initialized = True

        return (
            f"Register initialized. "
            f"{len(register.intents)} intents, "
            f"{len(register.constraints)} constraints tracked."
        )

    elif name == "mark_authenticated":
        register.authenticated = True
        return "User authenticated. Write operations are now unblocked on this condition."

    elif name == "update_intent_status":
        intent_id = arguments.get("intent_id")
        status = arguments.get("status")
        for intent in register.intents:
            if intent.id == intent_id:
                intent.status = status
                return f"Intent {intent_id} ('{intent.description}') updated to '{status}'."
        return f"Error: Intent with id {intent_id} not found."

    elif name == "verify_constraint":
        field = arguments.get("field")
        confirmed_value = arguments.get("confirmed_value")
        for constraint in register.constraints:
            if constraint.field == field:
                constraint.verified = True
                constraint.value = confirmed_value
                return (
                    f"Constraint '{field}' verified. "
                    f"Confirmed value: '{confirmed_value}'."
                )
        # Field not in register yet — add it as verified
        register.constraints.append(
            Constraint(
                field=field,
                value=confirmed_value,
                source="user_stated",
                verified=True,
            )
        )
        return (
            f"Constraint '{field}' was not in register. "
            f"Added and verified as '{confirmed_value}'."
        )

    elif name == "cite_policy":
        action = arguments.get("action")
        rule = arguments.get("rule")
        compliant = arguments.get("compliant", False)
        register.policy_citations.append(
            PolicyCitation(action=action, rule=rule, compliant=compliant)
        )
        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        return (
            f"Policy cited for '{action}' [{status}]: \"{rule}\""
        )

    else:
        return f"Error: Unknown register tool '{name}'."
