REGISTER_TOOLS = [
    {
        "name": "initialize_register",
        "description": "Call this FIRST before any other action. Decompose the user task into intents and extract all constraints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "description": {"type": "string"}
                        },
                        "required": ["id", "description"]
                    }
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "value": {"type": "string"},
                            "source": {"type": "string", "enum": ["user_stated", "assumed"]}
                        },
                        "required": ["field", "value", "source"]
                    }
                }
            },
            "required": ["intents", "constraints"]
        }
    },
    {
        "name": "update_intent_status",
        "description": "Update the status of an intent after completing or blocking it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "complete", "blocked"]
                }
            },
            "required": ["intent_id", "status"]
        }
    },
    {
        "name": "verify_constraint",
        "description": "Mark a constraint as verified after confirming its value via a tool call or user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "confirmed_value": {"type": "string"}
            },
            "required": ["field", "confirmed_value"]
        }
    },
    {
        "name": "cite_policy",
        "description": "REQUIRED before any write operation. Quote the exact policy rule that permits the action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "rule": {"type": "string"},
                "compliant": {"type": "boolean"}
            },
            "required": ["action", "rule", "compliant"]
        }
    }
]
