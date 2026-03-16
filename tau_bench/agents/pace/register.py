# tau_bench/agents/pace/register.py

import json
from dataclasses import dataclass, field
from typing import List, Literal, Tuple


@dataclass
class Intent:
    id: int
    description: str
    status: Literal["pending", "in_progress", "complete", "blocked"] = "pending"


@dataclass
class Constraint:
    field: str
    value: str
    source: Literal["user_stated", "assumed"]
    verified: bool = False


@dataclass
class PolicyCitation:
    action: str
    rule: str
    compliant: bool


class ConstraintRegister:
    def __init__(self):
        self.intents: List[Intent] = []
        self.constraints: List[Constraint] = []
        self.policy_citations: List[PolicyCitation] = []
        self.authenticated: bool = False
        self.completed_actions: List[str] = []
        self.initialized: bool = False

    def to_json(self) -> str:
        """Serialize current register state for injection into prompt."""
        return json.dumps(
            {
                "initialized": self.initialized,
                "authenticated": self.authenticated,
                "intents": [
                    {
                        "id": i.id,
                        "description": i.description,
                        "status": i.status,
                    }
                    for i in self.intents
                ],
                "constraints": [
                    {
                        "field": c.field,
                        "value": c.value,
                        "source": c.source,
                        "verified": c.verified,
                    }
                    for c in self.constraints
                ],
                "policy_citations": [
                    {
                        "action": p.action,
                        "rule": p.rule,
                        "compliant": p.compliant,
                    }
                    for p in self.policy_citations
                ],
                "completed_actions": self.completed_actions,
            },
            indent=2,
        )

    def all_intents_complete(self) -> bool:
        if not self.intents:
            return False
        return all(i.status == "complete" for i in self.intents)

    def incomplete_intents(self) -> List[str]:
        return [
            i.description
            for i in self.intents
            if i.status not in ("complete", "blocked")
        ]

    def pre_action_check(self, action_name: str) -> Tuple[bool, str]:
        """
        Hard gate called before any write operation.
        Returns (can_proceed, reason).
        """
        if not self.authenticated:
            return False, "User is not authenticated yet. Call an authentication tool first."

        unverified = [c for c in self.constraints if not c.verified]
        if unverified:
            fields = [c.field for c in unverified]
            return (
                False,
                f"These constraints are unverified: {fields}. "
                f"Use a lookup tool to confirm them before proceeding.",
            )

        cited = [p for p in self.policy_citations if p.action == action_name]
        if not cited:
            return (
                False,
                f"No policy citation found for '{action_name}'. "
                f"Call cite_policy with the exact rule that permits this action.",
            )

        if not cited[-1].compliant:
            return (
                False,
                f"Your policy citation for '{action_name}' is marked non-compliant. "
                f"Do not proceed with this action.",
            )

        return True, "OK"
