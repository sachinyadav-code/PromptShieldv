"""Policy enforcement engine for PromptShield Phase 1.

Maps inspection findings into definitive ALLOW / BLOCK decisions and computes
explainable heuristic risk scores.
"""

from dataclasses import dataclass
from typing import List, Literal
from app.detection.rules import DetectionRule


@dataclass
class PolicyDecision:
    """The outcome of evaluating detection findings against gateway policies."""

    decision: Literal["ALLOW", "BLOCK"]
    risk_score: float
    matched_rule_ids: List[str]
    explanation: str


def evaluate_policy(matched_rules: List[DetectionRule]) -> PolicyDecision:
    """Evaluate matched rules and determine gateway action.

    Rules:
    - If ANY matched rule has action 'BLOCK', decision is strictly 'BLOCK'.
    - Blocking is enforced independently of the numerical risk score.
    - An 'ALLOW' decision does NOT mean the prompt is proven safe; it only indicates
      that none of the documented deterministic heuristic rules matched.
    - Risk score is a normalized heuristic sum in range [0.0, 1.0], not a statistical probability.
    """
    matched_ids = [rule.rule_id for rule in matched_rules]

    # Calculate explainable heuristic risk score
    raw_score = sum(rule.risk_weight for rule in matched_rules)
    heuristic_risk_score = round(min(1.0, raw_score), 2)

    # Check for any blocking rule
    blocking_rules = [r for r in matched_rules if r.action == "BLOCK"]

    if blocking_rules:
        reasons = [f"{r.rule_id} ({r.name})" for r in blocking_rules]
        explanation = (
            f"Request blocked by security policy: Matched blocking rule(s): {', '.join(reasons)}. "
            f"Heuristic risk score: {heuristic_risk_score}."
        )
        return PolicyDecision(
            decision="BLOCK",
            risk_score=heuristic_risk_score,
            matched_rule_ids=matched_ids,
            explanation=explanation,
        )

    # If no blocking rules matched
    explanation = (
        "Request allowed: No known heuristic rules matched. "
        "Notice: An ALLOW decision indicates absence of matched patterns, not proof of safety or absence of adversarial intent."
    )
    return PolicyDecision(
        decision="ALLOW",
        risk_score=heuristic_risk_score,
        matched_rule_ids=matched_ids,
        explanation=explanation,
    )
