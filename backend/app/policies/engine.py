"""Policy enforcement engine for PromptShield (Phase 1 & Phase 2 Hybrid).

Maps deterministic inspection findings and ML detection assessments into authoritative
ALLOW / BLOCK decisions, compute explainable risk scores, and track decision origin.
"""

from dataclasses import dataclass
from enum import Enum
import os
import threading
from typing import Any, List, Literal, Optional, Union

from app.detection.rules import DetectionRule
from app.schemas import MLAssessment

PolicyDecisionSource = Literal["deterministic_rule", "ml_enforcement", "allow_default"]


@dataclass
class PolicyDecision:
    """The outcome of evaluating detection findings against gateway policies."""

    decision: Literal["ALLOW", "BLOCK"]
    risk_score: float
    matched_rule_ids: List[str]
    explanation: str
    decision_source: PolicyDecisionSource = "allow_default"


class MLEnforcementMode(str, Enum):
    """Supported ML enforcement policy modes for PromptShield."""

    OBSERVE = "observe"
    BLOCK_ON_THRESHOLD = "block_on_threshold"


DEFAULT_ML_ENFORCEMENT_MODE = MLEnforcementMode.OBSERVE.value
SUPPORTED_ML_ENFORCEMENT_MODES = {
    MLEnforcementMode.OBSERVE.value,
    MLEnforcementMode.BLOCK_ON_THRESHOLD.value,
}

_current_mode: Optional[str] = None
_mode_lock = threading.Lock()


def validate_ml_enforcement_mode(mode: Any) -> str:
    """Validate and normalize an ML enforcement mode string or enum.

    Raises:
        ValueError: If the mode is not supported or not a valid string.
    """
    if isinstance(mode, MLEnforcementMode):
        return mode.value
    if isinstance(mode, bool) or not isinstance(mode, str):
        raise ValueError(
            f"Invalid ML enforcement mode type: {type(mode).__name__}. "
            f"Expected string or MLEnforcementMode."
        )
    normalized = mode.strip().lower()
    if normalized not in SUPPORTED_ML_ENFORCEMENT_MODES:
        raise ValueError(
            f"Unsupported ML enforcement mode: '{mode}'. "
            f"Supported modes are: {sorted(SUPPORTED_ML_ENFORCEMENT_MODES)}."
        )
    return normalized


def get_ml_enforcement_mode() -> str:
    """Get the active ML enforcement mode.

    Precedence:
    1. Runtime setting via set_ml_enforcement_mode().
    2. Primary environment variable: PROMPTSHIELD_ML_ENFORCEMENT_MODE.
    3. Fallback environment variable: ML_ENFORCEMENT_MODE.
    4. Default: 'observe'.

    Raises:
        ValueError: If the resolved mode or environment variable is invalid.
    """
    with _mode_lock:
        if _current_mode is not None:
            return _current_mode
        primary_env = os.getenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE")
        if primary_env is not None:
            return validate_ml_enforcement_mode(primary_env)
        fallback_env = os.getenv("ML_ENFORCEMENT_MODE")
        if fallback_env is not None:
            return validate_ml_enforcement_mode(fallback_env)
        return DEFAULT_ML_ENFORCEMENT_MODE


def validate_startup_configuration() -> str:
    """Validate effective policy configuration at startup.

    Resolves and strictly validates the effective policy mode according to documented precedence:
    1. Programmatic runtime override
    2. Primary environment variable (PROMPTSHIELD_ML_ENFORCEMENT_MODE)
    3. Fallback environment variable (ML_ENFORCEMENT_MODE)
    4. Default ('observe')

    Returns:
        The validated effective ML enforcement mode string ('observe' or 'block_on_threshold').

    Raises:
        ValueError: If the effective configuration is invalid, empty, or unsupported.
    """
    return get_ml_enforcement_mode()


def set_ml_enforcement_mode(mode: Union[str, MLEnforcementMode]) -> None:
    """Set the runtime ML enforcement mode with strict validation.

    Raises:
        ValueError: If the mode is not supported.
    """
    validated = validate_ml_enforcement_mode(mode)
    with _mode_lock:
        global _current_mode
        _current_mode = validated


def reset_ml_enforcement_mode() -> None:
    """Reset the runtime ML enforcement mode back to default (or environment variable)."""
    with _mode_lock:
        global _current_mode
        _current_mode = None


def evaluate_hybrid_policy(
    matched_rules: List[DetectionRule],
    ml_assessment: Optional[Union[MLAssessment, Any]] = None,
    mode: Optional[Union[str, MLEnforcementMode]] = None,
) -> PolicyDecision:
    """Evaluate matched deterministic rules and optional ML assessment under the configured policy mode.

    Precedence and Security Rules:
    1. Deterministic Sovereignty: If ANY matched rule has action 'BLOCK', decision is strictly 'BLOCK',
       regardless of the ML assessment or policy mode. An ML benign prediction can NEVER override a
       deterministic rule block.
    2. Explicit Policy Mode: If deterministic rules do not block:
       - 'observe' (default): ML assessment is purely observational. An ML injection prediction does
         not alter the decision.
       - 'block_on_threshold': If ML assessment is available and its validated would_block value is True,
         the request is blocked.
    3. Unavailable ML Resiliency: If ML assessment is unavailable (available=False or None), it is never
       interpreted as benign. In 'block_on_threshold' mode, an unavailable ML assessment does NOT block
       the request on its own; the existing deterministic decision ('ALLOW') is preserved.
    4. Heuristic Scores: All risk scores are normalized heuristic contributions [0.0, 1.0], never
       calibrated Bayesian probabilities.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    matched_ids = [rule.rule_id for rule in matched_rules]
    raw_rule_score = sum(rule.risk_weight for rule in matched_rules)
    heuristic_risk_score = round(min(1.0, raw_rule_score), 2)

    # 1. Deterministic rule check (Sovereign Precedence)
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
            decision_source="deterministic_rule",
        )

    # Resolve and validate policy mode
    effective_mode = validate_ml_enforcement_mode(mode) if mode is not None else get_ml_enforcement_mode()

    # Adapt input to MLAssessment if necessary
    normalized_ml: Optional[MLAssessment] = None
    if ml_assessment is not None:
        if isinstance(ml_assessment, MLAssessment):
            normalized_ml = ml_assessment
        else:
            normalized_ml = MLAssessment.from_inference_result(ml_assessment)

    # 2. Hybrid Evaluation: Deterministic rules did not produce a block
    if effective_mode == MLEnforcementMode.BLOCK_ON_THRESHOLD.value:
        if normalized_ml is not None and normalized_ml.available and normalized_ml.would_block:
            ml_score = normalized_ml.score if normalized_ml.score is not None else 1.0
            combined_risk_score = round(max(heuristic_risk_score, ml_score), 2)
            explanation = (
                f"Request blocked by security policy: ML prompt injection detection threshold met "
                f"(score: {normalized_ml.score:.4f}, threshold: {normalized_ml.threshold:.2f}, mode: {effective_mode}). "
                f"Notice: Model score is an uncalibrated heuristic estimate, not a probability guarantee."
            )
            return PolicyDecision(
                decision="BLOCK",
                risk_score=combined_risk_score,
                matched_rule_ids=matched_ids,
                explanation=explanation,
                decision_source="ml_enforcement",
            )

    # 3. Default Allow Path (no deterministic block and ML did not block or was observational/unavailable)
    explanation = (
        "Request allowed: No known heuristic rules matched. "
        "Notice: An ALLOW decision indicates absence of matched patterns, not proof of safety or absence of adversarial intent."
    )
    return PolicyDecision(
        decision="ALLOW",
        risk_score=heuristic_risk_score,
        matched_rule_ids=matched_ids,
        explanation=explanation,
        decision_source="allow_default",
    )


def evaluate_policy(
    matched_rules: List[DetectionRule],
    ml_assessment: Optional[Union[MLAssessment, Any]] = None,
    mode: Optional[Union[str, MLEnforcementMode]] = None,
) -> PolicyDecision:
    """Authoritative gateway policy evaluator.

    Preserves 100% backward compatibility with evaluate_policy(matched_rules) while
    supporting hybrid evaluation when ml_assessment is provided.
    """
    return evaluate_hybrid_policy(
        matched_rules=matched_rules,
        ml_assessment=ml_assessment,
        mode=mode,
    )
