"""Unit tests for PromptShield Policy Engine and Hybrid Evaluation (Phase 2 Stage 3 Step 4)."""

import pytest

from app.detection.rules import DetectionRule
from app.policies.engine import (
    DEFAULT_ML_ENFORCEMENT_MODE,
    MLEnforcementMode,
    PolicyDecision,
    evaluate_hybrid_policy,
    evaluate_policy,
    get_ml_enforcement_mode,
    reset_ml_enforcement_mode,
    set_ml_enforcement_mode,
    validate_ml_enforcement_mode,
    validate_startup_configuration,
)
from app.schemas import MLAssessment


# Helper fixture / rules for testing
@pytest.fixture
def blocking_rule():
    return DetectionRule(
        rule_id="RULE_TEST_BLOCK",
        name="Test Blocking Rule",
        description="Rule used for testing deterministic blocking.",
        patterns=[],
        severity="HIGH",
        action="BLOCK",
        risk_weight=0.85,
    )


@pytest.fixture
def flag_rule():
    return DetectionRule(
        rule_id="RULE_TEST_FLAG",
        name="Test Flagging Rule",
        description="Rule used for testing non-blocking flag.",
        patterns=[],
        severity="LOW",
        action="FLAG",
        risk_weight=0.20,
    )


@pytest.fixture
def available_injection_ml():
    return MLAssessment(
        available=True,
        score=0.92,
        threshold=0.50,
        predicted_label="injection",
        would_block=True,
        model_version="v1",
    )


@pytest.fixture
def available_benign_ml():
    return MLAssessment(
        available=True,
        score=0.15,
        threshold=0.50,
        predicted_label="benign",
        would_block=False,
        model_version="v1",
    )


@pytest.fixture
def unavailable_ml():
    return MLAssessment.unavailable(
        threshold=0.50,
        error_code="MODEL_UNAVAILABLE",
    )


# ---------------------------------------------------------------------------
# 1. Configuration tests
# ---------------------------------------------------------------------------

def test_default_mode_is_observe():
    """Verify default ML enforcement mode is 'observe'."""
    assert DEFAULT_ML_ENFORCEMENT_MODE == "observe"
    assert get_ml_enforcement_mode() == "observe"


def test_validate_ml_enforcement_mode_accepts_valid_modes():
    """Verify validate_ml_enforcement_mode accepts valid strings and enums."""
    assert validate_ml_enforcement_mode("observe") == "observe"
    assert validate_ml_enforcement_mode("block_on_threshold") == "block_on_threshold"
    assert validate_ml_enforcement_mode("OBSERVE") == "observe"
    assert validate_ml_enforcement_mode("BLOCK_ON_THRESHOLD") == "block_on_threshold"
    assert validate_ml_enforcement_mode(MLEnforcementMode.OBSERVE) == "observe"
    assert validate_ml_enforcement_mode(MLEnforcementMode.BLOCK_ON_THRESHOLD) == "block_on_threshold"


def test_validate_ml_enforcement_mode_rejects_unsupported_modes():
    """Verify validate_ml_enforcement_mode strictly rejects unsupported values."""
    invalid_modes = ["block_always", "strict", "allow_all", "", "   ", "random"]
    for inv in invalid_modes:
        with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
            validate_ml_enforcement_mode(inv)

    invalid_types = [None, True, False, 123, 0.5, ["observe"], {"mode": "observe"}]
    for inv_type in invalid_types:
        with pytest.raises(ValueError, match="Invalid ML enforcement mode type"):
            validate_ml_enforcement_mode(inv_type)


def test_get_ml_enforcement_mode_respects_environment_variables(monkeypatch):
    """Verify environment variables PROMPTSHIELD_ML_ENFORCEMENT_MODE and ML_ENFORCEMENT_MODE configure mode."""
    # PROMPTSHIELD_ML_ENFORCEMENT_MODE
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "block_on_threshold")
    assert get_ml_enforcement_mode() == "block_on_threshold"

    # ML_ENFORCEMENT_MODE fallback
    monkeypatch.delenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE")
    monkeypatch.setenv("ML_ENFORCEMENT_MODE", "block_on_threshold")
    assert get_ml_enforcement_mode() == "block_on_threshold"

    # Invalid env var raises ValueError
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "unsupported_mode")
    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        get_ml_enforcement_mode()


def test_set_and_reset_ml_enforcement_mode():
    """Verify set_ml_enforcement_mode overrides config and reset_ml_enforcement_mode restores it."""
    assert get_ml_enforcement_mode() == "observe"

    set_ml_enforcement_mode("block_on_threshold")
    assert get_ml_enforcement_mode() == "block_on_threshold"

    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        set_ml_enforcement_mode("nonexistent")

    reset_ml_enforcement_mode()
    assert get_ml_enforcement_mode() == "observe"


# ---------------------------------------------------------------------------
# 2. Deterministic Sovereignty tests
# ---------------------------------------------------------------------------

def test_deterministic_block_sovereign_in_observe_mode(blocking_rule, available_benign_ml):
    """Verify deterministic block takes absolute precedence over benign ML in observe mode."""
    decision = evaluate_policy(
        matched_rules=[blocking_rule],
        ml_assessment=available_benign_ml,
        mode="observe",
    )
    assert decision.decision == "BLOCK"
    assert decision.decision_source == "deterministic_rule"
    assert "RULE_TEST_BLOCK" in decision.matched_rule_ids
    assert "Request blocked by security policy" in decision.explanation


def test_deterministic_block_sovereign_in_block_on_threshold_mode(blocking_rule, available_benign_ml):
    """Verify deterministic block takes absolute precedence over benign ML in block_on_threshold mode."""
    decision = evaluate_policy(
        matched_rules=[blocking_rule],
        ml_assessment=available_benign_ml,
        mode="block_on_threshold",
    )
    assert decision.decision == "BLOCK"
    assert decision.decision_source == "deterministic_rule"
    assert "RULE_TEST_BLOCK" in decision.matched_rule_ids


def test_deterministic_block_sovereign_when_ml_is_unavailable(blocking_rule, unavailable_ml):
    """Verify deterministic block is enforced even when ML is unavailable."""
    decision = evaluate_policy(
        matched_rules=[blocking_rule],
        ml_assessment=unavailable_ml,
        mode="block_on_threshold",
    )
    assert decision.decision == "BLOCK"
    assert decision.decision_source == "deterministic_rule"
    assert "RULE_TEST_BLOCK" in decision.matched_rule_ids


# ---------------------------------------------------------------------------
# 3. Observe Mode tests
# ---------------------------------------------------------------------------

def test_observe_mode_preserves_allow_even_when_ml_would_block(available_injection_ml):
    """Verify observe mode does not block when ML would_block is True but rules didn't block."""
    decision = evaluate_policy(
        matched_rules=[],
        ml_assessment=available_injection_ml,
        mode="observe",
    )
    assert decision.decision == "ALLOW"
    assert decision.decision_source == "allow_default"
    assert decision.matched_rule_ids == []
    assert "Request allowed" in decision.explanation


def test_observe_mode_preserves_allow_with_flag_rule(flag_rule, available_injection_ml):
    """Verify observe mode does not block with non-blocking FLAG rule and ML injection."""
    decision = evaluate_policy(
        matched_rules=[flag_rule],
        ml_assessment=available_injection_ml,
        mode="observe",
    )
    assert decision.decision == "ALLOW"
    assert decision.decision_source == "allow_default"
    assert decision.risk_score == 0.20
    assert "RULE_TEST_FLAG" in decision.matched_rule_ids


# ---------------------------------------------------------------------------
# 4. Block On Threshold Mode tests
# ---------------------------------------------------------------------------

def test_block_on_threshold_blocks_when_ml_threshold_met(available_injection_ml):
    """Verify block_on_threshold mode blocks an ML-positive prompt when rules didn't match."""
    decision = evaluate_policy(
        matched_rules=[],
        ml_assessment=available_injection_ml,
        mode="block_on_threshold",
    )
    assert decision.decision == "BLOCK"
    assert decision.decision_source == "ml_enforcement"
    assert decision.risk_score == 0.92
    assert "ML prompt injection detection threshold met" in decision.explanation
    assert "score: 0.9200" in decision.explanation
    assert "threshold: 0.50" in decision.explanation


def test_block_on_threshold_allows_when_ml_is_benign(available_benign_ml):
    """Verify block_on_threshold mode allows when ML predicts benign and rules didn't match."""
    decision = evaluate_policy(
        matched_rules=[],
        ml_assessment=available_benign_ml,
        mode="block_on_threshold",
    )
    assert decision.decision == "ALLOW"
    assert decision.decision_source == "allow_default"
    assert decision.risk_score == 0.0
    assert "Request allowed" in decision.explanation


# ---------------------------------------------------------------------------
# 5. Unavailable ML Resiliency tests
# ---------------------------------------------------------------------------

def test_ml_unavailable_in_observe_mode_does_not_block(unavailable_ml):
    """Verify unavailable ML in observe mode preserves deterministic allow."""
    decision = evaluate_policy(
        matched_rules=[],
        ml_assessment=unavailable_ml,
        mode="observe",
    )
    assert decision.decision == "ALLOW"
    assert decision.decision_source == "allow_default"
    assert decision.risk_score == 0.0


def test_ml_unavailable_in_block_on_threshold_mode_does_not_block(unavailable_ml):
    """Verify unavailable ML in block_on_threshold mode does not trigger an ML-only block."""
    decision = evaluate_policy(
        matched_rules=[],
        ml_assessment=unavailable_ml,
        mode="block_on_threshold",
    )
    assert decision.decision == "ALLOW"
    assert decision.decision_source == "allow_default"
    assert decision.risk_score == 0.0


def test_none_ml_assessment_preserves_deterministic_behavior():
    """Verify passing ml_assessment=None preserves deterministic decisions in all modes."""
    # Observe mode
    dec_obs = evaluate_policy(matched_rules=[], ml_assessment=None, mode="observe")
    assert dec_obs.decision == "ALLOW"
    assert dec_obs.decision_source == "allow_default"

    # Block on threshold mode
    dec_blk = evaluate_policy(matched_rules=[], ml_assessment=None, mode="block_on_threshold")
    assert dec_blk.decision == "ALLOW"
    assert dec_blk.decision_source == "allow_default"


# ---------------------------------------------------------------------------
# 6. Evaluation error handling
# ---------------------------------------------------------------------------

def test_evaluate_policy_rejects_invalid_mode():
    """Verify evaluate_policy raises ValueError when an unsupported mode is passed directly."""
    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        evaluate_policy(matched_rules=[], mode="invalid_mode")


# ---------------------------------------------------------------------------
# 7. Startup Configuration Validation tests (Stage 3 Step 5)
# ---------------------------------------------------------------------------

def test_validate_startup_configuration_defaults_to_observe():
    """Verify validate_startup_configuration returns 'observe' when no env vars or overrides exist."""
    assert validate_startup_configuration() == "observe"


def test_validate_startup_configuration_accepts_valid_modes(monkeypatch):
    """Verify validate_startup_configuration accepts both valid modes."""
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "observe")
    assert validate_startup_configuration() == "observe"

    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "block_on_threshold")
    assert validate_startup_configuration() == "block_on_threshold"


def test_validate_startup_configuration_primary_precedence_over_fallback(monkeypatch):
    """Verify primary env var takes precedence over fallback when both are set."""
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "block_on_threshold")
    monkeypatch.setenv("ML_ENFORCEMENT_MODE", "observe")
    assert validate_startup_configuration() == "block_on_threshold"


def test_validate_startup_configuration_invalid_primary_fails_even_with_valid_fallback(monkeypatch):
    """Verify invalid primary env var raises ValueError without falling back to secondary."""
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "invalid_mode")
    monkeypatch.setenv("ML_ENFORCEMENT_MODE", "observe")
    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        validate_startup_configuration()


def test_validate_startup_configuration_empty_primary_fails_even_with_valid_fallback(monkeypatch):
    """Verify empty/whitespace primary env var raises ValueError without falling back to secondary."""
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "")
    monkeypatch.setenv("ML_ENFORCEMENT_MODE", "observe")
    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        validate_startup_configuration()

    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "   ")
    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        validate_startup_configuration()


def test_validate_startup_configuration_invalid_fallback_fails_when_primary_absent(monkeypatch):
    """Verify invalid fallback env var raises ValueError when primary is absent."""
    monkeypatch.delenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", raising=False)
    monkeypatch.setenv("ML_ENFORCEMENT_MODE", "invalid_fallback")
    with pytest.raises(ValueError, match="Unsupported ML enforcement mode"):
        validate_startup_configuration()


def test_validate_startup_configuration_programmatic_override_precedence(monkeypatch):
    """Verify programmatic override takes precedence over both environment variables."""
    monkeypatch.setenv("PROMPTSHIELD_ML_ENFORCEMENT_MODE", "invalid_mode")
    set_ml_enforcement_mode("block_on_threshold")
    assert validate_startup_configuration() == "block_on_threshold"
    reset_ml_enforcement_mode()
