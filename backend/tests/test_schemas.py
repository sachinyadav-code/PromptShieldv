"""Unit tests for PromptShield API schemas (Phase 2 Stage 3 Step 2).

Verifies MLAssessment schema validation, unavailable-result semantics,
uncalibrated score boundaries, field consistency, serialization/deserialization,
and backward compatibility of InspectResponse and ProxyResponse.
"""

import json
import pytest
from pydantic import ValidationError

from app.detection.ml_classifier import MLInferenceResult
from app.schemas import (
    InspectResponse,
    MLAnalysisResult,
    MLAssessment,
    MLDetectionResult,
    MLInferenceAssessment,
    MockDownstreamResponse,
    PromptRequest,
    ProxyResponse,
    RuleInfo,
)


# ---------------------------------------------------------------------------
# 1. VALID AVAILABLE ASSESSMENT TESTS
# ---------------------------------------------------------------------------


def test_valid_available_injection_assessment():
    """Verify available injection assessment accepts valid fields and provides property access."""
    assessment = MLAssessment(
        available=True,
        score=0.8523,
        threshold=0.50,
        predicted_label="injection",
        would_block=True,
        model_version="v1",
        error_code=None,
    )
    assert assessment.available is True
    assert assessment.score == 0.8523
    assert assessment.threshold == 0.50
    assert assessment.predicted_label == "injection"
    assert assessment.would_block is True
    assert assessment.threshold_met is True  # Alias property
    assert assessment.model_version == "v1"
    assert assessment.error_code is None


def test_valid_available_benign_assessment():
    """Verify available benign assessment accepts valid fields and provides property access."""
    assessment = MLAssessment(
        available=True,
        score=0.1245,
        threshold=0.50,
        predicted_label="benign",
        would_block=False,
        model_version="v1",
        error_code=None,
    )
    assert assessment.available is True
    assert assessment.score == 0.1245
    assert assessment.threshold == 0.50
    assert assessment.predicted_label == "benign"
    assert assessment.would_block is False
    assert assessment.threshold_met is False  # Alias property
    assert assessment.model_version == "v1"
    assert assessment.error_code is None


def test_valid_available_with_threshold_met_alias():
    """Verify validation_alias accepts threshold_met during initialization."""
    assessment = MLAssessment(
        available=True,
        score=0.91,
        threshold=0.50,
        predicted_label="injection",
        threshold_met=True,  # Alias for would_block
        model_version="v1",
    )
    assert assessment.would_block is True
    assert assessment.threshold_met is True


def test_valid_assessment_at_exact_boundaries():
    """Verify valid assessment at extreme numerical boundaries [0.0, 1.0]."""
    # Lower boundary: score 0.0, threshold 0.0, would_block True
    a_low = MLAssessment(
        available=True,
        score=0.0,
        threshold=0.0,
        predicted_label="injection",
        would_block=True,
    )
    assert a_low.score == 0.0
    assert a_low.threshold == 0.0

    # Upper boundary: score 1.0, threshold 1.0, would_block True
    a_high = MLAssessment(
        available=True,
        score=1.0,
        threshold=1.0,
        predicted_label="injection",
        would_block=True,
    )
    assert a_high.score == 1.0
    assert a_high.threshold == 1.0

    # Zero score benign
    a_zero = MLAssessment(
        available=True,
        score=0.0,
        threshold=0.50,
        predicted_label="benign",
        would_block=False,
    )
    assert a_zero.score == 0.0
    assert a_zero.predicted_label == "benign"


# ---------------------------------------------------------------------------
# 2. VALID UNAVAILABLE ASSESSMENT TESTS
# ---------------------------------------------------------------------------


def test_valid_unavailable_assessment_with_error_code():
    """Verify valid unavailable assessment with explicit error code."""
    assessment = MLAssessment(
        available=False,
        score=None,
        threshold=0.50,
        predicted_label=None,
        would_block=False,
        error_code="MODEL_UNAVAILABLE",
    )
    assert assessment.available is False
    assert assessment.score is None
    assert assessment.predicted_label is None
    assert assessment.would_block is False
    assert assessment.threshold_met is False
    assert assessment.error_code == "MODEL_UNAVAILABLE"


def test_valid_unavailable_assessment_without_error_code():
    """Verify unavailable assessment without error code is permitted."""
    assessment = MLAssessment(
        available=False,
        score=None,
        threshold=0.50,
        predicted_label=None,
        would_block=False,
        error_code=None,
    )
    assert assessment.available is False
    assert assessment.score is None
    assert assessment.predicted_label is None
    assert assessment.would_block is False
    assert assessment.error_code is None


def test_unavailable_factory_helper():
    """Verify MLAssessment.unavailable factory method constructs valid instances with explicit threshold."""
    a = MLAssessment.unavailable(
        threshold=0.50,
        error_code="CORRUPT_ARTIFACT",
        model_version="v1",
    )
    assert a.available is False
    assert a.score is None
    assert a.predicted_label is None
    assert a.would_block is False
    assert a.threshold == 0.50
    assert a.model_version == "v1"
    assert a.error_code == "CORRUPT_ARTIFACT"


def test_omitting_threshold_fails_validation():
    """Verify omitting the threshold fails validation for available, unavailable, and helper paths."""
    # Available omitting threshold
    with pytest.raises(ValidationError) as exc_avail:
        MLAssessment(  # type: ignore
            available=True,
            score=0.85,
            predicted_label="injection",
            would_block=True,
        )
    assert "threshold" in str(exc_avail.value)

    # Unavailable omitting threshold
    with pytest.raises(ValidationError) as exc_unavail:
        MLAssessment(  # type: ignore
            available=False,
            would_block=False,
            error_code="MODEL_UNAVAILABLE",
        )
    assert "threshold" in str(exc_unavail.value)

    # Helper unavailable() omitting threshold raises TypeError
    with pytest.raises(TypeError):
        MLAssessment.unavailable()  # type: ignore


def test_available_and_unavailable_preserve_non_default_threshold():
    """Verify available and unavailable assessments preserve arbitrary non-default threshold such as 0.37."""
    custom_threshold = 0.37

    # Available injection assessment with 0.37
    a_avail = MLAssessment(
        available=True,
        score=0.45,
        threshold=custom_threshold,
        predicted_label="injection",
        would_block=True,
        model_version="v1",
    )
    assert a_avail.threshold == custom_threshold
    assert a_avail.model_dump()["threshold"] == custom_threshold

    # Available benign assessment with 0.37
    a_benign = MLAssessment(
        available=True,
        score=0.20,
        threshold=custom_threshold,
        predicted_label="benign",
        would_block=False,
        model_version="v1",
    )
    assert a_benign.threshold == custom_threshold

    # Unavailable assessment with 0.37
    a_unavail = MLAssessment(
        available=False,
        threshold=custom_threshold,
        would_block=False,
        error_code="MODEL_UNAVAILABLE",
    )
    assert a_unavail.threshold == custom_threshold

    # Unavailable via factory helper with 0.37
    a_helper = MLAssessment.unavailable(
        threshold=custom_threshold,
        error_code="CORRUPT_ARTIFACT",
        model_version="v1",
    )
    assert a_helper.threshold == custom_threshold


def test_conversion_from_inference_result_retains_original_threshold():
    """Verify conversion from MLInferenceResult retains arbitrary non-default threshold."""
    custom_threshold = 0.37

    # Case 1: Available MLInferenceResult with threshold 0.37
    res_avail = MLInferenceResult(
        available=True,
        score=0.45,
        threshold=custom_threshold,
        predicted_label="injection",
        would_block=True,
        model_version="v2",
        error_code=None,
    )
    assessment_avail = MLAssessment.from_inference_result(res_avail)
    assert assessment_avail.threshold == custom_threshold
    assert assessment_avail.score == 0.45
    assert assessment_avail.would_block is True

    # Case 2: Unavailable MLInferenceResult with threshold 0.37
    res_unavail = MLInferenceResult(
        available=False,
        score=None,
        threshold=custom_threshold,
        predicted_label=None,
        would_block=False,
        model_version="v2",
        error_code="ARTIFACT_NOT_FOUND",
    )
    assessment_unavail = MLAssessment.from_inference_result(res_unavail)
    assert assessment_unavail.threshold == custom_threshold
    assert assessment_unavail.available is False
    assert assessment_unavail.error_code == "ARTIFACT_NOT_FOUND"

    # Case 3: Dictionary representation with threshold 0.37
    dict_payload = {
        "available": True,
        "score": 0.20,
        "threshold": custom_threshold,
        "predicted_label": "benign",
        "would_block": False,
        "model_version": "v1",
        "error_code": None,
    }
    assessment_dict = MLAssessment.from_inference_result(dict_payload)
    assert assessment_dict.threshold == custom_threshold


# ---------------------------------------------------------------------------
# 3. UNAVAILABLE SEMANTICS TESTS (MUST NOT IMPLY BENIGN / SAFE)
# ---------------------------------------------------------------------------


def test_unavailable_assessment_rejects_benign_label():
    """Unavailable assessment must NOT represent result as benign (never imply safety)."""
    with pytest.raises(ValidationError, match="predicted_label must be None when inference is unavailable"):
        MLAssessment(
            available=False,
            predicted_label="benign",
            would_block=False,
            threshold=0.50,
        )


def test_unavailable_assessment_rejects_injection_label():
    """Unavailable assessment must NOT specify an injection label."""
    with pytest.raises(ValidationError, match="predicted_label must be None when inference is unavailable"):
        MLAssessment(
            available=False,
            predicted_label="injection",
            would_block=False,
            threshold=0.50,
        )


def test_unavailable_assessment_rejects_score():
    """Unavailable assessment must NOT have a score present."""
    with pytest.raises(ValidationError, match="score must be None when inference is unavailable"):
        MLAssessment(
            available=False,
            score=0.25,
            would_block=False,
            threshold=0.50,
        )


def test_unavailable_assessment_rejects_would_block_true():
    """Unavailable assessment cannot claim threshold was met."""
    with pytest.raises(ValidationError, match="would_block must be False"):
        MLAssessment(
            available=False,
            would_block=True,
            threshold=0.50,
        )


# ---------------------------------------------------------------------------
# 4. AVAILABLE INCONSISTENT FIELDS TESTS
# ---------------------------------------------------------------------------


def test_available_assessment_requires_score():
    """Available assessment requires a non-None numeric score."""
    with pytest.raises(ValidationError, match="score is required when inference is available"):
        MLAssessment(
            available=True,
            score=None,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
        )


def test_available_assessment_requires_predicted_label():
    """Available assessment requires a non-None predicted_label."""
    with pytest.raises(ValidationError, match="predicted_label is required when inference is available"):
        MLAssessment(
            available=True,
            score=0.75,
            threshold=0.50,
            predicted_label=None,
            would_block=True,
        )


def test_available_assessment_rejects_error_code():
    """Available assessment must not report an error_code."""
    with pytest.raises(ValidationError, match="error_code must be None when inference succeeded"):
        MLAssessment(
            available=True,
            score=0.75,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
            error_code="MODEL_UNAVAILABLE",
        )


def test_available_assessment_rejects_label_and_blocking_mismatch():
    """Verify inconsistency between predicted_label and would_block is rejected."""
    # Case A: predicted_label is 'injection', but would_block is False
    with pytest.raises(ValidationError, match="predicted_label is 'injection' but would_block is False"):
        MLAssessment(
            available=True,
            score=0.85,
            threshold=0.50,
            predicted_label="injection",
            would_block=False,
        )

    # Case B: predicted_label is 'benign', but would_block is True
    with pytest.raises(ValidationError, match="predicted_label is 'benign' but would_block is True"):
        MLAssessment(
            available=True,
            score=0.15,
            threshold=0.50,
            predicted_label="benign",
            would_block=True,
        )


# ---------------------------------------------------------------------------
# 5. NUMERICAL BOUNDS & TYPE VALIDATION TESTS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_score", [-0.01, -1.0, 1.01, 100.0])
def test_rejects_out_of_bounds_score(invalid_score):
    """Verify score strictly outside [0.0, 1.0] is rejected."""
    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=invalid_score,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
        )


@pytest.mark.parametrize("invalid_thresh", [-0.01, -0.5, 1.01, 2.0])
def test_rejects_out_of_bounds_threshold(invalid_thresh):
    """Verify threshold strictly outside [0.0, 1.0] is rejected."""
    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=0.50,
            threshold=invalid_thresh,
            predicted_label="injection",
            would_block=True,
        )


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_values(bad_val):
    """Verify NaN, Inf, and -Inf are rejected on score and threshold."""
    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=bad_val,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
        )
    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=0.50,
            threshold=bad_val,
            predicted_label="injection",
            would_block=True,
        )


@pytest.mark.parametrize("bad_bool", [True, False])
def test_rejects_booleans_passed_as_floats(bad_bool):
    """Verify Python booleans are strictly rejected for float fields."""
    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=bad_bool,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
        )
    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=0.50,
            threshold=bad_bool,
            predicted_label="injection",
            would_block=True,
        )


def test_rejects_non_boolean_available():
    """Verify non-boolean available values are rejected in strict mode."""
    with pytest.raises(ValidationError):
        MLAssessment(
            available="true",  # type: ignore
            threshold=0.50,
            would_block=False,
        )
    with pytest.raises(ValidationError):
        MLAssessment(
            available=1,  # type: ignore
            threshold=0.50,
            would_block=False,
        )


# ---------------------------------------------------------------------------
# 6. SENSITIVE LEAKAGE PREVENTION & FORMAT SAFETY TESTS
# ---------------------------------------------------------------------------


def test_error_code_format_safety():
    """Verify error_code only accepts safe identifier tokens and rejects paths or traces."""
    # Valid screaming snake case tokens
    for code in [
        "MODEL_UNAVAILABLE",
        "CORRUPT_ARTIFACT",
        "ARTIFACT_NOT_FOUND",
        "MALFORMED_OUTPUT",
        "COMPLEX_VALUED_OUTPUT",
        "NON_NUMERIC_OUTPUT",
        "NON_FINITE_OUTPUT",
        "INVALID_PROBABILITY_RANGE",
        "INVALID_PROBABILITY_SUM",
        "INFERENCE_FAILED",
    ]:
        a = MLAssessment.unavailable(threshold=0.50, error_code=code)
        assert a.error_code == code

    # Reject stack trace
    with pytest.raises(ValidationError):
        MLAssessment.unavailable(threshold=0.50, error_code="Traceback (most recent call last):\n  File ...")

    # Reject file paths
    with pytest.raises(ValidationError):
        MLAssessment.unavailable(threshold=0.50, error_code="C:\\app\\models\\prompt_injection_v1.joblib")

    with pytest.raises(ValidationError):
        MLAssessment.unavailable(threshold=0.50, error_code="/etc/shadow")


def test_model_version_format_safety():
    """Verify model_version only accepts safe identifier tokens and rejects file paths."""
    for ver in ["v1", "v2.0", "prompt_injection_v1", "model-2026-10"]:
        a = MLAssessment(
            available=True,
            score=0.8,
            threshold=0.5,
            predicted_label="injection",
            would_block=True,
            model_version=ver,
        )
        assert a.model_version == ver

    with pytest.raises(ValidationError):
        MLAssessment(
            available=True,
            score=0.8,
            threshold=0.5,
            predicted_label="injection",
            would_block=True,
            model_version="C:\\Windows\\system32",
        )


# ---------------------------------------------------------------------------
# 7. FACTORY FROM CLASSIFIER RESULT TESTS
# ---------------------------------------------------------------------------


def test_from_inference_result_with_dataclass():
    """Verify MLAssessment.from_inference_result cleanly ingests MLInferenceResult."""
    # Available result
    res_avail = MLInferenceResult(
        available=True,
        score=0.8251,
        threshold=0.50,
        predicted_label="injection",
        would_block=True,
        model_version="v1",
        error_code=None,
    )
    schema_avail = MLAssessment.from_inference_result(res_avail)
    assert schema_avail.available is True
    assert schema_avail.score == 0.8251
    assert schema_avail.threshold == 0.50
    assert schema_avail.predicted_label == "injection"
    assert schema_avail.would_block is True
    assert schema_avail.model_version == "v1"
    assert schema_avail.error_code is None

    # Unavailable result
    res_unavail = MLInferenceResult(
        available=False,
        score=None,
        threshold=0.50,
        predicted_label=None,
        would_block=False,
        model_version="v1",
        error_code="CORRUPT_ARTIFACT",
    )
    schema_unavail = MLAssessment.from_inference_result(res_unavail)
    assert schema_unavail.available is False
    assert schema_unavail.score is None
    assert schema_unavail.predicted_label is None
    assert schema_unavail.would_block is False
    assert schema_unavail.error_code == "CORRUPT_ARTIFACT"


def test_from_inference_result_with_dict():
    """Verify MLAssessment.from_inference_result ingests dictionary representations."""
    d = {
        "available": True,
        "score": 0.15,
        "threshold": 0.50,
        "predicted_label": "benign",
        "would_block": False,
        "model_version": "v1",
        "error_code": None,
    }
    schema = MLAssessment.from_inference_result(d)
    assert schema.available is True
    assert schema.score == 0.15
    assert schema.predicted_label == "benign"
    assert schema.would_block is False


# ---------------------------------------------------------------------------
# 8. SERIALIZATION & ROUND-TRIP TESTS
# ---------------------------------------------------------------------------


def test_serialization_to_dict_and_json():
    """Verify model_dump and model_dump_json produce expected clean output."""
    assessment = MLAssessment(
        available=True,
        score=0.88,
        threshold=0.50,
        predicted_label="injection",
        would_block=True,
        model_version="v1",
        error_code=None,
    )

    data_dict = assessment.model_dump()
    assert data_dict == {
        "available": True,
        "score": 0.88,
        "threshold": 0.50,
        "predicted_label": "injection",
        "would_block": True,
        "model_version": "v1",
        "error_code": None,
    }

    json_str = assessment.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["available"] is True
    assert parsed["score"] == 0.88
    assert parsed["threshold"] == 0.50
    assert parsed["predicted_label"] == "injection"
    assert parsed["would_block"] == True
    assert parsed["model_version"] == "v1"
    assert parsed["error_code"] is None


def test_round_trip_json_deserialization():
    """Verify JSON serialization can be re-validated into an identical MLAssessment."""
    orig = MLAssessment(
        available=True,
        score=0.2345,
        threshold=0.50,
        predicted_label="benign",
        would_block=False,
        model_version="v1",
        error_code=None,
    )
    json_data = orig.model_dump_json()
    restored = MLAssessment.model_validate_json(json_data)
    assert restored == orig


# ---------------------------------------------------------------------------
# 9. INTEGRATION WITH GATEWAY RESPONSES & BACKWARD COMPATIBILITY
# ---------------------------------------------------------------------------


def test_inspect_response_backward_compatibility():
    """Verify InspectResponse works without ml_assessment (Phase 1) and with ml_assessment."""
    # Phase 1 usage: ml_assessment omitted
    resp_p1 = InspectResponse(
        decision="ALLOW",
        risk_score=0.0,
        matched_rules=[],
        explanation="Request allowed: no threat patterns detected.",
        request_id="test-req-123",
    )
    assert resp_p1.decision == "ALLOW"
    assert resp_p1.ml_assessment is None
    assert resp_p1.ml_detection is None  # Alias property

    # Phase 2 usage: ml_assessment provided
    ml_eval = MLAssessment(
        available=True,
        score=0.05,
        threshold=0.50,
        predicted_label="benign",
        would_block=False,
        model_version="v1",
    )
    resp_p2 = InspectResponse(
        decision="ALLOW",
        risk_score=0.05,
        matched_rules=[],
        explanation="Request allowed: no threat patterns detected.",
        request_id="test-req-123",
        ml_assessment=ml_eval,
    )
    assert resp_p2.ml_assessment is ml_eval
    assert resp_p2.ml_detection is ml_eval


def test_proxy_response_backward_compatibility():
    """Verify ProxyResponse works without ml_assessment (Phase 1) and with ml_assessment."""
    # Phase 1 usage: ml_assessment omitted
    resp_p1 = ProxyResponse(
        decision="ALLOW",
        request_id="test-proxy-456",
        matched_rules=[],
        explanation="Request allowed: forwarded to downstream service.",
        downstream_response=MockDownstreamResponse(
            message="OK",
            received_prompt_length=15,
        ),
    )
    assert resp_p1.decision == "ALLOW"
    assert resp_p1.ml_assessment is None
    assert resp_p1.ml_detection is None

    # Phase 2 usage: ml_assessment provided
    ml_eval = MLAssessment(
        available=True,
        score=0.92,
        threshold=0.50,
        predicted_label="injection",
        would_block=True,
        model_version="v1",
    )
    resp_p2 = ProxyResponse(
        decision="BLOCK",
        request_id="test-proxy-456",
        matched_rules=["RULE_INSTRUCTION_OVERRIDE"],
        explanation="Request blocked by security rules.",
        downstream_response=None,
        ml_assessment=ml_eval,
    )
    assert resp_p2.ml_assessment is ml_eval
    assert resp_p2.ml_detection is ml_eval


def test_schema_aliases_are_identical():
    """Verify all defined schema aliases reference the exact same MLAssessment class."""
    assert MLDetectionResult is MLAssessment
    assert MLInferenceAssessment is MLAssessment
    assert MLAnalysisResult is MLAssessment


def test_response_schemas_decision_source_field():
    """Verify InspectResponse and ProxyResponse support decision_source field and validate allowed values."""
    for valid_source in ["deterministic_rule", "ml_enforcement", "allow_default", None]:
        resp_i = InspectResponse(
            decision="ALLOW",
            risk_score=0.0,
            matched_rules=[],
            explanation="OK",
            request_id="req-1",
            decision_source=valid_source,
        )
        assert resp_i.decision_source == valid_source

        resp_p = ProxyResponse(
            decision="ALLOW",
            request_id="req-2",
            matched_rules=[],
            explanation="OK",
            decision_source=valid_source,
        )
        assert resp_p.decision_source == valid_source

    # Invalid value rejected by Pydantic literal validation
    with pytest.raises(Exception):
        InspectResponse(
            decision="ALLOW",
            risk_score=0.0,
            matched_rules=[],
            explanation="OK",
            request_id="req-1",
            decision_source="invalid_source",
        )
