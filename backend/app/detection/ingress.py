"""Ingress defense pipeline for PromptShield Phase 7 Stage 7.2.

Provides bounded Unicode normalization and escape/credential decoding
for incoming user prompts, protecting deterministic rules and ML classifiers
against zero-width, homoglyph, percent-encoded, hex-escaped, and HTML-encoded
evasion attempts while strictly preserving the original raw prompt for
downstream forwarding and safe audit fingerprinting.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from app.detection.ml_classifier import MLInferenceResult, get_classifier
from app.detection.rules import DetectionRule, scan_prompt
from app.output.decoder import (
    MAX_DECODED_TEXT_LENGTH,
    OutputDecodedLengthExceededError,
    OutputDecodingError,
    decode_for_detection,
)
from app.output.normalizer import (
    MAX_NORMALIZED_TEXT_LENGTH,
    OutputLengthExceededError,
    OutputNormalizationError,
    OutputNormalizedLengthExceededError,
    normalize_for_detection,
)
from app.schemas import MLAssessment

logger = logging.getLogger("promptshield.detection.ingress")

# Pre-normalization / decoding input bound (aligned with PromptRequest max_length=10,000)
MAX_INGRESS_INPUT_LENGTH: int = 10_000

# Post-transformation bound for ingress detection views (prevents expansion bombs / ReDoS)
MAX_INGRESS_VIEW_LENGTH: int = 50_000


class IngressTransformationError(ValueError):
    """Exception raised when ingress prompt transformation fails closed."""

    def __init__(self, error_code: str, detail: str) -> None:
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


RULE_MALFORMED_ENCODING = DetectionRule(
    rule_id="RULE_MALFORMED_ENCODING",
    name="Malformed or Obfuscated Ingress Encoding",
    description="Detects malformed escape sequences, invalid unicode/byte encoding, or dangerous expansion in prompt ingress.",
    patterns=[],
    severity="HIGH",
    action="BLOCK",
    risk_weight=0.95,
)


@dataclass(frozen=True)
class IngressInspectionViews:
    """Bounded representations of an incoming user prompt for security inspection."""

    raw: str
    normalized: Optional[str]
    decoded: Optional[str]
    combined_normalized_decoded: Optional[str]
    decoded_normalized: Optional[str] = None
    error_code: Optional[str] = None


@dataclass(frozen=True)
class IngressInspectionResult:
    """Aggregated inspection result across all evaluated views."""

    matched_rules: List[DetectionRule]
    ml_assessment: MLAssessment
    transformation_error_code: Optional[str] = None
    views: Optional[IngressInspectionViews] = None


def generate_ingress_views(prompt: str) -> IngressInspectionViews:
    """Generate bounded normalized and decoded views of the prompt for inspection.

    Safety Invariants:
    1. Input length is validated (<= 10,000 characters).
    2. Strips 17 invisible characters and applies NFKC normalization.
    3. Single-pass decodes URL-percent, hex, unicode, and HTML numeric references.
    4. Rejects malformed encoding sequences (e.g. invalid UTF-8 bytes, null bytes)
       and expansion bombs (> 50,000 characters) by raising IngressTransformationError.
    5. Never executes decoded content or modifies the caller's raw prompt.
    """
    if not isinstance(prompt, str):
        raise IngressTransformationError(
            error_code="INVALID_PROMPT_TYPE",
            detail="Prompt must be a string.",
        )

    if len(prompt) > MAX_INGRESS_INPUT_LENGTH:
        raise IngressTransformationError(
            error_code="PROMPT_LENGTH_EXCEEDED",
            detail=f"Prompt length exceeds maximum allowed limit of {MAX_INGRESS_INPUT_LENGTH} characters.",
        )

    # 1. Unicode Normalization (invisible stripping + NFKC)
    try:
        normalized = normalize_for_detection(
            prompt,
            max_input_length=MAX_INGRESS_INPUT_LENGTH,
            max_normalized_length=MAX_INGRESS_VIEW_LENGTH,
        )
    except (OutputLengthExceededError, OutputNormalizedLengthExceededError) as exc:
        logger.warning("Ingress normalization length limit exceeded: %s", type(exc).__name__)
        raise IngressTransformationError(
            error_code="NORMALIZATION_LIMIT_EXCEEDED",
            detail="Prompt exceeds normalized character expansion limit.",
        ) from exc
    except OutputNormalizationError as exc:
        logger.warning("Ingress normalization internal failure: %s", type(exc).__name__)
        raise IngressTransformationError(
            error_code="NORMALIZATION_FAILED",
            detail="Prompt normalization failed.",
        ) from exc

    # 2. Credential and Escape Sequence Decoding of Raw Input
    try:
        decoded = decode_for_detection(
            prompt,
            max_input_length=MAX_INGRESS_INPUT_LENGTH,
            max_decoded_length=MAX_INGRESS_VIEW_LENGTH,
        )
    except (OutputLengthExceededError, OutputDecodedLengthExceededError) as exc:
        logger.warning("Ingress decoding length limit exceeded: %s", type(exc).__name__)
        raise IngressTransformationError(
            error_code="DECODING_LIMIT_EXCEEDED",
            detail="Prompt exceeds decoded character expansion limit.",
        ) from exc
    except OutputDecodingError as exc:
        logger.warning("Ingress decoding rejected malformed sequence: %s", type(exc).__name__)
        raise IngressTransformationError(
            error_code="MALFORMED_ENCODING",
            detail="Prompt contains malformed or forbidden escape sequences.",
        ) from exc

    # 3. Combined Normalized + Decoded View (normalizes the decoded representation)
    try:
        combined = normalize_for_detection(
            decoded,
            max_input_length=MAX_INGRESS_VIEW_LENGTH,
            max_normalized_length=MAX_INGRESS_VIEW_LENGTH,
        )
    except Exception as exc:
        logger.warning("Combined normalization of decoded view failed: %s", type(exc).__name__)
        raise IngressTransformationError(
            error_code="COMBINED_TRANSFORMATION_FAILED",
            detail="Combined prompt transformation failed.",
        ) from exc

    # 4. Decoded Normalized View (decodes normalized view to reveal intra-token split escapes)
    decoded_normalized = None
    if normalized != prompt:
        try:
            raw_decoded_norm = decode_for_detection(
                normalized,
                max_input_length=MAX_INGRESS_VIEW_LENGTH,
                max_decoded_length=MAX_INGRESS_VIEW_LENGTH,
            )
            # Normalize the decoded result if decoding occurred
            if raw_decoded_norm != normalized:
                decoded_normalized = normalize_for_detection(
                    raw_decoded_norm,
                    max_input_length=MAX_INGRESS_VIEW_LENGTH,
                    max_normalized_length=MAX_INGRESS_VIEW_LENGTH,
                )
            else:
                decoded_normalized = raw_decoded_norm
        except (OutputLengthExceededError, OutputDecodedLengthExceededError) as exc:
            logger.warning("Ingress normalized decoding length limit exceeded: %s", type(exc).__name__)
            raise IngressTransformationError(
                error_code="DECODING_LIMIT_EXCEEDED",
                detail="Normalized prompt exceeds decoded character expansion limit.",
            ) from exc
        except OutputDecodingError as exc:
            logger.warning("Ingress normalized decoding rejected malformed sequence: %s", type(exc).__name__)
            raise IngressTransformationError(
                error_code="MALFORMED_ENCODING",
                detail="Normalized prompt contains malformed or forbidden escape sequences.",
            ) from exc
        except Exception as exc:
            logger.warning("Normalization of decoded normalized view failed: %s", type(exc).__name__)
            raise IngressTransformationError(
                error_code="COMBINED_TRANSFORMATION_FAILED",
                detail="Decoded normalized prompt transformation failed.",
            ) from exc

    return IngressInspectionViews(
        raw=prompt,
        normalized=normalized,
        decoded=decoded,
        combined_normalized_decoded=combined,
        decoded_normalized=decoded_normalized,
        error_code=None,
    )


def evaluate_ingress_views_ml(
    views: IngressInspectionViews,
    candidate_texts: Optional[List[str]] = None,
    classifier: Optional[Any] = None,
) -> MLAssessment:
    """Evaluate ML classifier across ingress prompt views, selecting the highest risk assessment.

    Evaluates the raw prompt view first, and then compares against all distinct transformed views
    (normalized, decoded, combined, decoded_normalized). The assessment with the highest valid
    risk score is returned, ensuring obfuscated injection attempts in any view are captured.
    If an inference fails for a specific view, valid assessments are never overridden by unavailable ones.
    """
    if candidate_texts is None:
        candidate_texts = [views.raw]
        if views.normalized and views.normalized != views.raw:
            candidate_texts.append(views.normalized)
        if views.decoded and views.decoded != views.raw:
            candidate_texts.append(views.decoded)
        if views.combined_normalized_decoded and views.combined_normalized_decoded not in candidate_texts:
            candidate_texts.append(views.combined_normalized_decoded)
        if views.decoded_normalized and views.decoded_normalized not in candidate_texts:
            candidate_texts.append(views.decoded_normalized)

    best_ml_assessment = _evaluate_ml_view(views.raw, classifier=classifier)

    for text_view in candidate_texts[1:]:
        view_ml = _evaluate_ml_view(text_view, classifier=classifier)
        if view_ml.available:
            if not best_ml_assessment.available:
                best_ml_assessment = view_ml
            elif view_ml.score is not None and (best_ml_assessment.score is None or view_ml.score > best_ml_assessment.score):
                best_ml_assessment = view_ml

    return best_ml_assessment


def evaluate_ingress_prompt_ml(
    prompt: str,
    classifier: Optional[Any] = None,
) -> MLAssessment:
    """Safely generate ingress views for a prompt and evaluate ML classifier across all views.

    If an IngressTransformationError occurs, falls back gracefully to evaluating the raw prompt.
    """
    try:
        views = generate_ingress_views(prompt)
        return evaluate_ingress_views_ml(views, classifier=classifier)
    except IngressTransformationError:
        return _evaluate_ml_view(prompt, classifier=classifier)


def inspect_ingress_prompt(prompt: str) -> IngressInspectionResult:
    """Inspect user prompt across raw, normalized, and decoded views.

    Combines:
    1. Deterministic rules across all views (union of matched rules, preserving priority).
    2. ML classifier assessment across views (selecting the highest risk assessment).
    """
    views = generate_ingress_views(prompt)

    # 1. Deterministic Rule Matching across all views
    candidate_texts = [views.raw]
    if views.normalized and views.normalized != views.raw:
        candidate_texts.append(views.normalized)
    if views.decoded and views.decoded != views.raw:
        candidate_texts.append(views.decoded)
    if views.combined_normalized_decoded and views.combined_normalized_decoded not in candidate_texts:
        candidate_texts.append(views.combined_normalized_decoded)
    if views.decoded_normalized and views.decoded_normalized not in candidate_texts:
        candidate_texts.append(views.decoded_normalized)

    matched_dict: Dict[str, DetectionRule] = {}
    for text_view in candidate_texts:
        for rule in scan_prompt(text_view):
            if rule.rule_id not in matched_dict:
                matched_dict[rule.rule_id] = rule

    all_matched_rules = list(matched_dict.values())

    # 2. ML Classifier Assessment across all views
    best_ml_assessment = evaluate_ingress_views_ml(views, candidate_texts=candidate_texts)

    return IngressInspectionResult(
        matched_rules=all_matched_rules,
        ml_assessment=best_ml_assessment,
        transformation_error_code=None,
        views=views,
    )


def _evaluate_ml_view(text: str, classifier: Optional[Any] = None) -> MLAssessment:
    """Evaluate ML classifier for a specific text view with resilient fallback."""
    try:
        clf = classifier if classifier is not None else get_classifier()
        inference_result = clf.predict(text)
        if isinstance(inference_result, MLAssessment):
            return inference_result
        return MLAssessment.from_inference_result(inference_result)
    except Exception as exc:
        logger.warning("ML assessment failed on view: %s", type(exc).__name__)
        fallback_threshold = 0.50
        try:
            clf = classifier if classifier is not None else get_classifier()
            fallback_threshold = clf.threshold
        except Exception:
            pass
        return MLAssessment.unavailable(
            threshold=fallback_threshold,
            error_code="INFERENCE_FAILED",
        )