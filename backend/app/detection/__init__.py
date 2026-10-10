"""Detection package for deterministic rule and ingress inspection."""

from app.detection.ingress import (
    ALLOWED_INGRESS_ERROR_CODES,
    ALLOWED_INGRESS_VIEWS,
    IngressInspectionResult,
    IngressInspectionViews,
    IngressTransformationError,
    RULE_MALFORMED_ENCODING,
    evaluate_ingress_prompt_ml,
    evaluate_ingress_views_ml,
    extract_ingress_telemetry,
    generate_ingress_views,
    inspect_ingress_prompt,
)
from app.detection.rules import DetectionRule, get_configured_rules, scan_prompt

__all__ = [
    "ALLOWED_INGRESS_ERROR_CODES",
    "ALLOWED_INGRESS_VIEWS",
    "DetectionRule",
    "IngressInspectionResult",
    "IngressInspectionViews",
    "IngressTransformationError",
    "RULE_MALFORMED_ENCODING",
    "evaluate_ingress_prompt_ml",
    "evaluate_ingress_views_ml",
    "extract_ingress_telemetry",
    "generate_ingress_views",
    "get_configured_rules",
    "inspect_ingress_prompt",
    "scan_prompt",
]
