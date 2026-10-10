"""Detection package for deterministic rule and ingress inspection."""

from app.detection.ingress import (
    IngressInspectionResult,
    IngressInspectionViews,
    IngressTransformationError,
    RULE_MALFORMED_ENCODING,
    generate_ingress_views,
    inspect_ingress_prompt,
)
from app.detection.rules import DetectionRule, get_configured_rules, scan_prompt

__all__ = [
    "DetectionRule",
    "IngressInspectionResult",
    "IngressInspectionViews",
    "IngressTransformationError",
    "RULE_MALFORMED_ENCODING",
    "generate_ingress_views",
    "get_configured_rules",
    "inspect_ingress_prompt",
    "scan_prompt",
]
