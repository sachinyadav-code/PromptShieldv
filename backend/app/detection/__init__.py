"""Detection package for deterministic rule inspection."""

from app.detection.rules import DetectionRule, get_configured_rules, scan_prompt

__all__ = ["DetectionRule", "get_configured_rules", "scan_prompt"]
