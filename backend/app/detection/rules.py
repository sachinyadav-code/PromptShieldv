"""Deterministic prompt inspection rules for PromptShield Phase 1.

Important Notice:
Rule-based regex matching provides a transparent, deterministic baseline detector.
It can produce false negatives (evaded via semantic paraphrasing, obfuscation,
or encoding) and false positives (benign security discussions). It does not provide
comprehensive prompt injection security on its own.
"""

from dataclasses import dataclass, field
import re
from typing import List, Literal


@dataclass
class DetectionRule:
    """Definition for a deterministic security inspection rule."""

    rule_id: str
    name: str
    description: str
    patterns: List[re.Pattern] = field(repr=False)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    action: Literal["ALLOW", "FLAG", "BLOCK"]
    risk_weight: float  # Heuristic contribution between 0.0 and 1.0


# Configured starter rules for Phase 1
CONFIGURED_RULES: List[DetectionRule] = [
    DetectionRule(
        rule_id="RULE_INSTRUCTION_OVERRIDE",
        name="Instruction Override Detection",
        description="Detects explicit attempts to ignore, disregard, or override prior instructions or system prompts.",
        patterns=[
            re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.IGNORECASE),
            re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.IGNORECASE),
            re.compile(r"\bforget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:rules|instructions)\b", re.IGNORECASE),
            re.compile(r"\boverride\s+(?:the\s+|all\s+)?(?:system|developer)\s+(?:prompts?|instructions?|rules?)\b", re.IGNORECASE),
        ],
        severity="HIGH",
        action="BLOCK",
        risk_weight=0.85,
    ),
    DetectionRule(
        rule_id="RULE_SYSTEM_PROMPT_LEAK",
        name="System Prompt Extraction Detection",
        description="Detects attempts to leak, repeat, or display confidential system or developer prompts.",
        patterns=[
            re.compile(r"\b(?:reveal|show|print|display|dump|repeat|leak)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+prompt|developer\s+instructions?|system\s+instructions?|initial\s+prompt)\b", re.IGNORECASE),
            re.compile(r"\bwhat\s+(?:was|is)\s+your\s+(?:system\s+prompt|initial\s+instructions?)\b", re.IGNORECASE),
        ],
        severity="HIGH",
        action="BLOCK",
        risk_weight=0.80,
    ),
    DetectionRule(
        rule_id="RULE_SAFETY_BYPASS",
        name="Safety Filter Bypass Detection",
        description="Detects adversarial jailbreak personas and direct safety filter bypass attempts.",
        patterns=[
            re.compile(r"\b(?:dan|jailbreak)\s+mode\b", re.IGNORECASE),
            re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
            re.compile(r"\bbypass\s+(?:all\s+)?(?:safety|content)\s+(?:filters?|restrictions?|guidelines?)\b", re.IGNORECASE),
            re.compile(r"\bdisable\s+(?:all\s+)?(?:safety|content|ethical)\s+(?:filters?|guards?|restrictions?)\b", re.IGNORECASE),
        ],
        severity="HIGH",
        action="BLOCK",
        risk_weight=0.90,
    ),
    DetectionRule(
        rule_id="RULE_CREDENTIAL_ACCESS",
        name="Credential & Environment Extraction",
        description="Detects attempts to access internal environment variables, secrets, API keys, or system credentials.",
        patterns=[
            re.compile(r"\b(?:reveal|dump|print|display|leak)\s+(?:your\s+|the\s+)?(?:api[\s_-]?keys?|secret[\s_-]?keys?|environment\s+variables?|env\s+vars?)\b", re.IGNORECASE),
            re.compile(r"\bcat\s+/etc/(?:passwd|shadow)\b", re.IGNORECASE),
            re.compile(r"\baws_secret_access_key\b", re.IGNORECASE),
            re.compile(r"\b(?:process\.env|os\.environ)\b", re.IGNORECASE),
        ],
        severity="CRITICAL",
        action="BLOCK",
        risk_weight=0.95,
    ),
]


def get_configured_rules() -> List[DetectionRule]:
    """Return the list of active detection rules."""
    return CONFIGURED_RULES


def scan_prompt(prompt: str) -> List[DetectionRule]:
    """Evaluate prompt against all configured rules and return matching rules."""
    matched: List[DetectionRule] = []
    for rule in CONFIGURED_RULES:
        for pattern in rule.patterns:
            if pattern.search(prompt):
                matched.append(rule)
                break  # Record rule once if any of its patterns match
    return matched
