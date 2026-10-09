"""Typed schemas and data models for PromptShield Phase 4 Output Security.

IMPORTANT SAFETY & ARCHITECTURAL CONVENTIONS:
- These schemas represent technical output-security inspection and redaction outcomes.
- They do NOT make policy decisions (e.g. ALLOW vs BLOCK vs REVIEW). Policy decisions
  are reserved exclusively for the gateway policy engine in Stage 4.
- Raw secret materials, plain-text token fragments, and unmasked credentials are
  strictly prohibited from being included in these models.
- Concepts for completed clean scan, completed scan with findings, and failed/exceeded
  scans are represented distinctly and unambiguously.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence

from app.output.rules import OutputFinding


class OutputScanStatus(str, Enum):
    """Technical status of an output security scan (distinct from policy decisions)."""

    CLEAN = "clean"
    FINDINGS_DETECTED = "findings_detected"
    FAILED_OR_EXCEEDED = "failed_or_exceeded"


@dataclass(frozen=True)
class OutputScanResult:
    """Structured technical outcome of scanning downstream plain-text output.

    Attributes:
        has_findings: True if one or more sensitive findings were detected.
        findings: Sequence of OutputFinding items, ordered deterministically.
        finding_count: Total number of resolved findings.
        is_complete: True if the scan completed all rule checks without early failure.
        exceeded_safety_limit: True if a resource/safety limit was reached.
        text_length: Length of original scanned text in characters.
        duration_ms: Wall-clock execution time in milliseconds.
    """

    has_findings: bool
    findings: List[OutputFinding]
    finding_count: int
    is_complete: bool
    exceeded_safety_limit: bool
    text_length: int
    duration_ms: float = 0.0

    @property
    def is_clean(self) -> bool:
        """True if and only if the scan completed fully with zero findings and no limit overflow."""
        return self.is_complete and not self.has_findings and not self.exceeded_safety_limit

    @property
    def matched_rule_ids(self) -> List[str]:
        """List of unique rule IDs that matched in the scanned text."""
        return sorted(list({f.rule_id for f in self.findings}))

    @property
    def status(self) -> OutputScanStatus:
        """Categorical technical status of the scan."""
        if not self.is_complete or self.exceeded_safety_limit:
            return OutputScanStatus.FAILED_OR_EXCEEDED
        if self.has_findings:
            return OutputScanStatus.FINDINGS_DETECTED
        return OutputScanStatus.CLEAN

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scan result to a dictionary omitting all sensitive text."""
        return {
            "status": self.status.value,
            "has_findings": self.has_findings,
            "is_clean": self.is_clean,
            "finding_count": self.finding_count,
            "is_complete": self.is_complete,
            "exceeded_safety_limit": self.exceeded_safety_limit,
            "text_length": self.text_length,
            "matched_rule_ids": self.matched_rule_ids,
            "findings": [f.to_dict() for f in self.findings],
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class RedactionResult:
    """Structured outcome of redacting sensitive spans in plain text.

    Attributes:
        redacted_text: Sanitized text with sensitive spans replaced by non-secret masks.
        redaction_count: Number of distinct spans replaced.
        redacted_rule_ids: Rule IDs associated with the redacted spans.
        original_length: Character length of original input text.
        redacted_length: Character length of resulting redacted text.
    """

    redacted_text: str
    redaction_count: int
    redacted_rule_ids: List[str]
    original_length: int
    redacted_length: int

    @property
    def was_modified(self) -> bool:
        """True if one or more spans were replaced, modifying original text."""
        return self.redaction_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize redaction metadata without leaking raw texts."""
        return {
            "redaction_count": self.redaction_count,
            "was_modified": self.was_modified,
            "redacted_rule_ids": self.redacted_rule_ids,
            "original_length": self.original_length,
            "redacted_length": self.redacted_length,
        }
