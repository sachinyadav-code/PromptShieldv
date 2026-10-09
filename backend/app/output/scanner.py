"""Output security scanner module for PromptShield Phase 4.

Provides a reusable scanner that evaluates plain-text model responses against
deterministic DLP rules and returns structured OutputScanResult models.
"""

import time
from typing import Optional

from app.output.rules import (
    MAX_FINDINGS_COUNT,
    MAX_OUTPUT_TEXT_LENGTH,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
    scan_output_text,
)
from app.output.schemas import OutputScanResult, OutputScanStatus


class OutputScanner:
    """Reusable plain-text output security scanner.

    Invokes the deterministic DLP rules in app.output.rules, enforcing
    input length limits, finding count ceilings, and fail-closed exception
    propagation.
    """

    def __init__(
        self,
        max_length: int = MAX_OUTPUT_TEXT_LENGTH,
        max_findings: int = MAX_FINDINGS_COUNT,
        resolve_overlaps: bool = True,
    ) -> None:
        """Initialize OutputScanner with configurable boundaries.

        Args:
            max_length: Maximum allowed text length in characters (default: 25,000).
            max_findings: Maximum allowed finding count before overflow (default: 100).
            resolve_overlaps: If True, resolves overlapping spans into strictly disjoint
                findings according to the strict priority order (default: True).
        """
        self.max_length = max_length
        self.max_findings = max_findings
        self.resolve_overlaps = resolve_overlaps

    def scan(self, text: str) -> OutputScanResult:
        """Scan plain-text output and return a structured OutputScanResult.

        IMPORTANT SAFETY INVARIANTS:
        - A scan failure or safety limit overflow is NEVER swallowed or returned as clean;
          it raises an explicit, safe typed exception.
        - Completed clean scans return `is_clean=True` and `has_findings=False`.
        - Completed scans with findings return `is_clean=False` and `has_findings=True`.

        Args:
            text: Plain-text string to scan.

        Returns:
            OutputScanResult containing findings metadata and technical metrics.

        Raises:
            TypeError: If text is not a string.
            OutputLengthExceededError: If text length exceeds max_length.
            OutputFindingLimitExceededError: If finding count exceeds max_findings.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")

        start_time = time.perf_counter()

        # scan_output_text enforces length and finding limits, raising typed errors on overflow
        findings = scan_output_text(
            text,
            max_length=self.max_length,
            resolve_overlaps=self.resolve_overlaps,
            max_findings=self.max_findings,
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return OutputScanResult(
            has_findings=len(findings) > 0,
            findings=findings,
            finding_count=len(findings),
            is_complete=True,
            exceeded_safety_limit=False,
            text_length=len(text),
            duration_ms=duration_ms,
        )


def scan_output(
    text: str,
    max_length: int = MAX_OUTPUT_TEXT_LENGTH,
    max_findings: int = MAX_FINDINGS_COUNT,
    resolve_overlaps: bool = True,
) -> OutputScanResult:
    """Convenience function to scan plain-text output using an ephemeral OutputScanner.

    Args:
        text: Plain-text string to scan.
        max_length: Maximum allowed text length in characters (default: 25,000).
        max_findings: Maximum allowed finding count (default: 100).
        resolve_overlaps: If True, resolves overlapping spans deterministically (default: True).

    Returns:
        Structured OutputScanResult.

    Raises:
        TypeError: If text is not a string.
        OutputLengthExceededError: If text exceeds max_length.
        OutputFindingLimitExceededError: If finding count exceeds max_findings.
    """
    scanner = OutputScanner(
        max_length=max_length,
        max_findings=max_findings,
        resolve_overlaps=resolve_overlaps,
    )
    return scanner.scan(text)
