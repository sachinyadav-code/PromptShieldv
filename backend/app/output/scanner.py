"""Output security scanner module for PromptShield Phase 4.

Provides a reusable scanner that evaluates plain-text model responses against
deterministic DLP rules and returns structured OutputScanResult models.
"""

import time
from typing import Optional

from app.output.decoder import (
    MAX_DECODED_TEXT_LENGTH,
    OutputDecodedLengthExceededError,
    OutputDecodingError,
    decode_for_detection,
)
from app.output.normalizer import (
    MAX_NORMALIZED_TEXT_LENGTH,
    normalize_for_detection,
)
from app.output.rules import (
    MAX_FINDINGS_COUNT,
    MAX_OUTPUT_TEXT_LENGTH,
    OutputFinding,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
    scan_output_text,
)
from app.output.schemas import OutputScanResult, OutputScanStatus


class OutputScanner:
    """Reusable plain-text output security scanner.

    Invokes bounded Unicode normalization, bounded credential decoding, and
    deterministic DLP rules, enforcing input length limits, finding count ceilings,
    and fail-closed exception propagation.
    """

    def __init__(
        self,
        max_length: int = MAX_OUTPUT_TEXT_LENGTH,
        max_findings: int = MAX_FINDINGS_COUNT,
        resolve_overlaps: bool = True,
        normalize: bool = True,
        max_normalized_length: int = MAX_NORMALIZED_TEXT_LENGTH,
        decode: bool = True,
        max_decoded_length: int = MAX_DECODED_TEXT_LENGTH,
    ) -> None:
        """Initialize OutputScanner with configurable boundaries.

        Args:
            max_length: Maximum allowed input text length in characters (default: 25,000).
            max_findings: Maximum allowed finding count before overflow (default: 100).
            resolve_overlaps: If True, resolves overlapping spans into strictly disjoint
                findings according to the strict priority order (default: True).
            normalize: If True, normalizes plain text (stripping 17 invisible chars and
                applying standard NFKC) into a bounded copy before detection (default: True).
            max_normalized_length: Maximum allowed normalized length (default: 100,000).
            decode: If True, decodes URL percent-encoded bytes and hex escape sequences
                into a bounded copy before detection (default: True).
            max_decoded_length: Maximum allowed decoded length (default: 100,000).
        """
        self.max_length = max_length
        self.max_findings = max_findings
        self.resolve_overlaps = resolve_overlaps
        self.normalize = normalize
        self.max_normalized_length = max_normalized_length
        self.decode = decode
        self.max_decoded_length = max_decoded_length

    def scan(self, text: str) -> OutputScanResult:
        """Scan plain-text output and return a structured OutputScanResult.

        IMPORTANT SAFETY INVARIANTS:
        - A scan failure or safety limit overflow is NEVER swallowed or returned as clean;
          it raises an explicit, safe typed exception.
        - Completed clean scans return `is_clean=True` and `has_findings=False`.
        - Completed scans with findings return `is_clean=False` and `has_findings=True`.
        - When normalization or decoding is active, DLP detection operates on a bounded
          inspection scratchpad. If transformations altered the input text, all findings
          are flagged with `is_normalized=True` to prevent unsafe partial slicing.

        Args:
            text: Plain-text string to scan.

        Returns:
            OutputScanResult containing findings metadata and technical metrics.

        Raises:
            TypeError: If text is not a string.
            OutputLengthExceededError: If text length exceeds max_length, normalized limit, or decoded limit.
            OutputFindingLimitExceededError: If finding count exceeds max_findings.
            OutputNormalizationError: If Unicode normalization fails.
            OutputDecodingError: If encoded candidate decoding fails.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")

        start_time = time.perf_counter()

        if self.normalize:
            # 1. Bounded normalization for detection scratchpad
            scratchpad = normalize_for_detection(
                text,
                max_input_length=self.max_length,
                max_normalized_length=self.max_normalized_length,
            )
            # 2. Bounded decoding for detection scratchpad (if enabled)
            if self.decode:
                decoded = decode_for_detection(
                    scratchpad,
                    max_input_length=self.max_normalized_length,
                    max_decoded_length=self.max_decoded_length,
                )
                if decoded != scratchpad:
                    scratchpad = normalize_for_detection(
                        decoded,
                        max_input_length=self.max_normalized_length,
                        max_normalized_length=self.max_normalized_length,
                    )
                else:
                    scratchpad = decoded

            is_altered = (scratchpad != text)

            # 3. Deterministic DLP regex scanning on inspection copy
            raw_findings = scan_output_text(
                scratchpad,
                max_length=self.max_normalized_length,
                resolve_overlaps=self.resolve_overlaps,
                max_findings=self.max_findings,
            )

            # 4. If inspection scratchpad differs from original, mark every finding as normalized.
            # Never attempt to infer offset correspondence through substring comparisons.
            if is_altered:
                findings = [
                    OutputFinding(
                        rule_id=f.rule_id,
                        category=f.category,
                        severity=f.severity,
                        start=f.start,
                        end=f.end,
                        description=f.description,
                        is_normalized=True,
                    )
                    for f in raw_findings
                ]
            else:
                findings = raw_findings
        else:
            # Unnormalized direct scanning path (with optional decoding if enabled)
            if self.decode:
                decoded = decode_for_detection(
                    text,
                    max_input_length=self.max_length,
                    max_decoded_length=self.max_decoded_length,
                )
                is_altered = (decoded != text)
                raw_findings = scan_output_text(
                    decoded,
                    max_length=self.max_decoded_length,
                    resolve_overlaps=self.resolve_overlaps,
                    max_findings=self.max_findings,
                )
                if is_altered:
                    findings = [
                        OutputFinding(
                            rule_id=f.rule_id,
                            category=f.category,
                            severity=f.severity,
                            start=f.start,
                            end=f.end,
                            description=f.description,
                            is_normalized=True,
                        )
                        for f in raw_findings
                    ]
                else:
                    findings = raw_findings
            else:
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
    normalize: bool = True,
    max_normalized_length: int = MAX_NORMALIZED_TEXT_LENGTH,
    decode: bool = True,
    max_decoded_length: int = MAX_DECODED_TEXT_LENGTH,
) -> OutputScanResult:
    """Convenience function to scan plain-text output using an ephemeral OutputScanner.

    Args:
        text: Plain-text string to scan.
        max_length: Maximum allowed text length in characters (default: 25,000).
        max_findings: Maximum allowed finding count (default: 100).
        resolve_overlaps: If True, resolves overlapping spans deterministically (default: True).
        normalize: If True, normalizes plain text before detection (default: True).
        max_normalized_length: Maximum allowed normalized length (default: 100,000).
        decode: If True, decodes URL percent-encoded bytes and hex escape sequences
            before detection (default: True).
        max_decoded_length: Maximum allowed decoded length (default: 100,000).

    Returns:
        Structured OutputScanResult.

    Raises:
        TypeError: If text is not a string.
        OutputLengthExceededError: If text exceeds max_length, normalized limit, or decoded limit.
        OutputFindingLimitExceededError: If finding count exceeds max_findings.
        OutputNormalizationError: If normalization fails.
        OutputDecodingError: If decoding fails.
    """
    scanner = OutputScanner(
        max_length=max_length,
        max_findings=max_findings,
        resolve_overlaps=resolve_overlaps,
        normalize=normalize,
        max_normalized_length=max_normalized_length,
        decode=decode,
        max_decoded_length=max_decoded_length,
    )
    return scanner.scan(text)
