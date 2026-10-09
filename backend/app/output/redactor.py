"""Deterministic output redaction module for PromptShield Phase 4.

Redacts sensitive character spans using fixed, non-secret masks (e.g. '[REDACTED]'),
preserving all surrounding text byte-for-byte.
"""

from typing import Any, List, Optional, Sequence

from app.output.rules import OutputFinding, OutputScanError
from app.output.schemas import RedactionResult

DEFAULT_REDACTION_MASK = "[REDACTED]"


class RedactionError(OutputScanError, ValueError):
    """Base exception for output redaction errors."""
    pass


class RedactionSpanError(RedactionError):
    """Raised when finding span offsets are invalid (out of bounds, negative, or inverted)."""
    pass


class RedactionOverlapError(RedactionError):
    """Raised when overlapping spans are provided to the redactor."""
    pass


class RedactionMalformedFindingError(RedactionError):
    """Raised when a finding object is missing required span/start/end attributes or has invalid types."""
    pass


class OutputRedactor:
    """Deterministic plain-text redactor using validated character spans."""

    def __init__(self, mask: str = DEFAULT_REDACTION_MASK) -> None:
        """Initialize redactor with a fixed non-secret replacement mask.

        Args:
            mask: Fixed replacement string for sensitive spans (default: '[REDACTED]').
        """
        if not isinstance(mask, str):
            raise TypeError(f"Redaction mask must be a string, got {type(mask).__name__}")
        self.mask = mask

    def redact(self, text: str, findings: Sequence[Any]) -> RedactionResult:
        """Redact sensitive spans in text and return a structured RedactionResult.

        IMPORTANT SAFETY & PRIVACY INVARIANTS:
        - Replaces sensitive spans with the fixed mask. Zero secret fragments are exposed.
        - Text outside the selected spans is preserved character-for-character.
        - Spans are strictly validated against text length: 0 <= start < end <= len(text).
        - If overlapping spans are supplied unexpectedly, raises RedactionOverlapError.
        - Neither the input text nor the input finding objects are mutated.

        Args:
            text: Original plain-text string.
            findings: Sequence of findings or objects with `start` and `end` integer attributes.

        Returns:
            RedactionResult containing the sanitized text and redaction metadata.

        Raises:
            TypeError: If text is not a string.
            RedactionMalformedFindingError: If a finding is malformed or missing integer offsets.
            RedactionSpanError: If finding spans are out of bounds or inverted.
            RedactionOverlapError: If findings contain overlapping spans.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")

        orig_len = len(text)

        if not findings:
            return RedactionResult(
                redacted_text=text,
                redaction_count=0,
                redacted_rule_ids=[],
                original_length=orig_len,
                redacted_length=orig_len,
            )

        # 1. Validate and extract span offsets without mutating original findings
        validated_items: List[tuple[int, int, str]] = []
        for idx, f in enumerate(findings):
            # Extract start and end offsets
            if hasattr(f, "start") and hasattr(f, "end"):
                start, end = f.start, f.end
            elif isinstance(f, dict) and "start" in f and "end" in f:
                start, end = f["start"], f["end"]
            else:
                raise RedactionMalformedFindingError(
                    f"Finding at index {idx} is missing required 'start' and 'end' offsets."
                )

            # Validate types
            if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
                raise RedactionMalformedFindingError(
                    f"Finding at index {idx} has non-integer offsets: start={type(start).__name__}, end={type(end).__name__}."
                )

            # Validate bounds
            if start < 0:
                raise RedactionSpanError(
                    f"Finding at index {idx} has negative start offset ({start})."
                )
            if end <= start:
                raise RedactionSpanError(
                    f"Finding at index {idx} has invalid span: end ({end}) must be strictly greater than start ({start})."
                )
            if end > orig_len:
                raise RedactionSpanError(
                    f"Finding at index {idx} span end ({end}) exceeds text length ({orig_len})."
                )

            rule_id = getattr(f, "rule_id", "") or (f.get("rule_id", "") if isinstance(f, dict) else "")
            validated_items.append((start, end, str(rule_id)))

        # 2. Sort by start offset ascending (and end offset ascending)
        validated_items.sort(key=lambda item: (item[0], item[1]))

        # 3. Validate strict non-overlapping invariant
        for i in range(len(validated_items) - 1):
            curr_start, curr_end, _ = validated_items[i]
            next_start, next_end, _ = validated_items[i + 1]
            if curr_end > next_start:
                raise RedactionOverlapError(
                    f"Overlapping spans detected: ({curr_start}, {curr_end}) and ({next_start}, {next_end}). "
                    "Redaction requires strictly disjoint spans."
                )

        # 4. Construct sanitized string using forward slice reconstruction
        pieces: List[str] = []
        last_idx = 0
        redacted_rule_ids: List[str] = []

        for start, end, rule_id in validated_items:
            # Preserve all characters preceding the span
            pieces.append(text[last_idx:start])
            # Inject fixed replacement mask
            pieces.append(self.mask)
            last_idx = end
            if rule_id:
                redacted_rule_ids.append(rule_id)

        # Append remaining characters after the final span
        pieces.append(text[last_idx:])
        redacted_text = "".join(pieces)

        return RedactionResult(
            redacted_text=redacted_text,
            redaction_count=len(validated_items),
            redacted_rule_ids=redacted_rule_ids,
            original_length=orig_len,
            redacted_length=len(redacted_text),
        )


def redact_output(
    text: str,
    findings: Sequence[Any],
    mask: str = DEFAULT_REDACTION_MASK,
) -> RedactionResult:
    """Convenience function to redact sensitive spans using an ephemeral OutputRedactor.

    Args:
        text: Original plain-text string.
        findings: Sequence of OutputFinding items or objects with integer start and end offsets.
        mask: Fixed replacement mask (default: '[REDACTED]').

    Returns:
        Structured RedactionResult.
    """
    redactor = OutputRedactor(mask=mask)
    return redactor.redact(text, findings)
