"""Bounded Unicode normalization module for PromptShield Phase 5 Output Security.

Provides bounded normalization for detection scratchpads, enforcing input length
limits, removal of a closed 17-character invisible set, standard Python NFKC
normalization, and post-normalization length limits to protect downstream DLP regex
scanning from expansion bombs and ReDoS.

IMPORTANT ALLOCATION AND SAFETY REALITIES:
1. Python's standard library `unicodedata.normalize()` is implemented in C and does
   NOT expose pre-allocation size bounds or memory limits to Python callers. The C
   layer allocates its internal working buffers before returning the normalized `str`.
2. The pre-normalization input limit (25,000 characters) provides the coarse boundary
   limiting intermediate CPython allocation. In Unicode 16.0.0, maximum single-character
   NFKC expansion is 18 code points (e.g. U+FDFA), bounding worst-case intermediate
   allocations to ~450,000 characters (~1.8 MB).
3. The post-normalization check (100,000 characters) strictly bounds the output size
   admitted to downstream regex inspection, rejecting expansion bombs before rule matching.
4. This module does NOT construct or expose character-offset mappings back to the
   original text. Normalization is used exclusively for whole-response detection.
"""

from typing import FrozenSet
import unicodedata

from app.output.rules import (
    MAX_OUTPUT_TEXT_LENGTH,
    OutputLengthExceededError,
    OutputScanError,
)

# Upper bound on original plain-text input before normalization
MAX_INPUT_TEXT_LENGTH: int = MAX_OUTPUT_TEXT_LENGTH  # 25,000 characters

# Hard upper bound on normalized text size before downstream regex scanning
MAX_NORMALIZED_TEXT_LENGTH: int = 100_000  # 100,000 characters

# Exactly the 17 closed invisible and formatting characters evaluated for evasion
INVISIBLE_CHARACTERS: FrozenSet[str] = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BYTE ORDER MARK
        "\u2060",  # WORD JOINER
        "\u00ad",  # SOFT HYPHEN
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    }
)

# C-accelerated translation table for fast, allocation-efficient invisible stripping
_INVISIBLE_TRANSLATION_TABLE = {ord(ch): None for ch in INVISIBLE_CHARACTERS}


class OutputNormalizationError(OutputScanError):
    """Base exception for Unicode normalization failures."""
    pass


class OutputNormalizedLengthExceededError(OutputLengthExceededError, OutputNormalizationError):
    """Raised when normalized output text exceeds the maximum permitted limit (expansion bomb)."""

    def __init__(self, length: int, max_length: int) -> None:
        super().__init__(length=length, max_length=max_length)


def strip_invisible_characters(text: str) -> str:
    """Remove only characters in the closed 17-character INVISIBLE_CHARACTERS set.

    All other characters—including standard ASCII whitespace, non-breaking spaces,
    and regular multilingual text—are strictly preserved.

    Args:
        text: Plain-text string to filter.

    Returns:
        Filtered string with listed invisible formatting characters removed.
    """
    return text.translate(_INVISIBLE_TRANSLATION_TABLE)


def normalize_for_detection(
    text: str,
    max_input_length: int = MAX_INPUT_TEXT_LENGTH,
    max_normalized_length: int = MAX_NORMALIZED_TEXT_LENGTH,
) -> str:
    """Produce a bounded, normalized copy of plain text for output DLP detection.

    Steps:
    1. Validates input type (`isinstance(text, str)`).
    2. Enforces input length limit before normalization (`len(text) <= max_input_length`).
    3. Strips only the 17 listed invisible characters.
    4. Computes standard Python `unicodedata.normalize('NFKC', stripped)`.
    5. Enforces post-normalization output limit (`len(normalized) <= max_normalized_length`).

    IMPORTANT LIMITATION:
    The post-normalization check does NOT prevent `unicodedata.normalize()` itself
    from allocating an oversized intermediate result in C memory. Python does not
    expose pre-allocation bounds. The `max_input_length` check provides the coarse
    safeguard against unbounded allocations.

    Args:
        text: Plain-text string to normalize.
        max_input_length: Maximum allowed input length (default: 25,000).
        max_normalized_length: Maximum allowed normalized length (default: 100,000).

    Returns:
        Normalized string suitable for regex DLP scanning.

    Raises:
        TypeError: If text is not a string.
        OutputLengthExceededError: If input length exceeds max_input_length.
        OutputNormalizedLengthExceededError: If normalized length exceeds max_normalized_length.
        OutputNormalizationError: If unicodedata.normalize encounters an internal error.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")

    input_len = len(text)
    if input_len > max_input_length:
        raise OutputLengthExceededError(length=input_len, max_length=max_input_length)

    # Empty string is a valid edge case that normalizes directly to empty string
    if input_len == 0:
        return ""

    # Strip only the explicit 17 invisible characters
    stripped = strip_invisible_characters(text)

    # Perform standard CPython NFKC compatibility normalization
    try:
        normalized = unicodedata.normalize("NFKC", stripped)
    except Exception as exc:
        raise OutputNormalizationError(f"Unicode normalization failed: {exc}") from exc

    # Enforce hard output ceiling to protect downstream regex scanners from expansion bombs
    norm_len = len(normalized)
    if norm_len > max_normalized_length:
        raise OutputNormalizedLengthExceededError(
            length=norm_len, max_length=max_normalized_length
        )

    return normalized
