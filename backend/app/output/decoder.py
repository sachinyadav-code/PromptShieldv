r"""Bounded encoded credential decoding module for PromptShield Phase 5 Output Security.

Provides single-pass bounded decoding for detection scratchpads, supporting:
1. URL percent encoding (e.g. %2D, %3A, %2F)
2. Textual hexadecimal escape sequences (e.g. \x73\x6b)
3. Textual Unicode escape sequences (e.g. \u0073\u006b, \U00000073\U0000006b)

Enforces strict input length bounds, decoded buffer bounds, and fail-closed handling
for malformed encoding sequences (such as invalid UTF-8 byte sequences, embedded null bytes,
or invalid Unicode code points like surrogates and out-of-range scalars),
while preserving ordinary text containing literal percent signs, incomplete markers,
and unrelated backslashes.

IMPORTANT SAFETY INVARIANTS:
1. Decoding is performed in a single bounded pass. Double-encoded inputs (e.g. %252D,
   \x252D, \u005c\u00750073) are NOT recursively decoded.
2. Decoded text is NEVER executed or treated as executable code.
3. Decoded buffer sizes are strictly bounded by MAX_DECODED_TEXT_LENGTH.
4. Benign text containing literal %, incomplete markers (%2, %zz), and unrelated
   backslashes (\n, \t, \path, \u, \u1, \U, \U0073) are preserved without causing
   scan errors or blocking.
5. Candidates selected for decoding (%HH, \xHH, \uXXXX, \UXXXXXXXX) that cannot be safely
   processed as valid UTF-8 or valid Unicode code points raise OutputDecodingError,
   failing closed at the service boundary.
"""

import re
from typing import Optional

from app.output.rules import (
    MAX_OUTPUT_TEXT_LENGTH,
    OutputLengthExceededError,
    OutputScanError,
)

# Maximum input plain-text length before decoding (aligned with MAX_OUTPUT_TEXT_LENGTH)
MAX_INPUT_TEXT_LENGTH: int = MAX_OUTPUT_TEXT_LENGTH  # 25,000 characters

# Maximum allowed decoded buffer length
MAX_DECODED_TEXT_LENGTH: int = 100_000  # 100,000 characters

# Regex matching contiguous blocks of URL percent escapes, hex escapes, and Unicode escapes
# Matches contiguous sequences of %HH, \xHH, \XHH, \uXXXX, or \UXXXXXXXX
_PATTERN_ESCAPE_BLOCK = re.compile(
    r"(?:%[0-9a-fA-F]{2}|\\[xX][0-9a-fA-F]{2}|\\[u][0-9a-fA-F]{4}|\\[U][0-9a-fA-F]{8})+"
)


class OutputDecodingError(OutputScanError):
    """Base exception for encoded credential decoding failures."""
    pass


class OutputDecodedLengthExceededError(OutputLengthExceededError, OutputDecodingError):
    """Raised when decoded output text exceeds the maximum permitted buffer limit."""

    def __init__(self, length: int, max_length: int) -> None:
        super().__init__(length=length, max_length=max_length)


def decode_for_detection(
    text: str,
    max_input_length: int = MAX_INPUT_TEXT_LENGTH,
    max_decoded_length: int = MAX_DECODED_TEXT_LENGTH,
) -> str:
    r"""Produce a single bounded decoded copy of plain text for output DLP detection.

    Decoding rules:
    - Decodes URL percent-encoded bytes (`%HH`), textual hex escapes (`\xHH`),
      and textual Unicode escapes (`\uXXXX`, `\UXXXXXXXX`).
    - Performs a single bounded pass; double-encoded inputs (e.g. `%252D`, `\x252D`,
      `\u005c\u00750073`) are NOT recursively decoded.
    - Never executes decoded text or treats it as code.
    - Preserves literal `%`, incomplete markers (`%`, `%1`, `%zz`), and unrelated
      backslashes (`\n`, `\t`, `\path`, `\x`, `\u`, `\u1`, `\U`, `\U0073`).
    - Rejects malformed candidate escape sequences (invalid UTF-8 bytes, truncated
      multibyte sequences, embedded null bytes, surrogate code points, out-of-range scalars)
      by raising OutputDecodingError.
    - Enforces input-length and decoded-buffer boundaries.

    Args:
        text: Plain-text string to decode.
        max_input_length: Maximum allowed input length (default: 25,000).
        max_decoded_length: Maximum allowed decoded length (default: 100,000).

    Returns:
        Decoded plain-text string suitable for DLP regex scanning.

    Raises:
        TypeError: If text is not a string.
        OutputLengthExceededError: If input length exceeds max_input_length.
        OutputDecodedLengthExceededError: If decoded length exceeds max_decoded_length.
        OutputDecodingError: If an escape candidate cannot be decoded safely as valid UTF-8
            or contains forbidden null bytes / invalid code points.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")

    input_len = len(text)
    if input_len > max_input_length:
        raise OutputLengthExceededError(length=input_len, max_length=max_input_length)

    if input_len == 0:
        return ""

    # Fast-path check: if none of '%', '\\x', '\\X', '\\u', '\\U' are present, no decoding is needed
    if (
        "%" not in text
        and r"\x" not in text
        and r"\X" not in text
        and r"\u" not in text
        and r"\U" not in text
    ):
        return text

    def _replace_escape_block(match: re.Match) -> str:
        block = match.group(0)
        raw_bytes = bytearray()
        result_chars: list[str] = []
        i = 0
        block_len = len(block)

        def _flush_byte_run() -> None:
            if not raw_bytes:
                return
            try:
                decoded_str = raw_bytes.decode("utf-8")
                result_chars.append(decoded_str)
            except UnicodeDecodeError as exc:
                raise OutputDecodingError(
                    f"Malformed encoded escape candidate cannot be decoded as valid UTF-8: {exc}"
                ) from exc
            finally:
                raw_bytes.clear()

        while i < block_len:
            if block[i] == "%":
                byte_val = int(block[i + 1 : i + 3], 16)
                if byte_val == 0:
                    raise OutputDecodingError(
                        "Null byte (%00) in percent-encoded candidate is not permitted."
                    )
                raw_bytes.append(byte_val)
                i += 3
            elif block[i : i + 2] in ("\\x", "\\X"):
                byte_val = int(block[i + 2 : i + 4], 16)
                if byte_val == 0:
                    raise OutputDecodingError(
                        "Null byte (\\x00) in hex escape candidate is not permitted."
                    )
                raw_bytes.append(byte_val)
                i += 4
            elif block[i : i + 2] == "\\u":
                _flush_byte_run()
                code_point = int(block[i + 2 : i + 6], 16)
                if code_point == 0:
                    raise OutputDecodingError(
                        "Null code point (\\u0000) in Unicode escape candidate is not permitted."
                    )
                if 0xD800 <= code_point <= 0xDFFF:
                    raise OutputDecodingError(
                        f"Surrogate code point (U+{code_point:04X}) in Unicode escape candidate is not permitted."
                    )
                if code_point > 0x10FFFF:
                    raise OutputDecodingError(
                        f"Code point out of range (U+{code_point:X} > U+10FFFF) in Unicode escape candidate is not permitted."
                    )
                result_chars.append(chr(code_point))
                i += 6
            elif block[i : i + 2] == "\\U":
                _flush_byte_run()
                code_point = int(block[i + 2 : i + 10], 16)
                if code_point == 0:
                    raise OutputDecodingError(
                        "Null code point (\\U00000000) in Unicode escape candidate is not permitted."
                    )
                if 0xD800 <= code_point <= 0xDFFF:
                    raise OutputDecodingError(
                        f"Surrogate code point (U+{code_point:08X}) in Unicode escape candidate is not permitted."
                    )
                if code_point > 0x10FFFF:
                    raise OutputDecodingError(
                        f"Code point out of range (U+{code_point:X} > U+10FFFF) in Unicode escape candidate is not permitted."
                    )
                result_chars.append(chr(code_point))
                i += 10
            else:
                raise OutputDecodingError(
                    f"Unexpected token in escape block: {block[i:]}"
                )

        _flush_byte_run()
        return "".join(result_chars)

    try:
        decoded = _PATTERN_ESCAPE_BLOCK.sub(_replace_escape_block, text)
    except OutputDecodingError:
        raise
    except Exception as exc:
        raise OutputDecodingError(f"Unexpected decoding failure: {exc}") from exc

    decoded_len = len(decoded)
    if decoded_len > max_decoded_length:
        raise OutputDecodedLengthExceededError(
            length=decoded_len, max_length=max_decoded_length
        )

    return decoded
