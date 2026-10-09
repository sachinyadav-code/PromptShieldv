"""Comprehensive unit tests for PromptShield Phase 4 Stage 3 Output Scanner, Redactor, and Schemas."""

import pytest

from app.output.redactor import (
    DEFAULT_REDACTION_MASK,
    OutputRedactor,
    RedactionError,
    RedactionMalformedFindingError,
    RedactionOverlapError,
    RedactionSpanError,
    redact_output,
)
from app.output.rules import (
    DLP_RULE_AWS_ACCESS_KEY_ID,
    DLP_RULE_BEARER_TOKEN,
    DLP_RULE_DATABASE_URI_CREDENTIAL,
    DLP_RULE_OPENAI_API_KEY,
    DLP_RULE_PRIVATE_KEY_HEADER,
    DLP_RULE_US_SSN,
    MAX_OUTPUT_TEXT_LENGTH,
    OutputFinding,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
)
from app.output.scanner import OutputScanner, scan_output
from app.output.schemas import OutputScanResult, OutputScanStatus, RedactionResult


# ===========================================================================
# 1. Scanner Tests (Requirements 1 - 8)
# ===========================================================================
class TestOutputScanner:
    """Unit tests for OutputScanner and scan_output function."""

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n\t  \r\n"])
    def test_empty_and_whitespace_input_produces_clean_result(self, empty_input: str):
        result = scan_output(empty_input)
        assert isinstance(result, OutputScanResult)
        assert result.is_clean is True
        assert result.has_findings is False
        assert result.finding_count == 0
        assert result.findings == []
        assert result.matched_rule_ids == []
        assert result.is_complete is True
        assert result.exceeded_safety_limit is False
        assert result.status == OutputScanStatus.CLEAN

    def test_ordinary_non_sensitive_text_produces_clean_result(self):
        text = "PromptShield provides deterministic and ML-based security gateways for enterprise LLMs."
        result = scan_output(text)
        assert result.is_clean is True
        assert result.has_findings is False
        assert result.finding_count == 0
        assert result.findings == []
        assert result.is_complete is True
        assert result.exceeded_safety_limit is False
        assert result.status == OutputScanStatus.CLEAN

    def test_synthetic_test_secret_produces_finding_without_secret_leakage(self):
        secret = "sk-abcdef1234567890abcdef1234567890"
        text = f"Your API key has been generated: {secret}"

        result = scan_output(text)
        assert result.is_clean is False
        assert result.has_findings is True
        assert result.finding_count == 1
        assert result.is_complete is True
        assert result.status == OutputScanStatus.FINDINGS_DETECTED
        assert result.matched_rule_ids == [DLP_RULE_OPENAI_API_KEY]

        # Verify privacy: raw secret must NOT appear in serialized or string representations
        serialized = result.to_dict()
        assert secret not in str(serialized)
        assert secret not in repr(result)
        assert secret not in result.findings[0].description

    def test_multiple_findings_produce_deterministic_metadata_and_counts(self):
        text = (
            "AWS Credential: AKIA1234567890ABCDEF\n"
            "SSN Record: 123-45-6789\n"
            "Key Header: -----BEGIN RSA PRIVATE KEY-----\n"
        )
        result = scan_output(text)
        assert result.is_clean is False
        assert result.has_findings is True
        assert result.finding_count == 3
        assert result.matched_rule_ids == sorted(
            [DLP_RULE_AWS_ACCESS_KEY_ID, DLP_RULE_US_SSN, DLP_RULE_PRIVATE_KEY_HEADER]
        )
        assert result.is_complete is True
        assert result.exceeded_safety_limit is False
        assert len(result.findings) == 3

        # Verify ordering is strictly by start offset ascending
        assert result.findings[0].start < result.findings[1].start < result.findings[2].start

    def test_input_length_overflow_propagates_typed_exception(self):
        oversized = "A" * (MAX_OUTPUT_TEXT_LENGTH + 1)
        scanner = OutputScanner()
        with pytest.raises(OutputLengthExceededError) as exc_info:
            scanner.scan(oversized)

        err_msg = str(exc_info.value)
        assert str(MAX_OUTPUT_TEXT_LENGTH) in err_msg
        assert str(len(oversized)) in err_msg
        assert "AAAAA" not in err_msg

    def test_finding_count_overflow_propagates_typed_exception(self):
        # 4 distinct SSN findings with max_findings=3
        text = "SSNs: 123-45-6789, 219-09-9999, 078-05-1120, 123-45-6780"
        scanner = OutputScanner(max_findings=3)
        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scanner.scan(text)

        err = exc_info.value
        assert err.count == 4
        assert err.max_count == 3

    def test_invalid_input_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Input text must be a string"):
            scan_output(12345)  # type: ignore

        with pytest.raises(TypeError, match="Input text must be a string"):
            scan_output(None)  # type: ignore

    def test_failed_scan_is_never_reported_as_clean_or_complete(self):
        # When an error occurs, scanner does NOT return an object with is_clean=True
        with pytest.raises(OutputLengthExceededError):
            scan_output("X" * (MAX_OUTPUT_TEXT_LENGTH + 5))

        # Direct model construction verify
        incomplete_result = OutputScanResult(
            has_findings=False,
            findings=[],
            finding_count=0,
            is_complete=False,
            exceeded_safety_limit=True,
            text_length=100,
        )
        assert incomplete_result.is_clean is False
        assert incomplete_result.status == OutputScanStatus.FAILED_OR_EXCEEDED


# ===========================================================================
# 2. Redactor Tests (Requirements 9 - 18)
# ===========================================================================
class TestOutputRedactor:
    """Unit tests for OutputRedactor and redact_output function."""

    def test_single_finding_is_replaced_with_redacted_mask(self):
        text = "Your active key: AKIA1234567890ABCDEF has been verified."
        scan_res = scan_output(text)
        assert scan_res.finding_count == 1

        redaction_res = redact_output(text, scan_res.findings)
        assert isinstance(redaction_res, RedactionResult)
        assert redaction_res.redaction_count == 1
        assert redaction_res.was_modified is True
        assert redaction_res.redacted_text == "Your active key: [REDACTED] has been verified."
        assert redaction_res.redacted_rule_ids == [DLP_RULE_AWS_ACCESS_KEY_ID]
        assert "AKIA1234567890ABCDEF" not in redaction_res.redacted_text

    def test_multiple_disjoint_findings_redacted_correctly(self):
        text = (
            "AWS: AKIA1234567890ABCDEF; "
            "SSN: 123-45-6789; "
            "Key: -----BEGIN RSA PRIVATE KEY-----"
        )
        scan_res = scan_output(text)
        assert scan_res.finding_count == 3

        redaction_res = redact_output(text, scan_res.findings)
        assert redaction_res.redaction_count == 3
        expected = "AWS: [REDACTED]; SSN: [REDACTED]; Key: [REDACTED]"
        assert redaction_res.redacted_text == expected
        assert len(redaction_res.redacted_rule_ids) == 3

    def test_surrounding_text_remains_strictly_unchanged(self):
        prefix = "Heading with special punctuation: @#$%^&*()\n"
        secret = "AKIA1234567890ABCDEF"
        middle = "\nMiddle paragraph with unicode: café, naïve, 🚀.\n"
        secret2 = "123-45-6789"
        suffix = "\nTrailing sentence."
        text = f"{prefix}{secret}{middle}{secret2}{suffix}"

        scan_res = scan_output(text)
        redaction_res = redact_output(text, scan_res.findings)

        expected = f"{prefix}[REDACTED]{middle}[REDACTED]{suffix}"
        assert redaction_res.redacted_text == expected

    def test_adjacent_spans_handled_correctly(self):
        text = "ABCDEF"
        # Two adjacent disjoint findings: [0, 3) and [3, 6)
        f1 = OutputFinding(
            rule_id="RULE_A",
            category="CUSTOM",
            severity="HIGH",
            start=0,
            end=3,
            description="First 3 chars",
        )
        f2 = OutputFinding(
            rule_id="RULE_B",
            category="CUSTOM",
            severity="HIGH",
            start=3,
            end=6,
            description="Next 3 chars",
        )

        redaction_res = redact_output(text, [f1, f2])
        assert redaction_res.redaction_count == 2
        assert redaction_res.redacted_text == "[REDACTED][REDACTED]"

    @pytest.mark.parametrize(
        ("start", "end", "err_type"),
        [
            (-1, 5, RedactionSpanError),  # Negative start
            (5, 5, RedactionSpanError),  # Zero-length span
            (10, 5, RedactionSpanError),  # Inverted span
            (0, 100, RedactionSpanError),  # End exceeds text length
        ],
    )
    def test_invalid_spans_are_rejected(self, start: int, end: int, err_type: type):
        text = "Short text"  # len = 10
        finding = {"start": start, "end": end, "rule_id": "RULE_TEST"}
        with pytest.raises(err_type):
            redact_output(text, [finding])

    def test_unexpected_overlapping_spans_are_rejected(self):
        text = "The quick brown fox jumps over the lazy dog."
        # Overlapping spans [10, 25) and [20, 35)
        f1 = OutputFinding(
            rule_id="RULE_1",
            category="TEST",
            severity="HIGH",
            start=10,
            end=25,
            description="Span 1",
        )
        f2 = OutputFinding(
            rule_id="RULE_2",
            category="TEST",
            severity="HIGH",
            start=20,
            end=35,
            description="Span 2",
        )

        with pytest.raises(RedactionOverlapError, match="Overlapping spans detected"):
            redact_output(text, [f1, f2])

    @pytest.mark.parametrize(
        "malformed_finding",
        [
            {"start": "not_an_int", "end": 10},
            {"start": 0, "end": "not_an_int"},
            {"no_start_key": 0},
            object(),
        ],
    )
    def test_malformed_findings_are_rejected(self, malformed_finding):
        text = "Sample text for testing."
        with pytest.raises(RedactionMalformedFindingError):
            redact_output(text, [malformed_finding])

    def test_original_text_and_finding_objects_remain_unmodified(self):
        orig_text = "API key: sk-abcdef1234567890abcdef1234567890"
        copy_text = str(orig_text)
        scan_res = scan_output(orig_text)

        orig_start = scan_res.findings[0].start
        orig_end = scan_res.findings[0].end

        redact_output(orig_text, scan_res.findings)

        # Confirm original text string and finding offsets are immutable and unchanged
        assert orig_text == copy_text
        assert scan_res.findings[0].start == orig_start
        assert scan_res.findings[0].end == orig_end

    def test_repeated_redaction_is_strictly_deterministic(self):
        text = (
            "AWS: AKIA1234567890ABCDEF; "
            "SSN: 123-45-6789; "
            "DB: postgres://admin:secret123@localhost:5432/app"
        )
        scan_res = scan_output(text)
        baseline = redact_output(text, scan_res.findings)

        for _ in range(10):
            repeated = redact_output(text, scan_res.findings)
            assert repeated.redacted_text == baseline.redacted_text
            assert repeated.redaction_count == baseline.redaction_count
            assert repeated.redacted_rule_ids == baseline.redacted_rule_ids

    def test_no_secret_fragment_appears_in_redacted_output(self):
        secret_key = "AKIA1234567890ABCDEF"
        secret_ssn = "123-45-6789"
        text = f"Records: {secret_key} and {secret_ssn}"

        scan_res = scan_output(text)
        redaction_res = redact_output(text, scan_res.findings)

        assert secret_key not in redaction_res.redacted_text
        assert secret_ssn not in redaction_res.redacted_text
        assert "AKIA" not in redaction_res.redacted_text
        assert "123-45" not in redaction_res.redacted_text

    def test_custom_mask_is_supported(self):
        text = "AWS key: AKIA1234567890ABCDEF"
        scan_res = scan_output(text)
        redaction_res = redact_output(text, scan_res.findings, mask="***MASKED***")
        assert redaction_res.redacted_text == "AWS key: ***MASKED***"

    def test_empty_findings_returns_unmodified_text(self):
        text = "Clean text with no secrets."
        redaction_res = redact_output(text, [])
        assert redaction_res.redacted_text == text
        assert redaction_res.redaction_count == 0
        assert redaction_res.was_modified is False
        assert redaction_res.original_length == len(text)
        assert redaction_res.redacted_length == len(text)


# ===========================================================================
# 3. Unicode and Boundary Tests (Stage 5.2)
# ===========================================================================
class TestOutputUnicodeAndBoundaryCases:
    """Regression tests for Unicode handling, detector offsets, and redaction boundaries."""

    def test_emoji_surrounding_detected_secret_preserves_offsets_and_redaction(self):
        """Verifies that multi-byte and astral-plane emoji characters before and after a secret
        do not cause offset drift or character corruption during scanning and redaction.
        """
        secret = "sk-abcdef1234567890abcdef1234567890"
        # Emojis including single-codepoint (🔑, 🚀) and multi-codepoint sequence (🛡️ with U+FE0F)
        prefix = "🔑 Secret credential 🛡️: "
        suffix = " 🚀 [Do not share! ✨]"
        text = f"{prefix}{secret}{suffix}"

        # 1. Verify Python string index expectations
        expected_start = len(prefix)
        expected_end = len(prefix) + len(secret)
        assert text[expected_start:expected_end] == secret

        # 2. Scan output and verify exact finding boundaries
        scan_res = scan_output(text)
        assert scan_res.has_findings is True
        assert scan_res.finding_count == 1
        finding = scan_res.findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY
        assert finding.start == expected_start
        assert finding.end == expected_end
        assert text[finding.start:finding.end] == secret

        # 3. Redact and verify precise preservation of emoji text
        redaction_res = redact_output(text, scan_res.findings)
        assert redaction_res.redaction_count == 1
        assert redaction_res.was_modified is True
        assert redaction_res.redacted_text == f"{prefix}[REDACTED]{suffix}"
        assert secret not in redaction_res.redacted_text
        assert prefix in redaction_res.redacted_text
        assert suffix in redaction_res.redacted_text

    def test_cjk_surrounding_detected_secret_preserves_offsets_and_redaction(self):
        """Verifies that non-Latin (Chinese/Japanese/Korean) multi-byte text preserves
        character code-point indexing, detector boundaries, and clean redaction.
        """
        secret = "sk-abcdef1234567890abcdef1234567890"
        cjk_prefix = "系统提示：您的访问密钥为："
        cjk_suffix = "，请务必妥善保存并定期轮换。"
        text = f"{cjk_prefix}{secret}{cjk_suffix}"

        expected_start = len(cjk_prefix)
        expected_end = len(cjk_prefix) + len(secret)
        assert text[expected_start:expected_end] == secret

        # Unnormalized mode (Phase 4 baseline): preserves character code-point indexing
        scan_unnorm = scan_output(text, normalize=False)
        assert scan_unnorm.has_findings is True
        assert scan_unnorm.finding_count == 1
        finding = scan_unnorm.findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY
        assert finding.start == expected_start
        assert finding.end == expected_end
        assert text[finding.start:finding.end] == secret

        redaction_res = redact_output(text, scan_unnorm.findings)
        assert redaction_res.redaction_count == 1
        assert redaction_res.redacted_text == f"{cjk_prefix}[REDACTED]{cjk_suffix}"
        assert secret not in redaction_res.redacted_text
        assert cjk_prefix in redaction_res.redacted_text
        assert cjk_suffix in redaction_res.redacted_text

        # Normalized mode: \\uFF1A and \\uFF0C are compatibility characters altered by NFKC.
        # Zero-offset architecture marks findings normalized and blocks partial slicing.
        scan_norm = scan_output(text, normalize=True)
        assert scan_norm.has_findings is True
        assert scan_norm.findings[0].is_normalized is True
        with pytest.raises(RedactionError):
            redact_output(text, scan_norm.findings)
        red_whole = redact_output(text, scan_norm.findings, whole_response=True)
        assert red_whole.redacted_text == "[REDACTED]"

    def test_non_breaking_spaces_near_secret_preserves_offsets_and_redaction(self):
        """Verifies that non-breaking spaces (\\u00A0) surrounding secrets and in headers
        match correctly without offset drift and are preserved in redacted output.
        """
        secret_key = "AKIA1234567890ABCDEF"
        bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ90123456789"

        # Test 1: Credential preceded and followed by non-breaking space (unnormalized mode)
        text_nbsp = f"AWS_KEY:\u00A0{secret_key}\u00A0(production)"
        scan_nbsp = scan_output(text_nbsp, normalize=False)
        assert scan_nbsp.has_findings is True
        assert scan_nbsp.finding_count == 1
        finding = scan_nbsp.findings[0]
        assert finding.rule_id == DLP_RULE_AWS_ACCESS_KEY_ID
        assert text_nbsp[finding.start:finding.end] == secret_key

        red_nbsp = redact_output(text_nbsp, scan_nbsp.findings)
        assert red_nbsp.redacted_text == "AWS_KEY:\u00A0[REDACTED]\u00A0(production)"
        assert "\u00A0" in red_nbsp.redacted_text

        # Under normalized mode: \\u00A0 is altered to regular space ' ', so finding is marked is_normalized=True
        scan_nbsp_norm = scan_output(text_nbsp, normalize=True)
        assert scan_nbsp_norm.findings[0].is_normalized is True
        with pytest.raises(RedactionError):
            redact_output(text_nbsp, scan_nbsp_norm.findings)
        red_whole = redact_output(text_nbsp, scan_nbsp_norm.findings, whole_response=True)
        assert red_whole.redacted_text == "[REDACTED]"

        # Test 2: Bearer scheme with non-breaking space (unnormalized direct match)
        text_bearer = f"Authorization:\u00A0Bearer\u00A0{bearer_token}"
        scan_bearer = scan_output(text_bearer, normalize=False)
        assert scan_bearer.has_findings is True
        assert scan_bearer.finding_count == 1
        b_finding = scan_bearer.findings[0]
        assert b_finding.rule_id == DLP_RULE_BEARER_TOKEN
        assert text_bearer[b_finding.start:b_finding.end] == f"Bearer\u00A0{bearer_token}"

        red_bearer = redact_output(text_bearer, scan_bearer.findings)
        assert red_bearer.redacted_text == "Authorization:\u00A0[REDACTED]"

    def test_zero_width_space_adjacent_and_internal_boundary_behavior(self):
        """Documents and verifies boundary behavior for zero-width spaces (\\u200B):
        - Direct unnormalized mode: Adjacent ZWSP allows regex boundary; internal ZWSP disrupts regex.
        - Stage 5.3 Bounded normalization mode: Strips ZWSP, detects secret, guards against partial slicing.
        """
        secret = "sk-abcdef1234567890abcdef1234567890"

        # Case A1: Unnormalized mode (Phase 4 baseline)
        text_adj = f"token:\u200B{secret}\u200Bend"
        scan_unnorm = scan_output(text_adj, normalize=False)
        assert scan_unnorm.has_findings is True
        assert scan_unnorm.finding_count == 1
        f_unnorm = scan_unnorm.findings[0]
        assert text_adj[f_unnorm.start:f_unnorm.end] == secret
        red_unnorm = redact_output(text_adj, scan_unnorm.findings)
        assert red_unnorm.redacted_text == "token:\u200B[REDACTED]\u200Bend"

        # Case A2: Stage 5.3 Bounded normalization mode
        # \\u200B is stripped during normalization, so offset diverges from original text
        scan_norm = scan_output(text_adj, normalize=True)
        assert scan_norm.has_findings is True
        assert scan_norm.finding_count == 1
        f_norm = scan_norm.findings[0]
        assert f_norm.is_normalized is True
        # Partial redaction on normalized finding is strictly blocked
        with pytest.raises(RedactionError):
            redact_output(text_adj, scan_norm.findings)
        # Whole-response redaction safely replaces entire response
        red_whole = redact_output(text_adj, scan_norm.findings, whole_response=True)
        assert red_whole.redacted_text == "[REDACTED]"

        # Case B1: Unnormalized mode - internal ZWSP disrupts regex (Phase 4 baseline)
        text_internal = "sk-abcdef1234\u200B567890abcdef1234567890"
        scan_internal_unnorm = scan_output(text_internal, normalize=False)
        assert scan_internal_unnorm.is_clean is True
        assert scan_internal_unnorm.has_findings is False

        # Case B2: Stage 5.3 Bounded normalization mode
        # \\u200B is stripped during normalization, credential is detected, partial slicing blocked
        scan_internal_norm = scan_output(text_internal, normalize=True)
        assert scan_internal_norm.has_findings is True
        assert scan_internal_norm.finding_count == 1
        f_internal = scan_internal_norm.findings[0]
        assert f_internal.rule_id == DLP_RULE_OPENAI_API_KEY
        assert f_internal.is_normalized is True
        with pytest.raises(RedactionError):
            redact_output(text_internal, scan_internal_norm.findings)
        red_internal_whole = redact_output(text_internal, scan_internal_norm.findings, whole_response=True)
        assert red_internal_whole.redacted_text == "[REDACTED]"

    def test_mixed_multilingual_and_emoji_with_multiple_secrets(self):
        """Verifies that multiple secrets embedded within mixed CJK, Arabic RTL, and emoji
        text are detected at correct offsets and redacted without offset skew.
        """
        key1 = "sk-abcdef1234567890abcdef1234567890"
        key2 = "AKIA1234567890ABCDEF"
        text = (
            f"🔐 [安全凭据] OpenAI密钥: {key1} (请勿泄露) 🚀\n"
            f"مفتاح AWS السحابي: {key2} 🛡️"
        )

        scan_res = scan_output(text)
        assert scan_res.finding_count == 2
        # Findings sorted strictly by start offset ascending
        assert scan_res.findings[0].start < scan_res.findings[1].start
        assert text[scan_res.findings[0].start:scan_res.findings[0].end] == key1
        assert text[scan_res.findings[1].start:scan_res.findings[1].end] == key2

        red_res = redact_output(text, scan_res.findings)
        assert red_res.redaction_count == 2
        assert key1 not in red_res.redacted_text
        assert key2 not in red_res.redacted_text
        assert "[安全凭据]" in red_res.redacted_text
        assert "مفتاح AWS السحابي:" in red_res.redacted_text
        assert "🔐" in red_res.redacted_text
        assert "🛡️" in red_res.redacted_text

    def test_case_insensitive_credential_prefix_detection_and_redaction(self):
        """Verifies end-to-end scanning and redaction of credentials with case-insensitive prefixes."""
        sk_key = "SK-abcdef1234567890abcdef1234567890"
        bearer_entry = "bearer a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        text = f"OpenAI Key: {sk_key}\nAuth: {bearer_entry}"

        scan_res = scan_output(text)
        assert scan_res.finding_count == 2
        assert scan_res.matched_rule_ids == sorted([DLP_RULE_OPENAI_API_KEY, DLP_RULE_BEARER_TOKEN])
        assert text[scan_res.findings[0].start : scan_res.findings[0].end] == sk_key
        assert text[scan_res.findings[1].start : scan_res.findings[1].end] == bearer_entry

        red_res = redact_output(text, scan_res.findings)
        assert red_res.redaction_count == 2
        assert sk_key not in red_res.redacted_text
        assert bearer_entry not in red_res.redacted_text
        assert red_res.redacted_text == "OpenAI Key: [REDACTED]\nAuth: [REDACTED]"
