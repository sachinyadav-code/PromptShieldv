"""Comprehensive unit and integration tests for bounded Unicode normalization and safe output enforcement.

Covers all 18 required Stage 5.3 scenarios:
1. ASCII input remains unchanged after normalization.
2. Full-width credential prefixes are normalized and detected.
3. Compatibility ligatures and expansion boundaries do not leak fragments.
4. Combining-mark sequences and Hangul Jamo do not crash inspection.
5. Credentials containing listed invisible characters are detected after filtering.
6. Characters outside the explicit invisible list are not silently stripped.
7. Empty input and all-invisible input follow documented policy.
8. Input length at 25,000 characters and above the limit.
9. Normalized output at and above the 100,000-character limit (expansion bombs).
10. Normalization exceptions fail closed.
11. Candidate overflow follows existing fail-closed behavior.
12. Placeholder examples remain excluded only by exact matches; near-misses remain findings.
13. Case-insensitive Stage 5.1 prefixes remain functional under normalization.
14. A redactable finding replaces the entire response; no original prefix or suffix survives.
15. A critical finding blocks egress.
16. Scanner, normalizer, or service exceptions never return the original response.
17. No-finding responses preserve the original text exactly.
18. Existing schemas, audit behavior, and response contracts remain compatible.

All credentials and tokens in this test suite are synthetic test fixtures.
"""

from unittest.mock import patch
import pytest
import unicodedata

from app.output.normalizer import (
    INVISIBLE_CHARACTERS,
    MAX_INPUT_TEXT_LENGTH,
    MAX_NORMALIZED_TEXT_LENGTH,
    OutputNormalizationError,
    OutputNormalizedLengthExceededError,
    normalize_for_detection,
    strip_invisible_characters,
)
from app.output.redactor import OutputRedactor, RedactionError, redact_output, redact_whole_output
from app.output.rules import (
    DLP_RULE_AWS_ACCESS_KEY_ID,
    DLP_RULE_BEARER_TOKEN,
    DLP_RULE_DATABASE_URI_CREDENTIAL,
    DLP_RULE_OPENAI_API_KEY,
    DLP_RULE_PRIVATE_KEY_HEADER,
    OutputFinding,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
)
from app.output.scanner import OutputScanner, scan_output
from app.output.schemas import OutputScanResult, OutputScanStatus, RedactionResult
from app.output.service import enforce_output_security
from app.schemas import MockDownstreamResponse


# ===========================================================================
# 1. Normalizer Unit Tests (Requirements 1, 5, 6, 7, 8, 9, 10)
# ===========================================================================

class TestBoundedNormalizerUnit:
    """Unit tests for the bounded normalization module."""

    def test_ascii_input_remains_strictly_unchanged(self):
        """Requirement 1: ASCII input is identical before and after normalization."""
        sample_ascii = "Standard ASCII response with OpenAI Key: sk-abcdef1234567890abcdef1234567890."
        normalized = normalize_for_detection(sample_ascii)
        assert normalized == sample_ascii
        assert len(normalized) == len(sample_ascii)

    def test_invisible_character_set_contains_exactly_17_codepoints(self):
        """Verifies the closed set invariant: exactly 17 invisible code points."""
        assert len(INVISIBLE_CHARACTERS) == 17
        expected_codepoints = {
            "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u00ad",
            "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d",
            "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
        }
        assert set(INVISIBLE_CHARACTERS) == expected_codepoints

    def test_all_17_invisible_characters_are_stripped(self):
        """Requirement 5: Every single character in the 17-character set is stripped."""
        for inv_char in INVISIBLE_CHARACTERS:
            text = f"prefix{inv_char}middle{inv_char}suffix"
            stripped = strip_invisible_characters(text)
            assert stripped == "prefixmiddlesuffix", f"Failed to strip {ord(inv_char):#x}"
            normalized = normalize_for_detection(text)
            assert normalized == "prefixmiddlesuffix"

    def test_characters_outside_invisible_list_are_not_silently_stripped(self):
        """Requirement 6: Non-breaking space, ideographic space, and other format chars are preserved."""
        # Non-breaking space (\u00A0) normalizes to ASCII space under NFKC, NOT stripped
        text_nbsp = "hello\u00a0world"
        assert strip_invisible_characters(text_nbsp) == text_nbsp
        assert normalize_for_detection(text_nbsp) == "hello world"

        # Ideographic space (\u3000) normalizes to ASCII space under NFKC, NOT stripped
        text_ideo = "hello\u3000world"
        assert strip_invisible_characters(text_ideo) == text_ideo
        assert normalize_for_detection(text_ideo) == "hello world"

        # Standard spaces and newlines strictly preserved
        whitespace = "hello \t\n\r world"
        assert strip_invisible_characters(whitespace) == whitespace
        assert normalize_for_detection(whitespace) == whitespace

    def test_empty_and_all_invisible_inputs(self):
        """Requirement 7: Empty string and strings containing only invisible characters."""
        # Empty string
        assert normalize_for_detection("") == ""

        # String composed entirely of invisible characters
        all_invis = "".join(INVISIBLE_CHARACTERS) * 5
        assert strip_invisible_characters(all_invis) == ""
        assert normalize_for_detection(all_invis) == ""

    def test_input_length_boundary_25000_characters(self):
        """Requirement 8: Exactly 25,000 characters succeeds; 25,001 characters raises OutputLengthExceededError."""
        valid_at_limit = "A" * MAX_INPUT_TEXT_LENGTH
        assert len(valid_at_limit) == 25000
        normalized = normalize_for_detection(valid_at_limit)
        assert len(normalized) == 25000

        oversized = "A" * (MAX_INPUT_TEXT_LENGTH + 1)
        assert len(oversized) == 25001
        with pytest.raises(OutputLengthExceededError) as exc_info:
            normalize_for_detection(oversized)
        assert exc_info.value.length == 25001
        assert exc_info.value.max_length == 25000

    def test_normalized_output_length_overflow_expansion_bomb(self):
        """Requirement 9: Normalized length exceeding 100,000 raises OutputNormalizedLengthExceededError."""
        # In Unicode 16.0.0, U+FDFA expands to 18 characters under NFKC
        # 10,000 * U+FDFA is 10,000 input characters (<= 25,000) but expands to 180,000 characters
        expansion_bomb = "\uFDFA" * 10000
        assert len(expansion_bomb) == 10000 <= MAX_INPUT_TEXT_LENGTH

        with pytest.raises(OutputNormalizedLengthExceededError) as exc_info:
            normalize_for_detection(expansion_bomb)
        assert exc_info.value.length == 180000
        assert exc_info.value.max_length == MAX_NORMALIZED_TEXT_LENGTH

    def test_normalization_failure_fails_closed(self):
        """Requirement 10: Underlying normalization exceptions propagate as OutputNormalizationError."""
        with patch("unicodedata.normalize", side_effect=ValueError("Simulated UCD corruption")):
            with pytest.raises(OutputNormalizationError) as exc_info:
                normalize_for_detection("test input")
            assert "Simulated UCD corruption" in str(exc_info.value)

    def test_invalid_input_type_raises_type_error(self):
        """Non-string inputs must raise TypeError."""
        for invalid in [None, 12345, ["list"], {"key": "val"}, b"bytes"]:
            with pytest.raises(TypeError):
                normalize_for_detection(invalid)


# ===========================================================================
# 2. Scanner Detection with Bounded Normalization (Requirements 2, 3, 4, 11, 12, 13)
# ===========================================================================

class TestScannerNormalizedDetection:
    """Tests for OutputScanner integrating bounded Unicode normalization."""

    def test_full_width_credential_prefixes_detected(self):
        """Requirement 2: Full-width compatibility prefixes (e.g. \uff53\uff4b\uff0d) are detected."""
        # Full-width 'sk-' followed by ASCII random body
        full_width_prefix = "\uff53\uff4b\uff0d"  # ｓｋ－
        body = "abcdef1234567890abcdef1234567890"
        text = f"API Token: {full_width_prefix}{body}"

        scan_res = scan_output(text)
        assert scan_res.has_findings is True
        assert scan_res.finding_count == 1
        finding = scan_res.findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY
        assert finding.category == "API_KEY"
        assert finding.severity == "HIGH"
        assert finding.is_normalized is True

    def test_full_width_bearer_scheme_detected(self):
        """Requirement 2: Full-width Bearer scheme is normalized and detected."""
        # Full-width 'Bearer '
        fw_bearer = "\uff22\uff45\uff41\uff52\uff45\uff52 "  # Ｂｅａｒｅｒ
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ90123456789"
        text = f"Authorization: {fw_bearer}{token}"

        scan_res = scan_output(text)
        assert scan_res.has_findings is True
        assert scan_res.finding_count == 1
        finding = scan_res.findings[0]
        assert finding.rule_id == DLP_RULE_BEARER_TOKEN
        assert finding.is_normalized is True

    def test_compatibility_ligatures_before_credential_do_not_leak_fragments(self):
        """Requirement 3: Ligature expansion preceding a credential shifts normalized offsets safely."""
        # Ligature 'ffi' (\uFB03) expands from 1 char to 3 chars ('ffi')
        ligature = "\uFB03"
        secret = "sk-abcdef1234567890abcdef1234567890"
        text = f"Document header {ligature} key: {secret}"

        scan_res = scan_output(text)
        assert scan_res.has_findings is True
        finding = scan_res.findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY
        # Because ligature expanded, offsets in normalized text diverged from original text
        assert finding.is_normalized is True

        # Attempting partial redaction on original text MUST fail safely
        with pytest.raises(RedactionError):
            redact_output(text, scan_res.findings)

        # Whole-response redaction safely replaces entire message
        red_res = redact_output(text, scan_res.findings, whole_response=True)
        assert red_res.redacted_text == "[REDACTED]"
        assert secret not in red_res.redacted_text
        assert ligature not in red_res.redacted_text

    def test_combining_marks_and_hangul_jamo_do_not_crash(self):
        """Requirement 4: 50 combining accents and Hangul Conjoining Jamo sequences execute safely."""
        # Base 'e' + 50 combining acute accents (\u0301)
        combining_seq = "e" + ("\u0301" * 50)
        # Conjoining Hangul Jamo: Choseong Giyeok (\u1100) + Jungseong A (\u1161) + Jongseong Giyeok (\u11A8) -> Gak (\uAC01)
        hangul_jamo = "\u1100\u1161\u11A8"
        secret = "sk-abcdef1234567890abcdef1234567890"
        text = f"Testing {combining_seq} and {hangul_jamo} preceding {secret}."

        scan_res = scan_output(text)
        assert scan_res.has_findings is True
        assert scan_res.finding_count == 1
        finding = scan_res.findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY

    def test_credentials_with_internal_invisible_characters_detected(self):
        """Requirement 5: Credential with zero-width characters internally is detected."""
        # Zero-width spaces, joiners, and soft hyphens inserted inside the key
        obfuscated_key = "s\u200bk-\u200cproj-\u200d1234567890abcdef\u00ad1234567890abcdef123456"
        text = f"Exported key: {obfuscated_key}"

        scan_res = scan_output(text)
        assert scan_res.has_findings is True
        assert scan_res.finding_count == 1
        finding = scan_res.findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY
        assert finding.is_normalized is True

    def test_candidate_overflow_follows_existing_fail_closed_behavior(self):
        """Requirement 11: Dense finding overflow raises OutputFindingLimitExceededError."""
        # Dense repeated synthetic AWS keys exceeding limit (100)
        dense_keys = "\n".join([f"AKIA{i:016d}" for i in range(150)])
        scanner = OutputScanner(max_findings=100)
        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scanner.scan(dense_keys)
        assert exc_info.value.max_count == 100

    def test_placeholder_allowlist_matches_exact_only_near_miss_remains_finding(self):
        """Requirement 12: Exact placeholder is excluded; near-miss is detected as a finding."""
        # Exact OpenAI documentation placeholder
        exact_placeholder = "sk-00000000000000000000000000000000"
        scan_exact = scan_output(f"Configure: {exact_placeholder}")
        assert scan_exact.is_clean is True
        assert scan_exact.has_findings is False

        # Near-miss placeholder (ends in '1' instead of all zeros)
        near_miss = "sk-00000000000000000000000000000001"
        scan_near = scan_output(f"Configure: {near_miss}")
        assert scan_near.has_findings is True
        assert scan_near.finding_count == 1
        assert scan_near.findings[0].rule_id == DLP_RULE_OPENAI_API_KEY

    def test_stage_51_case_insensitive_prefixes_functional_under_normalization(self):
        """Requirement 13: Uppercase and mixed-case prefixes work under normalization."""
        # Uppercase full-width prefix ＳＫ－
        fw_upper_sk = "\uff33\uff2b\uff0d" + "abcdef1234567890abcdef1234567890"
        scan_fw = scan_output(f"Key: {fw_upper_sk}")
        assert scan_fw.has_findings is True
        assert scan_fw.findings[0].rule_id == DLP_RULE_OPENAI_API_KEY

        # Mixed-case Bearer scheme with full-width characters
        fw_bearer = "\uff42\uff25\uff41\uff52\uff45\uff52 " + "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ90123456789"
        scan_b = scan_output(f"Header: {fw_bearer}")
        assert scan_b.has_findings is True
        assert scan_b.findings[0].rule_id == DLP_RULE_BEARER_TOKEN


# ===========================================================================
# 3. Safe Whole-Response Service Enforcement (Requirements 14, 15, 16, 17, 18)
# ===========================================================================

class TestSafeOutputEnforcementService:
    """Tests for enforce_output_security enforcing whole-response neutralization."""

    def test_redactable_finding_replaces_entire_response(self):
        """Requirement 14: Redactable finding replaces message entirely with [REDACTED]; no prefix/suffix survives."""
        prefix = "IMPORTANT INSTRUCTIONS: Before deployment, configure this secret: "
        secret = "sk-abcdef1234567890abcdef1234567890"
        suffix = ". Please save it to your local environment file immediately."
        orig_message = f"{prefix}{secret}{suffix}"

        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=orig_message,
            received_prompt_length=25,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        # Entire message MUST be exactly [REDACTED]
        assert res.downstream_response.message == "[REDACTED]"
        # Zero original prefix or suffix survives
        assert prefix not in res.downstream_response.message
        assert suffix not in res.downstream_response.message
        assert secret not in res.downstream_response.message
        assert res.finding_count == 1
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_critical_finding_blocks_egress(self):
        """Requirement 15: Critical finding (e.g. private key header) blocks egress completely."""
        pkey = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAzsyntheticKeyData\n-----END RSA PRIVATE KEY-----"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Server private key generated:\n{pkey}",
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert res.risk_score == 1.0
        assert DLP_RULE_PRIVATE_KEY_HEADER in res.matched_rule_ids

    def test_clean_response_preserves_original_text_exactly(self):
        """Requirement 17: No-finding response returns original downstream message byte-for-byte."""
        clean_text = "Here is a safe analysis of public cryptography algorithms: RSA, ECC, and Dilithium."
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=clean_text,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "clean"
        assert res.downstream_response is not None
        assert res.downstream_response.message == clean_text
        assert res.finding_count == 0
        assert res.matched_rule_ids == []

    def test_all_invisible_input_returns_clean_preserved(self):
        """Requirement 7: All-invisible input completes inspection and is preserved as clean."""
        invisible_text = "\u200B\u200C\u200D\u2060"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=invisible_text,
            received_prompt_length=4,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "clean"
        assert res.downstream_response is not None
        assert res.downstream_response.message == invisible_text

    def test_scanner_or_normalizer_exceptions_fail_closed_never_leak_original(self):
        """Requirement 16: Exceptions during inspection fail closed to BLOCK without data leak."""
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message="Sensitive secret text sk-abcdef1234567890abcdef1234567890 that should not leak.",
            received_prompt_length=10,
        )

        # 1. Normalization exception fails closed
        with patch("app.output.scanner.normalize_for_detection", side_effect=ValueError("Simulated normalizer fail")):
            res = enforce_output_security(mock_resp)
            assert res.effective_decision == "BLOCK"
            assert res.downstream_response is None
            assert res.audit_outcome == "scan_error"
            assert res.error_code == "OUTPUT_INSPECTION_FAILED"

        # 2. Length exceeded fails closed
        oversized_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message="A" * (MAX_INPUT_TEXT_LENGTH + 10),
            received_prompt_length=10,
        )
        res_len = enforce_output_security(oversized_resp)
        assert res_len.effective_decision == "BLOCK"
        assert res_len.downstream_response is None
        assert res_len.audit_outcome == "scan_error"
        assert res_len.error_code == "OUTPUT_LENGTH_EXCEEDED"

    def test_existing_schemas_and_contracts_remain_compatible(self):
        """Requirement 18: OutputEnforcementResult, OutputScanResult, and RedactionResult serialization contracts."""
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message="Key: sk-abcdef1234567890abcdef1234567890",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert isinstance(res.scan_result, OutputScanResult)
        assert isinstance(res.redaction_result, RedactionResult)

        # OutputScanResult serialization check
        scan_dict = res.scan_result.to_dict()
        assert "status" in scan_dict
        assert "has_findings" in scan_dict
        assert "finding_count" in scan_dict
        assert scan_dict["has_findings"] is True
        assert scan_dict["finding_count"] == 1

        # Assert serialized finding retains exact original contract keys
        finding_dict = scan_dict["findings"][0]
        expected_keys = {"rule_id", "category", "severity", "start", "end", "description"}
        assert set(finding_dict.keys()) == expected_keys
        assert "is_normalized" not in finding_dict

        # Direct OutputFinding.to_dict() serialization check
        finding_obj = res.scan_result.findings[0]
        assert set(finding_obj.to_dict().keys()) == expected_keys
        assert "is_normalized" not in finding_obj.to_dict()

        # Verify internal attribute is_normalized remains available for redaction guard
        assert hasattr(finding_obj, "is_normalized")

        # RedactionResult serialization check
        red_dict = res.redaction_result.to_dict()
        assert "redaction_count" in red_dict
        assert "was_modified" in red_dict
        assert "redacted_rule_ids" in red_dict
        assert red_dict["was_modified"] is True
        assert red_dict["redaction_count"] == 1
        assert DLP_RULE_OPENAI_API_KEY in red_dict["redacted_rule_ids"]

    def test_coincidental_substring_match_marks_all_findings_normalized(self):
        """Safety Regression: Even when a shifted finding coincidentally matches the slice
        in the original text, comparing normalized_text with original text ensures ALL
        findings are marked is_normalized=True without relying on slice heuristics.
        """
        secret = "sk-abcdef1234567890abcdef1234567890"
        # 36 ligatures \uFB00 ('ff', 1 char in source expanding to 2 chars in normalized text).
        # Shift is exactly 36 chars.
        # In source: prefix (36) + ' ' (1) + secret (35) + ' ' (1) + secret (35).
        # Secret 1 is at 37:72, Secret 2 is at 73:108.
        # In normalized text: prefix expands to 72 chars.
        # Secret 1 is shifted to 73:108 in normalized text, where text[73:108] coincidentally
        # equals Secret 2.
        prefix = "\uFB00" * 36
        text = prefix + " " + secret + " " + secret

        scan_res = scan_output(text, normalize=True)
        assert scan_res.finding_count == 2
        # Crucial invariant: ALL findings are marked normalized because normalization altered text
        assert all(f.is_normalized is True for f in scan_res.findings)
        assert scan_res.findings[0].is_normalized is True
        assert scan_res.findings[1].is_normalized is True

        # Attempting partial redaction on findings MUST be rejected
        with pytest.raises(RedactionError):
            redact_output(text, scan_res.findings)

        # Whole-response replacement safely neutralizes the entire response
        red_res = redact_output(text, scan_res.findings, whole_response=True)
        assert red_res.redacted_text == "[REDACTED]"
        assert secret not in red_res.redacted_text

    def test_expanding_compatibility_character_preceding_credential_marks_finding_normalized(self):
        """Verifies that an expanding compatibility character before a credential
        marks all findings normalized and forbids partial slicing.
        """
        secret = "sk-abcdef1234567890abcdef1234567890"
        # Trademark \u2122 (1 char) expands to 'TM' (2 chars) under NFKC
        text = f"Product\u2122 api key is {secret}"

        scan_res = scan_output(text, normalize=True)
        assert scan_res.finding_count == 1
        assert scan_res.findings[0].is_normalized is True
        assert all(f.is_normalized is True for f in scan_res.findings)

        with pytest.raises(RedactionError):
            redact_output(text, scan_res.findings)

        red_res = redact_output(text, scan_res.findings, whole_response=True)
        assert red_res.redacted_text == "[REDACTED]"
        assert secret not in red_res.redacted_text

    def test_identical_normalized_text_preserves_unnormalized_finding_behavior(self):
        """When normalized text is identical to original text, findings are NOT marked normalized."""
        secret = "sk-abcdef1234567890abcdef1234567890"
        clean_text = f"Here is the credential: {secret}"

        scan_res = scan_output(clean_text, normalize=True)
        assert scan_res.finding_count == 1
        finding = scan_res.findings[0]
        assert finding.is_normalized is False

        # Direct partial redaction is permitted for unaltered text
        red_res = redact_output(clean_text, scan_res.findings)
        assert red_res.redacted_text == "Here is the credential: [REDACTED]"
        assert secret not in red_res.redacted_text
