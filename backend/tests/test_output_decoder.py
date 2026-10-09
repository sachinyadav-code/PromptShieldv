"""Tests for PromptShield Phase 5 Stage 5.4: Bounded Encoded Credential Scanning.

Verifies detection of credentials that evade detection via URL percent encoding
and hexadecimal escape sequences, enforcing single-pass bounded decoding,
zero-offset whole-response neutralization, fail-closed handling for malformed candidates,
and preservation of benign percent/backslash text.
"""

import json
import pytest

from app.audit.logger import (
    clear_audit_events,
    get_audit_events,
    log_output_security_event,
)
from app.schemas import MockDownstreamResponse
from app.output.decoder import (
    MAX_DECODED_TEXT_LENGTH,
    MAX_INPUT_TEXT_LENGTH,
    OutputDecodedLengthExceededError,
    OutputDecodingError,
    decode_for_detection,
)
from app.output.normalizer import OutputNormalizedLengthExceededError
from app.output.redactor import RedactionError, redact_output
from app.output.rules import (
    DLP_RULE_AWS_ACCESS_KEY_ID,
    DLP_RULE_BEARER_TOKEN,
    DLP_RULE_DATABASE_URI_CREDENTIAL,
    DLP_RULE_OPENAI_API_KEY,
    DLP_RULE_PRIVATE_KEY_HEADER,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
)
from app.output.scanner import OutputScanner, scan_output
from app.output.schemas import OutputScanResult, RedactionResult
from app.output.service import enforce_output_security


class TestBoundedDecoderUnit:
    """Unit tests for the decode_for_detection module."""

    def test_empty_string_returns_empty_string(self):
        assert decode_for_detection("") == ""

    def test_clean_ascii_without_markers_returns_same_string(self):
        text = "Ordinary prose without any escape characters."
        assert decode_for_detection(text) == text

    def test_percent_decoding_standard_ascii(self):
        raw = "sk%2Dabcdef1234567890abcdef1234567890"
        expected = "sk-abcdef1234567890abcdef1234567890"
        assert decode_for_detection(raw) == expected

    def test_percent_decoding_database_uri(self):
        raw = "postgres%3A%2F%2Fadmin%3Asecret123%40localhost%3A5432%2Fdb"
        expected = "postgres://admin:secret123@localhost:5432/db"
        assert decode_for_detection(raw) == expected

    def test_hexadecimal_escape_decoding_standard_ascii(self):
        raw = r"\x73\x6b-abcdef1234567890abcdef1234567890"
        expected = "sk-abcdef1234567890abcdef1234567890"
        assert decode_for_detection(raw) == expected

    def test_hexadecimal_escape_uppercase_x(self):
        raw = r"\X73\X6b-abcdef1234567890abcdef1234567890"
        expected = "sk-abcdef1234567890abcdef1234567890"
        assert decode_for_detection(raw) == expected

    def test_mixed_percent_and_hex_contiguous_escapes(self):
        raw = r"\x73%6b-abcdef1234567890abcdef1234567890"
        expected = "sk-abcdef1234567890abcdef1234567890"
        assert decode_for_detection(raw) == expected

    def test_multibyte_utf8_percent_and_hex_decoding(self):
        # UTF-8 for 'é' is 0xC3 0xA9
        assert decode_for_detection("%C3%A9") == "é"
        assert decode_for_detection(r"\xc3\xa9") == "é"
        assert decode_for_detection(r"%C3\xa9") == "é"

    def test_double_encoded_content_not_recursively_decoded(self):
        # %25 is '%'. In a single pass, %252D becomes %2D, NOT '-'
        assert decode_for_detection("%252D") == "%2D"
        assert decode_for_detection(r"\x252D") == "%2D"
        # %5C is '\'. %5Cx73 becomes \x73, NOT 's'
        assert decode_for_detection(r"%5Cx73") == r"\x73"

    def test_benign_prose_containing_percent_and_backslashes_preserved(self):
        benign = r"100% of employees love the 50% discount. Path C:\x\foo and \n \t \r %2 %zz % \x \x1 \xzz"
        assert decode_for_detection(benign) == benign

    def test_malformed_percent_candidate_fails_closed(self):
        # %FF is not a valid UTF-8 start byte
        with pytest.raises(OutputDecodingError):
            decode_for_detection("Key: %FF_invalid")

    def test_malformed_hex_candidate_fails_closed(self):
        # \xff is not a valid UTF-8 start byte
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Key: \xff_invalid")

    def test_truncated_multibyte_escape_fails_closed(self):
        # %C3 requires a continuation byte
        with pytest.raises(OutputDecodingError):
            decode_for_detection("Key: %C3_truncated")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Key: \xc3_truncated")

    def test_null_byte_in_percent_or_hex_fails_closed(self):
        with pytest.raises(OutputDecodingError):
            decode_for_detection("Secret: %00token")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Secret: \x00token")

    def test_max_input_length_exceeded_raises_error(self):
        oversized = "A" * (MAX_INPUT_TEXT_LENGTH + 1)
        with pytest.raises(OutputLengthExceededError):
            decode_for_detection(oversized)

    def test_max_decoded_buffer_exceeded_raises_error(self):
        with pytest.raises(OutputDecodedLengthExceededError):
            decode_for_detection("sk%2Dkey", max_decoded_length=3)

    def test_non_string_input_raises_type_error(self):
        with pytest.raises(TypeError):
            decode_for_detection(12345)  # type: ignore


class TestEncodedCredentialScanningAndRedaction:
    """Integration tests for OutputScanner and enforce_output_security with encoded credentials."""

    def test_percent_encoded_openai_key_detected_and_whole_response_redacted(self):
        """Requirement 1 & 5: Percent-encoded OpenAI key prefix is detected and redacted whole."""
        secret = "sk%2Dabcdef1234567890abcdef1234567890"
        message = f"Here is your deployment key: {secret}"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=message,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert res.finding_count == 1
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids
        # Crucial safety check: zero secret fragments survive
        assert "sk" not in res.downstream_response.message
        assert "%2D" not in res.downstream_response.message
        assert "abcdef" not in res.downstream_response.message

    def test_percent_encoded_database_uri_blocks_egress(self):
        """Requirement 2 & 4: Percent-encoded database URI credentials block egress completely."""
        db_uri = "postgres%3A%2F%2Fadmin%3Asecret123%40localhost%3A5432%2Fdb"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Database connection string: {db_uri}",
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert res.risk_score == 1.0
        assert DLP_RULE_DATABASE_URI_CREDENTIAL in res.matched_rule_ids

    def test_textual_hex_escapes_openai_key_detected(self):
        """Requirement 3: Textual hex escapes reconstructing credential prefix are detected."""
        secret = r"\x73\x6b-abcdef1234567890abcdef1234567890"
        message = f"Token generated: {secret}"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=message,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response.message == "[REDACTED]"
        assert res.finding_count == 1
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_hex_escaped_critical_private_key_blocks_egress(self):
        """Requirement 4: Hex-escaped critical findings (private key header) block egress."""
        # Hex-escaped "BEGIN RSA PRIVATE KEY"
        header = r"-----\x42\x45\x47\x49\x4e RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAzsyntheticKeyData\n-----END RSA PRIVATE KEY-----"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Generated key:\n{header}",
            received_prompt_length=20,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert DLP_RULE_PRIVATE_KEY_HEADER in res.matched_rule_ids

    def test_benign_text_with_percent_and_backslashes_preserved(self):
        """Requirement 6: Benign text with literal percent signs and backslashes is allowed unchanged."""
        clean_msg = r"Progress: 100% complete. Results stored in C:\users\docs\report.txt with 50% discount on items (%2 and %zz)."
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=clean_msg,
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "clean"
        assert res.downstream_response is not None
        assert res.downstream_response.message == clean_msg
        assert res.finding_count == 0

    def test_mixed_encoded_and_ordinary_content(self):
        """Requirement 7: Mixed encoded and ordinary prose detects credential safely."""
        secret = r"\x73\x6b%2Dproj%2D1234567890abcdef1234567890abcdef123456"
        message = f"100% confidence score. File saved to C:\\exports\\key.txt: {secret}."
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=message,
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response.message == "[REDACTED]"
        assert res.finding_count == 1
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_double_encoded_content_not_recursively_decoded_into_finding(self):
        """Requirement 8: Double-encoded input (%252D) is not recursively decoded into a credential."""
        # %252D decodes in single pass to %2D, which does NOT match the '-' required by the regex
        double_encoded = "sk%252Dabcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Safe double encoded token: {double_encoded}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        # Because it wasn't recursively decoded to 'sk-', it remains clean!
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "clean"
        assert res.downstream_response.message == f"Safe double encoded token: {double_encoded}"

    def test_maximum_input_and_decoded_buffer_boundaries_fail_closed(self):
        """Requirement 9: Oversized inputs fail closed at service boundary."""
        oversized = "A" * (MAX_INPUT_TEXT_LENGTH + 10)
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=oversized,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.downstream_response is None
        assert res.audit_outcome == "scan_error"
        assert res.error_code == "OUTPUT_LENGTH_EXCEEDED"

    def test_finding_count_and_scan_limit_enforcement(self):
        """Requirement 10: Finding overflow in decoded representation raises limit error and fails closed."""
        # 150 encoded AWS keys (%41%4B%49%41 -> AKIA)
        dense_encoded = "\n".join([f"%41%4B%49%41{i:016d}" for i in range(150)])
        scanner = OutputScanner(max_findings=100)
        with pytest.raises(OutputFindingLimitExceededError):
            scanner.scan(dense_encoded)

        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=dense_encoded,
            received_prompt_length=10,
        )
        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.downstream_response is None
        assert res.audit_outcome == "scan_error"
        assert res.error_code == "FINDING_LIMIT_EXCEEDED"

    def test_decoding_exceptions_fail_closed_safely(self):
        """Requirement 11: Exceptions during decoding fail closed without leaking original text."""
        malformed_message = "Dangerous text with malformed escape: %FF and sk-abcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=malformed_message,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.downstream_response is None
        assert res.audit_outcome == "scan_error"
        assert res.error_code == "OUTPUT_SCAN_ERROR"

    def test_audit_logs_contain_neither_original_secrets_nor_decoded_text(self):
        """Requirement 12: Audit logs omit raw secret strings and decoded intermediate text."""
        clear_audit_events()
        secret = "sk%2Dabcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Key: {secret}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        log_output_security_event(
            request_id="req-audit-test",
            endpoint="/api/v1/proxy",
            outcome=res.audit_outcome,
            matched_rule_ids=res.matched_rule_ids,
            finding_count=res.finding_count,
            client_request_id="client-test-42",
        )

        events = get_audit_events()
        assert len(events) > 0
        last_event = events[-1]
        serialized = json.dumps(last_event)

        # Invariant: Neither the encoded key nor the decoded key appears anywhere in audit log
        assert "sk%2D" not in serialized
        assert "sk-abcdef1234567890abcdef1234567890" not in serialized
        assert "abcdef1234567890" not in serialized
        clear_audit_events()

    def test_unicode_normalization_and_encoding_combination(self):
        """Requirement 13: Combination of Unicode full-width and percent encoding is detected."""
        # Full-width 'ｓｋ' (\uff53\uff4b) + percent encoded '-' (%2D)
        combo = "\uff53\uff4b%2Dabcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Secret: {combo}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response.message == "[REDACTED]"
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_partial_redaction_guard_blocks_slicing_on_decoded_findings(self):
        """Direct partial redaction call on decoded finding raises RedactionError."""
        secret = "sk%2Dabcdef1234567890abcdef1234567890"
        scan_res = scan_output(secret)
        assert scan_res.has_findings is True
        assert scan_res.findings[0].is_normalized is True

        with pytest.raises(RedactionError):
            redact_output(secret, scan_res.findings)

    def test_nfkc_expanded_scratchpad_exceeding_25k_processed_without_false_length_error(self):
        """Short input whose NFKC form exceeds 25,000 chars (within 100,000) does not fail length check."""
        # U+FDFA expands to 18 code points under NFKC: 1500 * 18 = 27,000 characters
        text = "\uFDFA" * 1500
        assert len(text) == 1500
        scan_res = scan_output(text)
        assert scan_res.has_findings is False
        assert scan_res.is_complete is True
        assert scan_res.text_length == 1500

    def test_nfkc_expanded_scratchpad_with_encoded_credential_detected_and_redacted(self):
        """Short input expanding to >25,000 chars containing encoded credential is detected and redacted."""
        # 1500 * U+FDFA (27,000 chars NFKC) + space + percent-encoded OpenAI key
        text = "\uFDFA" * 1500 + " sk%2Dabcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=text,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert res.finding_count == 1
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_nfkc_expanded_scratchpad_with_encoded_critical_credential_blocks_egress(self):
        """Short input expanding to >25,000 chars containing encoded critical secret blocks egress."""
        text = "\uFDFA" * 1500 + " postgres%3A%2F%2Fadmin%3Asecret123%40localhost%3A5432%2Fdb"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=text,
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert DLP_RULE_DATABASE_URI_CREDENTIAL in res.matched_rule_ids

    def test_original_input_exceeding_25000_fails_closed(self):
        """Input exceeding MAX_OUTPUT_TEXT_LENGTH (25,000) fails closed with OUTPUT_LENGTH_EXCEEDED."""
        oversized = "A" * (MAX_INPUT_TEXT_LENGTH + 1)
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=oversized,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.downstream_response is None
        assert res.audit_outcome == "scan_error"
        assert res.error_code == "OUTPUT_LENGTH_EXCEEDED"

    def test_normalized_text_exceeding_100000_fails_closed(self):
        """Input within 25,000 chars that expands beyond 100,000 NFKC limit fails closed."""
        # 6000 * U+FDFA = 6,000 input chars (<25,000), but 6,000 * 18 = 108,000 NFKC chars (>100,000)
        expansion_bomb = "\uFDFA" * 6000
        assert len(expansion_bomb) == 6000
        with pytest.raises(OutputNormalizedLengthExceededError):
            scan_output(expansion_bomb)

        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=expansion_bomb,
            received_prompt_length=10,
        )
        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.downstream_response is None
        assert res.audit_outcome == "scan_error"
        assert res.error_code == "OUTPUT_LENGTH_EXCEEDED"
