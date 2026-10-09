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

    def test_unicode_escape_decoding_standard_ascii(self):
        raw = r"\u0073\u006b-abcdef1234567890abcdef1234567890"
        expected = "sk-abcdef1234567890abcdef1234567890"
        assert decode_for_detection(raw) == expected

    def test_unicode_escape_8_digit_standard_ascii(self):
        raw = r"\U00000073\U0000006B-abcdef1234567890abcdef1234567890"
        expected = "sk-abcdef1234567890abcdef1234567890"
        assert decode_for_detection(raw) == expected

    def test_unicode_escape_multibyte_and_supplementary(self):
        assert decode_for_detection(r"\u00E9") == "é"
        assert decode_for_detection(r"\u00e9") == "é"
        assert decode_for_detection(r"\U0001F600") == "😀"
        assert decode_for_detection(r"\U0001f600") == "😀"

    def test_mixed_byte_and_unicode_contiguous_escapes(self):
        raw = r"\u0073%6b\x2d\U00000070"
        expected = "sk-p"
        assert decode_for_detection(raw) == expected

    def test_disallowed_4_digit_uppercase_u_preserved_literally(self):
        raw = r"\U0073\U006b-test"
        assert decode_for_detection(raw) == raw

    def test_truncated_byte_run_before_unicode_escape_fails_closed(self):
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"%E2\u0073")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\xe2\u0073")

    def test_truncated_byte_run_after_unicode_escape_fails_closed(self):
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\u0073%E2")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\u0073\xe2")

    def test_null_code_point_in_unicode_escape_fails_closed(self):
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Key: \u0000secret")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Key: \U00000000secret")

    def test_surrogate_code_points_fail_closed(self):
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Token: \uD800")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Token: \uDFFF")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Token: \U0000D800")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Token: \U0000DFFF")

    def test_valid_surrogate_pairs_decode_to_unicode_scalars(self):
        """Stage 5.6: Valid UTF-16 surrogate pairs decode to 21-bit Unicode scalars."""
        assert decode_for_detection(r"\uD83D\uDE00") == "\U0001F600"
        assert decode_for_detection(r"\uD83C\uDF89") == "\U0001F389"
        assert (
            decode_for_detection(r"\uD83D\uDE00\uD83C\uDF89")
            == "\U0001F600\U0001F389"
        )

    def test_surrogate_pairs_mixed_case_hexadecimal_digits(self):
        """Stage 5.6: Surrogate pairs with uppercase, lowercase, and mixed-case hex digits decode."""
        assert decode_for_detection(r"\ud83d\ude00") == "\U0001F600"
        assert decode_for_detection(r"\uD83d\uDe00") == "\U0001F600"
        assert decode_for_detection(r"\uD83C\udf89") == "\U0001F389"

    def test_unpaired_high_surrogate_fails_closed(self):
        """Stage 5.6: Unpaired high surrogates fail closed in all contexts."""
        # At end of input
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Prefix \uD83D")
        # Followed by another high surrogate
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D\uD83D")
        # Followed by ordinary non-surrogate Unicode escape
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D\u0073")
        # Followed by ordinary text
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D_test")

    def test_unpaired_low_surrogate_fails_closed(self):
        """Stage 5.6: Isolated low surrogates fail closed."""
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Prefix \uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\u0073\uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"test\uDE00")

    def test_surrogates_separated_by_text_whitespace_or_byte_escape_fail_closed(self):
        """Stage 5.6: Surrogates separated by text, whitespace, or byte escapes fail closed."""
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D \uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83Dfoo\uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D%20\uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D\x20\uDE00")

    def test_valid_byte_run_adjacent_to_surrogate_pair(self):
        """Stage 5.6: Valid byte runs adjacent to a surrogate pair decode cleanly."""
        assert decode_for_detection(r"%C3%A9\uD83D\uDE00") == "é\U0001F600"
        assert decode_for_detection(r"\xc3\xa9\uD83D\uDE00") == "é\U0001F600"
        assert decode_for_detection(r"\uD83D\uDE00%C3%A9") == "\U0001F600é"
        assert decode_for_detection(r"%C3%A9\uD83D\uDE00%C3%A9") == "é\U0001F600é"

    def test_malformed_byte_runs_adjacent_to_surrogate_pair_fail_closed(self):
        """Stage 5.6: Malformed byte runs adjacent to a surrogate pair fail closed."""
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"%E2\uD83D\uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\xe2\uD83D\uDE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D\uDE00%E2")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\uD83D\uDE00\xe2")

    def test_eight_digit_surrogates_remain_strictly_rejected(self):
        """Stage 5.6: Eight-digit \\UXXXXXXXX escapes never permit surrogate code points."""
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\U0000D83D")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\U0000DE00")
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"\U0000D83D\U0000DE00")

    def test_single_pass_behavior_with_surrogates(self):
        """Stage 5.6: Single-pass decoding preserves produced escapes without recursion."""
        assert (
            decode_for_detection(r"\u005c\u00750061\uD83D\uDE00")
            == r"\u0061" + "\U0001F600"
        )

    def test_out_of_range_code_point_fails_closed(self):
        with pytest.raises(OutputDecodingError):
            decode_for_detection(r"Token: \U00110000")

    def test_maximum_valid_code_point_decodes_successfully(self):
        assert decode_for_detection(r"\U0010FFFF") == chr(0x10FFFF)

    def test_incomplete_and_non_hex_unicode_escapes_preserved(self):
        assert decode_for_detection(r"\u") == r"\u"
        assert decode_for_detection(r"\u1") == r"\u1"
        assert decode_for_detection(r"\u12") == r"\u12"
        assert decode_for_detection(r"\u123") == r"\u123"
        assert decode_for_detection(r"\U") == r"\U"
        assert decode_for_detection(r"\U1") == r"\U1"
        assert decode_for_detection(r"\U1234567") == r"\U1234567"
        assert decode_for_detection(r"\uZZZZ") == r"\uZZZZ"
        assert decode_for_detection(r"\UZZZZZZZZ") == r"\UZZZZZZZZ"
        assert decode_for_detection(r"\upload\file.txt") == r"\upload\file.txt"

    def test_windows_paths_with_u_preserved(self):
        path = r"C:\Users\admin\Documents\notes.txt and D:\unique\data.csv"
        assert decode_for_detection(path) == path

    def test_unicode_escape_double_encoded_not_recursively_decoded(self):
        assert decode_for_detection(r"\u005c\u00750073") == r"\u0073"
        assert decode_for_detection(r"\u00252D") == "%2D"
        assert decode_for_detection(r"\u005c\u007873") == r"\x73"


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

    def test_unicode_escaped_openai_key_detected_and_whole_response_redacted(self):
        """Stage 5.5: Unicode-escaped OpenAI key (\\u0073\\u006b) is detected and whole response redacted."""
        secret = r"\u0073\u006b-abcdef1234567890abcdef1234567890"
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
        assert "sk" not in res.downstream_response.message
        assert r"\u0073" not in res.downstream_response.message

    def test_8_digit_unicode_escaped_openai_key_detected_and_redacted(self):
        """Stage 5.5: 8-digit Unicode-escaped OpenAI key (\\U00000073\\U0000006B) is detected and redacted."""
        secret = r"\U00000073\U0000006B-abcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Key: {secret}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_unicode_escaped_database_uri_blocks_egress(self):
        """Stage 5.5: Unicode-escaped database URI credential blocks egress completely."""
        db_uri = r"postgres\u003a\u002f\u002fadmin\u003asecret123\u0040localhost\u003a5432\u002fdb"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Database config: {db_uri}",
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert res.risk_score == 1.0
        assert DLP_RULE_DATABASE_URI_CREDENTIAL in res.matched_rule_ids

    def test_unicode_escaped_private_key_blocks_egress(self):
        """Stage 5.5: Unicode-escaped private key header blocks egress completely."""
        header = r"-----\u0042\u0045\u0047\u0049\u004e RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAzsyntheticKeyData\n-----END RSA PRIVATE KEY-----"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Key dump:\n{header}",
            received_prompt_length=20,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert DLP_RULE_PRIVATE_KEY_HEADER in res.matched_rule_ids

    def test_unicode_escaped_aws_key_detected_and_redacted(self):
        """Stage 5.5: Unicode-escaped AWS access key ID is detected and redacted."""
        secret = r"\u0041\u004B\u0049\u00411234567890ABCDEF"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"AWS token: {secret}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert DLP_RULE_AWS_ACCESS_KEY_ID in res.matched_rule_ids

    def test_json_structured_response_with_unicode_escaped_credential_redacted(self):
        """Stage 5.5: JSON response with Unicode-escaped credentials is safe and whole-response redacted."""
        json_payload = r'{"api_key": "\u0073\u006b-abcdef1234567890abcdef1234567890", "status": "ok"}'
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=json_payload,
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_unicode_escaped_fullwidth_normalized_and_detected(self):
        """Stage 5.5: Unicode escape of full-width characters is decoded and NFKC-normalized to trigger DLP."""
        secret = r"\uFF53\uFF4B-abcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Token: {secret}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_unicode_decoding_exception_fails_closed_safely(self):
        """Stage 5.5: Malformed Unicode escape candidates fail closed without leaking original text."""
        malformed = r"Dangerous message with surrogate: \uD800 and key \u0073\u006b-abcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=malformed,
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.downstream_response is None
        assert res.audit_outcome == "scan_error"
        assert res.error_code == "OUTPUT_SCAN_ERROR"

    def test_benign_windows_paths_and_u_prose_allowed_cleanly(self):
        """Stage 5.5: Windows paths and text containing \\u / \\Users are allowed without false positives."""
        clean_msg = r"Check file C:\Users\admin\Documents\report.txt or \upload\file.doc and note \unique behavior."
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

    def test_audit_logs_omit_unicode_escaped_secrets_and_decoded_text(self):
        """Stage 5.5: Audit logs omit raw Unicode-escaped secrets and decoded intermediate text."""
        clear_audit_events()
        secret = r"\u0073\u006b-abcdef1234567890abcdef1234567890"
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=f"Key: {secret}",
            received_prompt_length=10,
        )

        res = enforce_output_security(mock_resp)
        log_output_security_event(
            request_id="req-audit-unicode-test",
            endpoint="/api/v1/proxy",
            outcome=res.audit_outcome,
            matched_rule_ids=res.matched_rule_ids,
            finding_count=res.finding_count,
            client_request_id="client-test-55",
        )

        events = get_audit_events()
        assert len(events) > 0
        last_event = events[-1]
        serialized = json.dumps(last_event)

        assert r"\u0073\u006b" not in serialized
        assert "sk-abcdef1234567890abcdef1234567890" not in serialized
        assert "abcdef1234567890" not in serialized
        clear_audit_events()

    def test_json_response_with_surrogate_pair_allowed_and_audited_clean(self):
        """Stage 5.6: Valid JSON response containing an escaped surrogate pair is allowed and clean."""
        json_msg = r'{"status": "ok", "emoji": "\uD83D\uDE00", "count": 42}'
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=json_msg,
            received_prompt_length=15,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "clean"
        assert res.downstream_response is not None
        assert res.downstream_response.message == json_msg
        assert res.finding_count == 0

    def test_json_response_with_surrogate_pair_and_escaped_credential_redacted(self):
        """Stage 5.6: Surrogate pair alongside an escaped credential triggers whole-response redaction."""
        json_msg = (
            r'{"emoji": "\uD83D\uDE00", '
            r'"api_key": "\u0073\u006b-abcdef1234567890abcdef1234567890"}'
        )
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=json_msg,
            received_prompt_length=20,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "ALLOW"
        assert res.audit_outcome == "redacted"
        assert res.downstream_response is not None
        assert res.downstream_response.message == "[REDACTED]"
        assert res.finding_count == 1
        assert DLP_RULE_OPENAI_API_KEY in res.matched_rule_ids

    def test_json_response_with_surrogate_pair_and_critical_secret_blocked(self):
        """Stage 5.6: Surrogate pair alongside an escaped critical credential blocks egress."""
        json_msg = (
            r'{"emoji": "\uD83D\uDE00", '
            r'"db": "postgres\u003a\u002f\u002fadmin\u003asecret123\u0040localhost\u003a5432\u002fdb"}'
        )
        mock_resp = MockDownstreamResponse(
            simulated=True,
            provider="mock-llm-service",
            message=json_msg,
            received_prompt_length=20,
        )

        res = enforce_output_security(mock_resp)
        assert res.effective_decision == "BLOCK"
        assert res.audit_outcome == "blocked"
        assert res.downstream_response is None
        assert DLP_RULE_DATABASE_URI_CREDENTIAL in res.matched_rule_ids
