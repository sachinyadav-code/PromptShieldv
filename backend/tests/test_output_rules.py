"""Unit tests for PromptShield Phase 4 Stage 2 Output Security DLP rules."""

import pytest

from app.output.rules import (
    CONFIGURED_OUTPUT_RULES,
    DLP_RULE_AWS_ACCESS_KEY_ID,
    DLP_RULE_BEARER_TOKEN,
    DLP_RULE_DATABASE_URI_CREDENTIAL,
    DLP_RULE_OPENAI_API_KEY,
    DLP_RULE_PAYMENT_CARD,
    DLP_RULE_PRIVATE_KEY_HEADER,
    DLP_RULE_SECRET_ASSIGNMENT,
    DLP_RULE_US_SSN,
    MAX_FINDINGS_COUNT,
    MAX_OUTPUT_TEXT_LENGTH,
    OutputFinding,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
    OutputScanError,
    get_configured_output_rules,
    luhn_checksum_valid,
    scan_output_text,
)


class TestConfiguredOutputRulesCatalog:
    """Verify metadata catalog integrity for configured DLP rules."""

    def test_catalog_has_all_eight_required_rules(self):
        rules = get_configured_output_rules()
        assert len(rules) == 8
        rule_ids = {r.rule_id for r in rules}
        expected_ids = {
            DLP_RULE_PRIVATE_KEY_HEADER,
            DLP_RULE_AWS_ACCESS_KEY_ID,
            DLP_RULE_OPENAI_API_KEY,
            DLP_RULE_BEARER_TOKEN,
            DLP_RULE_DATABASE_URI_CREDENTIAL,
            DLP_RULE_PAYMENT_CARD,
            DLP_RULE_US_SSN,
            DLP_RULE_SECRET_ASSIGNMENT,
        }
        assert rule_ids == expected_ids

    def test_rule_severities_and_categories_documented(self):
        for rule in get_configured_output_rules():
            assert rule.severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            assert rule.category in (
                "PRIVATE_KEY",
                "CLOUD_CREDENTIAL",
                "API_KEY",
                "BEARER_TOKEN",
                "DATABASE_CREDENTIAL",
                "PAYMENT_CARD",
                "GOVERNMENT_ID",
                "SECRET_ASSIGNMENT",
            )
            assert len(rule.description) > 20
            # Documented disclaimers in rule descriptions
            assert (
                "does not" in rule.description.lower()
                or "header detection only" in rule.description.lower()
                or "potential false positive" in rule.description.lower()
                or "excludes" in rule.description.lower()
            )


class TestPrivateKeyHeaderDetection:
    """Positive and negative tests for private-key header detection."""

    @pytest.mark.parametrize(
        "header_text",
        [
            "-----BEGIN RSA PRIVATE KEY-----",
            "-----BEGIN EC PRIVATE KEY-----",
            "-----BEGIN DSA PRIVATE KEY-----",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
            "-----BEGIN PRIVATE KEY-----",
            "-----BEGIN ENCRYPTED PRIVATE KEY-----",
            "-----begin rsa private key-----",
        ],
    )
    def test_detects_all_supported_private_key_headers(self, header_text: str):
        content = f"Here is the exported key:\n{header_text}\nMIIEowIBAAKCAQ..."
        findings = scan_output_text(content)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_PRIVATE_KEY_HEADER
        assert finding.category == "PRIVATE_KEY"
        assert finding.severity == "CRITICAL"
        assert content[finding.start : finding.end].lower() == header_text.lower()
        assert 0 <= finding.start < finding.end <= len(content)

    @pytest.mark.parametrize(
        "benign_text",
        [
            "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...",
            "-----BEGIN CERTIFICATE-----\nMIIDXTCCAkWgAwIBAgIJAP3G5A...",
            "Use ssh-keygen -t ed25519 to generate a new key pair.",
            "The system does not expose your private key.",
        ],
    )
    def test_ignores_non_private_key_headers(self, benign_text: str):
        findings = scan_output_text(benign_text)
        assert len(findings) == 0


class TestAWSAccessKeyIDDetection:
    """Positive, negative, and placeholder tests for AWS access-key IDs."""

    def test_detects_synthetic_aws_access_key(self):
        text = "Your access key is AKIA1234567890ABCDEF for development."
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_AWS_ACCESS_KEY_ID
        assert finding.category == "CLOUD_CREDENTIAL"
        assert finding.severity == "HIGH"
        assert text[finding.start : finding.end] == "AKIA1234567890ABCDEF"

    def test_does_not_ignore_key_merely_containing_test(self):
        # Must NOT ignore a candidate merely because it contains 'TEST'
        text = "Deploy using key AKIATEST0123456789AB in testing."
        findings = scan_output_text(text)
        assert len(findings) == 1
        assert findings[0].rule_id == DLP_RULE_AWS_ACCESS_KEY_ID
        assert text[findings[0].start : findings[0].end] == "AKIATEST0123456789AB"

    def test_excludes_exact_canonical_aws_documentation_placeholder(self):
        # AKIAIOSFODNN7EXAMPLE is the canonical AWS documentation example
        text = "Example configuration: export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        findings = scan_output_text(text)
        # The exact documentation placeholder should not be reported as a leaked credential
        assert not any(f.rule_id == DLP_RULE_AWS_ACCESS_KEY_ID for f in findings)

    @pytest.mark.parametrize(
        "invalid_key",
        [
            "AKIA123",  # Too short
            "BKIA1234567890ABCDEF",  # Wrong prefix
            "akia1234567890abcdef",  # Lowercase
            "AKIA1234567890ABCDEFG_EXTRA",  # Non-word-boundary continuation
        ],
    )
    def test_ignores_invalid_aws_key_formats(self, invalid_key: str):
        text = f"Invalid token format: {invalid_key}"
        findings = scan_output_text(text)
        assert not any(f.rule_id == DLP_RULE_AWS_ACCESS_KEY_ID for f in findings)


class TestOpenAIApiKeyDetection:
    """Positive, negative, and placeholder tests for OpenAI-style API keys."""

    @pytest.mark.parametrize(
        "synthetic_key",
        [
            "sk-abcdef1234567890abcdef1234567890",
            "sk-proj-1234567890abcdef1234567890abcdef123456",
            "sk-abc12345678901234567890xyz",
        ],
    )
    def test_detects_synthetic_openai_style_keys(self, synthetic_key: str):
        text = f"Configured client with OPENAI_API_KEY={synthetic_key}"
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_OPENAI_API_KEY
        assert finding.category == "API_KEY"
        assert finding.severity == "HIGH"
        assert text[finding.start : finding.end] == synthetic_key

    @pytest.mark.parametrize(
        "short_or_benign",
        [
            "Import sklearn as sk for modeling: sk.learn",
            "Model checkpoint sk-123",
            "Command line arg --sk-test",
            "The library is scikit-learn (sk-learn)",
        ],
    )
    def test_ignores_short_or_benign_sk_mentions(self, short_or_benign: str):
        findings = scan_output_text(short_or_benign)
        assert len(findings) == 0

    def test_excludes_exact_openai_documentation_placeholder(self):
        text = "Set OPENAI_API_KEY=sk-00000000000000000000000000000000 in your .env"
        findings = scan_output_text(text)
        assert not any(f.rule_id == DLP_RULE_OPENAI_API_KEY for f in findings)


class TestBearerTokenDetection:
    """Positive, negative, and placeholder tests for Bearer tokens."""

    def test_detects_synthetic_jwt_bearer_token(self):
        token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        text = f"Request headers: Authorization: {token}"
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_BEARER_TOKEN
        assert finding.category == "BEARER_TOKEN"
        assert finding.severity == "HIGH"
        assert text[finding.start : finding.end] == token

    def test_detects_synthetic_hex_bearer_token(self):
        token = "Bearer a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        text = f"curl -H 'Authorization: {token}' https://api.internal"
        findings = scan_output_text(text)
        assert len(findings) == 1
        assert findings[0].rule_id == DLP_RULE_BEARER_TOKEN
        assert text[findings[0].start : findings[0].end] == token

    @pytest.mark.parametrize(
        "placeholder_header",
        [
            "Authorization: Bearer <TOKEN>",
            "Authorization: Bearer [REDACTED]",
            "Authorization: Bearer {token}",
            "Authorization: Bearer YOUR_TOKEN_HERE",
            "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE",
            "Authorization: Bearer short_token",
        ],
    )
    def test_excludes_generic_bearer_placeholders(self, placeholder_header: str):
        findings = scan_output_text(placeholder_header)
        assert not any(f.rule_id == DLP_RULE_BEARER_TOKEN for f in findings)


class TestDatabaseURICredentialDetection:
    """Positive and negative tests for credential-bearing database URIs."""

    @pytest.mark.parametrize(
        "uri",
        [
            "postgres://app_user:s3cr3t_pass123@localhost:5432/production_db",
            "postgresql://admin:TopSecretPassword@10.0.0.5:5432/mydb",
            "mysql://root:toor_secret_pwd@127.0.0.1:3306/users",
            "mongodb://service_user:p%40ssw0rd@cluster0.example.com:27017/analytics",
            "mongodb+srv://admin_user:ClusterSecret99@cluster0.mongodb.net/test",
            "redis://:cache_secret_token_123@redis-master.internal:6379/0",
            "rediss://admin:tls_redis_pass@cache.internal:6380",
        ],
    )
    def test_detects_credential_bearing_database_uris(self, uri: str):
        text = f"Connection string is: {uri}"
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_DATABASE_URI_CREDENTIAL
        assert finding.category == "DATABASE_CREDENTIAL"
        assert finding.severity == "CRITICAL"
        assert text[finding.start : finding.end] == uri

    @pytest.mark.parametrize(
        "benign_uri",
        [
            "postgres://localhost:5432/production_db",
            "postgresql://db.internal:5432/mydb",
            "mysql://127.0.0.1:3306/users",
            "mongodb://localhost:27017/analytics",
            "redis://redis-master.internal:6379/0",
            "postgres://app_user@localhost:5432/db",  # User only, no password delimiter
        ],
    )
    def test_ignores_database_uris_without_credentials(self, benign_uri: str):
        text = f"Connect to database at {benign_uri}"
        findings = scan_output_text(text)
        assert len(findings) == 0


class TestPaymentCardCandidateDetection:
    """Tests for payment-card candidate detection with Luhn verification."""

    def test_luhn_checksum_algorithm_direct(self):
        # 16-digit valid Luhn number
        assert luhn_checksum_valid("4000000000000002") is True
        assert luhn_checksum_valid("4000-0000-0000-0002") is True
        assert luhn_checksum_valid("4000 0000 0000 0002") is True

        # 15-digit valid Luhn Amex
        assert luhn_checksum_valid("378282246310005") is True

        # Invalid Luhn numbers
        assert luhn_checksum_valid("4000000000000003") is False
        assert luhn_checksum_valid("1234567812345678") is False

        # Edge cases: leading zero or all zeros
        assert luhn_checksum_valid("0000000000000000") is False
        assert luhn_checksum_valid("1111111111111111") is False

        # Out-of-bounds length
        assert luhn_checksum_valid("123456789012") is False  # 12 digits
        assert luhn_checksum_valid("12345678901234567890") is False  # 20 digits

    @pytest.mark.parametrize(
        "card_number",
        [
            "4000000000000002",
            "4000-0000-0000-0002",
            "4000 0000 0000 0002",
            "3782-822463-10005",
        ],
    )
    def test_detects_luhn_valid_card_candidates(self, card_number: str):
        text = f"Transaction billed to card {card_number} on file."
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_PAYMENT_CARD
        assert finding.category == "PAYMENT_CARD"
        assert finding.severity == "CRITICAL"
        assert text[finding.start : finding.end] == card_number

    @pytest.mark.parametrize(
        "invalid_candidate",
        [
            "4000-0000-0000-0003",  # Invalid checksum
            "1234-5678-9012-3456",  # Invalid checksum
            "0000-0000-0000-0000",  # All zeros
            "Invoice # 2024-10-09-1234",  # Order / date numbers
            "1234567890",  # 10 digits
        ],
    )
    def test_ignores_luhn_invalid_or_non_card_candidates(self, invalid_candidate: str):
        text = f"Reference: {invalid_candidate}"
        findings = scan_output_text(text)
        assert not any(f.rule_id == DLP_RULE_PAYMENT_CARD for f in findings)


class TestUSSSNCandidateDetection:
    """Tests for US SSN format and valid range detection."""

    @pytest.mark.parametrize(
        "valid_ssn",
        [
            "123-45-6789",
            "219-09-9999",
            "078-05-1120",
        ],
    )
    def test_detects_valid_format_ssn_candidates(self, valid_ssn: str):
        text = f"Customer SSN on application: {valid_ssn}"
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == DLP_RULE_US_SSN
        assert finding.category == "GOVERNMENT_ID"
        assert finding.severity == "HIGH"
        assert text[finding.start : finding.end] == valid_ssn

    @pytest.mark.parametrize(
        "invalid_ssn",
        [
            "000-12-3456",  # Area 000 is invalid
            "666-12-3456",  # Area 666 is invalid
            "901-12-3456",  # Area 900-999 is invalid
            "123-00-4567",  # Group 00 is invalid
            "123-45-0000",  # Serial 0000 is invalid
            "000-00-0000",  # All zeros
            "2024-05-12",  # Date format
            "123456789",  # Missing hyphens (unbounded 9 digits)
        ],
    )
    def test_ignores_invalid_ssn_ranges_and_formats(self, invalid_ssn: str):
        text = f"Identifier code: {invalid_ssn}"
        findings = scan_output_text(text)
        assert not any(f.rule_id == DLP_RULE_US_SSN for f in findings)


class TestSecretAssignmentDetection:
    """Tests for password/secret assignments and placeholder exclusions."""

    @pytest.mark.parametrize(
        ("expr", "expected_rule"),
        [
            ('password="SuperSecret123!"', DLP_RULE_SECRET_ASSIGNMENT),
            ("passwd: SecretPass999", DLP_RULE_SECRET_ASSIGNMENT),
            ("client_secret=sec_abcdef123456", DLP_RULE_SECRET_ASSIGNMENT),
            ('api_key = "internal_secret_val_456"', DLP_RULE_SECRET_ASSIGNMENT),
            ('pwd := "DatabasePassword888"', DLP_RULE_SECRET_ASSIGNMENT),
        ],
    )
    def test_detects_secret_assignments(self, expr: str, expected_rule: str):
        text = f"Configuration setting: {expr} in file."
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == expected_rule
        assert finding.category == "SECRET_ASSIGNMENT"
        assert finding.severity == "HIGH"
        assert text[finding.start : finding.end] == expr

    def test_does_not_ignore_assignment_merely_containing_test_or_dummy(self):
        text = 'config: password="my_test_password_123"'
        findings = scan_output_text(text)
        assert len(findings) == 1
        assert findings[0].rule_id == DLP_RULE_SECRET_ASSIGNMENT

    @pytest.mark.parametrize(
        "placeholder_expr",
        [
            'password=""',
            "password=''",
            "password=",
            "passwd: ",
            'password="your_password"',
            'password="<password>"',
            'password="[REDACTED]"',
            'password="changeme"',
            'password="placeholder"',
            'password="TODO"',
            'password="none"',
            'password="null"',
            'password="***"',
        ],
    )
    def test_excludes_generic_secret_assignment_placeholders(self, placeholder_expr: str):
        text = f"Template setup: {placeholder_expr}"
        findings = scan_output_text(text)
        assert len(findings) == 0


class TestMultipleFindingsAndOverlapResolution:
    """Tests for multiple findings, overlap resolution, and deterministic ordering."""

    def test_multiple_distinct_findings_ordered_by_offset(self):
        text = (
            "Environment Report:\n"
            "AWS Key: AKIA1234567890ABCDEF\n"
            "SSN: 123-45-6789\n"
            "Key Header: -----BEGIN RSA PRIVATE KEY-----\n"
        )
        findings = scan_output_text(text)
        assert len(findings) == 3

        # Must be ordered by start offset ascending
        assert findings[0].start < findings[1].start < findings[2].start
        assert findings[0].rule_id == DLP_RULE_AWS_ACCESS_KEY_ID
        assert findings[1].rule_id == DLP_RULE_US_SSN
        assert findings[2].rule_id == DLP_RULE_PRIVATE_KEY_HEADER

    def test_overlap_resolution_prefers_specific_rule_over_generic_assignment(self):
        # api_key="sk-abcdef1234567890abcdef1234567890" overlaps:
        # 1. DLP_RULE_SECRET_ASSIGNMENT (covers the full assignment)
        # 2. DLP_RULE_OPENAI_API_KEY (covers the token)
        text = 'export api_key="sk-abcdef1234567890abcdef1234567890"'

        # When resolve_overlaps=True (default), specific API key rule wins
        resolved = scan_output_text(text, resolve_overlaps=True)
        assert len(resolved) == 1
        assert resolved[0].rule_id == DLP_RULE_OPENAI_API_KEY

        # When resolve_overlaps=False, both findings are retained with exact dedup
        unresolved = scan_output_text(text, resolve_overlaps=False)
        assert len(unresolved) == 2
        rule_ids = {f.rule_id for f in unresolved}
        assert rule_ids == {DLP_RULE_SECRET_ASSIGNMENT, DLP_RULE_OPENAI_API_KEY}

    def test_all_resolved_spans_are_strictly_disjoint(self):
        text = (
            'export api_key="sk-abcdef1234567890abcdef1234567890"; '
            "db=postgres://user:pass123@localhost:5432/app; "
            "card=4000-0000-0000-0002"
        )
        findings = scan_output_text(text, resolve_overlaps=True)
        assert len(findings) >= 2
        for i in range(len(findings) - 1):
            assert findings[i].end <= findings[i + 1].start


class TestPrivacyInvariantsAndLimits:
    """Ensure raw secrets are never leaked in finding objects or errors."""

    def test_finding_dataclass_contains_zero_raw_secret_data(self):
        secret = "AKIA1234567890ABCDEF"
        text = f"AWS access token: {secret}"
        findings = scan_output_text(text)
        assert len(findings) == 1
        finding = findings[0]

        # Verify attributes
        assert not hasattr(finding, "matched_value")
        assert not hasattr(finding, "raw_secret")
        assert not hasattr(finding, "secret")

        # Verify serialized dict
        d = finding.to_dict()
        assert secret not in str(d)
        assert secret not in finding.description
        assert secret not in repr(finding)

    def test_all_spans_within_text_bounds(self):
        text = "Private: -----BEGIN RSA PRIVATE KEY----- and card 4000000000000002."
        findings = scan_output_text(text)
        for f in findings:
            assert 0 <= f.start < f.end <= len(text)
            assert f.length == (f.end - f.start)
            assert f.span == (f.start, f.end)

    def test_empty_and_whitespace_input_returns_empty_list(self):
        assert scan_output_text("") == []
        assert scan_output_text("   \n\t  ") == []

    def test_non_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Input text must be a string"):
            scan_output_text(12345)  # type: ignore

    def test_text_at_max_length_succeeds(self):
        text = "Clean text " * (MAX_OUTPUT_TEXT_LENGTH // 11)
        assert len(text) <= MAX_OUTPUT_TEXT_LENGTH
        findings = scan_output_text(text)
        assert findings == []

    def test_text_exceeding_max_length_raises_safe_error(self):
        oversized = "A" * (MAX_OUTPUT_TEXT_LENGTH + 1)
        with pytest.raises(OutputLengthExceededError) as exc_info:
            scan_output_text(oversized)

        err_msg = str(exc_info.value)
        assert str(MAX_OUTPUT_TEXT_LENGTH) in err_msg
        assert str(len(oversized)) in err_msg
        # Must NOT include any of the text content itself
        assert "AAAAA" not in err_msg


class TestStage21OverlapResolutionHardening:
    """Regression tests for Stage 2.1 strict deterministic overlap resolution."""

    def test_higher_priority_finding_starting_inside_lower_priority_is_retained(self):
        # Generic assignment starts at index 0 and extends to index 59
        # CRITICAL private key header starts at index 12 and extends to index 47
        text = 'secret_key = "-----BEGIN RSA PRIVATE KEY-----"'
        findings = scan_output_text(text, resolve_overlaps=True)

        assert len(findings) == 1
        # The higher-severity / higher-priority finding MUST be retained
        assert findings[0].rule_id == DLP_RULE_PRIVATE_KEY_HEADER
        assert findings[0].severity == "CRITICAL"
        assert text[findings[0].start : findings[0].end] == "-----BEGIN RSA PRIVATE KEY-----"

    def test_specific_credential_findings_take_precedence_over_generic_assignments(self):
        # 1. AWS key inside assignment
        text_aws = 'export aws_key="AKIA1234567890ABCDEF"'
        findings_aws = scan_output_text(text_aws, resolve_overlaps=True)
        assert len(findings_aws) == 1
        assert findings_aws[0].rule_id == DLP_RULE_AWS_ACCESS_KEY_ID

        # 2. Database URI inside assignment
        text_db = 'database_url = "postgres://appuser:mypassword123@localhost:5432/db"'
        findings_db = scan_output_text(text_db, resolve_overlaps=True)
        assert len(findings_db) == 1
        assert findings_db[0].rule_id == DLP_RULE_DATABASE_URI_CREDENTIAL
        assert findings_db[0].severity == "CRITICAL"

        # 3. OpenAI key inside assignment
        text_openai = 'api_key="sk-abcdef1234567890abcdef1234567890"'
        findings_openai = scan_output_text(text_openai, resolve_overlaps=True)
        assert len(findings_openai) == 1
        assert findings_openai[0].rule_id == DLP_RULE_OPENAI_API_KEY

    def test_lower_priority_overlapping_multiple_higher_priority_findings_is_discarded(self):
        # Suppose a generic assignment pattern overlaps two disjoint specific credentials
        # We simulate this via synthetic findings directly in deduplicate_and_order_findings
        f_generic = OutputFinding(
            rule_id=DLP_RULE_SECRET_ASSIGNMENT,
            category="SECRET_ASSIGNMENT",
            severity="HIGH",
            start=0,
            end=100,
            description="Generic assignment",
        )
        f_specific_1 = OutputFinding(
            rule_id=DLP_RULE_AWS_ACCESS_KEY_ID,
            category="CLOUD_CREDENTIAL",
            severity="HIGH",
            start=10,
            end=30,
            description="AWS key 1",
        )
        f_specific_2 = OutputFinding(
            rule_id=DLP_RULE_AWS_ACCESS_KEY_ID,
            category="CLOUD_CREDENTIAL",
            severity="HIGH",
            start=50,
            end=70,
            description="AWS key 2",
        )

        from app.output.rules import deduplicate_and_order_findings

        resolved = deduplicate_and_order_findings(
            [f_generic, f_specific_1, f_specific_2],
            text_length=120,
            resolve_overlaps=True,
        )

        # Both specific findings MUST be preserved; generic one discarded
        assert len(resolved) == 2
        assert resolved[0].span == (10, 30)
        assert resolved[1].span == (50, 70)
        assert resolved[0].rule_id == DLP_RULE_AWS_ACCESS_KEY_ID
        assert resolved[1].rule_id == DLP_RULE_AWS_ACCESS_KEY_ID

    def test_non_overlapping_findings_remain_present(self):
        text = (
            "1. Header: -----BEGIN EC PRIVATE KEY-----\n"
            "2. DB: mysql://root:rootpass@localhost:3306/db\n"
            "3. SSN: 123-45-6789\n"
            "4. Card: 4000-0000-0000-0002"
        )
        findings = scan_output_text(text, resolve_overlaps=True)
        assert len(findings) == 4
        expected_rules = [
            DLP_RULE_PRIVATE_KEY_HEADER,
            DLP_RULE_DATABASE_URI_CREDENTIAL,
            DLP_RULE_US_SSN,
            DLP_RULE_PAYMENT_CARD,
        ]
        assert [f.rule_id for f in findings] == expected_rules

    def test_repeated_scans_produce_strictly_identical_results(self):
        text = (
            'export config="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"; '
            'password="SuperSecretPassword123!"; '
            "card=4000 0000 0000 0002; "
            "AKIA1234567890ABCDEF"
        )
        baseline = scan_output_text(text, resolve_overlaps=True)
        for _ in range(10):
            repeated = scan_output_text(text, resolve_overlaps=True)
            assert repeated == baseline

    def test_all_returned_spans_are_valid_and_strictly_disjoint(self):
        text = (
            'export api_key="sk-abcdef1234567890abcdef1234567890"; '
            "db=postgres://user:pass123@localhost:5432/app; "
            "card=4000-0000-0000-0002; "
            "ssn=123-45-6789"
        )
        findings = scan_output_text(text, resolve_overlaps=True)
        for i in range(len(findings)):
            assert 0 <= findings[i].start < findings[i].end <= len(text)
            if i > 0:
                assert findings[i - 1].end <= findings[i].start

    def test_resolve_overlaps_false_preserves_all_distinct_findings(self):
        text = 'export api_key="sk-abcdef1234567890abcdef1234567890"'
        findings = scan_output_text(text, resolve_overlaps=False)
        assert len(findings) == 2
        rule_ids = {f.rule_id for f in findings}
        assert rule_ids == {DLP_RULE_SECRET_ASSIGNMENT, DLP_RULE_OPENAI_API_KEY}


class TestStage21FindingLimitOverflow:
    """Regression tests for explicit finding-limit overflow handling."""

    def test_overflow_is_explicitly_reported_via_exception(self):
        # 4 distinct SSN findings with max_findings=3
        text = "SSNs: 123-45-6789, 219-09-9999, 078-05-1120, 123-45-6780"
        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scan_output_text(text, max_findings=3)

        err = exc_info.value
        assert err.count == 4
        assert err.max_count == 3
        assert "Detected finding count (4) exceeds the maximum allowed limit of 3 findings." in str(err)

    def test_overflow_never_returns_partial_clean_finding_list(self):
        # When limit is exceeded, it must NOT return a partial 3-item list
        text = "SSNs: 123-45-6789, 219-09-9999, 078-05-1120, 123-45-6780"
        result = None
        try:
            result = scan_output_text(text, max_findings=3)
        except OutputFindingLimitExceededError:
            pass
        assert result is None, "Must not return partial findings when limit is exceeded"

    def test_exception_message_contains_zero_matched_secrets_or_text(self):
        secret_ssn = "123-45-6789"
        text = f"SSNs: {secret_ssn}, 219-09-9999"
        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scan_output_text(text, max_findings=1)

        err_msg = str(exc_info.value)
        assert secret_ssn not in err_msg
        assert "219-09-9999" not in err_msg
        assert "SSNs:" not in err_msg

    def test_normal_inputs_at_or_below_limit_retain_existing_behavior(self):
        # 3 findings with max_findings=3: exactly at limit, must succeed
        text = "SSNs: 123-45-6789, 219-09-9999, 078-05-1120"
        findings = scan_output_text(text, max_findings=3)
        assert len(findings) == 3

        # 3 findings with max_findings=5: below limit, must succeed
        findings_under = scan_output_text(text, max_findings=5)
        assert len(findings_under) == 3

    def test_adversarial_dense_payload_bounded_early(self):
        # Generate dense valid SSNs exceeding raw_ceiling
        dense_ssns = " ".join(["123-45-6789"] * 550)
        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scan_output_text(dense_ssns, max_findings=100)

        err = exc_info.value
        assert err.count >= 500
        assert err.max_count == 100


class TestStage22BoundedProcessingVerification:
    """Targeted verification for Stage 2.2 bounded processing, complexity, and fail-closed safety."""

    def test_unresolved_overlap_mode_enforces_limit_overflow(self):
        # When resolve_overlaps=False, finding limits must still be strictly enforced
        text = "SSNs: 123-45-6789, 219-09-9999, 078-05-1120, 123-45-6780"
        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scan_output_text(text, resolve_overlaps=False, max_findings=2)

        err = exc_info.value
        assert err.count == 4
        assert err.max_count == 2
        assert "Detected finding count (4) exceeds the maximum allowed limit of 2 findings." in str(err)

    def test_candidate_cap_fails_closed_without_omitting_critical_findings(self):
        # Dense input containing 600 lower-severity matches followed by a CRITICAL private key
        # The scan must explicitly fail-closed (raise exception) rather than silently returning
        # a partial list of findings that omitted the critical finding.
        dense_prefix = " ".join(["password='mypassword123'"] * 550)
        full_text = f"{dense_prefix}\n-----BEGIN RSA PRIVATE KEY-----"

        with pytest.raises(OutputFindingLimitExceededError) as exc_info:
            scan_output_text(full_text[:MAX_OUTPUT_TEXT_LENGTH], max_findings=100)

        err = exc_info.value
        assert err.count >= 500
        assert err.max_count == 100
        # Critical secret is NEVER leaked in the exception message
        assert "RSA PRIVATE KEY" not in str(err)

    def test_worst_case_dense_execution_time_is_bounded(self):
        import time

        # Generate dense text up to near max length (24,000 chars)
        dense_text = ("123-45-6789 " * 2000)[:24000]

        start_time = time.perf_counter()
        with pytest.raises(OutputFindingLimitExceededError):
            scan_output_text(dense_text, max_findings=100)
        elapsed_s = time.perf_counter() - start_time

        # Bounded processing must complete in well under 1 second (typically < 30ms)
        assert elapsed_s < 1.0, f"Dense scanning took too long: {elapsed_s:.3f}s"

    def test_spans_overlap_boundary_semantics(self):
        from app.output.rules import spans_overlap

        f1 = OutputFinding(
            rule_id=DLP_RULE_US_SSN,
            category="GOVERNMENT_ID",
            severity="HIGH",
            start=10,
            end=20,
            description="Finding 1",
        )
        f_touching = OutputFinding(
            rule_id=DLP_RULE_US_SSN,
            category="GOVERNMENT_ID",
            severity="HIGH",
            start=20,
            end=30,
            description="Touching adjacent span",
        )
        f_overlap_1char = OutputFinding(
            rule_id=DLP_RULE_US_SSN,
            category="GOVERNMENT_ID",
            severity="HIGH",
            start=19,
            end=30,
            description="1-character overlap",
        )
        f_contained = OutputFinding(
            rule_id=DLP_RULE_US_SSN,
            category="GOVERNMENT_ID",
            severity="HIGH",
            start=12,
            end=18,
            description="Contained inside f1",
        )

        # Touching boundary: [10, 20) and [20, 30) do NOT overlap
        assert spans_overlap(f1, f_touching) is False
        assert spans_overlap(f_touching, f1) is False

        # 1-char overlap: [10, 20) and [19, 30) DO overlap
        assert spans_overlap(f1, f_overlap_1char) is True
        assert spans_overlap(f_overlap_1char, f1) is True

        # Contained: [12, 18) inside [10, 20) DO overlap
        assert spans_overlap(f1, f_contained) is True
        assert spans_overlap(f_contained, f1) is True
