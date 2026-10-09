"""Deterministic output-security and Data Loss Prevention (DLP) rules for PromptShield Phase 4.

IMPORTANT SECURITY, STATISTICAL, AND COMPLIANCE NOTICE:
- Rule-based regex pattern and checksum matching provides a deterministic, transparent
  first-line egress defense against high-confidence sensitive data leakage.
- Format and checksum validation (e.g. Luhn algorithm for payment cards, format ranges
  for US SSNs, AKIA prefix for AWS access keys) does NOT prove or guarantee that a detected
  value is an active, issued, funded, or genuine credential or personal identity.
- Conversely, lack of a match does NOT prove that output is completely harmless; obfuscated,
  paraphrased, or non-standard token representations can produce false negatives.
- Semantic system-prompt leakage detection, PII inference (e.g., names, addresses),
  unbounded high-entropy scanning, and ML-based output classification are intentionally
  deferred from this deterministic module.
- Raw secret values, credentials, or customer data are STRICTLY PROHIBITED from being
  recorded in logs, exception messages, diagnostic payloads, or finding representations.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Resource and Safety Ceilings
MAX_OUTPUT_TEXT_LENGTH: int = 25000  # Enforced maximum character length for scanned text
MAX_FINDINGS_COUNT: int = 100        # Maximum number of findings returned per scan

# Type aliases
OutputSeverity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
OutputCategory = Literal[
    "PRIVATE_KEY",
    "CLOUD_CREDENTIAL",
    "API_KEY",
    "BEARER_TOKEN",
    "DATABASE_CREDENTIAL",
    "PAYMENT_CARD",
    "GOVERNMENT_ID",
    "SECRET_ASSIGNMENT",
]

# Stable Rule Identifier Constants
DLP_RULE_PRIVATE_KEY_HEADER = "DLP_RULE_PRIVATE_KEY_HEADER"
DLP_RULE_AWS_ACCESS_KEY_ID = "DLP_RULE_AWS_ACCESS_KEY_ID"
DLP_RULE_OPENAI_API_KEY = "DLP_RULE_OPENAI_API_KEY"
DLP_RULE_BEARER_TOKEN = "DLP_RULE_BEARER_TOKEN"
DLP_RULE_DATABASE_URI_CREDENTIAL = "DLP_RULE_DATABASE_URI_CREDENTIAL"
DLP_RULE_PAYMENT_CARD = "DLP_RULE_PAYMENT_CARD"
DLP_RULE_US_SSN = "DLP_RULE_US_SSN"
DLP_RULE_SECRET_ASSIGNMENT = "DLP_RULE_SECRET_ASSIGNMENT"

# Aliases for flexible integration
RULE_DLP_PRIVATE_KEY = DLP_RULE_PRIVATE_KEY_HEADER
RULE_DLP_AWS_KEY = DLP_RULE_AWS_ACCESS_KEY_ID
RULE_DLP_OPENAI_KEY = DLP_RULE_OPENAI_API_KEY
RULE_DLP_BEARER_TOKEN = DLP_RULE_BEARER_TOKEN
RULE_DLP_DB_CREDENTIAL_URI = DLP_RULE_DATABASE_URI_CREDENTIAL
RULE_DLP_PAYMENT_CARD = DLP_RULE_PAYMENT_CARD
RULE_DLP_US_SSN = DLP_RULE_US_SSN
RULE_DLP_SECRET_ASSIGNMENT = DLP_RULE_SECRET_ASSIGNMENT


class OutputScanError(Exception):
    """Base exception for output security scanning errors."""
    pass


class OutputLengthExceededError(OutputScanError, ValueError):
    """Raised when output text exceeds the maximum permitted length.

    IMPORTANT: To prevent data leakage, this exception message only reports
    character counts and limits, never the underlying text content.
    """

    def __init__(self, length: int, max_length: int) -> None:
        super().__init__(
            f"Output text length ({length} characters) exceeds the maximum allowed limit of {max_length} characters."
        )
        self.length = length
        self.max_length = max_length


class OutputFindingLimitExceededError(OutputScanError, ValueError):
    """Raised when the number of detected findings exceeds the configured maximum limit.

    IMPORTANT PRIVACY INVARIANT:
    To prevent data leakage, this exception message only reports finding counts
    and limits, never matched values, text content, or secrets.
    """

    def __init__(self, count: int, max_count: int) -> None:
        super().__init__(
            f"Detected finding count ({count}) exceeds the maximum allowed limit of {max_count} findings."
        )
        self.count = count
        self.max_count = max_count


@dataclass(frozen=True)
class OutputFinding:
    """Deterministic security finding in plain-text output.

    IMPORTANT PRIVACY INVARIANT:
    This data structure NEVER contains the raw matched secret, plain-text token,
    or key material. It stores only non-sensitive categorization, rule metadata,
    and character offsets.
    """

    rule_id: str
    category: str
    severity: OutputSeverity
    start: int
    end: int
    description: str
    is_normalized: bool = False

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start offset must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"end offset ({self.end}) must be strictly greater than start offset ({self.start})"
            )

    @property
    def span(self) -> Tuple[int, int]:
        """Return the (start, end) character tuple."""
        return (self.start, self.end)

    @property
    def length(self) -> int:
        """Return the span character length."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Serialize finding to dictionary format without sensitive data."""
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity,
            "start": self.start,
            "end": self.end,
            "description": self.description,
        }


@dataclass(frozen=True)
class OutputDetectionRule:
    """Metadata describing an output security rule."""

    rule_id: str
    name: str
    category: OutputCategory
    severity: OutputSeverity
    description: str


# Configured rule catalog
CONFIGURED_OUTPUT_RULES: List[OutputDetectionRule] = [
    OutputDetectionRule(
        rule_id=DLP_RULE_PRIVATE_KEY_HEADER,
        name="Private Key Header Detection",
        category="PRIVATE_KEY",
        severity="CRITICAL",
        description=(
            "Detects standard PEM/OpenSSH private key header blocks (RSA, EC, DSA, generic, and OpenSSH). "
            "Header detection only; does not perform cryptographic validation."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_AWS_ACCESS_KEY_ID,
        name="AWS Access Key ID Detection",
        category="CLOUD_CREDENTIAL",
        severity="HIGH",
        description=(
            "Detects 20-character AWS IAM access key ID format (AKIA followed by 16 uppercase alphanumeric characters). "
            "Format match only; does not verify whether the credential is active or valid in AWS."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_OPENAI_API_KEY,
        name="OpenAI-Style API Key Detection",
        category="API_KEY",
        severity="HIGH",
        description=(
            "Detects OpenAI-style synthetic key patterns (sk- or sk-proj- followed by 20+ alphanumeric characters). "
            "Pattern heuristic only; does not verify key validity or active account status."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_BEARER_TOKEN,
        name="Bearer Token Detection",
        category="BEARER_TOKEN",
        severity="HIGH",
        description=(
            "Detects plausible token strings following a Bearer authorization scheme. "
            "Excludes documentation placeholders such as <TOKEN> and [REDACTED]."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_DATABASE_URI_CREDENTIAL,
        name="Credential-Bearing Database URI Detection",
        category="DATABASE_CREDENTIAL",
        severity="CRITICAL",
        description=(
            "Detects database connection URIs (PostgreSQL, MySQL, MongoDB, Redis) containing embedded user:password credentials. "
            "Format match only; does not test network reachability or authentication."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_PAYMENT_CARD,
        name="Payment Card Candidate Detection",
        category="PAYMENT_CARD",
        severity="CRITICAL",
        description=(
            "Detects bounded 13-19 digit candidate numbers with valid Luhn checksums. "
            "Checksum validation only; does not prove the number corresponds to an issued or active card."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_US_SSN,
        name="US Social Security Number Candidate Detection",
        category="GOVERNMENT_ID",
        severity="HIGH",
        description=(
            "Detects US Social Security Number candidates in XXX-XX-XXXX format with invalid area, group, and serial ranges excluded. "
            "Format match only; does not verify whether the number is assigned to an individual."
        ),
    ),
    OutputDetectionRule(
        rule_id=DLP_RULE_SECRET_ASSIGNMENT,
        name="Credential and Secret Assignment Detection",
        category="SECRET_ASSIGNMENT",
        severity="HIGH",
        description=(
            "Detects direct password, client_secret, or api_key assignment patterns. "
            "Potential false positives in configuration templates and documentation examples."
        ),
    ),
]


def get_configured_output_rules() -> List[OutputDetectionRule]:
    """Return the list of configured output detection rules."""
    return list(CONFIGURED_OUTPUT_RULES)


# ---------------------------------------------------------------------------
# Documented Exact Placeholders (Narrowly Defined)
# ---------------------------------------------------------------------------
# Strict canonical examples published in official documentation/RFCs.
# A candidate is ONLY treated as a placeholder when the complete value matches
# one of these exact entries.
EXACT_AWS_DOCUMENTATION_PLACEHOLDERS: Set[str] = {
    "AKIAIOSFODNN7EXAMPLE",  # Official AWS IAM documentation example
}

EXACT_OPENAI_DOCUMENTATION_PLACEHOLDERS: Set[str] = {
    "sk-00000000000000000000000000000000",
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-your-api-key-here-1234567890",
}

EXACT_BEARER_PLACEHOLDERS: Set[str] = {
    "YOUR_TOKEN_HERE",
    "YOUR_ACCESS_TOKEN_HERE",
    "YOUR_API_TOKEN_HERE",
    "INSERT_TOKEN_HERE",
    "PLACEHOLDER_TOKEN_HERE",
    "000000000000000000000000",
}

EXACT_SECRET_ASSIGNMENT_PLACEHOLDERS: Set[str] = {
    "your_password",
    "your-password",
    "your_secret",
    "your_secret_here",
    "your_api_key",
    "your_token",
    "<password>",
    "[password]",
    "<secret>",
    "[secret]",
    "[redacted]",
    "<redacted>",
    "placeholder",
    "changeme",
    "change_me",
    "example",
    "todo",
    "none",
    "null",
    "true",
    "false",
    "******",
    "***",
    "xxxxxx",
    "xxxxxxxx",
}


# ---------------------------------------------------------------------------
# Compiled Bounded Regular Expressions
# ---------------------------------------------------------------------------
# 1. Private Key Header: matches PEM / OpenSSH private key header lines
PATTERN_PRIVATE_KEY_HEADER = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
    re.IGNORECASE,
)

# 2. AWS Access Key ID: AKIA prefix + 16 uppercase alphanumeric chars
PATTERN_AWS_KEY = re.compile(r"\b(AKIA[0-9A-Z]{16})\b")

# 3. OpenAI-Style API Key: sk- or sk-proj- prefix + 20 to 100 safe token characters (scoped case-insensitive prefix)
PATTERN_OPENAI_KEY = re.compile(r"\b((?i:sk-(?:proj-)?)[a-zA-Z0-9_\-]{20,100})\b")

# 4. Bearer Token: Bearer scheme followed by 20 to 256 plausible token characters (scoped case-insensitive scheme)
PATTERN_BEARER_TOKEN = re.compile(r"\b(?i:bearer)\s+([a-zA-Z0-9_\-\.]{20,256})\b")

# 5. Database URI: Supported schemes with embedded user:password credentials
PATTERN_DB_URI_CREDENTIAL = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?):\/\/[^\s/:@]*:[^\s/@]+@[^\s/:]+(?::[0-9]{1,5})?(?:\/[^\s\"'>]*)?",
    re.IGNORECASE,
)

# 6. Payment Card Candidates: 13-19 digit candidates (contiguous or standard dash/space groups)
PATTERN_PAYMENT_CARD = re.compile(
    r"\b(?:\d{4}[ -]\d{6}[ -]\d{5}|\d{4}(?:[ -]\d{4}){2,3}(?:[ -]\d{1,4})?|\d{13,19})\b"
)

# 7. US SSN Candidates: XXX-XX-XXXX with excluded invalid area (000, 666, 900-999), group (00), serial (0000)
PATTERN_US_SSN = re.compile(
    r"\b(?!000|666|9\d{2})(\d{3})-(?!00)(\d{2})-(?!0000)(\d{4})\b"
)

# 8. Password/Secret Assignment: conservative keyword=... / keyword: ... patterns
PATTERN_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|pwd|client_secret|secret_key|api_key)\s*(?:=|:|:=)\s*"
    r"(?:(['\"])([^'\"\r\n]{4,128})\2|([a-zA-Z0-9_\-!@#$%^&*+=.]{6,128}))"
)


# ---------------------------------------------------------------------------
# Luhn Algorithm Checksum
# ---------------------------------------------------------------------------
def luhn_checksum_valid(number_str: str) -> bool:
    """Validate candidate payment card number using the Luhn mod 10 algorithm.

    IMPORTANT STATISTICAL AND COMPLIANCE NOTICE:
    Passing the Luhn checksum indicates only that the sequence of digits conforms
    to the ISO/IEC 7812 check digit specification. It does NOT prove that the number
    corresponds to an issued, active, funded, or genuine payment card.

    Args:
        number_str: Candidate number string possibly containing hyphens or spaces.

    Returns:
        True if valid Luhn checksum and within valid digit boundaries; False otherwise.
    """
    digits = [int(c) for c in number_str if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    if digits[0] == 0:  # Major Industry Identifier (MII) 0 is not used for payment cards
        return False
    if len(set(digits)) == 1:  # Repeating single digit sequences (e.g., all 1s)
        return False

    total = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Internal Detection Scanner Helpers
# ---------------------------------------------------------------------------
def _scan_private_key_headers(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_PRIVATE_KEY_HEADER.finditer(text):
        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_PRIVATE_KEY_HEADER,
                category="PRIVATE_KEY",
                severity="CRITICAL",
                start=match.start(),
                end=match.end(),
                description="Private key header detected (format check, not cryptographic validation).",
            )
        )
    return findings


def _scan_aws_keys(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_AWS_KEY.finditer(text):
        key_value = match.group(1)
        if key_value in EXACT_AWS_DOCUMENTATION_PLACEHOLDERS:
            continue
        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_AWS_ACCESS_KEY_ID,
                category="CLOUD_CREDENTIAL",
                severity="HIGH",
                start=match.start(),
                end=match.end(),
                description="AWS access key ID candidate detected (format check, does not verify active status).",
            )
        )
    return findings


def _scan_openai_keys(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_OPENAI_KEY.finditer(text):
        token_candidate = match.group(1)
        if (
            token_candidate in EXACT_OPENAI_DOCUMENTATION_PLACEHOLDERS
            or token_candidate.lower() in EXACT_OPENAI_DOCUMENTATION_PLACEHOLDERS
        ):
            continue
        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_OPENAI_API_KEY,
                category="API_KEY",
                severity="HIGH",
                start=match.start(),
                end=match.end(),
                description="OpenAI-style API key candidate detected (pattern check, does not prove key is genuine).",
            )
        )
    return findings


def _scan_bearer_tokens(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_BEARER_TOKEN.finditer(text):
        token_value = match.group(1)
        if (
            token_value in EXACT_BEARER_PLACEHOLDERS
            or token_value.upper() in EXACT_BEARER_PLACEHOLDERS
        ):
            continue
        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_BEARER_TOKEN,
                category="BEARER_TOKEN",
                severity="HIGH",
                start=match.start(),
                end=match.end(),
                description="Bearer authentication token detected (pattern check, does not verify token validity).",
            )
        )
    return findings


def _scan_db_uris(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_DB_URI_CREDENTIAL.finditer(text):
        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_DATABASE_URI_CREDENTIAL,
                category="DATABASE_CREDENTIAL",
                severity="CRITICAL",
                start=match.start(),
                end=match.end(),
                description="Credential-bearing database URI detected (format check, does not verify database connectivity).",
            )
        )
    return findings


def _scan_payment_cards(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_PAYMENT_CARD.finditer(text):
        candidate = match.group(0)
        if luhn_checksum_valid(candidate):
            findings.append(
                OutputFinding(
                    rule_id=DLP_RULE_PAYMENT_CARD,
                    category="PAYMENT_CARD",
                    severity="CRITICAL",
                    start=match.start(),
                    end=match.end(),
                    description="Payment card candidate detected via Luhn check (does not prove card is issued or active).",
                )
            )
    return findings


def _scan_us_ssn(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_US_SSN.finditer(text):
        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_US_SSN,
                category="GOVERNMENT_ID",
                severity="HIGH",
                start=match.start(),
                end=match.end(),
                description="US Social Security Number candidate detected (format check, does not verify issued status).",
            )
        )
    return findings


def _scan_secret_assignments(text: str) -> List[OutputFinding]:
    findings: List[OutputFinding] = []
    for match in PATTERN_SECRET_ASSIGNMENT.finditer(text):
        # Extract quoted or unquoted value
        raw_val = match.group(3) if match.group(3) is not None else match.group(4)
        if not raw_val:
            continue
        cleaned_val = raw_val.strip()
        val_lower = cleaned_val.lower()

        # Check exact placeholders
        if val_lower in EXACT_SECRET_ASSIGNMENT_PLACEHOLDERS:
            continue
        # Check angle/square bracket placeholders like <TOKEN> or [REDACTED]
        if (cleaned_val.startswith("<") and cleaned_val.endswith(">")) or (
            cleaned_val.startswith("[") and cleaned_val.endswith("]")
        ):
            continue

        findings.append(
            OutputFinding(
                rule_id=DLP_RULE_SECRET_ASSIGNMENT,
                category="SECRET_ASSIGNMENT",
                severity="HIGH",
                start=match.start(),
                end=match.end(),
                description="Sensitive credential assignment pattern detected (format check, potential false positive in configuration templates).",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Deduplication, Overlap Resolution, and Ordering
# ---------------------------------------------------------------------------
SEVERITY_PRIORITY: Dict[OutputSeverity, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}

RULE_SPECIFICITY: Dict[str, int] = {
    DLP_RULE_PRIVATE_KEY_HEADER: 10,
    DLP_RULE_DATABASE_URI_CREDENTIAL: 9,
    DLP_RULE_PAYMENT_CARD: 8,
    DLP_RULE_AWS_ACCESS_KEY_ID: 7,
    DLP_RULE_OPENAI_API_KEY: 6,
    DLP_RULE_BEARER_TOKEN: 5,
    DLP_RULE_US_SSN: 4,
    DLP_RULE_SECRET_ASSIGNMENT: 1,  # Generic assignment has lowest specificity
}


def spans_overlap(f1: OutputFinding, f2: OutputFinding) -> bool:
    """Return True if two findings have overlapping character intervals."""
    return max(f1.start, f2.start) < min(f1.end, f2.end)


def finding_priority_key(f: OutputFinding) -> Tuple[int, int, int, int, str]:
    """Strict total priority key for deterministic finding selection.

    Documented priority hierarchy:
    1. Higher severity first (CRITICAL > HIGH > MEDIUM > LOW)
    2. Higher rule specificity next (specific credential detectors > generic assignments)
    3. Longer span next (wider matched boundary)
    4. Earlier start offset next (earlier in document)
    5. Rule ID as final deterministic tie-breaker (alphabetical)
    """
    return (
        -SEVERITY_PRIORITY.get(f.severity, 0),
        -RULE_SPECIFICITY.get(f.rule_id, 0),
        -f.length,
        f.start,
        f.rule_id,
    )


def deduplicate_and_order_findings(
    findings: Sequence[OutputFinding],
    text_length: int,
    resolve_overlaps: bool = True,
    max_findings: int = MAX_FINDINGS_COUNT,
) -> List[OutputFinding]:
    """Deterministically order, deduplicate, and resolve overlapping findings.

    Priority Selection Strategy for Overlapping Spans:
    1. Sort all candidates according to the strict priority key:
       - Higher severity first
       - Higher rule specificity next
       - Longer span next
       - Earlier start offset next
       - Rule ID as final deterministic tie-breaker
    2. Greedily select non-overlapping candidates:
       - Process candidates in priority order.
       - A candidate is accepted if it does not overlap with any previously accepted finding.
       - If it overlaps with any previously accepted finding, it is discarded because the
         conflicting accepted finding has strictly higher priority.
    3. Final Ordering:
       - Sort the selected non-overlapping findings by start offset ascending for the public result.
    4. Strict Finding-Limit Enforcement:
       - If the finding count exceeds max_findings, raises OutputFindingLimitExceededError.
         Never silently truncates results.

    Args:
        findings: Sequence of raw findings.
        text_length: Length of original input string.
        resolve_overlaps: If True, resolves overlapping spans into disjoint findings.
        max_findings: Maximum allowed findings count (default: 100).

    Returns:
        List of deterministically ordered and bounded findings.

    Raises:
        ValueError: If any span is out of bounds [0, text_length].
        OutputFindingLimitExceededError: If finding count exceeds max_findings.
    """
    if not findings:
        return []

    # 1. Validate raw spans
    for f in findings:
        if not (0 <= f.start < f.end <= text_length):
            raise ValueError(
                f"Invalid finding span ({f.start}, {f.end}) for text of length {text_length}"
            )

    # 2. Check early bound to prevent unbounded processing on adversarial payloads
    raw_ceiling = max(max_findings * 5, 500)
    if len(findings) > raw_ceiling:
        raise OutputFindingLimitExceededError(count=len(findings), max_count=max_findings)

    if not resolve_overlaps:
        # Exact deduplication only
        seen: Set[Tuple[int, int, str]] = set()
        exact_deduped: List[OutputFinding] = []
        for f in sorted(findings, key=lambda x: (x.start, x.end, x.rule_id)):
            key = (f.start, f.end, f.rule_id)
            if key not in seen:
                seen.add(key)
                exact_deduped.append(f)
        if len(exact_deduped) > max_findings:
            raise OutputFindingLimitExceededError(
                count=len(exact_deduped), max_count=max_findings
            )
        return exact_deduped

    # 3. Priority-based interval selection
    sorted_by_priority = sorted(findings, key=finding_priority_key)
    selected: List[OutputFinding] = []
    for candidate in sorted_by_priority:
        if not any(spans_overlap(candidate, chosen) for chosen in selected):
            selected.append(candidate)
            # Early check if selected exceeds max_findings
            if len(selected) > max_findings:
                raise OutputFindingLimitExceededError(
                    count=len(selected), max_count=max_findings
                )

    if len(selected) > max_findings:
        raise OutputFindingLimitExceededError(
            count=len(selected), max_count=max_findings
        )

    # 4. Sort selected findings by start offset ascending for public consumption
    selected.sort(key=lambda f: (f.start, f.end, f.rule_id))

    # 5. Guarantee disjoint spans
    for i in range(len(selected) - 1):
        assert selected[i].end <= selected[i + 1].start, "Resolved spans must be strictly disjoint"

    return selected


# ---------------------------------------------------------------------------
# Public Scanner Interface
# ---------------------------------------------------------------------------
def scan_output_text(
    text: str,
    max_length: int = MAX_OUTPUT_TEXT_LENGTH,
    resolve_overlaps: bool = True,
    max_findings: int = MAX_FINDINGS_COUNT,
) -> List[OutputFinding]:
    """Scan plain-text downstream output for deterministic DLP findings.

    Args:
        text: Plain-text string to scan.
        max_length: Maximum allowed character length (default: 25,000).
        resolve_overlaps: If True, resolves overlapping spans deterministically (default: True).
        max_findings: Maximum allowed findings count (default: 100).

    Returns:
        List of OutputFinding instances, deterministically ordered and bounded.

    Raises:
        TypeError: If input is not a string.
        OutputLengthExceededError: If input length exceeds max_length characters.
        OutputFindingLimitExceededError: If finding count exceeds max_findings.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")

    text_len = len(text)
    if text_len > max_length:
        raise OutputLengthExceededError(length=text_len, max_length=max_length)

    # Empty or whitespace-only text produces no findings safely
    if not text.strip():
        return []

    raw_findings: List[OutputFinding] = []

    # Run all 8 deterministic detectors
    raw_findings.extend(_scan_private_key_headers(text))
    raw_findings.extend(_scan_aws_keys(text))
    raw_findings.extend(_scan_openai_keys(text))
    raw_findings.extend(_scan_bearer_tokens(text))
    raw_findings.extend(_scan_db_uris(text))
    raw_findings.extend(_scan_payment_cards(text))
    raw_findings.extend(_scan_us_ssn(text))
    raw_findings.extend(_scan_secret_assignments(text))

    return deduplicate_and_order_findings(
        raw_findings,
        text_length=text_len,
        resolve_overlaps=resolve_overlaps,
        max_findings=max_findings,
    )
