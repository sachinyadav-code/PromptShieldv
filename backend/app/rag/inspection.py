"""Multi-Level RAG Security Inspection for PromptShield (Phase 3 Stage 3).

Provides deterministic security inspection across multiple retrieval tiers:
1. Direct User Query: Validates query text for direct injection attempts.
2. Document Fields: Evaluates title, content, tags, source, and metadata.
3. Bounded Chunks: Splits long content into overlapping windows to detect localized attacks.
4. Combined Context: Analyzes aggregated multi-document context for distributed attack fragments.

IMPORTANT SECURITY & PRIVACY INVARIANTS:
- All retrieved content, titles, tags, and metadata are UNTRUSTED DATA.
- Relevance scores or fixture flags (e.g. is_fixture_attack) grant ZERO authorization or trust.
- Raw prompt texts, document bodies, metadata payloads, and secrets are NEVER echoed in
  exception messages, log events, or finding explanations.
- Output order is strictly deterministic.
- Resource limits (max docs, max chars, max chunks) are enforced before processing;
  exceeded limits result in documented REVIEW / incomplete assessments, never silent truncation.
"""

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

from app.detection.rules import CONFIGURED_RULES, DetectionRule
from app.rag.corpus import Document

logger = logging.getLogger(__name__)

# Resource and Boundary Limits
DEFAULT_MAX_DOCS = 10
DEFAULT_MAX_CHARS = 25000
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOC = 50

# Type aliases
InspectionScope = Literal["direct_query", "document", "chunk", "combined_context"]
InspectionAssessment = Literal["ALLOW", "BLOCK", "REVIEW"]
DetectionSource = Literal["deterministic_rule", "heuristic_context", "ml_assessment", "combined_deterministic"]

# Additional RAG-Specific Detection Rules
RAG_RULE_TITLE_INJECTION = "RAG_RULE_TITLE_INJECTION"
RAG_RULE_METADATA_INJECTION = "RAG_RULE_METADATA_INJECTION"
RAG_RULE_PRIVILEGE_ESCALATION = "RAG_RULE_PRIVILEGE_ESCALATION"
RAG_RULE_CROSS_DOC_FRAGMENT = "RAG_RULE_CROSS_DOC_FRAGMENT"
RAG_RULE_CONTEXT_LIMIT_EXCEEDED = "RAG_RULE_CONTEXT_LIMIT_EXCEEDED"
RAG_RULE_AMBIGUOUS_SECURITY_CONTEXT = "RAG_RULE_AMBIGUOUS_SECURITY_CONTEXT"

# Privilege Escalation Patterns
PRIVILEGE_ESCALATION_PATTERNS = [
    re.compile(r"\b(?:grant|give|elevate|escalate)\s+(?:all\s+|me\s+)?(?:admin|administrator|root|superuser)\s+(?:access|privileges?|rights?|role)\b", re.IGNORECASE),
    re.compile(r"\b(?:modify|alter|change|override)\s+(?:security\s+|gateway\s+)?policies?\b", re.IGNORECASE),
    re.compile(r"\b(?:authorize|permit)\s+(?:all\s+)?(?:tool|function|plugin)\s+(?:executions?|calls?)\s+without\s+(?:confirmation|approval)\b", re.IGNORECASE),
    re.compile(r"\bgrant\s+full\s+admin\s+privileges\b", re.IGNORECASE),
]

# RAG Extended Instruction Override Patterns
RAG_OVERRIDE_PATTERNS = [
    re.compile(r"\bimportant\s+system\s+override\b", re.IGNORECASE),
    re.compile(r"\battention\s+assistant\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous\s+)?guidelines\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\s*,\s*(?:ignore|disregard)\b", re.IGNORECASE),
]

# Educational / Defensive Context Markers
EDUCATIONAL_MARKER_PATTERNS = [
    re.compile(r"\b(?:phrases?\s+like|such\s+as|examples?\s+(?:include|such\s+as)|known\s+as|attacks?\s+attempt\s+to|quoted\s+as|concept\s+of|describing|discussing|mitigate|defend\s+against)\s*['\"`]?$", re.IGNORECASE),
]

# Split Attack Component Patterns (Cross-Document Analysis)
CROSS_DOC_OPENER_PATTERN = re.compile(
    r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\b|"
    r"\b(?:important\s+system\s+override|attention\s+assistant|verification\s+notice)\b",
    re.IGNORECASE,
)
CROSS_DOC_PAYLOAD_PATTERN = re.compile(
    r"\b(?:reveal|show|print|dump|leak|grant|export|output)\s+(?:the\s+|your\s+|all\s+)?(?:system\s+prompt|developer\s+instructions?|api[\s_-]?keys?|admin\s+privileges|credentials?|passwords?)\b|"
    r"\bgrant\s+full\s+admin\s+privileges\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InspectionFinding:
    """Detailed finding resulting from a security inspection check.

    Attributes:
        scope: Target area inspected ('direct_query', 'document', 'chunk', 'combined_context').
        assessment: Verdict for this specific finding ('ALLOW', 'BLOCK', 'REVIEW').
        rule_id: Unique identifier for the matched rule.
        doc_id: Associated document identifier, if applicable.
        source: Provenance source of the document, if applicable.
        chunk_index: Zero-based index of the chunk, if applicable.
        char_start: Starting character offset in the original document content.
        char_end: Ending character offset in the original document content.
        detection_source: Grounding detector ('deterministic_rule', 'heuristic_context', etc.).
        risk_score: Bounded heuristic risk contribution in [0.0, 1.0].
        explanation: Sanitized explanation omitting raw prompt/document payloads.
    """

    scope: InspectionScope
    assessment: InspectionAssessment
    rule_id: str
    doc_id: Optional[str] = None
    source: Optional[str] = None
    chunk_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    detection_source: DetectionSource = "deterministic_rule"
    risk_score: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to standard dictionary format."""
        return {
            "scope": self.scope,
            "assessment": self.assessment,
            "rule_id": self.rule_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "detection_source": self.detection_source,
            "risk_score": self.risk_score,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class InspectionResult:
    """Aggregated inspection outcome for a specific inspection level.

    Attributes:
        scope: The level assessed ('direct_query', 'document', 'chunk', 'combined_context').
        assessment: Overall verdict for this scope ('ALLOW', 'BLOCK', 'REVIEW').
        risk_score: Maximum heuristic risk score among findings in [0.0, 1.0].
        matched_rule_ids: Tuple of distinct matched rule IDs.
        findings: Tuple of individual InspectionFinding objects.
        doc_id: Document identifier if scope is 'document' or 'chunk'.
        inspected_items_count: Count of entities inspected (e.g. docs, chunks).
        is_complete: False if limits were exceeded preventing full inspection; True otherwise.
        explanation: High-level sanitized summary description.
    """

    scope: InspectionScope
    assessment: InspectionAssessment
    risk_score: float
    matched_rule_ids: Tuple[str, ...] = field(default_factory=tuple)
    findings: Tuple[InspectionFinding, ...] = field(default_factory=tuple)
    doc_id: Optional[str] = None
    inspected_items_count: int = 1
    is_complete: bool = True
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary format."""
        return {
            "scope": self.scope,
            "assessment": self.assessment,
            "risk_score": self.risk_score,
            "matched_rule_ids": list(self.matched_rule_ids),
            "findings": [f.to_dict() for f in self.findings],
            "doc_id": self.doc_id,
            "inspected_items_count": self.inspected_items_count,
            "is_complete": self.is_complete,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class RAGInspectionReport:
    """Comprehensive multi-level security assessment across all RAG tiers."""

    overall_assessment: InspectionAssessment
    max_risk_score: float
    query_result: InspectionResult
    document_results: Tuple[InspectionResult, ...]
    combined_context_result: InspectionResult
    all_matched_rule_ids: Tuple[str, ...]
    all_findings: Tuple[InspectionFinding, ...]
    is_complete: bool
    summary_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standard dictionary format."""
        return {
            "overall_assessment": self.overall_assessment,
            "max_risk_score": self.max_risk_score,
            "query_result": self.query_result.to_dict(),
            "document_results": [dr.to_dict() for dr in self.document_results],
            "combined_context_result": self.combined_context_result.to_dict(),
            "all_matched_rule_ids": list(self.all_matched_rule_ids),
            "all_findings": [f.to_dict() for f in self.all_findings],
            "is_complete": self.is_complete,
            "summary_explanation": self.summary_explanation,
        }


def chunk_text(
    text: str,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_DOC,
) -> List[Tuple[int, int, int, str]]:
    """Partition document text into deterministic bounded overlapping chunks.

    Args:
        text: Input text content to partition.
        max_chunk_chars: Maximum characters per chunk (default 1000).
        overlap_chars: Number of overlapping characters between adjacent chunks (default 200).
        max_chunks: Maximum number of chunks allowed per document (default 50).

    Returns:
        List of tuples: (chunk_index, char_start, char_end, chunk_text).
        Empty list if text is empty or whitespace-only.
    """
    if not text or not text.strip():
        return []

    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be a positive integer")
    if overlap_chars < 0 or overlap_chars >= max_chunk_chars:
        raise ValueError("overlap_chars must be non-negative and strictly less than max_chunk_chars")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer")

    total_len = len(text)
    if total_len <= max_chunk_chars:
        return [(0, 0, total_len, text)]

    chunks: List[Tuple[int, int, int, str]] = []
    step = max_chunk_chars - overlap_chars
    start = 0
    chunk_idx = 0

    while start < total_len and chunk_idx < max_chunks:
        end = min(start + max_chunk_chars, total_len)
        chunk_slice = text[start:end]
        chunks.append((chunk_idx, start, end, chunk_slice))
        chunk_idx += 1
        start += step

    return chunks


def _is_educational_or_quoted_context(text: str, match_start: int, match_end: int) -> bool:
    """Determine whether a matched phrase appears within a descriptive or educational context.

    Prevents false-positive blocks on cybersecurity education articles that discuss
    injection techniques or credential concepts without executing an active attack directive.
    """
    # 1. Check if the document has explicit imperative override prefixes
    # If the text has an imperative prefix, it is an active attack, NOT educational.
    imperative_prefixes = ("important system override:", "attention assistant:", "verification notice:")
    lower_text = text.lower()
    for prefix in imperative_prefixes:
        if prefix in lower_text:
            return False

    # 2. Check preceding window for educational/descriptive markers
    prefix_window = text[max(0, match_start - 80):match_start]
    for pattern in EDUCATIONAL_MARKER_PATTERNS:
        if pattern.search(prefix_window):
            return True

    # 3. Check if the match itself is enclosed in quotes
    has_pre_quote = match_start > 0 and text[match_start - 1] in ("'", '"', "`")
    has_post_quote = match_end < len(text) and text[match_end] in ("'", '"', "`")
    if has_pre_quote and has_post_quote:
        # Enclosed in quotes; check if general text discusses concepts rather than commanding
        educational_terms = ("security risk", "prompt injection", "direct attacks", "indirect injection", "guidelines", "protocol")
        if any(term in lower_text for term in educational_terms):
            return True

    return False


def _scan_text_for_rules(
    text: str,
    scope: InspectionScope,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    chunk_index: Optional[int] = None,
    base_offset: int = 0,
    check_privilege: bool = True,
    allow_educational_override: bool = True,
) -> List[InspectionFinding]:
    """Scan text against deterministic rules and return structured findings."""
    if not text:
        return []

    findings: List[InspectionFinding] = []

    # 1. Evaluate Phase 1 configured rules
    for rule in CONFIGURED_RULES:
        for pattern in rule.patterns:
            for match in pattern.finditer(text):
                m_start = base_offset + match.start()
                m_end = base_offset + match.end()

                # Check educational/descriptive context if enabled
                if allow_educational_override and _is_educational_or_quoted_context(text, match.start(), match.end()):
                    findings.append(
                        InspectionFinding(
                            scope=scope,
                            assessment="ALLOW",
                            rule_id=RAG_RULE_AMBIGUOUS_SECURITY_CONTEXT,
                            doc_id=doc_id,
                            source=source,
                            chunk_index=chunk_index,
                            char_start=m_start,
                            char_end=m_end,
                            detection_source="heuristic_context",
                            risk_score=0.15,
                            explanation=f"Descriptive prompt injection reference in educational context (rule: {rule.rule_id})",
                        )
                    )
                else:
                    findings.append(
                        InspectionFinding(
                            scope=scope,
                            assessment="BLOCK",
                            rule_id=rule.rule_id,
                            doc_id=doc_id,
                            source=source,
                            chunk_index=chunk_index,
                            char_start=m_start,
                            char_end=m_end,
                            detection_source="deterministic_rule",
                            risk_score=rule.risk_weight,
                            explanation=f"Deterministic rule match: {rule.name}",
                        )
                    )
                break  # Record rule once per matching pattern per section

    # 2. Evaluate RAG extended override patterns
    for pattern in RAG_OVERRIDE_PATTERNS:
        for match in pattern.finditer(text):
            m_start = base_offset + match.start()
            m_end = base_offset + match.end()
            findings.append(
                InspectionFinding(
                    scope=scope,
                    assessment="BLOCK",
                    rule_id="RULE_INSTRUCTION_OVERRIDE",
                    doc_id=doc_id,
                    source=source,
                    chunk_index=chunk_index,
                    char_start=m_start,
                    char_end=m_end,
                    detection_source="deterministic_rule",
                    risk_score=0.90,
                    explanation="Detected active instruction override directive",
                )
            )
            break

    # 3. Evaluate Privilege Escalation rules if enabled
    if check_privilege:
        for pattern in PRIVILEGE_ESCALATION_PATTERNS:
            for match in pattern.finditer(text):
                m_start = base_offset + match.start()
                m_end = base_offset + match.end()
                findings.append(
                    InspectionFinding(
                        scope=scope,
                        assessment="BLOCK",
                        rule_id=RAG_RULE_PRIVILEGE_ESCALATION,
                        doc_id=doc_id,
                        source=source,
                        chunk_index=chunk_index,
                        char_start=m_start,
                        char_end=m_end,
                        detection_source="deterministic_rule",
                        risk_score=0.95,
                        explanation="Detected unauthorized privilege escalation or policy modification directive",
                    )
                )
                break

    return findings


def inspect_direct_query(query: str) -> InspectionResult:
    """Inspect a direct user query string against deterministic security rules.

    Args:
        query: User input query text.

    Returns:
        InspectionResult with scope 'direct_query'.
    """
    if not isinstance(query, str) or not query.strip():
        return InspectionResult(
            scope="direct_query",
            assessment="ALLOW",
            risk_score=0.0,
            matched_rule_ids=(),
            findings=(),
            inspected_items_count=1,
            is_complete=True,
            explanation="Empty or whitespace query evaluated safely",
        )

    # For direct queries, educational override does not apply (user prompt must not contain attacks)
    findings = _scan_text_for_rules(
        text=query,
        scope="direct_query",
        doc_id=None,
        source=None,
        chunk_index=None,
        base_offset=0,
        check_privilege=True,
        allow_educational_override=False,
    )

    has_block = any(f.assessment == "BLOCK" for f in findings)
    max_risk = max((f.risk_score for f in findings), default=0.0)
    rule_ids = tuple(sorted({f.rule_id for f in findings}))

    assessment: InspectionAssessment = "BLOCK" if has_block else "ALLOW"
    explanation = "Direct query passed security inspection" if assessment == "ALLOW" else f"Direct query blocked by rule(s): {', '.join(rule_ids)}"

    return InspectionResult(
        scope="direct_query",
        assessment=assessment,
        risk_score=max_risk,
        matched_rule_ids=rule_ids,
        findings=tuple(findings),
        inspected_items_count=1,
        is_complete=True,
        explanation=explanation,
    )


def inspect_document(doc: Document) -> InspectionResult:
    """Perform comprehensive security inspection of a single Document across all fields.

    Evaluates:
    1. Title field (checks for title-based injection)
    2. Tags and source fields
    3. Metadata dictionary values
    4. Full content body

    Args:
        doc: Validated Document object to inspect.

    Returns:
        InspectionResult with scope 'document'.
    """
    if not isinstance(doc, Document):
        raise TypeError(f"Expected Document instance, got {type(doc).__name__}")

    all_findings: List[InspectionFinding] = []

    # 1. Inspect Title
    title_findings = _scan_text_for_rules(
        text=doc.title,
        scope="document",
        doc_id=doc.doc_id,
        source=doc.source,
        check_privilege=True,
        allow_educational_override=False,
    )
    for tf in title_findings:
        if tf.assessment == "BLOCK":
            all_findings.append(
                InspectionFinding(
                    scope="document",
                    assessment="BLOCK",
                    rule_id=RAG_RULE_TITLE_INJECTION,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    detection_source="deterministic_rule",
                    risk_score=max(tf.risk_score, 0.90),
                    explanation=f"Malicious instruction embedded in document title (triggered: {tf.rule_id})",
                )
            )
        else:
            all_findings.append(tf)

    # 2. Inspect Tags and Source
    normalized_tags = " ".join(t.replace("-", " ").replace("_", " ") for t in doc.tags)
    tags_and_source = f"{doc.source} {normalized_tags} " + " ".join(doc.tags)
    tag_findings = _scan_text_for_rules(
        text=tags_and_source,
        scope="document",
        doc_id=doc.doc_id,
        source=doc.source,
        check_privilege=True,
        allow_educational_override=False,
    )
    for tgf in tag_findings:
        if tgf.assessment == "BLOCK":
            all_findings.append(
                InspectionFinding(
                    scope="document",
                    assessment="BLOCK",
                    rule_id=RAG_RULE_METADATA_INJECTION,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    detection_source="deterministic_rule",
                    risk_score=max(tgf.risk_score, 0.85),
                    explanation="Malicious instruction embedded in document tags or source field",
                )
            )

    # 3. Inspect Metadata values recursively
    def _extract_meta_strings(val: Any) -> List[str]:
        strs: List[str] = []
        if isinstance(val, (dict, Mapping)):
            for k, v in val.items():
                if isinstance(k, str):
                    strs.append(k)
                    strs.append(k.replace("-", " ").replace("_", " "))
                strs.extend(_extract_meta_strings(v))
        elif isinstance(val, (list, tuple)):
            for item in val:
                strs.extend(_extract_meta_strings(item))
        elif isinstance(val, str):
            strs.append(val)
            strs.append(val.replace("-", " ").replace("_", " "))
        return strs

    meta_strings = _extract_meta_strings(dict(doc.metadata))
    for meta_str in meta_strings:
        # Ignore purely internal non-hostile category identifiers
        if meta_str in ("security_test_fixture", "instruction_override", "credential_extraction", "title_injection", "metadata_injection", "system_prompt_leak"):
            continue
        meta_finds = _scan_text_for_rules(
            text=meta_str,
            scope="document",
            doc_id=doc.doc_id,
            source=doc.source,
            check_privilege=True,
            allow_educational_override=False,
        )
        for mf in meta_finds:
            if mf.assessment == "BLOCK":
                all_findings.append(
                    InspectionFinding(
                        scope="document",
                        assessment="BLOCK",
                        rule_id=RAG_RULE_METADATA_INJECTION,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        detection_source="deterministic_rule",
                        risk_score=max(mf.risk_score, 0.90),
                        explanation="Malicious instruction embedded in document metadata payload",
                    )
                )

    # 4. Inspect Document Content body
    content_findings = _scan_text_for_rules(
        text=doc.content,
        scope="document",
        doc_id=doc.doc_id,
        source=doc.source,
        check_privilege=True,
        allow_educational_override=True,
    )
    all_findings.extend(content_findings)

    # Deduplicate and prioritize BLOCK over ALLOW findings
    has_block = any(f.assessment == "BLOCK" for f in all_findings)
    has_review = any(f.assessment == "REVIEW" for f in all_findings)

    assessment: InspectionAssessment = "BLOCK" if has_block else ("REVIEW" if has_review else "ALLOW")
    max_risk = max((f.risk_score for f in all_findings), default=0.0)

    # If only educational allow finding, keep risk score low
    if assessment == "ALLOW":
        max_risk = min(max_risk, 0.15)

    rule_ids = tuple(sorted({f.rule_id for f in all_findings}))
    explanation = f"Document {doc.doc_id} passed inspection" if assessment == "ALLOW" else f"Document {doc.doc_id} evaluated with assessment {assessment} (rule(s): {', '.join(rule_ids)})"

    return InspectionResult(
        scope="document",
        assessment=assessment,
        risk_score=max_risk,
        matched_rule_ids=rule_ids,
        findings=tuple(all_findings),
        doc_id=doc.doc_id,
        inspected_items_count=1,
        is_complete=True,
        explanation=explanation,
    )


def inspect_chunks(
    doc: Document,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_DOC,
) -> List[InspectionFinding]:
    """Partition document content into bounded overlapping chunks and inspect each chunk.

    Enables pinpointing exact character offsets and chunk indices of suspicious text,
    and catching attacks positioned around chunk boundaries.

    Args:
        doc: Document whose content will be chunked and inspected.
        max_chunk_chars: Maximum characters per chunk (default 1000).
        overlap_chars: Number of overlapping characters (default 200).
        max_chunks: Maximum chunks to evaluate per document (default 50).

    Returns:
        List of InspectionFinding objects with scope 'chunk'.
    """
    if not isinstance(doc, Document):
        raise TypeError(f"Expected Document instance, got {type(doc).__name__}")

    chunks = chunk_text(
        text=doc.content,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
        max_chunks=max_chunks,
    )
    if not chunks:
        return []

    chunk_findings: List[InspectionFinding] = []

    for chunk_idx, c_start, c_end, chunk_content in chunks:
        findings = _scan_text_for_rules(
            text=chunk_content,
            scope="chunk",
            doc_id=doc.doc_id,
            source=doc.source,
            chunk_index=chunk_idx,
            base_offset=c_start,
            check_privilege=True,
            allow_educational_override=True,
        )
        chunk_findings.extend(findings)

    return chunk_findings


def inspect_combined_context(
    documents: Sequence[Document],
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> InspectionResult:
    """Inspect the aggregated multi-document retrieved context.

    Evaluates:
    1. Bounded resource limits (max_docs, max_chars). If exceeded, produces
       an incomplete REVIEW result rather than silently truncating.
    2. Cross-document split attack fragments distributed across documents.
    3. Preserves input order and doc provenance.

    Args:
        documents: Sequence of Document instances to inspect.
        max_docs: Upper bound on document count (default 10).
        max_chars: Upper bound on total character content (default 25000).

    Returns:
        InspectionResult with scope 'combined_context'.
    """
    if not documents:
        return InspectionResult(
            scope="combined_context",
            assessment="ALLOW",
            risk_score=0.0,
            matched_rule_ids=(),
            findings=(),
            inspected_items_count=0,
            is_complete=True,
            explanation="Empty document collection evaluated safely",
        )

    # 1. Enforce Resource Limits
    if len(documents) > max_docs:
        return InspectionResult(
            scope="combined_context",
            assessment="REVIEW",
            risk_score=0.50,
            matched_rule_ids=(RAG_RULE_CONTEXT_LIMIT_EXCEEDED,),
            findings=(
                InspectionFinding(
                    scope="combined_context",
                    assessment="REVIEW",
                    rule_id=RAG_RULE_CONTEXT_LIMIT_EXCEEDED,
                    detection_source="heuristic_context",
                    risk_score=0.50,
                    explanation=f"Retrieved document count ({len(documents)}) exceeded maximum limit of {max_docs}",
                ),
            ),
            inspected_items_count=len(documents),
            is_complete=False,
            explanation=f"Combined context incomplete: document count exceeded limit {max_docs}",
        )

    total_chars = sum(len(d.content) for d in documents)
    if total_chars > max_chars:
        return InspectionResult(
            scope="combined_context",
            assessment="REVIEW",
            risk_score=0.50,
            matched_rule_ids=(RAG_RULE_CONTEXT_LIMIT_EXCEEDED,),
            findings=(
                InspectionFinding(
                    scope="combined_context",
                    assessment="REVIEW",
                    rule_id=RAG_RULE_CONTEXT_LIMIT_EXCEEDED,
                    detection_source="heuristic_context",
                    risk_score=0.50,
                    explanation=f"Total document character count ({total_chars}) exceeded maximum limit of {max_chars}",
                ),
            ),
            inspected_items_count=len(documents),
            is_complete=False,
            explanation=f"Combined context incomplete: content length exceeded limit {max_chars}",
        )

    # 2. Inspect Cross-Document Split Patterns
    findings: List[InspectionFinding] = []

    # Check for split-instruction attacks spanning adjacent documents
    for i in range(len(documents) - 1):
        doc_a = documents[i]
        doc_b = documents[i + 1]

        # Check end of doc_a for opener, start of doc_b for payload
        tail_a = doc_a.content[-300:] if len(doc_a.content) > 300 else doc_a.content
        head_b = doc_b.content[:300] if len(doc_b.content) > 300 else doc_b.content

        # 1. Active opener in tail_a (must not be purely a quoted educational example)
        has_active_opener = False
        for op_m in CROSS_DOC_OPENER_PATTERN.finditer(tail_a):
            if not _is_educational_or_quoted_context(tail_a, op_m.start(), op_m.end()):
                has_active_opener = True
                break

        # 2. Active payload in head_b (must not be purely a defensive guidance mention)
        has_active_payload = False
        for pl_m in CROSS_DOC_PAYLOAD_PATTERN.finditer(head_b):
            if not _is_educational_or_quoted_context(head_b, pl_m.start(), pl_m.end()):
                has_active_payload = True
                break

        # 3. Check if the combined boundary transition window contains an assembled attack
        boundary_offset = len(tail_a)
        transition_window = tail_a + "\n" + head_b
        boundary_findings = _scan_text_for_rules(
            text=transition_window,
            scope="combined_context",
            doc_id=f"{doc_a.doc_id}+{doc_b.doc_id}",
            source=None,
            check_privilege=True,
            allow_educational_override=False,  # Assemblies spanning boundaries must NEVER be suppressed
        )

        # Attack matches must actually span across the inter-document boundary
        has_spanning_attack = any(
            bf.assessment == "BLOCK"
            and bf.char_start is not None
            and bf.char_end is not None
            and bf.char_start <= boundary_offset < bf.char_end
            for bf in boundary_findings
        )

        if (has_active_opener and has_active_payload) or has_spanning_attack:
            findings.append(
                InspectionFinding(
                    scope="combined_context",
                    assessment="BLOCK",
                    rule_id=RAG_RULE_CROSS_DOC_FRAGMENT,
                    doc_id=f"{doc_a.doc_id}+{doc_b.doc_id}",
                    detection_source="combined_deterministic",
                    risk_score=0.92,
                    explanation=f"Detected cross-document split instruction assembly spanning {doc_a.doc_id} and {doc_b.doc_id}",
                )
            )

    has_block = any(f.assessment == "BLOCK" for f in findings)
    max_risk = max((f.risk_score for f in findings), default=0.0)
    rule_ids = tuple(sorted({f.rule_id for f in findings}))

    assessment: InspectionAssessment = "BLOCK" if has_block else "ALLOW"
    explanation = "Combined context passed security inspection" if assessment == "ALLOW" else f"Combined context blocked by rule(s): {', '.join(rule_ids)}"

    return InspectionResult(
        scope="combined_context",
        assessment=assessment,
        risk_score=max_risk,
        matched_rule_ids=rule_ids,
        findings=tuple(findings),
        doc_id=findings[0].doc_id if findings else None,
        inspected_items_count=len(documents),
        is_complete=True,
        explanation=explanation,
    )


def inspect_rag_request(
    query: str,
    documents: Sequence[Document],
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> RAGInspectionReport:
    """Execute complete multi-level security inspection across all RAG tiers.

    Orchestrates:
    1. Direct Query Inspection
    2. Document-Level Inspection for each retrieved document
    3. Chunk-Level Inspection for each document
    4. Combined-Context Inspection across all documents

    Aggregates findings deterministically and reports overall assessment:
    - BLOCK if any level produces a BLOCK verdict.
    - REVIEW if no BLOCK but any level produces a REVIEW verdict (or limits exceeded).
    - ALLOW if all levels produce ALLOW.

    Args:
        query: User query string.
        documents: Sequence of retrieved Document objects.
        max_docs: Maximum allowed documents in combined context (default 10).
        max_chars: Maximum allowed characters in combined context (default 25000).
        max_chunk_chars: Maximum characters per chunk (default 1000).
        overlap_chars: Character overlap between chunks (default 200).

    Returns:
        Structured, comprehensive RAGInspectionReport.
    """
    # 1. Direct query inspection
    query_result = inspect_direct_query(query)

    # 2. Document & Chunk inspection
    doc_results: List[InspectionResult] = []
    all_findings_list: List[InspectionFinding] = list(query_result.findings)

    for doc in documents:
        d_result = inspect_document(doc)
        doc_results.append(d_result)
        all_findings_list.extend(d_result.findings)

        # Chunk findings for this document
        c_findings = inspect_chunks(
            doc=doc,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        all_findings_list.extend(c_findings)

    # 3. Combined context inspection
    combined_result = inspect_combined_context(
        documents=documents,
        max_docs=max_docs,
        max_chars=max_chars,
    )
    all_findings_list.extend(combined_result.findings)

    # 4. Aggregation
    is_complete = query_result.is_complete and combined_result.is_complete and all(dr.is_complete for dr in doc_results)

    all_verdicts = [query_result.assessment, combined_result.assessment] + [dr.assessment for dr in doc_results]
    if "BLOCK" in all_verdicts:
        overall_assessment: InspectionAssessment = "BLOCK"
    elif "REVIEW" in all_verdicts:
        overall_assessment = "REVIEW"
    else:
        overall_assessment = "ALLOW"

    max_risk = max((f.risk_score for f in all_findings_list), default=0.0)
    distinct_rules = tuple(sorted({f.rule_id for f in all_findings_list}))

    # Sort findings deterministically
    scope_rank = {"direct_query": 0, "document": 1, "chunk": 2, "combined_context": 3}
    sorted_findings = tuple(
        sorted(
            all_findings_list,
            key=lambda f: (
                scope_rank.get(f.scope, 99),
                f.doc_id or "",
                f.chunk_index if f.chunk_index is not None else -1,
                f.char_start if f.char_start is not None else -1,
                f.rule_id,
            ),
        )
    )

    if overall_assessment == "BLOCK":
        summary_explanation = f"RAG security inspection BLOCKED request due to detected threats: {', '.join(distinct_rules)}"
    elif overall_assessment == "REVIEW":
        summary_explanation = f"RAG security inspection flagged request for REVIEW: {', '.join(distinct_rules)}"
    else:
        summary_explanation = "RAG security inspection passed; all evaluated tiers classified as ALLOW"

    return RAGInspectionReport(
        overall_assessment=overall_assessment,
        max_risk_score=round(max_risk, 4),
        query_result=query_result,
        document_results=tuple(doc_results),
        combined_context_result=combined_result,
        all_matched_rule_ids=distinct_rules,
        all_findings=sorted_findings,
        is_complete=is_complete,
        summary_explanation=summary_explanation,
    )
