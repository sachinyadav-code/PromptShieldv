"""Deterministic keyword and token retriever for PromptShield Phase 3 (RAG).

Provides a transparent, offline keyword-overlap retrieval mechanism over a
validated collection of documents.

IMPORTANT SECURITY NOTICE:
- Retrieval relevance scores measure keyword overlap for ranking only.
- Relevance scores are NOT security scores, safety guarantees, or authorization signals.
- Content retrieved by this module is UNTRUSTED DATA and must never be interpreted
  as instructions to the retriever, gateway, or downstream models.
- Does not make external network requests, load remote databases, or invoke LLMs.
"""

from collections import Counter
from dataclasses import dataclass
import logging
import math
import re
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

from app.rag.corpus import Document

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 1000
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 10
DEFAULT_MIN_RELEVANCE_SCORE = 0.30
MIN_RELEVANCE_SCORE = 0.0
MAX_RELEVANCE_SCORE = 1.0

# Deterministic standard English stopwords for query keyword filtering
DEFAULT_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
})

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_-]+\b")


def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens from text.

    Args:
        text: Input string to tokenize.

    Returns:
        List of lowercase string tokens.
    """
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def validate_top_k(top_k: Any) -> int:
    """Validate top_k parameter strictly.

    Rejects:
    - Booleans (isinstance(top_k, bool)).
    - Non-integer types (floats, strings, None, etc.).
    - Values outside [MIN_TOP_K, MAX_TOP_K].

    Raises:
        ValueError: If top_k is invalid.
    """
    if isinstance(top_k, bool):
        raise ValueError(f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, not a boolean.")
    if not isinstance(top_k, int):
        raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if top_k < MIN_TOP_K or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}.")
    return top_k


def validate_query(query: Any) -> str:
    """Validate query input.

    Raises:
        ValueError: If query is not a string, is a boolean, or exceeds max length.
    """
    if isinstance(query, bool) or not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.")
    return query


def validate_min_score(min_score: Any) -> float:
    """Validate min_score parameter strictly.

    Rejects:
    - Booleans (isinstance(min_score, bool)).
    - Non-numeric types (strings, None, lists, etc.).
    - NaN or infinite values.
    - Values outside [MIN_RELEVANCE_SCORE, MAX_RELEVANCE_SCORE].

    Raises:
        ValueError: If min_score is invalid.
    """
    if isinstance(min_score, bool):
        raise ValueError(
            f"min_score must be a float between {MIN_RELEVANCE_SCORE} and {MAX_RELEVANCE_SCORE}, not a boolean."
        )
    if not isinstance(min_score, (int, float)):
        raise ValueError(f"min_score must be a numeric float, got {type(min_score).__name__}.")
    if math.isnan(min_score) or math.isinf(min_score):
        raise ValueError("min_score must be a finite number.")
    val = float(min_score)
    if val < MIN_RELEVANCE_SCORE or val > MAX_RELEVANCE_SCORE:
        raise ValueError(
            f"min_score must be between {MIN_RELEVANCE_SCORE} and {MAX_RELEVANCE_SCORE}, got {min_score}."
        )
    return val


@dataclass(frozen=True)
class RetrievalResult:
    """Structured result of retrieving a candidate document.

    Attributes:
        document: The matched immutable Document.
        score: Keyword overlap relevance score in [0.0, 1.0].
               This is purely an information retrieval ranking metric,
               NOT a security clearance or safety assessment.
        matched_tokens: Sorted tuple of query tokens found in the document.
    """

    document: Document
    score: float
    matched_tokens: Tuple[str, ...]

    @property
    def doc_id(self) -> str:
        return self.document.doc_id

    @property
    def title(self) -> str:
        return self.document.title

    @property
    def source(self) -> str:
        return self.document.source

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to standard dictionary."""
        return {
            "doc_id": self.document.doc_id,
            "title": self.document.title,
            "source": self.document.source,
            "score": self.score,
            "matched_tokens": list(self.matched_tokens),
            "document": self.document.to_dict(),
        }


def retrieve_documents(
    query: str,
    documents: Sequence[Document],
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    stopwords: Optional[Collection[str]] = DEFAULT_STOPWORDS,
) -> List[RetrievalResult]:
    """Retrieve and rank documents relevant to a query using deterministic token overlap.

    Algorithm:
    1. Validate query, top_k, and min_score bounds.
    2. Tokenize query into distinct lowercase terms.
    3. Filter out common stopwords using the provided stopwords set (default: DEFAULT_STOPWORDS).
       If query is blank or produces zero searchable content tokens (e.g. stopword-only queries),
       return empty list immediately.
    4. For each document, extract tokens from title, tags, and content.
    5. Calculate normalized field-weighted overlap score:
       - Title matches: up to 0.40
       - Tag matches: up to 0.30
       - Content matches: up to 0.30
       Avoids artificially inflating single generic words to 0.70 for one-off passing body mentions.
    6. Filter out documents with zero token overlap or score strictly less than min_score.
    7. Rank deterministically:
       - Primary: Score descending.
       - Secondary: Total token occurrence frequency descending.
       - Tertiary: doc_id ascending (lexicographical tie-breaker).
    8. Return top_k highest-ranking results.

    Limitations:
    - Keyword overlap does not capture semantic synonyms, paraphrases, or intent.
    - An adversarial document that repeats query keywords may score highly.
      Detection and security policy must be applied in subsequent pipeline stages.

    Args:
        query: User search query string.
        documents: Collection of validated Document objects.
        top_k: Maximum number of documents to return (1-10, default 3).
        min_score: Minimum relevance score threshold in [0.0, 1.0] (default 0.30).
                   Candidates scoring below this threshold are excluded from selection.
        stopwords: Optional collection of lowercase stopword strings to ignore during
                   relevance calculation (default: DEFAULT_STOPWORDS). Pass None or () to disable.

    Returns:
        List of RetrievalResult instances (at most top_k).
    """
    valid_query = validate_query(query)
    valid_top_k = validate_top_k(top_k)
    valid_min_score = validate_min_score(min_score)

    if not valid_query.strip() or not documents:
        return []

    q_tokens = tokenize(valid_query)
    if not q_tokens:
        return []

    if stopwords is not None:
        stop_set = frozenset(s.lower() for s in stopwords)
        content_tokens = [t for t in q_tokens if t not in stop_set]
    else:
        content_tokens = q_tokens

    if not content_tokens:
        return []

    q_unique: Set[str] = set(content_tokens)
    total_q_tokens = len(q_unique)

    candidates: List[Tuple[float, int, str, RetrievalResult]] = []

    for doc in documents:
        title_tokens = set(tokenize(doc.title))
        tags_tokens = set(tokenize(" ".join(doc.tags)))
        content_tokens = tokenize(doc.content)
        content_counts = Counter(content_tokens)
        content_set = set(content_counts.keys())

        all_doc_tokens = title_tokens | tags_tokens | content_set
        matched = q_unique & all_doc_tokens
        if not matched:
            continue

        title_matches = len(q_unique & title_tokens)
        tag_matches = len(q_unique & tags_tokens)

        if total_q_tokens == 1:
            tok = next(iter(q_unique))
            has_title = bool(q_unique & title_tokens)
            has_tag = bool(q_unique & tags_tokens)
            if has_title or has_tag:
                base_overlap = 0.70
                title_bonus = 0.20 if has_title else 0.0
                tag_bonus = 0.10 if has_tag else 0.0
            else:
                tf = content_counts.get(tok, 0)
                base_overlap = 0.50 if tf >= 2 else 0.35
                title_bonus = 0.0
                tag_bonus = 0.0
        else:
            base_overlap = 0.70 * (len(matched) / total_q_tokens)
            title_bonus = 0.20 * (title_matches / total_q_tokens)
            tag_bonus = 0.10 * (tag_matches / total_q_tokens)

        final_score = round(min(1.0, base_overlap + title_bonus + tag_bonus), 4)
        if final_score < valid_min_score:
            continue

        all_counts = Counter(tokenize(doc.title) + tokenize(" ".join(doc.tags)) + content_tokens)
        total_occurrences = sum(all_counts[t] for t in matched)

        result = RetrievalResult(
            document=doc,
            score=final_score,
            matched_tokens=tuple(sorted(matched)),
        )

        candidates.append((final_score, total_occurrences, doc.doc_id, result))

    # Sort deterministically: highest score first, then highest frequency, then doc_id ascending
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))

    return [c[3] for c in candidates[:valid_top_k]]
