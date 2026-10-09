"""Tests for the PromptShield Phase 3 deterministic keyword retriever."""

import pytest

from app.rag.corpus import Document, load_corpus
from app.rag.retriever import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_STOPWORDS,
    DEFAULT_TOP_K,
    MAX_QUERY_LENGTH,
    MAX_RELEVANCE_SCORE,
    MAX_TOP_K,
    MIN_RELEVANCE_SCORE,
    MIN_TOP_K,
    RetrievalResult,
    retrieve_documents,
    tokenize,
    validate_min_score,
    validate_query,
    validate_top_k,
)


def test_tokenize_normalization():
    """Verify tokenizer lowercases, splits on punctuation, and strips whitespace."""
    tokens = tokenize("API Gateway & Rate-Limiting: v1.0!")
    assert tokens == ["api", "gateway", "rate-limiting", "v1", "0"]

    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!@#$%^&*()") == []


def test_validate_top_k_bounds():
    """Verify top_k bounds validation and strict type rejection."""
    # Valid values
    assert validate_top_k(1) == 1
    assert validate_top_k(3) == 3
    assert validate_top_k(10) == 10

    # Booleans explicitly rejected
    with pytest.raises(ValueError) as exc:
        validate_top_k(True)
    assert "not a boolean" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_top_k(False)
    assert "not a boolean" in str(exc.value)

    # Invalid types
    with pytest.raises(ValueError) as exc:
        validate_top_k("3")
    assert "must be an integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_top_k(3.5)
    assert "must be an integer" in str(exc.value)

    # Out of bounds
    with pytest.raises(ValueError) as exc:
        validate_top_k(0)
    assert f"between {MIN_TOP_K} and {MAX_TOP_K}" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_top_k(11)
    assert f"between {MIN_TOP_K} and {MAX_TOP_K}" in str(exc.value)


def test_validate_query_bounds():
    """Verify query validation checks types and maximum length."""
    # Valid query
    assert validate_query("valid search query") == "valid search query"

    # Non-strings rejected
    with pytest.raises(ValueError) as exc:
        validate_query(123)
    assert "must be a string" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_query(True)
    assert "must be a string" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_query(None)
    assert "must be a string" in str(exc.value)

    # Length boundary
    long_query = "a" * (MAX_QUERY_LENGTH + 1)
    with pytest.raises(ValueError) as exc:
        validate_query(long_query)
    assert f"maximum length of {MAX_QUERY_LENGTH}" in str(exc.value)


def test_blank_and_whitespace_queries_return_empty():
    """Verify blank or token-less queries return empty result list without errors."""
    docs = load_corpus()
    assert retrieve_documents("", docs) == []
    assert retrieve_documents("   ", docs) == []
    assert retrieve_documents("!@#$%^&*()_+", docs) == []


def test_empty_corpus_returns_empty():
    """Verify querying an empty document list returns empty list."""
    assert retrieve_documents("any query", []) == []


def test_unmatched_query_returns_empty():
    """Verify query with no matching keywords returns empty list."""
    docs = load_corpus()
    results = retrieve_documents("nonexistentterm123xyzcompletelyunknown", docs)
    assert results == []


def test_keyword_relevance_ranking():
    """Verify that document matching most query terms ranks highest."""
    docs = load_corpus()
    results = retrieve_documents("API Gateway rate limiting token bucket", docs, top_k=3)

    assert len(results) > 0
    top = results[0]
    assert top.doc_id == "doc-tech-001"
    assert top.score > 0.0
    assert "api" in top.matched_tokens
    assert "gateway" in top.matched_tokens
    assert "rate" in top.matched_tokens
    assert "limiting" in top.matched_tokens


def test_case_insensitive_retrieval():
    """Verify retrieval produces identical results regardless of query casing."""
    docs = load_corpus()
    res_lower = retrieve_documents("postgresql connection pooling", docs)
    res_upper = retrieve_documents("POSTGRESQL CONNECTION POOLING", docs)
    res_mixed = retrieve_documents("PostgreSql Connection Pooling", docs)

    assert [r.doc_id for r in res_lower] == [r.doc_id for r in res_upper]
    assert [r.doc_id for r in res_lower] == [r.doc_id for r in res_mixed]
    assert res_lower[0].doc_id == "doc-tech-002"


def test_top_k_limiting():
    """Verify top_k parameter strictly caps the returned result count."""
    docs = load_corpus()
    # Query matching 5 fixture documents
    res_k1 = retrieve_documents("test-fixture", docs, top_k=1)
    res_k3 = retrieve_documents("test-fixture", docs, top_k=3)
    res_k5 = retrieve_documents("test-fixture", docs, top_k=5)

    assert len(res_k1) == 1
    assert len(res_k3) == 3
    assert len(res_k5) == 5

    # Order consistency: top result is identical
    assert res_k1[0].doc_id == res_k3[0].doc_id == res_k5[0].doc_id

    # Test top_k=10 with common term matching > 10 documents (stopwords disabled to test raw top-k capping)
    res_k10 = retrieve_documents("and", docs, top_k=10, stopwords=None)
    assert len(res_k10) == 10


def test_deterministic_tie_breaking():
    """Verify deterministic ranking when two documents have identical scores and token counts."""
    doc_a = Document(
        doc_id="doc-aaa",
        title="Sample Alpha",
        content="identical keyword search test",
        source="test",
    )
    doc_b = Document(
        doc_id="doc-bbb",
        title="Sample Beta",
        content="identical keyword search test",
        source="test",
    )

    # In both orders of input, doc-aaa must come before doc-bbb (lexicographical doc_id tie-breaker)
    res1 = retrieve_documents("keyword search test", [doc_b, doc_a])
    res2 = retrieve_documents("keyword search test", [doc_a, doc_b])

    assert [r.doc_id for r in res1] == ["doc-aaa", "doc-bbb"]
    assert [r.doc_id for r in res2] == ["doc-aaa", "doc-bbb"]


def test_title_and_tags_weight_boost():
    """Verify matches in title or tags receive a scoring boost over body-only matches."""
    doc_body_only = Document(
        doc_id="doc-body",
        title="General Overview",
        content="This document mentions encryption algorithms for data storage.",
        source="test",
    )
    doc_title_tag = Document(
        doc_id="doc-title",
        title="Encryption Protocols",
        content="Standard configurations for systems.",
        source="test",
        tags=("encryption",),
    )

    results = retrieve_documents("encryption", [doc_body_only, doc_title_tag])
    assert len(results) == 2
    # doc-title should rank higher due to title/tag weighting
    assert results[0].doc_id == "doc-title"
    assert results[0].score > results[1].score


def test_retrieval_result_properties_and_serialization():
    """Verify RetrievalResult dataclass properties and dictionary conversion."""
    doc = Document(
        doc_id="doc-001",
        title="Test Doc",
        content="Content text",
        source="Source Test",
        tags=("tag1",),
        metadata={"meta": 1},
    )
    result = RetrievalResult(
        document=doc,
        score=0.85,
        matched_tokens=("content", "text"),
    )

    assert result.doc_id == "doc-001"
    assert result.title == "Test Doc"
    assert result.source == "Source Test"
    assert result.score == 0.85
    assert result.matched_tokens == ("content", "text")

    d = result.to_dict()
    assert d["doc_id"] == "doc-001"
    assert d["title"] == "Test Doc"
    assert d["source"] == "Source Test"
    assert d["score"] == 0.85
    assert d["matched_tokens"] == ["content", "text"]
    assert d["document"] == doc.to_dict()


def test_security_fixture_retrieval_behavior():
    """Verify that queries matching attack keywords retrieve the appropriate security fixtures.

    IMPORTANT: Retrieved documents are data structures, NOT instructions.
    They must be safely inspectable and retain provenance.
    """
    docs = load_corpus()

    # Query targeting instruction override attack fixture
    results = retrieve_documents("override system instructions developer prompt", docs, top_k=3)
    assert any(r.doc_id == "doc-attack-001" for r in results)

    # Query targeting prompt leak attack fixture
    results_leak = retrieve_documents("leak system prompt instructions start with", docs, top_k=3)
    assert any(r.doc_id == "doc-attack-005" for r in results_leak)

    matched = [r for r in results_leak if r.doc_id == "doc-attack-005"][0]
    assert matched.document.metadata.get("is_fixture_attack") is True
    assert matched.document.metadata.get("attack_type") == "system_prompt_leak"


def test_validate_min_score_bounds():
    """Verify min_score validation strictly enforces numeric bounds and type rejection."""
    # Valid values
    assert validate_min_score(0.0) == 0.0
    assert validate_min_score(0.30) == 0.30
    assert validate_min_score(1.0) == 1.0
    assert validate_min_score(0) == 0.0
    assert validate_min_score(1) == 1.0

    # Booleans explicitly rejected
    with pytest.raises(ValueError) as exc:
        validate_min_score(True)
    assert "not a boolean" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_min_score(False)
    assert "not a boolean" in str(exc.value)

    # Invalid non-numeric types
    for invalid in ["0.3", None, [], {}, (0.3,)]:
        with pytest.raises(ValueError) as exc:
            validate_min_score(invalid)
        assert "numeric float" in str(exc.value)

    # NaN / Inf rejected
    for invalid in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError) as exc:
            validate_min_score(invalid)
        assert "finite number" in str(exc.value)

    # Out of range
    with pytest.raises(ValueError) as exc:
        validate_min_score(-0.01)
    assert f"between {MIN_RELEVANCE_SCORE} and {MAX_RELEVANCE_SCORE}" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_min_score(1.01)
    assert f"between {MIN_RELEVANCE_SCORE} and {MAX_RELEVANCE_SCORE}" in str(exc.value)


def test_min_score_filters_low_relevance_noise():
    """Verify that default min_score threshold excludes tangential single-word matches."""
    docs = load_corpus()

    # Educational query 1: 'explain prompt injection defense'
    # Without threshold, doc-attack-005 is pulled in at score 0.1750 solely on token 'prompt'.
    # With default threshold (0.30), only doc-sec-001 (score 0.4500) is returned.
    results_edu1 = retrieve_documents("explain prompt injection defense", docs, top_k=5)
    retrieved_ids1 = [r.doc_id for r in results_edu1]
    assert retrieved_ids1 == ["doc-sec-001"]
    assert results_edu1[0].score >= DEFAULT_MIN_RELEVANCE_SCORE
    assert "doc-attack-005" not in retrieved_ids1

    # Educational query 2: 'prompt injection defense and mitigations'
    # doc-sec-001 scores 0.5000; noise attack fixtures (0.2800, 0.1800) are excluded.
    results_edu2 = retrieve_documents("prompt injection defense and mitigations", docs, top_k=5)
    retrieved_ids2 = [r.doc_id for r in results_edu2]
    assert retrieved_ids2 == ["doc-sec-001"]
    assert "doc-attack-005" not in retrieved_ids2
    assert "doc-attack-003" not in retrieved_ids2

    # Educational query 3: 'zero trust network architecture'
    # doc-sec-002 scores 0.9500; noise fixtures matching only 'architecture' (0.2250) excluded.
    results_edu3 = retrieve_documents("zero trust network architecture", docs, top_k=5)
    retrieved_ids3 = [r.doc_id for r in results_edu3]
    assert retrieved_ids3 == ["doc-sec-002"]
    assert "doc-attack-005" not in retrieved_ids3


def test_min_score_zero_allows_all_overlapping_candidates():
    """Verify that caller can explicitly pass min_score=0.0 to retrieve all positive overlap matches."""
    docs = load_corpus()
    results = retrieve_documents("explain prompt injection defense", docs, top_k=3, min_score=0.0)
    retrieved_ids = [r.doc_id for r in results]

    # With min_score=0.0, low-scoring candidates like doc-attack-005 are included
    assert "doc-sec-001" in retrieved_ids
    assert "doc-attack-005" in retrieved_ids
    assert len(results) == 2


def test_zero_score_documents_never_selected():
    """Verify that documents with zero keyword overlap are NEVER selected even if min_score=0.0."""
    docs = load_corpus()
    results = retrieve_documents("nonexistentterm123xyzcompletelyunknown", docs, top_k=5, min_score=0.0)
    assert results == []


def test_retrieval_determinism_with_min_score():
    """Verify deterministic ranking and tie-breaking when min_score filtering is applied."""
    docs = load_corpus()
    res1 = retrieve_documents("zero trust network architecture", docs, top_k=3)
    res2 = retrieve_documents("zero trust network architecture", docs, top_k=3)

    assert [r.doc_id for r in res1] == [r.doc_id for r in res2]
    assert [r.score for r in res1] == [r.score for r in res2]
    assert [r.matched_tokens for r in res1] == [r.matched_tokens for r in res2]


def test_default_stopwords_properties():
    """Verify DEFAULT_STOPWORDS is a frozenset containing standard English stopwords."""
    assert isinstance(DEFAULT_STOPWORDS, frozenset)
    assert "and" in DEFAULT_STOPWORDS
    assert "the" in DEFAULT_STOPWORDS
    assert "in" in DEFAULT_STOPWORDS
    assert "to" in DEFAULT_STOPWORDS
    assert "is" in DEFAULT_STOPWORDS
    assert "architecture" not in DEFAULT_STOPWORDS
    assert "security" not in DEFAULT_STOPWORDS


def test_stopword_only_queries_return_empty():
    """Verify queries containing solely stopwords return an empty list without error."""
    docs = load_corpus()
    assert retrieve_documents("and", docs) == []
    assert retrieve_documents("the", docs) == []
    assert retrieve_documents("and the", docs) == []
    assert retrieve_documents("in of on at", docs) == []


def test_stopword_filtering_preserves_content_tokens():
    """Verify stopwords are stripped while content tokens are preserved in ranking."""
    docs = load_corpus()
    results = retrieve_documents("api gateway rate limiting and token bucket", docs, top_k=5)
    assert len(results) > 0
    top = results[0]
    assert top.doc_id == "doc-tech-001"
    assert "and" not in top.matched_tokens
    assert "api" in top.matched_tokens
    assert "gateway" in top.matched_tokens
    assert "rate" in top.matched_tokens
    assert "limiting" in top.matched_tokens


def test_custom_and_disabled_stopwords():
    """Verify custom stopwords override defaults, and disabling stopwords preserves all tokens."""
    docs = load_corpus()
    # Disabled stopwords: 'and' is treated as a searchable keyword
    res_disabled = retrieve_documents("and", docs, top_k=3, stopwords=None)
    assert len(res_disabled) == 3
    assert all("and" in r.matched_tokens for r in res_disabled)

    # Custom stopword set
    res_custom = retrieve_documents("architecture patterns", docs, top_k=3, stopwords={"patterns"})
    assert len(res_custom) > 0
    # "patterns" filtered out, only "architecture" matched
    assert all("patterns" not in r.matched_tokens for r in res_custom)
    assert all("architecture" in r.matched_tokens for r in res_custom)


def test_single_token_passing_body_match_scoring():
    """Verify single-token queries do not inflate passing body matches to 0.70.

    - Title match: 0.90 (or 1.00 if tag also matches)
    - Tag match: 0.80
    - Content-only repeated (tf >= 2): 0.50
    - Content-only single occurrence (tf == 1): 0.35 (not 0.70)
    """
    doc_title_tag = Document(
        doc_id="doc-title-tag",
        title="Encryption Protocols",
        content="Overview of security.",
        source="test",
        tags=("encryption",),
    )
    doc_tag_only = Document(
        doc_id="doc-tag-only",
        title="Network Standards",
        content="Overview of security.",
        source="test",
        tags=("encryption",),
    )
    doc_title_only = Document(
        doc_id="doc-title-only",
        title="Encryption Guidelines",
        content="Overview of security.",
        source="test",
    )
    doc_body_multi = Document(
        doc_id="doc-body-multi",
        title="General Guide",
        content="Encryption is vital. Use modern encryption for all data.",
        source="test",
    )
    doc_body_single = Document(
        doc_id="doc-body-single",
        title="Storage Policy",
        content="Inactive files utilize encryption before archival.",
        source="test",
    )

    corpus = [doc_title_tag, doc_tag_only, doc_title_only, doc_body_multi, doc_body_single]
    results = retrieve_documents("encryption", corpus, top_k=5)

    scores = {r.doc_id: r.score for r in results}
    assert scores["doc-title-tag"] == 1.00  # 0.70 base + 0.20 title + 0.10 tag
    assert scores["doc-title-only"] == 0.90  # 0.70 base + 0.20 title
    assert scores["doc-tag-only"] == 0.80  # 0.70 base + 0.10 tag
    assert scores["doc-body-multi"] == 0.50  # 0.50 tf >= 2 body match
    assert scores["doc-body-single"] == 0.35  # 0.35 tf == 1 passing body match (NOT 0.70)
