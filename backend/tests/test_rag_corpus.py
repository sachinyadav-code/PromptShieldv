"""Tests for the PromptShield Phase 3 local RAG knowledge corpus loader."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import pytest

from app.rag.corpus import (
    DEFAULT_CORPUS_PATH,
    CorpusLoadError,
    Document,
    MAX_CORPUS_FILE_SIZE,
    MAX_DOC_ID_LENGTH,
    MAX_TITLE_LENGTH,
    MAX_CONTENT_LENGTH,
    MAX_SOURCE_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS_COUNT,
    MAX_METADATA_DEPTH,
    MAX_METADATA_KEYS_COUNT,
    MAX_METADATA_SERIALIZED_BYTES,
    load_corpus,
)


def test_default_corpus_loads_successfully():
    """Verify that the production default corpus loads all documents cleanly."""
    docs = load_corpus()
    assert isinstance(docs, list)
    assert len(docs) == 14

    # Verify all doc_ids are distinct
    doc_ids = [d.doc_id for d in docs]
    assert len(set(doc_ids)) == 14

    # Verify expected categories are present
    categories = {d.source for d in docs}
    assert "engineering_wiki" in categories
    assert "support_knowledgebase" in categories
    assert "process_handbook" in categories
    assert "security_education_portal" in categories
    assert "synthetic_security_fixtures" in categories

    # Verify specific documents
    doc_map = {d.doc_id: d for d in docs}
    assert "doc-tech-001" in doc_map
    assert "doc-sec-001" in doc_map
    assert "doc-attack-001" in doc_map

    fixture_doc = doc_map["doc-attack-001"]
    assert fixture_doc.metadata.get("is_fixture_attack") is True
    assert fixture_doc.metadata.get("attack_type") == "instruction_override"
    assert "test-fixture" in fixture_doc.tags


def test_document_immutability():
    """Verify that Document dataclass is frozen and fields cannot be mutated."""
    doc = Document(
        doc_id="TEST-001",
        title="Test Title",
        content="Test content body",
        source="Test Source",
        tags=("tag1", "tag2"),
        metadata={"key": "val"},
    )
    with pytest.raises(FrozenInstanceError):
        doc.title = "Modified Title"  # type: ignore

    with pytest.raises(FrozenInstanceError):
        doc.doc_id = "MODIFIED-001"  # type: ignore


def test_document_to_dict():
    """Verify to_dict returns clean, expected serialization."""
    doc = Document(
        doc_id="TEST-001",
        title="Test Title",
        content="Test content body",
        source="Test Source",
        tags=("tag1", "tag2"),
        metadata={"key": "val"},
    )
    d = doc.to_dict()
    assert d == {
        "doc_id": "TEST-001",
        "title": "Test Title",
        "content": "Test content body",
        "source": "Test Source",
        "tags": ["tag1", "tag2"],
        "metadata": {"key": "val"},
    }


def test_missing_corpus_file(tmp_path):
    """Verify that a nonexistent file raises CorpusLoadError with clear path."""
    nonexistent = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(nonexistent)
    assert "does not exist" in str(exc_info.value).lower()
    assert nonexistent.name in str(exc_info.value)


def test_corpus_path_is_directory(tmp_path):
    """Verify that passing a directory path raises CorpusLoadError."""
    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(tmp_path)
    assert "not a file" in str(exc_info.value).lower()


def test_malformed_json_line(tmp_path):
    """Verify that invalid JSON raises CorpusLoadError with line number."""
    corpus_file = tmp_path / "malformed.jsonl"
    corpus_file.write_text('{"doc_id": "VALID-01", "title": "T", "content": "C", "source": "S"}\n{invalid json}\n', encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(corpus_file)
    msg = str(exc_info.value).lower()
    assert "line 2" in msg
    assert "malformed json" in msg


def test_empty_line_rejected(tmp_path):
    """Verify that empty or blank lines are rejected under strict formatting."""
    corpus_file = tmp_path / "blank_line.jsonl"
    corpus_file.write_text('{"doc_id": "VALID-01", "title": "T", "content": "C", "source": "S"}\n   \n', encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(corpus_file)
    msg = str(exc_info.value).lower()
    assert "line 2" in msg
    assert "empty line" in msg


def test_non_dict_json_record(tmp_path):
    """Verify that JSON records that are not objects raise CorpusLoadError."""
    corpus_file = tmp_path / "array.jsonl"
    corpus_file.write_text('["not", "a", "dict"]\n', encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(corpus_file)
    msg = str(exc_info.value).lower()
    assert "line 1" in msg
    assert "must be a json object" in msg


def test_missing_required_fields(tmp_path):
    """Verify that records missing required fields raise CorpusLoadError."""
    required = ["doc_id", "title", "content", "source"]
    base_record = {
        "doc_id": "TEST-01",
        "title": "Test Title",
        "content": "Test content",
        "source": "Test Source",
    }

    for req in required:
        bad = dict(base_record)
        del bad[req]
        corpus_file = tmp_path / f"missing_{req}.jsonl"
        corpus_file.write_text(json.dumps(bad) + "\n", encoding="utf-8")

        with pytest.raises(CorpusLoadError) as exc_info:
            load_corpus(corpus_file)
        msg = str(exc_info.value).lower()
        assert "line 1" in msg
        assert f"missing required field '{req}'" in msg


def test_invalid_doc_id_format(tmp_path):
    """Verify validation of doc_id format, length, and types."""
    invalid_cases = [
        (123, "must be a string"),
        ("", "empty doc_id"),
        ("   ", "empty doc_id"),
        ("A" * (MAX_DOC_ID_LENGTH + 1), "exceeds maximum length"),
        ("DOC ID WITH SPACES", "invalid characters"),
        ("DOC@SPECIAL!CHAR", "invalid characters"),
        ("DOC/SLASH", "invalid characters"),
    ]

    for bad_id, expected_reason in invalid_cases:
        corpus_file = tmp_path / "bad_id.jsonl"
        record = {
            "doc_id": bad_id,
            "title": "Test",
            "content": "Content",
            "source": "Source",
        }
        corpus_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(CorpusLoadError) as exc_info:
            load_corpus(corpus_file)
        assert expected_reason in str(exc_info.value).lower()


def test_duplicate_doc_id(tmp_path):
    """Verify that duplicate doc_ids across records are rejected."""
    corpus_file = tmp_path / "dup.jsonl"
    r1 = {"doc_id": "SAME-ID", "title": "First", "content": "Content", "source": "S"}
    r2 = {"doc_id": "SAME-ID", "title": "Second", "content": "Content", "source": "S"}
    corpus_file.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n", encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(corpus_file)
    msg = str(exc_info.value).lower()
    assert "line 2" in msg
    assert "duplicate doc_id 'same-id'" in msg


def test_oversized_and_invalid_string_fields(tmp_path):
    """Verify size and type limits on title, content, and source."""
    base = {"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S"}

    # Oversized title
    f = tmp_path / "big_title.jsonl"
    bad = dict(base, title="A" * (MAX_TITLE_LENGTH + 1))
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "title exceeds maximum length" in str(exc.value).lower()

    # Empty content
    f = tmp_path / "empty_content.jsonl"
    bad = dict(base, content="")
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "empty or whitespace-only" in str(exc.value).lower()

    # Oversized content
    f = tmp_path / "big_content.jsonl"
    bad = dict(base, content="A" * (MAX_CONTENT_LENGTH + 1))
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "content exceeds maximum length" in str(exc.value).lower()

    # Oversized source
    f = tmp_path / "big_source.jsonl"
    bad = dict(base, source="A" * (MAX_SOURCE_LENGTH + 1))
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "source exceeds maximum length" in str(exc.value).lower()


def test_invalid_tags_validation(tmp_path):
    """Verify validation of tags list structure and contents."""
    base = {"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S"}

    # Tags is not a list
    f = tmp_path / "tags_not_list.jsonl"
    bad = dict(base, tags="single_tag")
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "must be a list of strings" in str(exc.value).lower()

    # Tag item is not a string
    f = tmp_path / "tag_not_str.jsonl"
    bad = dict(base, tags=["valid", 123])
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "must be a string" in str(exc.value).lower()

    # Tag item is empty
    f = tmp_path / "tag_empty.jsonl"
    bad = dict(base, tags=[""])
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "cannot be empty" in str(exc.value).lower()

    # Too many tags
    f = tmp_path / "too_many_tags.jsonl"
    bad = dict(base, tags=[f"t{i}" for i in range(MAX_TAGS_COUNT + 1)])
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "exceeds maximum of" in str(exc.value).lower()


def test_invalid_metadata_validation(tmp_path):
    """Verify validation of metadata dictionary."""
    base = {"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S"}

    # Metadata not a dict
    f = tmp_path / "meta_not_dict.jsonl"
    bad = dict(base, metadata=["not", "dict"])
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "must be a dictionary" in str(exc.value).lower()

    # Too many metadata keys
    f = tmp_path / "meta_too_many.jsonl"
    bad = dict(base, metadata={f"k{i}": i for i in range(MAX_METADATA_KEYS_COUNT + 1)})
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "metadata keys exceed" in str(exc.value).lower()


def test_privacy_preservation_in_exceptions(tmp_path):
    """Ensure sensitive raw text in document content is never leaked into error messages."""
    sensitive_token = "SUPER_SECRET_INTERNAL_API_KEY_NEVER_LOG_998877"
    corpus_file = tmp_path / "sensitive.jsonl"
    record = {
        "doc_id": "INVALID DOC ID WITH SPACES",
        "title": "Title",
        "content": f"Confidential payload contains {sensitive_token}",
        "source": "Source",
    }
    corpus_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc_info:
        load_corpus(corpus_file)

    error_message = str(exc_info.value)
    assert sensitive_token not in error_message
    assert "invalid characters" in error_message.lower()


def test_document_metadata_immutability():
    """Verify that Document.metadata cannot be mutated directly."""
    doc = Document(
        doc_id="DOC-IMMUTABLE",
        title="Title",
        content="Content",
        source="Source",
        metadata={"category": "sec", "nested": {"level": 1}},
    )
    # Direct item assignment should raise TypeError
    with pytest.raises(TypeError):
        doc.metadata["category"] = "hacked"  # type: ignore

    # Nested item assignment should raise TypeError
    with pytest.raises(TypeError):
        doc.metadata["nested"]["level"] = 2  # type: ignore

    # Mutating dictionary returned by to_dict() must not alter Document.metadata
    d = doc.to_dict()
    d["metadata"]["category"] = "tampered"
    assert doc.metadata["category"] == "sec"


def test_metadata_depth_limit_exceeded(tmp_path):
    """Verify metadata deeper than MAX_METADATA_DEPTH is rejected."""
    base = {"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S"}
    deep_meta = {"l1": {"l2": {"l3": {"l4": "too_deep"}}}}
    bad = dict(base, metadata=deep_meta)

    f = tmp_path / "deep_meta.jsonl"
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "exceeds maximum depth" in str(exc.value).lower()


def test_metadata_serialized_size_limit_exceeded(tmp_path):
    """Verify metadata exceeding serialized size limit is rejected."""
    base = {"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S"}
    huge_meta = {f"k_{i}": "x" * 500 for i in range(25)}  # > 10,000 bytes
    bad = dict(base, metadata=huge_meta)

    f = tmp_path / "huge_meta.jsonl"
    f.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "exceeds maximum serialized size" in str(exc.value).lower()


def test_metadata_non_finite_numeric_rejected(tmp_path):
    """Verify non-finite numeric values (NaN, Inf) are rejected in metadata."""
    corpus_file = tmp_path / "nan_meta.jsonl"
    corpus_file.write_text('{"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S", "metadata": {"val": NaN}}\n', encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(corpus_file)
    assert "non-finite numeric value" in str(exc.value).lower() or "malformed json" in str(exc.value).lower()


def test_corpus_file_size_limit_exceeded(tmp_path, monkeypatch):
    """Verify corpus file exceeding MAX_CORPUS_FILE_SIZE is rejected before reading."""
    f = tmp_path / "oversized.jsonl"
    f.write_text('{"doc_id": "DOC-1", "title": "T", "content": "C", "source": "S"}\n', encoding="utf-8")

    # Temporarily monkeypatch MAX_CORPUS_FILE_SIZE in corpus module to 10 bytes
    monkeypatch.setattr("app.rag.corpus.MAX_CORPUS_FILE_SIZE", 10)
    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "exceeds maximum size limit" in str(exc.value).lower()


def test_corpus_document_count_limit_exceeded(tmp_path, monkeypatch):
    """Verify corpus file exceeding MAX_DOCUMENTS_COUNT is rejected."""
    from app.rag.corpus import MAX_DOCUMENTS_COUNT

    f = tmp_path / "too_many_docs.jsonl"
    # Write 1001 valid minimal lines
    lines = ['{"doc_id": "D-%d", "title": "T", "content": "C", "source": "S"}' % i for i in range(MAX_DOCUMENTS_COUNT + 1)]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(f)
    assert "maximum document count limit" in str(exc.value).lower()


def test_nested_metadata_deep_immutability():
    """Verify that nested dictionaries and lists inside Document.metadata are completely immutable."""
    doc = Document(
        doc_id="DOC-DEEP-IMMUTABLE",
        title="Deep Immutability Test",
        content="Testing nested immutability.",
        source="internal",
        tags=("security", "immutability"),
        metadata={
            "level1": {
                "level2": {
                    "level3": "value3",
                    "items": [1, 2, 3],
                }
            }
        },
    )
    # Direct dictionary mutation on nested levels must raise TypeError
    with pytest.raises(TypeError):
        doc.metadata["level1"]["level2"]["level3"] = "tampered"  # type: ignore

    with pytest.raises(TypeError):
        doc.metadata["level1"]["level2"]["items"][0] = 999  # type: ignore

    # Verify tags cannot be mutated
    with pytest.raises(AttributeError):
        doc.tags.append("new_tag")  # type: ignore


def test_set_rag_corpus_stores_immutable_tuple():
    """Verify set_rag_corpus converts mutable lists to an immutable tuple."""
    from app.main import get_rag_corpus, set_rag_corpus, reset_rag_corpus

    doc = Document(
        doc_id="DOC-MUTABLE-TEST",
        title="Title",
        content="Content",
        source="internal",
    )
    mutable_list = [doc]
    set_rag_corpus(mutable_list)

    cached = get_rag_corpus()
    assert isinstance(cached, tuple)
    assert len(cached) == 1
    assert cached[0].doc_id == "DOC-MUTABLE-TEST"

    # Mutating the original list must not affect the cached corpus
    mutable_list.append(Document(doc_id="DOC-SECOND", title="T2", content="C2", source="S"))
    cached_after = get_rag_corpus()
    assert len(cached_after) == 1

    reset_rag_corpus()


def test_corpus_cache_multithreaded_concurrent_access():
    """Verify thread-safe concurrent access and resets of the cached corpus."""
    import concurrent.futures
    from app.main import get_rag_corpus, reset_rag_corpus

    def worker(worker_id: int):
        for _ in range(20):
            if worker_id % 5 == 0:
                reset_rag_corpus()
            corpus = get_rag_corpus()
            assert isinstance(corpus, tuple)
            assert len(corpus) > 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Will re-raise any exception from worker threads

    reset_rag_corpus()
