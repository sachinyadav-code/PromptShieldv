"""Local document corpus loader for PromptShield Phase 3 (RAG).

Provides deterministic loading, strict field validation, and safe error handling
for the synthetic knowledge base. Every corpus field is treated as untrusted data.
Raw document contents are never logged or leaked in error messages.
"""

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent.parent
DEFAULT_CORPUS_PATH = REPO_ROOT / "datasets" / "rag_corpus_v1.jsonl"

MAX_CORPUS_FILE_SIZE = 10 * 1024 * 1024  # 10 MB maximum corpus file size
MAX_DOCUMENTS_COUNT = 1000  # 1,000 documents maximum in a single corpus file
MAX_DOC_ID_LENGTH = 64
MAX_TITLE_LENGTH = 256
MAX_CONTENT_LENGTH = 20000
MAX_SOURCE_LENGTH = 256
MAX_TAG_LENGTH = 64
MAX_TAGS_COUNT = 50
MAX_METADATA_KEYS_COUNT = 50
MAX_METADATA_DEPTH = 3
MAX_METADATA_SERIALIZED_BYTES = 10000
MAX_METADATA_KEY_LENGTH = 64
MAX_METADATA_VALUE_LENGTH = 2000

DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class CorpusLoadError(ValueError):
    """Exception raised when loading or validating a corpus fails.

    Preserves privacy: Never embeds raw document text or sensitive content
    in exception messages.
    """
    pass


def _freeze_value(val: Any) -> Any:
    """Recursively freeze mapping and sequence objects into immutable proxies/tuples."""
    if isinstance(val, (MappingProxyType, dict)):
        return MappingProxyType({k: _freeze_value(v) for k, v in val.items()})
    if isinstance(val, list):
        return tuple(_freeze_value(item) for item in val)
    return val


def _unfreeze_value(val: Any) -> Any:
    """Recursively unfreeze mapping and sequence objects into independent mutable dicts/lists."""
    if isinstance(val, (MappingProxyType, dict)):
        return {k: _unfreeze_value(v) for k, v in val.items()}
    if isinstance(val, (tuple, list)):
        return [_unfreeze_value(item) for item in val]
    return val


@dataclass(frozen=True)
class Document:
    """Immutable representation of a document in the RAG corpus.

    All fields are untrusted data and must never be treated as authoritative
    system or developer instructions.
    """

    doc_id: str
    title: str
    content: str
    source: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", _freeze_value(dict(self.metadata)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to a standard dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "tags": list(self.tags),
            "metadata": _unfreeze_value(self.metadata),
        }


def _validate_metadata_node(node: Any, line_num: int, current_depth: int) -> None:
    """Validate metadata structure recursively without echoing raw values."""
    if current_depth > MAX_METADATA_DEPTH:
        raise CorpusLoadError(f"metadata at line {line_num} exceeds maximum depth of {MAX_METADATA_DEPTH}")

    if isinstance(node, dict):
        if len(node) > MAX_METADATA_KEYS_COUNT:
            raise CorpusLoadError(f"metadata keys exceed maximum of {MAX_METADATA_KEYS_COUNT} at line {line_num}")
        for k, v in node.items():
            if isinstance(k, bool) or not isinstance(k, str):
                raise CorpusLoadError(f"metadata key must be string at line {line_num}")
            if not k or len(k) > MAX_METADATA_KEY_LENGTH:
                raise CorpusLoadError(f"metadata key at line {line_num} exceeds maximum length of {MAX_METADATA_KEY_LENGTH}")
            _validate_metadata_node(v, line_num, current_depth + 1)
    elif isinstance(node, list):
        if len(node) > 100:
            raise CorpusLoadError(f"metadata list at line {line_num} exceeds maximum items limit")
        for item in node:
            _validate_metadata_node(item, line_num, current_depth + 1)
    elif isinstance(node, str):
        if len(node) > MAX_METADATA_VALUE_LENGTH:
            raise CorpusLoadError(f"metadata string value at line {line_num} exceeds maximum length of {MAX_METADATA_VALUE_LENGTH}")
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        if not math.isfinite(node):
            raise CorpusLoadError(f"metadata at line {line_num} contains non-finite numeric value")
    elif node is None:
        pass
    else:
        raise CorpusLoadError(f"metadata at line {line_num} contains unsupported value type {type(node).__name__}")


def load_corpus(corpus_path: Optional[Union[Path, str]] = None) -> List[Document]:
    """Load and validate all documents from a JSONL corpus file.

    Args:
        corpus_path: Optional path to a JSONL corpus file. If None, uses
                     the default repository path at datasets/rag_corpus_v1.jsonl.

    Returns:
        A list of validated immutable Document instances in file order.

    Raises:
        CorpusLoadError: If the file does not exist, contains invalid JSON,
                         has missing or invalid fields, duplicate doc_ids,
                         or exceeds field size boundaries.
    """
    target_path = Path(corpus_path) if corpus_path is not None else DEFAULT_CORPUS_PATH

    if not target_path.exists():
        raise CorpusLoadError(f"Corpus file does not exist: {target_path.name}")
    if not target_path.is_file():
        raise CorpusLoadError(f"Corpus path is not a file: {target_path.name}")
    if target_path.stat().st_size > MAX_CORPUS_FILE_SIZE:
        raise CorpusLoadError(f"Corpus file exceeds maximum size limit of {MAX_CORPUS_FILE_SIZE} bytes")

    try:
        raw_text = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise CorpusLoadError(f"Failed to read corpus file {target_path.name}: {type(exc).__name__}") from exc

    lines = raw_text.splitlines()
    if not lines:
        return []

    if len(lines) > MAX_DOCUMENTS_COUNT:
        raise CorpusLoadError(f"Corpus exceeds maximum document count limit of {MAX_DOCUMENTS_COUNT}")

    documents: List[Document] = []
    seen_ids: set[str] = set()

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            raise CorpusLoadError(f"Empty line encountered at line {line_num} in {target_path.name}")

        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CorpusLoadError(f"Malformed JSON at line {line_num} in {target_path.name}: {exc.msg}") from exc

        if not isinstance(record, dict) or isinstance(record, bool):
            raise CorpusLoadError(f"Record at line {line_num} in {target_path.name} must be a JSON object")

        # Required fields check
        required_fields = ("doc_id", "title", "content", "source")
        for req_field in required_fields:
            if req_field not in record:
                raise CorpusLoadError(f"Missing required field '{req_field}' at line {line_num} in {target_path.name}")
            val = record[req_field]
            if val is None or isinstance(val, bool) or not isinstance(val, str):
                raise CorpusLoadError(f"Field '{req_field}' at line {line_num} must be a string, got {type(val).__name__}")

        # doc_id validation
        doc_id = record["doc_id"].strip()
        if not doc_id:
            raise CorpusLoadError(f"Empty doc_id at line {line_num} in {target_path.name}")
        if len(doc_id) > MAX_DOC_ID_LENGTH:
            raise CorpusLoadError(f"doc_id exceeds maximum length of {MAX_DOC_ID_LENGTH} at line {line_num}")
        if not DOC_ID_PATTERN.match(doc_id):
            raise CorpusLoadError(f"doc_id contains invalid characters at line {line_num}")
        if doc_id in seen_ids:
            raise CorpusLoadError(f"Duplicate doc_id '{doc_id}' at line {line_num} in {target_path.name}")
        seen_ids.add(doc_id)

        # title validation
        title = record["title"]
        if len(title) > MAX_TITLE_LENGTH:
            raise CorpusLoadError(f"title exceeds maximum length of {MAX_TITLE_LENGTH} at line {line_num}")

        # content validation
        content = record["content"]
        if not content.strip():
            raise CorpusLoadError(f"content cannot be empty or whitespace-only at line {line_num}")
        if len(content) > MAX_CONTENT_LENGTH:
            raise CorpusLoadError(f"content exceeds maximum length of {MAX_CONTENT_LENGTH} at line {line_num}")

        # source validation
        source = record["source"]
        if len(source) > MAX_SOURCE_LENGTH:
            raise CorpusLoadError(f"source exceeds maximum length of {MAX_SOURCE_LENGTH} at line {line_num}")

        # tags validation (optional)
        tags_raw = record.get("tags")
        tags: List[str] = []
        if tags_raw is not None:
            if isinstance(tags_raw, (str, bool)) or not isinstance(tags_raw, list):
                raise CorpusLoadError(f"tags at line {line_num} must be a list of strings")
            if len(tags_raw) > MAX_TAGS_COUNT:
                raise CorpusLoadError(f"tags count exceeds maximum of {MAX_TAGS_COUNT} at line {line_num}")
            for tag_idx, tag in enumerate(tags_raw):
                if isinstance(tag, bool) or not isinstance(tag, str):
                    raise CorpusLoadError(f"Tag at index {tag_idx} at line {line_num} must be a string")
                clean_tag = tag.strip()
                if not clean_tag:
                    raise CorpusLoadError(f"Tag at index {tag_idx} at line {line_num} cannot be empty")
                if len(clean_tag) > MAX_TAG_LENGTH:
                    raise CorpusLoadError(f"Tag at index {tag_idx} at line {line_num} exceeds max length {MAX_TAG_LENGTH}")
                tags.append(clean_tag)

        # metadata validation (optional)
        meta_raw = record.get("metadata")
        metadata: Dict[str, Any] = {}
        if meta_raw is not None:
            if isinstance(meta_raw, bool) or not isinstance(meta_raw, dict):
                raise CorpusLoadError(f"metadata at line {line_num} must be a dictionary")
            try:
                serialized_meta = json.dumps(meta_raw)
            except (TypeError, ValueError) as exc:
                raise CorpusLoadError(f"metadata at line {line_num} cannot be serialized to JSON") from exc
            if len(serialized_meta.encode("utf-8")) > MAX_METADATA_SERIALIZED_BYTES:
                raise CorpusLoadError(f"metadata at line {line_num} exceeds maximum serialized size of {MAX_METADATA_SERIALIZED_BYTES} bytes")
            _validate_metadata_node(meta_raw, line_num, current_depth=1)
            metadata = dict(meta_raw)

        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                content=content,
                source=source,
                tags=tuple(tags),
                metadata=metadata,
            )
        )

    logger.info("Loaded %d documents from %s", len(documents), target_path.name)
    return documents
