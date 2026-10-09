"""Machine Learning Prompt Injection Classifier for PromptShield.

Provides a thread-safe, resilient classifier that evaluates user prompts against
the trained prompt injection linear classifier (TF-IDF + Logistic Regression).

IMPORTANT SECURITY & STATISTICAL NOTICE:
- The classifier output score is an uncalibrated heuristic estimate, derived from
  the logistic sigmoid function over synthetic training data. It is NOT a Bayesian
  posterior probability of attack success and NOT a safety guarantee.
- ML predictions must NEVER be the sole safeguard for sensitive actions.
  Deterministic security rules (regex and policy guards) take sovereign precedence.
- An unavailable or failing ML classifier must never be silently treated as a benign
  clearance or as permission to bypass deterministic rules.
- Raw prompt texts are NEVER logged or persisted by this module.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, Literal, Optional, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Resolve default model paths relative to this module directory (robust against CWD changes)
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODULE_DIR / "models" / "prompt_injection_v1.joblib"
DEFAULT_METADATA_PATH = MODULE_DIR / "models" / "prompt_injection_v1_meta.json"
DEFAULT_FALLBACK_THRESHOLD = 0.50


def is_valid_threshold(val: Any) -> bool:
    """Strictly validate whether a value is a valid numeric decision threshold.

    Rejects:
    - Booleans (Python bool or numpy bool_).
    - Non-numeric types (str, list, dict, None, etc.).
    - Non-finite numbers (NaN, Inf, -Inf).
    - Numeric values strictly outside the closed interval [0.0, 1.0].
    """
    if isinstance(val, (bool, np.bool_)):
        return False
    if not isinstance(val, (int, float, np.number)):
        return False
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(f_val):
        return False
    return 0.0 <= f_val <= 1.0


@dataclass(frozen=True)
class MLInferenceResult:
    """Structured result of ML prompt injection inference.

    Attributes:
        available: True if the model is loaded and inference succeeded; False otherwise.
        score: Uncalibrated heuristic confidence score in [0.0, 1.0] indicating likelihood
               of prompt injection. None if the model was unavailable or inference failed.
        threshold: The decision threshold applied for blocking (e.g. 0.50).
        predicted_label: 'injection' or 'benign' if available; None if unavailable.
        would_block: True if available and unrounded raw score >= threshold; False otherwise.
        model_version: Identifier of the model version (e.g. 'v1'), or None if unknown.
        error_code: High-level error identifier if unavailable; None on success.
    """

    available: bool
    score: Optional[float]
    threshold: float
    predicted_label: Optional[Literal["benign", "injection"]]
    would_block: bool
    model_version: Optional[str]
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


class MLPromptInjectionClassifier:
    """Thread-safe, resilient prompt injection classifier wrapper.

    Encapsulates artifact loading, input validation, feature inference, and graceful
    fallback handling for the scikit-learn TF-IDF + Logistic Regression pipeline.
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        metadata_path: Optional[Union[Path, str]] = None,
        threshold: Optional[Any] = None,
    ) -> None:
        """Initialize the classifier configuration without loading from disk immediately.

        Args:
            model_path: Path to serialized .joblib pipeline artifact.
            metadata_path: Path to metadata .json artifact.
            threshold: Explicit decision threshold in [0.0, 1.0]. If invalid or None,
                       resolved from metadata artifact, or defaults to 0.50.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

        self._model: Any = None
        self._model_version: Optional[str] = None
        self._injection_class_index: Optional[int] = None
        self._classes: list = []
        self._load_count: int = 0
        self._load_attempted: bool = False
        self._load_error_code: Optional[str] = None
        self._lock = threading.RLock()

        # Resolve threshold and metadata
        self.threshold = self._resolve_threshold_and_metadata(threshold)

    def _resolve_threshold_and_metadata(self, explicit_threshold: Optional[Any]) -> float:
        """Resolve threshold strictly: explicit valid > metadata valid > fallback default."""
        # 1. Attempt explicit threshold
        if explicit_threshold is not None:
            if is_valid_threshold(explicit_threshold):
                self._load_metadata_version_only()
                return float(explicit_threshold)
            logger.warning(
                "Explicit threshold %r is invalid; attempting metadata fallback.",
                type(explicit_threshold).__name__,
            )

        # 2. Attempt metadata threshold
        if self.metadata_path and self.metadata_path.is_file():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._model_version = meta.get("model_version")
                meta_threshold = meta.get("threshold_selection", {}).get("selected_threshold")
                if is_valid_threshold(meta_threshold):
                    return float(meta_threshold)
                logger.warning("Metadata threshold %r is invalid; using default.", meta_threshold)
            except Exception:
                logger.warning(
                    "Could not read threshold from metadata; using default %.2f.",
                    DEFAULT_FALLBACK_THRESHOLD,
                )

        return DEFAULT_FALLBACK_THRESHOLD

    def _load_metadata_version_only(self) -> None:
        """Attempt to read model_version from metadata if available."""
        if self.metadata_path and self.metadata_path.is_file():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._model_version = meta.get("model_version")
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Return True if model is loaded, validated, and ready for inference."""
        return self._model is not None and self._injection_class_index is not None

    @property
    def is_loaded(self) -> bool:
        """Alias for is_available."""
        return self.is_available

    @property
    def model_version(self) -> Optional[str]:
        """Return the model version string if known."""
        return self._model_version

    @property
    def load_count(self) -> int:
        """Return the number of times the model artifact was loaded from disk."""
        return self._load_count

    def get_status(self) -> Dict[str, Any]:
        """Inspect the current operational status of the classifier."""
        return {
            "available": self.is_available,
            "model_version": self._model_version,
            "threshold": self.threshold,
            "load_count": self._load_count,
            "load_attempted": self._load_attempted,
            "error_code": self._load_error_code if not self.is_available else None,
        }

    def load(self, force_reload: bool = False) -> bool:
        """Load and validate the model artifact under thread safety lock.

        Retry and Caching Policy:
        - Successful loads are cached: once is_available is True, subsequent calls
          to load() or predict() will not reload from disk unless force_reload=True.
        - Failed loads (e.g. missing or corrupt artifacts) are recorded via
          _load_attempted=True. On subsequent inference requests, predict() avoids
          repeated uncontrolled disk I/O and deserialization attempts, immediately
          returning an unavailable result with the cached error code.
        - Explicit retries: Callers can deliberately trigger a fresh load attempt
          by calling load(force_reload=True) or reload().
        - A failed reload leaves the classifier in a consistent, safely unavailable
          state: in-memory model references and class indices are cleared to ensure
          no partial or stale state remains.

        Args:
            force_reload: If True, resets cached state and attempts a fresh disk load.

        Returns:
            True if model was successfully loaded and validated, False otherwise.
        """
        with self._lock:
            if self.is_available and not force_reload:
                return True

            if self._load_attempted and not force_reload and not self.is_available:
                return False

            if force_reload:
                self._model = None
                self._injection_class_index = None
                self._classes = []
                self._load_error_code = None

            self._load_attempted = True

            if not self.model_path.is_file():
                logger.warning("ML model artifact not found: %s", self.model_path.name)
                self._load_error_code = "ARTIFACT_NOT_FOUND"
                return False

            try:
                loaded_obj = joblib.load(self.model_path)
            except Exception as e:
                logger.error("Failed to deserialize ML model artifact: %s", type(e).__name__)
                self._load_error_code = "CORRUPT_ARTIFACT"
                return False

            if not self._validate_model(loaded_obj):
                return False

            self._model = loaded_obj
            self._load_count += 1
            self._load_error_code = None
            logger.info(
                "ML Prompt Injection Classifier loaded successfully (version: %s, threshold: %.2f)",
                self._model_version or "unknown",
                self.threshold,
            )
            return True

    def reload(self) -> bool:
        """Explicitly retry/reload the model artifact from disk."""
        return self.load(force_reload=True)

    def _validate_model(self, candidate_model: Any) -> bool:
        """Validate candidate model attributes, class labels, and inference capability."""
        if not hasattr(candidate_model, "classes_"):
            logger.error("Candidate model missing 'classes_' attribute.")
            self._load_error_code = "INVALID_MODEL_STRUCTURE"
            return False

        if not hasattr(candidate_model, "predict_proba") or not callable(candidate_model.predict_proba):
            logger.error("Candidate model missing callable 'predict_proba'.")
            self._load_error_code = "INVALID_MODEL_STRUCTURE"
            return False

        try:
            raw_classes = candidate_model.classes_
            classes = [str(c) for c in raw_classes]
        except Exception:
            logger.error("Unable to parse candidate model classes.")
            self._load_error_code = "INVALID_MODEL_CLASSES"
            return False

        # Require exactly the expected binary classes 'benign' and 'injection'
        # No duplicates, no extra classes, exactly 2 elements
        if len(classes) != 2 or set(classes) != {"benign", "injection"}:
            logger.error(
                "Candidate model classes must be exactly {'benign', 'injection'} without duplicates or extras; got %r",
                classes,
            )
            self._load_error_code = "INVALID_MODEL_CLASSES"
            return False

        self._classes = classes
        self._injection_class_index = classes.index("injection")
        return True

    def predict(self, prompt: str) -> MLInferenceResult:
        """Evaluate a prompt string for injection indicators.

        Args:
            prompt: User prompt string to inspect.

        Returns:
            MLInferenceResult containing structured classification findings.
        """
        with self._lock:
            # Automatic single load attempt if not yet loaded
            if not self.is_available and not self._load_attempted:
                self.load()

            if not self.is_available:
                return MLInferenceResult(
                    available=False,
                    score=None,
                    threshold=self.threshold,
                    predicted_label=None,
                    would_block=False,
                    model_version=self._model_version,
                    error_code=self._load_error_code or "MODEL_UNAVAILABLE",
                )

            # Safely capture references while under lock
            model = self._model
            injection_idx = self._injection_class_index

        if not isinstance(prompt, str):
            logger.error("Predict received non-string input of type: %s", type(prompt).__name__)
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="INVALID_INPUT",
            )

        try:
            probas = model.predict_proba([prompt])
        except Exception as e:
            logger.error("Inference execution failed: %s", type(e).__name__)
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="INFERENCE_FAILED",
            )

        # 1. Output shape validation: must be 2D array of shape (1, 2)
        if not isinstance(probas, np.ndarray) or probas.shape != (1, 2):
            logger.error(
                "Inference output has invalid shape or type: %s",
                getattr(probas, "shape", type(probas).__name__),
            )
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="MALFORMED_OUTPUT",
            )

        # 2. Output real-valued numeric dtype validation (reject non-numeric and complex numbers)
        if np.iscomplexobj(probas):
            logger.error("Inference output contains complex-valued numbers.")
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="COMPLEX_VALUED_OUTPUT",
            )

        if not (np.issubdtype(probas.dtype, np.floating) or np.issubdtype(probas.dtype, np.integer)):
            logger.error("Inference output has non-numeric dtype: %s", probas.dtype)
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="NON_NUMERIC_OUTPUT",
            )

        # 3. Finite numerical integrity (no NaN, Inf, -Inf)
        if not np.all(np.isfinite(probas)):
            logger.error("Inference output contains non-finite values (NaN/Inf).")
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="NON_FINITE_OUTPUT",
            )

        # 4. Range validation: each probability must be in [0.0, 1.0]
        if np.any(probas < 0.0) or np.any(probas > 1.0):
            logger.error("Inference output probabilities outside valid range [0.0, 1.0].")
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="INVALID_PROBABILITY_RANGE",
            )

        # 5. Row sum validation: row must sum to approximately 1.0 (tolerance 1e-3)
        row_sum = float(np.sum(probas))
        if abs(row_sum - 1.0) > 1e-3:
            logger.error("Inference output row does not sum to 1.0 (sum=%.4f).", row_sum)
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="INVALID_PROBABILITY_SUM",
            )

        if injection_idx is None or injection_idx >= 2:
            return MLInferenceResult(
                available=False,
                score=None,
                threshold=self.threshold,
                predicted_label=None,
                would_block=False,
                model_version=self._model_version,
                error_code="INDEX_OUT_OF_BOUNDS",
            )

        # 6. Decision logic: strictly compare unrounded probability against threshold
        raw_score = float(probas[0, injection_idx])
        would_block = bool(raw_score >= self.threshold)
        predicted_label: Literal["benign", "injection"] = "injection" if would_block else "benign"

        # Display score is rounded to four decimal places for presentation only;
        # rounding never alters would_block or predicted_label.
        display_score = round(raw_score, 4)

        return MLInferenceResult(
            available=True,
            score=display_score,
            threshold=self.threshold,
            predicted_label=predicted_label,
            would_block=would_block,
            model_version=self._model_version,
            error_code=None,
        )


_global_classifier: Optional[MLPromptInjectionClassifier] = None
_global_lock = threading.Lock()


def get_classifier() -> MLPromptInjectionClassifier:
    """Return the shared global ML classifier instance."""
    global _global_classifier
    with _global_lock:
        if _global_classifier is None:
            _global_classifier = MLPromptInjectionClassifier()
        return _global_classifier
