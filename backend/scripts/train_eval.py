"""PromptShield Phase 2 — Stage 2: Model Training and Evaluation Pipeline.

Ensures strict featurization ordering (fitted strictly on training partition),
validation-driven threshold selection, independent final holdout evaluation,
and complete provenance metadata reporting with evaluation integrity disclosures.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
import sklearn
import scipy


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for provenance tracking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def format_portable_path(filepath: Path, repo_root: Optional[Path] = None) -> str:
    """Format a dataset filepath for portable metadata representation.

    If the dataset file is located inside the repository root, returns a clean,
    forward-slash POSIX-style path relative to the repository root (e.g.,
    'datasets/prompts_dataset_v1.jsonl').

    If the dataset is outside the repository or path resolution fails, retains
    the resolved absolute path in normalized POSIX format so that provenance is
    documented without breaking training or falsely assuming repo locality.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    try:
        resolved_path = filepath.resolve()
        resolved_root = repo_root.resolve()
        rel_path = resolved_path.relative_to(resolved_root)
        return rel_path.as_posix()
    except (ValueError, RuntimeError):
        return filepath.resolve().as_posix()


def validate_records(
    records: List[Dict[str, Any]],
    source_name: str,
    allowed_splits: set,
) -> None:
    """Validate record structure, required fields, labels, and splits."""
    required_fields = {"id", "text", "label", "category", "split", "source"}
    valid_labels = {"benign", "injection"}

    seen_ids = set()
    seen_texts = set()

    for idx, record in enumerate(records, start=1):
        for field in required_fields:
            if field not in record or record[field] is None:
                raise ValueError(f"Missing or null field '{field}' in {source_name} at index {idx}")

        record_id = str(record["id"]).strip()
        if not record_id or record_id in seen_ids:
            raise ValueError(f"Duplicate or empty id '{record_id}' in {source_name} at index {idx}")
        seen_ids.add(record_id)

        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"Empty text in {source_name} for id '{record_id}'")
        norm_text = text.lower()
        if norm_text in seen_texts:
            raise ValueError(f"Duplicate text content in {source_name}: {text[:50]}...")
        seen_texts.add(norm_text)

        label = str(record["label"]).strip()
        if label not in valid_labels:
            raise ValueError(f"Invalid label '{label}' in {source_name} at index {idx}")

        split = str(record["split"]).strip()
        if split not in allowed_splits:
            raise ValueError(f"Invalid split '{split}' in {source_name}. Allowed: {allowed_splits}")


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSON Lines records from a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_pipeline() -> Pipeline:
    """Construct TF-IDF + Logistic Regression pipeline with fixed hyperparameters."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    C=1.0,
                    max_iter=1000,
                ),
            ),
        ]
    )


def select_blocking_threshold(
    y_val: List[str],
    val_scores: np.ndarray,
) -> Dict[str, Any]:
    """Empirically select decision/blocking threshold using validation split findings.

    Criterion:
    1. Primary Goal: Zero False Positives (FP == 0) on benign validation data to prevent
       disrupting legitimate enterprise user traffic.
    2. Secondary Goal: Subject to FP == 0, maximize Recall (minimize FN) on injection data.
    3. Tie-breaking: If multiple candidate thresholds achieve optimal performance (FP == 0, FN == 0),
       select the midpoint of the optimal candidate threshold interval for stability.
    4. Conservative Fallback: If no candidate achieves FP == 0, choose threshold with lowest FP
       and FPR <= 0.05 while maximizing F1; otherwise default conservatively to 0.50.
    """
    candidate_thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
    candidate_results: List[Dict[str, Any]] = []

    for tau in candidate_thresholds:
        preds = ["injection" if s >= tau else "benign" for s in val_scores]
        cm = confusion_matrix(y_val, preds, labels=["benign", "injection"])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        f1 = float(f1_score(y_val, preds, pos_label="injection", zero_division=0))
        prec = float(precision_score(y_val, preds, pos_label="injection", zero_division=0))
        rec = float(recall_score(y_val, preds, pos_label="injection", zero_division=0))
        fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
        candidate_results.append(
            {
                "threshold": float(tau),
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn": tn,
                "fpr": fpr,
                "f1_injection": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
        )

    zero_fp_candidates = [c for c in candidate_results if c["fp"] == 0]

    criterion_description = (
        "Empirical search over thresholds in [0.10, 0.90] with step 0.01 on validation split. "
        "Priority 1: Zero False Positives (FP == 0) on benign prompts to avoid false blocking. "
        "Priority 2: Maximize Recall on injection prompts (minimize FN). "
        "Tie-breaking: Select the median/midpoint of the optimal candidate interval. "
        "Fallback: If no zero-FP threshold exists, select lowest FP with FPR <= 0.05 maximizing F1; "
        "otherwise default to threshold 0.50 (or closest)."
    )

    if zero_fp_candidates:
        min_fn = min(c["fn"] for c in zero_fp_candidates)
        optimal_candidates = [c for c in zero_fp_candidates if c["fn"] == min_fn]
        mid_idx = len(optimal_candidates) // 2
        selected = optimal_candidates[mid_idx]
        selected_threshold = selected["threshold"]
        fallback_used = False
    else:
        fallback_used = True
        eligible_candidates = [c for c in candidate_results if c["fpr"] <= 0.05]
        if eligible_candidates:
            sorted_eligible = sorted(
                eligible_candidates,
                key=lambda c: (c["fp"], -c["f1_injection"]),
            )
            selected = sorted_eligible[0]
            selected_threshold = selected["threshold"]
            criterion_description += (
                " [FALLBACK APPLIED: Selected threshold with lowest FP and FPR <= 0.05 maximizing F1]"
            )
        else:
            candidates_at_half = [c for c in candidate_results if abs(c["threshold"] - 0.50) < 1e-4]
            if candidates_at_half:
                selected = candidates_at_half[0]
            else:
                selected = min(candidate_results, key=lambda c: abs(c["threshold"] - 0.50))
            selected_threshold = selected["threshold"]
            criterion_description += (
                " [FALLBACK APPLIED: No candidate achieved FPR <= 0.05; defaulted to candidate closest to 0.50]"
            )

    return {
        "selected_threshold": selected_threshold,
        "selection_criterion": criterion_description,
        "fallback_used": fallback_used,
        "validation_metrics_at_selected_threshold": selected,
        "candidate_sweep_sample": [
            c for c in candidate_results if round(c["threshold"] * 100) % 5 == 0
        ],
    }


def evaluate_split(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a set of predictions."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_benign = float(precision_score(y_true, y_pred, pos_label="benign", zero_division=0))
    rec_benign = float(recall_score(y_true, y_pred, pos_label="benign", zero_division=0))
    f1_benign = float(f1_score(y_true, y_pred, pos_label="benign", zero_division=0))

    prec_inj = float(precision_score(y_true, y_pred, pos_label="injection", zero_division=0))
    rec_inj = float(recall_score(y_true, y_pred, pos_label="injection", zero_division=0))
    f1_inj = float(f1_score(y_true, y_pred, pos_label="injection", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=["benign", "injection"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            "benign": {
                "precision": round(prec_benign, 4),
                "recall": round(rec_benign, 4),
                "f1_score": round(f1_benign, 4),
                "support": int(sum(1 for y in y_true if y == "benign")),
            },
            "injection": {
                "precision": round(prec_inj, 4),
                "recall": round(rec_inj, 4),
                "f1_score": round(f1_inj, 4),
                "support": int(sum(1 for y in y_true if y == "injection")),
            },
        },
        "confusion_matrix": {
            "true_negatives_benign": tn,
            "false_positives_falsely_flagged": fp,
            "false_negatives_missed_attacks": fn,
            "true_positives_detected_attacks": tp,
        },
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def extract_top_features(pipeline: Pipeline, top_n: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top positive (injection) and negative (benign) n-gram feature coefficients."""
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    top_pos_indices = np.argsort(coefficients)[-top_n:][::-1]
    top_neg_indices = np.argsort(coefficients)[:top_n]

    top_injection_features = [
        {"feature": str(feature_names[i]), "coefficient": round(float(coefficients[i]), 4)}
        for i in top_pos_indices
    ]
    top_benign_features = [
        {"feature": str(feature_names[i]), "coefficient": round(float(coefficients[i]), 4)}
        for i in top_neg_indices
    ]

    return {
        "top_injection_indicators": top_injection_features,
        "top_benign_indicators": top_benign_features,
    }


def train_and_evaluate(
    primary_dataset_path: Path,
    holdout_dataset_path: Path,
    output_dir: Path,
    model_version: str = "v1",
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute leakage-free training, validation threshold selection, and independent evaluation."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    print(f"[*] Step 1: Validating primary developmental dataset at {primary_dataset_path}...")
    primary_records = load_jsonl(primary_dataset_path)
    validate_records(primary_records, "primary_dataset", {"train", "validation", "test"})
    primary_hash = compute_file_hash(primary_dataset_path)
    print(f"    - Validated. Total: {len(primary_records)} records. SHA-256: {primary_hash[:16]}...")

    print(f"[*] Step 2: Validating independent final holdout dataset at {holdout_dataset_path}...")
    holdout_records = load_jsonl(holdout_dataset_path)
    validate_records(holdout_records, "holdout_dataset", {"independent_holdout"})
    holdout_hash = compute_file_hash(holdout_dataset_path)
    print(f"    - Validated. Total: {len(holdout_records)} records. SHA-256: {holdout_hash[:16]}...")

    # Verify no ID or text leakage between datasets
    prim_texts = {r["text"].strip().lower() for r in primary_records}
    hold_texts = {r["text"].strip().lower() for r in holdout_records}
    assert len(prim_texts & hold_texts) == 0, "Data leakage detected: overlapping prompts between datasets!"

    # Partition separation
    train_records = [r for r in primary_records if r["split"] == "train"]
    val_records = [r for r in primary_records if r["split"] == "validation"]
    exploratory_test_records = [r for r in primary_records if r["split"] == "test"]

    print(f"[*] Step 3: Partition sizes:")
    print(f"    - Training Partition:       {len(train_records)} (benign: {sum(1 for r in train_records if r['label'] == 'benign')}, injection: {sum(1 for r in train_records if r['label'] == 'injection')})")
    print(f"    - Validation Partition:     {len(val_records)} (benign: {sum(1 for r in val_records if r['label'] == 'benign')}, injection: {sum(1 for r in val_records if r['label'] == 'injection')})")
    print(f"    - Exploratory Test Split:   {len(exploratory_test_records)} (benign: {sum(1 for r in exploratory_test_records if r['label'] == 'benign')}, injection: {sum(1 for r in exploratory_test_records if r['label'] == 'injection')})")
    print(f"    - Independent Holdout:      {len(holdout_records)} (benign: {sum(1 for r in holdout_records if r['label'] == 'benign')}, injection: {sum(1 for r in holdout_records if r['label'] == 'injection')})")

    X_train = [r["text"] for r in train_records]
    y_train = [r["label"] for r in train_records]

    X_val = [r["text"] for r in val_records]
    y_val = [r["label"] for r in val_records]

    X_exp_test = [r["text"] for r in exploratory_test_records]
    y_exp_test = [r["label"] for r in exploratory_test_records]

    X_holdout = [r["text"] for r in holdout_records]
    y_holdout = [r["label"] for r in holdout_records]

    # Step 4: Fit strictly on training partition
    print("[*] Step 4: Fitting TF-IDF vectorizer and Logistic Regression strictly on TRAINING partition...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    classes = list(pipeline.classes_)  # ['benign', 'injection']
    inj_idx = classes.index("injection")

    # Step 5: Score validation data & select threshold
    print("[*] Step 5: Computing decision scores on VALIDATION partition & selecting threshold...")
    val_scores = pipeline.predict_proba(X_val)[:, inj_idx]
    threshold_info = select_blocking_threshold(y_val, val_scores)
    selected_threshold = threshold_info["selected_threshold"]
    print(f"    - Selected threshold: {selected_threshold:.2f}")
    print(f"    - Selection criterion: {threshold_info['selection_criterion']}")

    val_preds = ["injection" if s >= selected_threshold else "benign" for s in val_scores]
    val_metrics = evaluate_split(y_val, val_preds)

    # Step 6: Evaluate exploratory test split (with disclosure)
    print("[*] Step 6: Evaluating exploratory developmental test split...")
    exp_test_scores = pipeline.predict_proba(X_exp_test)[:, inj_idx]
    exp_test_preds = ["injection" if s >= selected_threshold else "benign" for s in exp_test_scores]
    exp_test_metrics = evaluate_split(y_exp_test, exp_test_preds)

    # Step 7: Evaluate genuinely independent final holdout
    print("[*] Step 7: Evaluating on UNINSPECTED INDEPENDENT FINAL HOLDOUT...")
    holdout_scores = pipeline.predict_proba(X_holdout)[:, inj_idx]
    holdout_preds = ["injection" if s >= selected_threshold else "benign" for s in holdout_scores]
    holdout_metrics = evaluate_split(y_holdout, holdout_preds)

    top_features = extract_top_features(pipeline, top_n=15)

    # Assemble metadata
    metadata = {
        "model_version": model_version,
        "artifact_file": f"prompt_injection_{model_version}.joblib",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "developmental_dataset": {
                "path": format_portable_path(primary_dataset_path, repo_root=repo_root),
                "sha256": primary_hash,
                "total_samples": len(primary_records),
                "split_counts": {
                    "train": {"total": len(train_records), "benign": sum(1 for r in train_records if r["label"] == "benign"), "injection": sum(1 for r in train_records if r["label"] == "injection")},
                    "validation": {"total": len(val_records), "benign": sum(1 for r in val_records if r["label"] == "benign"), "injection": sum(1 for r in val_records if r["label"] == "injection")},
                    "exploratory_test": {"total": len(exploratory_test_records), "benign": sum(1 for r in exploratory_test_records if r["label"] == "benign"), "injection": sum(1 for r in exploratory_test_records if r["label"] == "injection")},
                },
            },
            "independent_holdout_dataset": {
                "path": format_portable_path(holdout_dataset_path, repo_root=repo_root),
                "sha256": holdout_hash,
                "total_samples": len(holdout_records),
                "benign": sum(1 for r in holdout_records if r["label"] == "benign"),
                "injection": sum(1 for r in holdout_records if r["label"] == "injection"),
            },
        },
        "feature_configuration": {
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "max_features": 5000,
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "lowercase": True,
            "fitted_vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        },
        "classifier_configuration": {
            "algorithm": "LogisticRegression",
            "class_weight": "balanced",
            "random_state": 42,
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
        },
        "threshold_selection": {
            "selected_threshold": selected_threshold,
            "selection_criterion": threshold_info["selection_criterion"],
            "fallback_used": threshold_info["fallback_used"],
            "validation_metrics": val_metrics,
            "threshold_sweep_sample": threshold_info["candidate_sweep_sample"],
        },
        "exploratory_developmental_test_evaluation": {
            "integrity_disclosure": (
                "Individual prediction scores for this partition were displayed in exploratory logs during threshold development. "
                "Therefore, results on this split are reported as exploratory developmental metrics rather than an uninspected test."
            ),
            "sample_count": len(exploratory_test_records),
            "metrics": exp_test_metrics,
        },
        "independent_final_holdout_evaluation": {
            "evaluation_design": (
                "Genuinely new, independently authored synthetic evaluation set. "
                "Held strictly untouched and never inspected during model training or threshold tuning."
            ),
            "sample_count": len(holdout_records),
            "metrics": holdout_metrics,
        },
        "package_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "top_features": top_features,
        "score_nature_and_limitations": {
            "uncalibrated_scores": (
                "Logistic regression decision values are monotonic heuristic indicator scores derived from the sigmoid link function. "
                "They are uncalibrated and must not be interpreted as Bayesian probabilities of attack success or safety guarantees."
            ),
            "dataset_limitations": (
                "Trained on a synthetic benchmark dataset (84 train records). Evaluated on 18 validation, 18 exploratory test, and 20 independent holdout records. "
                "The small dataset cannot represent open-domain user variance or evolving multi-turn adversarial injection tactics. "
                "Deterministic rules remain sovereign and must always take precedence over ML predictions."
            ),
        },
    }

    # Step 8: Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = output_dir / f"prompt_injection_{model_version}.joblib"
    meta_path = output_dir / f"prompt_injection_{model_version}_meta.json"

    print(f"[*] Step 8: Saving model pipeline artifact to {model_artifact_path}...")
    joblib.dump(pipeline, model_artifact_path)

    print(f"[*] Step 9: Saving metadata report to {meta_path}...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 — STAGE 2: CORRECTED TRAINING & EVALUATION REPORT")
    print("=" * 70)
    print(f"Selected Threshold (via Validation): {selected_threshold:.2f}")
    print(f"Validation Performance:               F1={val_metrics['macro_f1']:.4f} | FP={val_metrics['confusion_matrix']['false_positives_falsely_flagged']} | FN={val_metrics['confusion_matrix']['false_negatives_missed_attacks']}")
    print("-" * 70)
    print("1. Exploratory Developmental Test Split (18 samples, previously observed):")
    print(f"   Accuracy: {exp_test_metrics['accuracy']:.4f} | Macro F1: {exp_test_metrics['macro_f1']:.4f} | FP: {exp_test_metrics['confusion_matrix']['false_positives_falsely_flagged']} | FN: {exp_test_metrics['confusion_matrix']['false_negatives_missed_attacks']}")
    print("-" * 70)
    print(f"2. Independent Final Holdout Set ({len(holdout_records)} samples: 10 benign, 10 injection):")
    print(f"   Accuracy:                {holdout_metrics['accuracy']:.4f}")
    print(f"   Macro F1-Score:          {holdout_metrics['macro_f1']:.4f}")
    print(f"   Benign Prec / Recall:    {holdout_metrics['per_class']['benign']['precision']:.4f} / {holdout_metrics['per_class']['benign']['recall']:.4f}")
    print(f"   Injection Prec / Recall: {holdout_metrics['per_class']['injection']['precision']:.4f} / {holdout_metrics['per_class']['injection']['recall']:.4f}")
    print(f"   Confusion Matrix:        TN={holdout_metrics['confusion_matrix']['true_negatives_benign']}, FP={holdout_metrics['confusion_matrix']['false_positives_falsely_flagged']}, FN={holdout_metrics['confusion_matrix']['false_negatives_missed_attacks']}, TP={holdout_metrics['confusion_matrix']['true_positives_detected_attacks']}")
    print(f"   False Positive Rate:     {holdout_metrics['false_positive_rate']:.4f}")
    print(f"   False Negative Rate:     {holdout_metrics['false_negative_rate']:.4f}")
    print("=" * 70 + "\n")

    return metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    dev_dataset = project_root / "datasets" / "prompts_dataset_v1.jsonl"
    holdout_dataset = project_root / "datasets" / "prompts_independent_holdout.jsonl"
    default_output_dir = project_root / "backend" / "app" / "detection" / "models"

    train_and_evaluate(dev_dataset, holdout_dataset, default_output_dir, repo_root=project_root)
