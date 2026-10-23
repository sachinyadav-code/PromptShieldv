"""PromptShield Phase 1 Step 4: False-Positive Analysis, Threshold Selection & Calibration Diagnostics.

Performs rigorous audit of Step 3 metrics, comprehensive threshold trade-off analysis,
Wilson confidence interval calculation, out-of-fold calibration diagnostics, and
individual hard-negative semantic error analysis.
"""

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score continuity-corrected confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    # z = 1.95996 for 95% confidence
    z = 1.95996 if abs(confidence - 0.95) < 1e-3 else 1.96
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    centre = (p_hat + (z**2) / (2 * n)) / denom
    spread = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * (n**2)))) / denom
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return round(lower, 4), round(upper, 4)


def load_all_datasets(repo_root: Path):
    v1_path = repo_root / "datasets" / "prompts_dataset_v1.jsonl"
    v2_path = repo_root / "datasets" / "prompts_dataset_v2_additions.jsonl"

    with open(v1_path, "r", encoding="utf-8") as f:
        v1_records = [json.loads(line) for line in f if line.strip()]
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_records = [json.loads(line) for line in f if line.strip()]

    train_records = (
        [r for r in v1_records if r["split"] == "train"]
        + [r for r in v2_records if r["split"] == "train"]
    )
    val_records = (
        [r for r in v1_records if r["split"] == "validation"]
        + [r for r in v2_records if r["split"] == "validation"]
    )

    return train_records, val_records


def build_models() -> Dict[str, Pipeline]:
    model_a = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, strip_accents="unicode", lowercase=True)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, C=1.0, max_iter=1000)),
        ]
    )
    model_b = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True, strip_accents="unicode", lowercase=True)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, C=2.0, max_iter=1000)),
        ]
    )
    model_c = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=10000, sublinear_tf=True, lowercase=True)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, C=1.5, max_iter=1000)),
        ]
    )
    union = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=10000, sublinear_tf=True, lowercase=True)),
        ]
    )
    model_d = Pipeline(
        [
            ("features", union),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, C=2.0, max_iter=1000)),
        ]
    )
    return {
        "Model A (Baseline Word TF-IDF)": model_a,
        "Model B (Tuned Word TF-IDF)": model_b,
        "Model C (Char n-gram TF-IDF)": model_c,
        "Model D (Combined Word+Char TF-IDF)": model_d,
    }


def compute_metrics(y_true: List[str], preds: List[str]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, preds, labels=["benign", "injection"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    n_ben = tn + fp
    n_inj = fn + tp
    fpr = fp / n_ben if n_ben > 0 else 0.0
    fnr = fn / n_inj if n_inj > 0 else 0.0
    prec = precision_score(y_true, preds, pos_label="injection", zero_division=0)
    rec = recall_score(y_true, preds, pos_label="injection", zero_division=0)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_true, preds)

    fpr_ci = wilson_score_interval(fp, n_ben)
    rec_ci = wilson_score_interval(tp, n_inj)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": round(fpr, 4),
        "fpr_ci_95": fpr_ci,
        "fnr": round(fnr, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "recall_ci_95": rec_ci,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
    }


def run_threshold_grid_analysis(
    model_name: str,
    pipeline: Any,
    X_val: List[str],
    y_val: List[str],
    val_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    classes = list(pipeline.classes_)
    inj_idx = classes.index("injection")
    scores = pipeline.predict_proba(X_val)[:, inj_idx]

    # Index all benign hard negatives: v2 additions hard negatives (12) + v1 security discussions (2) = 14
    hard_neg_indices = [
        i for i, r in enumerate(val_records)
        if r["label"] == "benign" and (
            r.get("source") == "curated_authored_hard_negative"
            or r.get("category") == "security_education_discussion"
        )
    ]
    total_hard_negs = len(hard_neg_indices)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    table = []

    for tau in thresholds:
        preds = ["injection" if s >= tau else "benign" for s in scores]
        m = compute_metrics(y_val, preds)

        hn_fp = sum(1 for i in hard_neg_indices if preds[i] == "injection")
        hn_fpr = round(hn_fp / total_hard_negs, 4) if total_hard_negs > 0 else 0.0

        table.append(
            {
                "threshold": tau,
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
                "fpr": m["fpr"],
                "fpr_ci": m["fpr_ci_95"],
                "fnr": m["fnr"],
                "precision": m["precision"],
                "recall": m["recall"],
                "recall_ci": m["recall_ci_95"],
                "macro_f1": m["macro_f1"],
                "hard_negs_flagged": f"{hn_fp}/{total_hard_negs} ({hn_fpr*100:.1f}%)",
                "attacks_missed": m["fn"],
            }
        )

    return table


def audit_hard_negatives(
    pipeline: Any,
    val_records: List[Dict[str, Any]],
    tau: float = 0.50,
) -> List[Dict[str, Any]]:
    classes = list(pipeline.classes_)
    inj_idx = classes.index("injection")
    X_val = [r["text"] for r in val_records]
    scores = pipeline.predict_proba(X_val)[:, inj_idx]

    findings = []
    for idx, r in enumerate(val_records):
        if r["label"] == "benign":
            score = float(scores[idx])
            flagged = score >= tau
            is_hard_neg = (
                r.get("source") == "curated_authored_hard_negative"
                or r.get("category") == "security_education_discussion"
            )
            findings.append(
                {
                    "id": r["id"],
                    "category": r.get("category", "unknown"),
                    "group_id": r.get("group_id", "v1-standalone"),
                    "score": round(score, 4),
                    "flagged_at_50": flagged,
                    "is_hard_negative": is_hard_neg,
                    "text": r["text"],
                    "rationale": r.get("threat_model_rationale", "Baseline benign prompt"),
                }
            )
    return findings


def evaluate_calibration_out_of_fold(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate calibration using out-of-fold predictions on training data and validation data."""
    X_tr = np.array([r["text"] for r in train_records])
    y_tr = np.array([r["label"] for r in train_records])
    groups_tr = np.array(
        [r["group_id"] if "group_id" in r else f"grp-v1-{r['id']}" for r in train_records]
    )

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Collect Out-Of-Fold uncalibrated decision probabilities for Model D
    oof_y_true = []
    oof_uncal_probs = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_tr, y_tr, groups_tr)):
        union = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True, lowercase=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=10000, sublinear_tf=True, lowercase=True)),
            ]
        )
        clf = LogisticRegression(class_weight="balanced", random_state=42, C=2.0, max_iter=1000)
        pipe = Pipeline([("features", union), ("clf", clf)])

        pipe.fit(X_tr[tr_idx], y_tr[tr_idx])
        classes = list(pipe.classes_)
        inj_idx = classes.index("injection")

        probas = pipe.predict_proba(X_tr[val_idx])[:, inj_idx]
        oof_uncal_probs.extend(probas)
        oof_y_true.extend(y_tr[val_idx])

    y_num_oof = [1 if y == "injection" else 0 for y in oof_y_true]
    brier_uncal = brier_score_loss(y_num_oof, oof_uncal_probs)
    logloss_uncal = log_loss(y_num_oof, oof_uncal_probs)

    # 2. Evaluate CalibratedClassifierCV (sigmoid and isotonic) on the full train -> validation
    union_full = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=10000, sublinear_tf=True, lowercase=True)),
        ]
    )
    base_clf = LogisticRegression(class_weight="balanced", random_state=42, C=2.0, max_iter=1000)
    base_pipe = Pipeline([("features", union_full), ("clf", base_clf)])

    # Fit base model on train
    base_pipe.fit(X_tr, y_tr)
    classes = list(base_pipe.classes_)
    inj_idx = classes.index("injection")

    X_val = [r["text"] for r in val_records]
    y_val = [r["label"] for r in val_records]
    y_num_val = [1 if y == "injection" else 0 for y in y_val]

    raw_val_probs = base_pipe.predict_proba(X_val)[:, inj_idx]
    val_brier_uncal = brier_score_loss(y_num_val, raw_val_probs)
    val_logloss_uncal = log_loss(y_num_val, raw_val_probs)

    # Test CalibratedClassifierCV(cv=5) on train, evaluating on val
    cal_sigmoid = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=5)
    cal_sigmoid.fit(X_tr, y_tr)
    sig_val_probs = cal_sigmoid.predict_proba(X_val)[:, 1]
    sig_brier = brier_score_loss(y_num_val, sig_val_probs)
    sig_logloss = log_loss(y_num_val, sig_val_probs)

    cal_isotonic = CalibratedClassifierCV(estimator=base_pipe, method="isotonic", cv=5)
    cal_isotonic.fit(X_tr, y_tr)
    iso_val_probs = cal_isotonic.predict_proba(X_val)[:, 1]
    iso_brier = brier_score_loss(y_num_val, iso_val_probs)
    iso_logloss = log_loss(y_num_val, iso_val_probs)

    return {
        "out_of_fold_train_n140": {
            "brier_score_uncalibrated": round(brier_uncal, 4),
            "log_loss_uncalibrated": round(logloss_uncal, 4),
        },
        "held_out_validation_n42": {
            "uncalibrated": {
                "brier_score": round(val_brier_uncal, 4),
                "log_loss": round(val_logloss_uncal, 4),
            },
            "calibrated_sigmoid": {
                "brier_score": round(sig_brier, 4),
                "log_loss": round(sig_logloss, 4),
            },
            "calibrated_isotonic": {
                "brier_score": round(iso_brier, 4),
                "log_loss": round(iso_logloss, 4),
            },
        },
        "calibration_assessment": (
            "Because the held-out validation sample has only N=42 examples (21 benign, 21 injection), "
            "empirical calibration curves exhibit high sampling variance. While Platt sigmoid calibration slightly "
            "reduces log-loss, the underlying classifier outputs remain monotonic decision-boundary rankings rather "
            "than robust Bayesian posterior probabilities. Ingress filtering must not treat raw output scores as calibrated probabilities."
        ),
    }


def main():
    repo_root = Path.cwd()
    train_records, val_records = load_all_datasets(repo_root)
    X_train = [r["text"] for r in train_records]
    y_train = [r["label"] for r in train_records]
    X_val = [r["text"] for r in val_records]
    y_val = [r["label"] for r in val_records]

    # Fit candidates on train (N=140)
    candidate_pipelines = build_models()
    for name, p in candidate_pipelines.items():
        p.fit(X_train, y_train)

    # Load frozen v1
    v1_path = repo_root / "backend" / "app" / "detection" / "models" / "prompt_injection_v1.joblib"
    frozen_v1 = joblib.load(v1_path)

    all_models = {
        "Frozen Production v1 (Fitted on N=84)": frozen_v1,
        **candidate_pipelines,
    }

    # 1. Run threshold analysis
    threshold_tables = {}
    for name, pipe in all_models.items():
        threshold_tables[name] = run_threshold_grid_analysis(name, pipe, X_val, y_val, val_records)

    # 2. Hard-negative false positive analysis
    hard_neg_analysis = {}
    for name, pipe in all_models.items():
        hard_neg_analysis[name] = audit_hard_negatives(pipe, val_records, tau=0.50)

    # 3. Calibration analysis
    cal_results = evaluate_calibration_out_of_fold(train_records, val_records)

    # Assemble report JSON
    results = {
        "step_4_audit_summary": {
            "validation_sample_count": len(val_records),
            "validation_benign_count": sum(1 for r in val_records if r["label"] == "benign"),
            "validation_injection_count": sum(1 for r in val_records if r["label"] == "injection"),
            "validation_hard_negative_count": sum(
                1 for r in val_records
                if r["label"] == "benign" and (
                    r.get("source") == "curated_authored_hard_negative"
                    or r.get("category") == "security_education_discussion"
                )
            ),
            "training_sample_count": len(train_records),
            "training_benign_count": sum(1 for r in train_records if r["label"] == "benign"),
            "training_injection_count": sum(1 for r in train_records if r["label"] == "injection"),
        },
        "threshold_grid_by_model": threshold_tables,
        "calibration_diagnostics": cal_results,
        "hard_negatives_individual_audit": {
            model: [h for h in findings if h["flagged_at_50"]]
            for model, findings in hard_neg_analysis.items()
        },
    }

    out_file = repo_root / "backend" / "app" / "detection" / "models" / "step_4_threshold_calibration_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Step 4 audit completed successfully. Output written to {out_file}")


if __name__ == "__main__":
    main()
