"""PromptShield Phase 1 Step 3: Leakage-Free Cross-Validation and Model Exploration.

Compares four lightweight prompt-injection detection models via 5-fold StratifiedGroupKFold
on the expanded developmental training partition (N=140), evaluates candidates on the held-out
validation partition (N=42), and tests the existing production v1 artifact for baseline comparison.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


def compute_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_dataset(repo_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    v1_path = repo_root / "datasets" / "prompts_dataset_v1.jsonl"
    v2_path = repo_root / "datasets" / "prompts_dataset_v2_additions.jsonl"

    with open(v1_path, "r", encoding="utf-8") as f:
        v1_records = [json.loads(line) for line in f if line.strip()]
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_records = [json.loads(line) for line in f if line.strip()]

    # Filter into training and validation partitions
    train_records = (
        [r for r in v1_records if r["split"] == "train"]
        + [r for r in v2_records if r["split"] == "train"]
    )
    val_records = (
        [r for r in v1_records if r["split"] == "validation"]
        + [r for r in v2_records if r["split"] == "validation"]
    )
    exp_test_records = [r for r in v1_records if r["split"] == "test"]

    return train_records, val_records, exp_test_records


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """Define the 4 candidate model pipelines."""
    # Model A: Baseline Word TF-IDF + LogisticRegression
    model_a = Pipeline(
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

    # Model B: Tuned Word TF-IDF + LogisticRegression (with min_df=2 and tuned C=2.0)
    model_b = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
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
                    C=2.0,
                    max_iter=1000,
                ),
            ),
        ]
    )

    # Model C: Character-level TF-IDF (char_wb 3-5) + LogisticRegression
    model_c = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=10000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    C=1.5,
                    max_iter=1000,
                ),
            ),
        ]
    )

    # Model D: Combined Word + Char FeatureUnion + LogisticRegression
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=5000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=10000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ]
    )
    model_d = Pipeline(
        [
            ("features", union),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    C=2.0,
                    max_iter=1000,
                ),
            ),
        ]
    )

    return {
        "Model A (Baseline Word TF-IDF)": model_a,
        "Model B (Tuned Word TF-IDF)": model_b,
        "Model C (Char n-gram TF-IDF)": model_c,
        "Model D (Combined Word+Char TF-IDF)": model_d,
    }


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_inj = float(precision_score(y_true, y_pred, pos_label="injection", zero_division=0))
    rec_inj = float(recall_score(y_true, y_pred, pos_label="injection", zero_division=0))
    f1_inj = float(f1_score(y_true, y_pred, pos_label="injection", zero_division=0))

    prec_ben = float(precision_score(y_true, y_pred, pos_label="benign", zero_division=0))
    rec_ben = float(recall_score(y_true, y_pred, pos_label="benign", zero_division=0))
    f1_ben = float(f1_score(y_true, y_pred, pos_label="benign", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=["benign", "injection"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "prec_inj": prec_inj,
        "rec_inj": rec_inj,
        "f1_inj": f1_inj,
        "prec_ben": prec_ben,
        "rec_ben": rec_ben,
        "f1_ben": f1_ben,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "fnr": fnr,
    }


def run_cross_validation(
    train_records: List[Dict[str, Any]],
    pipelines_factory: Any,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute leakage-free 5-Fold StratifiedGroupKFold on train records."""
    X = np.array([r["text"] for r in train_records])
    y = np.array([r["label"] for r in train_records])
    groups = np.array(
        [r["group_id"] if "group_id" in r else f"grp-v1-{r['id']}" for r in train_records]
    )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results: Dict[str, List[Dict[str, float]]] = {}
    models = pipelines_factory()
    for name in models:
        cv_results[name] = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        for name, pipe in pipelines_factory().items():
            pipe.fit(X_tr, y_tr)
            classes = list(pipe.classes_)
            inj_idx = classes.index("injection")
            scores = pipe.predict_proba(X_vl)[:, inj_idx]
            preds = ["injection" if s >= 0.50 else "benign" for s in scores]
            metrics = evaluate_predictions(list(y_vl), preds)
            cv_results[name].append(metrics)

    # Summarize mean and std across folds
    summary: Dict[str, Any] = {}
    for name, fold_metrics in cv_results.items():
        macro_f1s = [m["macro_f1"] for m in fold_metrics]
        prec_injs = [m["prec_inj"] for m in fold_metrics]
        rec_injs = [m["rec_inj"] for m in fold_metrics]
        f1_injs = [m["f1_inj"] for m in fold_metrics]
        prec_bens = [m["prec_ben"] for m in fold_metrics]
        rec_bens = [m["rec_ben"] for m in fold_metrics]
        f1_bens = [m["f1_ben"] for m in fold_metrics]
        accuracies = [m["accuracy"] for m in fold_metrics]

        total_tn = sum(m["tn"] for m in fold_metrics)
        total_fp = sum(m["fp"] for m in fold_metrics)
        total_fn = sum(m["fn"] for m in fold_metrics)
        total_tp = sum(m["tp"] for m in fold_metrics)
        agg_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        agg_fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0

        summary[name] = {
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "inj_precision_mean": float(np.mean(prec_injs)),
            "inj_precision_std": float(np.std(prec_injs)),
            "inj_recall_mean": float(np.mean(rec_injs)),
            "inj_recall_std": float(np.std(rec_injs)),
            "inj_f1_mean": float(np.mean(f1_injs)),
            "inj_f1_std": float(np.std(f1_injs)),
            "ben_precision_mean": float(np.mean(prec_bens)),
            "ben_precision_std": float(np.std(prec_bens)),
            "ben_recall_mean": float(np.mean(rec_bens)),
            "ben_recall_std": float(np.std(rec_bens)),
            "ben_f1_mean": float(np.mean(f1_bens)),
            "ben_f1_std": float(np.std(f1_bens)),
            "aggregate_confusion_matrix": {"tn": total_tn, "fp": total_fp, "fn": total_fn, "tp": total_tp},
            "aggregate_fpr": float(agg_fpr),
            "aggregate_fnr": float(agg_fnr),
            "per_fold": fold_metrics,
        }

    return summary


def evaluate_on_validation_set(
    fitted_pipelines: Dict[str, Pipeline],
    val_records: List[Dict[str, Any]],
    production_v1_pipeline: Any = None,
) -> Dict[str, Any]:
    """Evaluate fitted candidate models and production v1 on the held-out validation set (N=42)."""
    X_val = [r["text"] for r in val_records]
    y_val = [r["label"] for r in val_records]

    # Identify hard-negative indices (benign records from v2 additions: quoted analysis & sysprompt safety)
    hard_neg_indices = [
        i for i, r in enumerate(val_records)
        if r["label"] == "benign" and r.get("source") == "curated_authored_hard_negative"
    ]

    results: Dict[str, Any] = {}

    all_models = dict(fitted_pipelines)
    if production_v1_pipeline is not None:
        all_models["Production v1 Model (Frozen Artifact)"] = production_v1_pipeline

    for name, pipe in all_models.items():
        classes = list(pipe.classes_)
        inj_idx = classes.index("injection")
        scores = pipe.predict_proba(X_val)[:, inj_idx]

        # Standard threshold tau = 0.50 evaluation
        preds_50 = ["injection" if s >= 0.50 else "benign" for s in scores]
        eval_50 = evaluate_predictions(y_val, preds_50)

        # Hard negatives slice at tau = 0.50
        hn_scores = [scores[i] for i in hard_neg_indices]
        hn_preds = [preds_50[i] for i in hard_neg_indices]
        hn_fp = sum(1 for p in hn_preds if p == "injection")
        hn_total = len(hard_neg_indices)
        hn_fpr = hn_fp / hn_total if hn_total > 0 else 0.0

        # Optimal threshold sweep on validation
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
        candidates = []
        for t in thresholds:
            p_t = ["injection" if s >= t else "benign" for s in scores]
            ev = evaluate_predictions(y_val, p_t)
            candidates.append({"tau": float(t), **ev})

        zero_fp = [c for c in candidates if c["fp"] == 0]
        if zero_fp:
            best_fn = min(c["fn"] for c in zero_fp)
            optimal = [c for c in zero_fp if c["fn"] == best_fn]
            selected_tau = optimal[len(optimal) // 2]["tau"]
            best_metrics = optimal[len(optimal) // 2]
        else:
            selected_tau = 0.50
            best_metrics = eval_50

        results[name] = {
            "metrics_at_default_50": eval_50,
            "hard_negative_slice": {
                "total_hard_negatives": hn_total,
                "false_positives": hn_fp,
                "fpr": hn_fpr,
                "individual_scores": [
                    {"id": val_records[i]["id"], "score": round(float(scores[i]), 4), "text": val_records[i]["text"][:60]}
                    for i in hard_neg_indices
                ],
            },
            "optimal_threshold": {
                "selected_tau": selected_tau,
                "metrics": best_metrics,
            },
            "threshold_sweep_sample": [c for c in candidates if round(c["tau"] * 100) % 10 == 0],
        }

    return results


def measure_inference_and_size(pipeline: Pipeline, sample_texts: List[str]) -> Dict[str, Any]:
    # Warmup
    for _ in range(50):
        pipeline.predict_proba(sample_texts[:5])

    # Benchmark 100 iterations of 10 prompts
    batch = sample_texts[:10]
    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        pipeline.predict_proba(batch)
    elapsed = time.perf_counter() - start

    total_prompts = n_iters * len(batch)
    micros_per_prompt = (elapsed / total_prompts) * 1_000_000

    return {
        "micros_per_prompt": round(micros_per_prompt, 2),
        "prompts_per_second": round(total_prompts / elapsed, 1),
    }


def main():
    repo_root = Path.cwd()
    print("[*] Loading datasets...")
    train_records, val_records, exp_test_records = load_dataset(repo_root)

    print(f"    - Training Partition:   {len(train_records)} ({Counter(r['label'] for r in train_records)})")
    print(f"    - Validation Partition: {len(val_records)} ({Counter(r['label'] for r in val_records)})")
    print(f"    - Exploratory Test:     {len(exp_test_records)} ({Counter(r['label'] for r in exp_test_records)})")

    # 1. Run 5-Fold StratifiedGroupKFold on train records
    print("\n[*] Running 5-Fold StratifiedGroupKFold Cross-Validation on Training Partition (N=140)...")
    cv_summary = run_cross_validation(train_records, build_candidate_pipelines, n_splits=5, random_state=42)

    for name, res in cv_summary.items():
        print(f"\n--- {name} ---")
        print(f"  Macro F1:    {res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f}")
        print(f"  Accuracy:    {res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f}")
        print(f"  Inj Recall:  {res['inj_recall_mean']:.4f} +/- {res['inj_recall_std']:.4f} | Inj Prec: {res['inj_precision_mean']:.4f} +/- {res['inj_precision_std']:.4f}")
        print(f"  Agg CM:      TN={res['aggregate_confusion_matrix']['tn']}, FP={res['aggregate_confusion_matrix']['fp']}, FN={res['aggregate_confusion_matrix']['fn']}, TP={res['aggregate_confusion_matrix']['tp']}")
        print(f"  Agg FPR:     {res['aggregate_fpr']:.4f} ({res['aggregate_confusion_matrix']['fp']}/{res['aggregate_confusion_matrix']['tn'] + res['aggregate_confusion_matrix']['fp']})")
        print(f"  Agg FNR:     {res['aggregate_fnr']:.4f} ({res['aggregate_confusion_matrix']['fn']}/{res['aggregate_confusion_matrix']['fn'] + res['aggregate_confusion_matrix']['tp']})")

    # 2. Fit candidate pipelines on full training partition (N=140)
    print("\n[*] Fitting Candidate Pipelines on Full Training Partition (N=140)...")
    X_train = [r["text"] for r in train_records]
    y_train = [r["label"] for r in train_records]

    fitted_pipelines = build_candidate_pipelines()
    for name, pipe in fitted_pipelines.items():
        pipe.fit(X_train, y_train)

    # 3. Load production v1 model for baseline comparison
    v1_artifact_path = repo_root / "backend" / "app" / "detection" / "models" / "prompt_injection_v1.joblib"
    prod_v1 = joblib.load(v1_artifact_path)

    # 4. Evaluate on Held-out Development Validation Set (N=42)
    print("\n[*] Evaluating Candidates & Production v1 on Held-out Validation Partition (N=42)...")
    val_results = evaluate_on_validation_set(fitted_pipelines, val_records, production_v1_pipeline=prod_v1)

    for name, vres in val_results.items():
        m50 = vres["metrics_at_default_50"]
        hn = vres["hard_negative_slice"]
        opt = vres["optimal_threshold"]
        print(f"\n--- {name} (Validation Set N=42) ---")
        print(f"  Default (tau=0.50): Macro F1={m50['macro_f1']:.4f} | FP={m50['fp']} | FN={m50['fn']} | FPR={m50['fpr']:.4f}")
        print(f"  Hard-Negatives Slice (N={hn['total_hard_negatives']}): FP={hn['false_positives']} | FPR={hn['fpr']:.4f}")
        print(f"  Optimal Threshold (tau={opt['selected_tau']:.2f}): Macro F1={opt['metrics']['macro_f1']:.4f} | FP={opt['metrics']['fp']} | FN={opt['metrics']['fn']}")

    # 5. Benchmark inference latency
    print("\n[*] Measuring Inference Latency...")
    sample_texts = [r["text"] for r in val_records]
    latency_results = {}
    for name, pipe in fitted_pipelines.items():
        lat = measure_inference_and_size(pipe, sample_texts)
        latency_results[name] = lat
        print(f"  {name}: {lat['micros_per_prompt']} us/prompt ({lat['prompts_per_second']} prompts/sec)")

    # 6. Select winning candidate and save to candidate path (never overwriting v1)
    # Model D (Combined Word+Char TF-IDF) is our primary candidate: superior character resilience and zero hard-negative FP
    winning_model_name = "Model D (Combined Word+Char TF-IDF)"
    winning_pipeline = fitted_pipelines[winning_model_name]
    winning_val = val_results[winning_model_name]

    models_dir = repo_root / "backend" / "app" / "detection" / "models"
    candidate_joblib_path = models_dir / "prompt_injection_v2_candidate.joblib"
    candidate_meta_path = models_dir / "prompt_injection_v2_candidate_meta.json"

    print(f"\n[*] Saving winning candidate ({winning_model_name}) to {candidate_joblib_path}...")
    joblib.dump(winning_pipeline, candidate_joblib_path)
    artifact_size_bytes = candidate_joblib_path.stat().st_size

    # Prepare candidate metadata
    v1_hash = compute_hash(repo_root / "datasets" / "prompts_dataset_v1.jsonl")
    v2_add_hash = compute_hash(repo_root / "datasets" / "prompts_dataset_v2_additions.jsonl")

    # Extract top features
    union_step = winning_pipeline.named_steps["features"]
    clf_step = winning_pipeline.named_steps["clf"]
    feature_names = np.array(union_step.get_feature_names_out())
    coeffs = clf_step.coef_[0]

    top_pos_idx = np.argsort(coeffs)[-15:][::-1]
    top_neg_idx = np.argsort(coeffs)[:15]

    top_features = {
        "top_injection_indicators": [
            {"feature": str(feature_names[i]), "coefficient": round(float(coeffs[i]), 4)}
            for i in top_pos_idx
        ],
        "top_benign_indicators": [
            {"feature": str(feature_names[i]), "coefficient": round(float(coeffs[i]), 4)}
            for i in top_neg_idx
        ],
    }

    metadata = {
        "candidate_version": "v2_candidate",
        "model_name": winning_model_name,
        "artifact_file": "prompt_injection_v2_candidate.joblib",
        "artifact_size_bytes": artifact_size_bytes,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": {
            "v1_baseline_dataset": {
                "path": "datasets/prompts_dataset_v1.jsonl",
                "sha256": v1_hash,
                "train_samples_used": sum(1 for r in train_records if r["id"].startswith("b-") or r["id"].startswith("i-")),
            },
            "v2_additions_dataset": {
                "path": "datasets/prompts_dataset_v2_additions.jsonl",
                "sha256": v2_add_hash,
                "train_samples_used": sum(1 for r in train_records if r["id"].startswith("v2-")),
            },
            "total_training_samples": len(train_records),
            "class_distribution_train": dict(Counter(r["label"] for r in train_records)),
            "validation_samples": len(val_records),
            "class_distribution_validation": dict(Counter(r["label"] for r in val_records)),
        },
        "cross_validation_5fold_stratified_group": cv_summary[winning_model_name],
        "validation_evaluation": winning_val,
        "model_comparison_summary": {
            m: {
                "cv_macro_f1": f"{cv_summary[m]['macro_f1_mean']:.4f} +/- {cv_summary[m]['macro_f1_std']:.4f}",
                "cv_agg_fpr": cv_summary[m]["aggregate_fpr"],
                "cv_agg_fnr": cv_summary[m]["aggregate_fnr"],
                "val_macro_f1_default": val_results[m]["metrics_at_default_50"]["macro_f1"],
                "val_hard_negative_fp": val_results[m]["hard_negative_slice"]["false_positives"],
                "val_optimal_tau": val_results[m]["optimal_threshold"]["selected_tau"],
                "val_optimal_macro_f1": val_results[m]["optimal_threshold"]["metrics"]["macro_f1"],
                "latency_us_per_prompt": latency_results[m]["micros_per_prompt"],
            }
            for m in fitted_pipelines
        },
        "top_features": top_features,
        "status": "candidate_unpromoted",
        "limitations_and_disclosures": {
            "unpromoted": "This is an experimental candidate trained for Phase 1 Step 3 evaluation. It has NOT been promoted into production and production inference contracts remain unchanged.",
            "uninspected_holdout_disclosure": "The independent holdout dataset (prompts_independent_holdout.jsonl) was NOT used in any part of training, cross-validation, feature selection, or threshold tuning. As disclosed in Phase 2 baseline metadata, the holdout was evaluated in previous historical runs, so it constitutes a fixed independent benchmark rather than a blind holdout.",
            "uncalibrated_scores": "Logistic regression decision values are monotonic heuristic indicator scores derived from the sigmoid link function. They are uncalibrated and must not be interpreted as Bayesian probabilities.",
        },
    }

    print(f"[*] Writing candidate metadata to {candidate_meta_path}...")
    with open(candidate_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Verification of candidate reload
    print("[*] Verifying candidate artifact reload and consistency...")
    reloaded = joblib.load(candidate_joblib_path)
    sample_preds_orig = winning_pipeline.predict_proba(sample_texts[:5])
    sample_preds_reloaded = reloaded.predict_proba(sample_texts[:5])
    assert np.allclose(sample_preds_orig, sample_preds_reloaded), "Artifact reload produced inconsistent probabilities!"
    print("    - Candidate artifact reload verified successfully.")


if __name__ == "__main__":
    main()
