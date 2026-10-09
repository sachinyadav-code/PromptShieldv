# PromptShield Datasets and Evaluation Partitions

This directory contains the synthetic prompt injection and benign prompt datasets used to train, tune, and evaluate PromptShield's machine learning detectors.

---

## Evaluation Integrity & Dataset Design

To guarantee scientific and evaluation integrity, the evaluation framework distinguishes between exploratory developmental partitions and the uninspected final holdout:

1. **Developmental Dataset** (`prompts_dataset_v1.jsonl`):
   - **Training Partition (`train`)**: 84 records (42 benign, 42 injection). Used strictly to fit feature extraction (TF-IDF) and model parameters.
   - **Validation Partition (`validation`)**: 18 records (9 benign, 9 injection). Used strictly to select and tune the decision/blocking threshold ($\tau$).
   - **Exploratory Test Partition (`test`)**: 18 records (9 benign, 9 injection).
     > **Integrity Disclosure**: During Stage 2 development, individual decision scores for this partition were displayed in exploratory command-line logs prior to the final script execution. Therefore, this split is designated and reported as an **exploratory developmental test partition**, rather than an untouched holdout.

2. **Genuinely Independent Final Holdout** (`prompts_independent_holdout.jsonl`):
   - **Holdout Partition (`independent_holdout`)**: 20 completely new, independently authored records (10 benign, 10 injection).
   - Authored in entirely distinct domains and novel attack vectors (e.g. medical history, astrophysics, bio-informatics, authority impersonation, delimiter escapes, diagnostic probing, reverse ciphers).
   - **Zero Template Overlap**: Verified with Jaccard token similarity checks (no pair exceeded 0.65 Jaccard similarity with training/validation sets).
   - This partition was held strictly untouched and never used during model fitting or threshold selection.

---

## Partition Summary Table

| Dataset File | Partition | Records | Benign | Injection | Role & Evaluation Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `prompts_dataset_v1.jsonl` | `train` | **84** | 42 | 42 | Model featurization & training |
| `prompts_dataset_v1.jsonl` | `validation` | **18** | 9 | 9 | Threshold selection & hyperparameter tuning |
| `prompts_dataset_v1.jsonl` | `test` | **18** | 9 | 9 | Exploratory developmental test split |
| `prompts_independent_holdout.jsonl` | `independent_holdout` | **20** | 10 | 10 | **Independent final evaluation holdout** |
| **Total Benchmark** | | **140** | **70** | **70** | |

---

## Data Schema

Each JSON Lines record conforms to the following schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique identifier (e.g. `b-trn-001`, `ih-b-001`, `ih-i-001`) |
| `text` | `string` | Prompt text payload |
| `label` | `string` | Ground truth target: `benign` or `injection` |
| `category` | `string` | Subcategory for slice-based error analysis |
| `split` | `string` | Designated split (`train`, `validation`, `test`, `independent_holdout`) |
| `source` | `string` | Provenance tag (`synthetic_curated`, `independent_synthetic_evaluation`) |

---

## Integrity & Quality Checks

1. **Format & Schema Validation**: 100% of records in both files conform to schema without null or empty fields.
2. **Exact Duplicate Check**: 0 duplicate prompts exist within or across both files.
3. **Cross-Partition Leakage Check**: Maximum pairwise token Jaccard similarity between the independent holdout and the training/validation data is under 0.65 (zero near-duplicate template leakage).

---

---

## Phase 3 Local RAG Knowledge Corpus (`rag_corpus_v1.jsonl`)

The RAG document corpus is an offline, synthetic document collection designed to evaluate retrieval-augmented generation and indirect prompt injection defenses in Phase 3.

> **Crucial Distinction**:
> - Phase 2 datasets (`prompts_dataset_v1.jsonl` and `prompts_independent_holdout.jsonl`) are **prompt-level datasets** used strictly to train and evaluate the linear injection classifier.
> - `rag_corpus_v1.jsonl` is a **document-level corpus** representing external, untrusted knowledge sources retrieved during RAG execution. Documents in this corpus are treated as **untrusted data**, never as trusted system instructions.

### Corpus Summary (14 Synthetic Documents)

| Category | Count | Sample Doc IDs | Purpose & Characteristics |
| :--- | :---: | :--- | :--- |
| **Benign Technical Docs** | 2 | `doc-tech-001`, `doc-tech-002` | Real-world engineering architecture & database documentation. |
| **Mock Customer Support** | 2 | `doc-support-001`, `doc-support-002` | Customer refund rules and account/MFA troubleshooting guides. |
| **Enterprise Processes** | 2 | `doc-proc-001`, `doc-proc-002` | Post-mortem guidelines and hardware procurement refresh procedures. |
| **Cybersecurity Education** | 3 | `doc-sec-001` - `doc-sec-003` | Educational articles explaining prompt injection, Zero Trust, and credential security. Used to evaluate **false positive resilience**. |
| **Security Test Fixtures** | 5 | `doc-attack-001` - `doc-attack-005` | Labelled test fixtures with indirect injection in body, title, or metadata. Used to evaluate **indirect prompt injection detection**. |

### RAG Document Schema

Each line in `rag_corpus_v1.jsonl` is a valid JSON object with the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `doc_id` | `string` | Unique identifier (e.g. `doc-tech-001`, `doc-attack-001`). |
| `title` | `string` | Document title (untrusted, scanned for injection attempts). |
| `content` | `string` | Document text body (untrusted content). |
| `source` | `string` | Document origin identifier (e.g. `engineering_wiki`, `synthetic_security_fixtures`). |
| `tags` | `array[string]` | Keyword categorization tags (untrusted, included in token search). |
| `metadata` | `object` | Category, fixture flags, and attack type metadata. |

---

## Limitations

1. **Synthetic Nature**: All prompts and corpus documents are synthetically generated. Real-world adversaries use diverse, multi-turn, and evolving adversarial strategies that synthetic benchmarks cannot fully reflect.
2. **Uncalibrated Model Scores**: Model outputs from Logistic Regression are heuristic decision scores, not true Bayesian probabilities.
3. **Balanced vs. Production Distribution**: Both benchmarks maintain a balanced 50/50 class distribution for symmetric evaluation. Production traffic will feature severe class imbalance ($>99:1$ benign).
4. **Untrusted Data Boundary**: Presence in `rag_corpus_v1.jsonl` grants zero trust or authorization. Test fixture tags and metadata are for testing categorization only and never exempt documents from security inspection.
