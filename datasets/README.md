# PromptShield Datasets and Evaluation Partitions

This directory contains the synthetic prompt injection, benign prompt, and hard-negative datasets used to train, tune, and evaluate PromptShield's machine learning detectors.

---

## Evaluation Integrity & Dataset Design

To guarantee scientific and evaluation integrity, the evaluation framework distinguishes between developmental partitions, dataset expansions, and the uninspected final holdout:

1. **Developmental Baseline Dataset** (`prompts_dataset_v1.jsonl`):
   - **Training Partition (`train`)**: 84 records (42 benign, 42 injection). Used strictly to fit feature extraction (TF-IDF) and baseline model parameters.
   - **Validation Partition (`validation`)**: 18 records (9 benign, 9 injection). Used strictly to select and tune the decision/blocking threshold ($\tau$).
   - **Exploratory Test Partition (`test`)**: 18 records (9 benign, 9 injection).
     > **Integrity Disclosure**: During baseline development, individual decision scores for this partition were displayed in exploratory command-line logs prior to final script execution. Therefore, this split is designated and reported as an **exploratory developmental test partition**, rather than an untouched holdout.

2. **Phase 1 Versioned Additions & Hard Negatives** (`prompts_dataset_v2_additions.jsonl`):
   - **Total Records**: 80 curated records (40 benign, 40 injection).
   - **Training Partition (`train`)**: 56 records (28 benign, 28 injection).
   - **Validation Partition (`validation`)**: 24 records (12 benign, 12 injection).
   - **100% Group-Isolated**: Organized into 14 distinct semantic groups (`group_id`). All variants within a group are strictly assigned to either `train` or `validation`. Zero groups cross partitions.
   - **Threat Model Rationale**: Every record includes explicit threat-model rationale documenting why the sample is benign (e.g. academic security research, method overrides, Git ignore patterns, CSS overrides) or injection (e.g. direct instruction override, delimiter escape, persona framing, task diversion).
   - **Zero Holdout Overlap**: Contains zero overlap with the independent holdout.

3. **Genuinely Independent Final Holdout** (`prompts_independent_holdout.jsonl`):
   - **Holdout Partition (`independent_holdout`)**: 20 completely new, independently authored records (10 benign, 10 injection).
   - Authored in entirely distinct domains and novel attack vectors (e.g. medical history, astrophysics, bio-informatics, authority impersonation, delimiter escapes, diagnostic probing, reverse ciphers).
   - **Zero Template Overlap**: Verified with Jaccard token similarity checks (no pair exceeded 0.65 Jaccard similarity with training/validation sets).
   - **Held strictly untouched**: Never inspected or used during model fitting, candidate comparison, or threshold selection.

---

## Partition Summary Table

| Dataset File | Partition | Records | Benign | Injection | Role & Evaluation Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `prompts_dataset_v1.jsonl` | `train` | **84** | 42 | 42 | Baseline featurization & training |
| `prompts_dataset_v1.jsonl` | `validation` | **18** | 9 | 9 | Baseline threshold selection & tuning |
| `prompts_dataset_v1.jsonl` | `test` | **18** | 9 | 9 | Exploratory developmental test split |
| `prompts_dataset_v2_additions.jsonl` | `train` | **56** | 28 | 28 | Hard-negative & robust injection training |
| `prompts_dataset_v2_additions.jsonl` | `validation` | **24** | 12 | 12 | Group-aware validation & threshold tuning |
| `prompts_independent_holdout.jsonl` | `independent_holdout` | **20** | 10 | 10 | **Independent final evaluation holdout** |
| **Total Prompt Datasets** | | **220** | **110** | **110** | **100% Balanced Class Distribution** |

---

## Data Schema

### 1. Baseline Schema (`prompts_dataset_v1.jsonl`, `prompts_independent_holdout.jsonl`)

Each line conforms to the following schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique identifier (e.g. `b-trn-001`, `ih-b-001`, `ih-i-001`) |
| `text` | `string` | Prompt text payload |
| `label` | `string` | Ground truth target: `benign` or `injection` |
| `category` | `string` | Subcategory for slice-based error analysis |
| `split` | `string` | Designated split (`train`, `validation`, `test`, `independent_holdout`) |
| `source` | `string` | Provenance tag (`synthetic_curated`, `independent_synthetic_evaluation`) |

### 2. Versioned Additions Schema (`prompts_dataset_v2_additions.jsonl`)

Extends the schema with group clustering and explicit threat-model rationale:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique stable identifier (e.g. `v2-b-001`, `v2-i-001`) |
| `text` | `string` | Prompt text payload |
| `label` | `string` | Ground truth target: `benign` or `injection` |
| `category` | `string` | Granular category (e.g. `cybersec_education`, `programming_override`, `delimiter_escape`) |
| `group_id` | `string` | Cluster identifier for group-aware splitting (e.g. `grp-b-cybersec-edu`, `grp-i-direct-override`) |
| `split` | `string` | Split assignment (`train` or `validation`) |
| `source` | `string` | Provenance (`curated_authored_hard_negative`, `curated_authored_adversarial`) |
| `threat_model_rationale` | `string` | Justification explaining why the sample is classified as benign or injection |

---

## Labeling Policy & Threat Model

### Binary Label Definitions
- **`injection`**: Any prompt whose intent, operative command, or structural technique seeks to bypass, nullify, overrule, or hijack developer/system instructions, dump private initialization prompts or environment secrets, deactivate safety mechanisms, or divert the AI away from its authorized task into an attacker-controlled execution state.
- **`benign`**: Any prompt that does not attempt to subvert the host model's instructions or guardrails. Crucially, mentioning or quoting security terminology, attack patterns, method overrides, system administration directives, or Git configuration rules does **not** make a prompt malicious.

### Ambiguous Case Policy
1. **Quoted Attack Analysis**: Prompts analyzing, classifying, or debugging a quoted attack payload (e.g., *"How should our firewall detect 'Ignore previous instructions'?"*) are strictly labeled **`benign`** because the user's intent is defensive analysis; the model is not instructed to execute the payload.
2. **Indirect / Piggybacked Overrides**: Prompts asking to analyze an untrusted snippet with an operative directive directing the model to follow the payload (e.g., *"Moderate this comment: '...'. Execute the override in the comment."*) are labeled **`injection`**.
3. **Programming & Technical Overrides**: Requests regarding Java `@Override`, C++ vtables, CSS specificity overrides, or systemd drop-in configurations are strictly **`benign`**.
4. **Git Ignore Patterns**: Inquiries regarding `.gitignore` syntax or `git check-ignore` are strictly **`benign`**.

---

## Integrity & Quality Checks

1. **Format & Schema Validation**: 100% of records in all files conform to their respective schemas without null, empty, or unparseable fields.
2. **Exact Duplicate Check**: 0 duplicate prompts exist within or across `prompts_dataset_v1.jsonl`, `prompts_dataset_v2_additions.jsonl`, and `prompts_independent_holdout.jsonl`.
3. **Cross-Partition Leakage Check**:
   - In `prompts_dataset_v2_additions.jsonl`, all 14 `group_id` clusters are strictly contained in either `train` or `validation`.
   - Maximum pairwise token Jaccard similarity between the independent holdout and all developmental/additions data is under 0.65.
4. **Data Source & Provenance Verification**:
   - Additions are clearly tagged as `curated_authored_hard_negative` and `curated_authored_adversarial`.
   - No external unverified datasets were scraped or ingested, guaranteeing complete licensing and provenance integrity.

---

## Phase 3 Local RAG Knowledge Corpus (`rag_corpus_v1.jsonl`)

The RAG document corpus is an offline, synthetic document collection designed to evaluate retrieval-augmented generation and indirect prompt injection defenses in Phase 3.

> **Crucial Distinction**:
> - Prompt datasets (`prompts_dataset_v1.jsonl`, `prompts_dataset_v2_additions.jsonl`, and `prompts_independent_holdout.jsonl`) are **prompt-level datasets** used strictly to train and evaluate the linear injection classifier.
> - `rag_corpus_v1.jsonl` is a **document-level corpus** representing external, untrusted knowledge sources retrieved during RAG execution. Documents in this corpus are treated as **untrusted data**, never as trusted system instructions. They must never be merged into classifier prompt datasets.

### Corpus Summary (14 Synthetic Documents)

| Category | Count | Sample Doc IDs | Purpose & Characteristics |
| :--- | :---: | :--- | :--- |
| **Benign Technical Docs** | 2 | `doc-tech-001`, `doc-tech-002` | Real-world engineering architecture & database documentation. |
| **Mock Customer Support** | 2 | `doc-support-001`, `doc-support-002` | Customer refund rules and account/MFA troubleshooting guides. |
| **Enterprise Processes** | 2 | `doc-proc-001`, `doc-proc-002` | Post-mortem guidelines and hardware procurement refresh procedures. |
| **Cybersecurity Education** | 3 | `doc-sec-001` - `doc-sec-003` | Educational articles explaining prompt injection, Zero Trust, and credential security. Used to evaluate **false positive resilience**. |
| **Security Test Fixtures** | 5 | `doc-attack-001` - `doc-attack-005` | Labelled test fixtures with indirect injection in body, title, or metadata. Used to evaluate **indirect prompt injection detection**. |

---

## Limitations

1. **Synthetic Nature**: All prompts and corpus documents are synthetically curated. Real-world adversaries use diverse, multi-turn, and evolving adversarial strategies that synthetic benchmarks cannot fully reflect.
2. **Uncalibrated Model Scores**: Model outputs from Logistic Regression are heuristic decision scores, not true Bayesian probabilities.
3. **Balanced vs. Production Distribution**: Both benchmarks maintain a balanced 50/50 class distribution for symmetric evaluation. Production traffic will feature severe class imbalance ($>99:1$ benign).
4. **Untrusted Data Boundary**: Presence in `rag_corpus_v1.jsonl` grants zero trust or authorization. Test fixture tags and metadata are for testing categorization only and never exempt documents from security inspection.
