# ERM / TF-IDF Cross-Domain Masking Experiment
## Diagnostic Analysis Report

Date: 2026-10-02

## Scope and source of truth

The repository does not contain a file named `analysis.csv`. The row-level input used by the experiment is `analysis_set.csv`.

The executable definitions are in:

- `exp_tfidf.py`
- `analyze.py`
- `build_analysis_set.py`
- `results_tfidf_a_config.json`
- `results_tfidf_b_config.json`
- `results_tfidf_c_config.json`
- the corresponding `*_methods.json` files

The PDF in `ResearchPaper/paper1.pdf` appears to describe a different topic/temporal-split study and does not define the exact symbols used by this experiment. The code and result files were therefore treated as authoritative.

No files were modified during the diagnostic analysis.

## 1. Experiment structure

### Datasets

There are three datasets:

| Dataset | Documents in `analysis_set.csv` |
|---|---:|
| ISOT | 38,030 |
| McIntire | 6,005 |
| WELFake residual | 17,162 |

`build_analysis_set.py` expected 17,161 WELFake residual documents, so the actual output differs from the expected count by one document.

There are six ordered source-to-target pairs:

1. ISOT -> McIntire
2. ISOT -> WELFake residual
3. McIntire -> ISOT
4. McIntire -> WELFake residual
5. WELFake residual -> ISOT
6. WELFake residual -> McIntire

There are five seeds: `0, 1, 2, 3, 4`.

### Conditions

The main factorial contains eight conditions:

`C0, S, L, E, SL, SE, LE, SLE`

There are also three random masking controls:

`R_S, R_E, R_SE`

The exploratory temporal extension contains:

`T, SLET, R_T`

Each condition has 45 result rows:

- 30 cross-domain rows: 5 seeds x 6 ordered pairs
- 15 in-domain rows: 5 seeds x 3 datasets

The three result files contain:

| File | Seeds | Rows |
|---|---|---:|
| `results_tfidf_a.csv` | 0, 1 | 252 |
| `results_tfidf_b.csv` | 2, 3 | 252 |
| `results_tfidf_c.csv` | 4 | 126 |
| Total | 0-4 | 630 |

Each row represents one TF-IDF/logistic-regression evaluation for one seed, condition, source dataset, target dataset, and evaluation type.

### Condition definitions

- `C0`: unmodified text.
- `S`: source/publisher artifacts removed using regex patterns. These include URLs, datelines, agency markers, outlet names, social and boilerplate text, credits, email addresses, and similar artifacts.
- `L`: length-matched sampling, class-balanced within dataset-specific length bins. This is a sampling intervention, not token masking.
- `E`: named entities removed using spaCy `en_core_web_sm`.
- `SL`, `SE`, `LE`, `SLE`: combinations of the corresponding factors.
- `R_S`, `R_E`, `R_SE`: random word deletion controls. They remove the same estimated number of words as the corresponding real masking operation, but at random positions.
- `T`: temporal expressions removed, including weekdays, months, years, dates, and clock times.
- `SLET`: `SLE` plus temporal masking.
- `R_T`: random deletion control for temporal masking.

### Sampling sizes

The configured targets were:

- Training: 1,500 documents per class.
- In-domain test: 500 documents per class.
- Cross-domain test: 800 documents per class.
- Universe: up to 4,000 documents per class per dataset.

Length matching made the actual sizes slightly smaller:

- Training: 2,988-2,994 documents total.
- Cross-domain test: 1,588-1,590 documents total.
- In-domain test:
  - ISOT: 988-992 total.
  - McIntire: 802-836 total.
  - WELFake residual: 988-992 total.

The same effective sizes are used across conditions for a given seed/source/target role.

### Gap metrics

For every source dataset, seed, and condition:

```text
gap_f1  = in_domain_f1  - cross_domain_f1
gap_auc = in_domain_auc - cross_domain_auc
```

The gap reduction reported in `table_gap_reduction.csv` is:

```text
gap reduction = gap_C0 - gap_condition
```

Thus, a positive gap reduction means the gap became smaller.

Factorial effects use the eight cells `C0, S, L, E, SL, SE, LE, SLE`, coded with each factor as `-1` or `+1`. For an outcome `y` and coded term `z`, the effect is:

```text
factorial effect = 2 * mean(y * z)
```

## 2. Seed consistency

The following values are cross-domain AUC averaged over the six ordered pairs within each seed.

| Condition | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.7761 | 0.7769 | 0.7718 | 0.7719 | 0.7717 | 0.7737 | 0.0026 |
| S | 0.7598 | 0.7532 | 0.7670 | 0.7610 | 0.7528 | 0.7588 | 0.0059 |
| L | 0.8039 | 0.8116 | 0.8045 | 0.8034 | 0.8017 | 0.8050 | 0.0038 |
| E | 0.7860 | 0.7840 | 0.7897 | 0.7877 | 0.7891 | 0.7873 | 0.0023 |
| SL | 0.7971 | 0.8066 | 0.8035 | 0.8018 | 0.7941 | 0.8006 | 0.0050 |
| SE | 0.7692 | 0.7799 | 0.7713 | 0.7866 | 0.7786 | 0.7771 | 0.0070 |
| LE | 0.8197 | 0.8235 | 0.8233 | 0.8352 | 0.8259 | 0.8255 | 0.0058 |
| SLE | 0.8107 | 0.8138 | 0.8194 | 0.8179 | 0.8167 | 0.8157 | 0.0034 |

### Seed-level changes from C0

| Contrast | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| LE - C0 | +0.0436 | +0.0466 | +0.0514 | +0.0633 | +0.0542 |
| SLE - C0 | +0.0346 | +0.0369 | +0.0475 | +0.0460 | +0.0450 |
| L - C0 | +0.0278 | +0.0347 | +0.0327 | +0.0315 | +0.0300 |
| E - C0 | +0.0099 | +0.0071 | +0.0178 | +0.0158 | +0.0174 |

### Interpretation

Observed result: aggregate AUC effects are directionally consistent across all five seeds, and seed-to-seed variation is small relative to the aggregate changes.

Qualification: this does not mean every ordered pair improves. The pair-level decomposition is substantially less uniform.

## 3. Ordered-pair analysis

Mean cross-domain AUC over five seeds:

| Pair | C0 | L | E | LE | SLE | LE - C0 | SLE - C0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISOT -> McIntire | 0.6234 | 0.6765 | 0.6438 | 0.6976 | 0.6889 | +0.0742 | +0.0655 |
| ISOT -> WELFake residual | 0.6320 | 0.6637 | 0.6292 | 0.6721 | 0.6705 | +0.0402 | +0.0385 |
| McIntire -> ISOT | 0.7456 | 0.8039 | 0.7921 | 0.8806 | 0.8583 | +0.1350 | +0.1127 |
| McIntire -> WELFake residual | 0.9146 | 0.9002 | 0.9082 | 0.8962 | 0.8916 | -0.0183 | -0.0230 |
| WELFake residual -> ISOT | 0.8031 | 0.8804 | 0.8253 | 0.8994 | 0.8943 | +0.0963 | +0.0912 |
| WELFake residual -> McIntire | 0.9235 | 0.9055 | 0.9252 | 0.9071 | 0.8907 | -0.0164 | -0.0328 |

Largest improvements:

- McIntire -> ISOT: `LE - C0 = +0.1350`.
- WELFake residual -> ISOT: `LE - C0 = +0.0963`.
- ISOT -> McIntire: `LE - C0 = +0.0742`.

Decreases:

- WELFake residual -> McIntire: `LE - C0 = -0.0164`, `SLE - C0 = -0.0328`.
- McIntire -> WELFake residual: `LE - C0 = -0.0183`, `SLE - C0 = -0.0230`.

### Gap-AUC changes

These are condition gap minus C0 gap. Negative values indicate a smaller gap.

| Pair | L | E | LE | SLE |
|---|---:|---:|---:|---:|
| ISOT -> McIntire | -0.0534 | -0.0218 | -0.0762 | -0.0698 |
| ISOT -> WELFake residual | -0.0320 | +0.0014 | -0.0422 | -0.0428 |
| McIntire -> ISOT | -0.0613 | -0.0577 | -0.1400 | -0.1240 |
| McIntire -> WELFake residual | +0.0114 | -0.0049 | +0.0133 | +0.0117 |
| WELFake residual -> ISOT | -0.0768 | -0.0319 | -0.1051 | -0.1144 |
| WELFake residual -> McIntire | +0.0184 | -0.0115 | +0.0076 | +0.0095 |

## 4. Difficulty dependence

| Pair | C0 AUC | LE AUC | LE - C0 |
|---|---:|---:|---:|
| ISOT -> McIntire | 0.6234 | 0.6976 | +0.0742 |
| ISOT -> WELFake residual | 0.6320 | 0.6721 | +0.0402 |
| McIntire -> ISOT | 0.7456 | 0.8806 | +0.1350 |
| McIntire -> WELFake residual | 0.9146 | 0.8962 | -0.0183 |
| WELFake residual -> ISOT | 0.8031 | 0.8994 | +0.0963 |
| WELFake residual -> McIntire | 0.9235 | 0.9071 | -0.0164 |

Exploratory correlations across only six pairs:

- Pearson: `r = -0.56`, `p = 0.25`.
- Spearman: `rho = -0.49`, `p = 0.33`.

Interpretation: there is a suggestive tendency for lower-baseline transfers to improve more, but six pairs are insufficient for a reliable relationship. The apparent relationship is strongly affected by the two high-baseline directions that decline.

## 5. Random-masking controls

Differences are real masking minus random masking. The intervals use the same 5,000-resample bootstrap style as the existing analysis pipeline.

| Comparison | AUC difference | F1 difference | gap-AUC difference | gap-F1 difference |
|---|---:|---:|---:|---:|
| S - R_S | -0.0159 [-0.0217, -0.0105] | -0.0193 [-0.0276, -0.0115] | +0.0085 [+0.0022, +0.0147] | +0.0050 [-0.0032, +0.0137] |
| E - R_E | +0.0213 [+0.0117, +0.0321] | +0.0149 [+0.0082, +0.0221] | -0.0263 [-0.0369, -0.0157] | -0.0260 [-0.0339, -0.0183] |
| SE - R_SE | +0.0178 [+0.0083, +0.0278] | +0.0105 [+0.0030, +0.0184] | -0.0295 [-0.0393, -0.0207] | -0.0317 [-0.0387, -0.0244] |

Observed result:

- `S` performs worse than random deletion.
- `E` performs better than random deletion.
- `SE` performs better than random deletion.

Interpretation: named-entity masking has an effect beyond merely deleting words. This does not establish why; entity deletion may alter several correlated properties of the text.

## 6. Factorial effects

Effects are averaged across the 30 seed/pair units. Negative gap effects indicate gap reduction.

| Term | AUC | F1 | gap-AUC | gap-F1 |
|---|---:|---:|---:|---:|
| S | -0.0098 [-0.0128, -0.0067] | -0.0138 [-0.0204, -0.0076] | +0.0025 [-0.0008, +0.0060] | -0.0007 [-0.0063, +0.0051] |
| L | +0.0375 [+0.0229, +0.0518] | +0.0136 [+0.0032, +0.0238] | -0.0368 [-0.0508, -0.0219] | -0.0137 [-0.0240, -0.0034] |
| E | +0.0169 [+0.0082, +0.0262] | +0.0167 [+0.0105, +0.0232] | -0.0226 [-0.0321, -0.0141] | -0.0287 [-0.0357, -0.0219] |
| S:L | +0.0027 [+0.0002, +0.0054] | +0.0001 [-0.0024, +0.0024] | -0.0026 [-0.0052, +0.0000] | -0.0001 [-0.0030, +0.0030] |
| S:E | -0.0002 [-0.0025, +0.0022] | +0.0030 [-0.0003, +0.0068] | +0.0003 [-0.0020, +0.0027] | -0.0016 [-0.0056, +0.0019] |
| L:E | +0.0009 [-0.0014, +0.0032] | +0.0029 [+0.0004, +0.0056] | +0.0001 [-0.0020, +0.0023] | -0.0013 [-0.0045, +0.0018] |
| S:L:E | -0.0025 [-0.0060, +0.0008] | -0.0018 [-0.0045, +0.0007] | +0.0019 [-0.0014, +0.0053] | +0.0017 [-0.0005, +0.0039] |

Key interpretation:

- `L` is the strongest primary AUC factor.
- `E` is positive and smaller but consistent.
- `S` is negative for AUC and F1.
- `S:L` is very small. Its AUC interval barely excludes zero, while its F1 and gap effects include zero.
- Higher-order interactions do not provide strong evidence of robust interaction effects.

## 7. In-domain versus cross-domain

| Condition | In-domain F1 | Cross-domain F1 | Cross-domain AUC | gap-AUC |
|---|---:|---:|---:|---:|
| C0 | 0.9430 | 0.7290 | 0.7737 | 0.2124 |
| S | 0.9270 | 0.7103 | 0.7588 | 0.2191 |
| L | 0.9412 | 0.7378 | 0.8050 | 0.1801 |
| E | 0.9278 | 0.7379 | 0.7873 | 0.1914 |
| SL | 0.9254 | 0.7230 | 0.8006 | 0.1777 |
| SE | 0.9149 | 0.7289 | 0.7771 | 0.1949 |
| LE | 0.9295 | 0.7562 | 0.8255 | 0.1553 |
| SLE | 0.9164 | 0.7438 | 0.8157 | 0.1575 |

For `LE` relative to `C0`:

- In-domain F1 decreases from 0.9430 to 0.9295.
- Cross-domain F1 increases from 0.7290 to 0.7562.
- Cross-domain AUC increases from 0.7737 to 0.8255.

Therefore, the smaller `LE` gap is not caused solely by a large in-domain collapse. Nevertheless, gap reduction alone should not be interpreted as improved generalization; the cross-domain AUC is the primary evidence.

## 8. Temporal expression analysis

| Contrast | AUC change | F1 change | gap-AUC change | gap-F1 change |
|---|---:|---:|---:|---:|
| T - C0 | -0.0124 [-0.0183, -0.0067] | -0.0059 [-0.0109, -0.0004] | +0.0086 [+0.0032, +0.0141] | -0.0045 [-0.0111, +0.0014] |
| T - R_T | -0.0085 [-0.0155, -0.0014] | -0.0020 [-0.0073, +0.0033] | +0.0065 [approximately 0, +0.0133] | -0.0048 [-0.0109, +0.0011] |
| SLET - SLE | -0.0160 [-0.0216, -0.0104] | -0.0097 [-0.0146, -0.0047] | +0.0105 [+0.0048, +0.0161] | -0.0022 [-0.0086, +0.0039] |

Temporal masking does not provide evidence comparable to `L` or `E`. It slightly worsens cross-domain AUC relative to both `C0` and its random control. It should remain exploratory.

## 9. Data and methodology checks

### Passed checks

- All five seeds are present.
- All 14 conditions are present.
- Every condition has 45 rows.
- Every `(seed, condition, train, test)` key is unique.
- There are no duplicate result rows.
- All source-target combinations are present.
- All in-domain and cross-domain evaluations are present.
- The result files correctly partition seeds 0-4.
- All configurations use the same model and sampling settings.
- All method files report spaCy `en_core_web_sm`.
- There is no suspicious systematic equality between F1 and AUC or between AUC and average precision.

### Problems and risks

#### Missing `analysis.csv`

The named file is absent. The actual experiment input is `analysis_set.csv`.

#### One residual duplicate text

`analysis_set.csv` contains identical text in:

- `ISOT-15508`
- `WELFake-44862`

Both have the same label. The records are extremely short URL-only documents, so their practical impact may be small, but the final analysis set is not perfectly cross-dataset deduplicated.

#### ISOT/McIntire overlap was not removed

The audit reports exact and near-duplicate overlaps between ISOT and McIntire, including label conflicts. `build_analysis_set.py` trims WELFake against ISOT and McIntire, but it leaves ISOT/McIntire overlap intact.

The split is group-aware within each dataset, but not across datasets. Because cross-domain evaluation samples target documents from the entire target dataset, a source training document can have an exact or near-duplicate counterpart in the target dataset.

This makes cross-domain contamination plausible for ISOT <-> McIntire. It is a methodological risk, not proof that the results are invalid.

#### Bootstrap interpretation

`analyze.py` bootstraps 30 `(seed, pair)` units. It does not resample the five seeds as independent clusters. These intervals are useful descriptive paired-unit intervals, but they should not be presented as strong uncertainty estimates over independent replications.

#### Silent duplicate removal in analysis

`analyze.py` uses `drop_duplicates` on `(seed, condition, train, test)`. No duplicates occur in the current result files, but this could conceal a future data-generation problem rather than fail loudly.

#### Slightly unequal sample sizes

Length matching causes small differences in effective train and test sizes. The differences are explained by bin feasibility and are consistent across conditions for each role. They are not evidence of an aggregation error, but should be reported.

## 10. Scientific interpretation

### A. What the experiment clearly establishes

- Cross-domain performance varies substantially by transfer direction.
- `L` and `E` improve aggregate cross-domain AUC relative to `C0`.
- `S` reduces aggregate cross-domain AUC.
- `LE` has the highest aggregate main-condition AUC: 0.8255.
- Entity masking beats matched random deletion for AUC, F1, and both gap metrics.
- Aggregate `L`, `E`, and `LE` improvements are directionally consistent across all five seeds.
- Temporal masking does not improve cross-domain AUC in this experiment.

### B. What it suggests but does not establish

- Length distributions and named entities may carry transferable domain-specific artifacts.
- Masking benefits may be strongest on harder transfer directions.
- Effects are direction-specific rather than universal.
- Entity masking removes more useful cross-domain signal than random word deletion.

### C. Claims to avoid

Avoid claiming that:

- masking causes better generalization;
- source markers are universally harmful;
- `LE` improves every source-to-target transfer;
- gap reduction alone proves improved generalization;
- the effects generalize beyond these datasets, this TF-IDF model, and this sampling protocol;
- the confidence intervals establish strong five-replication statistical significance;
- the remaining ISOT/McIntire overlap is harmless;
- temporal information is generally uninformative.

### D. Most important numerical findings

1. Aggregate cross-domain AUC is 0.7737 for `C0` and 0.8255 for `LE`, a gain of approximately +0.0518.
2. `LE` improves McIntire -> ISOT by +0.1350 AUC but decreases McIntire -> WELFake residual by -0.0183.
3. `LE` improves WELFake residual -> ISOT by +0.0963 but decreases WELFake residual -> McIntire by -0.0164.
4. Entity masking versus random deletion gives +0.0213 AUC for `E - R_E`.
5. Temporal masking reduces AUC by -0.0124 versus `C0` and -0.0085 versus its random control.

## Recommended next step

### 1. Exact hypothesis/question

Does the `LE` cross-domain AUC improvement persist after all cross-dataset exact and near-duplicate documents are removed, especially for ISOT <-> McIntire?

### 2. Exact experiment or analysis

Construct a fully cross-dataset-overlap-controlled analysis set, then rerun the paired evaluation using the same TF-IDF/logistic-regression pipeline and sampling protocol.

### 3. Conditions

Use:

`C0, L, E, LE, R_E, R_SE`

Evaluate all six ordered transfer directions.

### 4. Seeds

Use seeds 0, 1, 2, 3, and 4.

### 5. Primary metric

Cross-domain AUC, reported separately by ordered pair. The overall average should remain secondary to the pair-level results.

### 6. Interpretation of possible outcomes

- `LE` remains positive across the same directions: stronger evidence that the result is not driven by duplicate-content contamination.
- Gains disappear mainly for ISOT <-> McIntire: the original result was likely overlap-sensitive.
- Gains remain only for transfers into ISOT: report the finding as direction-specific, not as universal masking-based generalization.
- `E - R_E` remains positive: evidence for a named-entity-specific effect beyond deletion quantity.

### 7. Approximate computational cost

The focused rerun uses six conditions instead of fourteen, roughly 43% of the current model-fitting workload, or about 90 source/evaluation combinations across five seeds. Building the cleaned analysis set requires repeating the repository overlap-processing step; the existing script estimates approximately 15-25 minutes for that preprocessing, excluding model fitting.
