# Celebrity external validation, version 1

Frozen before external scoring, 12 November 2026. Machine specification:
`configs/external_celebrity_v1.json`. This is a separate extension, never an
addition to the original six-transfer pooled analysis. No Celebrity scores were
examined to choose this protocol. Audit text inspection is permitted for data
integrity, interpretation and exclusion; it must not change model interventions.

## Question and scope

Train separately on ISOT, McIntire and the constructed WELFake residual and test
transfer to the supplied Celebrity dataset. The existing paired study has seeds
0–7, 13 TF-IDF conditions (312 fits) and eight transformer conditions (192 fits per
transformer). Its analysis input SHA-256 is
`29e96d7b7fe60a94a2652c704f2bd4be9b5bb7e45f540eabeb19320276788ab5`.
Celebrity's 250 articles per label fail the original 3,000-per-class training
threshold. Do not train in the reverse direction. Celebrity is a celebrity/gossip
domain target with author-supplied labels, not a temporal holdout or proof of
universal generalization. One text explicitly identifies a 1974 interview.

Primary outcomes: E minus R_E and SE minus R_SE on all eligible target articles,
fixed across seeds, sources and models. Principal metric: ROC-AUC. Preserve F1
(fake=1, threshold 0.5), accuracy and average precision. C0 is a reference.
The added SE/R_SE pair makes the combined comparison feasible without target
label-based sampling. Original eight-condition compatibility is C0, E, R_E, L,
LE (=L+E), L_RE (=L+R_E), SLE (=S+L+E), L_RSE (=L+R_SE).

L, LE, L_RE, SLE and L_RSE use source-side length matching plus the **same one
frozen target length-matched subset**. These are secondary diagnostics using
target labels. Never compare their scores to primary scores as if they shared a
test set. Secondary contrasts are LE−L_RE and SLE−L_RSE. No choice of thresholds,
masking expressions, NER labels or checkpoints may use target performance.

## Data audit and exclusions

Read only `fakeNewsDatasets/celebrityDataset/` and retain the supplied top-level
README/citation separately. UTF-8 bytes and full text are never rewritten. The
first nonblank physical line is the headline; body is the exact suffix following
that line. Feature input is the entire original decoded file, including original
line breaks. UTF-8 decode failures are recorded and quarantined. Labels come from
the fake/legit folders; numeric IDs define provisional story groups only. IDs,
paths, labels and group numbers never enter text features.

Audit raw hashes, lengths, missing fields, text flags and internal duplicates.
Unusually long means >3,000 whitespace words (a flag, not removal). Publisher,
fact-checking and boilerplate flags are inspection aids, not masking changes or
automatic evidence of false labels. Hashes of repeated body lines with ≥8
normalized words help identify repeated story/boilerplate candidates without
exporting text. Semantic pairing and every label are not manually revalidated.

Match against **every row** of the hash-verified frozen source CSV, not just
sampled training documents. The builder concatenated title, newline, body: split
at the first newline for field recovery and also compare the original combined
text. Record this limitation if source titles themselves contained newlines.
Use repository `norm_light`, `norm_strip` and hashed shingles for:

- Nonempty body and combined-text exact matches under both normalizations.
- Artifact-stripped 300-character prefix, only with at least 300 characters.
- Full and first-100-word views: 5-shingle Jaccard ≥0.8 (primary), 3-shingle
  Jaccard ≥0.5 (sensitivity), or containment ≥0.9 with ≥50 shingles in the smaller
  set. Empty normalized strings/shingle sets cannot match.

The extension enumerates all positive shingle intersections using inverted
postings; it retains the established scoring/thresholds while avoiding LSH's
probabilistic candidate recall and oversized-bucket chaining. Internal full-text
5-shingle results independently check the preliminary screen. All method/field
evidence rows retain both article identifiers, labels, scores and manual status.

Conservatively exclude all cross-corpus exact, prefix, primary and sensitivity
candidates, without presenting them as manually confirmed duplicates. Close
exclusions over provisional numeric groups **and** internal duplicate candidate
components. Quarantine invalid/short/missing-body text and cross-label exact or
primary duplicates with those components. Do not remove internal same-label
copies merely to improve scores; related units remain linked for uncertainty.
No source article is modified. Audit counts before/after and evidence are saved.
If source CSV is unavailable, emit internal audit with cross-corpus incomplete;
selection/evaluation are blocked until the exact original CSV is available.

## Reproducible selection and interventions

Source universe, cluster split, per-class effective sizes, length-bin quotas and
condition-independent selections reuse `exp_tfidf`, `exp_transformer` and
`paired_sampling` without modification. Freeze eight complete original source
manifests, including unused original test roles for parity checks. Source universe
draw uses seed 0 and at most 4,000 per dataset/class, source split fraction 0.2,
train target 1,500/class and ten length bins. Paired effective sample sizes may
be smaller; record actual sizes.

Primary target order is lexicographic stable article ID order after audit
eligibility. Secondary ten-bin edges are global target word-length deciles;
within each bin use the minimum class count scaled/floored by the existing
`effective_n` rule, then the existing `draw(..., length_matched=True)`. RNG seed is
the first eight hex digits of the SHA-256 JSON digest of
`["celebrity-external-v1", "target-secondary", 20261112]`.
Save IDs, labels, provisional groups, edges, quotas and selection hashes.
Neither selection varies with training seed or model.

Reuse original source regex expressions and spaCy `en_core_web_sm` entity labels;
do not extend source terms based on Celebrity. Retain the original source cache,
copy spans into a separate external cache, extract target spans with the same
model family and record actual spaCy/model metadata. Historical complete NER
environment is unknown: same model name does not prove identical model version.
The production runner refuses regex fallback. Per-document matched random
deletion reuses CRC32(`seed-doc_id-E/SE`) mod 2^31 streams and the paired helper's
exact whole-word count, including all-word deletion to empty text. Save target
whole-word deletion counts for each job; IDs and original text remain external.

## Models, resumption and compute

TF-IDF is the primary extension: existing lowercased word (1,2)-grams,
min_df=2, max_df=.95, sublinear TF, ≤200,000 features; logistic regression C=1,
liblinear, max_iter=2,000. Preserve existing fitting behavior. Transformers reuse
the existing fit function: DistilBERT base uncased or RoBERTa base, 256 tokens,
two epochs, batch 16, AdamW lr=3e−5/weight decay .01, warmup 6%, gradient clip 1,
CUDA mixed precision. Log the actual dependency versions and code hashes.
CUDA numerical bitwise determinism is not guaranteed by the inherited recipe.
The extension resets NumPy to the training seed before each fit, including
TF-IDF's implicit liblinear RNG, so job assignment/resumption does not change it.

No usable trained checkpoints were found in the repository; archived CSV metrics
are not models. Retrain original source fits. Do not launch the full GPU grid as
part of preparation. Full grid: 3 sources × 8 seeds × 10 conditions = **240 fits
per model**, 720 across all three (120 primary, 120 secondary per model). Using
only the compatibility subset is 192 fits/model, but omits the primary SE/R_SE
pair and is not the default frozen grid. Each completed fit evaluates one target
set. Maximum source training size is 3,000 articles. Upper bound per transformer
fit: 2×ceil(3000/16)=376 optimizer steps; full grid ≤90,240 steps/model. Actual
sizes and timings are written per job.

CPU audit/TF-IDF: recommend 16 GB RAM, one or more modern CPU cores; full audit
runtime and TF-IDF fit timings must be measured, not inferred from GPU archives.
Transformer extension: one 16 GB CUDA GPU per process; independent fits may be
assigned to two T4s as in the original runs. Transformers require model downloads
and compatible torch/CUDA; disk also holds source data/cache. No paid job is
authorized by these commands. Wall-clock estimates require a measured pilot.

`plan` freezes the full job grid and input/code/environment signature before any
fit. Job index follows seed, condition, source order in configuration. Rank r of
W workers owns indices `[r::W]`; assignments do not depend on which jobs finished.
Only one rank per GPU, configured using CUDA_VISIBLE_DEVICES. Completed jobs are
atomic JSON records, checksummed and validated on resume. Changed inputs/code/
environment require a new output directory. Plans/preparation are serial;
parallel workers only write their own jobs. Exclusive per-job locks prevent
duplicate workers; remove a stale lock only after confirming its worker stopped.
Completion requires every expected job and valid metric/selection signatures.

## Analysis and interpretation

Report all completed outcomes with completion status, including adverse or
inconclusive effects. Each source-to-Celebrity transfer is reported separately;
this runner makes no pooled estimate. If pooling is later added, preregister its
weighting (e.g., equal source weight) and keep it separate from the original six
transfers. Differences are intervention AUC minus control AUC within the same
seed, source, model, training IDs and target IDs. Report mean, SD, minimum,
maximum and number of paired seeds; incomplete contrasts are not imputed.

Seed variability reflects training/random-deletion variation conditional on
these fixed articles, not independent target samples. No article-level interval
is implemented in v1. A later paired cluster bootstrap must resample frozen
eligibility components (numeric groups plus duplicate links), reuse the same
resample for intervention and control predictions, and state that this estimates
article/story sampling variation rather than variation across independently
collected datasets. Numbered groups are provisional; residual related stories
may remain. Never add placeholder manuscript results. Any protocol amendment
after scoring needs a new version, reason and explicit exploratory status.

## Rights and deliverables

Original extension code follows the repository MIT software license. Celebrity
article text and the authors' supplied README retain their own terms. The ZIP's
README supplies a citation to Pérez-Rosas, Kleinberg, Lefevre and Mihalcea,
*Automatic Detection of Fake News*, COLING 2018; preserve its bytes, including
the supplied encoding defects. No new data redistribution license is asserted.
The runnable package includes code, configuration, protocol, notices, audit
identifiers and evidence, never articles, original data ZIP, source CSV, model
weights or span caches. The user supplies those local inputs.
