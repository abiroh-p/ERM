# Current dataset provenance audit — 12 November 2026

This report supplements historical evidence. It does not replace either freeze
manifest, the primary audit tables, the historical diagnostic report, or completed
experiment/validation results. Article text is excluded from the report bundles.

## Verified input and counts

`data/processed/analysis_set.csv` has **61,197 rows**, 213,921,911 bytes, and SHA-256:

```text
29e96d7b7fe60a94a2652c704f2bd4be9b5bb7e45f540eabeb19320276788ab5
```

This hash and row count agree with `freeze/MANIFEST.json` and
`archive/freeze_validation_intermediate/MANIFEST.json`. The actual file at
`freeze/inputs/extra/analysis_set.csv` independently hashes to the same value.

| Corpus | Real (`is_fake=0`) | Fake (`is_fake=1`) | Total |
|---|---:|---:|---:|
| ISOT | 20,923 | 17,107 | 38,030 |
| McIntire | 2,964 | 3,041 | 6,005 |
| WELFake residual | 10,427 | 6,735 | 17,162 |
| Total | 34,314 | 26,883 | 61,197 |

There are no missing required values, invalid labels, duplicate identifiers,
identifier-label conflicts, malformed CSV rows, or discrepancies between recorded
`n_words` and final whitespace word counts. The audit streams quoted multiline CSV
records; physical line counts are not document counts.

| Corpus | Unique clusters | Clusters of size ≥2 | Documents in these clusters | Largest | Clusters with both labels |
|---|---:|---:|---:|---:|---:|
| ISOT | 37,722 | 245 | 553 | 5 | 6 |
| McIntire | 5,860 | 117 | 262 | 6 | 1 |
| WELFake residual | 16,912 | 219 | 469 | 4 | 0 |

ISOT/McIntire cluster summaries agree with the historical within-dataset report.
That report describes **full cleaned WELFake**, so its cluster totals cannot be
directly compared with the residual. Recorded clusters are inspected, not inferred
again by the lightweight audit.

## Demonstrated cause of 17,161 versus 17,162

Both counts describe real, different construction policies:

- `results/audit/primary/welfake_dry_run.csv` records 61,734 cleaned WELFake rows,
  44,573 union removals and **17,161 retained**. `triple_a_primary.csv` gives
  **10,427 real / 6,734 fake**.
- `build_analysis_set.py` historically used 17,161 as a dry-run reference and
  warned only for differences greater than 150. It never asserted equality or
  selected a target number of rows. The frozen experiment input retains **17,162**.
- Reapplying the historical primary flags to their recorded identifiers yields
  exactly the actual residual identifier set **minus `WELFake-44862`**. There are
  no historical-only identifiers. The extra row has `is_fake=1`.
- The historical flags mark that row with `exact_h_text_light` and
  `exact_h_both_light` against ISOT. The cleaned cache gives it the same lightly
  normalized body hash as `ISOT-15508`:
  `a885235c62d7b3007c6939e434a46b7e`. Both labels are fake.
- Both bodies become empty after artifact stripping, so stripped-body and
  stripped-title/body hashes and 300-character prefixes are missing. The recorded
  builder checks only the stripped-body exact hash plus prefix and primary near
  matching, whereas the dry run also checks light and title/body exact hashes.
  Empty shingles have Jaccard/containment zero and do not qualify as near matches.

This explains the entire one-row identifier difference. It is not a stale
assertion, header-count error, missing analysis label, or label-conflicting exact
duplicate group. The two URL-only rows pass the original short-document filter:
their lightly normalized bodies have 24 words, while the final title-plus-body
whitespace count is 2. That difference follows from the two different counting
rules; the final `n_words` column itself is correct.

The builder's original claim to trim “exactly as in the audit dry run” was false
and is corrected. The recorded construction is preserved; the historical number
is explicitly identified as a reference. Changing an overlap rule would create a
different scientific input and is not silently applied here.

Both `ISOT-15508` and `WELFake-44862` are already present in the historical
`data/audit/flagged_doc_ids.txt` and frozen exclusion list. This observation does
not establish the impact of this row on any fit or metric.

The historical `deviation_note.txt` reverses the WELFake real/fake wording: its
primary numbers should be read as **10,427 real / 6,734 fake**, and its sensitivity
numbers as **10,417 real / 6,573 fake**, consistent with the preserved CSV tables.
The note is retained unchanged as historical evidence; this is its current
correction. No manuscript count is changed on that note's authority.

## Source reconstruction

Original-named CSVs are available under `data/raw/`. A separate CPU reconstruction
completed into `data/provenance_reproduction_20261112/analysis_set.csv`, using
Python 3.12.4, NumPy 2.3.2 and pandas 2.3.3, matching the historical audit
environment. The input and frozen files were never overwritten.

**The reconstruction is byte-identical to the experiment input**, with SHA-256
`29e96d7b7fe60a94a2652c704f2bd4be9b5bb7e45f540eabeb19320276788ab5` and
213,921,911 bytes. Comparison found no missing or additional identifiers and zero
differences in `dataset`, `doc_id`, `text`, `is_fake`, `cluster_id` or `n_words`.
All per-corpus/per-label counts match. The fresh construction flagged 44,074
WELFake documents by exact/prefix rules and 44,564 by primary near matching;
their union removed 44,572 and retained 17,162. These are overlapping flag sets,
not additive removal counts.

The run used the original exact/near predicates with the new source-path and
no-overwrite support. A later refactor names the unchanged exact predicate in
`exact_prefix_flags`; its stripped-empty behavior is regression-tested. Code
hashes in the bundle identify the current on-disk source, not an independently
archived snapshot of the running Python process.

The recorded configuration is seed 13; 128 permutations; 32 bands × 4 rows;
primary 5-shingles/Jaccard 0.8; sensitivity 3-shingles/Jaccard 0.5; containment
0.9 with at least 50 shingles; full and first-100-word views; 300-character
prefix; minimum 20 lightly normalized body words; maximum LSH bucket 200.
Oversized candidate buckets are chained rather than exhaustively paired, so
pairwise audit searches and the builder's pooled search need not be identical
in general. This is a possible additional policy sensitivity, not the demonstrated
cause of this one-row difference.

Original raw-source hashes were not recorded. Current source hashes are included
in the evidence bundle; they cannot by themselves prove the sources are the
original snapshots. Label mapping and deduplication code are shared by the audit
and builder; no replacement data were downloaded.

The current raw CSVs parse to 23,481 ISOT fake rows, 21,417 ISOT real rows,
6,335 McIntire rows and 72,134 WELFake rows, matching the historical raw totals.
Every configuration key recorded by the primary audit matches the current
construction settings. These agreements do not replace raw-source byte hashes.

## Validation evidence and scope

The supplied task context reports completed document-paired validation over seeds
0–7: **312 TF-IDF fits, 192 DistilBERT fits and 192 RoBERTa fits**, with an analysis
hash matching the original freeze. These records and all experiment metrics are
preserved. This checkout's `results/validation/` contains the older seed-0 parity
configuration and a RoBERTa test record, not a complete record of those fit totals;
the totals are therefore attributed to the supplied validation context, not newly
certified by this audit. Additional validation JSON can be supplied using
`--record`, or its recorded hash using `--expected-sha256`.

Three separate conclusions must be maintained:

1. **Byte identity:** SHA-256 agreement identifies the exact recorded input bytes.
2. **Count consistency:** parsing those bytes verifies document and label totals.
3. **Construction reproduction:** original-source cleaning and matching must be
   rerun and compared separately. A hash or count alone does not reproduce it.

None independently validates article veracity or establishes exhaustive absence
of cross-corpus overlap. No GPU training, experiment reruns, pushes, releases,
data deletions, or manuscript submission are performed.

## Exact commands

From `D:\ERM`, using the existing local file (standard library only):

```powershell
python src/audit/provenance_audit.py --analysis-csv data/processed/analysis_set.csv --output analysis_input_audit.zip
```

Kaggle cell, with this checkout at `/kaggle/working/ERM` and the existing input
path recorded in `results/validation/parity_config.json`:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "/kaggle/working/ERM/src/audit/provenance_audit.py",
    "--analysis-csv", "/kaggle/input/datasets/abi8389/erm-data/analysis_set.csv",
    "--output", "/kaggle/working/analysis_input_audit.zip",
], check=False)
```

Return `analysis_input_audit.zip` to ChatGPT. It contains only summary reports,
hashes and configuration metadata. Exit code 1 still produces a ZIP when a data
or hash comparison fails. Use a new output filename if a ZIP already exists.

To independently repeat construction into a new local directory:

```powershell
python src/audit/build_analysis_set.py data/provenance_reproduction_new/analysis_set.csv --base-dir data/raw
python src/audit/construction_evidence.py --analysis-csv data/processed/analysis_set.csv --source-dir data/raw --reproduced-csv data/provenance_reproduction_new/analysis_set.csv --output construction_evidence.zip
```

## Manuscript corrections

- Methods: describe the actual stripped-body hash rule and distinguish the
  broader historical dry run; disclose the one retained exact match.
- Methods: state that the short-document filter uses lightly normalized body
  words before title concatenation, and conflicting exact-label groups are
  removed entirely. Primary clustering also includes qualifying containment.
- Figure 1 caption and limitations: avoid suggesting complete removal of all
  exact overlap; record the retained stripped-empty pair and existing exclusions.
- **No changes** to 17,162, 10,427 real, 6,735 fake, 61,197 total, 61,734 cleaned
  WELFake, or 44,572 builder removals are supported or needed: they agree with
  the actual frozen experiment input.

These corrections are implemented in the LaTeX sources. The existing manuscript
PDF is preserved; no LaTeX compiler is installed in this environment.

## Checks

The existing synthetic TF-IDF runner parity/resume test passed, without GPU
training. All 21 focused provenance/construction tests passed. These cover invalid
labels, duplicate/conflicting identifiers, hash/count mismatches, quoted multiline
CSV, missing fields, cluster mismatch reporting, text-free ZIP contents, overwrite
protection, flat/nested source layouts, and the stripped-empty historical-policy
regression. `git diff --check` passed.
