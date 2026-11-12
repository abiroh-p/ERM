# Results index

## Current document-paired validation

`paired/` is the primary result set used by the current manuscript.

| File/directory | Contents |
|---|---|
| `paired/accounts/ERM_Paired_Account_{A,B,C}.zip` | Unchanged source account outputs: A seeds 0–2, B 3–5, C 6–7 |
| `paired/provenance/` | Extracted configuration/protocol JSON for input auditing; article selections remain in the original account ZIPs |
| `paired/merged/results_tfidf.csv` | 936 evaluation rows; 312 fits, 13 conditions, eight seeds |
| `paired/merged/results_distilbert.csv` | 576 evaluation rows; 192 fits, eight conditions, eight seeds |
| `paired/merged/results_roberta.csv` | 576 evaluation rows; 192 fits, eight conditions, eight seeds |
| `paired/statistics/` | Pooled, per-pair, per-seed and model-difference estimates; raw gap reductions; historical comparison |
| `paired/figures/` | Paired scientific figure and corrected overlap schematic |
| `paired/integrity_report.json` | Source hashes, run grid, manifests and scope of checks |

Each fit yields one in-domain and two cross-domain evaluations. The experimental repeat is the seed; six ordered pairs are averaged within seed. Intervals are descriptive t-intervals over eight seed-level means. The six-contrast Holm family is retained; omitted transformer contrasts receive p=1. Pair-level results are exploratory.

Rebuild with `python scripts/rebuild_paired_results.py`, followed by `python scripts/generate_paired_tables.py`.

## Historical results

`raw/` retains original condition-specific samples, `sensitivity/` the archived duplicate-removal reruns, and `validation/` the historical seed-0 parity check and a RoBERTa smoke run. They must not be merged into the document-paired grid. Original TF-IDF configurations remain under `configs/tfidf/`.

Older derived folders formerly named `final/` have moved to `archive/legacy_derived/results/final/`. Exact duplicates already present under `freeze/outputs/` were removed and mapped in `docs/cleanup_manifest.json`. The redundant DistilBERT five-seed CSV was removed only after its complete row content matched the corresponding slice of the eight-seed file.

Frozen snapshots remain unchanged. Their input manifests may name analysis data excluded from Git; they are historical records, not proof that every named large input is distributed here. The original source paths and environments are described in the preserved metadata and the current environment document.
