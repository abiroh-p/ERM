# ERM: dataset artifacts and the cross-dataset gap

Code, audited results and manuscript for **How Much of the Cross-Dataset Gap Do Dataset Artifacts Explain? A Controlled Study of Fake-News Detectors**. The manuscript is being prepared for Applied Intelligence; no acceptance or publication is claimed.

## Current findings

The main findings use document-paired validation: **eight seeds (0–7), 312 TF-IDF fits, 192 DistilBERT fits and 192 RoBERTa fits**, yielding 2,088 evaluation rows. Masking conditions and random-deletion controls select identical articles within each length-matching setting and share selections across models. Length matching versus baseline intentionally changes article selection.

| Finding | Paired validation |
|---|---|
| TF-IDF S+L+E raw AUC gap reduction | **26.2%** (95% seed interval 24.5–27.8%) |
| TF-IDF entity masking minus random deletion | **+0.021 AUC**; advantage also holds after length matching |
| DistilBERT entity masking minus random deletion | **−0.028 AUC**, favoring random deletion |
| Both transformers: S+L+E minus length-matched random deletion | No established positive pooled advantage; neither result establishes equivalence |
| RoBERTa S+L+E raw gap reduction | **50.8%**; a raw reduction does not isolate a cue-specific benefit |

These are descriptive seed-level results on three fixed English news corpora, not generalization to independently collected future corpora or exclusive causal attribution. The transformer validation covers eight conditions; TF-IDF covers all thirteen. The archived RoBERTa R_SE-versus-C0 gain of +0.102 AUC was not directly retested in the focused run.

## Start here

- Current manuscript: [`paper/manuscript/ERM_Manuscript.pdf`](paper/manuscript/ERM_Manuscript.pdf).
- Main result files, statistics and saved run provenance: [`results/paired/`](results/paired/).
- Dataset audit and the resolved one-row construction difference: [`docs/reports/dataset_provenance_20261112.md`](docs/reports/dataset_provenance_20261112.md).
- Reference audit: [`docs/references/REFERENCE_AUDIT.md`](docs/references/REFERENCE_AUDIT.md).
- Cleanup changes: [`docs/CLEANUP_REPORT.md`](docs/CLEANUP_REPORT.md).
- Remaining release decisions: [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

## Reproduce statistics without a GPU or article text

```bash
python -m pip install -r requirements-analysis.txt
python scripts/rebuild_paired_results.py
python scripts/generate_paired_tables.py
python scripts/generate_overlap_figure.py
python -m pytest tests -q
```

The rebuild starts from the unchanged three account ZIPs, checks complete result grids and saved article pairing, verifies the original analysis-input hash and exact experiment-code hashes, and regenerates merged CSVs, contrasts and figures. The saved records are independently checkable; this command does not rerun GPU training, raw-text overlap matching or NER extraction.

## Build the manuscript

```bash
python scripts/build_manuscript.py
```

A complete TeX Live installation with `pdflatex` and `bibtex` is required. The script stages a flat source directory at `paper/submission/`, compiles it, and updates the manuscript PDF. Publisher class and bibliography-style files are included under `paper/sn/`. The editable sections and generated tables are the source of truth; `paper/submission/` is a generated, ignored build directory.

## Data and construction provenance

Celebrity is prepared as a separate held-out external-validation extension;
see [`results/external_celebrity/README.md`](results/external_celebrity/README.md)
for its audit, frozen protocol, preparation and execution commands. No external
scores are included in the original three-corpus findings above.

Article text and the large analysis CSV/NER cache are not redistributed here. Obtain ISOT (`Fake.csv`, `True.csv`), the 6,335-row McIntire mirror (`fake_or_real_news.csv`) and WELFake (`WELFake_Dataset.csv`) from the sources cited in the manuscript. Put them in flat or corpus-specific folders under `data/raw/`.

| Corpus | Real (0) | Fake (1) | Total |
|---|---:|---:|---:|
| ISOT | 20,923 | 17,107 | 38,030 |
| McIntire | 2,964 | 3,041 | 6,005 |
| WELFake residual | 10,427 | 6,735 | 17,162 |
| Total | 34,314 | 26,883 | 61,197 |

The frozen analysis CSV SHA-256 is `29e96d7b7fe60a94a2652c704f2bd4be9b5bb7e45f540eabeb19320276788ab5`. The residual is **overlap-trimmed**, not guaranteed free of overlap. The broader historical audit retained 17,161 rows, while the builder retained one additional fake-labelled, stripped-empty exact match. Both documents in that retained pair are absent from all saved paired-validation selections. The author-run reconstruction reports byte-identical output; original raw-source hashes were not recorded, so historical raw-source snapshot identity remains unproved.

```bash
python src/audit/build_analysis_set.py data/processed/analysis_set.csv --base-dir data/raw
python src/audit/provenance_audit.py --analysis-csv data/processed/analysis_set.csv --output analysis_input_audit.zip
```

Construction refuses to overwrite an existing output. The read-only audit emits a small text-free report ZIP. An expected mismatch against the broader historical dry-run count is explained in the current provenance report.

## Repeat training

Use `requirements.txt` for the broader training dependencies and install spaCy's `en_core_web_sm` model. See [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) for observed versions and incomplete historical records; there is no universal verified lockfile.

```bash
python scripts/run_paired_validation.py --stage prepare --analysis-csv data/processed/analysis_set.csv --spans-cache cache/spans_cache.pkl --output-dir runs/paired
python scripts/run_paired_validation.py --stage all --analysis-csv data/processed/analysis_set.csv --spans-cache cache/spans_cache.pkl --output-dir runs/paired --design focused
```

Each transformer stage distributes independent fits over two GPUs. Keep new run outputs separate from the released results. To reproduce the three-account split, use seeds 0–2, 3–5 and 6–7 with `src/experiments/exp_transformer.py run --sampling paired`; the exact original commands and versions are recorded under `results/paired/provenance/`. Do not mix `legacy` and `paired` cells.

## Layout and historical evidence

| Path | Role |
|---|---|
| `src/audit/`, `src/experiments/`, `src/analysis/` | Maintained code, including paired sampling and provenance checks |
| `scripts/` | Rebuild, manuscript staging and run entry points |
| `results/paired/` | Current document-paired evidence and derived results |
| `results/raw/`, `results/sensitivity/`, `results/validation/` | Historical condition-specific experiments and checks; not the main paired results |
| `results/audit/primary/`, `results/audit/smoke/` | Historical audit tables; smoke counts are not publication estimates |
| `freeze/`, `archive/freeze_validation_intermediate/` | Preserved historical frozen snapshots; some manifests name large data intentionally not redistributed |
| `archive/legacy_derived/` | Older derived exports retained for provenance, not current headline results |
| `configs/`, `docs/`, `tests/`, `paper/` | Original run configurations, evidence, checks and manuscript |

## Citation and licensing

`CITATION.cff` identifies this research repository. The manuscript remains unpublished; update citation metadata when a public paper identifier exists. Original ERM code and its associated technical documentation are licensed under the [MIT License](LICENSE). The manuscript, research/result archives, publisher template files and other third-party materials are outside this code-license grant; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Source datasets retain their own terms.
