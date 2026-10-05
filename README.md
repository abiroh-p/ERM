# ERM: Fake-news attribution and transfer study

This repository contains the reorganized, publication-ready analysis pipeline for the cross-dataset misinformation study. The project preserves the original methodology, sampling logic, random seeds, and results while normalizing file layout and path resolution for reproducible execution from a clean repo structure.

## Repository layout

- `src/experiments/` — experiment runners and masking logic
  - `src/experiments/exp_tfidf.py` — primary TF-IDF + logistic regression factorial experiment
- `src/analysis/` — publication analysis and output generation
  - `src/analysis/analyze.py` — validates result CSVs and writes paper tables/figure outputs
- `src/audit/` — audit, filtering, and analysis-set construction scripts
- `data/raw/` — raw source corpora and upstream datasets
- `data/processed/` — cleaned analysis set and derived processed files
- `data/audit/` — flagged or excluded document IDs and audit artifacts
- `results/raw/` — raw experimental result CSVs
- `results/sensitivity/` — focused smoke and sensitivity runs
- `results/final/` — primary analysis tables and publication outputs, grouped
  under `tfidf/`, `distilbert/`, `model_comparison/`, and `figures/`
- `freeze/` — canonical frozen inputs, manifest, and regenerated analysis outputs
- `configs/` — configuration files
- `docs/` — reports and documentation
- `paper/` — manuscript and supporting paper assets
- `archive/` — historical and superseded materials kept for provenance

## Quick start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Run the canonical smoke test:

   ```bash
   python src/experiments/exp_tfidf.py run --smoke
   ```

3. Run the publication analysis on the smoke result files:

   ```bash
   python src/analysis/analyze.py results/sensitivity/tfidf/smoke/results_tfidf_smoke.csv --expected-seeds 0 1 --output-dir results/final/tfidf
   ```

4. Inspect the generated outputs in `results/final/tfidf/`.

For the publication freeze, run `freeze_and_analyze.py` with the existing
result CSVs. It copies and hashes the inputs before regenerating outputs under
`freeze/`; it does not retrain either model.

## Notes

- The project resolves key files relative to the repository root so it remains operational after the repo restructuring.
- `exp_tfidf.py` relies on the `en_core_web_sm` spaCy model for named-entity masking when available; otherwise it falls back to a regex-based entity finder.
- Historical versions and superseded outputs remain archived under `archive/` for reproducibility and provenance.
