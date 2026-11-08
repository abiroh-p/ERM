# Results manifest

Every file under `results/raw/` holds metrics only (no article text). One row = one (seed, condition, training dataset, test dataset) cell. A complete cell grid is 3 training datasets x 3 test datasets = 9 rows per (seed, condition).

Datasets: `ISOT`, `McIntire`, `WELFake_residual` (the WELFake residual after removing documents that overlap ISOT or McIntire; see the paper, Section VI-A).

Conditions: `C0` baseline; `S` source-marker masking; `L` length matching; `E` entity masking; two- and three-letter codes combine them (`SE`, `LE`, `SLE`, `SL`); `R_x` random deletion of as many words as `x` removes; `L_RE`, `L_RSE` length matching plus random deletion matched to `E` / `S+E` (the matched controls for `L+E` and `S+L+E`); `T` temporal expressions and `S_W` / `S_B` (wire/outlet vs web-boilerplate part of `S`) are exploratory.

## Raw results

| File | Model | Seeds | Conditions | Rows | Produced by | Environment |
|---|---|---|---|---|---|---|
| `raw/tfidf/results_tfidf_a.csv` | TF-IDF + LR | 0-1 | 14 (primary + `T`, `SLET`, `R_T`) | 252 | `src/experiments/exp_tfidf.py` | author's own system |
| `raw/tfidf/results_tfidf_b.csv` | TF-IDF + LR | 2-3 | same 14 | 252 | same | author's own system |
| `raw/tfidf/results_tfidf_c.csv` | TF-IDF + LR | 4 | same 14 | 126 | same | author's own system |
| `raw/tfidf/results_tfidf_e.csv` | TF-IDF + LR | 5-7 | 18 (adds `S_W`/`S_B`, `R_W`/`R_B`) | 486 | same | author's own system |
| `raw/tfidf/results_tfidf_d1.csv`, `_d2.csv`, `_d3.csv` | TF-IDF + LR | 0-1, 2-3, 4 | `S_W`, `S_B`, `R_W`, `R_B` | 72, 72, 36 | same | author's own system |
| `raw/tfidf/results_tfidf_newconds.csv` | TF-IDF + LR | 0-7 | `L_RE`, `L_RSE` | 144 | `src/experiments/exp_transformer.py --backend tfidf` | Kaggle (see config) |
| `raw/distilbert/results_distilbert.csv` | DistilBERT | 0-7 | 11 (`C0 S L E SE SL LE SLE R_S R_E R_SE`) | 792 | `src/experiments/exp_distilbert.py` | Google Colab T4 (versions not recorded; see below) |
| `raw/distilbert/results_distilbert_5seeds.csv` | DistilBERT | 0-4 | same 11 | 495 | same | same; a subset of the 8-seed file |
| `raw/distilbert/results_distilbert_newconds.csv` | DistilBERT | 0-7 | `L_RE`, `L_RSE` | 144 | `exp_transformer.py --model distilbert-base-uncased --gpus 0 1` | Kaggle 2x T4 |
| `raw/roberta/results_roberta.csv` | RoBERTa-base | 0-7 | 12 (`C0 S L E SE LE SLE R_S R_E R_SE L_RE L_RSE`) | 864 | `exp_transformer.py --model roberta-base --gpus 0 1` | Kaggle 2x T4 |

The TF-IDF 8-seed analysis uses files `a`, `b`, `c`, `e` (seeds 0-7, no overlap). Files `d1`-`d3` and `e` carry the exploratory `S_W`/`S_B` split.

Each file produced by `exp_transformer.py` has a `*_config.json` next to it with the exact arguments and library versions.

## Environments

- New runs (`*_newconds`, `results_roberta`, `validation/`): Kaggle, 2x NVIDIA T4, Python 3.12.13, torch 2.10.0+cu128, transformers 5.0.0 (from the config files).
- Archived DistilBERT runs: most probably Google Colab (T4). The original runner did not record library versions, so the versions are not known for certain. A seed-0 re-run of conditions `C0` and `L` on Kaggle reproduced the archived values exactly (see `validation/`). Report this as a reproduction under different library versions only if the Colab versions are confirmed to differ.
- Archived TF-IDF runs: the author's own system (CPU). The config files do not record library versions; the software list in the paper's Reproducibility section (NumPy 2.3.2, pandas 2.3.3, scikit-learn 1.8.0, SciPy 1.16.3, spaCy 3.8.16 with en_core_web_sm 3.8.0) presumably describes this system and should be confirmed.

## Validation

`validation/parity.csv` (+ `parity_config.json`): 6 fine-tunes (seed 0, `C0` and `L`, 3 training datasets) with the multi-GPU runner. All 18 cells match `raw/distilbert/results_distilbert.csv` exactly (AUC, F1, accuracy and AP, max absolute difference 0.0), with identical sample sizes.
`validation/results_roberta_test.csv`: one-job smoke test of the RoBERTa path (not used in any analysis).
`tests/test_runner_parity.py` checks on synthetic data that the sharded runner reproduces `exp_tfidf.py` exactly and resumes correctly.

## Derived statistics

`final/confirmatory/confirmatory_<model>_auc.csv`: seed-level paired contrasts from `src/analysis/confirmatory.py` (mean, 95% t-interval over seeds, seeds agreeing in sign, exact sign-flip p, Holm p within the six primary contrasts). Regenerate:

```bash
python src/analysis/confirmatory.py --out results/final/confirmatory/confirmatory_tfidf --results \
  results/raw/tfidf/results_tfidf_a.csv results/raw/tfidf/results_tfidf_b.csv results/raw/tfidf/results_tfidf_c.csv \
  results/raw/tfidf/results_tfidf_e.csv results/raw/tfidf/results_tfidf_newconds.csv
python src/analysis/confirmatory.py --out results/final/confirmatory/confirmatory_distilbert --results \
  results/raw/distilbert/results_distilbert.csv results/raw/distilbert/results_distilbert_newconds.csv
python src/analysis/confirmatory.py --out results/final/confirmatory/confirmatory_roberta --results results/raw/roberta/results_roberta.csv
```

Note: `confirmatory.py` needs `model` and `kind` columns, which all files listed above contain. The seed intervals describe resampling variability on one fixed document universe, not generalization to other datasets. The "primary" contrasts were defined after the earlier exploratory results, so Holm-adjusted p-values here are not confirmatory.

## Known gaps

- Configs for the archived TF-IDF runs (`results_tfidf_*_config.json`, `*_methods.json`) are in `configs/tfidf/`, not next to the CSVs; configs of runs made with `exp_transformer.py` are written next to their CSV.
- `results/final/` holds several overlapping legacy folders (`tfidf`, `tfidf_8seed_final`, `confirmatory_results_tfIdf`, `distilbert`, `distilbert_8seed`); these predate this manifest and are to be consolidated.
- No fourth, independent dataset has been run yet.
