# ERM: how much of the cross-dataset gap do dataset artifacts explain?

Code, results and analysis for *"How Much of the Cross-Dataset Gap Do Dataset Artifacts Explain? A Controlled Study of Fake-News Detectors"*.

The study asks whether removing suspected dataset artifacts (source markers, document length, named entities) closes the gap between in-dataset and cross-dataset performance of fake-news detectors, and whether any improvement exceeds what deleting the same number of random words achieves. It audits three public English news corpora (ISOT, McIntire, WELFake), applies a factorial set of interventions with size-matched random-deletion controls, and evaluates TF-IDF + logistic regression, DistilBERT and RoBERTa-base.

**Scope.** Three English news corpora from 2016-2017, labels that may reflect publisher rather than veracity, and the WELFake residual is constructed (not independently collected). Seed intervals describe resampling variability on one fixed document universe, not generalization to other datasets. See the paper's Limitations section.

## Headline results

- TF-IDF: length matching plus entity masking (with source masking) removes about a quarter of the cross-dataset AUC gap (0.213 to 0.160); source masking does not reduce it. The effect depends strongly on the dataset pair.
- DistilBERT and RoBERTa: masking raises cross-dataset AUC, but deleting the same number of random words does about as well, so targeted removal does not outperform matched random deletion.
- Full numbers, intervals and per-pair tables: the paper, `results/final/`, and `results/README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `src/audit/` | dataset audit and analysis-set construction (`audit_01_05.py`, `build_analysis_set.py`, `drop_flagged.py`), adding a new corpus (`add_dataset.py`) |
| `src/experiments/` | `exp_tfidf.py` (TF-IDF arm and masking logic), `exp_distilbert.py` (original single-GPU runner), `exp_transformer.py` (multi-GPU, resumable runner for DistilBERT / RoBERTa / BERT / TF-IDF) |
| `src/analysis/` | `analyze.py`, `compare_models.py`, `compare_contamination.py`, `confirmatory.py`, `label_agreement.py` |
| `results/raw/` | raw per-cell results (one CSV per model and run); see `results/README.md` for the file-by-file manifest |
| `results/final/`, `results/sensitivity/`, `results/audit/` | derived tables and figures, duplicate-removal reruns, audit outputs |
| `configs/` | run configurations of the archived TF-IDF runs |
| `freeze/` | hashed copies of the inputs used for the paper and the regenerated tables |
| `tests/` | synthetic-data tests of the multi-GPU runner |
| `docs/` | `KAGGLE_2xT4.md` (running the transformer arms on two GPUs) and reports |
| `paper/` | manuscript |
| `archive/` | superseded material kept for provenance |

## Setup

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm        # entity masking uses spaCy en_core_web_sm
python -m pytest tests -q                        # about 30 s, no GPU or real data needed
```

`requirements-lock.txt` lists the exact versions reported in the paper. The runs behind the results used different environments (author's own system for the archived TF-IDF runs, Google Colab for the archived DistilBERT runs, Kaggle 2x T4 for the newer runs); see `results/README.md` and the `*_config.json` files.

Without the spaCy model, `exp_tfidf.py` silently falls back to a regex entity finder. Do not use that fallback for results you intend to compare with the paper; `exp_transformer.py prepare` refuses to run without the model unless you pass `--allow-regex-ner`.

## Data

The datasets are **not redistributed** here. Download them yourself and cite the original publications:

| Dataset | Source used in the paper | File |
|---|---|---|
| ISOT | Kaggle: `rahulogoel/isot-fake-news-dataset` | `Fake.csv`, `True.csv` |
| McIntire "Fake or Real News" | GitHub mirror `joolsa/fake_real_news_dataset` (6,335 articles) | `fake_or_real_news.csv` |
| WELFake | Kaggle: `studymart/welfake-dataset-for-fake-news` | `WELFake_Dataset.csv` |

The audit scripts read these four files from the current working directory (`load_all(base_dir=".")`); the loader also accepts the repository's `data/raw/{ISOT,McIntire,WELFake}/` layout. Build the analysis set once the audit has been run:

```bash
python src/audit/audit_01_05.py            # audit steps 01-05, writes reports
python src/audit/build_analysis_set.py     # writes analysis_set.csv (ISOT, McIntire, WELFake residual)
```

The frozen experiment input has 61,197 documents (ISOT 20,923 real / 17,107 fake, McIntire 2,964 / 3,041, WELFake residual 10,427 / 6,735). These are observed input counts, not construction targets. The historical primary audit dry run retained 17,161 WELFake rows (10,427 real / 6,734 fake). Its broader exact-match policy removed `WELFake-44862`, a fake-labelled lightly normalized body match to `ISOT-15508`; the builder's artifact-stripped hash is missing for both bodies and retains that WELFake row. The builder preserves the recorded experiment construction and refuses to overwrite an existing output. See [the current provenance report](docs/reports/dataset_provenance_20261112.md). The label mapping is 0 = real, 1 = fake; these remain dataset labels, not independent article-veracity assessments.

Audit an existing analysis input without changing it or training models (Python standard library only):

```bash
python src/audit/provenance_audit.py --analysis-csv data/processed/analysis_set.csv --output analysis_input_audit.zip
```

The ZIP contains only `audit.json`, `SUMMARY.md` and `metadata.json`: hashes, counts, missing/invalid-field and identifier checks, cluster summaries, and configuration hashes. It contains no articles. The command streams CSV records and hashes the file incrementally. It compares known freeze records and recognized analysis-hash fields in experiment configs; pass `--record path/to/validation.json` for additional records or `--expected-sha256 <recorded hash>`. An unavailable experiment record is not treated as a verified validation. Exit code 1 means a data check or recorded input/cluster comparison failed; a difference from the explicitly historical dry-run claim is reported separately. Existing ZIPs are never overwritten.

To reconstruct from accessible source CSVs, use a new, ignored output directory and compare it with the exact experiment input:

```bash
python src/audit/build_analysis_set.py data/provenance_reproduction_new/analysis_set.csv --base-dir data/raw
python src/audit/construction_evidence.py --analysis-csv data/processed/analysis_set.csv --source-dir data/raw --reproduced-csv data/provenance_reproduction_new/analysis_set.csv --output construction_evidence.zip
```

Reconstruction uses NumPy and pandas; inspecting the historical cleaned Parquet cache also requires pyarrow. The old audit configuration records NumPy 2.3.2 and pandas 2.3.3. Current source hashes cannot prove original source-snapshot identity because the historical audit did not record raw-source hashes. Byte identity, count agreement and reproduction are distinct checks. Neither counts nor hashes independently verify overlap or article veracity. No dataset upload or Git commit is needed.

## Reproducing the experiments

TF-IDF arm (CPU), conditions and seeds as in the paper:

```bash
python src/experiments/exp_tfidf.py spans                       # cache entity spans once
python src/experiments/exp_tfidf.py run --smoke                 # quick pipeline check
python src/experiments/exp_tfidf.py run --seeds 0 1 --tag a     # writes results_tfidf_a.csv; repeat for other seeds / tags
```

Transformer arms (GPU), resumable, one worker per visible GPU. On Kaggle with two T4s, see `docs/KAGGLE_2xT4.md`:

```bash
python src/experiments/exp_transformer.py run --model roberta-base --gpus 0 1 --seeds 0 1 2 3 4 5 6 7 \
    --conds C0 S L E SE LE SLE R_S R_E R_SE L_RE L_RSE \
    --analysis-csv data/processed/analysis_set.csv --spans-cache cache/spans_cache.pkl --out results_roberta.csv
```

`exp_transformer.py` reproduces the original runners exactly for the same seeds (checked bit-for-bit on DistilBERT, seed 0, `C0` and `L`; and by `tests/test_runner_parity.py` for the TF-IDF path). Each run writes a `*_config.json` with its arguments and library versions.

Duplicate-removal robustness check: `src/audit/drop_flagged.py` writes `flagged_doc_ids.txt`; rerun conditions with `--exclude flagged_doc_ids.txt` (see the script's docstring).

## Analysis

```bash
python src/analysis/analyze.py <result csvs> --expected-seeds 0 1 2 3 4 5 6 7 --output-dir results/final/tfidf
python src/analysis/compare_models.py --tfidf <tfidf csvs> --bert <distilbert csv>
python src/analysis/confirmatory.py --results <csvs> --out results/final/confirmatory/<name>
python src/analysis/label_agreement.py <annotation sheet> <label key>      # Cohen's kappa for a manual label check
python freeze_and_analyze.py --tfidf <csvs> --bert <csv>                   # hash inputs and regenerate all tables
```

The unit of inference is the random seed; the six ordered dataset pairs within a seed are not independent. Intervals are 95% t-intervals over seeds, reported with how many seeds agree in sign. `confirmatory.py` adds exact sign-flip p-values with Holm correction within a six-contrast family. That family was defined after the earlier exploratory results, so treat its adjusted p-values as exploratory.

## Adding a new corpus

`src/audit/add_dataset.py` applies the same cleaning and overlap audit to a new corpus, removes documents already present in the existing corpora, reports the overlap rate (a high rate means the corpus is not independent), checks the paper's size rule, and writes a blind sheet for manual label checking. Check the label direction on known content before using any new corpus.

## Citation and license

Citation: [add the paper reference and a `CITATION.cff` once the paper is accepted or on a preprint server].
License: [choose and add a `LICENSE` file before making the repository public; the datasets keep their own licenses].
