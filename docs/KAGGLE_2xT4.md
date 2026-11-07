# Running the experiments on Kaggle (2 x T4)

Settings -> Accelerator -> **GPU T4 x2**, Internet **on**. Upload `analysis_set.csv` (+ `spans_cache.pkl`) as a Kaggle Dataset,
or build them with `src/audit/build_analysis_set.py` / `src/audit/add_dataset.py`.

```python
# cell 1: setup
!git clone https://github.com/abiroh-p/ERM.git /kaggle/working/ERM
%cd /kaggle/working/ERM
!pip -q install transformers spacy && python -m spacy download en_core_web_sm
!nvidia-smi -L                       # must list two GPUs

# cell 2: NER spans for any new documents (CPU, once)
!python src/experiments/exp_transformer.py prepare \
    --analysis-csv /kaggle/input/erm-data/analysis_set_plus.csv --spans-cache /kaggle/working/spans_cache.pkl

# cell 3: one model across BOTH GPUs (one worker per GPU, jobs interleaved; resumable)
!python src/experiments/exp_transformer.py run --model roberta-base --gpus 0 1 --seeds 0 1 2 3 4 5 6 7 \
    --analysis-csv /kaggle/input/erm-data/analysis_set_plus.csv --spans-cache /kaggle/working/spans_cache.pkl \
    --out /kaggle/working/results_roberta.csv
```

* If the 12 h session ends, **rerun the same cell**: finished (seed, condition, train-dataset) cells are skipped.
  Save `/kaggle/working` as a version (or copy the CSV to a Dataset) before the session closes.
* Worker output is prefixed `[w0|gpu0]` / `[w1|gpu1]`. A failed worker does not stop the other; rerun to finish.
* Other arms: `--model distilbert-base-uncased`, `--model bert-base-uncased`, `--backend tfidf --workers 4`.
* Temporal test: `--split temporal --date-col date --datasets <dated corpora>`.
* Pair-level random control for S+L+E is the new `L_RSE` condition (and `L_RE` for L+E); both are in the default list.
* Statistics: `python src/analysis/confirmatory.py --results <csvs> --out results/final/confirmatory [--metric gap] [--loo]`.

## Fourth corpus: NELA-GT-2020 (outlet-level labels, 2020)
```
python src/audit/prepare_nela.py --db nela-gt-2020.db --labels labels.csv --out data/raw/NELA/nela2020_sample.csv --per-class 6000
python src/audit/add_dataset.py --input data/raw/NELA/nela2020_sample.csv --name NELA2020 --text-col content --title-col title \
    --label-col label --fake-values fake --date-col date --analysis-csv data/processed/analysis_set.csv \
    --out data/processed/analysis_set_plus.csv --sensitivity
```
Labels are outlet-level (reliable -> real, unreliable -> fake, mixed dropped) and the release replaces ~7% of tokens with `@`; state both in the paper.
