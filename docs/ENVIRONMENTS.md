# Environment evidence

There is no single verified software environment for all experiments.

| Work | Evidence | Recorded environment |
|---|---|---|
| Current paired validation, all three accounts/models | `results/paired/provenance/` | Python 3.13.16; NumPy 2.1.3; pandas 2.3.3; scikit-learn 1.6.1; SciPy 1.16.3; torch 2.11.0+cu130; transformers 5.20.2; two T4 GPUs per transformer session |
| Original TF-IDF runs | Old configuration files and author manuscript statements | Historical installed versions incompletely verified; do not treat a later local installation as proof |
| Original DistilBERT runs | Original runner did not log package versions | Library versions unknown; Google Colab T4 reported |
| Added legacy conditions, original RoBERTa and parity run | `results/raw/*/*_config.json`, `results/validation/` | Python 3.12.13; torch 2.10.0+cu128; transformers 5.0.0 |
| Author provenance reconstruction | Supplied provenance report | Python 3.12.4; NumPy 2.3.2; pandas 2.3.3 |

`requirements-analysis.txt` provides CPU analysis/test dependencies. `requirements.txt` includes training dependencies. `requirements-paired-observed.txt` records the observed numerical versions, with comments identifying the transformer stack; it is not a complete transitive lock or a universal CUDA install recipe. Kaggle's existing compatible CUDA/PyTorch installation should be preserved when repeating training.

The old universal `requirements-lock.txt` mixed unverified historical attributions. It is retained as evidence under `archive/environment_records/requirements-lock_legacy.txt` and is not an installation recommendation.

The paired cache records spaCy `en_core_web_sm` as its entity method; its complete historical generation environment is not established merely by the cache hash. The paired runner refuses an unapproved regex fallback and rejects resumes with incompatible protocol/software records. TensorFlow is not required.
