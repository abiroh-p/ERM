# Celebrity external-validation extension

Preparation only; **no Celebrity model results have been generated**. This stays
separate from the released three-corpus results and manuscript. Start with
[`AUDIT_REPORT.md`](AUDIT_REPORT.md), the frozen
[`protocol`](../../docs/EXTERNAL_CELEBRITY_PROTOCOL_v1.md) and
[`configuration`](../../configs/external_celebrity_v1.json).

The archive and exact frozen analysis input were available and hash-verified.
After conservative group exclusions: **496 primary articles (248 per label)**;
**428 secondary articles (214 per label)**. Numeric story groups are provisional.
The original paired source selections match all 24 saved model/seed manifests.
The required spaCy English model is currently absent; an attempted model download
did not complete. The production NER stage refuses regex fallback, so scoring
remains blocked until the separate external cache is prepared.

## Local CPU commands

Run at repository/package root. On this machine use `.venv/Scripts/python.exe`
in place of `python`; elsewhere create a compatible environment first:

```powershell
python -m pip install -r requirements-analysis.txt "spacy>=3.7"
python -m spacy download en_core_web_sm --no-deps --no-cache-dir
$celebrityZip = 'C:\Users\abish\Downloads\fakeNewsDatasets.zip'
```

The supplied audit/selections are already frozen. To verify/reuse selections:

```powershell
python scripts/external_celebrity.py select --zip $celebrityZip
python scripts/external_celebrity.py ner --zip $celebrityZip
python scripts/external_celebrity.py plan --zip $celebrityZip --model tfidf
python scripts/external_celebrity.py run --zip $celebrityZip --model tfidf
python scripts/external_celebrity.py check --zip $celebrityZip --model tfidf
python scripts/external_celebrity.py analyze --zip $celebrityZip --model tfidf
```

Default inputs are `data/processed/analysis_set.csv`, `cache/spans_cache.pkl`;
provide explicit `--analysis-csv` and `--spans-cache` if elsewhere. The source
cache hash observed here is
`f16342d4339666ac924f42b61eb42282d70ed16f5128a3bc4ba2ef1f13ac80bc`, identical
to the saved paired protocol. The old cache is read only; the new cache goes
under `results/external_celebrity/prepared_v1/spans.pkl` (ignored, not packaged).
Keep the new cache and NER provenance together. Historical exact NER package
version is unknown; current target extraction metadata will be recorded.

To rerun the full audit, use a **new empty output directory**. Existing reports
are deliberately not overwritten. These commands do not train or score models:

```powershell
python src/audit/celebrity_audit.py --zip $celebrityZip --analysis-csv data/processed/analysis_set.csv --out runs/celebrity_audit_recheck
python scripts/external_celebrity.py select --zip $celebrityZip --audit runs/celebrity_audit_recheck --prepared runs/celebrity_prepared_recheck
```

If the source CSV is missing, omit `--analysis-csv` in the audit command. It will
record cross-corpus overlap as incomplete; source selection/scoring will refuse
to proceed. A different CSV is rejected by hash. The original source archive
and data ZIP are not included in the runnable package.

Optional full-repository supplemental verification (requires the three unchanged
account ZIPs in `results/paired/accounts/`; their verified hashes/24 comparisons
are already saved in `prepared_v1/verification.json`):

```powershell
python scripts/verify_celebrity_preparation.py --zip $celebrityZip
```

## Transformer commands

Install the existing training requirements in an appropriate GPU environment;
retain compatible CUDA/PyTorch rather than replacing a working CUDA stack.
No trained task checkpoints were found: every job below retrains on a source
corpus. Each model needs 240 fits for the complete frozen ten-condition design.

```powershell
python -m pip install -r requirements.txt
python scripts/external_celebrity.py plan --zip $celebrityZip --model distilbert
python scripts/external_celebrity.py plan --zip $celebrityZip --model roberta
```

For each model, run these in two separate terminals, with one GPU per process:

```powershell
# Terminal 1
$env:CUDA_VISIBLE_DEVICES = '0'
python scripts/external_celebrity.py run --zip $celebrityZip --model distilbert --rank 0 --world-size 2
```

```powershell
# Terminal 2
$env:CUDA_VISIBLE_DEVICES = '1'
python scripts/external_celebrity.py run --zip $celebrityZip --model distilbert --rank 1 --world-size 2
```

Repeat both worker commands with `--model roberta` after the DistilBERT workers
finish. One GPU can run sequentially with `--rank 0 --world-size 1`. Plan on the
same software stack used by workers; plan freezes actual versions and code.
TF-IDF can also be partitioned with rank/world-size in separate CPU terminals.

```powershell
python scripts/external_celebrity.py check --zip $celebrityZip --model distilbert
python scripts/external_celebrity.py analyze --zip $celebrityZip --model distilbert
python scripts/external_celebrity.py check --zip $celebrityZip --model roberta
python scripts/external_celebrity.py analyze --zip $celebrityZip --model roberta
```

All stages accept `--output` to choose a new run root (default
`runs/external_celebrity_v1`). Model subdirectories contain frozen plans, one
atomic result per fit, completion manifests and separate source-level analyses.
Repeat the same run command to resume. Check returns exit code 2 for incomplete
grids and refuses corrupt or incompatible results. Analysis may summarize partial
outcomes but marks incompleteness; it never imputes missing cells.

Expected compute: 240 TF-IDF CPU fits, 240 DistilBERT GPU fits and 240 RoBERTa GPU
fits. Actual frozen source training sizes range from 2,988–2,994 documents.
Recommend 16 GB host RAM and one 16 GB CUDA GPU per transformer worker. At two
epochs/batch 16, each actual transformer fit takes 374–376 optimizer steps,
at most 90,240/model. Audit/preparation are CPU tasks; no paid/GPU jobs were
launched. No measured full-model runtime estimate is available: measure pilot
job elapsed times before reserving compute. A run processes jobs in stable order;
interrupting leaves only the active fit to repeat, with a stale-lock check.

## Checks and package

```powershell
New-Item -ItemType Directory -Path runs -Force | Out-Null
python -m pytest tests/test_external_celebrity.py -q -p no:cacheprovider --basetemp runs/celebrity_tests_fresh
python scripts/package_external_celebrity.py
```

Use a fresh test directory for each invocation. See
[`VALIDATION.md`](VALIDATION.md) for the actual completed checks. Full repository
tests use `python -m pytest tests -q`; the smaller runnable package includes the
extension's self-contained test module. Package outputs are a repository patch,
a runnable ZIP and SHA-256 checksums under `deliverables/external_celebrity_v1/`.
The package contains scripts, protocol, IDs and evidence, not article text or
weights. Its `PACKAGE_MANIFEST.json` hashes every included file. A code-license
grant does not extend to third-party article text; supplied notices are retained.
