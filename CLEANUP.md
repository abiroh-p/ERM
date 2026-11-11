# Removing dataset text from the repository

Why: the paper and README say the datasets are not redistributed, but the repository tracks the McIntire CSV, a parquet file with 8,670 full-text documents from all three corpora, and audit tables holding 200-character excerpts, and the history holds older copies under other paths (e.g. `fake_or_real_news.csv`, `results/audit_01_05/`). Check each dataset's license; this procedure makes the statement true.

Everything below was tested end to end on scratch copies (pack size 37.6 MiB to 2.6 MiB, 23 text-bearing paths to 0). It has not been run on your GitHub repository.

## Step A. Stop tracking (safe, reversible; no history rewrite)

```bash
python scripts/repo_cleanup.py                 # dry run: shows what changes
python scripts/repo_cleanup.py --apply         # untracks files (they stay on disk), strips text_i/text_j from the audit tables
git commit -m "Stop tracking dataset text; strip excerpts from audit tables"
git push
```
Add `--keep-label-checks` if you want to keep the `label_check_*.txt` excerpts. This alone fixes the current files but not old commits.

## Step B. (Optional but recommended) purge history

Old commits still contain the data, and the repository is public. Rewriting history changes every commit hash and needs a force-push.

1. Back up first: `git clone --mirror https://github.com/abiroh-p/ERM.git ERM_backup.git`
2. `pip install git-filter-repo`
3. List every path that ever held dataset text (run in your normal repo, after step A):
   ```bash
   python scripts/find_text_in_history.py --write-paths ../purge_paths.txt
   ```
   Read the list. It is a heuristic that over-reports rather than misses; remove a line only if you are sure that path never held text.
4. Rewrite a fresh clone (never your working copy):
   ```bash
   cd ..
   git clone https://github.com/abiroh-p/ERM.git ERM_purged
   cd ERM_purged
   git filter-repo --force --invert-paths --paths-from-file ../purge_paths.txt
   ```
5. The purge also removes the four text-free audit tables from the current tree. Copy them back from your working repo and commit:
   ```bash
   python - <<'PY'
   import shutil, pathlib
   for f in ["results/audit/primary/overlap_pairs.csv", "results/audit/smoke/overlap_pairs.csv",
             "results/audit/primary/borderline_to_inspect.csv", "results/audit/smoke/borderline_to_inspect.csv"]:
       dst = pathlib.Path(f); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(pathlib.Path("../ERM") / f, dst)
   PY
   git add -A && git commit -m "Add text-free audit tables"
   ```
   (On Windows PowerShell, run the Python snippet from a file instead of the here-document.)
6. Verify: `python ../ERM/scripts/find_text_in_history.py` must print `0 paths`; `git count-objects -vH` shows the new size.
7. Publish: `git remote add origin https://github.com/abiroh-p/ERM.git` then `git push --force --all` (and `--tags` if you have tags).
8. Re-clone the repository everywhere you use it (including Kaggle notebooks), and re-run `python -m pytest tests -q`.

## Limits you should know
- Forks, clones and caches made earlier still hold the old data; GitHub may keep unreachable commits viewable by hash until you ask GitHub support to clear cached views.
- Any DOI/archive (e.g. Zenodo) or paper that cites an old commit hash will point to a commit that no longer exists. Do the purge before you cite a commit.
- The audit tables' `i`/`j` columns are row positions in the cleaned corpus, so they are only meaningful together with a re-run of the audit; the dataset files themselves must be downloaded from the original sources (README, Data).
