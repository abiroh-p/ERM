#!/usr/bin/env python3
"""
repo_cleanup.py -- stop tracking dataset text in the ERM repository (dry run by default).

    python scripts/repo_cleanup.py            # show what would change
    python scripts/repo_cleanup.py --apply    # do it (does NOT commit or push)

What it does
  * UNTRACKS (git rm --cached; your local files stay) every tracked file that holds dataset article text:
      - data/raw/McIntire/fake_or_real_news.csv           the McIntire dataset itself
      - results/audit/smoke/cleaned_all.parquet           8,670 full-text documents from all three corpora
      - results/audit/*/label_check_*.txt                 hand-check excerpts (skip with --keep-label-checks)
  * STRIPS the text columns (text_i, text_j: 200-character excerpts) from the audit tables, keeping every other column:
      - results/audit/primary|smoke/overlap_pairs.csv
      - results/audit/primary|smoke/borderline_to_inspect.csv
    and stages the text-free versions.
  * appends the matching patterns to .gitignore.
It does not rewrite history: files already committed stay in old commits (see docs/DATA_HANDLING.md for the optional purge).
"""
import argparse, pathlib, subprocess, sys
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
STRIP = ["results/audit/primary/overlap_pairs.csv", "results/audit/smoke/overlap_pairs.csv",
         "results/audit/primary/borderline_to_inspect.csv", "results/audit/smoke/borderline_to_inspect.csv"]
TEXT_COLS = ["text_i", "text_j"]
IGNORE = ["# dataset text is not redistributed (see README, Data)", "data/raw/", "results/audit/smoke/cleaned_all.parquet",
          "results/audit/*/label_check_*.txt", "results/audit/*/*_with_text.csv"]


def git(*args, capture=False):
    r = subprocess.run(["git", *args], cwd=REPO, check=True, text=True, capture_output=capture)
    return r.stdout if capture else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--keep-label-checks", action="store_true")
    a = ap.parse_args()
    tracked = set(git("ls-files", capture=True).splitlines())
    untrack = [f for f in ["data/raw/McIntire/fake_or_real_news.csv", "results/audit/smoke/cleaned_all.parquet"] if f in tracked]
    if not a.keep_label_checks:
        untrack += sorted(f for f in tracked if "/label_check_" in f and f.endswith(".txt"))
    strip = [f for f in STRIP if f in tracked]
    ignore = list(IGNORE) if not a.keep_label_checks else [x for x in IGNORE if "label_check" not in x]

    print(("APPLYING" if a.apply else "DRY RUN") + f" in {REPO}\n")
    print("untrack (files stay on disk):"); [print("  ", f) for f in untrack]
    print("strip text columns", TEXT_COLS, "and re-stage:")
    for f in strip:
        p = REPO / f; cols = list(pd.read_csv(p, nrows=0).columns)
        print("  ", f, "->", [c for c in cols if c not in TEXT_COLS])
    gi = REPO / ".gitignore"; have = gi.read_text().splitlines() if gi.exists() else []
    add = [x for x in ignore if x not in have]
    print(".gitignore additions:"); [print("  ", x) for x in add]
    if not a.apply:
        print("\nNothing changed. Re-run with --apply."); return

    for f in strip:
        p = REPO / f; d = pd.read_csv(p)
        d.drop(columns=[c for c in TEXT_COLS if c in d.columns]).to_csv(p, index=False)
        git("add", f)
    for f in untrack:
        git("rm", "--cached", "--quiet", f)
    if add:
        with gi.open("a") as h:
            h.write("\n" + "\n".join(add) + "\n")
        git("add", ".gitignore")
    # verify
    now = set(git("ls-files", capture=True).splitlines())
    left = [f for f in untrack if f in now]
    bad = [f for f in strip if any(c in pd.read_csv(REPO / f, nrows=0).columns for c in TEXT_COLS)]
    if left or bad:
        sys.exit(f"verification failed: still tracked {left}, text columns remain in {bad}")
    print("\nDone and verified. Review with `git status`, then:\n  git commit -m \"Stop tracking dataset text; strip excerpts from audit tables\"")
    print("History still contains the old files; see docs/DATA_HANDLING.md before making the repository public or citing it.")


if __name__ == "__main__":
    main()
