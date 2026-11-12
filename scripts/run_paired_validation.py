"""Kaggle entry point; independent transformer jobs are distributed across both T4s."""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
import exp_transformer as T

FOCUSED = ["C0", "E", "R_E", "L", "LE", "L_RE", "SLE", "L_RSE"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "tfidf", "distilbert", "roberta", "all"], required=True)
    ap.add_argument("--analysis-csv", required=True)
    ap.add_argument("--spans-cache", required=True)
    ap.add_argument("--output-dir", default="/kaggle/working/erm_paired_v1")
    ap.add_argument("--design", choices=["focused", "full"], default="focused")
    a = ap.parse_args()
    data = str(pathlib.Path(a.analysis_csv).resolve())
    cache = str(pathlib.Path(a.spans_cache).resolve())
    out = pathlib.Path(a.output_dir).resolve()
    runner = str(ROOT / "src/experiments/exp_transformer.py")
    common = ["--analysis-csv", data, "--spans-cache", cache]
    if a.stage == "prepare":
        subprocess.run([sys.executable, runner, "prepare", *common], check=True, cwd=ROOT)
        return
    stages = ["tfidf", "distilbert", "roberta"] if a.stage == "all" else [a.stage]
    for stage in stages:
        # CPU runs can economically cover the complete design; do not confuse the
        # focused transformer subset with a full factorial replication.
        conditions = T.PRIMARY_CONDS if stage == "tfidf" or a.design == "full" else FOCUSED
        target = out / stage / f"results_{stage}.csv"
        settings = ["--backend", "tfidf", "--workers", "2"] if stage == "tfidf" else [
            "--model", {"distilbert": "distilbert-base-uncased", "roberta": "roberta-base"}[stage], "--gpus", "0", "1"]
        command = [sys.executable, runner, "run", "--sampling", "paired", *common, *settings,
                   "--datasets", *T.LEGACY_ORDER, "--seeds", *map(str, range(8)), "--conds", *conditions,
                   "--out", str(target)]
        print(f"{stage}: {len(conditions)} conditions x 8 seeds x 3 training corpora; output {target}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
