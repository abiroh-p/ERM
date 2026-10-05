"""
freeze_and_analyze.py -- freeze the results used in the paper and regenerate EVERY statistical table from the frozen copies.

    python -X utf8 freeze_and_analyze.py ^
        --tfidf results_tfidf_a.csv results_tfidf_b.csv results_tfidf_c.csv results_tfidf_d1.csv results_tfidf_d2.csv results_tfidf_d3.csv ^
        --bert results_distilbert.csv ^
        --sens results_tfidf_x1.csv results_tfidf_x2.csv results_tfidf_x3.csv ^
        --extra flagged_doc_ids.txt analysis_set.csv

(^ is the Windows line-continuation character; or put it all on one line. --sens and --extra are optional.)

What it does
  1. Copies the input files to freeze/inputs/ and writes MANIFEST.json (SHA-256, row counts, library versions, timestamp).
  2. Runs, on the FROZEN COPIES, in separate output folders:
        outputs/tfidf/           analyze.py on the TF-IDF results (pooled, factorial, random controls, per-pair, exploratory)
        outputs/distilbert/      analyze.py on the DistilBERT results
        outputs/model_comparison compare_models.py (paired-by-seed model differences)
        outputs/sensitivity/     analyze.py on the duplicate-removal reruns (if --sens given)
     Each folder also gets analysis_log.txt with the full console output.
Nothing in freeze/inputs is ever modified. If you must change a result later, rerun this script into a NEW folder (--out).
Uses src/analysis/analyze.py and src/analysis/compare_models.py from this
repository. The fallback paths support the historical flat layout.
"""
import argparse, hashlib, json, pathlib, platform, shutil, subprocess, sys, datetime
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ANALYZE_SCRIPT = (HERE / "src" / "analysis" / "analyze.py").resolve()
COMPARE_SCRIPT = (HERE / "src" / "analysis" / "compare_models.py").resolve()

if not ANALYZE_SCRIPT.exists():
    ANALYZE_SCRIPT = (HERE / "analyze.py").resolve()
if not COMPARE_SCRIPT.exists():
    COMPARE_SCRIPT = (HERE / "compare_models.py").resolve()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd, log_name="analysis_log.txt"):
    cwd.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8"] + cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    (cwd / log_name).write_text(r.stdout + ("\n--- STDERR ---\n" + r.stderr if r.stderr else ""), encoding="utf-8")
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"  {status:14s} {cwd.name}")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", nargs="+", required=True)
    ap.add_argument("--bert", nargs="+", required=True)
    ap.add_argument("--sens", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[], help="other files to hash and keep (flagged ids, analysis set, configs)")
    ap.add_argument("--out", default="freeze")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if (out / "MANIFEST.json").exists():
        sys.exit(f"{out} already contains a frozen analysis. Use --out <new folder> instead of overwriting it.")
    inp = out / "inputs"
    for sub in ("tfidf", "distilbert", "sensitivity", "extra"):
        (inp / sub).mkdir(parents=True, exist_ok=True)

    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "files": {}}
    for lib in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            manifest[lib] = __import__(lib).__version__
        except Exception:
            manifest[lib] = "not installed"

    groups = {"tfidf": a.tfidf, "distilbert": a.bert, "sensitivity": a.sens, "extra": a.extra}
    frozen = {k: [] for k in groups}
    for grp, files in groups.items():
        for f in files:
            src = pathlib.Path(f)
            if not src.exists():
                sys.exit(f"missing input file: {src}")
            dst = inp / grp / src.name
            shutil.copy2(src, dst)
            info = {"sha256": sha(dst), "bytes": dst.stat().st_size}
            if dst.suffix == ".csv":
                try:
                    d = pd.read_csv(dst)
                    info["rows"] = len(d)
                    if {"seed", "cond"} <= set(d.columns):
                        info["seeds"] = sorted(int(s) for s in d.seed.unique()); info["conditions"] = sorted(d.cond.unique())
                except Exception as e:
                    info["read_error"] = str(e)
            manifest["files"][f"{grp}/{src.name}"] = info
            frozen[grp].append(str(dst.resolve()))
            try:
                dst.chmod(0o444)
            except Exception:
                pass
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"frozen {sum(len(v) for v in frozen.values())} files into {inp} (see MANIFEST.json)\nregenerating tables from the frozen copies:")

    o = out / "outputs"
    run([str(ANALYZE_SCRIPT), "--output-dir", str((o / "tfidf").resolve())] + frozen["tfidf"], o / "tfidf")
    run([str(ANALYZE_SCRIPT), "--output-dir", str((o / "distilbert").resolve())] + frozen["distilbert"], o / "distilbert")
    run([str(COMPARE_SCRIPT), "--tfidf"] + frozen["tfidf"] + ["--bert"] + frozen["distilbert"], o / "model_comparison")
    if frozen["sensitivity"]:
        run([str(ANALYZE_SCRIPT), "--output-dir", str((o / "sensitivity").resolve())] + frozen["sensitivity"], o / "sensitivity")
    print(f"\nDone. Every table in the paper should come from {o}. Console output of each step: <folder>/analysis_log.txt")


if __name__ == "__main__":
    main()
