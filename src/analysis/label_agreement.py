"""
label_agreement.py -- Cohen's kappa for the manual label check written by add_dataset.py (label_check_<name>.csv).

Fill `annotator_1_label` and `annotator_2_label` with fake/real (or 1/0) per document, then:
    python src/analysis/label_agreement.py data/audit/label_check_NewsX.csv data/audit/label_key_NewsX.csv
Reports: inter-annotator kappa, and the agreement of each annotator (and of their consensus) with the dataset label.
Disagreements are written next to the input so they can be documented in the paper.
"""
import sys
import numpy as np
import pandas as pd


def to01(s):
    return s.astype(str).str.strip().str.lower().map({"1": 1, "fake": 1, "f": 1, "0": 0, "real": 0, "r": 0})


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in (0, 1))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(a, b, n=2000, seed=0):
    rng, k = np.random.RandomState(seed), len(a)
    v = [kappa(a[i], b[i]) for i in (rng.randint(0, k, k) for _ in range(n))]
    return np.nanpercentile(v, [2.5, 97.5])


def main(path, key_path):
    d = pd.read_csv(path).merge(pd.read_csv(key_path), on="doc_id", how="left")
    d["a1"], d["a2"] = to01(d["annotator_1_label"]), to01(d["annotator_2_label"])
    d = d.dropna(subset=["a1", "a2"]).astype({"a1": int, "a2": int})
    if d.empty:
        sys.exit("no complete annotations found")
    lo, hi = boot_ci(d["a1"].to_numpy(), d["a2"].to_numpy())
    print(f"n={len(d)}  Cohen's kappa (annotator 1 vs 2) = {kappa(d['a1'], d['a2']):.3f}  (bootstrap 95% CI {lo:.3f}-{hi:.3f})")
    print(f"raw agreement between annotators: {np.mean(d['a1'] == d['a2']):.3f}")
    for k in ("a1", "a2"):
        print(f"  {k} agrees with dataset label: {np.mean(d[k] == d['is_fake']):.3f}   kappa {kappa(d[k], d['is_fake']):.3f}")
    both = d[d["a1"] == d["a2"]]
    print(f"  consensus (both agree, n={len(both)}) vs dataset label: {np.mean(both['a1'] == both['is_fake']):.3f}")
    dis = d[d["a1"] != d["a2"]]
    out = path.replace(".csv", "_disagreements.csv")
    dis.to_csv(out, index=False)
    print(f"{len(dis)} annotator disagreements -> {out}")
    for c in (0, 1):
        sub = both[both["is_fake"] == c]
        if len(sub):
            print(f"  dataset label {'fake' if c else 'real'}: consensus agrees on {np.mean(sub['a1'] == c):.3f} of {len(sub)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
