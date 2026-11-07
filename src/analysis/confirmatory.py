"""
confirmatory.py -- seed-level paired contrasts with exact sign-flip p-values and Holm correction.

    python src/analysis/confirmatory.py --results results/raw/tfidf/*.csv results/raw/roberta/results_roberta.csv \
        --out results/final/confirmatory

Unit of inference = the seed (as in the paper).  For each model and contrast (A - B) the script
  1. takes cross-dataset AUC (or the in-domain minus cross gap) per (seed, train, test),
  2. averages the paired difference over the ordered pairs -> one value per seed,
  3. reports mean, 95% t-interval, sign agreement, exact paired sign-flip p-value (2^n sign patterns),
  4. Holm-adjusts the p-values within the PRIMARY family only; everything else is labelled exploratory.
With n seeds the smallest attainable two-sided sign-flip p is 2/2^n (0.0078 for n=8, 0.0005 for n=12), so a family of m primary
contrasts can only survive Holm if 2/2^n <= alpha/m.  The script prints that bound; it is why more seeds matter for the
confirmatory family but not for exploratory work.  Seeds measure resampling noise on a fixed document universe, NOT
generalisation to other corpora: for that, report the leave-one-corpus-out column (--loo).
"""
import argparse, itertools, pathlib
import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = [("S_vs_R_S", "S", "R_S"), ("E_vs_R_E", "E", "R_E"), ("SE_vs_R_SE", "SE", "R_SE"),
           ("L_vs_C0", "L", "C0"), ("LE_vs_L_RE", "LE", "L_RE"), ("SLE_vs_L_RSE", "SLE", "L_RSE")]
EXPLORATORY = [("S_vs_C0", "S", "C0"), ("E_vs_C0", "E", "C0"), ("LE_vs_C0", "LE", "C0"), ("SLE_vs_C0", "SLE", "C0"),
               ("R_SE_vs_C0", "R_SE", "C0"), ("L_RSE_vs_L", "L_RSE", "L")]


def seed_values(df, ca, cb, metric, loo=None):
    cols = ["seed", "train", "test"]
    x = df[df["cond"] == ca].set_index(cols); y = df[df["cond"] == cb].set_index(cols)
    j = x.join(y, lsuffix="_a", rsuffix="_b", how="inner")
    if loo:
        j = j[(j.index.get_level_values("train") != loo) & (j.index.get_level_values("test") != loo)]
    return j.groupby(level="seed").apply(lambda g: (g[f"{metric}_a"] - g[f"{metric}_b"]).mean())


def cross_gap(df):
    cross = df[df["kind"] == "cross"].set_index(["seed", "cond", "train", "test"])["auc"]
    inn = df[df["kind"] == "in"].set_index(["seed", "cond", "train"])["auc"]
    out = df[df["kind"] == "cross"].copy()
    out["gap"] = [inn.get((s, c, tr), np.nan) - v for (s, c, tr, _), v in cross.items()]
    return out


def signflip_p(v):
    v = np.asarray(v, float); n = len(v)
    if n > 20:
        rng = np.random.RandomState(0)
        s = rng.choice([-1, 1], size=(200000, n)); return float((np.abs((s * v).mean(1)) >= abs(v.mean()) - 1e-15).mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    return float((np.abs((signs * v).mean(1)) >= abs(v.mean()) - 1e-15).mean())


def holm(p, alpha=0.05):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def summarise(v):
    n = len(v); m = float(np.mean(v)); se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    t = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    return dict(n_seeds=n, mean=m, lo=m - t * se, hi=m + t * se, same_sign=int((np.sign(v) == np.sign(m)).sum()), p_signflip=signflip_p(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="results/final/confirmatory")
    ap.add_argument("--metric", default="auc", choices=["auc", "gap"])
    ap.add_argument("--loo", action="store_true", help="also repeat every contrast leaving each corpus out (sensitivity to dataset choice)")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.results], ignore_index=True)
    rows = []
    for model, d in df.groupby("model"):
        base = cross_gap(d) if a.metric == "gap" else d[d["kind"] == "cross"]
        metric = "gap" if a.metric == "gap" else "auc"
        corpora = sorted(set(d["train"]))
        for fam, contrasts in (("primary", PRIMARY), ("exploratory", EXPLORATORY)):
            for name, ca, cb in contrasts:
                if not {ca, cb} <= set(d["cond"]):
                    continue
                for loo in ([None] + (corpora if a.loo else [])):
                    v = seed_values(base, ca, cb, metric, loo)
                    if len(v) >= 2:
                        rows.append(dict(model=model, family=fam, contrast=name, left_out=loo or "", **summarise(v.to_numpy())))
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for (model, fam), g in res[res["left_out"] == ""].groupby(["model", "family"]):
        if fam == "primary":
            res.loc[g.index, "p_holm"] = holm(g["p_signflip"])
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(str(out) + f"_{a.metric}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res[res["left_out"] == ""].round(4).to_string(index=False))
    n = int(res["n_seeds"].max()); m = len(PRIMARY)
    print(f"\nsmallest attainable sign-flip p with {n} seeds = {2 / 2 ** n:.4f}; Holm needs it <= 0.05/{m} = {0.05 / m:.4f} "
          f"for the strongest primary contrast -> {'attainable' if 2 / 2 ** n <= 0.05 / m else 'NOT attainable: use more seeds for the primary family'}")


if __name__ == "__main__":
    main()
