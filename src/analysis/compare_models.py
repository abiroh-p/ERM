"""
compare_models.py -- does the effect of an intervention differ between TF-IDF and DistilBERT?  (paired by seed)

    python compare_models.py --tfidf results_tfidf_a.csv results_tfidf_b.csv results_tfidf_c.csv results_tfidf_d1.csv results_tfidf_d2.csv results_tfidf_d3.csv
                             --bert results_distilbert.csv

Why: "significant in one model, not in the other" does NOT show that the models differ. The quantity of interest is the DIFFERENCE between
the two models' intervention effects, with its own uncertainty:

        D_s = (TF-IDF effect)_s - (DistilBERT effect)_s ,    s = seed.

Definitions
  * effect of contrast A-B for one model, seed s, ordered pair p :   metric(s, A, p) - metric(s, B, p)        (paired: same seed, same pair)
  * POOLED effect for seed s : the EQUAL-WEIGHT mean of that effect over the six ordered pairs (each direction counts 1/6).
    This is a descriptive summary of these six pairs, not an estimate for "datasets in general"; see the per-pair table below.
  * Unit of inference = SEED (the pairs within a seed share training sets and are not independent replicates). The seeds all draw from the
    same fixed document universe and the same three datasets, so seed-to-seed variation reflects resampling and training randomness only.
  * Intervals are 95% t-intervals over the n seed values (df = n-1); '(k/n)' = seeds agreeing in sign with the mean.
    These are DESCRIPTIVE evidence, not a significance rule. No p-values: with five seeds an exact paired sign-flip test cannot go below
    p = 2/32 = 0.0625.
  * Pairing of the two arms assumes both used identical documents for each (seed, condition, train dataset); the script checks that the
    sample sizes agree cell by cell and stops if they do not.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

CONTRASTS = [("S", "C0"), ("L", "C0"), ("E", "C0"), ("LE", "C0"), ("SLE", "C0"),
             ("S", "R_S"), ("E", "R_E"), ("SE", "R_SE")]
OUTCOMES = [("auc", "cross-dataset AUC"), ("gap_auc", "AUC gap")]


REQUIRED = ["seed", "cond", "train", "test", "kind", "auc", "n_train", "n_test", "model"]


def load(paths, name, model_values):
    r = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    miss = [c for c in REQUIRED if c not in r.columns]
    if miss:
        raise SystemExit(f"[{name}] missing columns: {miss}")
    if not np.isfinite(r["auc"]).all():
        raise SystemExit(f"[{name}] non-finite AUC values in {int((~np.isfinite(r['auc'])).sum())} rows")
    got = set(r["model"].unique())
    if not got <= set(model_values):
        raise SystemExit(f"[{name}] model column is {sorted(got)}, expected {sorted(model_values)} -- did you swap --tfidf and --bert?")
    d = r.duplicated(subset=["seed", "cond", "train", "test"], keep=False)
    if d.any():
        raise SystemExit(f"[{name}] {int(d.sum())} duplicated (seed, cond, train, test) rows")
    ds = sorted(set(r.train) | set(r.test))
    n_pairs = len(ds) * (len(ds) - 1)
    cross = r[r.kind == "cross"]
    cnt = cross.groupby(["seed", "cond"]).size()
    bad = cnt[cnt != n_pairs]
    if len(bad):
        raise SystemExit(f"[{name}] incomplete ordered-pair sets (expected {n_pairs} per seed and condition):\n{bad.head(10)}")
    inn = r[r.kind == "in"][["seed", "cond", "train", "auc"]].rename(columns={"auc": "auc_in"})
    x = cross.merge(inn, on=["seed", "cond", "train"], how="left")
    if x["auc_in"].isna().any():
        raise SystemExit(f"[{name}] cross rows without a matching in-domain row")
    x["gap_auc"] = x.auc_in - x.auc
    x["pair"] = x.train + "->" + x.test
    return x.set_index(["seed", "pair"]), ds


def per_seed_effect(X, a, b, outcome):
    ea = X[X.cond == a][outcome]; eb = X[X.cond == b][outcome]
    d = (ea - eb).dropna()
    return d.groupby(level="seed").mean()


def summarise(v):
    v = np.asarray(v, float); n = len(v)
    m, sd = v.mean(), v.std(ddof=1) if n > 1 else np.nan
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return dict(mean=m, sd=sd, lo=m - h, hi=m + h, min=v.min(), max=v.max(), same_sign=int((np.sign(v) == np.sign(m)).sum()), n=n)


def fmt(s, d=3):
    return f"{s['mean']:+.{d}f} [{s['lo']:+.{d}f}, {s['hi']:+.{d}f}] ({s['same_sign']}/{s['n']})"


PAIR_CONTRASTS = [("SLE", "C0"), ("LE", "C0"), ("L", "C0"), ("E", "R_E"), ("S", "R_S")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", nargs="+", required=True)
    ap.add_argument("--bert", nargs="+", required=True)
    a = ap.parse_args()
    T, dsT = load(a.tfidf, "tfidf", ["tfidf_lr"])
    B, dsB = load(a.bert, "bert", ["distilbert"])
    if dsT != dsB:
        raise SystemExit(f"datasets differ between arms: {dsT} vs {dsB}")
    sT, sB = set(T.index.get_level_values("seed")), set(B.index.get_level_values("seed"))
    seeds = sorted(sT & sB)
    if not seeds:
        raise SystemExit("no shared seeds")
    if sT != sB:
        print(f"WARNING: seed sets differ (tfidf {sorted(sT)}, bert {sorted(sB)}); using the shared seeds {seeds}")
    # pairing check: same sample sizes cell by cell
    ta = T.reset_index().set_index(["seed", "cond", "train", "test"])[["n_train", "n_test"]]
    tb = B.reset_index().set_index(["seed", "cond", "train", "test"])[["n_train", "n_test"]]
    j = ta.join(tb, lsuffix="_t", rsuffix="_b", how="inner")
    mism = j[(j.n_train_t != j.n_train_b) | (j.n_test_t != j.n_test_b)]
    if len(mism):
        raise SystemExit(f"PAIRING INVALID: {len(mism)} of {len(j)} shared cells have different sample sizes in the two arms "
                         f"(different universe, --exclude file or configuration?). First rows:\n{mism.head()}")
    print(f"pairing check passed on {len(j)} shared cells; datasets {dsT}; seeds {seeds}")
    print("Unit = seed; pooled effect = equal-weight mean over the six ordered pairs; intervals = t, df = n-1; (k/n) = seeds agreeing in sign.\n")

    rows, per_seed, skipped = [], [], []
    for outcome, label in OUTCOMES:
        for ca, cb in CONTRASTS:
            if not all(c in set(T.cond) and c in set(B.cond) for c in (ca, cb)):
                skipped.append(f"{ca}-{cb}"); continue
            et = per_seed_effect(T, ca, cb, outcome).reindex(seeds)
            eb = per_seed_effect(B, ca, cb, outcome).reindex(seeds)
            diff = (et - eb)
            st, sb, sd = summarise(et.dropna()), summarise(eb.dropna()), summarise(diff.dropna())
            rows.append(dict(outcome=label, contrast=f"{ca} - {cb}", tfidf=fmt(st), distilbert=fmt(sb), difference_tfidf_minus_bert=fmt(sd),
                             tfidf_seed_range=f"{st['min']:+.3f}..{st['max']:+.3f}", bert_seed_range=f"{sb['min']:+.3f}..{sb['max']:+.3f}"))
            for s_ in seeds:
                per_seed.append(dict(outcome=label, contrast=f"{ca} - {cb}", seed=s_, tfidf=et[s_], distilbert=eb[s_], difference=diff[s_]))
    if skipped:
        print(f"NOTE: contrasts skipped because a condition is missing in one arm: {sorted(set(skipped))}\n")
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    for _, g in out.groupby("outcome", sort=False):
        print(f"=== POOLED over the six pairs, {g.outcome.iloc[0]}: effect per model and their difference ===")
        print(g.drop(columns="outcome").to_string(index=False)); print()
    out.to_csv("table_model_comparison.csv", index=False)
    ps = pd.DataFrame(per_seed); ps.to_csv("table_per_seed_effects.csv", index=False)

    # ---- pair-specific model contrasts (cross-dataset AUC) ----
    print("=== PER ORDERED PAIR, cross-dataset AUC: model difference (TF-IDF effect - DistilBERT effect), seed-based interval ===")
    prow, grid = [], {}
    pairs = sorted(set(T.index.get_level_values("pair")))
    for ca, cb in PAIR_CONTRASTS:
        if not all(c in set(T.cond) and c in set(B.cond) for c in (ca, cb)):
            continue
        for pr in pairs:
            ea = (T[T.cond == ca]["auc"] - T[T.cond == cb]["auc"]).xs(pr, level="pair").reindex(seeds)
            eb = (B[B.cond == ca]["auc"] - B[B.cond == cb]["auc"]).xs(pr, level="pair").reindex(seeds)
            sd_ = summarise((ea - eb).dropna())
            prow.append(dict(contrast=f"{ca} - {cb}", pair=pr, tfidf_effect=ea.mean(), bert_effect=eb.mean(), difference=sd_["mean"],
                             lo=sd_["lo"], hi=sd_["hi"], seeds_same_sign=f"{sd_['same_sign']}/{sd_['n']}"))
            grid.setdefault(pr, {})[f"{ca}-{cb}"] = fmt(sd_)
    pd.DataFrame(prow).to_csv("table_model_comparison_by_pair.csv", index=False)
    print(pd.DataFrame(grid).T.to_string())
    print("\nHow to read: an interval that excludes zero, with most seeds agreeing in sign, is evidence that the two models respond differently; "
          "anything else is 'not established' with this few seeds. These are descriptive intervals, not hypothesis tests, and nothing is corrected "
          "for multiple comparisons.\nSaved: table_model_comparison.csv, table_model_comparison_by_pair.csv, table_per_seed_effects.csv")


if __name__ == "__main__":
    main()
