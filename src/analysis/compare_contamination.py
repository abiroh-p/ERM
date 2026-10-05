"""
compare_contamination.py -- do the conclusions survive removal of the flagged documents?  (original run vs duplicate-cleaned run)

    TF-IDF   (uses the first five seeds of the main run, because the cleaned reruns used seeds 0-4):
    python compare_contamination.py --orig results_tfidf_a.csv results_tfidf_b.csv results_tfidf_c.csv --clean results_tfidf_x1.csv results_tfidf_x2.csv results_tfidf_x3.csv

    DistilBERT (only if you ran the optional cleaned DistilBERT jobs):
    python compare_contamination.py --orig results_distilbert.csv --clean results_distilbert_clean.csv

What it measures (descriptive, no p-values), using only seeds / conditions present in BOTH runs:
  * for each contrast (baseline level, L-C0, E-C0, LE-C0, SLE-C0, S-C0, S-R_S, E-R_E, SE-R_SE) and outcome (cross-dataset AUC, AUC gap):
        pooled effect per seed (equal-weight mean over the six ordered pairs), in the original and in the cleaned run
        Delta_s = cleaned effect - original effect      -> mean, SD, min/max, 95% t-interval over seeds, seeds agreeing in sign
  * per ordered pair: how many (pair, contrast) effects change sign, and the largest absolute change in a mean effect.
Caveat: the cleaned run uses a slightly different sampling universe, so the random draws differ; Delta therefore contains sampling noise as
well as any real contamination effect. Compare Delta with the seed-to-seed SD of the effect itself: a change well inside that SD is not detectable.
"""
import argparse
import numpy as np
import pandas as pd
from compare_models import load, per_seed_effect, summarise, fmt

CONTRASTS = [("L", "C0"), ("E", "C0"), ("LE", "C0"), ("SLE", "C0"), ("S", "C0"), ("S", "R_S"), ("E", "R_E"), ("SE", "R_SE")]
OUTCOMES = [("auc", "cross-dataset AUC"), ("gap_auc", "AUC gap")]
MODELS = ["tfidf_lr", "distilbert"]


def level(X, cond, outcome):
    return X[X.cond == cond][outcome].groupby(level="seed").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", nargs="+", required=True)
    ap.add_argument("--clean", nargs="+", required=True)
    a = ap.parse_args()
    O, dsO = load(a.orig, "orig", MODELS)
    C, dsC = load(a.clean, "clean", MODELS)
    if dsO != dsC:
        raise SystemExit(f"datasets differ: {dsO} vs {dsC}")
    if set(O.model.unique() if "model" in O else []) != set(C.model.unique() if "model" in C else []):
        raise SystemExit("the two runs use different models -- compare like with like")
    seeds = sorted(set(O.index.get_level_values("seed")) & set(C.index.get_level_values("seed")))
    conds = set(O.cond) & set(C.cond)
    if not seeds:
        raise SystemExit("no shared seeds")
    print(f"shared seeds: {seeds}; shared conditions: {sorted(conds)}")
    print("Delta_s = cleaned - original (pooled over six pairs, per seed). Intervals: t, df = n-1; (k/n) = seeds agreeing in sign.\n")
    O = O[O.index.get_level_values("seed").isin(seeds)]; C = C[C.index.get_level_values("seed").isin(seeds)]

    rows = []
    for outcome, label in OUTCOMES:
        if "C0" in conds:
            eo, ec = level(O, "C0", outcome).reindex(seeds), level(C, "C0", outcome).reindex(seeds)
            d = summarise((ec - eo).dropna())
            rows.append(dict(outcome=label, quantity="baseline C0 level", original=f"{eo.mean():.3f}", cleaned=f"{ec.mean():.3f}",
                             delta_cleaned_minus_original=fmt(d), seed_sd_of_effect=f"{eo.std(ddof=1):.3f}"))
        for ca, cb in CONTRASTS:
            if ca not in conds or cb not in conds:
                continue
            eo = per_seed_effect(O, ca, cb, outcome).reindex(seeds); ec = per_seed_effect(C, ca, cb, outcome).reindex(seeds)
            d = summarise((ec - eo).dropna())
            rows.append(dict(outcome=label, quantity=f"{ca} - {cb}", original=f"{eo.mean():+.3f}", cleaned=f"{ec.mean():+.3f}",
                             delta_cleaned_minus_original=fmt(d), seed_sd_of_effect=f"{eo.std(ddof=1):.3f}"))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for _, g in out.groupby("outcome", sort=False):
        print(f"=== POOLED, {g.outcome.iloc[0]} ===")
        print(g.drop(columns="outcome").to_string(index=False)); print()
    out.to_csv("table_contamination_pooled.csv", index=False)

    # per-pair: sign changes and largest change
    pairs = sorted(set(O.index.get_level_values("pair")))
    prow = []
    for ca, cb in CONTRASTS:
        if ca not in conds or cb not in conds:
            continue
        for pr in pairs:
            eo = (O[O.cond == ca]["auc"] - O[O.cond == cb]["auc"]).xs(pr, level="pair").reindex(seeds).mean()
            ec = (C[C.cond == ca]["auc"] - C[C.cond == cb]["auc"]).xs(pr, level="pair").reindex(seeds).mean()
            prow.append(dict(contrast=f"{ca} - {cb}", pair=pr, original=eo, cleaned=ec, change=ec - eo, sign_flip=bool(np.sign(eo) != np.sign(ec))))
    P = pd.DataFrame(prow); P.to_csv("table_contamination_by_pair.csv", index=False)
    if len(P):
        print(f"=== PER PAIR (cross-dataset AUC effects, mean over seeds) ===\n{len(P)} pair-contrast effects; sign flips: {int(P.sign_flip.sum())}; "
              f"largest absolute change: {P.change.abs().max():.3f} ({P.loc[P.change.abs().idxmax(), 'contrast']}, {P.loc[P.change.abs().idxmax(), 'pair']}); "
              f"median absolute change: {P.change.abs().median():.3f}")
        if P.sign_flip.any():
            print("effects whose sign flipped (check whether the original effect was near zero):")
            print(P[P.sign_flip].round(3).to_string(index=False))
    print("\nHow to read: a conclusion is stable if Delta is small compared with the seed-to-seed SD of the effect and its interval is centred near zero. "
          "'Stable' here means 'no change detectable at this sample size', not 'proven identical'.\nSaved: table_contamination_pooled.csv, table_contamination_by_pair.csv")


if __name__ == "__main__":
    main()
