"""
analyze.py -- turn the results CSV(s) into the paper's tables.

    python analyze.py results_tfidf_a.csv results_tfidf_b.csv results_tfidf_c.csv

Definitions (fake = positive class; every test set is class-balanced, so AUC is the headline cross-domain metric):
    gap(seed, cond, pair) = metric_in_domain(train dataset) - metric_cross(train -> test dataset)      (f1 and auc versions)
    Factorial effects use the 8 cells (C0,S,L,E,SL,SE,LE,SLE) coded -1/+1. An effect is the average change in the outcome
    when that factor (or interaction) goes from -1 to +1.  A NEGATIVE effect on a GAP means the factor shrinks the gap.
    'Specific effect' of masking X = outcome(X) - outcome(R_X): what real masking does beyond deleting the same number of
    random words.
Uncertainty: the 6 ordered pairs within a seed share training sets and the seed controls splits/sampling, so pairs are NOT
independent replicates. Every CI is a t-interval over SEED means (pairs averaged inside each seed first). These intervals
describe resampling noise for THESE THREE DATASETS under THIS sampling protocol; they are not intervals for 'fake-news
datasets in general', and pooled averages hide large differences between pairs (see section 9). '(k/n)' = number of seeds
whose sign agrees with the mean.
"""
import sys, pathlib
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
FACT = ["C0", "S", "L", "E", "SL", "SE", "LE", "SLE"]
CTRL = ["R_S", "R_E", "R_SE"]
EXPL = ["T", "SLET", "R_T"]
SPLIT = ["S_W", "S_B", "R_W", "R_B"]
CODE = {c: (-1 + 2 * ("S" in c), -1 + 2 * ("L" in c), -1 + 2 * ("E" in c)) for c in FACT}
CODE["C0"] = (-1, -1, -1)


def cboot(s):
    """s: Series indexed by (seed, pair).  Pairs are averaged inside each seed, then a t-interval (df = n_seeds - 1) is taken over
    the seed means.  Returns (mean, lo, hi, k, n_seeds).  (Name kept for backward compatibility; it is no longer a bootstrap.)"""
    from scipy import stats
    v = s.groupby(level="seed").mean().dropna().to_numpy()
    n = len(v)
    if n < 2:
        return (v.mean() if n else np.nan, np.nan, np.nan, 0, n)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean(), v.mean() - h, v.mean() + h, int((np.sign(v) == np.sign(v.mean())).sum()), n


def fmt(t, d=3):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}] ({t[3]}/{t[4]})"


def main(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    r = pd.concat([pd.read_csv(q) for q in paths], ignore_index=True)
    dup = r.duplicated(subset=["seed", "cond", "train", "test"], keep=False)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicated (seed, cond, train, test) rows across the result files -- "
                         f"the same run was probably passed twice or two runs overlap:\n"
                         f"{r[dup].groupby(['seed', 'cond']).size().head(20)}")
    print(f"read {len(paths)} file(s): {len(r):,} rows, seeds {sorted(int(s) for s in r.seed.unique())}, "
          f"conditions {sorted(r.cond.unique())}")
    out_base = pathlib.Path(paths[0]).name if len(paths) == 1 else "results_tfidf_merged.csv"      # always written into the current folder

    inn = r[r.kind == "in"][["seed", "cond", "train", "f1", "auc"]].rename(columns={"f1": "f1_in", "auc": "auc_in"})
    x = r[r.kind == "cross"].merge(inn, on=["seed", "cond", "train"])
    x["gap_f1"] = x.f1_in - x.f1
    x["gap_auc"] = x.auc_in - x.auc
    x["pair"] = x.train + "->" + x.test
    x.to_csv(out_base.replace(".csv", "_pairs.csv"), index=False)
    X = x.set_index(["seed", "pair"])
    order = [c for c in FACT + CTRL + EXPL + SPLIT if c in set(x.cond)]
    nseeds = x.seed.nunique()

    print(f"\n=== 1. Per condition: mean over pairs; CI = t-interval over {nseeds} seed means; (k/n) = seeds agreeing in sign ===")
    rows = []
    for c in order:
        g = X[X.cond == c]
        b = lambda col: cboot(g[col])
        rows.append(dict(cond=c, in_auc=round(g.auc_in.mean(), 3), cross_auc=f"{b('auc')[0]:.3f} [{b('auc')[1]:.3f}, {b('auc')[2]:.3f}]",
                         gap_auc=f"{b('gap_auc')[0]:.3f} [{b('gap_auc')[1]:.3f}, {b('gap_auc')[2]:.3f}]",
                         in_f1=round(g.f1_in.mean(), 3), cross_f1=f"{b('f1')[0]:.3f} [{b('f1')[1]:.3f}, {b('f1')[2]:.3f}]",
                         gap_f1=f"{b('gap_f1')[0]:.3f} [{b('gap_f1')[1]:.3f}, {b('gap_f1')[2]:.3f}]"))
    t1 = pd.DataFrame(rows).set_index("cond")
    print(t1.to_string()); t1.to_csv("table_conditions.csv")

    print("\n=== 2. Per ordered pair, C0 vs all-masked (SLE): cross AUC and AUC gap (mean over seeds) ===")
    pv = x[x.cond.isin(["C0", "SLE"])].pivot_table(index="pair", columns="cond", values=["auc", "gap_auc", "f1", "gap_f1"], aggfunc="mean").round(3)
    print(pv.to_string()); pv.to_csv("table_pairs.csv")

    print("\n=== 3. Share of the C0 gap removed by each intervention (1 - gap_cond / gap_C0), per seed; t-interval over seeds ===")
    sh = []
    for outcome in ("gap_auc", "gap_f1"):
        g0 = X[X.cond == "C0"][outcome].groupby(level="seed").mean()
        for c in [c for c in FACT[1:] + ["T", "SLET"] if c in set(x.cond)]:
            gc = X[X.cond == c][outcome].groupby(level="seed").mean()
            ratio = (1 - gc / g0).rename("v"); ratio.index = pd.MultiIndex.from_arrays([ratio.index, ["all"] * len(ratio)], names=["seed", "pair"])
            sh.append(dict(outcome=outcome, cond=c, share_removed=fmt(cboot(ratio), 3)))
    print(pd.DataFrame(sh).to_string(index=False)); pd.DataFrame(sh).to_csv("table_gap_share.csv", index=False)
    print("(Shares are not additive across interventions -- see the interaction terms below.)")

    print("\n=== 4. Factorial effects (8 cells C0,S,L,E,SL,SE,LE,SLE) ===")
    eff = []
    for outcome in ("auc", "f1", "gap_auc", "gap_f1"):
        recs = []
        for (seed, pair), g in X[X.cond.isin(FACT)].groupby(level=["seed", "pair"]):
            if g.cond.nunique() < 8:
                continue
            y = g.set_index("cond")[outcome].reindex(FACT).to_numpy()
            sg = np.array([CODE[c] for c in FACT])
            terms = {"S": sg[:, 0], "L": sg[:, 1], "E": sg[:, 2], "S:L": sg[:, 0] * sg[:, 1],
                     "S:E": sg[:, 0] * sg[:, 2], "L:E": sg[:, 1] * sg[:, 2], "S:L:E": sg[:, 0] * sg[:, 1] * sg[:, 2]}
            recs.append(dict(seed=seed, pair=pair, **{k: 2 * np.mean(y * v) for k, v in terms.items()}))
        if not recs:
            print(f"(skipped: need all 8 factorial cells for outcome '{outcome}' -- merge all result files)")
            continue
        U = pd.DataFrame(recs).set_index(["seed", "pair"])
        for k in U.columns:
            eff.append(dict(outcome=outcome, term=k, effect=fmt(cboot(U[k]), 3)))
    E = pd.DataFrame(eff)
    if len(E):
        print(E.to_string(index=False)); E.to_csv("table_effects.csv", index=False)

    print("\n=== 5. Specific effect of real masking vs random-word masking of the SAME size (outcome(X) - outcome(R_X)) ===")
    sp = []
    for real, ctrl in (("S", "R_S"), ("E", "R_E"), ("SE", "R_SE")):
        if real not in set(x.cond) or ctrl not in set(x.cond):
            continue
        a = X[X.cond == real]; b_ = X[X.cond == ctrl]; c0 = X[X.cond == "C0"]
        for outcome in ("auc", "f1", "gap_auc", "gap_f1"):
            sp.append(dict(masking=real, outcome=outcome, vs_random=fmt(cboot(a[outcome] - b_[outcome]), 3),
                           vs_C0=fmt(cboot(a[outcome] - c0[outcome]), 3)))
    S5 = pd.DataFrame(sp); print(S5.to_string(index=False)); S5.to_csv("table_specific_effects.csv", index=False)

    print("\n=== 6. Gap change vs C0 per ordered pair (negative AUC-gap change = gap shrinks), mean over seeds ===")
    c0 = x[x.cond == "C0"].groupby("pair").gap_auc.mean()
    g6 = x[x.cond.isin([c for c in FACT[1:] + ["T", "SLET"] if c in set(x.cond)])].groupby(["pair", "cond"]).gap_auc.mean().unstack()
    red = g6.sub(c0, axis=0).round(3); print(red.to_string()); red.to_csv("table_gap_change_by_pair.csv")

    print("\n=== 9. PER ORDERED PAIR (the primary way to report): change in cross-dataset AUC vs C0, t-interval over seeds ===")
    rows9 = []
    for pair, gp in X.groupby(level="pair"):
        base = gp[gp.cond == "C0"].auc
        row = {"pair": pair, "C0_auc": f"{base.mean():.3f}"}
        for c in [c for c in ["S", "L", "E", "LE", "SLE", "R_E"] if c in set(x.cond)]:
            d = gp[gp.cond == c].auc - base
            row[c] = fmt(cboot(d), 3)
        rows9.append(row)
    T9 = pd.DataFrame(rows9).set_index("pair"); print(T9.to_string()); T9.to_csv("table_pairwise_auc_change.csv")
    print("(positive = cross-dataset AUC improved. Pairs differ in sign -- do not summarise them with one pooled number.)")

    if {"T", "R_T", "SLET"} & set(x.cond):
        print("\n=== 7. EXPLORATORY: temporal expressions (T). Added after inspecting residual cues -> report as exploratory ===")
        rows7 = []
        for a_, b_, lab in (("T", "C0", "T alone vs C0"), ("T", "R_T", "T vs random words of the same size"),
                            ("SLET", "SLE", "T on top of S+L+E")):
            if a_ in set(x.cond) and b_ in set(x.cond):
                for outcome in ("auc", "f1", "gap_auc", "gap_f1"):
                    rows7.append(dict(contrast=lab, outcome=outcome, change=fmt(cboot(X[X.cond == a_][outcome] - X[X.cond == b_][outcome]), 3)))
        T7 = pd.DataFrame(rows7); print(T7.to_string(index=False)); T7.to_csv("table_temporal_extension.csv", index=False)

    if {"S_W", "S_B"} & set(x.cond):
        print("\n=== 8. EXPLORATORY: source markers split into W (wire datelines/tags + outlet names) and B (web boilerplate, credits, embeds, URLs) ===")
        rows8 = []
        for a_, ctrl, lab in (("S_W", "R_W", "W: wire/outlet markers"), ("S_B", "R_B", "B: web boilerplate & credits")):
            if a_ in set(x.cond):
                for outcome in ("auc", "gap_auc", "f1", "gap_f1"):
                    rows8.append(dict(part=lab, outcome=outcome, vs_C0=fmt(cboot(X[X.cond == a_][outcome] - X[X.cond == "C0"][outcome]), 3),
                                      vs_random=fmt(cboot(X[X.cond == a_][outcome] - X[X.cond == ctrl][outcome]), 3) if ctrl in set(x.cond) else "n/a"))
        T8 = pd.DataFrame(rows8); print(T8.to_string(index=False)); T8.to_csv("table_source_split.csv", index=False)
        cols8 = [c for c in ["S_W", "S_B"] if c in set(x.cond)]
        g8 = x[x.cond.isin(cols8)].groupby(["pair", "cond"]).gap_auc.mean().unstack().sub(c0, axis=0).round(3)
        print("\nAUC-gap change vs C0 per ordered pair (positive = gap grew):"); print(g8.to_string())

    print("\nNotes: report in-domain AND cross-domain performance, not only the gap (a gap can shrink because in-domain fell).")
    print("With balanced classes, 'predict fake for everything' gives F1 = 0.667 but AUC = 0.5 -- AUC is the headline cross-domain metric.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m = [cboot(X[X.cond == c].gap_auc) for c in order]
        cols = [("#888" if c == "C0" else "#4a7" if c in FACT else "#c84" if c in CTRL else "#a6c" if c in SPLIT else "#58b") for c in order]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.bar(order, [a[0] for a in m], yerr=[[a[0] - a[1] for a in m], [a[2] - a[0] for a in m]], capsize=3, color=cols)
        ax.set_ylabel("AUC gap (AUC_in - AUC_cross)"); ax.set_title(f"Cross-dataset AUC gap by condition (mean over pairs; 95% t-interval over {nseeds} seeds)")
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig("fig_gap_by_condition.png", dpi=200)
        print("figure: fig_gap_by_condition.png (green = factorial, orange = random-word controls, blue = exploratory T)")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else "results_tfidf.csv")
