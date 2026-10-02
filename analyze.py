"""
analyze.py -- turn results_tfidf.csv (or the DistilBERT results with the same columns) into the paper's tables.

    python analyze.py results_tfidf.csv            (or several files: python analyze.py results_tfidf_a.csv results_tfidf_b.csv)

Definitions (F1 = F1 of the 'fake' class; test sets are class-balanced in every condition):
    gap(seed, cond, pair)  = F1_in_domain(train dataset) - F1_cross(train -> test dataset)
    Factorial effects use the 8 cells (C0,S,L,E,SL,SE,LE,SLE) coded -1/+1. An effect is the average change in the
    outcome when that factor (or interaction) goes from -1 to +1.  Negative effect on GAP = the factor shrinks the gap.
    'Specific effect' of masking X = outcome(X) - outcome(R_X): what the real masking does beyond deleting the same
    number of random words.
CIs: bootstrap (5000 resamples) over (seed, ordered-pair) units, 95%.
"""
import sys, itertools
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
FACT = ["C0", "S", "L", "E", "SL", "SE", "LE", "SLE"]
CODE = {c: (-1 + 2 * ("S" in c), -1 + 2 * ("L" in c), -1 + 2 * ("E" in c)) for c in FACT}
CODE["C0"] = (-1, -1, -1)


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(t, d=3):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def main(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    r = pd.concat([pd.read_csv(q) for q in paths], ignore_index=True).drop_duplicates(subset=["seed", "cond", "train", "test"])
    print(f"read {len(paths)} file(s): {len(r):,} rows, seeds {sorted(r.seed.unique())}")
    path = paths[0] if len(paths) == 1 else "results_tfidf_merged.csv"
    inn = r[r.kind == "in"][["seed", "cond", "train", "f1", "auc"]].rename(columns={"f1": "f1_in", "auc": "auc_in"})
    x = r[r.kind == "cross"].merge(inn, on=["seed", "cond", "train"])
    x["gap_f1"] = x.f1_in - x.f1
    x["gap_auc"] = x.auc_in - x.auc
    x["pair"] = x.train + "->" + x.test
    x.to_csv(path.replace(".csv", "_pairs.csv"), index=False)

    print("\n=== 1. Per condition (mean over seeds and pairs, with 95% bootstrap CI) ===")
    rows = []
    for c, g in x.groupby("cond"):
        rows.append(dict(cond=c, in_f1=g.f1_in.mean(), cross_f1=fmt(boot_ci(g.f1), 3).split(" ")[0],
                         cross_f1_ci=fmt(boot_ci(g.f1), 3).split(" ", 1)[1], gap_f1=fmt(boot_ci(g.gap_f1), 3),
                         cross_auc=fmt(boot_ci(g.auc), 3).split(" ")[0], gap_auc=fmt(boot_ci(g.gap_auc), 3)))
    t1 = pd.DataFrame(rows).set_index("cond").loc[[c for c in FACT + ["R_S", "R_E", "R_SE"] if c in set(x.cond)]]
    print(t1.to_string())
    t1.to_csv("table_conditions.csv")

    print("\n=== 2. Per ordered pair: cross F1 and gap, C0 vs all-masked (SLE) ===")
    pv = x[x.cond.isin(["C0", "SLE"])].pivot_table(index="pair", columns="cond", values=["f1", "gap_f1"], aggfunc="mean").round(3)
    print(pv.to_string())
    pv.to_csv("table_pairs.csv")

    print("\n=== 3. Factorial effects (cells C0,S,L,E,SL,SE,LE,SLE) ===")
    eff_rows = []
    for outcome in ("f1", "auc", "gap_f1", "gap_auc"):
        units = []
        for (seed, pair), g in x[x.cond.isin(FACT)].groupby(["seed", "pair"]):
            if g.cond.nunique() < 8:
                continue
            y = g.set_index("cond")[outcome].reindex(FACT).to_numpy()
            sg = np.array([CODE[c] for c in FACT])
            terms = {"S": sg[:, 0], "L": sg[:, 1], "E": sg[:, 2], "S:L": sg[:, 0] * sg[:, 1],
                     "S:E": sg[:, 0] * sg[:, 2], "L:E": sg[:, 1] * sg[:, 2], "S:L:E": sg[:, 0] * sg[:, 1] * sg[:, 2]}
            units.append({k: 2 * np.mean(y * v) for k, v in terms.items()})
        U = pd.DataFrame(units)
        for k in U.columns:
            eff_rows.append(dict(outcome=outcome, term=k, effect=fmt(boot_ci(U[k]), 3), mean=U[k].mean()))
    E = pd.DataFrame(eff_rows)
    print(E.drop(columns="mean").to_string(index=False))
    E.to_csv("table_effects.csv", index=False)

    print("\n=== 4. Specific effect of real masking vs random-word masking of the SAME size (outcome(X) - outcome(R_X)) ===")
    sp = []
    for real, ctrl in (("S", "R_S"), ("E", "R_E"), ("SE", "R_SE")):
        a = x[x.cond == real].set_index(["seed", "pair"]); b = x[x.cond == ctrl].set_index(["seed", "pair"])
        j = a.join(b, lsuffix="_r", rsuffix="_c", how="inner")
        for outcome in ("f1", "auc", "gap_f1", "gap_auc"):
            sp.append(dict(masking=real, outcome=outcome, vs_random=fmt(boot_ci(j[f"{outcome}_r"] - j[f"{outcome}_c"]), 3)))
        c0 = x[x.cond == "C0"].set_index(["seed", "pair"])
        j0 = a.join(c0, lsuffix="_r", rsuffix="_0", how="inner")
        for outcome in ("f1", "auc", "gap_f1", "gap_auc"):
            sp.append(dict(masking=real, outcome=outcome + " vs C0", vs_random=fmt(boot_ci(j0[f"{outcome}_r"] - j0[f"{outcome}_0"]), 3)))
    print(pd.DataFrame(sp).to_string(index=False))
    pd.DataFrame(sp).to_csv("table_specific_effects.csv", index=False)

    print("\n=== 5. Gap reduction relative to C0 (positive = gap shrinks), per ordered pair ===")
    c0 = x[x.cond == "C0"].groupby("pair").gap_f1.mean()
    g5 = x[x.cond.isin(FACT[1:])].groupby(["pair", "cond"]).gap_f1.mean().unstack()
    red = (-(g5.sub(c0, axis=0))).round(3)
    print(red.to_string())
    red.to_csv("table_gap_reduction.csv")
    print("\n(Cross-check: report in-domain AND cross-domain performance, not only the gap -- a gap can shrink because in-domain fell.)")
    print("(With balanced classes, predicting 'fake' for everything gives F1 = 0.667 but AUC = 0.5. Treat AUC as the headline cross-domain metric; F1 near 0.65 can be chance level.)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = [c for c in FACT + ["R_S", "R_E", "R_SE"] if c in set(x.cond)]
        m = [boot_ci(x[x.cond == c].gap_f1) for c in order]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(order, [a[0] for a in m], yerr=[[a[0] - a[1] for a in m], [a[2] - a[0] for a in m]], capsize=3, color=["#888"] + ["#4a7"] * 7 + ["#c84"] * 3)
        ax.set_ylabel("cross-dataset gap (F1_in - F1_cross)"); ax.set_title("Gap by condition (mean, 95% CI)")
        plt.tight_layout(); plt.savefig("fig_gap_by_condition.png", dpi=200)
        print("\nfigure: fig_gap_by_condition.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else "results_tfidf.csv")
