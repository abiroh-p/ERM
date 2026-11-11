"""
paper_tables.py -- LaTeX tables and one figure for the three-model matched-control results.

    python src/analysis/paper_tables.py --tfidf combined_tfidf.csv --distilbert combined_distilbert.csv \
        --roberta results_roberta.csv --out paper

Inputs are the raw per-cell result CSVs; give several files per model and they are concatenated (TF-IDF: results_tfidf_a,b,c,e +
results_tfidf_newconds; DistilBERT: results_distilbert + results_distilbert_newconds; RoBERTa: results_roberta).  Every number is computed here from those files, per seed, then summarised
as mean and 95% t-interval over the eight seeds (as in the paper), with the number of seeds that agree in sign with the mean.
Outputs (under --out):
    tables/tab_matched_pooled.tex   pooled cross-dataset AUC contrasts, three models
    tables/tab_model_diff.tex       paired seed-wise model differences D for the same contrasts
    tables/tab_matched_pairs.tex    per-pair S+L+E vs L+R_SE and L+E vs L+R_E, three models
    figs/fig_matched_pairs.pdf      the per-pair contrasts as a dot-and-interval plot
"""
import argparse, pathlib
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("tfidf", "TF-IDF"), ("distilbert", "DistilBERT"), ("roberta", "RoBERTa")]
CONTRASTS = [  # (label, condition a, condition b, family)
    ("S $-$ R$_S$", "S", "R_S", "p"), ("E $-$ R$_E$", "E", "R_E", "p"), ("S+E $-$ R$_{SE}$", "SE", "R_SE", "p"),
    ("L $-$ C0", "L", "C0", "p"), ("L+E $-$ L+R$_E$", "LE", "L_RE", "p"), ("S+L+E $-$ L+R$_{SE}$", "SLE", "L_RSE", "p"),
    ("L+R$_{SE}$ $-$ L", "L_RSE", "L", "x"), ("R$_{SE}$ $-$ C0", "R_SE", "C0", "x")]
SHORT = {"WELFake_residual": "residual", "McIntire": "McIntire", "ISOT": "ISOT"}
KEYS = ["seed", "train", "test"]


def load(paths):
    paths = [paths] if isinstance(paths, str) else paths
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d = d[d["kind"] == "cross"]
    if d.duplicated(KEYS + ["cond"]).any():
        raise SystemExit(f"{paths}: duplicate (seed, train, test, cond) cells; fix the input before building tables")
    return d


def cells(d, a, b):
    x = d[d.cond == a].set_index(KEYS)["auc"]; y = d[d.cond == b].set_index(KEYS)["auc"]
    return (x - y).dropna()


def summ(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(mean=m, lo=m - h, hi=m + h, k=int((np.sign(v) == np.sign(m)).sum()), n=n)


def seed_vals(d, a, b):
    return cells(d, a, b).groupby(level="seed").mean()


def f3(x):
    return f"{x:+.3f}".replace("-", "$-$")


def cell_tex(s, bold_if_clear=True):
    txt = f"{f3(s['mean'])} [{f3(s['lo'])}, {f3(s['hi'])}] ({s['k']}/{s['n']})"
    clear = (s["lo"] > 0 or s["hi"] < 0)
    return f"\\textbf{{{txt}}}" if (bold_if_clear and clear) else txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", required=True, nargs="+"); ap.add_argument("--distilbert", required=True, nargs="+"); ap.add_argument("--roberta", required=True, nargs="+")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    out = pathlib.Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figs").mkdir(parents=True, exist_ok=True)
    D = {"tfidf": load(a.tfidf), "distilbert": load(a.distilbert), "roberta": load(a.roberta)}
    for k, d in D.items():
        print(k, "seeds", sorted(d.seed.unique().tolist()), "cells per cond", d.groupby("cond").size().median())

    # ---- pooled contrasts, three models
    rows = []
    for lab, ca, cb, fam in CONTRASTS:
        cs = [cell_tex(summ(seed_vals(D[m], ca, cb))) for m, _ in MODELS]
        rows.append(f"{lab} & " + " & ".join(cs) + " \\\\")
        if lab.startswith("L $-$ C0"):
            rows.append("\\midrule")
    (out / "tables" / "tab_matched_pooled.tex").write_text(
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nContrast & TF-IDF & DistilBERT & RoBERTa \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- paired model differences D
    pairs = [("tfidf", "distilbert"), ("tfidf", "roberta"), ("distilbert", "roberta")]
    names = dict(MODELS)
    rows = []
    for lab, ca, cb, fam in CONTRASTS:
        cs = []
        for m1, m2 in pairs:
            v1, v2 = seed_vals(D[m1], ca, cb), seed_vals(D[m2], ca, cb)
            cs.append(cell_tex(summ((v1 - v2).dropna())))
        rows.append(f"{lab} & " + " & ".join(cs) + " \\\\")
        if lab.startswith("L $-$ C0"):
            rows.append("\\midrule")
    hdr = " & ".join(f"{names[m1]} $-$ {names[m2]}" for m1, m2 in pairs)
    (out / "tables" / "tab_model_diff.tex").write_text(
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nContrast & " + hdr + " \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- per-pair tables and figure
    pair_order = [("ISOT", "McIntire"), ("ISOT", "WELFake_residual"), ("McIntire", "ISOT"),
                  ("McIntire", "WELFake_residual"), ("WELFake_residual", "ISOT"), ("WELFake_residual", "McIntire")]
    plabel = lambda p: f"{SHORT[p[0]]}$\\rightarrow${SHORT[p[1]]}"
    blocks, fig_data = [], {}
    for title, ca, cb in (("S+L+E $-$ L+R$_{SE}$", "SLE", "L_RSE"), ("L+E $-$ L+R$_{E}$", "LE", "L_RE")):
        blocks.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for p in pair_order:
            cs = []
            for m, _ in MODELS:
                c = cells(D[m], ca, cb)
                v = c.xs(p[0], level="train").xs(p[1], level="test") if True else None
                s = summ(v.to_numpy()); fig_data[(ca, m, p)] = s; cs.append(cell_tex(s))
            blocks.append(f"{plabel(p)} & " + " & ".join(cs) + " \\\\")
        blocks.append("\\midrule")
    (out / "tables" / "tab_matched_pairs.tex").write_text(
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nTransfer & TF-IDF & DistilBERT & RoBERTa \\\\\n\\midrule\n"
        + "\n".join(blocks[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    colors = {"tfidf": "#444444", "distilbert": "#1f77b4", "roberta": "#d95f02"}
    marks = {"tfidf": "o", "distilbert": "s", "roberta": "^"}
    for ax, (ca, title) in zip(axes, (("SLE", "S+L+E vs. L + random deletion"), ("LE", "L+E vs. L + random deletion"))):
        for j, (m, name) in enumerate(MODELS):
            for i, p in enumerate(pair_order):
                s = fig_data[(ca, m, p)]; y = i + (j - 1) * 0.22
                ax.plot([s["lo"], s["hi"]], [y, y], color=colors[m], lw=1.3)
                ax.plot(s["mean"], y, marks[m], color=colors[m], ms=4.5, label=name if i == 0 else None)
        ax.axvline(0, color="black", lw=0.8); ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("change in cross-dataset AUC (mean, 95% interval over seeds)", fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(range(len(pair_order)))
    axes[0].set_yticklabels([f"{SHORT[p[0]]}→{SHORT[p[1]]}" for p in pair_order], fontsize=7); axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "figs" / "fig_matched_pairs.pdf"); plt.close(fig)
    print("wrote tables and figure under", out)


if __name__ == "__main__":
    main()
