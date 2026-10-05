"""
analyze_v2.py

Publication-oriented statistical analysis for the cross-dataset fake-news study.

Usage (Windows):
    python -X utf8 analyze_v2.py results_tfidf_a.csv results_tfidf_b.csv results_tfidf_c.csv results_tfidf_d1.csv results_tfidf_d2.csv results_tfidf_d3.csv

    python -X utf8 analyze_v2.py results_distilbert.csv

Core statistical rule
---------------------
The independent experimental replication unit is the RANDOM SEED.
The six ordered transfer directions are repeated/blocked observations within a seed,
so they are averaged within seed before inference.

For pooled transfer results:
    1. compute the metric for each of the 6 ordered pairs within each seed;
    2. average those 6 pairs to obtain one seed-level value;
    3. estimate the mean across the 8 seeds;
    4. report a two-sided 95% t-interval (df = n_seed - 1).

The script is deliberately strict for the primary factorial design:
    - expected seeds default to 0,1,2,3,4,5,6,7;
    - exactly 3 datasets are required;
    - all 6 ordered transfer directions are required;
    - every seed x pair must contain all 8 factorial conditions;
    - every seed x condition x training-dataset cell must contain exactly one in-domain row;
    - every seed x condition x train x test cell must contain exactly one cross-domain row.

Multiple comparisons
--------------------
For each outcome separately, the seven factorial effects (S, L, E, S:L, S:E, L:E, S:L:E)
are treated as one family and Holm-adjusted paired t-test p-values are provided.
The three real-vs-random contrasts are treated as a separate family for each outcome.
Exact sign-flip p-values are also reported as a small-sample robustness check; with eight
non-zero seeds the smallest attainable two-sided exact sign-flip p-value is 0.0078125.

Important interpretation rule
-----------------------------
These intervals and tests describe seed-level variation under this exact experimental
protocol and the three evaluated datasets. They are not population-level intervals for
"fake-news datasets in general". Pair-specific tables remain descriptive because transfer
directions share the same seed-controlled training/sampling process.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_path(value: str | pathlib.Path | None) -> pathlib.Path:
    if value is None:
        return REPO_ROOT
    p = pathlib.Path(value)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


FACT = ["C0", "S", "L", "E", "SL", "SE", "LE", "SLE"]
CTRL = ["R_S", "R_E", "R_SE"]
EXPL = ["T", "SLET", "R_T"]
SPLIT = ["S_W", "S_B", "R_W", "R_B"]
ALL_PRIMARY_FACTORS = FACT[1:]
PRIMARY_SEEDS = list(range(8))
REQUIRED_COLS = {"seed", "cond", "train", "test", "kind", "f1", "auc"}
PRIMARY_OUTCOMES = ("auc", "gap_auc")
ALL_OUTCOMES = ("auc", "f1", "gap_auc", "gap_f1")

# Conventional +/-1 factorial coding.
CODE = {
    c: (-1 + 2 * ("S" in c), -1 + 2 * ("L" in c), -1 + 2 * ("E" in c))
    for c in FACT
}
CODE["C0"] = (-1, -1, -1)


def holm_adjust(p_values: pd.Series | list[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    idx = np.flatnonzero(finite)
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, pos in enumerate(order):
        val = min(1.0, (m - rank) * p[pos])
        running = max(running, val)
        adjusted[rank] = running
    # adjusted currently follows sorted positions; place back in original positions.
    for k, pos in enumerate(order):
        out[pos] = adjusted[k]
    return out


def exact_signflip_p(values: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value for a one-sample mean around zero."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    v = v[v != 0]
    n = len(v)
    if n == 0:
        return 1.0
    observed = abs(v.mean())
    # Enumerate all 2^n signs. n=5 here, so this is tiny and exact.
    null = np.array([np.mean(v * np.asarray(signs)) for signs in product([-1.0, 1.0], repeat=n)])
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def summarise(values: pd.Series | np.ndarray) -> dict:
    """Mean, SD, 95% t CI, raw paired t-test p, exact sign-flip p, sign agreement."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "min": np.nan,
            "max": np.nan,
            "same_sign": 0,
            "n": 0,
            "p_t_raw": np.nan,
            "p_signflip": np.nan,
        }
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    if n > 1:
        se = sd / np.sqrt(n)
        crit = stats.t.ppf(0.975, n - 1)
        lo, hi = m - crit * se, m + crit * se
        p_t = float(stats.ttest_1samp(v, 0.0, alternative="two-sided").pvalue)
    else:
        lo = hi = np.nan
        p_t = np.nan
    same = int(np.sum(np.sign(v) == np.sign(m))) if m != 0 else int(np.sum(v == 0))
    return {
        "mean": m,
        "sd": sd,
        "lo": float(lo),
        "hi": float(hi),
        "min": float(v.min()),
        "max": float(v.max()),
        "same_sign": same,
        "n": n,
        "p_t_raw": p_t,
        "p_signflip": exact_signflip_p(v),
    }


def fmt(s: dict, digits: int = 3) -> str:
    if s["n"] == 0:
        return "NA"
    if np.isnan(s["lo"]):
        return f"{s['mean']:+.{digits}f} [NA, NA] ({s['same_sign']}/{s['n']})"
    return (
        f"{s['mean']:+.{digits}f} [{s['lo']:+.{digits}f}, {s['hi']:+.{digits}f}] "
        f"({s['same_sign']}/{s['n']})"
    )


def require(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit("VALIDATION ERROR: " + message)


def load_and_validate(paths: list[str], expected_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[int]]:
    frames = []
    for p in paths:
        q = repo_path(p)
        require(q.exists(), f"missing input file: {q}")
        try:
            frames.append(pd.read_csv(q))
        except Exception as exc:
            raise SystemExit(f"VALIDATION ERROR: could not read {q}: {exc}")

    r = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLS - set(r.columns)
    require(not missing, f"missing required columns: {sorted(missing)}")

    try:
        r["seed"] = pd.to_numeric(r["seed"], errors="raise").astype(int)
    except Exception as exc:
        raise SystemExit(f"VALIDATION ERROR: seed column is not integer-like: {exc}")

    for c in ["cond", "train", "kind"]:
        r[c] = r[c].astype(str)

    require(set(r["kind"].unique()) <= {"in", "cross"}, f"kind contains unexpected values: {sorted(set(r.kind) - {'in', 'cross'})}")
    require(r["auc"].notna().all(), "auc contains missing values")
    require(r["f1"].notna().all(), "f1 contains missing values")

    # Exact cell uniqueness: never silently average duplicate run rows.
    dup = r.duplicated(subset=["seed", "cond", "train", "test", "kind"], keep=False)
    require(not dup.any(),
            "duplicate run cells found for (seed, cond, train, test, kind). "
            f"Duplicated rows: {int(dup.sum())}\n{r.loc[dup, ['seed','cond','train','test','kind']].head(20).to_string(index=False)}")

    datasets = sorted(set(r["train"].dropna()) | set(r["test"].dropna()))
    require(len(datasets) == 3, f"expected exactly 3 datasets, found {datasets}")

    observed_seeds = sorted(r.seed.unique().tolist())
    require(observed_seeds == expected_seeds,
            f"expected seeds {expected_seeds}, found {observed_seeds}. "
            "Do not silently intersect seeds for the primary paper analysis.")

    # Exactly one in-domain row per seed x condition x training dataset.
    inn = r[r.kind == "in"].copy()
    in_counts = inn.groupby(["seed", "cond", "train"], sort=False).size()
    bad_in = in_counts[in_counts != 1]
    require(len(bad_in) == 0,
            "every seed x condition x train cell must have exactly one in-domain row; "
            f"bad cells:\n{bad_in.head(20).to_string()}")

    cross = r[r.kind == "cross"].copy()
    cross_counts = cross.groupby(["seed", "cond", "train", "test"], sort=False).size()
    bad_cross = cross_counts[cross_counts != 1]
    require(len(bad_cross) == 0,
            "every seed x condition x train x test cell must have exactly one cross-domain row; "
            f"bad cells:\n{bad_cross.head(20).to_string()}")

    # Exact set of six ordered transfer directions for the primary design.
    pairs_expected = sorted(f"{s}->{t}" for s in datasets for t in datasets if s != t)
    cross["pair"] = cross["train"] + "->" + cross["test"]
    pairs_observed = sorted(cross.pair.unique().tolist())
    require(pairs_observed == pairs_expected,
            f"expected all 6 ordered transfer directions {pairs_expected}, found {pairs_observed}")

    # Primary factorial completeness: every seed x pair has all 8 factorial cells.
    primary_cross = cross[cross.cond.isin(FACT)].copy()
    seen = primary_cross.groupby(["seed", "pair"])["cond"].agg(lambda s: set(s))
    bad = [(seed, pair, sorted(set(FACT) - conds)) for (seed, pair), conds in seen.items() if set(FACT) != conds]
    # Also check seed/pair combinations absent completely.
    all_expected = {(s, pair) for s in expected_seeds for pair in pairs_expected}
    missing_groups = sorted(all_expected - set(seen.index.tolist()))
    for s, pair in missing_groups:
        bad.append((s, pair, FACT.copy()))
    require(not bad,
            "primary factorial design is incomplete. Every seed x transfer pair must contain all 8 cells. "
            f"Examples: {bad[:12]}")

    # Each primary condition must have the same set of seeds and all six pairs.
    for c in FACT:
        sub = primary_cross[primary_cross.cond == c]
        got = set((int(s), p) for s, p in zip(sub.seed, sub.pair))
        require(got == all_expected,
                f"condition {c} does not contain exactly the expected {len(expected_seeds)} seeds x 6 pairs")

    # Compute gaps from the single in-domain row.
    inn_small = inn[["seed", "cond", "train", "f1", "auc"]].rename(columns={"f1": "f1_in", "auc": "auc_in"})
    x = cross.merge(inn_small, on=["seed", "cond", "train"], how="left", validate="many_to_one")
    require(x["auc_in"].notna().all() and x["f1_in"].notna().all(),
            "some cross-domain rows have no matching in-domain result for the same seed x condition x train")
    x["gap_f1"] = x["f1_in"] - x["f1"]
    x["gap_auc"] = x["auc_in"] - x["auc"]

    # Helpful audit frame for condition-level in-domain means.
    return r, inn, x, datasets, pairs_expected, observed_seeds


def seed_pair_mean(frame: pd.DataFrame, value_col: str, require_pairs: list[str] | None = None) -> pd.Series:
    """Average transfer pairs within seed; result indexed by seed."""
    g = frame.groupby(["seed", "pair"], sort=True)[value_col].mean()
    if require_pairs is not None:
        expected = {(int(seed), pair) for seed in sorted(frame.seed.unique()) for pair in require_pairs}
        actual = set(g.index.tolist())
        missing = sorted(expected - actual)
        if missing:
            raise SystemExit(f"VALIDATION ERROR: missing seed x pair cells: {missing[:20]}")
    return g.groupby(level="seed").mean()


def seed_train_mean(inn: pd.DataFrame, value_col: str) -> pd.Series:
    """Average the 3 unique training-dataset in-domain values within seed."""
    return inn.groupby("seed", sort=True)[value_col].mean()


def contrast_seed_values(X: pd.DataFrame, a: str, b: str, outcome: str, pairs: list[str]) -> pd.Series:
    """Paired condition difference, averaged over six pairs inside each seed."""
    A = X[X.cond == a].set_index(["seed", "pair"])[outcome]
    B = X[X.cond == b].set_index(["seed", "pair"])[outcome]
    d = A - B
    expected = pd.MultiIndex.from_product([sorted(X.seed.unique()), pairs], names=["seed", "pair"])
    d = d.reindex(expected)
    require(d.notna().all(), f"contrast {a} - {b} has missing paired cells for outcome {outcome}")
    return d.groupby(level="seed").mean()


def factorial_seed_values(X: pd.DataFrame, outcome: str, pairs: list[str]) -> pd.DataFrame:
    """Return one row per seed with the factorial effects averaged equally over six pairs."""
    rows = []
    coding = np.array([CODE[c] for c in FACT], dtype=float)
    terms = {
        "S": coding[:, 0],
        "L": coding[:, 1],
        "E": coding[:, 2],
        "SL": coding[:, 0] * coding[:, 1],
        "SE": coding[:, 0] * coding[:, 2],
        "LE": coding[:, 1] * coding[:, 2],
        "SLE": coding[:, 0] * coding[:, 1] * coding[:, 2],
    }

    for seed in sorted(X.seed.unique()):
        seed_rows = []
        for pair in pairs:
            g = X[(X.seed == seed) & (X.pair == pair) & X.cond.isin(FACT)].set_index("cond")
            y = g.reindex(FACT)[outcome].to_numpy(dtype=float)
            require(np.isfinite(y).all(), f"missing/non-finite factorial outcome: seed={seed}, pair={pair}, outcome={outcome}")
            # For +/-1 coding, 2*mean(y*z) is the main effect / interaction contrast.
            seed_rows.append({term: float(2.0 * np.mean(y * z)) for term, z in terms.items()})
        pair_df = pd.DataFrame(seed_rows)
        vals = pair_df.mean(axis=0)
        rows.append(dict(seed=seed, **vals.to_dict()))
    return pd.DataFrame(rows).set_index("seed")


def add_holm(df: pd.DataFrame, group_cols: list[str], p_col: str = "p_t_raw") -> pd.DataFrame:
    out = df.copy()
    out["p_holm"] = np.nan
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, p_col].to_numpy(dtype=float))
    return out


def summary_row(values: pd.Series, label: str = "") -> dict:
    s = summarise(values)
    return {
        "label": label,
        "mean": s["mean"],
        "sd": s["sd"],
        "ci_lo": s["lo"],
        "ci_hi": s["hi"],
        "min": s["min"],
        "max": s["max"],
        "same_sign": s["same_sign"],
        "n_seeds": s["n"],
        "p_t_raw": s["p_t_raw"],
        "p_signflip": s["p_signflip"],
        "formatted": fmt(s),
    }


def main(paths: list[str], expected_seeds: list[int], output_dir: str | pathlib.Path = "results/final") -> None:
    r, inn, x, datasets, pairs, seeds = load_and_validate(paths, expected_seeds)
    conditions = sorted(r.cond.unique().tolist())
    order = [c for c in FACT + CTRL + EXPL + SPLIT if c in set(conditions)]
    X = x.set_index(["seed", "pair"], drop=False)
    out_dir = repo_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== DESIGN / VALIDATION AUDIT ===")
    print(f"files: {len(paths)}")
    print(f"rows: {len(r):,}")
    print(f"datasets: {datasets}")
    print(f"seeds: {seeds}")
    print(f"transfer pairs: {pairs}")
    print(f"conditions: {conditions}")
    print(f"primary factorial completeness: PASS ({len(seeds)} seeds x {len(pairs)} pairs x {len(FACT)} conditions)")
    print("duplicate cell check: PASS")
    print("in-domain cell check: PASS")
    print("cross-domain cell check: PASS")
    print("statistical unit: SEED")
    print("pooled transfer estimand: equal-weight mean over 6 ordered pairs within each seed")
    print("CI: two-sided 95% t interval over seed means (df = n_seeds - 1)")
    print()

    # ------------------------------------------------------------------
    # 1. Per-condition summaries
    # ------------------------------------------------------------------
    print("=== 1. PER CONDITION ===")
    cond_rows = []
    for c in order:
        gx = X[X.cond == c]
        # Pooled transfer outcomes: average six pairs inside each seed, then across seeds.
        cross_auc = seed_pair_mean(gx.reset_index(drop=True), "auc", pairs)
        cross_f1 = seed_pair_mean(gx.reset_index(drop=True), "f1", pairs)
        gap_auc = seed_pair_mean(gx.reset_index(drop=True), "gap_auc", pairs)
        gap_f1 = seed_pair_mean(gx.reset_index(drop=True), "gap_f1", pairs)
        # In-domain outcome: average the 3 unique train-dataset values inside each seed.
        gi = inn[inn.cond == c]
        in_auc = seed_train_mean(gi, "auc")
        in_f1 = seed_train_mean(gi, "f1")
        sa = summarise(cross_auc); sf = summarise(cross_f1); sga = summarise(gap_auc); sgf = summarise(gap_f1)
        sia = summarise(in_auc); sif = summarise(in_f1)
        cond_rows.append({
            "cond": c,
            "in_auc_mean": sia["mean"],
            "cross_auc_mean": sa["mean"], "cross_auc_ci_lo": sa["lo"], "cross_auc_ci_hi": sa["hi"],
            "gap_auc_mean": sga["mean"], "gap_auc_ci_lo": sga["lo"], "gap_auc_ci_hi": sga["hi"],
            "in_f1_mean": sif["mean"],
            "cross_f1_mean": sf["mean"], "cross_f1_ci_lo": sf["lo"], "cross_f1_ci_hi": sf["hi"],
            "gap_f1_mean": sgf["mean"], "gap_f1_ci_lo": sgf["lo"], "gap_f1_ci_hi": sgf["hi"],
            "n_seeds": sa["n"],
        })
    T1 = pd.DataFrame(cond_rows).set_index("cond")
    print(T1.round(4).to_string())
    T1.to_csv(out_dir / "table_conditions.csv")

    # Human-readable conditions table for paper drafting.
    pretty = []
    for _, row in T1.reset_index().iterrows():
        pretty.append({
            "cond": row["cond"],
            "in_auc": f"{row['in_auc_mean']:.3f}",
            "cross_auc": f"{row['cross_auc_mean']:.3f} [{row['cross_auc_ci_lo']:.3f}, {row['cross_auc_ci_hi']:.3f}]",
            "gap_auc": f"{row['gap_auc_mean']:.3f} [{row['gap_auc_ci_lo']:.3f}, {row['gap_auc_ci_hi']:.3f}]",
            "in_f1": f"{row['in_f1_mean']:.3f}",
            "cross_f1": f"{row['cross_f1_mean']:.3f} [{row['cross_f1_ci_lo']:.3f}, {row['cross_f1_ci_hi']:.3f}]",
            "gap_f1": f"{row['gap_f1_mean']:.3f} [{row['gap_f1_ci_lo']:.3f}, {row['gap_f1_ci_hi']:.3f}]",
            "n_seeds": int(row["n_seeds"]),
        })
    pd.DataFrame(pretty).to_csv(out_dir / "table_conditions_pretty.csv", index=False)

    # ------------------------------------------------------------------
    # 2. C0 vs SLE per pair
    # ------------------------------------------------------------------
    print("\n=== 2. PER ORDERED PAIR: C0 vs SLE ===")
    pair_rows = []
    for pair in pairs:
        gp = X[X.pair == pair]
        for cond in ["C0", "SLE"]:
            row = gp[gp.cond == cond]
            pair_rows.append({
                "pair": pair,
                "cond": cond,
                "cross_auc_mean_over_seeds": row["auc"].mean(),
                "gap_auc_mean_over_seeds": row["gap_auc"].mean(),
                "cross_f1_mean_over_seeds": row["f1"].mean(),
                "gap_f1_mean_over_seeds": row["gap_f1"].mean(),
            })
    T2 = pd.DataFrame(pair_rows)
    piv2 = T2.pivot(index="pair", columns="cond")
    print(T2.pivot(index="pair", columns="cond").round(4).to_string())
    T2.to_csv(out_dir / "table_pairs.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Gap share removed
    # ------------------------------------------------------------------
    print("\n=== 3. SHARE OF C0 GAP REMOVED ===")
    share_rows = []
    share_conds = [c for c in FACT[1:] + ["T", "SLET"] if c in conditions]
    for outcome in ("gap_auc", "gap_f1"):
        base = seed_pair_mean(X[X.cond == "C0"].reset_index(drop=True), outcome, pairs)
        if np.any(np.abs(base.to_numpy()) < 1e-12):
            raise SystemExit(f"VALIDATION ERROR: near-zero C0 gap encountered for {outcome}; gap-share ratio would be unstable")
        for c in share_conds:
            gc = seed_pair_mean(X[X.cond == c].reset_index(drop=True), outcome, pairs)
            ratio = 1.0 - gc / base
            s = summarise(ratio)
            share_rows.append({
                "outcome": outcome,
                "cond": c,
                "share_removed_mean": s["mean"],
                "ci_lo": s["lo"],
                "ci_hi": s["hi"],
                "p_t_raw": s["p_t_raw"],
                "p_signflip": s["p_signflip"],
                "same_sign": s["same_sign"],
                "n_seeds": s["n"],
                "formatted": fmt(s),
            })
    T3 = pd.DataFrame(share_rows)
    print(T3[["outcome","cond","share_removed_mean","ci_lo","ci_hi","same_sign","n_seeds"]].round(4).to_string(index=False))
    T3.to_csv(out_dir / "table_gap_share.csv", index=False)
    print("Note: gap-share estimates are descriptive and are not additive across interventions because the design contains interactions.")

    # ------------------------------------------------------------------
    # 4. Factorial effects, with Holm adjustment
    # ------------------------------------------------------------------
    print("\n=== 4. FACTORIAL EFFECTS (7 TERMS; HOLM WITHIN EACH OUTCOME FAMILY) ===")
    eff_rows = []
    seed_eff_frames = {}
    for outcome in ALL_OUTCOMES:
        U = factorial_seed_values(X, outcome, pairs)
        seed_eff_frames[outcome] = U
        for term in ALL_PRIMARY_FACTORS:
            s = summarise(U[term])
            eff_rows.append({
                "outcome": outcome,
                "term": term,
                "effect_mean": s["mean"],
                "sd": s["sd"],
                "ci_lo": s["lo"],
                "ci_hi": s["hi"],
                "p_t_raw": s["p_t_raw"],
                "p_signflip": s["p_signflip"],
                "same_sign": s["same_sign"],
                "n_seeds": s["n"],
                "formatted": fmt(s),
            })
    T4 = pd.DataFrame(eff_rows)
    T4 = add_holm(T4, ["outcome"], p_col="p_t_raw")
    print(T4[["outcome","term","effect_mean","ci_lo","ci_hi","p_t_raw","p_holm","p_signflip","same_sign","n_seeds"]].round(5).to_string(index=False))
    T4.to_csv(out_dir / "table_factorial_effects.csv", index=False)
    # Keep old filename too for compatibility with paper drafts.
    T4[["outcome","term","formatted","p_t_raw","p_holm","p_signflip"]].to_csv(out_dir / "table_effects.csv", index=False)

    seed_eff = []
    for outcome, U in seed_eff_frames.items():
        z = U.reset_index().melt(id_vars="seed", var_name="term", value_name="effect")
        z.insert(0, "outcome", outcome)
        seed_eff.append(z)
    pd.concat(seed_eff, ignore_index=True).to_csv(out_dir / "table_factorial_per_seed.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Specific real vs matched-random effects
    # ------------------------------------------------------------------
    print("\n=== 5. SPECIFIC EFFECT: REAL MASKING - MATCHED RANDOM MASKING ===")
    specific_rows = []
    for real, ctrl in (("S", "R_S"), ("E", "R_E"), ("SE", "R_SE")):
        if real not in conditions or ctrl not in conditions:
            continue
        for outcome in ALL_OUTCOMES:
            vals = contrast_seed_values(X, real, ctrl, outcome, pairs)
            s = summarise(vals)
            specific_rows.append({
                "contrast": f"{real} - {ctrl}",
                "masking": real,
                "control": ctrl,
                "outcome": outcome,
                "effect_mean": s["mean"],
                "sd": s["sd"],
                "ci_lo": s["lo"],
                "ci_hi": s["hi"],
                "p_t_raw": s["p_t_raw"],
                "p_signflip": s["p_signflip"],
                "same_sign": s["same_sign"],
                "n_seeds": s["n"],
                "formatted": fmt(s),
            })
    T5 = pd.DataFrame(specific_rows)
    if len(T5):
        T5 = add_holm(T5, ["outcome"], p_col="p_t_raw")
        print(T5[["contrast","outcome","effect_mean","ci_lo","ci_hi","p_t_raw","p_holm","p_signflip","same_sign","n_seeds"]].round(5).to_string(index=False))
        T5.to_csv(out_dir / "table_specific_effects.csv", index=False)

    # Also preserve real-vs-C0 changes as a separate table.
    rc_rows = []
    for real in ["S", "L", "E", "LE", "SE", "SL", "SLE"]:
        if real not in conditions:
            continue
        for outcome in ALL_OUTCOMES:
            vals = contrast_seed_values(X, real, "C0", outcome, pairs)
            s = summarise(vals)
            rc_rows.append({
                "contrast": f"{real} - C0",
                "outcome": outcome,
                "effect_mean": s["mean"],
                "ci_lo": s["lo"], "ci_hi": s["hi"],
                "p_t_raw": s["p_t_raw"], "p_signflip": s["p_signflip"],
                "same_sign": s["same_sign"], "n_seeds": s["n"],
                "formatted": fmt(s),
            })
    pd.DataFrame(rc_rows).to_csv(out_dir / "table_vs_C0_contrasts.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Per-pair gap change
    # ------------------------------------------------------------------
    print("\n=== 6. AUC-GAP CHANGE VS C0 BY ORDERED PAIR ===")
    gap_rows = []
    gap_conds = [c for c in FACT[1:] + ["T", "SLET"] if c in conditions]
    for pair in pairs:
        gp = X[X.pair == pair]
        base = gp[gp.cond == "C0"]["gap_auc"]
        for c in gap_conds:
            d = gp[gp.cond == c]["gap_auc"].to_numpy(dtype=float) - base.to_numpy(dtype=float)
            s = summarise(d)
            gap_rows.append({
                "pair": pair,
                "cond": c,
                "gap_change_mean": s["mean"],
                "ci_lo": s["lo"], "ci_hi": s["hi"],
                "same_sign": s["same_sign"], "n_seeds": s["n"],
                "formatted": fmt(s),
            })
    T6 = pd.DataFrame(gap_rows)
    print(T6[["pair","cond","gap_change_mean","ci_lo","ci_hi","same_sign","n_seeds"]].round(4).to_string(index=False))
    T6.to_csv(out_dir / "table_gap_change_by_pair.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Per-pair cross-AUC change vs C0
    # ------------------------------------------------------------------
    print("\n=== 7. CROSS-AUC CHANGE VS C0 BY ORDERED PAIR ===")
    pair_auc_rows = []
    auc_conds = [c for c in ["S", "L", "E", "LE", "SLE", "R_E"] if c in conditions]
    for pair in pairs:
        for c in auc_conds:
            vals = contrast_seed_values(X[X.pair == pair].reset_index(drop=True), c, "C0", "auc", [pair])
            s = summarise(vals)
            pair_auc_rows.append({
                "pair": pair,
                "cond": c,
                "auc_change_mean": s["mean"],
                "ci_lo": s["lo"], "ci_hi": s["hi"],
                "same_sign": s["same_sign"], "n_seeds": s["n"],
                "formatted": fmt(s),
            })
    T7 = pd.DataFrame(pair_auc_rows)
    print(T7[["pair","cond","auc_change_mean","ci_lo","ci_hi","same_sign","n_seeds"]].round(4).to_string(index=False))
    T7.to_csv(out_dir / "table_pairwise_auc_change.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Exploratory temporal extension
    # ------------------------------------------------------------------
    if {"T", "R_T", "SLET"}.intersection(conditions):
        print("\n=== 8. EXPLORATORY: TEMPORAL EXTENSION ===")
        rows8 = []
        for a_, b_, lab in (("T", "C0", "T - C0"), ("T", "R_T", "T - R_T"), ("SLET", "SLE", "SLET - SLE")):
            if a_ in conditions and b_ in conditions:
                for outcome in ALL_OUTCOMES:
                    vals = contrast_seed_values(X, a_, b_, outcome, pairs)
                    s = summarise(vals)
                    rows8.append({"contrast": lab, "outcome": outcome, **summary_row(vals)})
        T8 = pd.DataFrame(rows8)
        print(T8[["contrast","outcome","mean","ci_lo","ci_hi","same_sign","n_seeds"]].round(5).to_string(index=False))
        T8.to_csv(out_dir / "table_temporal_extension.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Exploratory source split
    # ------------------------------------------------------------------
    if {"S_W", "S_B"}.intersection(conditions):
        print("\n=== 9. EXPLORATORY: SOURCE-MARKER SPLIT ===")
        rows9 = []
        for a_, ctrl, lab in (("S_W", "R_W", "S_W"), ("S_B", "R_B", "S_B")):
            if a_ not in conditions:
                continue
            for outcome in ALL_OUTCOMES:
                v_c0 = contrast_seed_values(X, a_, "C0", outcome, pairs)
                row = {"part": lab, "outcome": outcome}
                row.update({f"vs_C0_{k}": v for k, v in summary_row(v_c0).items() if k != "label"})
                if ctrl in conditions:
                    v_r = contrast_seed_values(X, a_, ctrl, outcome, pairs)
                    s = summary_row(v_r)
                    row.update({f"vs_random_{k}": v for k, v in s.items() if k != "label"})
                rows9.append(row)
        T9 = pd.DataFrame(rows9)
        print(T9[["part","outcome","vs_C0_mean","vs_C0_ci_lo","vs_C0_ci_hi"]].round(5).to_string(index=False))
        T9.to_csv(out_dir / "table_source_split.csv", index=False)

    # ------------------------------------------------------------------
    # 10. Compact primary-results table for paper drafting
    # ------------------------------------------------------------------
    primary_eff = T4[T4.outcome == "auc"].copy()
    primary_random = T5[T5.outcome == "auc"].copy() if len(T5) else pd.DataFrame()
    compact = []
    for _, row in primary_eff.iterrows():
        compact.append({
            "family": "factorial_cross_auc",
            "contrast": row.term,
            "effect": row.effect_mean,
            "ci_lo": row.ci_lo,
            "ci_hi": row.ci_hi,
            "p_t_raw": row.p_t_raw,
            "p_holm": row.p_holm,
            "p_signflip": row.p_signflip,
            "same_sign": row.same_sign,
            "n_seeds": row.n_seeds,
        })
    if len(primary_random):
        for _, row in primary_random.iterrows():
            compact.append({
                "family": "random_control_cross_auc",
                "contrast": row.contrast,
                "effect": row.effect_mean,
                "ci_lo": row.ci_lo,
                "ci_hi": row.ci_hi,
                "p_t_raw": row.p_t_raw,
                "p_holm": row.p_holm,
                "p_signflip": row.p_signflip,
                "same_sign": row.same_sign,
                "n_seeds": row.n_seeds,
            })
    T10 = pd.DataFrame(compact)
    T10.to_csv(out_dir / "table_primary_results.csv", index=False)
    print("\n=== 10. PRIMARY CROSS-AUC EFFECTS (compact) ===")
    print(T10.round(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Reproducibility metadata
    # ------------------------------------------------------------------
    metadata = {
        "inputs": [str(pathlib.Path(p).resolve()) for p in paths],
        "rows": int(len(r)),
        "datasets": datasets,
        "seeds": seeds,
        "transfer_pairs": pairs,
        "conditions_present": conditions,
        "factorial_conditions": FACT,
        "factorial_terms": ALL_PRIMARY_FACTORS,
        "statistical_unit": "seed",
        "pooled_transfer_estimand": "equal-weight mean over six ordered transfer directions within seed",
        "ci": "two-sided 95% t interval over seed means",
        "primary_outcome": "cross-domain AUC",
        "factorial_multiple_comparison": "Holm adjustment across 7 factorial terms within each outcome family",
        "random_control_multiple_comparison": "Holm adjustment across 3 real-vs-random contrasts within each outcome family",
        "exact_signflip_note": "with eight non-zero seeds, the smallest attainable two-sided exact sign-flip p-value is 0.0078125",
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Figure: AUC gap by condition (no semantic claims from the graph itself)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = []
        los = []
        his = []
        labels = []
        for c in order:
            g = X[X.cond == c]
            s = summarise(seed_pair_mean(g.reset_index(drop=True), "gap_auc", pairs))
            labels.append(c); vals.append(s["mean"]); los.append(s["mean"] - s["lo"]); his.append(s["hi"] - s["mean"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, vals, yerr=[los, his], capsize=3)
        ax.set_ylabel("AUC gap (AUC_in - AUC_cross)")
        ax.set_title(f"Cross-dataset AUC gap by condition ({len(seeds)} seed-level replicates)")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_gap_by_condition.png", dpi=250)
        plt.close(fig)
        print(f"figure: {out_dir / 'fig_gap_by_condition.png'}")
    except Exception as exc:
        print("figure skipped:", exc)

    print("\nDONE.")
    print("Key files:")
    for f in [
        "table_conditions.csv",
        "table_conditions_pretty.csv",
        "table_factorial_effects.csv",
        "table_effects.csv",
        "table_factorial_per_seed.csv",
        "table_specific_effects.csv",
        "table_vs_C0_contrasts.csv",
        "table_gap_share.csv",
        "table_gap_change_by_pair.csv",
        "table_pairwise_auc_change.csv",
        "table_primary_results.csv",
        "analysis_metadata.json",
    ]:
        print("  ", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Strict seed-level statistical analysis for the cross-dataset study.")
    ap.add_argument("paths", nargs="+", help="result CSV files")
    ap.add_argument("--expected-seeds", nargs="+", type=int, default=PRIMARY_SEEDS,
                    help="expected independent seeds for the paper analysis (default: 0 1 2 3 4 5 6 7)")
    ap.add_argument("--output-dir", default="results/final",
                    help="directory for generated tables and figure outputs (default: results/final)")
    args = ap.parse_args()
    main(args.paths, sorted(args.expected_seeds), args.output_dir)
