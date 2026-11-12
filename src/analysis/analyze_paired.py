"""Validate paired result grids and report descriptive seed-level contrasts.

Use only NEW paired results, never a mixture of archived and validation cells.
The six-contrast Holm family is retained even when the focused run omits two
contrasts. No claim of prospective confirmation is made: the original results
were known when this validation protocol was written.
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
import paired_sampling as P
import exp_transformer as T
import confirmatory as C

FOCUSED = ["C0", "E", "R_E", "L", "LE", "L_RE", "SLE", "L_RSE"]


def validate(d, seeds, conditions, datasets):
    required = set(T.KEY + ["kind", "auc", "sampling_protocol", "train_sample_sha256",
                            "test_sample_sha256", "manifest_sha256"])
    if required - set(d):
        raise ValueError(f"missing columns: {sorted(required - set(d))}")
    if d.empty or not d.sampling_protocol.eq(P.PROTOCOL).all():
        raise ValueError("only paired-documents-v1 results are accepted")
    if d.model.nunique() != 1 or d.duplicated(T.KEY).any():
        raise ValueError("expected one model with unique result cells per file")
    expected = set(itertools.product(seeds, conditions, datasets, datasets))
    actual = set(d[["seed", "cond", "train", "test"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"result grid incomplete/unexpected: {len(expected - actual)} missing, {len(actual - expected)} extra")
    if d.auc.isna().any() or not np.isfinite(d.auc).all() or not d.auc.between(0, 1).all():
        raise ValueError("AUC values must be finite and within [0,1]")
    if not d.kind.eq(np.where(d.train == d.test, "in", "cross")).all():
        raise ValueError("invalid in-domain/cross-domain row labels")
    for column in ["train_sample_sha256", "test_sample_sha256", "manifest_sha256"]:
        if not d[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"invalid/missing {column}")
    if (d.groupby("seed").manifest_sha256.nunique() != 1).any():
        raise ValueError("a seed contains different sampling manifests")
    d = d.copy()
    d["length_matched"] = d.cond.map(lambda c: bool(T.X.CONDITIONS[c][1]))
    for _, g in d.groupby(["seed", "length_matched", "train", "test"]):
        if g.train_sample_sha256.nunique() != 1 or g.test_sample_sha256.nunique() != 1:
            raise ValueError("conditions use different document samples")
    for _, g in d[d.kind == "cross"].groupby(["seed", "length_matched", "test"]):
        if g.test_sample_sha256.nunique() != 1:
            raise ValueError("cross-domain test samples differ between training sources")
    return d


def summaries(d):
    rows = []
    for metric in ["auc", "gap"]:
        cross = d[d.kind == "cross"] if metric == "auc" else C.cross_gap(d)
        for family, contrasts in [("planned_descriptive", C.PRIMARY), ("exploratory", C.EXPLORATORY)]:
            for name, ca, cb in contrasts:
                if {ca, cb} <= set(d.cond):
                    v = C.seed_values(cross, ca, cb, metric)
                    rows.append(dict(model=d.model.iloc[0], metric=metric, family=family,
                                     contrast=name, **C.summarise(v.to_numpy())))
    res = pd.DataFrame(rows)
    res["p_holm_six"] = np.nan
    for metric in ["auc", "gap"]:
        selected = res[(res.metric == metric) & (res.family == "planned_descriptive")]
        ps = {row.contrast: row.p_signflip for row in selected.itertuples()}
        names = [item[0] for item in C.PRIMARY]
        adjusted = dict(zip(names, C.holm([ps.get(n, 1.0) for n in names])))
        res.loc[selected.index, "p_holm_six"] = [adjusted[n] for n in selected.contrast]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, nargs="+")
    ap.add_argument("--design", choices=["focused", "full"], default="focused")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--datasets", nargs="+", default=T.LEGACY_ORDER)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    conditions = FOCUSED if a.design == "focused" else T.PRIMARY_CONDS
    frames, results = [], []
    for path in a.results:
        d = validate(pd.read_csv(path), a.seeds, conditions, a.datasets)
        frames.append(d)
        results.append(summaries(d))
    if len({d.model.iloc[0] for d in frames}) != len(frames):
        raise ValueError("give one complete CSV per model")
    if len(frames) > 1:
        reference = frames[0][["seed", "manifest_sha256"]].drop_duplicates().sort_values("seed").reset_index(drop=True)
        for d in frames[1:]:
            other = d[["seed", "manifest_sha256"]].drop_duplicates().sort_values("seed").reset_index(drop=True)
            if not reference.equals(other):
                raise ValueError("models use different sampling manifests")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(results, ignore_index=True)
    table.to_csv(out / "paired_contrasts.csv", index=False)
    metadata = dict(protocol=P.PROTOCOL, design=a.design, seeds=a.seeds, conditions=conditions,
                    datasets=a.datasets, inputs={str(pathlib.Path(p).resolve()): P.file_hash(p) for p in a.results},
                    inference="Descriptive seed-level variability on these fixed corpora. Not equivalence or causal identification.",
                    correction="Holm over the six planned contrasts per model and outcome; omitted contrasts use p=1.")
    P.atomic_json(out / "analysis_metadata.json", metadata)
    print(table[table.metric == "auc"].round(5).to_string(index=False))
    print(f"Validated complete grids and document pairing; wrote {out}")


if __name__ == "__main__":
    main()
