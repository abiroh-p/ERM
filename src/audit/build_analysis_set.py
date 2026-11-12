"""
build_analysis_set.py -- run ONCE after the audit is finished and the label mapping is fixed.

Re-uses YOUR audit_01_05.py (load_all() with the fixed labels, step03_within, near_dup_pairs) so the cleaned
datasets are identical to the audited ones, then trims WELFake using the recorded builder policy:
artifact-stripped body hash OR 300-char prefix OR primary near-duplicate with ISOT/McIntire.
Near matching uses full/head views and Jaccard >= 0.8 OR qualifying containment >= 0.9.
The audit dry run additionally used lightly-normalized and title+body exact hashes.
Those policies differ for stripped-empty documents; do not treat the dry-run count as a target.

Output: analysis_set.csv  with columns
    dataset | doc_id | text | is_fake | cluster_id | n_words
    dataset in {ISOT, McIntire, WELFake_residual};  text = title + newline + body
ISOT 'subject' and 'date' never enter (they are not in the standard schema).

Historical dry-run sizes: ISOT 38,030 | McIntire 6,005 | WELFake_residual 17,161.
These are provenance references, not construction targets. Audit the actual experiment CSV separately.
Runtime: roughly 15-25 minutes (it repeats the within-dataset clustering).
"""
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import audit_01_05 as A

HISTORICAL_DRY_RUN = {"ISOT": 38030, "McIntire": 6005, "WELFake": 17161}
EXACT_OVERLAP_COLUMNS = ("h_text_strip",)


def exact_prefix_flags(ref, wel):
    """Recorded builder policy. Missing hashes never match other missing hashes.

    Deliberately preserves the experiment construction, including stripped-empty
    bodies that the broader historical audit's light hashes would flag.
    """
    bad = wel["prefix"].isin(set(ref["prefix"].dropna()))
    for column in EXACT_OVERLAP_COLUMNS:
        bad = bad | wel[column].isin(set(ref[column].dropna()))
    return bad


def main(out_path="analysis_set.csv", base_dir="."):
    if pathlib.Path(out_path).exists():
        raise FileExistsError("Refusing to overwrite an existing analysis file")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames = A.load_all(base_dir)
    cleaned = {}
    for df in frames:
        c, rep = A.step03_within(df)
        if "text_ns" not in c.columns or "prefix" not in c.columns:
            c = A.add_norm_and_hashes(c)
        cleaned[str(c["dataset"].iloc[0])] = c.reset_index(drop=True)
    for need in ("ISOT", "McIntire", "WELFake"):
        if need not in cleaned:
            sys.exit(f"dataset '{need}' not returned by load_all(); got {list(cleaned)}")

    ref = pd.concat([cleaned["ISOT"], cleaned["McIntire"]], ignore_index=True)
    wel = cleaned["WELFake"]

    bad = exact_prefix_flags(ref, wel)
    print(f"exact/prefix overlap flags on WELFake: {int(bad.sum()):,}")

    allx = pd.concat([ref, wel], ignore_index=True)
    is_wel = np.r_[np.zeros(len(ref), bool), np.ones(len(wel), bool)]
    pairs = A.near_dup_pairs(allx, A.JACC_PRIMARY, A.SHINGLE_K)
    pi, pj = pairs["i"].to_numpy(int), pairs["j"].to_numpy(int)
    cross = is_wel[pi] != is_wel[pj]
    wel_idx = np.where(is_wel[pi[cross]], pi[cross], pj[cross]) - len(ref)
    near = np.zeros(len(wel), bool); near[wel_idx] = True
    print(f"near-duplicate flags on WELFake: {int(near.sum()):,}")

    keep = ~(bad.to_numpy() | near)
    resid = wel[keep].copy()
    resid["dataset"] = "WELFake_residual"
    print(f"WELFake_residual: {len(resid):,}  (historical audit dry run said {HISTORICAL_DRY_RUN['WELFake']:,})")
    if len(resid) != HISTORICAL_DRY_RUN["WELFake"]:
        print("Count differs from the historical dry run; compare policies and input hashes before interpreting it.")

    parts = [cleaned["ISOT"], cleaned["McIntire"], resid]
    out = pd.concat(parts, ignore_index=True)
    out["text"] = out["title"].fillna("").astype(str) + "\n" + out["text"].fillna("").astype(str)
    out["n_words"] = out["text"].str.split().str.len()
    out = out[["dataset", "doc_id", "text", "is_fake", "cluster_id", "n_words"]]
    out.to_csv(out_path, index=False, mode="x")

    print("\nfinal analysis set:")
    print(out.groupby(["dataset", "is_fake"]).agg(n=("doc_id", "size"), words_mean=("n_words", "mean"),
                                                    words_median=("n_words", "median")).round(1))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_path", nargs="?", default="analysis_set.csv")
    parser.add_argument("--base-dir", default=".", help="Directory containing the four original CSVs")
    args = parser.parse_args()
    main(args.out_path, args.base_dir)
