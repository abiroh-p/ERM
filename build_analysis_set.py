"""
build_analysis_set.py -- run ONCE after the audit is finished and the label mapping is fixed.

Re-uses YOUR audit_01_05.py (load_all() with the fixed labels, step03_within, near_dup_pairs) so the cleaned
datasets are identical to the audited ones, then trims WELFake exactly as in the audit dry run
(primary tier: exact text OR 300-char prefix OR near-duplicate (5-shingle Jaccard >= 0.8) with ISOT/McIntire).

Output: analysis_set.csv  with columns
    dataset | doc_id | text | is_fake | cluster_id | n_words
    dataset in {ISOT, McIntire, WELFake_residual};  text = title + newline + body
ISOT 'subject' and 'date' never enter (they are not in the standard schema).

Expected sizes (from your audit): ISOT 38,030 | McIntire 6,005 | WELFake_residual 17,161.
Runtime: roughly 15-25 minutes (it repeats the within-dataset clustering).
"""
import sys
import numpy as np
import pandas as pd
import audit_01_05 as A

EXPECTED = {"ISOT": 38030, "McIntire": 6005, "WELFake": 17161}


def main(out_path="analysis_set.csv"):
    frames = A.load_all()
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

    bad = wel["h_text_strip"].isin(set(ref["h_text_strip"].dropna())) \
        | wel["prefix"].isin(set(ref["prefix"].dropna()))
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
    print(f"WELFake_residual: {len(resid):,}  (audit dry run said {EXPECTED['WELFake']:,})")
    if abs(len(resid) - EXPECTED["WELFake"]) > 150:
        print("!! size differs a lot from the audit dry run -- investigate before continuing")

    parts = [cleaned["ISOT"], cleaned["McIntire"], resid]
    out = pd.concat(parts, ignore_index=True)
    out["text"] = out["title"].fillna("").astype(str) + "\n" + out["text"].fillna("").astype(str)
    out["n_words"] = out["text"].str.split().str.len()
    out = out[["dataset", "doc_id", "text", "is_fake", "cluster_id", "n_words"]]
    out.to_csv(out_path, index=False)

    print("\nfinal analysis set:")
    print(out.groupby(["dataset", "is_fake"]).agg(n=("doc_id", "size"), words_mean=("n_words", "mean"),
                                                    words_median=("n_words", "median")).round(1))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
