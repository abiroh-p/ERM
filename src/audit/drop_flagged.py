"""
drop_flagged.py -- cross-dataset duplicate sensitivity check.

Finds documents that are exact / near-duplicates of a document in ANOTHER dataset inside analysis_set.csv, plus degenerate
documents (< 20 words once URLs/datelines/tags are stripped, e.g. URL-only posts), and writes their doc_ids to
flagged_doc_ids.txt.  From each duplicate pair the document in the LARGER dataset is flagged, so McIntire stays intact.

Then rerun the key conditions on the cleaned universe (cached entity spans stay valid):
    python -X utf8 exp_tfidf.py run --seeds 0 1 --conds C0 L E LE R_E R_SE --exclude flagged_doc_ids.txt --tag x1
    python -X utf8 exp_tfidf.py run --seeds 2 3 --conds C0 L E LE R_E R_SE --exclude flagged_doc_ids.txt --tag x2
    python -X utf8 exp_tfidf.py run --seeds 4   --conds C0 L E LE R_E R_SE --exclude flagged_doc_ids.txt --tag x3
    python -X utf8 analyze.py results_tfidf_x1.csv results_tfidf_x2.csv results_tfidf_x3.csv
Runtime: ~10 minutes here. Needs audit_01_05.py in the same folder.
"""
import numpy as np, pandas as pd
import audit_01_05 as A

df = pd.read_csv("analysis_set.csv")
df["title"] = ""                                    # title is already inside `text`
df["text"] = df["text"].fillna("").astype(str)
df = A.add_norm_and_hashes(df).reset_index(drop=True)
size = df["dataset"].value_counts().to_dict()
ds = df["dataset"].to_numpy()
flag, why = set(), {}

# degenerate after stripping source markers
deg = df["text_ns"].str.split().str.len().fillna(0) < A.MIN_WORDS
for i in np.where(deg.to_numpy())[0]:
    flag.add(i); why[i] = "degenerate"
print(f"degenerate (<{A.MIN_WORDS} words after stripping): {int(deg.sum()):,}")

def flag_pair(i, j, reason):
    a, b = (i, j) if size[ds[i]] >= size[ds[j]] else (j, i)       # flag the member from the larger dataset
    flag.add(a); why.setdefault(a, reason)

# exact + prefix
for col, reason in (("h_text_strip", "exact"), ("prefix", "prefix300")):
    g = df[df[col].notna()].groupby(col).indices
    n = 0
    for _, idx in g.items():
        if len(idx) > 1 and len(set(ds[idx])) > 1:
            for i in idx:
                for j in idx:
                    if i < j and ds[i] != ds[j]:
                        flag_pair(i, j, reason); n += 1
    print(f"cross-dataset {reason} pairs: {n:,}")

# near duplicates (primary tier k=5 J>=0.8 incl. truncated copies; sensitivity tier k=3 J>=0.5)
for thr, k, reason in ((A.JACC_PRIMARY, A.SHINGLE_K, "near_primary"), (A.JACC_SENS, A.SENS_SHINGLE_K, "near_sens")):
    pr = A.near_dup_pairs(df, thr, k)
    pr = pr[ds[pr["i"].to_numpy(int)] != ds[pr["j"].to_numpy(int)]]
    for i, j in zip(pr["i"].astype(int), pr["j"].astype(int)):
        flag_pair(i, j, reason)
    print(f"cross-dataset {reason} pairs: {len(pr):,}")

ids = df.loc[sorted(flag), "doc_id"]
open("flagged_doc_ids.txt", "w").write("\n".join(ids) + "\n")
tab = pd.DataFrame({"dataset": ds[sorted(flag)], "why": [why[i] for i in sorted(flag)]})
print("\nflagged documents by dataset and reason:")
print(tab.groupby(["dataset", "why"]).size().to_string())
print(f"\nTOTAL flagged: {len(ids):,}  (of {len(df):,})  -> flagged_doc_ids.txt")
