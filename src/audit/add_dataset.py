"""
add_dataset.py -- add ONE new corpus to analysis_set.csv under the same audit as the original three.

    python src/audit/add_dataset.py --input new.csv --name NewsX --text-col text --title-col title \
        --label-col label --fake-values fake 1 --date-col published \
        --analysis-csv data/processed/analysis_set.csv --out data/processed/analysis_set_plus.csv --check-sample 40

What it does (same code as the paper's audit, imported from audit_01_05.py):
  1. standardises the corpus (title + body, is_fake in {0,1}); you MUST state which raw label values mean "fake"
  2. within-dataset cleaning: docs < 20 words, label-conflicting duplicates and exact duplicates removed; near-duplicate
     clusters (5-shingle Jaccard >= 0.8) recorded as cluster_id so splits stay group-aware
  3. overlap audit against every document already in analysis_set.csv: exact (normalised) text, 300-char prefix and
     near-duplicate/containment matches.  Flagged documents are removed (as for the WELFake residual) and written to
     data/audit/flagged_<name>_ids.txt.  The overlap RATE is the headline independence statistic: if most of the new corpus
     matches the old ones it is not an independent corpus, and the script says so.
  4. optional sensitivity tiers (--sensitivity): other shingle/threshold settings, reported side by side
  5. writes a BLIND hand-check sample per class (--check-sample N; the dataset label goes to a separate key file) so the LABEL DIRECTION is verified on known content,
     which is exactly what went wrong in the WELFake documentation
  6. applies the paper's pre-specified size rule (every class >= 3,000 docs, minority share >= 40%) and reports pass/fail
The output keeps an optional `date` column (used only by the temporal experiment, never as a model input).
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import audit_01_05 as A                                                          # noqa: E402

MIN_PER_CLASS, MIN_MINORITY = 3000, 0.40                                         # the paper's pre-specified rule


def to_is_fake(raw, fake_values):
    fv = {str(v).strip().lower() for v in fake_values}
    return raw.astype(str).str.strip().str.lower().isin(fv).astype(int)


def combined(df):
    """Same representation as analysis_set.csv: text = title + newline + body (title column left empty for matching)."""
    out = df.copy()
    out["text"] = out["title"].fillna("").astype(str) + "\n" + out["text"].fillna("").astype(str)
    out["title"] = ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True, help="dataset name, e.g. NewsX (new names sort AFTER the original corpora)")
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--title-col")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--fake-values", nargs="+", required=True, help="raw label values that mean FAKE, e.g. fake 1")
    ap.add_argument("--date-col")
    ap.add_argument("--analysis-csv", default="data/processed/analysis_set.csv")
    ap.add_argument("--out", default="data/processed/analysis_set_plus.csv")
    ap.add_argument("--audit-dir", default="data/audit")
    ap.add_argument("--sensitivity", action="store_true", help="also report overlap under extra shingle/threshold tiers (slow)")
    ap.add_argument("--check-sample", type=int, default=40, help="docs per class written for manual label verification")
    ap.add_argument("--encoding", default="utf-8")
    a = ap.parse_args()

    raw = pd.read_csv(a.input, encoding=a.encoding, low_memory=False)
    for c in [a.text_col, a.label_col] + ([a.title_col] if a.title_col else []) + ([a.date_col] if a.date_col else []):
        if c not in raw.columns:
            sys.exit(f"column '{c}' not in {a.input}: {list(raw.columns)}")
    is_fake = to_is_fake(raw[a.label_col], a.fake_values)
    print(f"raw rows {len(raw):,}; label values {raw[a.label_col].value_counts().to_dict()} -> fake share {is_fake.mean():.3f}")
    new = A.standardize(a.name, raw[a.title_col] if a.title_col else None, raw[a.text_col], is_fake)
    date_of = dict(zip(new["doc_id"], raw[a.date_col])) if a.date_col else {}

    # 1) within-dataset cleaning + near-duplicate clusters
    clean, rep = A.step03_within(new)
    print("within-dataset cleaning:", rep)

    # 2) overlap audit against the existing analysis set
    ref = pd.read_csv(a.analysis_csv)
    ref["text"] = ref["text"].fillna("").astype(str)
    ref_f = A.add_norm_and_hashes(pd.DataFrame({"dataset": ref["dataset"], "doc_id": ref["doc_id"], "title": "",
                                                "text": ref["text"], "is_fake": ref["is_fake"]}))
    cand = A.add_norm_and_hashes(combined(clean))
    exact = cand["h_text_strip"].isin(set(ref_f["h_text_strip"].dropna()))
    pref = cand["prefix"].isin(set(ref_f["prefix"].dropna()))
    print(f"exact-text matches {int(exact.sum()):,} | 300-char prefix matches {int(pref.sum()):,} (of {len(cand):,})")

    def near_flags(thr, k):
        allx = pd.concat([ref_f, cand], ignore_index=True)
        is_new = np.r_[np.zeros(len(ref_f), bool), np.ones(len(cand), bool)]
        pairs = A.near_dup_pairs(allx, thr, k)
        pi, pj = pairs["i"].to_numpy(int), pairs["j"].to_numpy(int)
        cross = is_new[pi] != is_new[pj]
        idx = np.where(is_new[pi[cross]], pi[cross], pj[cross]) - len(ref_f)
        f = np.zeros(len(cand), bool); f[idx] = True
        return f

    near = near_flags(A.JACC_PRIMARY, A.SHINGLE_K)
    print(f"near-duplicate/containment matches (k={A.SHINGLE_K}, J>={A.JACC_PRIMARY}): {int(near.sum()):,}")
    if a.sensitivity:
        for k, thr in ((A.SENS_SHINGLE_K, A.JACC_SENS), (A.SHINGLE_K, 0.6), (A.SHINGLE_K, 0.9)):
            print(f"  sensitivity k={k} J>={thr}: {int(near_flags(thr, k).sum()):,} flagged")
    flagged = (exact | pref | near).to_numpy()
    rate = flagged.mean()
    print(f"OVERLAP with existing corpora: {int(flagged.sum()):,} of {len(cand):,} ({rate:.1%})")
    if rate > 0.20:
        print("!! more than 20% of this corpus already exists in the other three: it is not an independent corpus.")
    audit_dir = pathlib.Path(a.audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"flagged_{a.name}_ids.txt").write_text("\n".join(cand.loc[flagged, "doc_id"]) + "\n")
    keep = cand[~flagged].copy()

    # 3) output in the analysis_set schema (+ optional date)
    keep["n_words"] = keep["text"].str.split().str.len()
    keep["date"] = keep["doc_id"].map(date_of) if a.date_col else np.nan
    cols = ["dataset", "doc_id", "text", "is_fake", "cluster_id", "n_words", "date"]
    new_rows = keep[cols]
    if "date" not in ref.columns:
        ref["date"] = np.nan
    out = pd.concat([ref[cols], new_rows], ignore_index=True)
    out.to_csv(a.out, index=False)

    # 4) size rule + label-direction hand-check sample
    cnt = new_rows["is_fake"].value_counts().reindex([0, 1], fill_value=0)
    ok_n, ok_share = bool((cnt >= MIN_PER_CLASS).all()), bool(cnt.min() / max(cnt.sum(), 1) >= MIN_MINORITY)
    print(f"{a.name}: real {cnt[0]:,} / fake {cnt[1]:,} | size rule (>=3,000 per class): {'PASS' if ok_n else 'FAIL'} | "
          f"minority >= 40%: {'PASS' if ok_share else 'FAIL'}")
    chk = pd.concat([new_rows[new_rows["is_fake"] == c].sample(min(a.check_sample, int((new_rows["is_fake"] == c).sum())), random_state=0)
                     for c in (0, 1)]).sample(frac=1, random_state=0)             # shuffled, so class order gives nothing away
    blind = pd.DataFrame({"doc_id": chk["doc_id"], "text_preview": chk["text"].str.slice(0, 600),
                          "annotator_1_label": "", "annotator_2_label": ""})        # BLIND sheet: the dataset label is not shown
    chk_path, key_path = audit_dir / f"label_check_{a.name}.csv", audit_dir / f"label_key_{a.name}.csv"
    blind.to_csv(chk_path, index=False)
    chk[["doc_id", "is_fake"]].to_csv(key_path, index=False)
    print(f"wrote {a.out}\nwrote BLIND hand-check sheet {chk_path} and its key {key_path}\n"
          f"annotators fill the two label columns (fake/real) WITHOUT seeing the key; then: python src/analysis/label_agreement.py {chk_path} {key_path}")


if __name__ == "__main__":
    main()
