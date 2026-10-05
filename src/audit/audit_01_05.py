"""
audit_01_05.py  --  dataset audit, steps 01-05 (cell markers '# %%' work in VS Code / Jupytext / Colab).

What it does
    01  configuration + reproducibility record
    02  standardise every dataset to ONE schema (loaders are YOURS to fill in after 00_inspect_schema.py)
    03  within-dataset cleaning: drop too-short docs, drop exact-duplicate groups (label-conflicting groups
        are dropped entirely), assign near-duplicate cluster ids (nothing near-duplicate is deleted;
        the later split must be group-aware on cluster_id)
    04  basic statistics + label-verification samples to READ BY HAND
    05  cross-dataset overlap REPORT (exact, near-duplicate, prefix, containment, label conflicts)
        + a DRY RUN of "trim overlaps from one dataset" against pre-registered thresholds.
        Nothing is deleted across datasets in this script.

Standard schema (one row per article):
    dataset | doc_id | title | text | is_fake      (is_fake: 1 = fake, 0 = real -- you verify this in step 04)
"""
# %% 01 -- configuration & reproducibility
import collections, hashlib, itertools, json, pathlib, platform, random, re, sys
import numpy as np
import pandas as pd

SEED = 13
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = pathlib.Path("results") / "audit_01_05"

# near-duplicate machinery
SHINGLE_K = 5            # PRIMARY tier: word 5-shingles, Jaccard >= JACC_PRIMARY
SENS_SHINGLE_K = 3       # SENSITIVITY tier: word 3-shingles (tolerates scattered word edits), Jaccard >= JACC_SENS
NUM_PERM = 128           # MinHash permutations
BANDS, ROWS = 32, 4      # LSH banding; BANDS*ROWS must equal NUM_PERM. Candidate generation only;
assert BANDS * ROWS == NUM_PERM  # every candidate is re-checked with the TRUE Jaccard / containment.
JACC_PRIMARY = 0.8       # primary near-duplicate threshold
JACC_SENS = 0.5          # sensitivity threshold (computed with SENS_SHINGLE_K)
CONTAIN_THR = 0.9        # |A n B| / min(|A|,|B|) >= this -> "one is contained in the other"
MIN_SHINGLES_FOR_CONTAINMENT = 50   # ignore containment for tiny docs (too easy to be contained)
HEAD_WORDS = 100         # second view: first 100 words only (catches truncated copies)
PREFIX_CHARS = 300       # first-300-character prefix match (normalised text)
MIN_WORDS = 20           # step 03 drops docs shorter than this
MAX_BUCKET = 200         # LSH bucket larger than this is chained, not fully paired (logged)

# ---- PRE-REGISTERED DECISION RULE: EDIT THESE NUMBERS BEFORE YOU RUN STEP 05 ----------------------
PREREG = dict(
    min_docs_per_class=3000,   # after cleaning (and after trimming the trim-dataset), per dataset per class
    min_class_share=0.40,      # each class >= 40% of its dataset
    trim_dataset="WELFake",    # overlaps are removed from THIS (larger) dataset in the dry run
)
# ----------------------------------------------------------------------------------------------------

# %% 02 -- schema standardisation (loaders are intentionally NOT guessed)
STANDARD_COLS = ["dataset", "doc_id", "title", "text", "is_fake"]


def standardize(dataset, title, text, is_fake):
    """title/text: array-likes of str (title may be None). is_fake: array-like of 0/1 (1 = fake).
    You decide the 0/1 mapping from the inspect output + the hand check in step 04, then pass it here."""
    text = pd.Series(list(text), dtype="object")
    n = len(text)
    title = pd.Series([""] * n if title is None else list(title), dtype="object")
    df = pd.DataFrame({
        "dataset": dataset,
        "doc_id": [f"{dataset}-{i}" for i in range(n)],
        "title": title.fillna("").astype(str),
        "text": text.fillna("").astype(str),
        "is_fake": pd.Series(list(is_fake)).astype("int8"),
    })
    bad = set(df["is_fake"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{dataset}: is_fake contains values other than 0/1: {bad}")
    return df[STANDARD_COLS]


def load_all(base_dir="."):
    """Load the inspected files using their observed schemas and label mappings."""
    base_dir = pathlib.Path(base_dir)
    fake = pd.read_csv(base_dir / "Fake.csv")
    real = pd.read_csv(base_dir / "True.csv")
    isot = standardize(
        "ISOT",
        pd.concat([fake["title"], real["title"]], ignore_index=True),
        pd.concat([fake["text"], real["text"]], ignore_index=True),
        [1] * len(fake) + [0] * len(real),
    )

    mc = pd.read_csv(base_dir / "fake_or_real_news.csv")
    mcintire = standardize(
        "McIntire", mc["title"], mc["text"],
        mc["label"].map({"FAKE": 1, "REAL": 0}),
    )

    wel = pd.read_csv(base_dir / "WELFake_Dataset.csv")
    welfake = standardize(
        "WELFake", wel["title"], wel["text"],
        wel["label"].map({0: 0, 1: 1}),
    )
    return [isot, mcintire, welfake]


# %% normalisation, hashing, shingling, MinHash
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HANDLE_RE = re.compile(r"[@#]\w+")
DATELINE_RE = re.compile(r"^\s*[A-Za-z .,'/&-]{1,40}\(\s*(?:reuters|ap|afp|upi)\s*\)\s*[-\u2013\u2014:]*\s*", re.I)
TAG_RE = re.compile(r"\(\s*(?:reuters|ap|afp|upi)\s*\)", re.I)
REUTERS_RE = re.compile(r"\breuters\b", re.I)


def norm_light(s):
    """Pass A: lowercase, normalize punctuation/whitespace, retain content."""
    s = str(s).replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[^\w\s]", " ", s.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def norm_strip(s):
    """pass 2: ALSO strip URLs, @/# handles, a leading 'CITY (Reuters) -' dateline, '(Reuters)' tags and the word
    'reuters'.  Used for MATCHING ONLY -- so a Reuters story cleaned differently in two datasets still matches.
    (This is not the masking intervention of the main experiment.)"""
    s = URL_RE.sub(" ", str(s))
    s = DATELINE_RE.sub("", s, count=1)
    s = TAG_RE.sub(" ", s)
    s = HANDLE_RE.sub(" ", s)
    s = REUTERS_RE.sub(" ", s)
    return norm_light(s)


def h128(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def _hash_or_nan(s):
    return h128(s) if isinstance(s, str) and s else np.nan   # empty/NaN must never "match" each other


def add_norm_and_hashes(df):
    df = df.copy()
    df["n_words"] = df["text"].map(norm_light).str.split().str.len()
    t_l = df["text"].map(norm_light); t_s = df["text"].map(norm_strip)
    ti_l = df["title"].map(norm_light); ti_s = df["title"].map(norm_strip)
    df["text_ns"] = t_s                                           # kept for shingling / prefix
    df["h_text_light"] = t_l.map(_hash_or_nan)
    df["h_text_strip"] = t_s.map(_hash_or_nan)
    df["h_title_light"] = ti_l.map(_hash_or_nan)                  # titles: reported, NOT counted as duplicates
    df["h_title_strip"] = ti_s.map(_hash_or_nan)
    df["h_both_light"] = (ti_l + " | " + t_l).where(t_l != "").map(_hash_or_nan)
    df["h_both_strip"] = (ti_s + " | " + t_s).where(t_s != "").map(_hash_or_nan)
    df["prefix"] = t_s.str.slice(0, PREFIX_CHARS).where(t_s.str.len() >= PREFIX_CHARS).map(_hash_or_nan)
    return df


def shingles(words, k=SHINGLE_K):
    if not words:
        return np.empty(0, dtype=np.uint64)
    grams = [" ".join(words)] if len(words) < k else [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]
    arr = np.fromiter((int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") for g in grams),
                      dtype=np.uint64, count=len(grams))
    return np.unique(arr)                                         # sorted, unique


_rng = np.random.RandomState(SEED)
_A = _rng.randint(1, 2 ** 31, size=NUM_PERM).astype(np.uint64) * np.uint64(2) + np.uint64(1)
_B = _rng.randint(0, 2 ** 31, size=NUM_PERM).astype(np.uint64)


def minhash(h):
    if h.size == 0:
        return np.full(NUM_PERM, np.iinfo(np.uint32).max, dtype=np.uint32)
    with np.errstate(over="ignore"):
        m = (h[:, None] * _A[None, :] + _B[None, :]) >> np.uint64(32)
    return m.min(axis=0).astype(np.uint32)


def lsh_candidates(sigs):
    """candidate (i<j) pairs from banding. Only candidates -- verified later with true Jaccard."""
    n = len(sigs)
    pairs, big = set(), 0
    C = np.uint64(0x9E3779B97F4A7C15)
    for b in range(BANDS):
        band = sigs[:, b * ROWS:(b + 1) * ROWS].astype(np.uint64)
        with np.errstate(over="ignore"):
            key = ((band[:, 0] << np.uint64(32)) | band[:, 1]) ^ (((band[:, 2] << np.uint64(32)) | band[:, 3]) * C)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            idx = [int(x) for x in order[s:e]]
            if len(idx) > MAX_BUCKET:
                big += 1
                pairs.update(zip(idx[:-1], idx[1:]))              # chain only
            else:
                pairs.update(itertools.combinations(idx, 2))
    if big:
        print(f"    [note] {big} oversized LSH buckets were chained, not fully paired (boilerplate-heavy docs)")
    return pairs


def jac_con(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    m = min(a.size, b.size)
    return (inter / union if union else 0.0), (inter / m if m else 0.0), m


def near_dup_pairs(df, thr, k, verbose=True):
    """Pairs (positional i<j in df) whose TRUE k-shingle Jaccard >= thr OR containment >= CONTAIN_THR.
    Two views: full text and first HEAD_WORDS words. Returns max over views.
    NOTE: 5-shingle Jaccard is harsh on scattered edits (~5% of words changed -> J ~0.6), hence the k=3 tier."""
    words = df["text_ns"].str.split().tolist()
    out = {}
    for view, cut in (("full", None), ("head", HEAD_WORDS)):
        if verbose:
            print(f"  [{view}, k={k}] shingling + MinHash for {len(df):,} docs ...")
        sh = [shingles(w[:cut] if cut else w, k) for w in words]
        sigs = np.vstack([minhash(h) for h in sh])
        cand = lsh_candidates(sigs)
        if verbose:
            print(f"  [{view}] {len(cand):,} LSH candidates -> verifying with exact Jaccard/containment")
        for i, j in cand:
            jac, con, m = jac_con(sh[i], sh[j])
            con_ok = con >= CONTAIN_THR and m >= MIN_SHINGLES_FOR_CONTAINMENT
            if jac >= thr or con_ok:
                prev = out.get((i, j), (0.0, 0.0))
                out[(i, j)] = (max(prev[0], jac), max(prev[1], con if m >= MIN_SHINGLES_FOR_CONTAINMENT else 0.0))
    res = pd.DataFrame([(i, j, a, c) for (i, j), (a, c) in out.items()], columns=["i", "j", "jaccard", "containment"])
    return res


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# %% 03 -- within-dataset deduplication
def step03_within(df):
    """returns (clean_df_with_cluster_id, report_dict). Exact dups removed; near-dups only CLUSTERED."""
    name = df["dataset"].iloc[0]
    rep = {"dataset": name, "n_raw": len(df)}
    df = add_norm_and_hashes(df)
    short = df["n_words"].fillna(0) < MIN_WORDS
    rep["dropped_short_or_empty"] = int(short.sum())
    df = df[~short].copy()

    # Exact duplicates use the conservative lightly-normalized representation.
    g = df.groupby("h_text_light")["is_fake"]
    nuniq, size = g.transform("nunique"), g.transform("size")
    conflict = (nuniq > 1)
    rep["exact_dup_groups_with_label_conflict"] = int(df.loc[conflict, "h_text_light"].nunique())
    rep["dropped_docs_in_conflicting_groups"] = int(conflict.sum())
    df = df[~conflict].copy()
    dup = df.duplicated("h_text_light", keep="first")
    rep["dropped_exact_duplicates"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    # near-duplicate CLUSTERS (kept, not deleted)
    print(f"[03:{name}] near-duplicate clustering ({len(df):,} docs)")
    pairs = near_dup_pairs(df, JACC_PRIMARY, SHINGLE_K)
    sens = near_dup_pairs(df, JACC_SENS, SENS_SHINGLE_K)
    prim_set = set(zip(pairs["i"], pairs["j"]))
    rep["near_pairs_primary"] = len(pairs)
    rep["near_pairs_sensitivity_only"] = int(sum((i, j) not in prim_set for i, j in zip(sens["i"], sens["j"])))
    uf = UnionFind(len(df))
    for i, j, jac, con in pairs.itertuples(index=False):
        uf.union(int(i), int(j))
    df["cluster_id"] = [f"{name}-c{uf.find(i)}" for i in range(len(df))]
    cs = df.groupby("cluster_id")["is_fake"].agg(["size", "nunique"])
    rep["near_dup_clusters_size_ge2"] = int((cs["size"] >= 2).sum())
    rep["docs_in_near_dup_clusters"] = int(cs.loc[cs["size"] >= 2, "size"].sum())
    rep["largest_cluster"] = int(cs["size"].max())
    rep["near_dup_clusters_with_label_conflict"] = int((cs["nunique"] > 1).sum())
    rep["n_clean"] = len(df)
    return df, rep


# %% 04 -- statistics + label verification
def step04_stats(df, out_dir, n_samples=10):
    name = df["dataset"].iloc[0]
    rows = []
    for lab, sub in df.groupby("is_fake"):
        w = sub["n_words"]
        rows.append(dict(dataset=name, is_fake=lab, n=len(sub), share=len(sub) / len(df), words_mean=w.mean(),
                         words_median=w.median(), p5=w.quantile(.05), p95=w.quantile(.95),
                         title_present=(sub["title"].str.strip() != "").mean()))
    stats = pd.DataFrame(rows)
    # label verification: READ these by hand and confirm is_fake==1 really are fake articles
    path = out_dir / f"label_check_{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for lab in (1, 0):
            sub = df[df["is_fake"] == lab]
            f.write(f"\n{'#' * 20}  {name}  is_fake={lab} ({'FAKE' if lab else 'REAL'} per your mapping)\n")
            for _, r in sub.sample(min(n_samples, len(sub)), random_state=SEED).iterrows():
                f.write(f"\n- TITLE: {r['title'][:150]}\n  TEXT : {r['text'][:300].replace(chr(10), ' ')}\n")
    print(f"[04:{name}] label-check samples written to {path}  <-- READ THEM before trusting is_fake")
    return stats


# %% 05 -- cross-dataset overlap report (no deletion)
HASH_COLS = ["h_text_light", "h_text_strip", "h_both_light", "h_both_strip", "h_title_light", "h_title_strip"]


def step05_overlap(df_all, out_dir):
    names = list(df_all["dataset"].unique())
    df_all = df_all.reset_index(drop=True)
    ds = df_all["dataset"].values
    lab = df_all["is_fake"].values

    # Search each dataset pair independently; same-dataset near duplicates were already audited in step 03.
    print("[05] near-duplicate search for each dataset pair")
    pair_frames = []
    for a, b in itertools.combinations(names, 2):
        ab = df_all[df_all["dataset"].isin([a, b])].copy()
        positions = ab.index.to_numpy()
        p1 = near_dup_pairs(ab.reset_index(drop=True), JACC_PRIMARY, SHINGLE_K).astype({"i": int, "j": int})
        p1["tier"] = "primary"
        p2 = near_dup_pairs(ab.reset_index(drop=True), JACC_SENS, SENS_SHINGLE_K).astype({"i": int, "j": int})
        p2 = p2[~p2.set_index(["i", "j"]).index.isin(p1.set_index(["i", "j"]).index)].copy()
        p2["tier"] = "sensitivity"
        both = pd.concat([p1, p2], ignore_index=True)
        if not both.empty:
            both["i"] = positions[both["i"].to_numpy()]
            both["j"] = positions[both["j"].to_numpy()]
            pair_frames.append(both)
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
        columns=["i", "j", "jaccard", "containment", "tier"])
    pairs["ds_i"], pairs["ds_j"] = ds[pairs["i"].values], ds[pairs["j"].values]
    pairs["label_conflict"] = lab[pairs["i"].values] != lab[pairs["j"].values]

    flags = pd.DataFrame(index=df_all.index)   # per-doc overlap flags, used by the dry run
    report = []
    exact_rows = []
    for a, b in itertools.combinations(names, 2):
        A, B = df_all[df_all["dataset"] == a], df_all[df_all["dataset"] == b]
        blk = {"pair": f"{a} <-> {b}"}
        for col in HASH_COLS:
            m = A[["doc_id", col, "is_fake"]].dropna().merge(
                B[["doc_id", col, "is_fake"]].dropna(), on=col, suffixes=("_a", "_b"))
            conflict_rows = m[m["is_fake_a"] != m["is_fake_b"]]
            blk[f"exact[{col}] pairs"] = len(m)
            blk[f"exact[{col}] unique_docs_{a}"] = int(A[col].isin(B[col].dropna()).sum())
            blk[f"exact[{col}] unique_docs_{b}"] = int(B[col].isin(A[col].dropna()).sum())
            blk[f"exact[{col}] label_conflicts"] = len(conflict_rows)
            for _, row in conflict_rows.iterrows():
                exact_rows.append(dict(pair=f"{a} <-> {b}", evidence=col,
                                       doc_id_a=row["doc_id_a"], doc_id_b=row["doc_id_b"],
                                       is_fake_a=int(row["is_fake_a"]), is_fake_b=int(row["is_fake_b"]),
                                       jaccard=np.nan, containment=np.nan))
            if col in {"h_text_light", "h_text_strip", "h_both_light", "h_both_strip"}:
                flags[f"{a}~{b}:exact_{col}"] = df_all.index.isin(A.index[A[col].isin(B[col].dropna())]) | \
                                                   df_all.index.isin(B.index[B[col].isin(A[col].dropna())])
        pa, pb = set(A["prefix"].dropna()), set(B["prefix"].dropna())
        blk[f"prefix300 docs_{a}"] = int(A["prefix"].isin(pb).sum())
        blk[f"prefix300 docs_{b}"] = int(B["prefix"].isin(pa).sum())
        flags[f"{a}~{b}:prefix"] = df_all["prefix"].isin(pa & pb) & df_all["dataset"].isin([a, b])
        sub = pairs[((pairs["ds_i"] == a) & (pairs["ds_j"] == b)) | ((pairs["ds_i"] == b) & (pairs["ds_j"] == a))]
        for tier, t in sub.groupby("tier"):
            docs = set(t["i"]) | set(t["j"])
            blk[f"near[{tier}] pairs"] = len(t)
            blk[f"near[{tier}] docs_{a}"] = int(sum(ds[d] == a for d in docs))
            blk[f"near[{tier}] docs_{b}"] = int(sum(ds[d] == b for d in docs))
            blk[f"near[{tier}] label conflicts"] = int(t["label_conflict"].sum())
            blk[f"containment[{tier}] pairs"] = int((t["containment"] >= CONTAIN_THR).sum())
            blk[f"prefix300[{tier}] pairs"] = 0
        prim = sub[sub["tier"] == "primary"]
        flags[f"{a}~{b}:near"] = df_all.index.isin(set(prim["i"]) | set(prim["j"]))
        sens = sub[sub["tier"] == "sensitivity"]
        flags[f"{a}~{b}:nearsens"] = df_all.index.isin(set(sens["i"]) | set(sens["j"]))
        report.append(blk)

        for _, row in sub[sub["label_conflict"]].iterrows():
            exact_rows.append(dict(pair=f"{a} <-> {b}", evidence=f"near_{row['tier']}",
                                   doc_id_a=df_all.iloc[int(row["i"])] ["doc_id"],
                                   doc_id_b=df_all.iloc[int(row["j"])] ["doc_id"],
                                   is_fake_a=int(df_all.iloc[int(row["i"])] ["is_fake"]),
                                   is_fake_b=int(df_all.iloc[int(row["j"])] ["is_fake"]),
                                   jaccard=row["jaccard"], containment=row["containment"]))

    # outputs
    rep_df = pd.DataFrame(report).fillna(0).set_index("pair").T
    rep_df.to_csv(out_dir / "overlap_report.csv")
    prev = lambda idx: df_all.loc[idx, "text"].str.slice(0, 200).str.replace("\n", " ")
    pv = pairs.copy()
    pv["text_i"], pv["text_j"] = prev(pv["i"].values).values, prev(pv["j"].values).values
    pv.to_csv(out_dir / "overlap_pairs.csv", index=False)
    pv["prefix300_match"] = pv.apply(lambda r: df_all.iloc[int(r["i"])] ["prefix"] == df_all.iloc[int(r["j"])] ["prefix"], axis=1)
    border = pv[(pv["tier"] == "sensitivity") | ((pv["jaccard"] >= JACC_PRIMARY - 0.05) &
                                                      (pv["jaccard"] < JACC_PRIMARY))]
    parts = [g.sample(min(50, len(g)), random_state=SEED) for _, g in border.groupby(["ds_i", "ds_j"])]
    (pd.concat(parts) if parts else border).to_csv(out_dir / "borderline_to_inspect.csv", index=False)
    flags.insert(0, "doc_id", df_all["doc_id"].values)
    flags.to_csv(out_dir / "overlap_flags_per_doc.csv", index=False)
    pd.DataFrame(exact_rows, columns=["pair", "evidence", "doc_id_a", "doc_id_b", "is_fake_a", "is_fake_b", "jaccard", "containment"]
                 ).to_csv(out_dir / "label_conflicts.csv", index=False)
    pd.DataFrame([{"pair": r["pair"], "evidence": "exact_or_near", "label_conflicts": 1}
                  for r in exact_rows]).groupby(["pair", "evidence"], as_index=False).size().rename(columns={"size": "n"}).to_csv(
                      out_dir / "label_conflict_summary.csv", index=False)
    return rep_df, flags


def dry_run_trim(df_all, flags, prereg=PREREG, out_dir=OUT_DIR):
    """Pretend we removed from prereg['trim_dataset'] every doc flagged as overlapping another dataset.
    Variant 1: exact OR primary near-dup OR prefix.  Variant 2: variant 1 PLUS the sensitivity tier.
    Reports sizes/class balance against the pre-registered thresholds. Nothing is deleted."""
    t = prereg["trim_dataset"]
    rows = []
    for tier, suffixes in (("primary", (":exact_", ":near", ":prefix")),
                           ("primary+sensitivity", (":exact_", ":near", ":nearsens", ":prefix"))):
        def allowed(col):
            return (":exact_" in col or col.endswith(":prefix") or col.endswith(":near") or
                (":nearsens" in col and ":nearsens" in suffixes))
        for other in ("ISOT", "McIntire"):
            pair_cols = [c for c in flags.columns if c != "doc_id" and
                         t in c.split(":")[0].split("~") and other in c.split(":")[0].split("~") and
                         allowed(c)]
            hit = flags[pair_cols].any(axis=1) if pair_cols else pd.Series(False, index=flags.index)
            removed = hit & (df_all["dataset"] == t)
            rows.append(dict(tier=tier, removal_scenario=f"WELFake minus {other}",
                             original_welfake=int((df_all["dataset"] == t).sum()), removed=int(removed.sum()),
                             remaining_welfake=int((df_all["dataset"] == t).sum() - removed.sum())))
        union_cols = [c for c in flags.columns if c != "doc_id" and t in c.split(":")[0].split("~") and
                      allowed(c)]
        union_hit = flags[union_cols].any(axis=1) if union_cols else pd.Series(False, index=flags.index)
        union_removed = union_hit & (df_all["dataset"] == t)
        rows.append(dict(tier=tier, removal_scenario="WELFake minus ISOT and McIntire union",
                         original_welfake=int((df_all["dataset"] == t).sum()), removed=int(union_removed.sum()),
                         remaining_welfake=int((df_all["dataset"] == t).sum() - union_removed.sum())))
    dry = pd.DataFrame(rows)
    out_dir = pathlib.Path(out_dir)
    dry.to_csv(out_dir / "welfake_dry_run.csv", index=False)
    print("\n== WELFake overlap dry run ==\n", dry.to_string(index=False))

    base = [c for c in flags.columns if c != "doc_id" and t in c.split(":")[0].split("~")]
    results = {}
    for label, cols in (("primary", [c for c in base if not c.endswith(":nearsens")]), ("primary+sensitivity", base)):
        hit = flags[cols].any(axis=1) if cols else pd.Series(False, index=flags.index)
        keep = ~(hit & (df_all["dataset"] == t))
        out = []
        for d, sub in df_all[keep].groupby("dataset"):
            n0, n1 = int((sub["is_fake"] == 0).sum()), int((sub["is_fake"] == 1).sum())
            share = min(n0, n1) / max(1, n0 + n1)
            ok = min(n0, n1) >= prereg["min_docs_per_class"] and share >= prereg["min_class_share"]
            out.append(dict(dataset=d, real=n0, fake=n1, min_class_share=round(share, 3), PASS=ok))
        res = pd.DataFrame(out)
        removed = int((hit & (df_all["dataset"] == t)).sum())
        print(f"\n[dry run: {label}] would remove {removed:,} {t} docs; thresholds: {prereg}")
        print(res.to_string(index=False))
        print("DECISION RULE ->", "ALL PASS: Triple A viable" if res["PASS"].all() else
              "AT LEAST ONE FAILS: apply your fallback (C, or another independent full-text dataset)")
        results[label] = res
        res.assign(scenario=label).to_csv(out_dir / f"triple_a_{label}.csv", index=False)
    return results


# %% run everything
def run_audit(frames, out_dir=OUT_DIR, smoke_test=None):
    """smoke_test=3000 -> run on a random 3000 docs per dataset just to check loaders/labels/runtime.
    Overlap COUNTS from a smoke test are meaningless (sampling hides most duplicates); do the real run without it."""
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if smoke_test:
        print(f"!! SMOKE TEST: sampling {smoke_test} docs per dataset -- do not read overlap numbers from this run")
        frames = [f.sample(min(smoke_test, len(f)), random_state=SEED).reset_index(drop=True) for f in frames]
    json.dump({"seed": SEED, "shingle_k": SHINGLE_K, "num_perm": NUM_PERM, "bands": BANDS, "rows": ROWS,
               "jacc_primary": JACC_PRIMARY, "jacc_sens": JACC_SENS, "contain_thr": CONTAIN_THR,
               "head_words": HEAD_WORDS, "prefix_chars": PREFIX_CHARS, "min_words": MIN_WORDS,
               "prereg": PREREG, "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
               "platform": platform.platform()}, open(out_dir / "config.json", "w"), indent=2)
    pd.DataFrame([{"representation": "light_and_artifact_stripped", "auc": np.nan,
                   "status": "not_run; domain classifier training is outside Steps 01-05 and prohibited by request"}]).to_csv(
                       out_dir / "domain_auc.csv", index=False)
    cleaned, reps, stats = [], [], []
    for df in frames:
        c, r = step03_within(df)
        cleaned.append(c); reps.append(r)
        stats.append(step04_stats(c, out_dir))
    pd.DataFrame(reps).to_csv(out_dir / "within_dedup_report.csv", index=False)
    pd.concat(stats).to_csv(out_dir / "basic_stats.csv", index=False)
    print("\n== within-dataset cleaning ==\n", pd.DataFrame(reps).set_index("dataset").T.to_string())
    print("\n== basic stats ==\n", pd.concat(stats).round(2).to_string(index=False))
    df_all = pd.concat(cleaned, ignore_index=True)
    rep_df, flags = step05_overlap(df_all, out_dir)
    print("\n== cross-dataset overlap (non-zero rows) ==\n", rep_df[(rep_df != 0).any(axis=1)].to_string())
    dry_run_trim(df_all, flags, out_dir=out_dir)
    slim = df_all.drop(columns=["text_ns"])
    if _has_parquet():
        slim.to_parquet(out_dir / "cleaned_all.parquet")
    else:
        slim.to_csv(out_dir / "cleaned_all.csv", index=False)
    return df_all, rep_df, flags


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    run_audit(load_all())
