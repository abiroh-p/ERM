"""
prepare_nela.py -- turn a NELA-GT SQLite release into an outlet-capped real/fake sample for add_dataset.py.

    python src/audit/prepare_nela.py --db nela-gt-2020.db --labels labels.csv --out data/raw/NELA/nela2020_sample.csv \
        --per-class 6000

Facts about NELA-GT this script is built around (from the MELALab/nela-gt README and the NELA-GT-2020 paper):
  * table `newsdata`: id, date (YYYY-MM-DD), source, title, content, author, published, published_utc, collection_utc, url
  * labels are SOURCE-level (Media Bias/Fact Check): 0 reliable, 1 unreliable, 2 mixed, -1/null invalid.
    -> reliable => real (0), unreliable => fake (1); mixed and null are DROPPED.  The label reflects the outlet, not the article.
  * the text is deliberately degraded for copyright: tokens are replaced by '@' (7 per 100 tokens for long articles, 5 per 20 for
    short ones).  Standalone '@' markers are removed here (--keep-at to keep them), but the missing tokens stay missing.  The other
    corpora are not degraded this way; report it as a limitation (and as a reason DistilBERT-style random-deletion effects matter).
  * outlets differ hugely in volume; a per-class OUTLET CAP is found by bisection so the sample has as many different outlets as
    possible (every outlet contributes min(n_outlet, cap) articles) instead of a few prolific ones dominating a class.
The label direction is checked on outlets whose reliability is well known; the script aborts if it looks inverted, because
documentation of label direction has already been wrong once in this project (WELFake) and fixed once in the NELA README.
"""
import argparse, pathlib, re, sqlite3, sys
import numpy as np
import pandas as pd

AT_RE = re.compile(r"(?:(?<=\s)|^)@+(?=\s|$)")
KNOWN_RELIABLE = {"reuters", "apnews", "associatedpress", "bbc", "npr", "pbs", "theguardian", "thehill", "usatoday"}
KNOWN_UNRELIABLE = {"infowars", "naturalnews", "thegatewaypundit", "beforeitsnews", "zerohedge", "prisonplanet", "yournewswire"}


def load_labels(path, label_col=None):
    lab = pd.read_csv(path)
    print(f"label file columns: {list(lab.columns)}")
    src = lab.columns[0]
    cand = [c for c in lab.columns[1:] if label_col == c or (label_col is None and re.search("label|aggreg", c, re.I))]
    col = cand[0] if cand else lab.columns[1]
    print(f"using '{col}' as the aggregated label (override with --label-col); value counts:\n{lab[col].value_counts(dropna=False).to_string()}")
    lab = lab[[src, col]].rename(columns={src: "source", col: "raw"})
    lab["source"] = lab["source"].astype(str).str.strip().str.lower()
    lab["raw"] = pd.to_numeric(lab["raw"], errors="coerce")
    return lab.dropna().astype({"raw": int}).drop_duplicates("source")


def check_direction(lab):
    s = lab.set_index("source")["raw"]
    bad = [k for k in KNOWN_RELIABLE if k in s.index and s[k] == 1] + [k for k in KNOWN_UNRELIABLE if k in s.index and s[k] == 0]
    seen = [k for k in KNOWN_RELIABLE | KNOWN_UNRELIABLE if k in s.index]
    print(f"direction check on {len(seen)} well-known outlets present in the labels: "
          + ", ".join(f"{k}={s[k]}" for k in sorted(seen)))
    if bad:
        sys.exit(f"label direction looks INVERTED or mis-parsed for {bad}; fix --label-col / the mapping before sampling")
    if not seen:
        print("!! none of the well-known outlets are present: verify the direction by hand on a few outlets before using the sample")


def water_fill(counts, target):
    """smallest cap with sum(min(n_o, cap)) >= target (all outlets if the class is smaller than target)."""
    lo, hi = 1, int(counts.max())
    if counts.sum() <= target:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if np.minimum(counts, mid).sum() >= target else (mid + 1, hi)
    return lo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="NELA-GT SQLite file (table newsdata)")
    ap.add_argument("--labels", required=True, help="source labels CSV from the same release")
    ap.add_argument("--label-col")
    ap.add_argument("--out", default="data/raw/NELA/nela_sample.csv")
    ap.add_argument("--per-class", type=int, default=6000, help="target articles per class BEFORE the overlap/duplicate audit")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--date-from"); ap.add_argument("--date-to")
    ap.add_argument("--keep-at", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lab = load_labels(a.labels, a.label_col); check_direction(lab)
    lab = lab[lab["raw"].isin([0, 1])]
    print(f"{len(lab)} outlets with a usable label ({int((lab.raw == 0).sum())} reliable, {int((lab.raw == 1).sum())} unreliable); mixed/null dropped")

    con = sqlite3.connect(a.db)
    q = "SELECT id, source, date FROM newsdata"
    meta = pd.read_sql_query(q, con)
    meta["source"] = meta["source"].astype(str).str.strip().str.lower()
    meta = meta.merge(lab, on="source", how="inner")
    if a.date_from:
        meta = meta[meta["date"] >= a.date_from]
    if a.date_to:
        meta = meta[meta["date"] <= a.date_to]
    print(f"{len(meta):,} labelled articles in range")

    rng = np.random.RandomState(a.seed)
    picks = []
    for raw, name in ((0, "real"), (1, "fake")):
        g = meta[meta["raw"] == raw]
        counts = g.groupby("source").size()
        cap = water_fill(counts.to_numpy(), a.per_class)
        sel = g.assign(_r=rng.rand(len(g))).sort_values("_r").groupby("source").head(cap).drop(columns="_r")   # random <=cap per outlet
        if len(sel) > a.per_class:                                           # bisection overshoots by < one outlet-step
            sel = sel.sample(a.per_class, random_state=a.seed)
        print(f"{name}: {len(counts)} outlets, cap {cap}/outlet -> {len(sel):,} articles from {sel['source'].nunique()} outlets "
              f"(largest outlet share {sel['source'].value_counts(normalize=True).iloc[0]:.1%})")
        picks.append(sel.assign(label=name))
    meta = pd.concat(picks, ignore_index=True)

    ids = meta["id"].tolist(); parts = []
    for i in range(0, len(ids), 5000):                                       # fetch bodies only for the chosen ids
        chunk = ids[i:i + 5000]
        parts.append(pd.read_sql_query(f"SELECT id, title, content FROM newsdata WHERE id IN ({','.join('?' * len(chunk))})", con, params=chunk))
    df = meta.merge(pd.concat(parts), on="id", how="left")
    df["content"] = df["content"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    if not a.keep_at:
        df["content"] = df["content"].map(lambda t: re.sub(r"\s+", " ", AT_RE.sub(" ", t)).strip())
        df["title"] = df["title"].map(lambda t: re.sub(r"\s+", " ", AT_RE.sub(" ", t)).strip())
    df = df[df["content"].str.split().str.len() >= a.min_words]
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "source", "date", "title", "content", "label"]].to_csv(out, index=False)
    print(f"wrote {out}: {df['label'].value_counts().to_dict()}")
    print("next: python src/audit/add_dataset.py --input", out, "--name NELA2020 --text-col content --title-col title "
          "--label-col label --fake-values fake --date-col date")


if __name__ == "__main__":
    main()
