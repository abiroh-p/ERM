"""
exp_tfidf.py -- factorial artifact-attribution experiment, TF-IDF + Logistic Regression arm.

    python exp_tfidf.py diagnose            # artifact coverage per class + most label-predictive tokens (READ THIS)
    python exp_tfidf.py spans               # compute & cache source/entity spans for the sampling universe (slow once)
    python exp_tfidf.py run                 # the full factorial experiment -> results_tfidf.csv
    python exp_tfidf.py run --smoke         # tiny run to check everything works

DESIGN (all choices live in CFG below)
  Factors (each 0/1):  S = source/publisher markers removed
                       L = length-matched resampling (class-balanced within length bins -> no class/length link)
                       E = named entities removed
  8 factorial cells (C0,S,L,E,SL,SE,LE,SLE) + 3 random-word-masking controls R_S, R_E, R_SE that delete the SAME
  NUMBER of words per document as the real masking, at random positions.
  EXPLORATORY extension (added after inspecting residual cues): T = temporal expressions; conditions T, SLET, R_T.
  Constant across ALL conditions of a (dataset, seed): sample size and class balance (50/50), so conditions differ
  only in the intervention. Every transformation is applied identically to the training data and to the test data.
  Splits are group-aware (near-duplicate clusters never straddle train/test).
  Cross-dataset: train on dataset A (its train clusters), test on all of dataset B's sampled docs.
  In-domain: train on A (train clusters), test on A's held-out clusters.
"""
import argparse, itertools, json, pathlib, pickle, re, sys, time, zlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_path(value, *, fallback=None, default_dir=None):
    if value is None and fallback is not None:
        value = fallback
    if value is None:
        return default_dir
    p = pathlib.Path(value)
    if p.is_absolute():
        return p
    if default_dir is not None:
        return default_dir / p
    return REPO_ROOT / p


CFG = dict(
    analysis_csv=str(repo_path("data/processed/analysis_set.csv")),
    spans_cache=str(repo_path("cache/spans_cache.pkl")),
    out_csv=str(repo_path("results/raw/tfidf/results_tfidf.csv")),
    universe_per_class=4000,     # docs per dataset-class eligible for sampling (bounds NER time)
    n_train=1500,                # target docs PER CLASS for training (reduced automatically if L-matching can't reach it)
    n_test_in=500,               # per class, in-domain held-out test
    n_test_x=800,                # per class, cross-dataset test
    test_frac=0.2,               # fraction of near-duplicate clusters held out for in-domain testing
    length_bins=10,
    seeds=[0, 1, 2, 3, 4],
    entity_labels=["PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE"],
    spacy_model="en_core_web_sm",
)

# --------------------------------------------------------------------------------------------------------------
# Condition table: (S, L, E, T, random-control-of)
#   S source markers | L length matching | E named entities | T temporal expressions (EXPLORATORY: added after
#   inspecting the residual cues, so report it as such). The pre-planned design is the 8-cell S x L x E factorial.
CONDITIONS = {
    "C0": (0, 0, 0, 0, None), "S": (1, 0, 0, 0, None), "L": (0, 1, 0, 0, None), "E": (0, 0, 1, 0, None),
    "SL": (1, 1, 0, 0, None), "SE": (1, 0, 1, 0, None), "LE": (0, 1, 1, 0, None), "SLE": (1, 1, 1, 0, None),
    "R_S": (0, 0, 0, 0, "S"), "R_E": (0, 0, 0, 0, "E"), "R_SE": (0, 0, 0, 0, "SE"),
    # exploratory extension: temporal expressions alone, on top of everything else, and a size-matched random control
    "T": (0, 0, 0, 1, None), "SLET": (1, 1, 1, 1, None), "R_T": (0, 0, 0, 0, "T"),
    # exploratory split of S (is the source arm hurting because some markers are SHARED across low-quality sites?)
    "S_W": ("W", 0, 0, 0, None), "S_B": ("B", 0, 0, 0, None), "R_W": (0, 0, 0, 0, "W"), "R_B": (0, 0, 0, 0, "B"),
}

# --------------------------------------------------------------------------------------------------------------
# S: source / publisher markers.  Extend EXTRA_SOURCE_TERMS after reading `diagnose` output.
EXTRA_SOURCE_TERMS = []          # e.g. ["21st century wire", "natural news"]  (case-insensitive, whole words)
_SP = [
    (r"https?://\S+|www\.\S+", 0),
    (r"pic\.twitter\.com/\S+", re.I),
    (r"^\s*[A-Z][A-Za-z .,'/&-]{1,40}\(\s*(?:Reuters|AP|AFP|UPI)\s*\)\s*[-\u2013\u2014:]*", re.M),   # dateline
    (r"\(\s*(?:Reuters|AP|AFP|UPI)\s*\)", 0),
    (r"\b(?:Reuters|Breitbart|Infowars|21st Century Wire|Natural News|Daily Kos|Fox News|Washington Post|"
     r"New York Times|NY Times|Huffington Post|HuffPost|CNN|BBC|Associated Press|Getty Images)\b", re.I),
    (r"\bfeatured image\b[^\n.]{0,80}", re.I),
    (r"\b(?:image|photo|screenshot|video)\s+(?:via|by|credit)\b[^\n.]{0,80}", re.I),
    (r"\b(?:share this|share on facebook|tweet this|like us on facebook|follow us on|subscribe to)\b|\bread more\b\s*:?[^\n.]{0,60}", re.I),
    (r"google\s+pinterest\s+digg\s+linkedin\s+reddit\s+stumbleupon\s+print\s+delicious\s+pocket\s+tumblr", re.I),
    (r"\bfollow\b[^\n.]{0,100}?\b(?:twitter|facebook|instagram|periscope)\b(?:\s*(?:and|,)\s*(?:twitter|facebook|instagram|periscope)\b)*\s*:?\s*(?:@\w+)?", re.I),   # 'Follow X on Twitter @x'
    (r"^\s*(?:advertisements?|share|print|tweet|email|comments?|related(?: articles)?|sources?|via|read more)\s*:?\s*$", re.I | re.M),  # lines that are ONLY boilerplate
    (r"^\s*(?:source|sources|via|image|photo|credit)\s*:\s*[^\n]{0,100}$", re.I | re.M),                           # 'Source: ...' credit lines
    (r"\bshare\s+(?:this|on\s+\w+)(?:\s+(?:article|post|story))?(?:\s+tweet)?\s*:?", re.I),                      # 'Share this:' 'Share on Facebook Tweet'
    (r"\bprint\s+this\s+(?:post|page|article)\b[^\n.]{0,40}", re.I),
    (r"-?\s*\badvertisement\b(?:\s+square,\s+site\s+wide)?\s*-?", re.I),                                         # '- Advertisement -'
    (r"\bposted\s+(?:at\s+[\d:]+\s*[ap]m\s+)?on\s+\w+\s+\d{1,2},\s+\d{4}(?:\s+by\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,2})?", re.I),
    (r"\bSF Source\b[^\n]{0,60}", 0),                                                                             # 'SF Source The New American Nov. 2016'
    (r"\bsources?\s*:\s*[^\n.]{0,60}|\(\s*source\s*\)|\|\s*(?=sources?\s*:)", re.I),                                      # 'Source: ...' '( source )'
    (r"\b(?:or\s+)?e-?mail\s+(?:him|her|them|me|us)?\s*at\s+[\w.+-]+@[\w-]+\.\s?\w+|\b[\w.+-]+@[\w-]+\.\s?(?:com|org|net|edu|co)\b", re.I),
    (r"https?\s*:\s*/\s*/\s*\S*|https?\s*:\s*\.|\bpic\.?\s*twitter\.?(?:\s*com\S*)?|\bt\.\s?co/\S+", re.I),   # URLs broken by tokenisation
    (r"[\u2014\u2013-]\s+[^()\n]{1,60}\(@\w+\)(?:\s+[A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+(?:19|20)\d{2},?)?", 0),    # embedded-tweet attribution
    (r"\b[\w-]+\.\s?(?:com|org|net|edu)\b(?:/\S*)?", re.I),                                                       # 'breitbart. com'
    (r"\bclick\s+(?:here|below|to\s+\w+)\b[^\n.]{0,40}|\bleave\s+a\s+(?:comment|reply)\b|"
     r"\bplease\s+(?:feel\s+free\s+to\s+)?(?:share|like|follow|subscribe|support|donate|click|comment)\b[^\n.]{0,60}", re.I),
    (r"(?<!\w)[@#]\w+", 0),
]
_S_RES = [re.compile(p, f) for p, f in _SP]
if EXTRA_SOURCE_TERMS:
    _S_RES.append(re.compile(r"\b(?:" + "|".join(re.escape(t) for t in EXTRA_SOURCE_TERMS) + r")\b", re.I))
S_NAMES = ["url", "twitter_pic", "dateline", "agency_tag", "outlet_names", "featured_image",
           "image_credit", "boilerplate_cta", "social_share_strip", "follow_social", "boilerplate_line",
           "credit_line", "share_prompts", "print_prompt", "ad_markers", "posted_stamp", "sf_source", "source_credit", "email_address", "broken_urls", "tweet_attribution", "bare_domains", "click_prompts", "handles_hashtags"] + (["extra_terms"] if EXTRA_SOURCE_TERMS else [])


# T: temporal expressions -- weekdays, months (capitalised, so the modal verb 'may' is untouched), years,
# numeric dates and clock times.
_T_RES = [
    re.compile(r"\b(?:Mon|Tues?|Wed(?:nes)?|Thu(?:rs?)?|Fri|Sat(?:ur)?|Sun)(?:day)?\b\.?"),
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|"
               r"Nov(?:ember)?|Dec(?:ember)?)\b\.?(?:\s+\d{1,2}(?:st|nd|rd|th)?\b)?(?:,?\s+(?:19|20)\d{2}\b)?"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?", re.I),
]


def spans_time(text):
    out = []
    for rx in _T_RES:
        out.extend(m.span() for m in rx.finditer(text))
    return out


WIRE_NAMES = {"dateline", "agency_tag", "outlet_names"}          # S_W: newswire datelines/tags + outlet names
_WIRE_IDX = [i for i, n in enumerate(S_NAMES) if n in WIRE_NAMES]
_WEB_IDX = [i for i, n in enumerate(S_NAMES) if n not in WIRE_NAMES]   # S_B: everything else (share/ads/credits/embeds/URLs...)


def spans_source(text, which="all"):
    idx = {"all": range(len(_S_RES)), "W": _WIRE_IDX, "B": _WEB_IDX}[which]
    out = []
    for i in idx:
        out.extend(m.span() for m in _S_RES[i].finditer(text))
    return out


# --------------------------------------------------------------------------------------------------------------
# E: named entities (spaCy NER if available, otherwise a CRUDE capitalisation heuristic -- state which in the paper)
_ENT_FALLBACK = re.compile(r"\b(?:[A-Z][A-Za-z'\u2019.-]+)(?:\s+(?:of|the|de|von|van|and|for)?\s*[A-Z][A-Za-z'\u2019.-]+)+\b|\b[A-Z]{2,6}\b")


class EntityFinder:
    def __init__(self):
        self.method = "regex_fallback"
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load(CFG["spacy_model"], exclude=["parser", "lemmatizer", "attribute_ruler", "senter"])
            self.nlp.max_length = 3_000_000
            self.method = f"spacy:{CFG['spacy_model']}"
        except Exception as e:                       # noqa
            print(f"[E] spaCy not available ({type(e).__name__}); using the regex fallback. "
                  f"For the real run: pip install spacy && python -m spacy download {CFG['spacy_model']}")

    def spans(self, texts):
        if self.nlp is None:
            return [[m.span() for m in _ENT_FALLBACK.finditer(t)] for t in texts]
        keep = set(CFG["entity_labels"])
        out = []
        for doc in self.nlp.pipe(texts, batch_size=16):
            out.append([(e.start_char, e.end_char) for e in doc.ents if e.label_ in keep])
        return out


def remove_spans(text, spans):
    if not spans:
        return text
    out, last = [], 0
    for s, e in sorted(spans):
        if e <= last:
            continue
        if s > last:
            out.append(text[last:s])
        last = e
    out.append(text[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def random_drop_words(text, m, rng):
    w = text.split()
    if m <= 0 or not w:
        return text
    m = min(m, len(w) - 1)
    drop = set(rng.choice(len(w), m, replace=False).tolist())
    return " ".join(x for i, x in enumerate(w) if i not in drop)


# --------------------------------------------------------------------------------------------------------------
# data handling
def load_universe(cfg=CFG, seed=0):
    df = pd.read_csv(cfg["analysis_csv"])
    df["text"] = df["text"].fillna("").astype(str)
    df["n_words"] = df["text"].str.split().str.len()
    parts = []
    for (ds, lab), g in df.groupby(["dataset", "is_fake"]):
        parts.append(g.sample(min(cfg["universe_per_class"], len(g)), random_state=seed))
    u = pd.concat(parts, ignore_index=True)
    if cfg.get("exclude_file"):
        ids = set(pathlib.Path(cfg["exclude_file"]).read_text().split())
        n0 = len(u); u = u[~u["doc_id"].isin(ids)].reset_index(drop=True)
        print(f"[exclude] removed {n0 - len(u):,} of {n0:,} universe docs listed in {cfg['exclude_file']}")
    bins = np.zeros(len(u), int)
    for ds, idx in u.groupby("dataset").groups.items():
        nw = u.loc[idx, "n_words"].to_numpy()
        edges = np.unique(np.quantile(nw, np.linspace(0, 1, cfg["length_bins"] + 1)))
        bins[u.index.get_indexer(idx)] = np.clip(np.searchsorted(edges, nw, side="right") - 1, 0, max(len(edges) - 2, 0))
    u["bin"] = bins
    return u


def compute_spans(u, cfg=CFG):
    path = pathlib.Path(cfg["spans_cache"])
    cache = pickle.load(open(path, "rb")) if path.exists() else {"method": None, "spans": {}}
    ef = EntityFinder()
    if cache["method"] not in (None, ef.method):
        print(f"[spans] cache was built with {cache['method']} but now {ef.method}: rebuilding"); cache = {"method": None, "spans": {}}
    cache["method"] = ef.method
    todo = u[~u["doc_id"].isin(cache["spans"])]
    print(f"[spans] {len(todo):,} docs to process ({len(u) - len(todo):,} cached), entity method = {ef.method}")
    t0, B = time.time(), 500
    for i in range(0, len(todo), B):
        chunk = todo.iloc[i:i + B]
        ents = ef.spans(chunk["text"].tolist())
        for did, txt, e in zip(chunk["doc_id"], chunk["text"], ents):
            cache["spans"][did] = e          # NER spans only (S and T are regex, computed on the fly)
        pickle.dump(cache, open(path, "wb"))
        print(f"  {min(i + B, len(todo)):,}/{len(todo):,}  ({time.time() - t0:.0f}s)", flush=True)
    return cache


_MASK_CACHE, _RCACHE = {}, {}      # deterministic S/E/T texts (kept for the whole run); random controls (cleared per seed)


def _ent(v):                      # tolerate old caches that stored (source_spans, entity_spans) tuples
    return v[1] if isinstance(v, tuple) else v


def _spans_for(letters, t, ent):
    sp = []
    if "S" in letters:
        sp += spans_source(t)
    if "W" in letters:
        sp += spans_source(t, "W")
    if "B" in letters:
        sp += spans_source(t, "B")
    if "E" in letters:
        sp += ent
    if "T" in letters:
        sp += spans_time(t)
    return sp


def make_texts(df, cond, spans, seed):
    S, L, E, T, R = CONDITIONS[cond]
    letters = (S if isinstance(S, str) else ("S" if S else "")) + ("E" if E else "") + ("T" if T else "")
    out = []
    for did, t in zip(df["doc_id"], df["text"]):
        if not R and not letters:
            out.append(t); continue
        if R:
            v = _RCACHE.get((did, R))
            if v is None:
                real = _spans_for(R, t, _ent(spans[did]))
                m = len(t.split()) - len(remove_spans(t, real).split())
                rng = np.random.RandomState(zlib.crc32(f"{seed}-{did}-{R}".encode()) % (2 ** 31))
                v = _RCACHE[(did, R)] = random_drop_words(t, m, rng)
        else:
            v = _MASK_CACHE.get((did, letters))
            if v is None:
                sp = _spans_for(letters, t, _ent(spans[did]) if E else [])
                v = _MASK_CACHE[(did, letters)] = remove_spans(t, sp) if sp else t
        out.append(v)
    return out


def split_clusters(df, seed, test_frac):
    rng = np.random.RandomState(seed)
    cl = np.array(sorted(df["cluster_id"].unique()))
    rng.shuffle(cl)
    test = set(cl[:int(len(cl) * test_frac)])
    m = df["cluster_id"].isin(test)
    return df[~m], df[m]


def effective_n(pool, n_cfg):
    """per-class size that length-matched sampling can actually deliver; used for ALL conditions."""
    cnt = pool.groupby(["bin", "is_fake"]).size().unstack(fill_value=0)
    for c in (0, 1):
        if c not in cnt.columns:
            raise ValueError("a pool is missing a class")
    m = cnt.min(axis=1)
    total = int(m.sum())
    n = min(n_cfg, total)
    quota = np.floor(m * n / max(total, 1)).astype(int)
    return int(quota.sum()), quota


def draw(pool, n_eff, quota, length_matched, rng):
    parts = []
    if not length_matched:
        for _, g in pool.groupby("is_fake"):
            parts.append(g.sample(n_eff, random_state=rng.randint(2 ** 31 - 1)))
    else:
        for b, q in quota.items():
            if q <= 0:
                continue
            for lab in (0, 1):
                g = pool[(pool["bin"] == b) & (pool["is_fake"] == lab)]
                parts.append(g.sample(int(q), random_state=rng.randint(2 ** 31 - 1)))
    return pd.concat(parts)


def fit_eval(train_texts, y_train, test_sets):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=200000)
    X = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(X, y_train)
    res = {}
    for name, (texts, y) in test_sets.items():
        Xt = vec.transform(texts)
        p = clf.predict_proba(Xt)[:, 1]
        pred = (p >= 0.5).astype(int)
        res[name] = dict(f1=f1_score(y, pred), acc=accuracy_score(y, pred),
                         ap=average_precision_score(y, p), auc=roc_auc_score(y, p))
    return res


# --------------------------------------------------------------------------------------------------------------
def run(cfg=CFG, smoke=False, conditions=None):
    cfg = dict(cfg)
    cfg["analysis_csv"] = str(repo_path(cfg["analysis_csv"], default_dir=REPO_ROOT))
    cfg["spans_cache"] = str(repo_path(cfg["spans_cache"], default_dir=REPO_ROOT))
    cfg["out_csv"] = str(repo_path(cfg["out_csv"], default_dir=REPO_ROOT))
    if smoke:
        default_seeds, default_out = cfg["seeds"] == CFG["seeds"], cfg["out_csv"] == CFG["out_csv"]
        cfg.update(universe_per_class=300, n_train=150, n_test_in=60, n_test_x=100, spans_cache=str(repo_path("cache/spans_cache_smoke.pkl")))
        if default_seeds:
            cfg["seeds"] = [0, 1]
        if default_out:
            cfg["out_csv"] = str(repo_path("results/sensitivity/tfidf/results_tfidf_smoke.csv"))
    out_path = pathlib.Path(cfg["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u = load_universe(cfg)
    cache = compute_spans(u, cfg)
    spans = cache["spans"]
    datasets = sorted(u["dataset"].unique())
    conds = conditions or list(CONDITIONS)
    rows = []
    config_path = out_path.with_name(out_path.stem + "_config.json")
    methods_path = out_path.with_name(out_path.stem + "_methods.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in cfg.items()}, fh, indent=2)
    with open(methods_path, "w", encoding="utf-8") as fh:
        json.dump({"entity_method": cache["method"], "source_patterns": S_NAMES}, fh, indent=2)

    for seed in cfg["seeds"]:
        t0 = time.time()
        _RCACHE.clear()
        pools = {}
        for ds in datasets:
            d = u[u["dataset"] == ds]
            tr, te = split_clusters(d, seed, cfg["test_frac"])
            n_tr, q_tr = effective_n(tr, cfg["n_train"])
            n_in, q_in = effective_n(te, cfg["n_test_in"])
            n_x, q_x = effective_n(d, cfg["n_test_x"])
            pools[ds] = dict(tr=(tr, n_tr, q_tr), in_=(te, n_in, q_in), x=(d, n_x, q_x))
            if seed == cfg["seeds"][0]:
                print(f"[sizes] {ds}: train {n_tr}/class | in-domain test {n_in}/class | cross test {n_x}/class")
                for lab, got, want in (("train", n_tr, cfg["n_train"]), ("in-domain test", n_in, cfg["n_test_in"]), ("cross test", n_x, cfg["n_test_x"])):
                    if got < 0.5 * want:
                        print(f"   !! {ds} {lab}: length-matching leaves only {got}/class (wanted {want}). Because sizes are held constant "
                              f"across ALL conditions, this shrinks every condition for {ds}. Consider fewer length bins (CFG['length_bins']).")
        for cond in conds:
            S, L, E, T, R = CONDITIONS[cond]
            for tr_ds in datasets:
                rng = np.random.RandomState(zlib.crc32(f"{seed}-{cond}-{tr_ds}".encode()) % (2 ** 31))
                pool, n_eff, q = pools[tr_ds]["tr"]
                trs = draw(pool, n_eff, q, L, rng)
                tests = {}
                for te_ds in datasets:
                    role = "in_" if te_ds == tr_ds else "x"
                    p, n2, q2 = pools[te_ds][role]
                    ts = draw(p, n2, q2, L, rng)
                    tests[te_ds] = (make_texts(ts, cond, spans, seed), ts["is_fake"].to_numpy())
                res = fit_eval(make_texts(trs, cond, spans, seed), trs["is_fake"].to_numpy(), tests)
                for te_ds, m in res.items():
                    rows.append(dict(seed=seed, model="tfidf_lr", cond=cond, S=S, L=L, E=E, T=T, rand_of=R or "",
                                     train=tr_ds, test=te_ds, kind="in" if te_ds == tr_ds else "cross",
                                     n_train=len(trs), n_test=len(tests[te_ds][1]), **m))
        pd.DataFrame(rows).to_csv(cfg["out_csv"], index=False)
        print(f"seed {seed} done in {time.time() - t0:.0f}s -> {cfg['out_csv']}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------
def diagnose(cfg=CFG, after=False):
    u = load_universe(cfg)
    if after:
        print("== AFTER S+E masking: what is STILL predictive (needs `spans` to have been run) ==")
        cache = pickle.load(open(cfg["spans_cache"], "rb"))
        print(f"entity method used in cache: {cache['method']}")
        u = u[u["doc_id"].isin(cache["spans"])].copy()
        u["text"] = make_texts(u, "SE", cache["spans"], 0)
        for ds, g in u.groupby("dataset"):
            vec = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.5, sublinear_tf=True, max_features=50000)
            X = vec.fit_transform(g["text"])
            clf = LogisticRegression(max_iter=1000, solver="liblinear").fit(X, g["is_fake"])
            names = np.array(vec.get_feature_names_out()); co = clf.coef_[0]; o = np.argsort(co)
            print(f"\n[{ds}] -> real: {', '.join(names[o[:30]])}\n[{ds}] -> fake: {', '.join(names[o[::-1][:30]])}")
        return
    print("== S-pattern coverage: % of docs matched, by dataset and class (is_fake 0=real, 1=fake) ==")
    rows = []
    for (ds, lab), g in u.groupby(["dataset", "is_fake"]):
        row = {"dataset": ds, "is_fake": lab, "n": len(g)}
        for name, rx in zip(S_NAMES, _S_RES):
            row[name] = round(100 * g["text"].map(lambda t: bool(rx.search(t))).mean(), 1)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== length (words) by dataset and class ==")
    print(u.groupby(["dataset", "is_fake"])["n_words"].describe()[["mean", "50%", "min", "max"]].round(0).to_string())
    print("\n== most label-predictive unigrams per dataset (positive -> fake, negative -> real). Look for SOURCE tokens ==")
    for ds, g in u.groupby("dataset"):
        vec = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.5, sublinear_tf=True, max_features=50000)
        X = vec.fit_transform(g["text"])
        clf = LogisticRegression(max_iter=1000, solver="liblinear").fit(X, g["is_fake"])
        names = np.array(vec.get_feature_names_out()); co = clf.coef_[0]
        o = np.argsort(co)
        print(f"\n[{ds}] -> real: {', '.join(names[o[:25]])}\n[{ds}] -> fake: {', '.join(names[o[::-1][:25]])}")
    print("\nAdd any source/publisher-like tokens you see to EXTRA_SOURCE_TERMS at the top of exp_tfidf.py, then delete spans_cache.pkl.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "spans", "run"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--after", action="store_true", help="with diagnose: show predictive tokens left after S+E masking")
    ap.add_argument("--seeds", type=int, nargs="+", help="run only these seeds (e.g. --seeds 0 1) so seeds can run in parallel terminals")
    ap.add_argument("--conds", nargs="+", help="run only these conditions (e.g. --conds S_W S_B R_W R_B) -- use with --tag")
    ap.add_argument("--exclude", help="text file of doc_ids to drop from the sampling universe (see drop_flagged.py)")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. --tag a -> results_tfidf_a.csv")
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose(after=a.after)
    elif a.cmd == "spans":
        compute_spans(load_universe())
    else:
        c = dict(CFG)
        if a.seeds:
            c["seeds"] = a.seeds
        if a.tag:
            c["out_csv"] = f"results_tfidf_{a.tag}.csv"
        if a.exclude:
            c["exclude_file"] = a.exclude
        run(cfg=c, smoke=a.smoke, conditions=a.conds)
