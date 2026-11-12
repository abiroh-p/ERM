"""Document-paired validation helpers. Archived sampling remains in exp_tfidf.py.

Manifests contain identifiers, labels and cluster IDs, never article text. Ordinary
and length-matched samples use independent streams for each dataset and role.
"""
import hashlib
import json
import pathlib
import zlib

import numpy as np
import pandas as pd

import exp_tfidf as X

PROTOCOL = "paired-documents-v1"
ROLES = ("tr", "in_", "x")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_universe(u):
    if u["doc_id"].isna().any() or u["doc_id"].duplicated().any():
        raise ValueError("paired sampling requires globally unique, non-null doc_id values")
    if not u["doc_id"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("paired sampling requires string doc_id values")
    if u["cluster_id"].isna().any() or not set(u["is_fake"].unique()) <= {0, 1}:
        raise ValueError("paired sampling requires non-null cluster IDs and binary labels")


def make_manifest(pools, seed):
    records = []
    for dataset in sorted(pools):
        train_pool, test_pool = pools[dataset]["tr"][0], pools[dataset]["in_"][0]
        if set(train_pool.cluster_id) & set(test_pool.cluster_id):
            raise ValueError(f"{dataset}: train and in-domain test clusters overlap")
        for length_matched in (False, True):
            for role in ROLES:
                pool, n_eff, quota = pools[dataset][role]
                if n_eff <= 0:
                    raise ValueError(f"{dataset}/{role}: no balanced samples available")
                key = [PROTOCOL, int(seed), dataset, role, int(length_matched)]
                rng = np.random.RandomState(int(digest(key)[:8], 16))
                sampled = X.draw(pool, n_eff, quota, length_matched, rng)
                counts = sampled.is_fake.value_counts().to_dict()
                if counts != {0: n_eff, 1: n_eff}:
                    raise ValueError(f"{dataset}/{role}: sample is not balanced")
                if sampled.doc_id.duplicated().any():
                    raise ValueError(f"{dataset}/{role}: duplicate sampled identifiers")
                rows = [[str(d), int(y), str(c)] for d, y, c in
                        sampled[["doc_id", "is_fake", "cluster_id"]].itertuples(index=False, name=None)]
                records.append(dict(dataset=dataset, role=role, length_matched=length_matched,
                                    rows=rows, sample_sha256=digest(rows)))
    payload = dict(protocol=PROTOCOL, seed=int(seed), samples=records)
    return dict(**payload, manifest_sha256=digest(payload))


def save_manifest(path, manifest):
    path = pathlib.Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"sampling manifest changed: {path}; use a new output path")
    else:
        atomic_json(path, manifest)


def read_manifest(path, expected_sha256):
    value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    recorded = value.pop("manifest_sha256")
    if digest(value) != recorded or recorded != expected_sha256:
        raise ValueError(f"sampling manifest failed integrity check: {path}")
    return value


def select_sample(indexed_universe, manifest, dataset, role, length_matched):
    found = [r for r in manifest["samples"] if
             (r["dataset"], r["role"], r["length_matched"]) == (dataset, role, bool(length_matched))]
    if len(found) != 1:
        raise ValueError(f"missing/ambiguous sample {dataset}/{role}/{length_matched}")
    sample = found[0]
    if digest(sample["rows"]) != sample["sample_sha256"]:
        raise ValueError("sample hash mismatch")
    df = indexed_universe.loc[[r[0] for r in sample["rows"]]].copy()
    actual = [[str(d), int(y), str(c)] for d, y, c in
              df[["doc_id", "is_fake", "cluster_id"]].itertuples(index=False, name=None)]
    if actual != sample["rows"]:
        raise ValueError("sample labels/clusters do not match the manifest")
    return df, sample["sample_sha256"]


def make_texts(df, cond, spans, seed):
    """Exact whole-word-count control, including targeted deletion of all words.

    The legacy random-drop function keeps at least one word. Here an all-word
    deletion also yields empty text in the control. Deterministic masks are unchanged.
    """
    random_of = X.CONDITIONS[cond][4]
    if not random_of:
        return X.make_texts(df, cond, spans, seed)
    out = []
    for did, text in zip(df.doc_id, df.text):
        real_spans = X._spans_for(random_of, text, X._ent(spans[did]))
        masked = X.remove_spans(text, real_spans)
        words = text.split()
        removed = len(words) - len(masked.split())
        if not 0 <= removed <= len(words):
            raise ValueError(f"invalid deletion count for {did}/{random_of}")
        rng = np.random.RandomState(zlib.crc32(f"{seed}-{did}-{random_of}".encode()) % (2 ** 31))
        if removed == len(words):
            result = ""
        else:
            result = X.random_drop_words(text, removed, rng)
        if len(result.split()) != len(masked.split()):
            raise ValueError(f"random deletion count mismatch for {did}/{random_of}")
        out.append(result)
    return out


def validate_cache(cache, u, allow_regex=False):
    method = cache.get("method", "")
    if method != "spacy:en_core_web_sm" and not allow_regex:
        raise ValueError(f"paired publication runs require spaCy en_core_web_sm spans; found {method!r}. "
                         "Use --allow-regex-ner only for synthetic checks.")
    missing = set(u.doc_id) - set(cache.get("spans", {}))
    if missing:
        raise ValueError(f"{len(missing)} universe documents have no cached spans; run prepare first")
    return method


def complete_jobs(prior, model, datasets, protocol):
    if prior.empty:
        return set()
    if "sampling_protocol" not in prior or not prior.sampling_protocol.eq(protocol).all():
        raise ValueError("output contains a different/legacy sampling protocol; use a new output path")
    if not prior.model.eq(model).all():
        raise ValueError("output contains another model; use a separate output path")
    if prior.duplicated(["seed", "cond", "train", "test"]).any():
        raise ValueError("output contains duplicate result cells")
    done = set()
    for key, group in prior.groupby(["seed", "cond", "train"]):
        if set(group.test) == set(datasets) and len(group) == len(datasets):
            done.add(key)
    return done
