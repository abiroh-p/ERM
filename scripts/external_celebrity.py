"""Separate Celebrity extension. Audit/select/NER/plan/run/check/analyze are explicit stages.

No source training or target scoring occurs during preparation. Run one process
per rank (and per GPU for transformers); global job indexes never change on resume.
"""
import argparse
import importlib.metadata
import json
import os
import pathlib
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/experiments"), str(ROOT / "src/audit")]
import exp_transformer as T
import paired_sampling as P
import celebrity_audit as A
X = T.X


def freeze(path, value):
    """Only invoked in serial preparation/plan stages, before parallel workers."""
    path = pathlib.Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"frozen record changed: {path}; use a new version/directory")
    else:
        P.atomic_json(path, value)


def config(a):
    return json.loads(pathlib.Path(a.config).read_text())


def verify_inputs(a, require_complete=True):
    c = config(a)
    if P.file_hash(a.zip) != c["zip_sha256"] or P.file_hash(a.analysis_csv) != c["analysis_sha256"]:
        raise ValueError("frozen input hash mismatch")
    audit = pathlib.Path(a.audit)
    summary = json.loads((audit / "audit_summary.json").read_text())
    if summary["zip_sha256"] != c["zip_sha256"] or summary["config_sha256"] != P.file_hash(a.config):
        raise ValueError("audit input/config differs")
    if require_complete and (not summary["cross_corpus_complete"] or summary["analysis_sha256"] != c["analysis_sha256"]):
        raise ValueError("cross-corpus audit incomplete: evaluation is blocked")
    return c


def source_universe(a):
    c = dict(X.CFG, analysis_csv=a.analysis_csv)
    u = X.load_universe(c)
    if set(u.dataset) != set(config(a)["sources"]):
        raise ValueError("source corpus grid differs")
    P.validate_universe(u)
    return u, c


def target_frame(a):
    rows = A.read_articles(a.zip)
    df = pd.DataFrame(rows)
    eligibility = pd.read_csv(pathlib.Path(a.audit) / "eligibility.csv", keep_default_na=False)
    keep = eligibility.loc[~eligibility.excluded, "article_id"]
    df = df[df.doc_id.isin(keep)].copy().sort_values("doc_id").reset_index(drop=True)
    df["cluster_id"] = df.story_group
    if set(df.is_fake) != {0, 1}:
        raise ValueError("eligible target needs both labels")
    return df


def target_manifest(df, c):
    """Primary: every eligible article. Secondary: one frozen labelled length draw."""
    df = df.copy()
    edges = np.unique(np.quantile(df.n_words, np.linspace(0, 1, c["target_length_bins"] + 1)))
    df["bin"] = np.clip(np.searchsorted(edges, df.n_words, side="right")-1, 0, max(len(edges)-2, 0))
    n, quota = X.effective_n(df, int(df.is_fake.value_counts().min()))
    rng_key = [c["protocol"], "target-secondary", c["target_selection_seed"]]
    rng = np.random.RandomState(int(P.digest(rng_key)[:8], 16))
    if n < 1:
        raise ValueError("no secondary length-matched target selection available")
    secondary = X.draw(df, n, quota, True, rng)
    def sample(d):
        rows = [[r.doc_id, int(r.is_fake), r.story_group] for r in d.itertuples()]
        return dict(rows=rows, sample_sha256=P.digest(rows))
    return dict(protocol=c["protocol"], primary=sample(df), secondary=sample(secondary),
                secondary_rng_key=rng_key, secondary_rng_seed=int(P.digest(rng_key)[:8], 16),
                edges=edges.tolist(), quotas={str(k): int(v) for k, v in quota.items()})


def select(a):
    c = verify_inputs(a)
    out = pathlib.Path(a.prepared)
    audit_files = ["audit_summary.json", "inventory.csv", "eligibility.csv", "internal_candidates.csv", "cross_candidates.csv"]
    binding = dict(protocol=c["protocol"], config_sha256=P.file_hash(a.config),
                   analysis_sha256=c["analysis_sha256"], zip_sha256=c["zip_sha256"],
                   audit_files={p: P.file_hash(pathlib.Path(a.audit)/p) for p in audit_files})
    freeze(out / "inputs.json", binding)
    u, cfg = source_universe(a)
    target = target_manifest(target_frame(a), c)
    freeze(out / "target_selection.json", target)
    sampling = SimpleNamespace(split="cluster", date_col="date", cutoff_quantile=0.7)
    manifests = {}
    for seed in c["seeds"]:
        m = P.make_manifest(T.build_pools(u, c["sources"], seed, cfg, sampling), seed)
        freeze(out / f"source_seed_{seed}.json", m)
        manifests[str(seed)] = m["manifest_sha256"]
    freeze(out / "selection_index.json", dict(source_manifests=manifests, target_sha256=P.digest(target)))
    print(f"Selections frozen: primary {len(target['primary']['rows'])}, secondary {len(target['secondary']['rows'])}; 8 source manifests")


def checked_selections(a):
    c = verify_inputs(a)
    prep = pathlib.Path(a.prepared)
    binding = json.loads((prep / "inputs.json").read_text())
    for name, h in binding["audit_files"].items():
        if P.file_hash(pathlib.Path(a.audit)/name) != h:
            raise ValueError("audit/eligibility changed after selection freeze")
    if binding["config_sha256"] != P.file_hash(a.config):
        raise ValueError("configuration changed after selection freeze")
    index = json.loads((prep / "selection_index.json").read_text())
    target = json.loads((prep / "target_selection.json").read_text())
    if P.digest(target) != index["target_sha256"]:
        raise ValueError("target manifest changed")
    for value in (target["primary"], target["secondary"]):
        if P.digest(value["rows"]) != value["sample_sha256"]:
            raise ValueError("target selection checksum mismatch")
    for seed in c["seeds"]:
        P.read_manifest(prep/f"source_seed_{seed}.json", index["source_manifests"][str(seed)])
    return c, target, index


def ner(a):
    c, target, index = checked_selections(a)
    u, _ = source_universe(a)
    with open(a.spans_cache, "rb") as f:
        cache = pickle.load(f)  # trusted local repository cache only
    P.validate_cache(cache, u)
    # Do not mutate the historical cache, and do not let fallback NER rebuild it.
    ef = X.EntityFinder()
    if ef.method != "spacy:en_core_web_sm":
        raise ValueError("install en_core_web_sm; no regex fallback is permitted")
    df = target_frame(a)
    spans = {did: cache["spans"][did] for did in u.doc_id}
    for did, es in zip(df.doc_id, ef.spans(df.text.tolist())):
        spans[did] = es
    out = pathlib.Path(a.prepared)
    record = dict(method=ef.method, entity_labels=X.CFG["entity_labels"],
                  source_cache_sha256=P.file_hash(a.spans_cache), target_selection_sha256=P.digest(target),
                  spacy_version=importlib.metadata.version("spacy"), model_meta=ef.nlp.meta,
                  spans_sha256=P.digest({k: X._ent(v) for k, v in sorted(spans.items())}))
    freeze(out / "ner_provenance.json", record)
    dest = out / "spans.pkl"
    if dest.exists():
        with open(dest, "rb") as f:
            old = pickle.load(f)
        if P.digest({k: X._ent(v) for k, v in sorted(old["spans"].items())}) != record["spans_sha256"]:
            raise ValueError("existing external cache differs")
    else:
        with open(dest, "xb") as f:
            pickle.dump(dict(method=ef.method, spans=spans), f)
    print("Separate source+Celebrity spans saved; original cache preserved")


def jobs(c):
    return [dict(index=i, seed=s, cond=k, source=d) for i, (s, k, d) in enumerate(
        (s, k, d) for s in c["seeds"] for k in c["conditions"] for d in c["sources"])]


def signature(a):
    c, target, index = checked_selections(a)
    files = ["scripts/external_celebrity.py", "src/audit/celebrity_audit.py", "src/experiments/exp_tfidf.py",
             "src/experiments/paired_sampling.py", "src/experiments/exp_transformer.py", "docs/EXTERNAL_CELEBRITY_PROTOCOL_v1.md"]
    libs = ["numpy", "pandas", "scikit-learn", "scipy"]
    if a.model != "tfidf":
        libs += ["torch", "transformers", "tokenizers"]
    return dict(protocol=c["protocol"], model=a.model, config_sha256=P.file_hash(a.config),
                source_manifests=index["source_manifests"], target_sha256=P.digest(target),
                inputs_sha256=P.file_hash(pathlib.Path(a.prepared)/"inputs.json"),
                spans_sha256=P.file_hash(pathlib.Path(a.prepared)/"spans.pkl"),
                ner_provenance_sha256=P.file_hash(pathlib.Path(a.prepared)/"ner_provenance.json"),
                code_sha256={f: P.file_hash(ROOT/f) for f in files}, python=sys.version,
                libraries={n: importlib.metadata.version(n) for n in libs},
                source_cfg={k:v for k,v in X.CFG.items() if k not in {"analysis_csv", "spans_cache", "out_csv", "seeds"}},
                transformer=c["transformer"] if a.model != "tfidf" else None)


def plan(a):
    sig = signature(a)
    out = pathlib.Path(a.output)/a.model
    freeze(out/"run_protocol.json", sig)
    freeze(out/"jobs.json", jobs(config(a)))
    print(f"{len(jobs(config(a)))} fits for {a.model}; plan frozen; run workers next")


def validate_result(r, job, sig, target, index):
    body = {k:v for k,v in r.items() if k != "result_sha256"}
    if P.digest(body) != r.get("result_sha256") or r.get("signature_sha256") != P.digest(sig) or r.get("job") != job:
        raise ValueError("result checksum, protocol or job mismatch")
    selection = target["secondary" if X.CONDITIONS[job["cond"]][1] else "primary"]
    expected = selection["sample_sha256"]
    if r["target_sample_sha256"] != expected or r["source_manifest_sha256"] != index["source_manifests"][str(job["seed"])]:
        raise ValueError("result selection mismatch")
    if r["model"] != sig["model"] or r["n_test"] != len(selection["rows"]) or len(r["target_word_deletions"]) != r["n_test"]:
        raise ValueError("result model or target cardinality mismatch")
    if not all(np.isfinite(r["metrics"][k]) and 0 <= r["metrics"][k] <= 1 for k in ("auc", "ap", "f1", "acc")):
        raise ValueError("non-finite/out-of-range metrics")


def load_run(a):
    c, target, index = checked_selections(a)
    out = pathlib.Path(a.output)/a.model
    sig = json.loads((out/"run_protocol.json").read_text())
    if sig != signature(a):
        raise ValueError("planned code, inputs or environment changed; make a new plan/output")
    grid = jobs(c)
    if json.loads((out/"jobs.json").read_text()) != grid:
        raise ValueError("job plan differs")
    return c, target, index, out, sig, grid


def run(a):
    c, target, index, out, sig, grid = load_run(a)
    if a.world_size < 1 or not 0 <= a.rank < a.world_size:
        raise ValueError("rank must be in [0, world-size)")
    if a.model != "tfidf":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("transformer run requires CUDA; GPU training never starts in preparation")
    u, _ = source_universe(a)
    u = u.set_index("doc_id", drop=False)
    df = target_frame(a).set_index("doc_id", drop=False)
    with open(pathlib.Path(a.prepared)/"spans.pkl", "rb") as f:
        cache = pickle.load(f)
    P.validate_cache(cache, pd.concat([u, df], ignore_index=True))
    spans = cache["spans"]
    spans_record = json.loads((pathlib.Path(a.prepared)/"ner_provenance.json").read_text())
    if P.digest({k: X._ent(v) for k, v in sorted(spans.items())}) != spans_record["spans_sha256"]:
        raise ValueError("span contents differ from NER provenance")
    dest = out/"jobs"
    dest.mkdir(exist_ok=True)
    for job in grid[a.rank::a.world_size]:
        path = dest/f"job_{job['index']:04d}.json"
        if path.exists():
            validate_result(json.loads(path.read_text()), job, sig, target, index)
            continue
        # An exclusive lock prevents two identical ranks from fitting/writing one job.
        lock = path.with_suffix(".lock")
        try:
            handle = lock.open("x")
        except FileExistsError:
            raise ValueError(f"job locked: {lock}; verify worker stopped before removing stale lock")
        try:
            with handle:
                handle.write(str(os.getpid()))
            t0 = time.time()
            seed, cond, source = job["seed"], job["cond"], job["source"]
            m = P.read_manifest(pathlib.Path(a.prepared)/f"source_seed_{seed}.json", index["source_manifests"][str(seed)])
            length = X.CONDITIONS[cond][1]
            train, tr_hash = P.select_sample(u, m, source, "tr", length)
            selection = target["secondary" if length else "primary"]
            test = df.loc[[r[0] for r in selection["rows"]]]
            actual = [[r.doc_id, int(r.is_fake), r.story_group] for r in test.itertuples()]
            if actual != selection["rows"]:
                raise ValueError("target labels/groups differ from manifest")
            X._MASK_CACHE.clear(); X._RCACHE.clear()
            train_text = P.make_texts(train, cond, spans, seed)
            test_text = P.make_texts(test, cond, spans, seed)
            tests = {"Celebrity": (test_text, test.is_fake.to_numpy())}
            # liblinear uses the global stream when random_state is omitted.
            # Reset per job so rank count and resumption cannot alter that stream.
            np.random.seed(seed)
            if a.model == "tfidf":
                res = X.fit_eval(train_text, train.is_fake.to_numpy(), tests)
            else:
                settings = SimpleNamespace(model={"distilbert":"distilbert-base-uncased", "roberta":"roberta-base"}[a.model], **c["transformer"])
                res = T.fit_predict_transformer(train_text, train.is_fake.to_numpy(), tests, settings, seed)
            removed = [len(t.split())-len(v.split()) for t,v in zip(test.text, test_text)]
            record = dict(job=job, model=a.model, test="Celebrity", metrics=res["Celebrity"],
                          signature_sha256=P.digest(sig), train_sample_sha256=tr_hash,
                          target_sample_sha256=selection["sample_sha256"], source_manifest_sha256=index["source_manifests"][str(seed)],
                          n_train=len(train), n_test=len(test), target_word_deletions=removed,
                          elapsed_seconds=time.time()-t0)
            record["result_sha256"] = P.digest(record)
            validate_result(record, job, sig, target, index)
            P.atomic_json(path, record)
            print(f"completed {job}; {record['elapsed_seconds']:.1f}s", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def collect(a):
    c, target, index, out, sig, grid = load_run(a)
    rows, missing = [], []
    for job in grid:
        path = out/"jobs"/f"job_{job['index']:04d}.json"
        if not path.exists():
            missing.append(job)
            continue
        r = json.loads(path.read_text())
        validate_result(r, job, sig, target, index)
        rows.append(dict(**job, model=a.model, **r["metrics"], n_train=r["n_train"], n_test=r["n_test"],
                         train_sample_sha256=r["train_sample_sha256"], target_sample_sha256=r["target_sample_sha256"]))
    status = dict(expected=len(grid), completed=len(rows), missing=missing, complete=not missing)
    P.atomic_json(out/"completion.json", status)
    print(f"{a.model}: {len(rows)}/{len(grid)} validated fits")
    return c, out, pd.DataFrame(rows), status


def analyze(a):
    c, out, df, status = collect(a)
    if df.empty:
        raise ValueError("no outcomes: no placeholder results will be written")
    df.to_csv(out/"completed_metrics.csv", index=False)
    contrasts = []
    for scope, comparisons in (("primary", c["primary_contrasts"]), ("secondary", c["secondary_contrasts"])):
        for intervention, control in comparisons:
            left, right = df[df.cond.eq(intervention)], df[df.cond.eq(control)]
            pairs = left.merge(right, on=["model", "seed", "source"], suffixes=("_i", "_r"), validate="one_to_one")
            for r in pairs.itertuples():
                if r.target_sample_sha256_i != r.target_sample_sha256_r or r.train_sample_sha256_i != r.train_sample_sha256_r:
                    raise ValueError("unpaired contrast")
                contrasts.append(dict(model=a.model, source=r.source, seed=r.seed, scope=scope,
                                      contrast=f"{intervention}-{control}", auc_difference=r.auc_i-r.auc_r))
    cc = pd.DataFrame(contrasts, columns=["model", "source", "seed", "scope", "contrast", "auc_difference"])
    cc.to_csv(out/"matched_contrasts.csv", index=False)
    if not cc.empty:
        cc.groupby(["model", "source", "scope", "contrast"]).auc_difference.agg(["count", "mean", "std", "min", "max"]).to_csv(out/"seed_variability.csv")
    df.groupby(["model", "source", "cond"]).auc.agg(["count", "mean", "std", "min", "max"]).to_csv(out/"auc_by_source.csv")
    P.atomic_json(out/"analysis_limits.json", dict(complete=status["complete"],
                  unit="training/random-deletion seed on fixed target articles; not independent external datasets",
                  pooling="none; each source transfer reported separately", article_uncertainty="not estimated; use frozen eligibility components for any future cluster bootstrap"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "ner", "plan", "run", "check", "analyze"])
    p.add_argument("--config", default=str(A.CONFIG))
    p.add_argument("--zip", required=True)
    p.add_argument("--analysis-csv", default="data/processed/analysis_set.csv")
    p.add_argument("--spans-cache", default="cache/spans_cache.pkl")
    p.add_argument("--audit", default="results/external_celebrity/audit_v1")
    p.add_argument("--prepared", default="results/external_celebrity/prepared_v1")
    p.add_argument("--output", default="runs/external_celebrity_v1")
    p.add_argument("--model", choices=["tfidf", "distilbert", "roberta"], default="tfidf")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    a = p.parse_args()
    if a.stage == "check":
        _, _, _, status = collect(a)
        if not status["complete"]:
            sys.exit(2)
    else:
        globals()[a.stage](a)


if __name__ == "__main__":
    main()
