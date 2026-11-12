"""
exp_transformer.py -- multi-GPU, resumable runner for the ERM experiments (DistilBERT, RoBERTa, BERT, and TF-IDF).

WHY THIS DESIGN
  One fine-tune here is small (about 3,000 training documents, 256 tokens, 2 epochs), so splitting a single run across two
  GPUs with DDP buys little and adds synchronisation overhead.  The experiment is a grid of independent jobs
  (seed x condition x training dataset), so we parallelise at the JOB level instead:
      parent process  -> builds the list of unfinished jobs, starts one worker per GPU, merges the shards at the end
      worker r        -> sees exactly one GPU (CUDA_VISIBLE_DEVICES=<gpu r>), runs jobs[r::n_workers], appends to its own shard CSV
  Default --sampling legacy preserves exp_distilbert.py / exp_tfidf.py draws.
  --sampling paired uses saved, condition-independent document selections and
  separate result/provenance files; its cells must not be mixed with archived ones.
  Resumable: finished cells are read back from the output and shard files, so a Kaggle session that dies (12 h limit) loses at most
  the fine-tunes that were in flight.

COMMANDS
  python exp_transformer.py prepare  [--analysis-csv ... --spans-cache ...]        # NER spans for new documents (needs spaCy)
  python exp_transformer.py run --model roberta-base --gpus 0 1 --seeds 0 1 2 3 4 5 6 7
  python exp_transformer.py run --backend tfidf --workers 4 --conds C0 L_RSE ...     # CPU arm through the same pipeline
  python exp_transformer.py run --smoke --backend tfidf --workers 2                  # pipeline check without a GPU

NEW CONDITIONS (registered here, exp_tfidf.py is untouched)
  L_RE   length matching + random deletion of as many words as entity masking removes   (matched control for L+E)
  L_RSE  length matching + random deletion of as many words as S+E masking removes      (matched control for S+L+E)
  These supply the length-matched masking controls.

DATASET ORDER
  The three original corpora keep their original relative order, and any new dataset is placed after them, so the random
  streams of the original 3x3 cells are unchanged (exact parity with the published samples).

TEMPORAL MODE (--split temporal --date-col date)
  Train on documents before a date cutoff, test in-domain on later documents (clusters that straddle the cutoff are dropped
  from the test side), and compare that temporal gap with the usual cross-dataset gap.  Needs a `date` column in the analysis
  CSV (add_dataset.py keeps it).  The split is deterministic, so seeds then vary only the sample and the training.
"""
import argparse, gc, json, os, pathlib, platform, subprocess, sys, time, zlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp_tfidf as X                                                            # noqa: E402
import paired_sampling as P                                                     # noqa: E402

EXTRA_CONDITIONS = {"L_RE": (0, 1, 0, 0, "E"), "L_RSE": (0, 1, 0, 0, "SE")}
X.CONDITIONS.update(EXTRA_CONDITIONS)

PRIMARY_CONDS = ["C0", "S", "L", "E", "SL", "SE", "LE", "SLE", "R_S", "R_E", "R_SE", "L_RE", "L_RSE"]
LEGACY_ORDER = ["ISOT", "McIntire", "WELFake_residual"]
ALIASES = {"distilbert-base-uncased": "distilbert", "roberta-base": "roberta", "bert-base-uncased": "bert"}
KEY = ["seed", "model", "cond", "train", "test"]
REPO = X.REPO_ROOT


# ------------------------------------------------------------------------------------------------ shared helpers
def model_name(a):
    return "tfidf_lr" if a.backend == "tfidf" else ALIASES.get(a.model, a.model.replace("/", "_"))


def default_out(a):
    alias = model_name(a)
    suffix = "_temporal" if a.split == "temporal" else ""
    if a.sampling == "paired":
        return str(REPO / "results" / "paired_validation" / alias / f"results_{alias}{suffix}.csv")
    return str(REPO / "results" / "raw" / alias / f"results_{alias}{suffix}.csv")


def build_cfg(a):
    cfg = dict(X.CFG)
    cfg["analysis_csv"] = a.analysis_csv or cfg["analysis_csv"]
    cfg["spans_cache"] = a.spans_cache or cfg["spans_cache"]
    if a.exclude:
        cfg["exclude_file"] = a.exclude
    if a.smoke:
        cfg.update(universe_per_class=300, n_train=150, n_test_in=60, n_test_x=100)
        if not a.spans_cache:
            cfg["spans_cache"] = str(X.repo_path("cache/spans_cache_smoke.pkl"))
    return cfg


def order_datasets(names, keep=None):
    names = [n for n in names if (keep is None or n in keep)]
    return [n for n in LEGACY_ORDER if n in names] + sorted(n for n in names if n not in LEGACY_ORDER)


def split_temporal(d, date_col, q):
    dt = pd.to_datetime(d[date_col], errors="coerce", utc=True)
    if dt.isna().mean() > 0.2:
        raise ValueError(f"temporal split: more than 20% of '{date_col}' is missing/unparseable for one dataset; "
                         f"restrict the run with --datasets to dated corpora")
    d, ns = d[dt.notna()], dt[dt.notna()].astype("int64").to_numpy()
    cut = np.quantile(ns, q)
    tr, te = d[ns < cut], d[ns >= cut]
    return tr, te[~te["cluster_id"].isin(set(tr["cluster_id"]))]


def build_pools(u, datasets, seed, cfg, a):
    pools = {}
    for ds in datasets:
        d = u[u["dataset"] == ds]
        tr, te = (split_temporal(d, a.date_col, a.cutoff_quantile) if a.split == "temporal"
                  else X.split_clusters(d, seed, cfg["test_frac"]))
        try:
            n_tr, q_tr = X.effective_n(tr, cfg["n_train"])
            n_in, q_in = X.effective_n(te, cfg["n_test_in"])
        except ValueError as e:
            raise ValueError(f"{ds}: {e} (seed {seed}, split={a.split}); check class balance on both sides of the split") from e
        n_x, q_x = X.effective_n(d, cfg["n_test_x"])
        pools[ds] = dict(tr=(tr, n_tr, q_tr), in_=(te, n_in, q_in), x=(d, n_x, q_x))
    return pools


def read_rows(paths):
    frames = [pd.read_csv(p) for p in paths if pathlib.Path(p).exists() and pathlib.Path(p).stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_outputs(out, shards):
    df = read_rows([out] + list(shards))
    if df.empty:
        return df
    df = df.drop_duplicates(KEY, keep="last").sort_values(["seed", "cond", "train", "test"]).reset_index(drop=True)
    tmp = pathlib.Path(str(out) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)                                                    # atomic: a crash never leaves a half-written file
    return df


def detect_gpus():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        n = sum(1 for line in r.stdout.splitlines() if line.startswith("GPU "))
        if n:
            return list(range(n))
    except Exception:                                                       # noqa
        pass
    return []


# ------------------------------------------------------------------------------------------------ transformer arm
def fit_predict_transformer(train_texts, y_train, test_sets, a, seed):
    """Same training recipe as exp_distilbert.fit_predict_bert (kept identical so DistilBERT cells stay comparable)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")

    def batches(texts, y=None, shuffle=False):
        idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
        for i in range(0, len(idx), a.bs):
            j = idx[i:i + a.bs]
            b = enc([texts[k] for k in j])
            if y is not None:
                b["labels"] = torch.tensor([int(y[k]) for k in j])
            yield {k: v.to(dev) for k, v in b.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(train_texts) / a.bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    model.train()
    for _ in range(a.epochs):
        for b in batches(list(train_texts), np.asarray(y_train), shuffle=True):
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = model(**b).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
    model.eval(); res = {}
    with torch.no_grad():
        for name, (texts, y) in test_sets.items():
            probs = []
            for b in batches(list(texts)):
                with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    lg = model(**b).logits
                probs.append(torch.softmax(lg.float(), dim=-1)[:, 1].cpu().numpy())
            p = np.concatenate(probs); pred = (p >= 0.5).astype(int)
            res[name] = dict(f1=f1_score(y, pred), acc=accuracy_score(y, pred),
                             ap=average_precision_score(y, p), auc=roc_auc_score(y, p))
    del model, opt, sch, scaler
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------ worker
def worker_main(plan_path, rank):
    import types, pickle
    plan = json.loads(pathlib.Path(plan_path).read_text())
    a = types.SimpleNamespace(**plan["args"])
    cfg, datasets = plan["cfg"], plan["datasets"]
    jobs = [tuple(j) for j in plan["jobs"]][rank::plan["n_workers"]]
    shard = pathlib.Path(plan["shards"][rank])
    tag = f"[w{rank}|gpu{plan['gpus'][rank] if plan['gpus'] else '-'}]"
    if a.backend == "transformer":
        import torch
        print(f"{tag} device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU visible!)'}", flush=True)
        if not torch.cuda.is_available() and not a.allow_cpu:
            sys.exit(f"{tag} no CUDA device visible; pass --allow-cpu to run on CPU")
    u = X.load_universe(cfg)
    if a.sampling == "paired":
        u = u[u["dataset"].isin(datasets)].copy()
    cache = pickle.load(open(cfg["spans_cache"], "rb"))
    spans = cache["spans"]
    paired = a.sampling == "paired"
    if paired:
        P.validate_universe(u)
        P.validate_cache(cache, u, a.allow_regex_ner)
        indexed = u.set_index("doc_id", drop=False)
    missing = u.loc[~u["doc_id"].isin(spans), "doc_id"]
    if len(missing):
        sys.exit(f"{tag} {len(missing):,} universe documents have no cached spans -> run `exp_transformer.py prepare` first")
    print(f"{tag} {len(jobs)} jobs", flush=True)
    pools, cur = None, None
    for n, (seed, cond, tr_ds) in enumerate(jobs, 1):
        if seed != cur:                                                     # random-control cache is keyed without the seed
            X._RCACHE.clear(); pools, cur = build_pools(u, datasets, seed, cfg, a), seed
            if paired:
                manifest_info = plan["manifests"][str(seed)]
                manifest = P.read_manifest(manifest_info["path"], manifest_info["sha256"])
        t0 = time.time()
        S, L, E, T, R = X.CONDITIONS[cond]
        if paired:
            trs, train_hash = P.select_sample(indexed, manifest, tr_ds, "tr", L)
        else:
            rng = np.random.RandomState(zlib.crc32(f"{seed}-{cond}-{tr_ds}".encode()) % (2 ** 31))
            pool, n_eff, q = pools[tr_ds]["tr"]
            trs = X.draw(pool, n_eff, q, L, rng)
        tests = {}
        test_hashes = {}
        transform = P.make_texts if paired else X.make_texts
        for te_ds in datasets:                                              # same draw order as the original runners
            role = "in_" if te_ds == tr_ds else "x"
            if paired:
                ts, test_hashes[te_ds] = P.select_sample(indexed, manifest, te_ds, role, L)
            else:
                p, n2, q2 = pools[te_ds][role]
                ts = X.draw(p, n2, q2, L, rng)
            tests[te_ds] = (transform(ts, cond, spans, seed), ts["is_fake"].to_numpy())
        tr_texts, y_tr = transform(trs, cond, spans, seed), trs["is_fake"].to_numpy()
        res = X.fit_eval(tr_texts, y_tr, tests) if a.backend == "tfidf" else fit_predict_transformer(tr_texts, y_tr, tests, a, seed)
        rows = []
        for te_ds, m in res.items():
            row = dict(seed=seed, model=plan["model_name"], cond=cond, S=S, L=L, E=E, T=T, rand_of=R or "", train=tr_ds,
                       test=te_ds, kind="in" if te_ds == tr_ds else "cross", n_train=len(trs), n_test=len(tests[te_ds][1]), **m)
            if a.split == "temporal":
                row["split"] = "temporal"
            if paired:
                row.update(sampling_protocol=P.PROTOCOL, train_sample_sha256=train_hash,
                           test_sample_sha256=test_hashes[te_ds], manifest_sha256=manifest_info["sha256"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(shard, mode="a", header=not shard.exists(), index=False)   # one append per finished job
        print(f"{tag} {n}/{len(jobs)} seed {seed} {cond:6s} train={tr_ds:18s} {time.time() - t0:5.0f}s", flush=True)
    print(f"{tag} finished", flush=True)


# ------------------------------------------------------------------------------------------------ parent
def run_main(a):
    cfg = build_cfg(a)
    mname = model_name(a)
    out = pathlib.Path(a.out or default_out(a)); out.parent.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.seeds, a.conds = [0], ["C0", "LE", "L_RSE"]
        out = out.with_name(out.stem + "_smoke.csv")
        a.epochs = 1
    u = X.load_universe(cfg)
    if a.datasets and set(a.datasets) - set(u["dataset"].unique()):
        sys.exit(f"requested datasets absent from analysis CSV: {sorted(set(a.datasets) - set(u['dataset'].unique()))}")
    datasets = order_datasets(u["dataset"].unique(), a.datasets)
    if a.sampling == "paired":
        u = u[u["dataset"].isin(datasets)].copy()
    if not datasets:
        sys.exit("no datasets selected")
    if a.workers < 1 or not a.seeds or len(set(a.seeds)) != len(a.seeds) or len(set(a.conds)) != len(a.conds):
        sys.exit("workers must be positive; seeds and conditions must be unique")
    unknown = [c for c in a.conds if c not in X.CONDITIONS]
    if unknown:
        sys.exit(f"unknown conditions {unknown}; available: {list(X.CONDITIONS)}")
    if a.backend == "transformer":
        gpus = a.gpus if a.gpus is not None else detect_gpus()
        if len(set(gpus)) != len(gpus):
            sys.exit("GPU indices must be unique (one worker per GPU)")
        if not gpus and not a.allow_cpu:
            sys.exit("no GPU found (nvidia-smi); pass --gpus 0 1, or --allow-cpu for a (very slow) CPU run")
        n_workers = max(len(gpus), 1)
    else:
        gpus, n_workers = [], a.workers
    work = out.parent / (out.stem + "_work"); work.mkdir(exist_ok=True)
    shards = [str(work / f"shard{r}.csv") for r in range(n_workers)]
    # Retain shards from a session with a different worker count.
    all_shards = sorted(set(shards) | {str(p) for p in work.glob("shard*.csv")})
    prior = read_rows([out] + all_shards)
    paired = a.sampling == "paired"
    manifests = {}
    if paired:
        import pickle
        import importlib.metadata
        P.validate_universe(u)
        with open(cfg["spans_cache"], "rb") as f:
            entity_method = P.validate_cache(pickle.load(f), u, a.allow_regex_ner)
        signature = dict(protocol=P.PROTOCOL, datasets=datasets,
                         analysis_sha256=P.file_hash(cfg["analysis_csv"]),
                         spans_sha256=P.file_hash(cfg["spans_cache"]),
                         exclusion_sha256=P.file_hash(a.exclude) if a.exclude else None,
                         cfg={k: v for k, v in cfg.items() if k not in
                              {"analysis_csv", "spans_cache", "out_csv", "exclude_file", "seeds"}},
                         split=a.split, date_col=a.date_col, cutoff_quantile=a.cutoff_quantile,
                         backend=a.backend, model=a.model if a.backend == "transformer" else "tfidf_lr",
                         max_len=a.max_len, epochs=a.epochs, bs=a.bs, lr=a.lr, entity_method=entity_method,
                         libraries={name: importlib.metadata.version(name) for name in
                                    (["numpy", "pandas", "scikit-learn", "scipy"] +
                                     (["torch", "transformers", "tokenizers"] if a.backend == "transformer" else []))},
                         code_sha256={name: P.file_hash(pathlib.Path(__file__).with_name(name)) for name in
                                      ["exp_transformer.py", "exp_tfidf.py", "paired_sampling.py"]})
        guard = out.with_name(out.stem + "_protocol.json")
        if guard.exists():
            if json.loads(guard.read_text()) != signature:
                sys.exit("paired run inputs/settings/code changed; use a new output path to avoid mixing experiments")
        elif not prior.empty:
            sys.exit("existing results have no paired protocol record; use a new output path")
        else:
            P.atomic_json(guard, signature)
        manifest_dir = out.with_name(out.stem + "_samples")
        for seed in a.seeds:
            manifest = P.make_manifest(build_pools(u, datasets, seed, cfg, a), seed)
            path = manifest_dir / f"seed_{seed}.json"
            P.save_manifest(path, manifest)
            manifests[str(seed)] = dict(path=str(path.resolve()), sha256=manifest["manifest_sha256"])
    done = set()
    if paired:
        if not prior.empty and ("sampling_protocol" not in prior or
                                not prior.sampling_protocol.eq(P.PROTOCOL).all()):
            sys.exit("output contains legacy/different sampling protocol; use a new output path")
        done = P.complete_jobs(prior.drop_duplicates(KEY, keep="last") if not prior.empty else prior,
                               mname, datasets, P.PROTOCOL)
    elif not prior.empty:
        if "sampling_protocol" in prior.columns:
            sys.exit("paired results cannot be resumed with legacy sampling; use --sampling paired")
        p = prior[prior["model"] == mname]
        done = set(zip(p["seed"], p["cond"], p["train"]))
    jobs = [[s, c, d] for s in a.seeds for c in a.conds for d in datasets if (s, c, d) not in done]
    print(f"model={mname} backend={a.backend} sampling={a.sampling} datasets={datasets}")
    print(f"{len(done)} cells already done, {len(jobs)} to run on {n_workers} worker(s) (gpus={gpus or 'cpu'}) -> {out}")
    if not jobs:
        merge_outputs(out, all_shards); return
    plan = dict(args=vars(a), cfg=cfg, datasets=datasets, jobs=jobs, n_workers=n_workers, gpus=gpus,
                shards=shards, model_name=mname, manifests=manifests)
    plan_path = work / "plan.json"; plan_path.write_text(json.dumps(plan))
    meta = dict(args=vars(a), cfg=cfg, datasets=datasets, python=platform.python_version(), host=platform.node())
    if paired:
        import importlib.metadata
        meta.update(sampling_protocol=P.PROTOCOL, manifests=manifests,
                    libraries={name: importlib.metadata.version(name) for name in
                               ["numpy", "pandas", "scikit-learn", "scipy"]})
    try:
        import torch, transformers
        meta.update(torch=torch.__version__, transformers=transformers.__version__)
    except Exception:                                                       # noqa
        pass
    (out.with_name(out.stem + "_config.json")).write_text(json.dumps(meta, indent=2, default=str))

    procs = []
    for r in range(n_workers):
        env = os.environ.copy()
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[r])                      # each worker sees exactly one T4 as cuda:0
        env["TOKENIZERS_PARALLELISM"] = "false"
        procs.append(subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve()), "worker",
                                       "--plan", str(plan_path), "--rank", str(r)], env=env))
    codes = [p.wait() for p in procs]
    df = merge_outputs(out, all_shards)
    for s in all_shards:
        pathlib.Path(s).unlink(missing_ok=True)
    print(f"worker exit codes {codes}; {len(df)} rows in {out}")
    if any(codes):
        sys.exit("a worker failed; finished cells are saved - rerun the same command to resume")
    plan_path.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass


def prepare_main(a):
    import pickle
    cfg = build_cfg(a)
    u = X.load_universe(cfg)
    ef = X.EntityFinder()
    if ef.method.startswith("regex") and not a.allow_regex_ner:
        sys.exit("spaCy en_core_web_sm is not available: the entity masking would silently change to the regex fallback "
                 "and the cache would be rebuilt. Install it (python -m spacy download en_core_web_sm) or pass --allow-regex-ner.")
    X.compute_spans(u, cfg)
    spans = pickle.load(open(cfg["spans_cache"], "rb"))["spans"]
    print("universe documents:", len(u), "| without spans:", int((~u["doc_id"].isin(spans)).sum()))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--analysis-csv", help="default: data/processed/analysis_set.csv (on Kaggle: /kaggle/input/<dataset>/...)")
    common.add_argument("--spans-cache", help="default: cache/spans_cache.pkl")
    common.add_argument("--exclude", help="doc_id exclusion file (data/audit/flagged_doc_ids.txt for the duplicate-removal rerun)")
    common.add_argument("--smoke", action="store_true")
    pr = sub.add_parser("prepare", parents=[common])
    pr.add_argument("--allow-regex-ner", action="store_true")
    ru = sub.add_parser("run", parents=[common])
    ru.add_argument("--backend", default="transformer", choices=["transformer", "tfidf"])
    ru.add_argument("--sampling", choices=["legacy", "paired"], default="legacy",
                    help="legacy preserves archived draws; paired fixes documents across masking conditions")
    ru.add_argument("--allow-regex-ner", action="store_true", help="synthetic checks only; paired runs require spaCy spans by default")
    ru.add_argument("--model", default="distilbert-base-uncased", help="distilbert-base-uncased | roberta-base | bert-base-uncased")
    ru.add_argument("--gpus", type=int, nargs="+", help="GPU indices, one worker each (default: all that nvidia-smi lists)")
    ru.add_argument("--workers", type=int, default=2, help="CPU workers for --backend tfidf")
    ru.add_argument("--allow-cpu", action="store_true")
    ru.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ru.add_argument("--conds", nargs="+", default=PRIMARY_CONDS)
    ru.add_argument("--datasets", nargs="+", help="restrict to these datasets (default: all in the analysis CSV)")
    ru.add_argument("--out")
    ru.add_argument("--max_len", type=int, default=256)
    ru.add_argument("--epochs", type=int, default=2)
    ru.add_argument("--bs", type=int, default=16)
    ru.add_argument("--lr", type=float, default=3e-5)
    ru.add_argument("--split", default="cluster", choices=["cluster", "temporal"])
    ru.add_argument("--date-col", default="date")
    ru.add_argument("--cutoff-quantile", type=float, default=0.7, help="temporal: train on the oldest 70%% of documents")
    wk = sub.add_parser("worker")
    wk.add_argument("--plan", required=True)
    wk.add_argument("--rank", type=int, required=True)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.cmd == "worker":
        worker_main(args.plan, args.rank)
    elif args.cmd == "prepare":
        prepare_main(args)
    else:
        run_main(args)
