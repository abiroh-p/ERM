"""
exp_distilbert.py -- DistilBERT arm. Designed for Google Colab (free T4 GPU).

Uses EXACTLY the same universe, splits, sampling, masking and random controls as exp_tfidf.py (it imports it), so each
(seed, condition, train dataset, test dataset) cell is a paired comparison with the TF-IDF result. Writes a CSV with the
same columns (model = 'distilbert'). Resumable: finished (seed, cond, train) cells in --out are skipped on restart.

Colab setup (one cell):
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/drive/MyDrive/ERM          # folder holding: analysis_set.csv, spans_cache.pkl, exp_tfidf.py, exp_distilbert.py
    !pip -q install transformers
Quick check (about 3 minutes):   !python exp_distilbert.py --smoke
Real run (~2 min per fine-tune):  !python exp_distilbert.py --seeds 0 1 2 --out results_distilbert.csv

Needs spans_cache.pkl produced locally by `exp_tfidf.py spans` (upload it). spaCy is NOT needed on Colab.

IMPORTANT interpretation note: DistilBERT only reads the first `max_len` tokens (default 256). Datelines/credits at the start of an
article are visible to it, but document LENGTH is mostly invisible, so the L intervention may matter less here than for TF-IDF.
That is a finding to report, not a bug.
"""
import argparse, pathlib, pickle, time, zlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
import exp_tfidf as X

DEFAULT_CONDS = ["C0", "S", "L", "E", "SL", "SE", "LE", "SLE", "R_E"]

def fit_predict_bert(train_texts, y_train, test_sets, a, seed):
    import torch
    from torch.utils.data import DataLoader
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
    del model, opt; (torch.cuda.empty_cache() if dev == "cuda" else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conds", nargs="+", default=DEFAULT_CONDS)
    ap.add_argument("--out", default="results_distilbert.csv")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 1 seed, 2 conditions, small samples")
    ap.add_argument("--backend", default="bert", choices=["bert", "tfidf_check"], help="tfidf_check: pipeline test without a GPU")
    ap.add_argument("--exclude", help="doc_id exclusion file (as in exp_tfidf.py)")
    a = ap.parse_args()

    cfg = dict(X.CFG)
    if a.exclude:
        cfg["exclude_file"] = a.exclude
    if a.smoke:
        cfg.update(universe_per_class=300, n_train=150, n_test_in=60, n_test_x=100, spans_cache="spans_cache_smoke.pkl")
        a.seeds, a.conds, a.out, a.epochs = [0], ["C0", "LE"], a.out.replace(".csv", "_smoke.csv"), 1
    u = X.load_universe(cfg)
    spans = pickle.load(open(cfg["spans_cache"], "rb"))["spans"]
    datasets = sorted(u["dataset"].unique())
    out = pathlib.Path(a.out)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["seed"], r["cond"], r["train"]) for r in rows}
    print(f"{len(rows)} rows already in {out}; {len(done)} cells done")

    for seed in a.seeds:
        pools = {}
        for ds in datasets:                       # identical to exp_tfidf.run()
            d = u[u["dataset"] == ds]
            tr, te = X.split_clusters(d, seed, cfg["test_frac"])
            n_tr, q_tr = X.effective_n(tr, cfg["n_train"]); n_in, q_in = X.effective_n(te, cfg["n_test_in"]); n_x, q_x = X.effective_n(d, cfg["n_test_x"])
            pools[ds] = dict(tr=(tr, n_tr, q_tr), in_=(te, n_in, q_in), x=(d, n_x, q_x))
        X._RCACHE.clear()
        for cond in a.conds:
            S, L, E, T, R = X.CONDITIONS[cond]
            for tr_ds in datasets:
                if (seed, cond, tr_ds) in done:
                    continue
                t0 = time.time()
                rng = np.random.RandomState(zlib.crc32(f"{seed}-{cond}-{tr_ds}".encode()) % (2 ** 31))
                pool, n_eff, q = pools[tr_ds]["tr"]
                trs = X.draw(pool, n_eff, q, L, rng)
                tests = {}
                for te_ds in datasets:
                    role = "in_" if te_ds == tr_ds else "x"
                    p, n2, q2 = pools[te_ds][role]
                    ts = X.draw(p, n2, q2, L, rng)
                    tests[te_ds] = (X.make_texts(ts, cond, spans, seed), ts["is_fake"].to_numpy())
                tr_texts, y_tr = X.make_texts(trs, cond, spans, seed), trs["is_fake"].to_numpy()
                res = (fit_predict_bert(tr_texts, y_tr, tests, a, seed) if a.backend == "bert" else X.fit_eval(tr_texts, y_tr, tests))
                for te_ds, m in res.items():
                    rows.append(dict(seed=seed, model="distilbert" if a.backend == "bert" else "tfidf_check", cond=cond, S=S, L=L, E=E, T=T,
                                     rand_of=R or "", train=tr_ds, test=te_ds, kind="in" if te_ds == tr_ds else "cross",
                                     n_train=len(trs), n_test=len(tests[te_ds][1]), **m))
                pd.DataFrame(rows).to_csv(out, index=False)           # save after EVERY fine-tune
                print(f"seed {seed} {cond:5s} train={tr_ds:17s} done in {time.time() - t0:.0f}s", flush=True)
    print("finished ->", out)


if __name__ == "__main__":
    main()
