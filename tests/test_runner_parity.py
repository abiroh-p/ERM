"""Parity + resume check for the sharded runner on synthetic data (no GPU, no real datasets needed).
Run:  python -m pytest tests -q        (about 1 minute)"""
import pathlib, subprocess, sys
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
EXP = REPO / "src" / "experiments"
sys.path.insert(0, str(EXP))


def synth(path, names=("ISOT", "McIntire", "WELFake_residual", "Aardvark"), n=1500):
    rng = np.random.RandomState(1); vocab = [f"w{i}" for i in range(400)]; rows = []
    for ds in names:
        for lab in (0, 1):
            for _ in range(n):
                k = int(np.clip(rng.normal(60 + (20 if lab else -20) * (ds == "ISOT"), 20), 25, 200))
                w = list(rng.choice(vocab, k)); w[rng.randint(k)] = rng.choice(vocab[:40] if lab else vocab[40:80])
                rows.append(dict(dataset=ds, doc_id=f"{ds}-{len(rows)}", text=" ".join(w), is_fake=lab, cluster_id=f"{ds}-c{len(rows)}"))
    df = pd.DataFrame(rows); df["n_words"] = df["text"].str.split().str.len(); df.to_csv(path, index=False)


def run(*args):
    r = subprocess.run([sys.executable, str(EXP / "exp_transformer.py"), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_sharded_run_matches_original_runner_and_resumes(tmp_path):
    import exp_transformer as T, exp_tfidf as X
    full, three = tmp_path / "full.csv", tmp_path / "three.csv"
    synth(full); pd.read_csv(full).query("dataset != 'Aardvark'").to_csv(three, index=False)
    cache = tmp_path / "spans.pkl"
    run("prepare", "--analysis-csv", str(full), "--spans-cache", str(cache), "--allow-regex-ner")
    (tmp_path / "spans3.pkl").write_bytes(cache.read_bytes())
    conds = ["C0", "LE", "L_RSE"]
    out = tmp_path / "out.csv"
    args = ["run", "--backend", "tfidf", "--workers", "2", "--seeds", "0", "--conds", *conds,
            "--analysis-csv", str(full), "--spans-cache", str(cache), "--out", str(out)]
    run(*args)
    ref_cfg = dict(X.CFG, analysis_csv=str(three), spans_cache=str(tmp_path / "spans3.pkl"), out_csv=str(tmp_path / "ref.csv"), seeds=[0])
    X.run(ref_cfg, conditions=conds)
    a, b = pd.read_csv(tmp_path / "ref.csv"), pd.read_csv(out)
    b = b[(b.train != "Aardvark") & (b.test != "Aardvark")]
    m = a.merge(b, on=["seed", "cond", "train", "test"], suffixes=("_ref", "_new"))
    assert len(m) == len(a) > 0
    assert np.allclose(m["auc_ref"], m["auc_new"], atol=1e-12)          # a 4th dataset must not change the original cells
    assert "already done" in run(*args) and "0 to run" in run(*args)    # resume
