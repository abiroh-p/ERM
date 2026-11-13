"""Integrity/selection/runner checks; all model scores here use synthetic text."""
import copy
import importlib.util
import json
import pathlib
import pickle
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src/audit"), str(ROOT/"src/experiments")]
import celebrity_audit as A
import paired_sampling as P
spec = importlib.util.spec_from_file_location("celebrity_runner", ROOT/"scripts/external_celebrity.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def row(did, text, label=0, group=None):
    return dict(doc_id=did, text=text, body=text, title="", is_fake=label, story_group=group or did,
                decoding_error="", n_words=len(text.split()), missing_body=not bool(text))


def test_matcher_has_exhaustive_recall_and_empty_safety():
    cfg = json.loads(A.CONFIG.read_text())["near"]
    targets = [row("empty", ""), row("punct", "!!!"), row("real", " ".join(f"word{i}" for i in range(85)))]
    matcher = A.Matcher(targets, cfg)
    assert list(matcher.matches(row("blank", ""))) == []
    assert list(matcher.matches(row("blank", "!!!"))) == []
    found = list(matcher.matches(row("copy", targets[2]["text"], 1)))
    assert {m["article_id"] for m in found} == {"real"}
    assert all(m["cross_label"] for m in found)
    assert {m["method"] for m in found} >= {"exact_light", "exact_strip", "primary_full", "primary_head100", "sensitivity_full", "prefix300"}
    # Containment can trigger when Jaccard fails, and header copies can trigger alone.
    longer = row("long", targets[2]["text"] + " " + " ".join(f"other{i}" for i in range(300)))
    assert any(m["method"] == "primary_full" and m["jaccard"] < .8 and m["containment"] == 1 for m in matcher.matches(longer))
    # Cross-check every indexed near score against the original exact scorer.
    for m in matcher.matches(longer):
        if m["method"].startswith(("primary_", "sensitivity_")):
            k = 5 if m["method"].startswith("primary") else 3
            cut = 100 if m["method"].endswith("head100") else None
            x = A.A.shingles(A.A.norm_strip(targets[2]["text"]).split()[:cut], k)
            y = A.A.shingles(A.A.norm_strip(longer["text"]).split()[:cut], k)
            jac, con, _ = A.A.jac_con(x, y)
            assert m["jaccard"] == jac and m["containment"] == con


def test_exclusions_close_over_related_components():
    rows = [row("a", "text", group="1"), row("b", "text", group="1"), row("c", "text", group="2"), row("d", "text", group="3")]
    for r in rows:
        r["n_words"] = 50
    internal = [dict(article_id="b", other_id="c", cross_label=False, method="sensitivity_full")]
    cross = [dict(article_id="a")]
    result = A.component_exclusions(rows, internal, cross)
    assert [r["excluded"] for r in result] == [True, True, True, False]
    assert len({r["component"] for r in result[:3]}) == 1


def test_zip_scope_preserves_text_and_rejects_mismatched_names(tmp_path):
    zpath = tmp_path/"scope.zip"
    original = "\r\nHeadline\r\n\r\nOriginal body café.\r\n"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(A.PREFIX+"fake/001fake.txt", original.encode("utf-8"))
        z.writestr("fakeNewsDatasets/fakeNewsDataset/fake/001fake.txt", "Other dataset must not enter")
    rows = A.read_articles(zpath)
    assert len(rows) == 1 and rows[0]["text"] == original
    assert rows[0]["title"] == "Headline" and rows[0]["body"] == "\r\n\r\nOriginal body café.\r\n"
    with zipfile.ZipFile(zpath, "a") as z:
        z.writestr(A.PREFIX+"fake/002legit.txt", "folder/filename label disagreement")
    with pytest.raises(ValueError, match="unexpected Celebrity member"):
        A.read_articles(zpath)


def test_preliminary_screen_does_not_silently_apply_minimum_shingles():
    sys.path.insert(0, str(ROOT/"scripts"))
    from verify_celebrity_preparation import preliminary_screen
    text = " ".join(f"w{i}" for i in range(30))
    rows = [row("short", text), row("long", text+" "+" ".join(f"z{i}" for i in range(300)))]
    assert len(preliminary_screen(rows)) == 1
    cfg = json.loads(A.CONFIG.read_text())["near"]
    assert list(A.Matcher(rows[:1], cfg).matches(rows[1])) == []


def test_fixed_target_and_random_control_word_counts():
    c = json.loads(A.CONFIG.read_text())
    df = pd.DataFrame([row(f"a{i}", "Alice " + "word "*(20+i), i%2, str(i//2)) for i in range(30)])
    manifest = R.target_manifest(df, c)
    assert manifest == R.target_manifest(df, c)
    assert len(manifest["primary"]["rows"]) == len(df)
    assert len({r[0] for r in manifest["secondary"]["rows"]}) == len(manifest["secondary"]["rows"])
    df = pd.DataFrame([row("all", "Alice Smith", 1), row("some", "Alice enjoys music today", 0)])
    spans = {"all": [(0,11)], "some": [(0,5)]}
    for seed in (0, 7):
        for mask, control in (("E", "R_E"), ("LE", "L_RE"), ("SLE", "L_RSE"), ("SE", "R_SE")):
            R.X._MASK_CACHE.clear()
            masked = P.make_texts(df, mask, spans, seed)
            random = P.make_texts(df, control, spans, seed)
            assert [len(t.split()) for t in masked] == [len(t.split()) for t in random]
            assert random[0] == ""


def synthetic_setup(tmp_path):
    c = json.loads(A.CONFIG.read_text())
    c.update(sources=["ISOT"], seeds=[0], target_length_bins=2,
             conditions=["C0", "E", "R_E", "SE", "R_SE", "L", "LE", "L_RE", "SLE", "L_RSE"])
    zpath = tmp_path/"synthetic.zip"
    rng = np.random.RandomState(13)
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("fakeNewsDatasets/README", "Synthetic test fixture: no third-party text")
        for i in range(1, 21):
            for folder in ("fake", "legit"):
                words = " ".join(rng.choice([f"target{j}" for j in range(500)], 40+i))
                z.writestr(f"{A.PREFIX}{folder}/{i:03d}{folder}.txt", "Synthetic headline\n\n" + words)
    source = []
    for label in (0, 1):
        for i in range(120):
            text = "source headline\n" + " ".join(rng.choice([f"source{j}" for j in range(500)], 30+i%40))
            source.append(dict(dataset="ISOT", doc_id=f"ISOT-{label}-{i}", text=text, is_fake=label, cluster_id=f"{label}-{i}"))
    csv = tmp_path/"source.csv"
    pd.DataFrame(source).to_csv(csv, index=False)
    c.update(zip_sha256=P.file_hash(zpath), analysis_sha256=P.file_hash(csv))
    cfg = tmp_path/"config.json"
    P.atomic_json(cfg, c)
    a = SimpleNamespace(zip=str(zpath), analysis_csv=str(csv), config=str(cfg), audit=str(tmp_path/"audit"),
                        prepared=str(tmp_path/"prepared"), output=str(tmp_path/"outputs"), model="tfidf", rank=0, world_size=2)
    A.audit(zpath, csv, a.audit, cfg)
    R.select(a)
    # Explicit synthetic spans; production validation is relaxed only by this test's monkeypatch.
    spans = {r["doc_id"]: [(0,6)] for r in source}
    spans.update({r["doc_id"]: [(0,9)] for r in A.read_articles(zpath)})
    with open(pathlib.Path(a.prepared)/"spans.pkl", "wb") as f:
        pickle.dump(dict(method="synthetic-test-only", spans=spans), f)
    P.atomic_json(pathlib.Path(a.prepared)/"ner_provenance.json", dict(spans_sha256=P.digest(spans), method="synthetic-test-only"))
    return a


def test_synthetic_tfidf_end_to_end_resume_integrity_and_analysis(tmp_path, monkeypatch):
    a = synthetic_setup(tmp_path)
    original = P.validate_cache
    monkeypatch.setattr(P, "validate_cache", lambda cache, u: original(cache, u, allow_regex=True))
    R.plan(a)
    R.run(a)
    _, out, _, status = R.collect(a)
    assert status["completed"] == 5 and not status["complete"]
    a.rank = 1
    R.run(a)
    before = {p: p.read_bytes() for p in (out/"jobs").glob("*.json")}
    R.run(a)
    assert before == {p: p.read_bytes() for p in before}
    _, _, _, status = R.collect(a)
    assert status["complete"] and status["completed"] == 10
    R.analyze(a)
    contrasts = pd.read_csv(out/"matched_contrasts.csv")
    assert len(contrasts) == 4 and set(contrasts.scope) == {"primary", "secondary"}
    # Worker partitioning must not change outcomes or deletion streams.
    a.output = str(tmp_path/"serial_outputs")
    a.rank, a.world_size = 0, 1
    R.plan(a)
    R.run(a)
    serial = pathlib.Path(a.output)/"tfidf"/"jobs"
    for path, saved in before.items():
        x, y = json.loads(saved), json.loads((serial/path.name).read_bytes())
        assert x["metrics"] == y["metrics"]
        assert x["target_word_deletions"] == y["target_word_deletions"]
    a.output = str(tmp_path/"outputs")
    # Corrupted completed results must fail, never be silently deduplicated/accepted.
    path = next(iter(before))
    record = json.loads(path.read_text())
    record["metrics"]["auc"] = .123
    P.atomic_json(path, record)
    with pytest.raises(ValueError, match="checksum"):
        R.collect(a)


def test_incomplete_cross_audit_and_changed_manifest_block_evaluation(tmp_path):
    a = synthetic_setup(tmp_path)
    summary_path = pathlib.Path(a.audit)/"audit_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cross_corpus_complete"] = False
    P.atomic_json(summary_path, summary)
    with pytest.raises(ValueError, match="incomplete"):
        R.select(a)
    summary["cross_corpus_complete"] = True
    P.atomic_json(summary_path, summary)
    path = pathlib.Path(a.prepared)/"target_selection.json"
    manifest = json.loads(path.read_text())
    manifest["primary"]["rows"].pop()
    P.atomic_json(path, manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        R.checked_selections(a)
