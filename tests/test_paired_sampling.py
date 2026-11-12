"""Run with python -m unittest discover -s tests -p test_paired_sampling.py -v."""
import copy
import importlib.util
import json
import pathlib
import pickle
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
import exp_transformer as T
import exp_tfidf as X
import paired_sampling as P


def synthetic(path, per_class=160):
    rows, spans = [], {}
    rng = np.random.RandomState(29)
    for ds in T.LEGACY_ORDER:
        for label in (0, 1):
            for i in range(per_class):
                did = f"{ds}-{label}-{i}"
                words = ["Reuters", "Acme", "genuine" if label == 0 else "rumor"]
                words += [f"term{v}" for v in rng.randint(0, 80, size=rng.randint(22, 60))]
                text = " ".join(words)
                rows.append(dict(dataset=ds, doc_id=did, is_fake=label, text=text,
                                 cluster_id=f"{ds}-{label}-cluster{i // 2}"))
                spans[did] = [(8, 12)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return spans


class PairedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.csv = self.root / "analysis.csv"
        self.spans = synthetic(self.csv)
        self.cache = self.root / "spans.pkl"
        self.cache.write_bytes(pickle.dumps(dict(method="regex_fallback", spans=self.spans)))
        self.cfg = dict(X.CFG, analysis_csv=str(self.csv), spans_cache=str(self.cache))
        self.u = X.load_universe(self.cfg)
        self.args = SimpleNamespace(split="cluster", date_col="date", cutoff_quantile=0.7)

    def test_manifests_ignore_dataset_order_and_separate_clusters(self):
        pools = T.build_pools(self.u, T.LEGACY_ORDER, 0, self.cfg, self.args)
        first = P.make_manifest(pools, 0)
        second = P.make_manifest(dict(reversed(list(pools.items()))), 0)
        self.assertEqual(first, second)
        indexed = self.u.set_index("doc_id", drop=False)
        for ds in T.LEGACY_ORDER:
            for length in (False, True):
                train, _ = P.select_sample(indexed, first, ds, "tr", length)
                test, _ = P.select_sample(indexed, first, ds, "in_", length)
                self.assertFalse(set(train.cluster_id) & set(test.cluster_id))
        path = self.root / "manifest.json"
        P.save_manifest(path, first)
        P.read_manifest(path, first["manifest_sha256"])
        corrupt = copy.deepcopy(first)
        corrupt["samples"][0]["rows"][0][1] ^= 1
        path.write_text(json.dumps(corrupt))
        with self.assertRaisesRegex(ValueError, "integrity"):
            P.read_manifest(path, first["manifest_sha256"])

    def test_cache_and_identifier_guards(self):
        with self.assertRaisesRegex(ValueError, "spaCy"):
            P.validate_cache(dict(method="regex_fallback", spans=self.spans), self.u)
        P.validate_cache(dict(method="regex_fallback", spans=self.spans), self.u, True)
        with self.assertRaisesRegex(ValueError, "unique"):
            P.validate_universe(pd.concat([self.u, self.u.iloc[:1]]))

    def test_random_controls_match_every_document_including_empty(self):
        rows = pd.DataFrame([dict(doc_id="normal", text="Reuters Acme story text"),
                             dict(doc_id="empty", text="Reuters"),
                             dict(doc_id="nohit", text="some ordinary words")])
        spans = {"normal": [(8, 12)], "empty": [(0, 7)], "nohit": []}
        X._MASK_CACHE.clear()
        for real, random in [("S", "R_S"), ("E", "R_E"), ("SE", "R_SE"),
                             ("LE", "L_RE"), ("SLE", "L_RSE")]:
            a = P.make_texts(rows, real, spans, 7)
            b = P.make_texts(rows, random, spans, 7)
            self.assertEqual([len(t.split()) for t in a], [len(t.split()) for t in b])
            self.assertEqual(b, P.make_texts(rows, random, spans, 7))

    def test_incomplete_jobs_are_not_skipped(self):
        rows = [dict(seed=0, model="tfidf_lr", cond="C0", train="ISOT", test=d,
                     sampling_protocol=P.PROTOCOL) for d in T.LEGACY_ORDER[:2]]
        partial = pd.DataFrame(rows)
        self.assertEqual(P.complete_jobs(partial, "tfidf_lr", T.LEGACY_ORDER, P.PROTOCOL), set())
        with self.assertRaisesRegex(ValueError, "legacy"):
            P.complete_jobs(partial.drop(columns="sampling_protocol"), "tfidf_lr", T.LEGACY_ORDER, P.PROTOCOL)

    def run_cli(self, *extra, check=True):
        command = [sys.executable, str(ROOT / "src/experiments/exp_transformer.py"), "run",
                   "--sampling", "paired", "--backend", "tfidf", "--allow-regex-ner",
                   "--workers", "2", "--seeds", "0", "--conds", "C0", "E", "R_E", "L", "LE", "L_RE", "SLE", "L_RSE",
                   "--analysis-csv", str(self.csv), "--spans-cache", str(self.cache),
                   "--out", str(self.root / "results.csv"), *extra]
        result = subprocess.run(command, capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result

    def test_end_to_end_pairing_worker_count_resume_and_guards(self):
        self.run_cli()
        result_file = self.root / "results.csv"
        results = pd.read_csv(result_file)
        self.assertEqual(len(results), 8 * 9)
        self.assertFalse(results.duplicated(T.KEY).any())
        self.assertEqual(set(results.sampling_protocol), {P.PROTOCOL})
        for conditions in [["C0", "E", "R_E"], ["L", "LE", "L_RE", "SLE", "L_RSE"]]:
            for _, group in results[results.cond.isin(conditions)].groupby(["seed", "train", "test"]):
                self.assertEqual(group.train_sample_sha256.nunique(), 1)
                self.assertEqual(group.test_sample_sha256.nunique(), 1)
        for _, group in results[results.kind == "cross"].groupby(["cond", "test"]):
            self.assertEqual(group.test_sample_sha256.nunique(), 1)
        resumed = self.run_cli("--workers", "1")
        self.assertIn("0 to run", resumed.stdout)
        pd.testing.assert_frame_equal(results, pd.read_csv(result_file))
        other = self.root / "single.csv"
        self.run_cli("--workers", "1", "--out", str(other))
        pd.testing.assert_frame_equal(results, pd.read_csv(other))
        # A partial job must be recomputed, rather than silently skipped.
        results.iloc[1:].to_csv(result_file, index=False)
        self.assertIn("1 to run", self.run_cli().stdout)
        self.assertEqual(len(pd.read_csv(result_file)), 72)
        changed = self.run_cli("--max_len", "128", check=False)
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("changed", changed.stderr)
        # A legacy command must not append to a paired output.
        mixed = self.run_cli("--sampling", "legacy", check=False)
        self.assertNotEqual(mixed.returncode, 0)
        self.assertIn("legacy", mixed.stderr)


if __name__ == "__main__":
    unittest.main()
