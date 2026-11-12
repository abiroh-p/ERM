"""Analysis rejects mismatched/partial grids rather than hiding missing cells."""
import pathlib
import sys
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/analysis"))
import analyze_paired as A


def results():
    rows = []
    for seed in (0, 1):
        for cond in A.FOCUSED:
            length = bool(A.T.X.CONDITIONS[cond][1])
            for train in A.T.LEGACY_ORDER:
                for test in A.T.LEGACY_ORDER:
                    role = "in_" if train == test else "x"
                    rows.append(dict(seed=seed, cond=cond, train=train, test=test,
                                     model="tfidf_lr", kind="in" if train == test else "cross", auc=0.8,
                                     sampling_protocol=A.P.PROTOCOL,
                                     manifest_sha256=A.P.digest([seed]),
                                     train_sample_sha256=A.P.digest([seed, train, length, "tr"]),
                                     test_sample_sha256=A.P.digest([seed, test, length, role])))
    return pd.DataFrame(rows)


class PairedAnalysisTests(unittest.TestCase):
    def validate(self, d):
        return A.validate(d, [0, 1], A.FOCUSED, A.T.LEGACY_ORDER)

    def test_complete_grid_and_six_contrast_correction(self):
        d = self.validate(results())
        summaries = A.summaries(d)
        planned = summaries[summaries.family == "planned_descriptive"]
        self.assertEqual(len(planned), 8)  # four available contrasts x two outcomes
        self.assertEqual(set(planned.p_holm_six), {1.0})

    def test_partial_duplicate_and_bad_metrics_rejected(self):
        d = results()
        for broken in [d.iloc[:-1], pd.concat([d, d.iloc[:1]]), d.assign(auc=float('nan'))]:
            with self.assertRaises(ValueError):
                self.validate(broken)

    def test_wrong_document_hash_or_protocol_rejected(self):
        d = results()
        d.loc[0, 'test_sample_sha256'] = A.P.digest(['different'])
        with self.assertRaisesRegex(ValueError, 'different document'):
            self.validate(d)
        d = results().assign(sampling_protocol='legacy')
        with self.assertRaisesRegex(ValueError, 'only paired'):
            self.validate(d)


if __name__ == '__main__':
    unittest.main()
