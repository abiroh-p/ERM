"""Regression checks for the demonstrated historical policy difference."""
from pathlib import Path
import sys

import pandas as pd
import pytest

AUDIT = Path(__file__).resolve().parents[1] / "src/audit"
sys.path.insert(0, str(AUDIT))
import audit_01_05 as A
import build_analysis_set as B


def test_url_only_body_has_light_match_but_recorded_builder_keeps_it():
    url = "https://example.com/" + "/".join(f"segment{i}" for i in range(25))
    ref = A.add_norm_and_hashes(A.standardize("ISOT", [""], [url], [1]))
    wel = A.add_norm_and_hashes(A.standardize("WELFake", [""], [url], [1]))
    assert ref.n_words.iloc[0] >= A.MIN_WORDS
    assert ref.h_text_light.iloc[0] == wel.h_text_light.iloc[0]
    assert pd.isna(ref.h_text_strip.iloc[0]) and pd.isna(wel.h_text_strip.iloc[0])
    assert not B.exact_prefix_flags(ref, wel).iloc[0]
    assert wel.h_text_light.isin(ref.h_text_light.dropna()).iloc[0]
    assert A.near_dup_pairs(pd.concat([ref, wel]), A.JACC_PRIMARY, A.SHINGLE_K, verbose=False).empty


def test_builder_refuses_existing_output_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "original.csv"
    output.write_bytes(b"preserve original")
    def must_not_load(*args):
        pytest.fail("Existing-file guard must run before loading source data")
    monkeypatch.setattr(A, "load_all", must_not_load)
    with pytest.raises(FileExistsError):
        B.main(output)
    assert output.read_bytes() == b"preserve original"


@pytest.mark.parametrize("nested", [False, True])
def test_loader_supports_flat_and_repository_source_layout(tmp_path, nested):
    for corpus, filename, frame in [
        ("ISOT", "Fake.csv", pd.DataFrame({"title": [""], "text": ["fake"]})),
        ("ISOT", "True.csv", pd.DataFrame({"title": [""], "text": ["real"]})),
        ("McIntire", "fake_or_real_news.csv", pd.DataFrame({"title": [""], "text": ["mc"], "label": ["REAL"]})),
        ("WELFake", "WELFake_Dataset.csv", pd.DataFrame({"title": [""], "text": ["wel"], "label": [1]})),
    ]:
        folder = tmp_path / corpus if nested else tmp_path
        folder.mkdir(exist_ok=True)
        frame.to_csv(folder / filename, index=False)
    frames = A.load_all(tmp_path)
    assert [f.is_fake.tolist() for f in frames] == [[1, 0], [0], [1]]
    assert all(not f.title.iloc[0] for f in frames)
