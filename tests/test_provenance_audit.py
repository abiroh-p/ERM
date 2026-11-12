"""Synthetic input checks: no real articles, training, network or GPUs."""
import csv
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

MODULE = Path(__file__).resolve().parents[1] / "src/audit/provenance_audit.py"
spec = importlib.util.spec_from_file_location("provenance_audit", MODULE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def fixture_csv(tmp_path, rows=None, columns=A.REQUIRED):
    path = tmp_path / "analysis.csv"
    rows = rows if rows is not None else [
        ["ISOT", "id-1", "private article\nsecond line", "0", "c-1", "4"],
        ["WELFake_residual", "id-2", "another private article", "1", "c-2", "3"]]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_multiline_counts_and_incremental_hash(tmp_path):
    path = fixture_csv(tmp_path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = A.audit(path, [{"record": "test", "sha256": digest, "rows": 2}])
    assert report["input"]["sha256"] == digest
    assert report["total_rows"] == 2
    assert report["counts"]["ISOT"]["0"] == 1
    assert report["recorded_input_comparisons"][0]["hash_status"] == "match"
    assert not A.failed(report)


@pytest.mark.parametrize("bad", ["2", "-1", "0.5", "FAKE", "NaN", "", "inf"])
def test_invalid_labels_never_truncated(tmp_path, bad):
    report = A.audit(fixture_csv(tmp_path, [["ISOT", "id", "private", bad, "c", "1"]]))
    assert report["checks"]["invalid_or_missing_labels"] == 1
    assert A.failed(report)


def test_duplicate_and_conflicting_identifiers_global(tmp_path):
    rows = [["ISOT", "same", "secret", "0", "cluster", "1"],
            ["ISOT", "same", "secret", "0", "cluster", "1"],
            ["McIntire", "same", "secret", "1", "cluster", "1"]]
    report = A.audit(fixture_csv(tmp_path, rows))
    assert report["checks"]["duplicate_identifiers"] == 1
    assert report["checks"]["duplicate_identifier_extra_rows"] == 2
    assert report["checks"]["conflicting_label_identifiers"] == 1
    assert report["cluster_ids_shared_across_corpora"] == 1
    assert report["clusters"]["ISOT"]["clusters_with_both_labels"] == 0
    rows.append(["ISOT", "other", "secret", "1", "cluster", "1"])
    assert A.audit(fixture_csv(tmp_path, rows))["clusters"]["ISOT"]["clusters_with_both_labels"] == 1
    assert A.failed(report)


def test_hash_and_recorded_count_mismatch(tmp_path):
    report = A.audit(fixture_csv(tmp_path), [{"record": "test", "sha256": "0" * 64, "rows": 17}])
    ref = report["recorded_input_comparisons"][0]
    assert ref["hash_status"] == ref["rows_status"] == "mismatch"
    assert A.failed(report)


def test_missing_columns_fields_and_word_count(tmp_path):
    report = A.audit(fixture_csv(tmp_path, [["ISOT", "", "secret", "1"]], A.REQUIRED[:4]))
    assert report["missing_columns"] == ["cluster_id", "n_words"]
    assert report["missing_values"]["doc_id"] == 1
    assert A.failed(report)
    path = tmp_path / "words.csv"
    path.write_text("dataset,doc_id,text,is_fake,cluster_id,n_words\nISOT,id,secret,1,c,2\n")
    assert A.audit(path)["checks"]["n_words_text_mismatches"] == 1


def test_no_article_or_unknown_values_in_zip_and_no_overwrite(tmp_path):
    sentinel = "PRIVATE_ARTICLE_NEVER_EXPORT"
    path = fixture_csv(tmp_path, [[sentinel, sentinel, sentinel, sentinel, sentinel, "1"]])
    report = A.audit(path)
    output = tmp_path / "report.zip"
    A.write_bundle(output, report, {"data_included": False})
    with zipfile.ZipFile(output) as bundle:
        assert set(bundle.namelist()) == {"audit.json", "SUMMARY.md", "metadata.json"}
        assert all(sentinel.encode() not in bundle.read(n) for n in bundle.namelist())
    with pytest.raises(FileExistsError):
        A.write_bundle(output, report, {})
    with pytest.raises(SystemExit):
        A.main(["--analysis-csv", str(path), "--output", str(path)])


def test_record_discovery_does_not_use_result_hash(tmp_path):
    folder = tmp_path / "freeze"
    folder.mkdir()
    (folder / "MANIFEST.json").write_text(json.dumps({"files": {
        "tfidf/results.csv": {"sha256": "a" * 64},
        "extra/analysis_set.csv": {"sha256": "b" * 64, "rows": 2}}}))
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({"inputs": {"analysis_csv_sha256": "c" * 64}}))
    refs, inspected = A.recorded_references(tmp_path, [config])
    assert {r["sha256"] for r in refs} == {"b" * 64, "c" * 64}
    assert len(inspected) == 2


def test_parse_failure_is_partial_and_text_free(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text('dataset,doc_id,text,is_fake,cluster_id,n_words\nISOT,id,"PRIVATE_UNCLOSED')
    report = A.audit(path)
    assert report["checks"]["csv_parse_errors"] == 1
    assert "PRIVATE_UNCLOSED" not in json.dumps(report)
    assert A.failed(report)
