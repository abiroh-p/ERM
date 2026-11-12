"""Compare constructed files by identity and decoded fields, without article exports."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/audit"))
import construction_evidence as E
import provenance_audit as P


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(P.REQUIRED)
        writer.writerows(rows)


def test_construction_comparison_separates_byte_and_field_identity(tmp_path):
    rows = [["ISOT", "ISOT-1", "PRIVATE TEXT", "0", "c1", "2"],
            ["WELFake_residual", "WELFake-2", "OTHER PRIVATE", "1", "c2", "2"]]
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    write(left, rows)
    write(right, rows[::-1])
    report = E.compare_files(left, right)
    assert report["semantic_fields_identical_by_identifier"]
    assert not report["byte_identical"]
    assert "PRIVATE" not in str(report)
    rows[0][2] = "CHANGED PRIVATE"
    write(right, rows)
    report = E.compare_files(left, right)
    assert not report["semantic_fields_identical_by_identifier"]
    assert report["field_mismatches_for_common_identifiers"]["text"] == 1


def test_comparison_masks_unexpected_identifiers(tmp_path):
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    write(left, [["SECRET_DATASET", "SECRET_ID", "SECRET_BODY", "SECRET_LABEL", "c", "1"]])
    write(right, [])
    report = E.compare_files(left, right)
    assert report["original_only_identifiers"][0].startswith("sha256:")
    assert "SECRET" not in str(report)


def test_recorded_cluster_mismatch_is_reported_and_fails(tmp_path):
    path = tmp_path / "input.csv"
    write(path, [["ISOT", "ISOT-1", "one", "0", "c", "1"]])
    historical = tmp_path / "results/audit/primary"
    historical.mkdir(parents=True)
    (historical / "within_dedup_report.csv").write_text(
        "dataset,near_dup_clusters_size_ge2,docs_in_near_dup_clusters,largest_cluster,near_dup_clusters_with_label_conflict\n"
        "ISOT,2,4,2,1\nWELFake,10,20,2,0\n")
    report = P.audit(path)
    P.compare_recorded_clusters(report, tmp_path)
    assert report["recorded_cluster_comparisons"][0]["status"] == "mismatch"
    assert report["recorded_cluster_comparisons"][1]["status"] == "not_comparable"
    assert P.failed(report)
