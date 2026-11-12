"""Read-only, streaming audit of the experiment CSV; emits a text-free summary ZIP.

No training, data downloads, normalization, deduplication or overlap search is performed.
Memory is proportional to identifiers and clusters, never to the article-text collection.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

REPO = Path(__file__).resolve().parents[2]
REQUIRED = ("dataset", "doc_id", "text", "is_fake", "cluster_id", "n_words")
CORPORA = ("ISOT", "McIntire", "WELFake_residual")
# pandas read_csv's default NA strings, plus whitespace-only fields. Raw spelling
# is retained for identifier equality; no text or identifier values enter reports.
NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
      "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}
CLAIMS = {
    "manuscript_and_README": {"total": 61197, "counts": {
        "ISOT": {"0": 20923, "1": 17107}, "McIntire": {"0": 2964, "1": 3041},
        "WELFake_residual": {"0": 10427, "1": 6735}}},
    "historical_primary_dry_run": {"total": 61196, "counts": {
        "ISOT": {"0": 20923, "1": 17107}, "McIntire": {"0": 2964, "1": 3041},
        "WELFake_residual": {"0": 10427, "1": 6734}}},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing(value):
    return value is None or not value.strip() or value in NA


def label(value):
    """Accept binary numeric spellings accepted by the runners, never truncate."""
    from decimal import Decimal, InvalidOperation
    if missing(value):
        return None
    try:
        number = Decimal(value)
        return int(number) if number.is_finite() and number in (0, 1) else None
    except (InvalidOperation, ValueError):
        return None


def recorded_references(root, extra=()):
    """Read only known metadata, never load CSVs or treat result hashes as data hashes."""
    paths = {root / "freeze/MANIFEST.json", root / "archive/freeze_validation_intermediate/MANIFEST.json"}
    for folder in ("configs", "results/raw", "results/validation"):
        paths.update((root / folder).rglob("*_config.json"))
    paths.update(Path(p) for p in extra)
    refs, inspected = [], []
    for path in sorted(paths):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            inspected.append({"record": name, "status": "unreadable_metadata"})
            continue
        inspected.append({"record": name, "sha256": sha256(path)})
        # freeze_and_analyze.py records the analysis input in files/extra/analysis_set.csv.
        for key, item in obj.get("files", {}).items():
            if key.replace("\\", "/").split("/")[-1] in ("analysis_set.csv", "analysis_clean.csv", "analysis.csv"):
                if isinstance(item, dict):
                    refs.append({"record": name, "key": key, "sha256": item.get("sha256"),
                                 "rows": item.get("rows"), "bytes": item.get("bytes")})
        # Newer experiment records may store an explicit analysis-input hash.
        def visit(node, keypath=""):
            if not isinstance(node, dict):
                return
            for key, value in node.items():
                location = f"{keypath}/{key}"
                if key in ("analysis_csv_sha256", "analysis_sha256", "analysis_set_sha256"):
                    refs.append({"record": name, "key": location, "sha256": value})
                elif key in ("analysis_csv", "analysis_set", "analysis_input") and isinstance(value, dict):
                    if "sha256" in value:
                        refs.append({"record": name, "key": location, "sha256": value["sha256"],
                                     "rows": value.get("rows"), "bytes": value.get("bytes")})
                visit(value, location)
        visit(obj)
    return refs, inspected


def audit(path, references=()):
    path = Path(path)
    before = path.stat()
    digest = sha256(path)
    # Windows C long is smaller than sys.maxsize.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    counts = {ds: Counter() for ds in CORPORA}
    totals, missing_counts, problems = Counter(), Counter(), Counter()
    ids, id_labels, clusters = Counter(), defaultdict(set), {}
    cluster_corpora = defaultdict(set)
    n_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames or []
        absent = [col for col in REQUIRED if col not in header]
        duplicate_columns = len(header) - len(set(header))
        try:
            for row in reader:
                n_rows += 1
                if None in row or any(v is None for v in row.values()):
                    problems["malformed_rows"] += 1
                for col in REQUIRED:
                    if missing(row.get(col)):
                        missing_counts[col] += 1
                ds = row.get("dataset")
                ds = ds if ds in CORPORA else "unknown_or_missing"
                totals[ds] += 1
                lab = label(row.get("is_fake"))
                if lab is None:
                    problems["invalid_or_missing_labels"] += 1
                if ds in counts:
                    counts[ds][str(lab) if lab is not None else "invalid_or_missing"] += 1
                doc_id = row.get("doc_id")
                if not missing(doc_id):
                    ids[doc_id] += 1
                    if lab is not None:
                        id_labels[doc_id].add(lab)
                cluster = row.get("cluster_id")
                if not missing(cluster):
                    entry = clusters.setdefault((ds, cluster), [0, set(), set()])
                    cluster_corpora[cluster].add(ds)
                    entry[0] += 1
                    entry[1].add(ds)
                    if lab is not None:
                        entry[2].add(lab)
                words = row.get("n_words")
                if not missing(words):
                    try:
                        value = int(words)
                        if value < 0:
                            raise ValueError
                    except ValueError:
                        problems["invalid_n_words"] += 1
                    else:
                        # Match the final builder: title + newline + body, whitespace split.
                        if "text" in row and value != len((row["text"] or "").split()):
                            problems["n_words_text_mismatches"] += 1
        except csv.Error:
            # Do not expose parser content or article text in exceptions/reports.
            problems["csv_parse_errors"] += 1
    after = path.stat()
    unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    compared = []
    for reference in references:
        ref = dict(reference)
        expected_hash = ref.get("sha256")
        ref["hash_status"] = ("not_recorded" if not expected_hash else
                              "match" if str(expected_hash).lower() == digest else "mismatch")
        for key, actual in (("rows", n_rows), ("bytes", before.st_size)):
            ref[f"{key}_status"] = ("not_recorded" if ref.get(key) is None else
                                     "match" if ref[key] == actual else "mismatch")
        compared.append(ref)
    claim_checks = []
    for name, claim in CLAIMS.items():
        deltas = {ds: {lab: counts[ds][lab] - expected for lab, expected in per_label.items()}
                  for ds, per_label in claim["counts"].items()}
        claim_checks.append({"claim": name, "total_expected": claim["total"],
                             "total_delta": n_rows - claim["total"], "label_count_deltas": deltas,
                             "status": "match" if n_rows == claim["total"] and
                             not any(v for labs in deltas.values() for v in labs.values()) else "mismatch"})
    cluster_summary = {}
    for ds in CORPORA:
        entries = [v for v in clusters.values() if ds in v[1]]
        cluster_summary[ds] = {"unique_clusters": len(entries),
                               "clusters_size_ge2": sum(v[0] >= 2 for v in entries),
                               "documents_in_clusters_size_ge2": sum(v[0] for v in entries if v[0] >= 2),
                               "largest_cluster": max((v[0] for v in entries), default=0),
                               "clusters_with_both_labels": sum(len(v[2]) > 1 for v in entries)}
    issues = {key: problems[key] for key in ("malformed_rows", "csv_parse_errors", "invalid_or_missing_labels",
                                           "invalid_n_words", "n_words_text_mismatches")}
    issues.update(unknown_corpus_rows=totals["unknown_or_missing"],
                  duplicate_identifiers=sum(v > 1 for v in ids.values()),
                  duplicate_identifier_extra_rows=sum(v - 1 for v in ids.values()),
                  conflicting_label_identifiers=sum(len(v) > 1 for v in id_labels.values()))
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": {"sha256": digest, "bytes": before.st_size, "unchanged_during_audit": unchanged},
            "parser": "Python csv, UTF-8 BOM accepted, RFC-style quoted multiline fields; streams one record at a time",
            "missing_policy": "pandas default NA tokens and whitespace-only fields; no values printed",
            "label_meanings": {"0": "real (dataset label)", "1": "fake (dataset label)"},
            "label_policy": "binary numeric spellings accepted; nonbinary values never truncated",
            "required_columns": list(REQUIRED), "missing_columns": absent,
            "duplicate_header_columns": duplicate_columns, "missing_values": dict(missing_counts),
            "total_rows": n_rows, "counts": {ds: dict(c) for ds, c in counts.items()},
            "corpus_totals": dict(totals), "checks": issues,
            "clusters": cluster_summary,
            "cluster_ids_shared_across_corpora": sum(len(v) > 1 for v in cluster_corpora.values()),
            "recorded_input_comparisons": compared, "count_claim_comparisons": claim_checks,
            "construction_reproduction": "not_performed_by_this_command",
            "limitations": ["Byte-hash agreement establishes file identity only.",
                            "Counts and cluster summaries do not independently verify overlap or article veracity.",
                            "Construction reproduction requires original sources, rules and environment.",
                            "Cluster IDs are inspected as recorded; clusters are not recomputed.",
                            "Record totals after a parse error are partial."]}


def failed(report):
    return (bool(report["missing_columns"]) or report["duplicate_header_columns"] or
            any(report["missing_values"].values()) or any(report["checks"].values()) or
            not report["input"]["unchanged_during_audit"] or
            any(r["status"] == "mismatch" for r in report.get("recorded_cluster_comparisons", [])) or
            any(r["hash_status"] == "mismatch" or r["rows_status"] == "mismatch" or
                r["bytes_status"] == "mismatch" for r in report["recorded_input_comparisons"]))


def summary(report):
    lines = ["# Current analysis-input audit", "", f"Rows: {report['total_rows']}",
             f"SHA-256: {report['input']['sha256']}", "", "Labels: 0 = real; 1 = fake (source labels).", "",
             "| Corpus | Real (0) | Fake (1) | Invalid/missing |", "|---|---:|---:|---:|"]
    for ds, counts in report["counts"].items():
        lines.append(f"| {ds} | {counts.get('0', 0)} | {counts.get('1', 0)} | {counts.get('invalid_or_missing', 0)} |")
    lines += ["", "Byte identity comparisons:"]
    lines += [f"- {r['record']}: {r['hash_status']} (rows: {r['rows_status']}; bytes: {r['bytes_status']})"
              for r in report["recorded_input_comparisons"]] or ["- No recorded hash available."]
    lines += ["", "Count consistency:"]
    lines += [f"- {r['claim']}: {r['status']}; total delta {r['total_delta']:+d}." for r in report["count_claim_comparisons"]]
    lines += ["", "Data checks: " + ("ATTENTION REQUIRED" if failed(report) else "passed"),
              "See audit.json for missing fields, duplicate/conflicting identifiers and cluster summaries.",
              "", "Construction reproduction: not performed by this audit command.", ""] + report["limitations"]
    return "\n".join(lines) + "\n"


def write_bundle(output, report, metadata):
    # Exclusive creation protects datasets, historical reports and earlier audit evidence.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("audit.json", json.dumps(report, indent=2) + "\n")
        bundle.writestr("SUMMARY.md", summary(report))
        bundle.writestr("metadata.json", json.dumps(metadata, indent=2) + "\n")


def compare_recorded_clusters(report, root):
    path = root / "results/audit/primary/within_dedup_report.csv"
    checks = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                ds = row["dataset"]
                if ds == "WELFake":
                    checks.append({"corpus": "WELFake_residual", "status": "not_comparable",
                                   "reason": "historical report describes full cleaned WELFake, before residual trimming"})
                    continue
                if ds not in report["clusters"]:
                    continue
                pairs = {"near_dup_clusters_size_ge2": "clusters_size_ge2",
                         "docs_in_near_dup_clusters": "documents_in_clusters_size_ge2",
                         "largest_cluster": "largest_cluster",
                         "near_dup_clusters_with_label_conflict": "clusters_with_both_labels"}
                deltas = {current: report["clusters"][ds][current] - int(row[historical])
                          for historical, current in pairs.items()}
                checks.append({"corpus": ds, "status": "match" if not any(deltas.values()) else "mismatch",
                               "deltas": deltas})
    report["recorded_cluster_comparisons"] = checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New summary ZIP, never overwritten")
    parser.add_argument("--record", action="append", default=[], type=Path, help="Additional experiment/freeze JSON")
    parser.add_argument("--expected-sha256", help="Additional recorded analysis-input hash")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() == args.analysis_csv.resolve():
        parser.error("output must be a new file separate from the analysis input")
    for path in args.record:
        if not path.is_file():
            parser.error("an explicitly requested metadata record is missing")
    refs, inspected = recorded_references(REPO, args.record)
    if args.expected_sha256:
        if len(args.expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_sha256):
            parser.error("expected SHA-256 must contain 64 hexadecimal characters")
        refs.append({"record": "command_line_expected_sha256", "sha256": args.expected_sha256})
    report = audit(args.analysis_csv, refs)
    compare_recorded_clusters(report, REPO)
    config_paths = ["src/audit/build_analysis_set.py", "src/audit/audit_01_05.py", "src/audit/provenance_audit.py",
                    "results/audit/primary/config.json", "results/audit/primary/welfake_dry_run.csv",
                    "results/audit/primary/triple_a_primary.csv", "results/audit/primary/within_dedup_report.csv"]
    metadata = {"python": platform.python_version(), "records_inspected": inspected,
                "code_and_configuration_hashes": {p: sha256(REPO / p) for p in config_paths if (REPO / p).is_file()},
                "count_claims": CLAIMS, "command": "provenance_audit.py --analysis-csv <existing file> --output <new ZIP>",
                "data_included": False, "training_launched": False}
    write_bundle(args.output, report, metadata)
    print(summary(report))
    print(f"Summary ZIP written ({args.output.stat().st_size} bytes).")
    # Historical dry-run disagreement is reported, not a schema failure or a target to force.
    return 1 if failed(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
