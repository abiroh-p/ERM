"""Compare historical trim evidence and an optional fresh construction, without exporting text.

Requires pandas; historical Parquet metadata inspection also requires pyarrow.
This command reads inputs only. Run build_analysis_set.py separately into a NEW directory.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd

import audit_01_05 as A
import build_analysis_set as B
import provenance_audit as P


def safe_ids(values):
    return [v if re.fullmatch(r"(?:ISOT|McIntire|WELFake)-\d+", v) else
            "sha256:" + hashlib.sha256(v.encode()).hexdigest() for v in sorted(values)]


def fingerprints(path):
    """Keep only per-field digests in memory, preserving exact decoded field values."""
    csv.field_size_limit(100_000_000)
    rows, duplicates, counts = {}, 0, Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, strict=True):
            key = row["doc_id"]
            duplicates += key in rows
            rows[key] = tuple(hashlib.sha256(row[col].encode()).hexdigest() for col in P.REQUIRED)
            ds = row["dataset"] if row["dataset"] in P.CORPORA else "unknown_or_missing"
            lab = str(P.label(row["is_fake"])) if P.label(row["is_fake"]) is not None else "invalid_or_missing"
            counts[(ds, lab)] += 1
    return rows, duplicates, {f"{ds}/is_fake={lab}": n for (ds, lab), n in counts.items()}


def compare_files(original, reproduced):
    left, ld, lc = fingerprints(original)
    right, rd, rc = fingerprints(reproduced)
    common = left.keys() & right.keys()
    deltas = {col: sum(left[key][i] != right[key][i] for key in common) for i, col in enumerate(P.REQUIRED)}
    oh, rh = P.sha256(original), P.sha256(reproduced)
    return {"original_sha256": oh, "reproduced_sha256": rh, "byte_identical": oh == rh,
            "original_counts": lc, "reproduced_counts": rc,
            "original_duplicate_identifier_extra_rows": ld, "reproduced_duplicate_identifier_extra_rows": rd,
            "original_only_identifiers": safe_ids(left.keys() - right.keys()),
            "reproduced_only_identifiers": safe_ids(right.keys() - left.keys()),
            "field_mismatches_for_common_identifiers": deltas,
            "semantic_fields_identical_by_identifier": not ld and not rd and left.keys() == right.keys() and not any(deltas.values())}


def historical_evidence(analysis, historical):
    actual = pd.read_csv(analysis, usecols=["dataset", "doc_id", "is_fake"])
    flags = pd.read_csv(historical / "overlap_flags_per_doc.csv")
    columns = [c for c in flags if c != "doc_id" and "WELFake" in c.split(":")[0].split("~") and not c.endswith(":nearsens")]
    hit = flags[columns].any(axis=1)
    historical_ids = set(flags.loc[flags.doc_id.str.startswith("WELFake-") & ~hit, "doc_id"])
    actual_ids = set(actual.loc[actual.dataset.eq("WELFake_residual"), "doc_id"])
    extra = actual_ids - historical_ids
    result = {"historical_primary_residual_rows": len(historical_ids), "analysis_residual_rows": len(actual_ids),
              "analysis_only_identifiers": safe_ids(extra), "historical_only_identifiers": safe_ids(historical_ids - actual_ids),
              "analysis_only_labels": {safe_ids([key])[0]: int(actual.loc[actual.doc_id.eq(key), "is_fake"].iloc[0]) for key in extra},
              "historical_flags_for_analysis_only": []}
    for _, row in flags.loc[flags.doc_id.isin(extra)].iterrows():
        result["historical_flags_for_analysis_only"].append({"doc_id": safe_ids([row.doc_id])[0],
                                                            "true_flags": [c for c in columns if row[c]]})
    cache = historical / "cleaned_all.parquet"
    if cache.is_file():
        clean = pd.read_parquet(cache, columns=["dataset", "doc_id", "is_fake", "h_text_light", "h_text_strip",
                                               "h_both_light", "h_both_strip", "prefix", "n_words"])
        ref = clean[clean.dataset.ne("WELFake")]
        wel = clean[clean.dataset.eq("WELFake")]
        base = B.exact_prefix_flags(ref, wel)
        light = wel.h_text_light.isin(ref.h_text_light.dropna())
        result["exact_policy_counts_on_historical_cleaned_cache"] = {
            "recorded_builder_exact_or_prefix": int(base.sum()),
            "light_exact_not_flagged_by_builder_exact_or_prefix": int((light & ~base).sum())}
        records = []
        for _, row in clean.loc[clean.doc_id.isin(extra)].iterrows():
            matching = ref[ref.h_text_light.eq(row.h_text_light)]
            records.append({"doc_id": safe_ids([row.doc_id])[0], "is_fake": int(row.is_fake),
                            "matching_reference_identifiers": safe_ids(matching.doc_id),
                            "light_normalized_body_hash": row.h_text_light,
                            "stripped_body_hash_missing": bool(pd.isna(row.h_text_strip)),
                            "stripped_title_body_hash_missing": bool(pd.isna(row.h_both_strip)),
                            "prefix_hash_missing": bool(pd.isna(row.prefix)),
                            "light_normalized_body_word_count": int(row.n_words)})
        result["policy_difference_records"] = records
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-csv", required=True, type=Path)
    parser.add_argument("--historical-dir", type=Path, default=P.REPO / "results/audit/primary")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--reproduced-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new ZIP")
    current_config = dict(seed=A.SEED, shingle_k=A.SHINGLE_K, num_perm=A.NUM_PERM, bands=A.BANDS, rows=A.ROWS,
                          jacc_primary=A.JACC_PRIMARY, jacc_sens=A.JACC_SENS, contain_thr=A.CONTAIN_THR,
                          head_words=A.HEAD_WORDS, prefix_chars=A.PREFIX_CHARS, min_words=A.MIN_WORDS, prereg=A.PREREG)
    recorded = json.loads((args.historical_dir / "config.json").read_text())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_sha256": P.sha256(args.analysis_csv),
              "historical_evidence": historical_evidence(args.analysis_csv, args.historical_dir),
              "construction": "not_compared", "recorded_configuration": recorded,
              "current_configuration": current_config,
              "configuration_differences": {k: {"recorded": recorded.get(k), "current": v}
                                            for k, v in current_config.items() if recorded.get(k) != v},
              "additional_code_constants_not_in_old_config": {"sensitivity_shingle_k": A.SENS_SHINGLE_K,
                  "minimum_containment_shingles": A.MIN_SHINGLES_FOR_CONTAINMENT, "max_lsh_bucket": A.MAX_BUCKET,
                  "builder_exact_hash_columns": list(B.EXACT_OVERLAP_COLUMNS)},
              "current_environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
              "limitations": ["Original raw-source hashes were not recorded; current source hashes cannot prove original source snapshot identity.",
                              "Historical flags and cleaned cache are evidence from the earlier audit, not an independent overlap search.",
                              "Comparing an existing construction does not rerun the builder in this command.",
                              "No article-veracity validation or model fitting is performed."]}
    hashed = {}
    for name in ("config.json", "within_dedup_report.csv", "welfake_dry_run.csv", "triple_a_primary.csv",
                 "overlap_flags_per_doc.csv", "cleaned_all.parquet"):
        path = args.historical_dir / name
        if path.is_file():
            hashed[f"historical/{name}"] = P.sha256(path)
    for name in ("audit_01_05.py", "build_analysis_set.py", "construction_evidence.py"):
        hashed[f"code/{name}"] = P.sha256(Path(__file__).with_name(name))
    if args.source_dir:
        report["current_source_metadata"] = {}
        csv.field_size_limit(100_000_000)
        for corpus, name in (("ISOT", "Fake.csv"), ("ISOT", "True.csv"),
                             ("McIntire", "fake_or_real_news.csv"), ("WELFake", "WELFake_Dataset.csv")):
            path = args.source_dir / name
            if not path.is_file():
                path = args.source_dir / corpus / name
            key = f"current_sources/{corpus}/{name}"
            hashed[key] = P.sha256(path)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                next(reader)
                n_rows = sum(1 for row in reader if row)
            report["current_source_metadata"][key] = {"bytes": path.stat().st_size, "rows": n_rows,
                                                      "sha256": hashed[key]}
    report["evidence_hashes"] = hashed
    if args.reproduced_csv:
        report["construction"] = "compared_existing_separate_output"
        report["construction_comparison"] = compare_files(args.analysis_csv, args.reproduced_csv)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("construction_evidence.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"historical_residual": report["historical_evidence"]["historical_primary_residual_rows"],
                      "analysis_residual": report["historical_evidence"]["analysis_residual_rows"],
                      "construction": report["construction"], "summary_zip_bytes": args.output.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
