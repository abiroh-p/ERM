"""Supplement the audit with an unrestricted full-text screen and archive parity.

Does not score models, infer ground-truth labels or export article text.
"""
import argparse
from collections import Counter, defaultdict
import json
import pathlib
import sys
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src/audit"), str(ROOT/"src/experiments"), str(ROOT/"scripts")]
import celebrity_audit as A
import external_celebrity as R
P = R.P


def preliminary_screen(rows):
    """Independent all-pair full-text light-normalized 5-shingle screen.

    Unlike the established audit, containment has NO 50-shingle minimum here,
    matching the broader preliminary claim supplied in the task.
    """
    posts, sizes, found = defaultdict(list), [], []
    for i, r in enumerate(rows):
        sh = A.A.shingles(A.A.norm_light(r["text"]).split(), 5)
        counts = Counter(j for h in sh for j in posts[int(h)])
        for j, intersection in counts.items():
            minimum = min(len(sh), sizes[j])
            jac = intersection/(len(sh)+sizes[j]-intersection)
            con = intersection/minimum if minimum else 0
            if jac >= .8 or con >= .9:
                found.append(dict(article_id=r["doc_id"], other_id=rows[j]["doc_id"], jaccard=jac, containment=con))
        sizes.append(len(sh))
        for h in sh:
            posts[int(h)].append(i)
    return found


def verify(a):
    c, target, index = R.checked_selections(a)
    rows = A.read_articles(a.zip)
    screen = preliminary_screen(rows)
    parity = []
    archives = pathlib.Path(a.paired_archives)
    for path in sorted(archives.glob("ERM_Paired_Account_*.zip")):
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if "_samples/seed_" not in name or not name.endswith(".json"):
                    continue
                archived = json.loads(z.read(name))
                seed = archived["seed"]
                expected = json.loads((pathlib.Path(a.prepared)/f"source_seed_{seed}.json").read_text())
                equal = expected == archived
                parity.append(dict(archive=path.name, member=name, seed=seed, identical=equal,
                                   archive_sha256=P.file_hash(path), manifest_sha256=archived["manifest_sha256"]))
                if not equal:
                    raise ValueError(f"source selection differs from saved paired study: {name}")
    source_sizes = []
    for seed in c["seeds"]:
        manifest = P.read_manifest(pathlib.Path(a.prepared)/f"source_seed_{seed}.json", index["source_manifests"][str(seed)])
        for sample in manifest["samples"]:
            if sample["role"] == "tr":
                source_sizes.append(dict(seed=seed, source=sample["dataset"], length_matched=sample["length_matched"], n_train=len(sample["rows"])))
    inventory = pd.DataFrame(rows)
    flags = pd.read_csv(pathlib.Path(a.audit)/"language_flags.csv")
    by_label = flags.merge(inventory[["doc_id", "is_fake"]], left_on="article_id", right_on="doc_id").groupby(["kind", "is_fake"]).size()
    value = dict(preliminary_full_text_5shingle_candidates=screen,
                 primary_target_articles=len(target["primary"]["rows"]), secondary_target_articles=len(target["secondary"]["rows"]),
                 source_manifest_parity=parity, parity_complete=len(parity)==24,
                 source_train_sizes=source_sizes,
                 flag_counts_by_label=[dict(kind=k, is_fake=int(y), articles=int(n)) for (k,y),n in by_label.items()],
                 preexisting_replacement_character_articles=sum("\ufffd" in r["text"] for r in rows),
                 preexisting_replacement_character_count=sum(r["text"].count("\ufffd") for r in rows),
                 exact_raw_byte_duplicate_groups=sum(n>1 for n in Counter(r["file_sha256"] for r in rows).values()),
                 missing_headline_separator=[r["doc_id"] for r in rows if not r["body"].strip()],
                 repeated_nontrivial_line_hashes=len(json.loads((pathlib.Path(a.audit)/"repeated_lines.json").read_text())))
    R.freeze(pathlib.Path(a.prepared)/"verification.json", value)
    print(json.dumps({k:v for k,v in value.items() if k not in {"source_manifest_parity","source_train_sizes"}}, indent=2))
    print(f"Archived source manifest parity: {sum(r['identical'] for r in parity)}/{len(parity)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True)
    p.add_argument("--analysis-csv", default="data/processed/analysis_set.csv")
    p.add_argument("--config", default=str(A.CONFIG))
    p.add_argument("--audit", default="results/external_celebrity/audit_v1")
    p.add_argument("--prepared", default="results/external_celebrity/prepared_v1")
    p.add_argument("--paired-archives", default="results/paired/accounts")
    verify(p.parse_args())
