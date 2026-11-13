"""Read only the Celebrity ZIP subtree; exhaustive indexed overlap, no text exports.

Matching reuses audit_01_05 normalization, shingles and thresholds. Inverted
shingle postings enumerate every positive-intersection pair, avoiding LSH recall
limits. All scores are verified. The frozen CSV is never modified.
"""
import argparse
from collections import Counter, defaultdict
import json
import pathlib
import re
import sys
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/experiments"))
import paired_sampling as P
import audit_01_05 as A

PREFIX = "fakeNewsDatasets/celebrityDataset/"
CONFIG = ROOT / "configs/external_celebrity_v1.json"
COLUMNS = ["article_id", "other_id", "other_dataset", "label", "other_label", "field", "method", "jaccard", "containment", "cross_label", "manual_status"]
FLAGS = {
    "fact_check": r"\b(?:gossip\s*cop|fact[- ]check\w*|debunk\w*|not true|untrue|false (?:claim|report|rumor|rumour)\w*|no truth|fabricat\w*)\b",
    "publisher": r"\b(?:gossip\s*cop|radar\s*online|radaronline|hollywood\s*life|hollywoodlife|national enquirer|rolling stone|people magazine|entertainment weekly|tmz|vogue|e! news|us weekly)\b",
    "boilerplate": r"\b(?:subscribe|read more|share this|all rights reserved|click here|follow us|advertisement)\b",
    "historical_1974": r"\b1974\b"
}


def read_articles(zip_path):
    """First nonblank physical line is headline; body is the untouched suffix.

    Features use the entire original UTF-8 file, never reconstructed fields.
    Invalid UTF-8 is quarantined, never silently replacement-decoded.
    """
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.startswith(PREFIX) and not n.endswith("/"))
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")
        for name in names:
            m = re.fullmatch(re.escape(PREFIX) + r"(fake|legit)/(\d+)(fake|legit)\.txt", name)
            if not m or m[1] != m[3]:
                raise ValueError(f"unexpected Celebrity member: {name}")
            raw = z.read(name)
            error = ""
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                text, error = "", str(e)
            lines = list(re.finditer(r"[^\r\n]+", text))
            first = next((x for x in lines if x.group().strip()), None)
            title, body = (first.group(), text[first.end():]) if first else ("", "")
            rows.append(dict(doc_id="celebrity-" + P.digest(name)[:20], archive_path=name,
                             file_sha256=P.file_hash_bytes(raw) if hasattr(P, "file_hash_bytes") else __import__("hashlib").sha256(raw).hexdigest(),
                             is_fake=int(m[1] == "fake"), story_group="celebrity-story-" + str(int(m[2])),
                             text=text, title=title, body=body, decoding_error=error,
                             n_words=len(text.split()), body_words=len(body.split()), bytes=len(raw),
                             missing_title=not bool(title.strip()), missing_body=not bool(body.strip())))
    if not rows:
        raise ValueError("Celebrity subtree is empty")
    return rows


def views(row):
    return {"body": row["body"], "title_body": row["text"]}


class Matcher:
    def __init__(self, targets, config):
        self.targets, self.config = targets, config
        self.exact = defaultdict(lambda: defaultdict(list))
        self.posts = defaultdict(lambda: defaultdict(list))
        self.sizes = {}
        for i, row in enumerate(targets):
            for field, text in views(row).items():
                for norm, fn in (("light", A.norm_light), ("strip", A.norm_strip)):
                    value = fn(text)
                    if value:
                        self.exact[(field, "exact_" + norm)][value].append(i)
                    if norm == "strip":
                        if len(value) >= config["prefix_chars"]:
                            self.exact[(field, "prefix300")][value[:config["prefix_chars"]]].append(i)
                        words = value.split()
                        for tier, k in (("primary", config["primary_k"]), ("sensitivity", config["sensitivity_k"])):
                            for view, cut in (("full", None), ("head100", config["head_words"])):
                                key = (field, tier, view)
                                sh = A.shingles(words[:cut], k)
                                self.sizes[(key, i)] = len(sh)
                                for h in sh:
                                    self.posts[key][int(h)].append(i)

    def matches(self, row, internal_index=None):
        c = self.config
        def emit(i, field, method, jac=1., con=1.):
            a = self.targets[i]
            return dict(article_id=a["doc_id"], other_id=row["doc_id"], other_dataset=row.get("dataset", "Celebrity"),
                        label=a["is_fake"], other_label=row["is_fake"], field=field, method=method,
                        jaccard=jac, containment=con, cross_label=a["is_fake"] != row["is_fake"], manual_status="unreviewed_candidate")
        for field, text in views(row).items():
            for norm, fn in (("light", A.norm_light), ("strip", A.norm_strip)):
                value = fn(text)
                if not value:
                    continue
                keys = [("exact_" + norm, value)]
                if norm == "strip" and len(value) >= c["prefix_chars"]:
                    keys.append(("prefix300", value[:c["prefix_chars"]]))
                for method, v in keys:
                    for i in self.exact[(field, method)].get(v, ()):
                        if internal_index is None or i < internal_index:
                            yield emit(i, field, method)
                if norm != "strip":
                    continue
                words = value.split()
                for tier, k, threshold in (("primary", c["primary_k"], c["primary_jaccard"]), ("sensitivity", c["sensitivity_k"], c["sensitivity_jaccard"])):
                    for view, cut in (("full", None), ("head100", c["head_words"])):
                        key = (field, tier, view)
                        sh = A.shingles(words[:cut], k)
                        counts = Counter(i for h in sh for i in self.posts[key].get(int(h), ()) if internal_index is None or i < internal_index)
                        for i, intersection in counts.items():
                            n = self.sizes[(key, i)]
                            minimum = min(n, len(sh))
                            jac = intersection / (n + len(sh) - intersection)
                            con = intersection / minimum if minimum else 0.
                            if jac >= threshold or (con >= c["containment"] and minimum >= c["containment_min_shingles"]):
                                yield emit(i, field, f"{tier}_{view}", jac, con)


def component_exclusions(rows, internal, cross):
    """Close conservatively over numeric groups AND internal candidate components."""
    uf = A.UnionFind(len(rows))
    ids = {r["doc_id"]: i for i, r in enumerate(rows)}
    groups = {}
    for i, r in enumerate(rows):
        if r["story_group"] in groups:
            uf.union(i, groups[r["story_group"]])
        groups[r["story_group"]] = i
    for r in internal:
        uf.union(ids[r["article_id"]], ids[r["other_id"]])
    reasons = defaultdict(set)
    for r in cross:
        reasons[uf.find(ids[r["article_id"]])].add("detected_cross_corpus_candidate")
    for i, r in enumerate(rows):
        if r["decoding_error"] or r["n_words"] < 20 or r["missing_body"]:
            reasons[uf.find(i)].add("invalid_or_short_text")
    # Conflicting exact/primary duplicate labels are not resolved using target performance.
    for r in internal:
        if r["cross_label"] and r["method"].startswith(("exact_", "primary_")):
            reasons[uf.find(ids[r["article_id"]])].add("internal_duplicate_label_conflict")
    return [dict(article_id=r["doc_id"], story_group=r["story_group"],
                 component="celebrity-component-" + str(min(j for j in range(len(rows)) if uf.find(j) == uf.find(i))),
                 excluded=bool(reasons[uf.find(i)]), reasons=";".join(sorted(reasons[uf.find(i)]))) for i, r in enumerate(rows)]


def audit(zip_path, analysis, out, config_path=CONFIG):
    cfg = json.loads(pathlib.Path(config_path).read_text())
    out = pathlib.Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("audit destination must be new/empty; preserve previous audit")
    out.mkdir(parents=True, exist_ok=True)
    zip_hash = P.file_hash(zip_path)
    if zip_hash != cfg["zip_sha256"]:
        raise ValueError("ZIP hash differs from frozen configuration")
    rows = read_articles(zip_path)
    matcher = Matcher(rows, cfg["near"])
    internal = [m for i, r in enumerate(rows) for m in matcher.matches(r, i)]
    pd.DataFrame(internal, columns=COLUMNS).to_csv(out / "internal_candidates.csv", index=False)
    cross, analysis_hash, n_source, corpora = [], None, 0, Counter()
    if analysis and pathlib.Path(analysis).exists():
        analysis_hash = P.file_hash(analysis)
        if analysis_hash != cfg["analysis_sha256"]:
            raise ValueError("analysis CSV hash mismatch: no substitute accepted")
        for chunk in pd.read_csv(analysis, chunksize=1000, keep_default_na=False):
            for r in chunk.to_dict("records"):
                # Builder writes original title + '\n' + original body. Missing separator is a schema error.
                if "\n" not in r["text"]:
                    raise ValueError("frozen CSV title/body separator missing")
                r["title"], r["body"] = r["text"].split("\n", 1)
                cross.extend(matcher.matches(r))
                corpora[r["dataset"]] += 1
                n_source += 1
            if n_source % 5000 == 0:
                print(f"cross overlap: {n_source} source articles; {len(cross)} evidence rows", flush=True)
        if set(corpora) != set(cfg["sources"]):
            raise ValueError("unexpected source corpus grid")
    pd.DataFrame(cross, columns=COLUMNS).to_csv(out / "cross_candidates.csv", index=False)
    exclusions = component_exclusions(rows, internal, cross)
    pd.DataFrame(exclusions).to_csv(out / "eligibility.csv", index=False)
    inventory, flags, lines = [], [], defaultdict(set)
    for r in rows:
        inventory.append({k: v for k, v in r.items() if k not in {"text", "title", "body"}})
        for kind, pattern in FLAGS.items():
            found = list(re.finditer(pattern, r["text"], re.I))
            if found:
                flags.append(dict(article_id=r["doc_id"], kind=kind, count=len(found),
                                  matched_terms=";".join(sorted({m.group().lower() for m in found}))))
        # Hash repeated nontrivial physical lines; no text redistribution.
        for line in r["body"].splitlines():
            v = A.norm_light(line)
            if len(v.split()) >= 8:
                lines[A.h128(v)].add(r["doc_id"])
    pd.DataFrame(inventory).to_csv(out / "inventory.csv", index=False)
    pd.DataFrame(flags, columns=["article_id", "kind", "count", "matched_terms"]).to_csv(out / "language_flags.csv", index=False)
    P.atomic_json(out / "repeated_lines.json", {h: sorted(ids) for h, ids in lines.items() if len(ids) > 1})
    with zipfile.ZipFile(zip_path) as z:
        (out / "SUPPLIED_README.txt").write_bytes(z.read("fakeNewsDatasets/README"))
    keep = {r["article_id"] for r in exclusions if not r["excluded"]}
    counts = lambda rr: dict(Counter("fake" if r["is_fake"] else "legitimate" for r in rr))
    full_primary = [r for r in internal if r["method"] == "primary_full" and r["field"] == "title_body"]
    summary = dict(protocol=cfg["protocol"], zip_sha256=zip_hash, analysis_sha256=analysis_hash,
                   config_sha256=P.file_hash(config_path), cross_corpus_complete=analysis_hash is not None,
                   source_rows=dict(corpora), before=counts(rows), after=counts([r for r in rows if r["doc_id"] in keep]),
                   excluded=len(rows)-len(keep), utf8_errors=sum(bool(r["decoding_error"]) for r in rows),
                   empty=sum(not r["text"].strip() for r in rows), under_20_words=sum(r["n_words"] < 20 for r in rows),
                   missing_body=sum(r["missing_body"] for r in rows), missing_title=sum(r["missing_title"] for r in rows),
                   word_lengths=pd.Series([r["n_words"] for r in rows]).describe().to_dict(),
                   long_articles=[r["doc_id"] for r in rows if r["n_words"] > cfg["long_article_words"]],
                   exact_full_pairs=len({(r["article_id"],r["other_id"]) for r in internal if r["method"] == "exact_light" and r["field"] == "title_body"}),
                   full_5shingle_primary_pairs=len(full_primary), internal_evidence=len(internal), cross_evidence=len(cross),
                   cross_articles=len({r["article_id"] for r in cross}),
                   internal_cross_label_pairs=len({(r["article_id"],r["other_id"]) for r in internal if r["cross_label"]}),
                   language_flags=dict(Counter(r["kind"] for r in flags)),
                   provisional_story_groups=len({r["story_group"] for r in rows}),
                   manually_confirmed_overlap=0,
                   limitations=["Numeric pairs are provisional, not semantic verification.", "Lexical screens cannot establish independence or label correctness.", "1974 interview precludes a temporal holdout claim.", "Source title/body split follows builder's first newline; raw source fields not substituted."])
    P.atomic_json(out / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--analysis-csv")
    ap.add_argument("--out", default="results/external_celebrity/audit_v1")
    ap.add_argument("--config", default=str(CONFIG))
    a = ap.parse_args()
    audit(a.zip, a.analysis_csv, a.out, a.config)
