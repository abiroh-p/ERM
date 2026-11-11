#!/usr/bin/env python3
"""
find_text_in_history.py -- list every file path that EVER held dataset article text in this repository's history.

    python scripts/find_text_in_history.py [--write-paths purge_paths.txt]

Reads each distinct historical blob (not just the current tree) and flags CSV/TXT/JSON/parquet-like files whose header names a text
column (text, title, content, text_i, text_j) or whose first lines contain very long fields. Use the printed paths with
`git filter-repo` (see CLEANUP.md). Heuristic: review the list; it is meant to over-report rather than miss.
"""
import subprocess, re, collections, sys

def run(*a):
    return subprocess.run(["git", *a], capture_output=True).stdout

# every (blob, path) pair that ever appeared in any commit (rev-list --objects only shows one path per blob)
path_of = {}
log = run("log", "--all", "--raw", "--no-abbrev", "--no-renames", "--format=").decode(errors="replace").splitlines()
for line in log:
    if not line.startswith(":"):
        continue
    meta, _, path = line.partition("\t")
    new_sha = meta.split()[3]
    if set(new_sha) != {"0"}:
        path_of.setdefault(new_sha, set()).add(path)
batch = run("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)").decode().splitlines()
hits = collections.defaultdict(list)
for line in batch:
    sha, typ, size = line.split(); size = int(size)
    if typ != "blob" or sha not in path_of or size < 5000:
        continue
    paths = path_of[sha]
    if not any(p.lower().endswith((".csv", ".txt", ".json", ".tsv", ".parquet", ".md")) for p in paths):
        continue
    head = subprocess.run(["git", "cat-file", "blob", sha], capture_output=True).stdout[:6000]
    if head[:4] == b"PAR1":
        flag = "parquet"
    else:
        txt = head.decode("utf-8", errors="replace")
        first = txt.splitlines()[0] if txt.splitlines() else ""
        cols = [c.strip().strip('"').lower() for c in re.split(r"[,\t]", first)]
        text_col = any(c in ("text", "title", "content", "body", "text_i", "text_j") for c in cols)
        long_fields = any(len(l) > 1500 for l in txt.splitlines()[1:4])
        label_check = any("label_check" in p for p in paths)
        flag = "text column" if text_col else ("long lines" if long_fields else ("excerpts" if label_check else ""))
    if flag:
        for p in paths:
            hits[p].append((size, flag))
for p, v in sorted(hits.items(), key=lambda kv: -max(s for s, _ in kv[1])):
    print(f"{max(s for s, _ in v):>12,}  {v[0][1]:12s} {p}")
print(f"\n{len(hits)} paths; purge them with git filter-repo (see CLEANUP.md)")
if "--write-paths" in sys.argv:
    out = sys.argv[sys.argv.index("--write-paths") + 1]
    open(out, "w").write("\n".join(sorted(hits)) + "\n"); print("wrote", out)
