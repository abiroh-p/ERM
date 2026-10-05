"""kwic.py -- show words in context so you can design precise source-marker patterns.
Usage:  python -X utf8 kwic.py <dataset> <fake|real> <word> [n]
e.g.    python -X utf8 kwic.py McIntire fake share 15
"""
import re, sys
import pandas as pd
ds, cls, word = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
df = pd.read_csv("analysis_set.csv", usecols=["dataset", "text", "is_fake"])
g = df[(df.dataset == ds) & (df.is_fake == (1 if cls == "fake" else 0))].sample(frac=1, random_state=1)
rx = re.compile(r"\b" + re.escape(word) + r"\b", re.I)
shown = 0
for t in g.text:
    m = rx.search(t)
    if m:
        a, b = max(0, m.start() - 70), min(len(t), m.end() + 70)
        print("...", t[a:b].replace("\n", " ⏎ "), "...")
        shown += 1
        if shown >= n: break
print(f"[{shown} shown; {int(g.text.str.contains(rx).sum())} of {len(g)} {cls} docs in {ds} contain '{word}']")
