import pandas as pd

INPUT = "analysis_set.csv"
FLAGGED = "flagged_doc_ids.txt"
OUTPUT = "analysis_clean.csv"

with open(FLAGGED, "r", encoding="utf-8") as f:
    flagged = {line.strip() for line in f if line.strip()}

print(f"Flagged IDs: {len(flagged)}")

total = 0
removed = 0

first = True

for chunk in pd.read_csv(INPUT, chunksize=100_000):

    total += len(chunk)

    mask = chunk["doc_id"].astype(str).isin(flagged)

    removed += mask.sum()

    clean = chunk.loc[~mask]

    clean.to_csv(
        OUTPUT,
        mode="w" if first else "a",
        header=first,
        index=False
    )

    first = False

print("\n========== CLEANING COMPLETE ==========")
print(f"Original documents : {total}")
print(f"Removed flagged    : {removed}")
print(f"Clean documents    : {total - removed}")
print(f"Output             : {OUTPUT}")