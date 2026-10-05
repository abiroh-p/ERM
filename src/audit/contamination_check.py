import pandas as pd

ANALYSIS = "analysis_set.csv"
FLAGGED = "flagged_doc_ids.txt"

# Load flagged IDs
with open(FLAGGED, "r", encoding="utf-8") as f:
    flagged = {line.strip() for line in f if line.strip()}

print(f"Flagged IDs loaded: {len(flagged)}")

# Read only the header first
cols = pd.read_csv(ANALYSIS, nrows=0).columns.tolist()

print("\nColumns:")
for c in cols:
    print(" ", c)

# Read in chunks so the 200 MB file doesn't need to fit entirely in RAM
flagged_found = []
total_rows = 0
flagged_rows = 0

for chunk in pd.read_csv(ANALYSIS, chunksize=100_000):
    total_rows += len(chunk)

    # Find likely document-ID column
    id_col = None
    for candidate in ["doc_id", "document_id", "id"]:
        if candidate in chunk.columns:
            id_col = candidate
            break

    if id_col is None:
        raise RuntimeError(
            "Could not find document ID column. "
            f"Available columns: {list(chunk.columns)}"
        )

    mask = chunk[id_col].astype(str).isin(flagged)

    if mask.any():
        found = chunk.loc[mask, id_col].astype(str).tolist()
        flagged_found.extend(found)
        flagged_rows += len(found)

print("\n========== RESULT ==========")
print(f"Total analysis rows : {total_rows}")
print(f"Flagged rows found  : {flagged_rows}")
print(f"Unique flagged IDs  : {len(set(flagged_found))}")
print(f"Flagged IDs missing : {len(flagged - set(flagged_found))}")

missing = sorted(flagged - set(flagged_found))

if missing:
    print("\nFirst missing IDs:")
    for x in missing[:20]:
        print(x)