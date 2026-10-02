"""
00_inspect_schema.py  --  run this FIRST, paste the output back.

Usage:
    python 00_inspect_schema.py <path> [<path> ...]

Each <path> may be a .csv, a .zip containing CSVs, or a folder containing CSVs.
Prints everything needed to write loaders and to confirm WHICH Kaggle
dataset each file really is (ISOT vs McIntire vs Kaggle competition set).
"""
import sys, io, zipfile, hashlib, pathlib
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)


def iter_csvs(path):
    p = pathlib.Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*.csv")):
            yield str(f), f.read_bytes()
    elif p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    yield f"{p.name}::{n}", z.read(n)
    elif p.suffix.lower() == ".csv":
        yield str(p), p.read_bytes()
    else:
        print(f"[skip] {path}: not csv/zip/folder")


def describe(name, raw):
    print("=" * 100)
    print("FILE:", name)
    print("bytes:", len(raw), " md5:", hashlib.md5(raw).hexdigest())
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:  # encoding / parser problems are worth knowing about
        print("read_csv failed:", repr(e))
        df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="latin-1", on_bad_lines="skip")
        print("(fell back to latin-1, bad lines skipped)")
    print("shape:", df.shape)
    print("\ncolumns / dtype / nulls / nunique:")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                         "nulls": df.isna().sum(),
                         "nunique": df.nunique(dropna=True)})
    print(info)
    # low-cardinality columns: likely label / subject / category fields
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 20:
            print(f"\nvalue_counts[{c}]:")
            print(df[c].value_counts(dropna=False).head(20))
    # text-like columns: length stats + a peek
    for c in df.select_dtypes(include="object").columns:
        s = df[c].dropna().astype(str)
        if len(s) and s.str.len().median() > 150:
            w = s.str.split().str.len()
            print(f"\ntext column [{c}]: words mean={w.mean():.0f} median={w.median():.0f} "
                  f"p5={w.quantile(.05):.0f} p95={w.quantile(.95):.0f}  empty={int((w == 0).sum())}")
    print("\nfirst 3 rows (cells truncated to 120 chars):")
    cut = lambda x: str(x)[:120]
    head = df.head(3)
    print((head.map(cut) if hasattr(head, "map") else head.applymap(cut)).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        for name, raw in iter_csvs(arg):
            describe(name, raw)
