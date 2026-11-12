# Dataset handling

Raw article files, processed analysis CSVs and NER caches are local inputs, excluded from the repository. Saved metrics, identifiers, labels, cluster IDs and configuration/hash records are retained for reproducibility; they are not full article text.

The current source tree was checked for dataset filenames, article-text CSV columns, runtime caches and nested repository metadata during cleanup. This is a current-tree/package check, not a claim about every earlier clone or remote Git object.

`scripts/repo_cleanup.py` and `scripts/find_text_in_history.py` are historical maintenance tools. They are not part of the paired experiment pipeline. No history rewrite or remote push is performed by this cleanup.

The two historical frozen snapshots remain unchanged, including metadata naming inputs that are deliberately not redistributed. Read `docs/reports/dataset_provenance_20261112.md` before interpreting differences between historical audit counts and the constructed experiment file.
