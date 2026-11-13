# Validation record

All tests and smoke fits used synthetic text. No Celebrity scores or source
retraining results were produced during preparation.

Completed checks:

- SHA-256 verification of the original data ZIP, exact frozen analysis CSV and
  original source span cache: all match their supplied/saved hashes.
- Full exhaustive cross-corpus audit: 61,197 source rows, nine evidence records,
  two distinct target articles; original inputs were read only.
- Independent full-text 5-shingle screen, without the containment minimum:
  zero internal candidates. Raw-byte and normalized exact duplicates: zero.
- All eight regenerated source manifests match all 24 model/seed manifests
  saved in the three original account archives.
- `python -m pytest tests -q -p no:cacheprovider --basetemp
  runs/external_celebrity_full_tests_01`: **36 passed**, 16 preexisting pandas
  deprecation warnings, 163.78 seconds.
- After additional extension integrity/coverage checks,
  `python -m pytest tests/test_external_celebrity.py -q -p no:cacheprovider
  --basetemp runs/external_celebrity_test_02`: **7 passed**, 7.26 seconds.
- Final complete repository suite, including the new checks:
  `python -m pytest tests -q -p no:cacheprovider --basetemp
  runs/external_celebrity_full_tests_02`: **38 passed**, the same 16 preexisting
  pandas deprecation warnings, 102.96 seconds.
- Runnable ZIP extracted into a fresh directory, all 42 member-file hashes
  verified, ZIP decompression verified; its standalone extension tests:
  **7 passed**, 7.05 seconds. Final documentation-only package rebuild uses
  identical tested code/test bytes (verified by hash).
- Repository patch passed `git apply --check --whitespace=nowarn` against
  a staged copy of the original tracked files with all new paths absent.
- `git diff --check`: passed for tracked changes (Git reports routine line-ending
  conversion notices).

The extension smoke test fits ten synthetic TF-IDF conditions with two worker
assignments, verifies incomplete then complete status, resumes without changing
finished files, computes all four matched contrasts, then refits the same ten
synthetic jobs in a serial assignment and checks identical metrics/deletion
streams. It corrupts one completed record to verify rejection. Other tests cover
empty normalization, all-word deletion, exact scores against the original
scorer, cross-label detection, group-closure exclusions, containment minimums,
ZIP subtree isolation, original line-break preservation, label/path disagreement,
fixed target manifests and blocked evaluation after incomplete/changed audits.

The production NER command was exercised: it correctly refused to replace the
missing spaCy model with regex NER. An attempted model download failed under
network restrictions; the subsequent download request did not complete. No
separate production span cache or evaluation plan was created. Existing source
cache was not modified. Install the model and run the documented CPU NER stage
before actual model evaluation. No GPU fit or paid job was launched.

Observed local environment: Python 3.12.4; NumPy 2.5.4; pandas 3.1.1;
scikit-learn 1.9.1; SciPy 1.18.1; spaCy 3.8.18. This differs from the saved paired
training environment. Source selection parity was checked directly rather than
inferred from these versions. Actual evaluation plans freeze installed versions
and input/code hashes. Historical complete NER/model-version equivalence remains
unestablished.

The runnable ZIP builder hashes every included member, verifies decompression
and writes SHA-256 checksums for the package and patch. Separate package-check
records are delivered alongside the ZIP after extraction/testing and patch checks.
