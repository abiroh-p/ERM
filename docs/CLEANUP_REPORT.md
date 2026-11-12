# Repository cleanup and paired-results integration

Base: `83f807a0da71603347483a05587341ee9724fdd1` (author push: dataset provenance audits and manuscript clarifications).

## Result

The active repository now leads with the audited document-paired evidence and current manuscript. Dataset construction/provenance code and tests from the latest push are retained. No training or raw-data reconstruction was launched, no experiment metric was edited, and no commit, push, release, history rewrite or journal submission was made.

## Removed or relocated

The detailed action map is `docs/cleanup_manifest.json`.

- Removed 25 stale/redundant files: obsolete manuscript source ZIPs and preview files, an operating-system resource file, generated repository inventory, superseded analysis implementations, exact duplicate legacy table/figure exports already preserved in `freeze/outputs/`, and the redundant five-seed DistilBERT slice.
- Relocated 81 historical files: unique old derived result exports to `archive/legacy_derived/results/final/` and the old mixed-environment lockfile to `archive/environment_records/`.
- Preserved both frozen snapshots, original raw metric CSVs, audit tables, duplicate-removal sensitivity results and historical parity records. The frozen manifests continue to record inputs not redistributed with the repository.
- Verified the five-seed file against the eight-seed file field-for-field before removing the redundant slice. Historical duplicate exports were removed only after byte-hash equality with a retained frozen export.

## Added or updated

- Integrated the exact paired experiment implementation used by the supplied account runs, its sampler/analyzer and eight existing paired tests. Kept the latest audit construction code.
- Added `results/paired/`: unchanged source account ZIPs, merged results, saved protocol/configuration records and original selection manifests in the account ZIPs, statistical tables, figures and integrity evidence. Historical cells are not merged into this grid.
- Added reproducible CPU rebuild, table/figure generation and Springer staging/build scripts using repository-relative paths. Avoided a second copy of the paired code and legacy raw results.
- Replaced the old manuscript PDF/source with the audited paired revision, including the one-row construction disclosure, corrected overlap figure and reference audit.
- Updated root/results documentation and environment records. Removed the claim that one unsupported lockfile reproduces every historical environment.
- Extended the streaming provenance audit to discover all nine paired protocol input hashes; added a focused regression test for that discovery.
- Added repository citation metadata without inventing a publication identifier or granting an unapproved code license.

## Verification

- **31 tests passed**, covering historical runner parity/resume, paired sampling/analysis, provenance auditing and construction-policy regressions. A third-party pytest plugin emitted a deprecation warning; no test failed.
- Full CPU rebuild passed: eight seeds, 696 fits and 2,088 evaluation rows. All released merged/statistic CSVs agree with the previously audited manuscript package to numerical precision.
- The updated provenance discovery finds all nine paired protocol hashes, all matching the frozen analysis input.
- The official Springer build succeeds from the clean repository, producing the audited **30-page** manuscript. Extracted text agrees with the previously verified PDF; no undefined citations/references, placeholder question marks or overfull text boxes remain.
- Current-tree CSVs and nested account-result CSVs were checked for article-text columns. No raw article datasets or caches are included in the delivery.
- Python sources parse and `git diff --check` passes. The binary patch is checked against a fresh checkout of the stated base.

## Using the delivery

The ZIP contains a complete cleaned source tree plus an `APPLY_CLEANUP.patch` and application instructions. Applying the patch to the exact clean base handles deletions and moves. Copying the cleaned tree over an old checkout alone would leave stale deleted files behind.

The manuscript remains an author-review revision. See `docs/RELEASE_CHECKLIST.md` for declarations, licensing, historical provenance limits and release/submission steps.
