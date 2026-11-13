# Celebrity audit and suitability assessment

Audit completed against all 61,197 frozen source articles on 12 November 2026.
No external model scores were examined. Git began clean at `754c8e6`; after a
successful remote fetch, HEAD and origin/main had no commits ahead/behind. No
update was necessary. No applicable `AGENTS.md` exists in the repository or its
workspace parent. Existing corpora, caches, results, account ZIPs and manuscript
were preserved. Nothing was committed, pushed or submitted.

## Verified inputs and counts

The supplied ZIP hash is
`649dd52b0378b0f95a68afd4e55cfe8e3c70a6bd9d647e10194ebb69fb562f77`.
Only `fakeNewsDatasets/celebrityDataset/` contributes articles. The top-level
README is retained as a notice; FakeNewsAMT/other folders never enter the audit
or features. Inventory contains every original path, byte hash, stable article
ID, binary label and provisional numeric story group.

| Check | Observed |
|---|---:|
| Fake / legitimate articles | 250 / 250 |
| Numeric groups | 250; each contains one file per label |
| UTF-8 decoding failures / empty files | 0 / 0 |
| Articles below 20 whitespace words | 0 (minimum 28) |
| Missing headline / body under declared parsing | 0 / 1 |
| Raw-byte / light-normalized exact duplicate pairs | 0 / 0 |
| Internal full-text 5-shingle primary candidates | 0 |
| All internal body/combined, full/head primary/sensitivity evidence | 0 |
| Cross-corpus evidence rows / distinct target articles | 9 / 2 |
| Unusually long articles (>3,000 words) | 4, all legitimate |

Independent preliminary replication also found **zero** full-text light-normalized
5-shingle pairs with Jaccard ≥0.8 **or containment ≥0.9 without a minimum-shingle
restriction**. The established audit additionally checks artifact-stripped body
and combined fields, head-100 views and the 3-shingle sensitivity tier. There
are no internal cross-label duplicate candidates. This supports the preliminary
observations, but does not establish semantic independence.

Median full-file length is 315.5 words (quartiles 213.25, 431); mean 424.102,
maximum 14,777. The four long legitimate files are 060 (14,777 words), 112
(4,356), 144 (5,109) and 195 (3,600). They are retained, with no truncation at
preparation. Transformer token truncation follows the original 256-token recipe.
There are zero preexisting Unicode replacement characters; successful UTF-8
decoding nevertheless does not establish that original typography is intact.

First nonblank physical line is treated as headline; the remaining suffix is
body, preserving original bytes in the ZIP and full decoded feature text.
`fake/131fake.txt` consists of one 28-word line about wedding attire, with no body
under that rule. It is quarantined, explicitly, rather than silently rewritten.

## Cross-corpus overlap and exclusions

The actual `data/processed/analysis_set.csv` matches the recorded hash
`29e96d7b7fe60a94a2652c704f2bd4be9b5bb7e45f540eabeb19320276788ab5`.
Screened rows: ISOT 38,030; McIntire 6,005; WELFake residual 17,162. The screen
covers every source row, including rows absent from saved training selections.
No original source corpus is changed.

| Celebrity file | Frozen source ID | Evidence | Labels (target/source) |
|---|---|---|---|
| legit/062legit.txt | ISOT-12805 | Body prefix300; primary full containment .9373; primary head Jaccard .8641; sensitivity full/head | 0 / 1 |
| fake/062fake.txt | WELFake-58856 | Body primary full containment .9025; sensitivity body/combined full Jaccard .5344/.5370 | 1 / 1 |

Manual inspection confirmed shared content for both article pairs: the ISOT
article adapts a shorter version of the same report; the residual article shares
many passages of the same RuPaul allegation with additions/deletions. These are
**confirmed shared-content overlaps**, not byte-identical duplicates or an
independent adjudication of truth. The conflicting ISOT/Celebrity labels remain
unresolved. No McIntire candidates were detected. Exact full-body/combined matches
were absent. In candidate CSV rows, prefix/exact scores of 1 refer to equality
of that method's compared string; they do not imply full-article Jaccard 1.

Raw detector rows retain `unreviewed_candidate`; the separate
`audit_v1/manual_reviews.csv` records the two subsequent manual confirmations.
`audit_summary.json` is the detector-stage summary (its manual-confirmation count
is zero), not a substitute for that review file. All candidate scores and IDs are
available in `cross_candidates.csv`; no article excerpts are redistributed.

Frozen conservative policy excludes detected cross-corpus candidates and closes
over provisional numeric groups/internal duplicate components; invalid or
missing-body text is quarantined with the same closure. Thus:

| Exclusion reason | Groups | Articles (fake / legitimate) |
|---|---|---:|
| Detected and subsequently confirmed shared content | 062 | 2 (1 / 1) |
| Missing-body quarantine, including related partner | 131 | 2 (1 / 1) |
| Total removed | 2 | 4 (2 / 2) |
| Fixed primary target retained | 248 | **496 (248 / 248)** |
| Frozen secondary length-matched target | subset of retained | **428 (214 / 214)** |

All numeric pairs remain provisional except the limited inspection above. Group
closure is conservative; it does not validate every pairing. No model score was
used for exclusions. Publisher/fact-check flags and long files are retained.

## Language, repetition and unresolved suitability limits

Regex inspection aids flagged 36 articles for potential fact-checking/denial
language (18 per label); 167 for listed publisher names (103 fake, 64 legitimate);
and ten for boilerplate prompts (two fake, eight legitimate). These dictionaries
are audit-only, separate from the unchanged model intervention regexes. Counts
describe matching documents, not semantic determinations. Publisher vocabularies
are deliberately nonexhaustive and these flags are not source attribution.
Two normalized body-line hashes (≥8 words) recur across articles, despite no
threshold-level internal duplicate pair. Repeated story/boilerplate candidates
remain traceable through `repeated_lines.json` without reproducing the lines.

Legitimate file 060 identifies its interview as from the January 31, 1974 issue
of Rolling Stone. The dataset cannot be described as uniformly recent or a
temporal holdout. The supplied README describes paired web collection and
manual verification using gossip-checking sources. This does not independently
validate all supplied labels. The headline-only fake file, label conflict and
potential fact-checking language warrant explicit discussion in any manuscript.

After exclusions the dataset is usable as a **qualified, small, domain-specific
external test**, subject to label and collection limits. It is not an independent
fourth training corpus. The lexical screen cannot exclude paraphrased/syndicated
stories below thresholds, all related entities/events, or unobserved overlap.
Body recovery from frozen source text follows the builder's first-newline
separator; original raw field metadata is not silently substituted. No universal
generalization, strict independence, or causal attribution claim is warranted.

## Prepared implementation and remaining work

All source manifest contents match **24/24 archived model/seed records** from
the original three account ZIPs, despite a different local numerical environment.
The eight source manifests and both target selections are frozen and shared
across all models. See `prepared_v1/verification.json` for original archive hashes,
individual parity checks, source sizes, target counts and the independent screen.
Source NER cache hash also matches the saved paired protocol. No usable trained
task checkpoints were found; archived metric CSVs cannot be evaluated as models.

The English spaCy model is absent locally; network-restricted download did not
complete. Building the separate NER cache is the only uncompleted CPU preparation
step before planning actual model runs. Its command is in the README. Source
cache is untouched and regex fallback is rejected. Historical exact spaCy model
version remains unknown; actual new extraction versions must be logged.

No full TF-IDF/GPU evaluation or paid job was launched. Validation uses synthetic
text only; no placeholder scores or manuscript edits were made. The default grid
needs 240 fits/model (720 overall); the eight-condition subset plus SE/R_SE is
frozen before scores. Each source transfer and primary/secondary analysis remains
separate. Seed variability will not be represented as independent target data.
The scripts, configuration, protocol, text-free evidence, patch and runnable ZIP
provide a usable experiment even if later label review leads to rejecting this
dataset from the manuscript.
