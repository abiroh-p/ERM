# Response to the supplied detailed revision checklist

Revision: ERM-review-20261112. Review assessed against the current manuscript, experiment code, saved paired-account outputs and existing reference audit. The 227 checklist items are not 227 established errors.

## Applied changes

| Review items | Decision and action |
|---|---|
| 5-10 | Split the dense audit sentence in the abstract, identify the residual as the third analysis corpus, expand TF-IDF at first use. All principal estimates and qualifications remain. Abstract: 218 whitespace-delimited words. |
| 14, 27, 30, 41, 60-62 | Map research questions to objectives; correct O3 to apply random-deletion comparisons to masking, with L versus C0 for resampling. Define the AUC gap in prose after its equation and correct the formulation/intervention opening: length matching resamples rather than transforms every document's text. |
| 58-59 | Explicitly explain the focused transformer subset and the source-only contrasts it leaves untested. |
| 83-84, 105-109 and the specific Table 5 recommendation | Expand Table 5 from four to all six contrasts. Add Appendix A Table 7 listing each model's six contrasts, sign agreement, exact and Holm-adjusted AUC p-values. Mark unrun transformer contrasts explicitly: no estimated effect/test; p=1 enters correction only. Replace ambiguous 'planned' terminology in paired-result prose with 'descriptive family'. The figure and per-seed tables remain focused subsets. |
| 95 | Explain that adding source masking does not increase the raw TF-IDF gap reduction and that its negative matched-control contrast is consistent with this, without asserting a unique mechanism. |
| 158-162, 223-225 | Prepare Online Resource 1, ESM_1.zip, with code, original paired outputs and manifests, metrics, configurations, audit records and CPU reproduction instructions. Match its revision identifier to the review response. Add an explicit manuscript citation and caption. Exclude article text and entity caches. |
| 164, 226 | Retain approved AI acknowledgements; add a Methods disclosure because the assistance included code/analysis and drafting, beyond copy editing. Use the journal's 'Statements and Declarations' heading. |
| 189, 197 | Fix reference [15]. The publisher style mangled colon-based pages 109:1--109:40. Represent article 109 as a single article identifier with the explicit note 'Article 109, 40 pages'. Retain DOI and the author-hosted published PDF's 2020 date; no invented page range. |
| 201 | Reformat Table 1 from eight dense columns to five, grouping study-design summaries while retaining all nine selected studies, the matched-control distinction and the scoped novelty claim. |

## Already addressed; retained and checked

The frozen analysis input, one surviving normalized exact-match pair, construction-policy discrepancy, absence of both identifiers from all saved paired selections, original-source hash limitation, threshold deviation and constructed residual are already disclosed in Methods, Results and Limitations. These disclosures remain intact.

Archived condition-specific sampling and document-paired validation are already separated. Five-to-eight-seed extension belongs to the archived analysis; the paired validation used the eight fixed seed identifiers from the outset. The extension was not newly relabelled as an independent replication. Shared article identities do not imply shared deletion positions or identical optimization randomness.

The manuscript already documents target-label use in length matching, whole-word versus subword matching, 256-token truncation, post-exploration choices, model- and pair-specific responses, fixed-universe inference, and the absence of a fourth independent corpus. These remain explicit. No negative result or historical analysis was removed.

The existing entry-level reference audit records primary-text checks of the nine selected closest works and metadata checks of all 41 references; this revision does not claim to have reread all papers. The specifically flagged references [23] and [30] retain correct canonical DOIs, and [31] retains its working repository URL labelled as a mirror. Current primary ACL, Springer and GitHub pages were inspected. References [8] and [11] have arXiv IDs matching their DOI suffixes and the previous audit records; fresh arXiv retrieval failed in this pass, so new live resolution is not certified. Hyperlink annotations are present in the compiled PDF. This is not a claim that every external endpoint was freshly tested.

## Disagreement and qualification

- Items asking to explain *why* DistilBERT is worse or random deletion helps cannot be satisfied as causal findings by these experiments. Input-window shifts and regularization-like effects remain possible explanations, not demonstrated mechanisms.
- A nonsignificant RoBERTa contrast is inconclusive, not proof of no benefit or equivalence. Do not strengthen that wording.
- The anonymous-review checks are conditional. The journal page asks for author information and corresponding contact in supplementary files; this review alone does not establish a mandatory anonymous submission. Keep the approved named files and confirm any additional requirements shown by the live submission portal. Do not strip attribution speculatively.
- Agree that an extra dataset or architecture should not be added merely to increase counts. Independently collected or temporally separate evaluation would still be scientifically valuable future work; the present study does not contain it.
- A wholesale introduction/methods rewrite or extra speculative claims would add little. Make the concrete corrections above and preserve limitations and archived outcomes.

## Verification and limits

The fresh saved-output rebuild passed: 312 TF-IDF, 192 DistilBERT and 192 RoBERTa fits; 2,088 evaluation rows. It verifies original account archives, recorded experimental code hashes, selection hashes, cluster separation, analysis-input hash agreement and independently recomputes paired statistics. It does not rerun GPU training, NER or source-data matching.

The supplementary ZIP was extracted into a fresh directory, every included file hash was verified, and its documented CPU rebuild passed. All six checked numerical CSV outputs exactly match the canonical rebuild. Table generation reads these outputs, including the complete descriptive contrast family. No training code or saved experiment input was changed.

The flat manuscript source was compiled through pdflatex, BibTeX and two further pdflatex passes. Final build: 31 pages, with no undefined citation/reference or overfull-box warnings. The PDF is searchable and has active hyperlink annotations. Updated tables and the bibliography were visually inspected; page layout was checked after final compilation. Font embedding is checked with pdffonts; Type 3 fonts in existing plots are embedded glyph fonts, not missing fonts.

The 31-test suite had passed previously. It was not rerun for these manuscript/table-generation changes; the relevant fresh checks are the saved-output rebuild, extracted-archive rebuild and table/PDF validation. Original article text, caches and training results are preserved. No submission, repository push or acceptance is claimed.

## Primary sources consulted in this pass

- Article 109 published PDF, hosted by its author: https://reza.zafarani.net/publications/files/2020-CSUR-FakeNewsSurvey.pdf
- Author companion citation: https://fake-news.site/
- Applied Intelligence instructions: https://link.springer.com/journal/10489/submission-guidelines
- Schuster et al.: https://aclanthology.org/2020.cl-2.8/
- Ahmed et al.: https://link.springer.com/chapter/10.1007/978-3-319-69155-8_9
- Dataset mirror: https://github.com/joolsa/fake_real_news_dataset
