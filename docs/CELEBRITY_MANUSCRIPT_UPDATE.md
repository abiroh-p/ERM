# Celebrity manuscript integration - 13 November 2026

Base repository: f96905cba35e8526a00bf046555af98461da87de. No commit, push or journal submission has been performed by this update.

The manuscript adds a separate external-validation section, primary comparison table and source-specific entity figure. Appendix B adds baseline, secondary diagnostic and all per-seed contrasts; the archived appendix becomes C through automatic numbering. All original three-corpus numerical tables, metrics and figures remain unchanged. The archived appendix introduction now points to both analysis stages without changing its numerical findings.

Abstract, introduction/objective O4, discussion, limitations, conclusion, future work and declarations now reflect Celebrity as a small held-out external target, not a fourth training corpus or a temporal holdout. The new citation is Perez-Rosas et al., COLING 2018, pp. 3391-3401, checked against https://aclanthology.org/C18-1287/.

External evidence: 720 completed fits, 240 per model; 496 primary target articles and 428 secondary length-matched target articles. All three models reuse the frozen source selections and target IDs within each setting. Main tables show means and sample SD; the new figure shows observed seed minima/maxima, not confidence intervals. No external pooling, new p-value family or article/story uncertainty estimate was added. Positive means are not labelled significant. The raw-gain-versus-control examples are preserved, including McIntire-trained RoBERTa (C0 .6521, E .7365, R_E .7566).

The repository retains the exact uploaded result ZIP in results/external_celebrity/accounts/. scripts/rebuild_external_celebrity_results.py verifies saved fit/protocol/selection integrity and recomputes all 288 contrasts, 36 summaries and 90 source-condition AUC summaries. scripts/generate_external_celebrity_tables.py generates all six new numerical tables and the figure from those rebuilt values. A deliberately corrupted fit is rejected. The previous independent review and the rebuild agree on all contrast summaries.

The operational cache guard and --max-jobs pilot limit previously delivered with the run-ready package are included, as are packaging fixes for already-tracked files and standalone-package guidance. Frozen audit bytes are restored exactly from the result archive, and Git attributes preserve their hashes across checkouts. No input dataset, source-cache span, masking rule, model recipe, frozen scientific configuration, source selection or reported outcome is modified.

Online Resource 1 remains the unchanged three-corpus reproducibility archive. Online Resource 2 (ESM_2.zip) supplies the separate external result archive, protocol, text-free audit/selection records, code and CPU reconstruction instructions. It excludes raw news text, source data ZIPs, span caches and model weights. Its extracted CPU rebuild matches all five canonical numerical/verification files exactly.

Rebuild on CPU:

    python scripts/rebuild_external_celebrity_results.py
    python scripts/generate_external_celebrity_tables.py
    python scripts/build_manuscript.py

The last command requires a functioning LaTeX/BibTeX installation. The manuscript source ZIP is flat, includes the compiled bibliography and PDF, and is independently compile-checked after extraction. See CELEBRITY_MANUSCRIPT_VALIDATION.json for the final check record.

Scope limits: saved-output integrity and aggregation were checked; GPU training, source-data overlap and all labels were not independently reproduced. Per-article predicted scores were not exported. Target NER metadata is recorded, but complete historical source-cache generation and pretrained checkpoint revision identity are not reconstructed.
