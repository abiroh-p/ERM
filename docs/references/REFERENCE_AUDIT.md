# Reference and prior-work audit - 12 November 2026

This review accompanies the audit-reconciled, document-paired manuscript. It is a check of the cited records and the selected closest-work claims, not a systematic search proving priority over all published literature.

## Corrections made

- Corrected the WELFake coauthor spelling to **Ivone Amorim** and restored **Samuel R. Bowman** in the annotation-artifacts paper.
- Replaced provisional citations for the SoK, GUIDE and diachronic-bias papers with their published USENIX, KDD and W-NUT proceedings records.
- Completed the Information Sciences volume/article number, the Blackledge page range, and several missing DOIs.
- Used official NeurIPS BibTeX for the TruthOverTricks volume and pages. Crossref and NeurIPS disagree on this paper's pagination; the manuscript follows the official NeurIPS record (141018-141060), not Crossref's 156431-156473.
- Removed unverified historical software version attribution from the general spaCy citation; recorded experiment environments remain described separately.
- Removed stale bibliography comments saying entries were written from memory or that a particular reference-count target was required. No references were added merely to reach a count.
- Corrected the description of the SoK replication: original excerpts and fact-edited versions are distinct tests. The previous sentence conflated them.
- Replaced an exhaustive-looking novelty paragraph with a scoped description of the joint design. Random augmentation and randomized train splits in prior work are acknowledged explicitly and distinguished from per-document deletion-count controls.

## Verification depth and limits

There are 41 cited entries. Relevant sections of all nine selected closest-work papers were inspected through primary full texts: publisher PDFs/HTML or author-deposited preprints. For other entries the review generally checks bibliographic records rather than rereading the entire paper. The table below identifies the route and scope for every entry.

Publisher-deposited Crossref records are bibliographic evidence, not proof that every scientific sentence in an article has been verified. Title-search false positives were rejected rather than imported. In particular, the first Kaufman title search returned the 2011 conference paper rather than the cited 2012 journal article; the correct journal DOI was checked separately.

The Holm record retains its established citation with a stable JSTOR URL; full publisher metadata/text access was limited in this session. The McIntire mirror is correctly identified as a mirror, and its present ownership does not independently prove historical authorship. These provenance boundaries remain explicit. The Verma PDF's header date is inconsistent with the month encoded in its arXiv ID; the reference cites the confirmed ID and year without asserting a submission date.

The detailed table and JSON record which fields or claims were checked. Downloaded third-party full texts were used internally and are not redistributed in this archive. Relevant publisher-deposited metadata, including the conflicting NeurIPS/Crossref page record, are retained for traceability.

## Entry-level checks

| Citation key | Check scope | Primary or deposited source | Finding |
|---|---|---|---|
| ahmed2017isot | Publisher chapter record | [source](https://link.springer.com/chapter/10.1007/978-3-319-69155-8_9) | Authors, year, volume and pp. 127-138 agree. |
| verma2021welfake | Publisher-deposited DOI metadata | [source](https://doi.org/10.1109/tcss.2021.3068519) | Corrected Ivone Amorim; volume 8(4), pp. 881-893 and DOI agree with publisher-deposited record. |
| wang2017liar | ACL metadata | [source](https://aclanthology.org/P17-2067/) | Author, title, venue, pages and DOI agree. |
| shu2020fakenewsnet | Publisher-deposited DOI metadata | [source](https://doi.org/10.1089/big.2020.0062) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| mcintire2017fakeorreal | Dataset mirror inspected | [source](https://github.com/joolsa/fake_real_news_dataset) | Mirror link is valid. Historical McIntire attribution/year retained; mirror ownership does not independently establish original authorship. Raw-file byte hash is in the supplied construction audit. |
| shu2017survey | Publisher-deposited DOI metadata | [source](https://doi.org/10.1145/3137597.3137600) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| zhou2020survey | Publisher-deposited DOI metadata | [source](https://doi.org/10.1145/3395046) | Publisher metadata splits title and subtitle; combined title agrees. Kept article-based page notation rather than mistaking a short metadata title for a mismatch. |
| lazer2018science | Publisher-deposited DOI metadata | [source](https://doi.org/10.1126/science.aao2998) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| asr2019big | Publisher-deposited DOI metadata | [source](https://doi.org/10.1177/2053951719843310) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| janicka2019cross | Publisher-deposited DOI metadata | [source](https://doi.org/10.13053/cys-23-3-3281) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| hoy2022generalisability | Publisher-deposited DOI metadata | [source](https://doi.org/10.1109/bigdata55660.2022.10020583) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| bozarth2020toward | Publisher-deposited DOI metadata | [source](https://doi.org/10.1609/icwsm.v14i1.7279) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| zhu2022endef | Primary content and bibliographic record | [source](https://arxiv.org/pdf/2204.09484) | Primary full text checked: word/entity drop and mask are training augmentations, rather than the matched deletion comparator used here. |
| verma2026accuracy | Primary preprint full text | [source](https://arxiv.org/pdf/2609.25006) | Title and author confirmed. Relevant metadata/source-tag/deduplication, topic/temporal and LIAR sections checked. PDF header month/date is inconsistent with its arXiv ID; no submission date is asserted in the reference. |
| haroon2026benchmarks | Primary preprint v2 full text | [source](https://arxiv.org/pdf/2607.14131v2) | Expanded multilingual title/author and six-corpus design confirmed; length cap, exact-string overlap check and declared WELFake label mapping checked. |
| delloglio2026comparison | Primary content and bibliographic record | [source](https://arxiv.org/html/2603.25501v1) | Primary full text/metadata checked: 12 approaches, 10 datasets and four evaluation groups; added Information Sciences volume 745 and article 123407. |
| wan2025truth | Primary content and bibliographic record | [source](https://papers.nips.cc/paper_files/paper/2025/hash/ce5b4f79f4752b7f8e983a80ebcd9c7a-Abstract-Conference.html) | Primary proceedings PDF and BibTeX checked. Equal-size random training splits are distinct from document-matched random deletion. Official BibTeX gives pp. 141018-141060, differing from Crossref; publisher BibTeX takes precedence. |
| xiao2023sok | Proceedings metadata and primary full text | [source](https://www.usenix.org/conference/usenixsecurity25/presentation/xiao-madelyne) | Converted to USENIX Security 2025 proceedings citation, pp. 5247-5266. Corrected conflation of original news excerpts and fact-edited versions. |
| thibault2024guide | Primary content and bibliographic record | [source](https://arxiv.org/pdf/2411.05060) | Primary paper and ACM-deposited metadata checked: 75 datasets surveyed, 36 claims plus 9 paragraph datasets assessed; converted to KDD 2025 proceedings citation, pp. 5801-5809. |
| blackledge2021transforming | Primary content and bibliographic record | [source](https://arxiv.org/pdf/2109.09796) | Primary author preprint and publisher-deposited metadata checked: opinion filtering has different effects for models trained on ISOT versus Combined Corpus; added pp. 3960-3968. |
| murayama2021diachronic | Primary content and bibliographic record | [source](https://aclanthology.org/2021.wnut-1.21/) | Primary paper and ACL record checked: four corpora, person-name deletion/masking and Wikidata replacements; converted to W-NUT proceedings citation. |
| geirhos2020shortcut | Publisher-deposited DOI metadata | [source](https://doi.org/10.1038/s42256-020-00257-z) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| gururangan2018annotation | ACL metadata | [source](https://aclanthology.org/N18-2017/) | Corrected Samuel R. Bowman; remaining fields agree. |
| mccoy2019right | Publisher-deposited DOI metadata | [source](https://doi.org/10.18653/v1/p19-1334) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| poliak2018hypothesis | Publisher-deposited DOI metadata | [source](https://doi.org/10.18653/v1/s18-2023) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| lee2022dedup | Publisher-deposited DOI metadata | [source](https://doi.org/10.18653/v1/2022.acl-long.577) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| northcutt2021labelerrors | Official proceedings abstract | [source](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract-round1.html) | Author list, title and 2021 Datasets and Benchmarks track confirmed. No page range invented. |
| broder1997resemblance | Publisher-deposited DOI metadata | [source](https://doi.org/10.1109/sequen.1997.666900) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| pedregosa2011sklearn | Publisher metadata | [source](https://www.jmlr.org/papers/v12/pedregosa11a.html) | Full author list, journal volume 12, year and pp. 2825-2830 agree. |
| sanh2019distilbert | Author preprint metadata | [source](https://arxiv.org/abs/1910.01108) | Authors, title and ID confirmed; first submitted 2019, revised 2020. Added arXiv DOI; retain first-publication year. |
| devlin2019bert | ACL metadata | [source](https://aclanthology.org/N19-1423/) | Authors, title, venue, pages and DOI agree. |
| wolf2020transformers | Publisher-deposited DOI metadata | [source](https://doi.org/10.18653/v1/2020.emnlp-demos.6) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| honnibal2020spacy | Maintainer citation file | [source](https://github.com/explosion/spaCy/blob/master/CITATION.cff) | Authors/year/concept DOI agree with maintainer recommendation. Removed unsupported version-specific note; actual environment evidence remains in declarations. |
| holm1979simple | Stable publisher/archive identifier; limited metadata access | [source](https://www.jstor.org/stable/4615733) | Retained established 1979 citation, Scandinavian Journal of Statistics 6(2), 65-70. Added stable URL. Full publisher text/author list was not freshly retrieved; no unsupported DOI added. |
| liu2019roberta | Author preprint metadata | [source](https://arxiv.org/abs/1907.11692) | All ten authors, title, ID and year agree; added arXiv DOI. |
| schuster2020limitations | ACL journal metadata and abstract | [source](https://aclanthology.org/2020.cl-2.8/) | Authors, title, journal and pp. 499-510 agree; added DOI 10.1162/coli_a_00380. |
| niven2019probing | ACL metadata and abstract | [source](https://aclanthology.org/P19-1459/) | Authors, title, venue and pp. 4658-4664 agree; added DOI. |
| torralba2011unbiased | Publisher-deposited DOI metadata | [source](https://doi.org/10.1109/cvpr.2011.5995347) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| kaufman2012leakage | Publisher-deposited DOI metadata | [source](https://doi.org/10.1145/2382577.2382579) | Verified the 2012 four-author journal article and its DOI. Rejected the similarly titled 2011 conference result returned by a title search. |
| kapoor2023leakage | Publisher-deposited DOI metadata | [source](https://doi.org/10.1016/j.patter.2023.100804) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
| gorman2019standard | Publisher-deposited DOI metadata | [source](https://doi.org/10.18653/v1/p19-1267) | Identified cited work through exact DOI or a title/author-compatible record; retained matching bibliographic fields. |
