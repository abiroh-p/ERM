# License scope and third-party notices

## ERM original code

The root `LICENSE` applies to the original ERM software and its associated technical documentation. This includes original code under `src/`, `scripts/`, `tests/` and `freeze_and_analyze.py`. The notice uses the collective name "ERM contributors" and does not transfer ownership between contributors.

## Separate materials

The MIT code license does not assign a new license to the research manuscript, paper assets, result/data archives or third-party materials. Source datasets, model weights, software dependencies and any other third-party content retain their own terms. This repository does not redistribute the original news articles, analysis CSV or NER cache.

## Celebrity external-validation input

The separately supplied `fakeNewsDatasets.zip` is third-party data. Its Celebrity
subtree is used locally; article text is excluded from the extension package.
The supplied top-level README is preserved byte-for-byte under
`results/external_celebrity/audit_v1/SUPPLIED_README.txt`, including its original
encoding defects. It requests citation of Pérez-Rosas, Kleinberg, Lefevre and
Mihalcea, *Automatic Detection of Fake News*, COLING 2018. It supplies no explicit
license grant for redistribution of article text; this repository's MIT code
license does not grant rights to those articles or to that README.

## Springer Nature template files

The following files are supplied unchanged from the publisher template:

| File | Terms stated in the file |
|---|---|
| `paper/sn/sn-jnl.cls` | LaTeX Project Public License version 1.3c or later; original notices retained |
| `paper/sn/sn-mathphys-num.bst` | LaTeX Project Public License version 1.0 or later; original notice states that the style file may be used provided it is not changed |

These files remain under those terms, including when copied into the generated flat submission directory. The MIT code license does not replace their notices. A complete TeX distribution supplies the applicable LPPL materials.
