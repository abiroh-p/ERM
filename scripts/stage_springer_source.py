"""Stage a flat Springer source directory from the editable manuscript."""
from pathlib import Path
import re, shutil
REPO=Path(__file__).resolve().parents[1]
R=REPO/'results/paired'
S=REPO/'paper/sn'
DEST=REPO/'paper/submission'
DEST.mkdir(exist_ok=True)
figure=R/'figures/paired_validation_contrasts.pdf'
shutil.copy2(figure,S.parent/'figs'/figure.name)
if (R/'figures/audit_reconciled_overlap.pdf').exists():
 shutil.copy2(R/'figures/audit_reconciled_overlap.pdf',S.parent/'figs/fig_overlap.pdf')
def expand(text):
 def sub(match):
  name=match.group(1)
  path=S/name
  if not path.suffix:path=path.with_suffix('.tex')
  return expand(path.read_text())
 return re.sub(r'\\input\{([^}]+)\}',sub,text)
s=expand((S/'main_sn.tex').read_text())
s=re.sub(r'\\graphicspath\{\{\.\./figs/\}\{figs/\}\}',r'\\graphicspath{{./}}',s)
# Update comments for this flat source.
s=s.replace('%% Compile: pdflatex main_sn; bibtex main_sn; pdflatex main_sn; pdflatex main_sn.  Content lives in sections/*.tex and abstract.tex.', '%% Compile: pdflatex main; bibtex main; pdflatex main; pdflatex main.')
(DEST/'main.tex').write_text(s)
for name in ['sn-jnl.cls','sn-mathphys-num.bst','refs.bib']:
 shutil.copy2(S/name,DEST/name)
for p in (S.parent/'figs').glob('*.pdf'):shutil.copy2(p,DEST/p.name)
(DEST/'README.txt').write_text('Author-review revision. Compile main.tex with pdflatex, bibtex, pdflatex, pdflatex.\nSee ../../docs/RELEASE_CHECKLIST.md for pre-submission checks.\n')
print('Staged flat source:',DEST)
