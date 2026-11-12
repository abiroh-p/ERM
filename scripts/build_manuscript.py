"""Stage and compile the official Springer source; requires pdflatex and bibtex."""
from pathlib import Path
import subprocess
import sys
import shutil

REPO = Path(__file__).resolve().parents[1]

def main():
    subprocess.run([sys.executable, str(REPO/'scripts/stage_springer_source.py')], check=True)
    source = REPO/'paper/submission'
    for command in [
        ['pdflatex', '-output-format=pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
        ['bibtex', 'main'],
        ['pdflatex', '-output-format=pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
        ['pdflatex', '-output-format=pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
    ]:
        subprocess.run(command, cwd=source, check=True)
    log = (source/'main.log').read_text(errors='replace')
    if any(value in log for value in ['undefined references', 'undefined citations', 'Overfull']):
        raise SystemExit('Inspect paper/submission/main.log: unresolved references/citations or layout overflow.')
    destination = REPO/'paper/manuscript/ERM_Manuscript.pdf'
    shutil.copy2(source/'main.pdf', destination)
    print('Compiled:', destination)

if __name__ == '__main__':
    main()
