r"""Inline every \input{...} of main_sn.tex into one file (main_sn_flat.tex) for upload to the submission system."""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent
def inline(text):
    def sub(m):
        p = root / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return inline(p.read_text())
    return re.sub(r"\\input\{([^}]*)\}", sub, text)
(root / "main_sn_flat.tex").write_text(inline((root / "main_sn.tex").read_text()))
print("wrote", root / "main_sn_flat.tex")
