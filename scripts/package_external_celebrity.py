"""Build a code/identifier-only runnable ZIP and reviewable repository patch.

Archives, article text, analysis CSV, span caches and model weights are never
included. Does not stage, commit, push or mutate existing experiment archives.
"""
import argparse
import difflib
import hashlib
import json
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW_FILES = ["configs/external_celebrity_v1.json", "docs/EXTERNAL_CELEBRITY_PROTOCOL_v1.md",
             "scripts/external_celebrity.py", "scripts/verify_celebrity_preparation.py",
             "scripts/package_external_celebrity.py", "src/audit/celebrity_audit.py",
             "tests/test_external_celebrity.py"]
MODIFIED = [".gitignore", "README.md", "THIRD_PARTY_NOTICES.md"]
DEPENDENCIES = ["LICENSE", "requirements.txt", "requirements-analysis.txt", "requirements-paired-observed.txt",
                "src/audit/audit_01_05.py", "src/experiments/exp_tfidf.py",
                "src/experiments/exp_transformer.py", "src/experiments/paired_sampling.py"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def files():
    result = list(NEW_FILES)
    for p in sorted((ROOT/"results/external_celebrity").rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
            result.append(p.relative_to(ROOT).as_posix())
    return result


def new_diff(name):
    data = (ROOT/name).read_bytes().decode("utf-8")
    # Git patches use LF, including for supplied CRLF notices. Preserve the
    # original notice bytes in the runnable ZIP and recorded manifest.
    data = data.replace("\r\n", "\n")
    lines = list(difflib.unified_diff([], data.splitlines(keepends=True), fromfile="/dev/null", tofile="b/"+name))
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n\\ No newline at end of file\n"
    return "diff --git a/"+name+" b/"+name+"\nnew file mode 100644\n"+"".join(lines)


def build(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    added = files()
    patch_text = subprocess.run(["git", "diff", "--no-ext-diff", "HEAD", "--", *MODIFIED], cwd=ROOT,
                                check=True, capture_output=True).stdout.decode("utf-8")
    patch_text += "".join(new_diff(n) for n in added)
    patch = out/"external_celebrity_v1.patch"
    patch.write_bytes(patch_text.encode("utf-8"))
    included = sorted(set(added + MODIFIED + DEPENDENCIES))
    for name in included:
        if (ROOT/name).suffix.lower() in {".pkl", ".zip", ".pt", ".bin", ".safetensors"}:
            raise ValueError("forbidden data/cache/model material in package")
    manifest = dict(format="celebrity-code-and-identifiers-v1", files={n:sha((ROOT/n).read_bytes()) for n in included},
                    base_commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),
                    patch_sha256=sha(patch.read_bytes()),
                    excluded="article text, original ZIP, analysis CSV, span caches, model weights; supplied README retained as third-party notice")
    destination = out/"external_celebrity_v1_runnable.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in included:
            # Stable timestamps make identical inputs produce identical packages.
            info = zipfile.ZipInfo(name, date_time=(2026,11,12,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (ROOT/name).read_bytes())
        z.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity failure")
        for name,h in manifest["files"].items():
            if sha(z.read(name)) != h:
                raise ValueError("ZIP member hash failure")
    (out/"SHA256SUMS.txt").write_text(f"{sha(patch.read_bytes())}  {patch.name}\n{sha(destination.read_bytes())}  {destination.name}\n",encoding="utf-8")
    print(json.dumps(dict(files=len(included), patch=str(patch.resolve()), patch_bytes=patch.stat().st_size,
                         zip=str(destination.resolve()), zip_bytes=destination.stat().st_size),indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="deliverables/external_celebrity_v1")
    build(ap.parse_args().out)
