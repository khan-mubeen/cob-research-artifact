"""package the current review files without git history or local metadata."""
import argparse
import hashlib
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {
    ".git", ".venv", ".venv-audit", ".venv-report", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".run", "build", "dist",
    "submission", "models", "tools", "Python", "prompt-reports", "local_diagnostics",
    "unsloth_compiled_cache", "_incoming",
}
PRIVATE = {
    "docs/meeting-notes", "docs/cob_plan_milestones.pdf",
    "data/raw/DataCoSupplyChainDataset.csv", "data/finetune",
}

def payload():
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith(".venv"))
        for name in sorted(files):
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            if any(rel == p or rel.startswith(p + "/") for p in PRIVATE):
                continue
            if name in EXCLUDED or name in {".pypirc", ".netrc", ".npmrc", ".DS_Store", "Thumbs.db", "temp.py", "tempCodeRunnerFile.py"}:
                continue
            if name.startswith(".env") or name.endswith((".pyc", ".pem", ".key", ".log")):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"refusing a linked path: {rel}")
            yield rel, path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="submission/chain-of-bid-review.zip")
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT) or out.parent != ROOT / "submission":
        parser.error("output must be a file inside submission/")
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        parser.error("output already exists; choose another filename")
    sums = []
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for rel, path in sorted(payload()):
            entry = zipfile.ZipInfo("review/" + rel, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100755 if rel.endswith(".sh") else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            digest = hashlib.sha256()
            with path.open("rb") as src, archive.open(entry, "w", force_zip64=True) as dst:
                while block := src.read(1024 * 1024):
                    digest.update(block)
                    dst.write(block)
            sums.append(f"{digest.hexdigest()}  {rel}\n")
        entry = zipfile.ZipInfo("review/SHA256SUMS.txt", date_time=(1980, 1, 1, 0, 0, 0))
        entry.create_system = 3
        entry.external_attr = 0o100644 << 16
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, "".join(sums).encode("utf-8"))
        archive.comment = b""
    print(f"packed {len(sums)} files into {out.relative_to(ROOT).as_posix()}")
    print("no git history, archive comment or local file timestamps included")

if __name__ == "__main__":
    main()
