"""Model files for RetiVASC: they are too large for git and ship as two GitHub release assets.

  python tools/release.py fetch            download and install both packages
  python tools/release.py fetch --web      browser models only (enough for the web UI on its own)
  python tools/release.py fetch --vascx    the VascX models, from Eyened/vascx on Hugging Face (0.56 GB)
  python tools/release.py check            verify the installed files against the checksum lists
  python tools/release.py pack             (maintainers) build the two zip files to attach to a release

  retivasc-web-models-<version>.zip         webui/public/models/*.onnx              ~0.85 GB
  retivasc-automorph-weights-<version>.zip  automorph_v2/runtime/**/*.pth           ~1.7 GB, for the server

Only the Python standard library is used, so this runs before anything else is installed.
Every file is checked against webui/public/models/manifest.json or automorph_v2/models/manifest.json.
The repository is read from `git remote get-url origin`; pass --repo OWNER/NAME to override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
WEB_MANIFEST = ROOT / "webui" / "public" / "models" / "manifest.json"
AM_MANIFEST = ROOT / "automorph_v2" / "models" / "manifest.json"
PACKAGES = {
    "web": f"retivasc-web-models-{VERSION}.zip",
    "automorph": f"retivasc-automorph-weights-{VERSION}.zip",
}


def expected(package: str) -> dict[str, tuple[int, str]]:
    """Path relative to the repository -> (bytes, sha256), from the committed manifests."""
    if package == "web":
        m = json.loads(WEB_MANIFEST.read_text(encoding="utf-8"))
        return {f"webui/public/{p['file']}": (p["bytes"], p["sha256"])
                for model in m["models"].values() for p in model["parts"]}
    m = json.loads(AM_MANIFEST.read_text(encoding="utf-8"))
    return {f"automorph_v2/{f['path']}": (f["bytes"], f["sha256"]) for f in m["files"] if f["kind"] == "checkpoint"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def missing(package: str, quick: bool = False) -> list[str]:
    """Files absent or not matching the manifest; quick compares sizes only."""
    bad = []
    for rel, (size, digest) in expected(package).items():
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size != size or (not quick and sha256(p) != digest):
            bad.append(rel)
    return bad


def repo_name(arg: str | None) -> str:
    if arg:
        return arg
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit("cannot tell which GitHub repository to download from; pass --repo OWNER/NAME")
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", url)
    if not m:
        sys.exit(f"origin {url} is not a GitHub repository; pass --repo OWNER/NAME")
    return m.group(1)


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()


def fetch(packages: list[str], repo: str | None) -> int:
    todo = [p for p in packages if missing(p, quick=True)]
    if not todo:
        print("all model files are already installed (python tools/release.py check verifies their checksums)")
        return 0
    longest = max(len(str(ROOT / rel)) for p in todo for rel in expected(p))
    if sys.platform == "win32" and longest >= 260:
        # AutoMorph's checkpoints sit in deep folders; Windows refuses paths of 260+ characters
        sys.exit(f"the repository folder is too deep for Windows ({longest} characters for the longest model file, "
                 f"limit 259). Clone it to a short path such as C:\\RetiVASC and run this again.")
    repo = repo_name(repo)
    for package in todo:
        name = PACKAGES[package]
        url = f"https://github.com/{repo}/releases/download/v{VERSION}/{name}"
        print(f"downloading {name}\n  from {url}")
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / name
            try:
                download(url, zpath)
            except urllib.error.HTTPError as e:
                sys.exit(f"download failed ({e.code} {e.reason}): is there a release v{VERSION} with {name} in {repo}?")
            want = expected(package)
            with zipfile.ZipFile(zpath) as z:
                for info in z.infolist():
                    rel = info.filename
                    if rel not in want:  # only files named in the manifest, never paths outside the repository
                        continue
                    dest = ROOT / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
        bad = missing(package)
        if bad:
            print(f"FAILED: {len(bad)} files missing or not matching their checksum, e.g. {bad[0]}")
            return 1
        print(f"  installed and verified {len(want)} files")
    return 0


VASCX_MANIFEST = ROOT / "webui" / "public" / "models" / "vascx" / "manifest.json"


def fetch_vascx() -> int:
    """The VascX models are not ours to repackage (CC BY-NC 4.0): fetch them from the authors'
    Hugging Face repository, at the revision in webui/public/models/vascx/manifest.json, and check them."""
    m = json.loads(VASCX_MANIFEST.read_text(encoding="utf-8"))
    folder = VASCX_MANIFEST.parent
    print(f"VascX models: {m['license']}")
    for model in m["models"].values():
        for part in model["parts"]:
            dest = folder / part["file"]
            if dest.is_file() and dest.stat().st_size == part["bytes"] and sha256(dest) == part["sha256"]:
                continue
            url = f"{m['source']}/resolve/{m['revision']}/{part['file']}"
            print(f"downloading {part['file']}\n  from {url}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            try:
                download(url, tmp)
            except urllib.error.HTTPError as e:
                sys.exit(f"download failed ({e.code} {e.reason})")
            if sha256(tmp) != part["sha256"]:
                tmp.unlink()
                sys.exit(f"{part['file']} does not match its checksum")
            tmp.replace(dest)
    print("VascX models installed and verified")
    return 0


def pack(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for package, name in PACKAGES.items():
        bad = missing(package)
        if bad:
            print(f"cannot pack {name}: {len(bad)} files missing or changed, e.g. {bad[0]}")
            return 1
        zpath = out_dir / name
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:  # weights do not compress
            for rel in sorted(expected(package)):
                z.write(ROOT / rel, rel)
        print(f"{zpath}  {zpath.stat().st_size / 2**20:.0f} MB  sha256 {sha256(zpath)}")
    print(f"attach both files to the GitHub release v{VERSION}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python tools/release.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("fetch")
    q.add_argument("--web", action="store_true", help="browser models only")
    q.add_argument("--vascx", action="store_true", help="the VascX models (from Hugging Face), only these")
    q.add_argument("--repo", help="OWNER/NAME of the GitHub repository")
    sub.add_parser("check")
    q = sub.add_parser("pack")
    q.add_argument("--out", default=str(ROOT / "release"))
    a = p.parse_args()
    if a.cmd == "fetch":
        if a.vascx:
            return fetch_vascx()
        return fetch(["web"] if a.web else ["web", "automorph"], a.repo)
    if a.cmd == "check":
        status = 0
        for package in PACKAGES:
            bad = missing(package)
            print(f"{package:10s} {'ok' if not bad else f'{len(bad)} files missing or changed'}")
            status |= bool(bad)
        return status
    return pack(Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
