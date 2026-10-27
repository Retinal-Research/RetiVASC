"""Stage photos, run AutoMorph v2 on them and read its outputs.

AutoMorph v2 is called through its own command line, unmodified. Paths and the
GPU choice come from the environment so the module runs on any machine:
  RETIVASC_AUTOMORPH  AutoMorph v2 folder     (default: <retivasc>/automorph_v2)
  RETIVASC_GPU_PY     Python with torch       (default: the Python running this module)
  RETIVASC_GPU_ENV    extra environment, e.g. "CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=0"
The same keys (automorph, gpu_python, gpu_env) can be set in retivasc.local.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config

AUTOMORPH = Path(config.get("automorph", "RETIVASC_AUTOMORPH", str(config.ROOT / "automorph_v2")))
GPU_PY = Path(config.get("gpu_python", "RETIVASC_GPU_PY", sys.executable))
GPU_ENV = dict(
    kv.split("=", 1)
    for kv in config.get("gpu_env", "RETIVASC_GPU_ENV").split(";")
    if "=" in kv
)
LOG = "retivasc_automorph.log"
MANIFEST = "retivasc_staging.json"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}

# the 72 values of AutoMorph v2 M3, in its column order
FEATURES = [
    "Disc_height", "Disc_width", "Cup_height", "Cup_width", "CDR_vertical", "CDR_horizontal",
    *[f"{v}{m}" for v in ("", "Artery_", "Vein_") for m in (
        "Fractal_dimension", "Vessel_density", "Average_width", "Distance_tortuosity",
        "Squared_curvature_tortuosity", "Tortuosity_density")],
]
for _zone in ("b", "c"):
    for _v, _calibre in (("", None), ("Artery_", "CRAE"), ("Vein_", "CRVE")):
        FEATURES += [f"{_v}{m}_zone_{_zone}" for m in (
            "Fractal_dimension", "Vessel_density", "Average_width", "Distance_tortuosity",
            "Squared_curvature_tortuosity", "Tortuosity_density")]
        if _calibre:
            FEATURES += [f"{_calibre}_Hubbard_zone_{_zone}", f"{_calibre}_Knudtson_zone_{_zone}"]
    FEATURES += [f"AVR_Hubbard_zone_{_zone}", f"AVR_Knudtson_zone_{_zone}"]
assert len(FEATURES) == 72


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def stage(images_root: Path | str, data_root: Path | str) -> dict:
    """Copy every non-empty image under images_root into <data_root>/images, each under a
    unique, AutoMorph-safe name (AutoMorph keys its outputs by stem), and record where each
    copy came from in <data_root>/retivasc_staging.json."""
    images_root, data_root = Path(images_root), Path(data_root)
    images = data_root / "images"
    images.mkdir(parents=True, exist_ok=False)
    entries, stems = [], set()
    for path in sorted(p for p in images_root.rglob("*") if p.is_file()):
        if path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS or path.stat().st_size == 0:
            continue
        rel = path.relative_to(images_root).as_posix()
        base = _safe(Path(rel).with_suffix("").as_posix().replace("/", "__")) or "photo"
        stem, n = base, 2
        while stem.lower() in stems:
            stem, n = f"{base}__{n}", n + 1
        stems.add(stem.lower())
        run_name = stem + path.suffix.lower()
        digest = hashlib.sha256()
        with open(path, "rb") as src, open(images / run_name, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                digest.update(chunk)
                dst.write(chunk)
        entries.append({"run_name": run_name, "stem": stem, "rel_path": rel, "size": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "image_root": str(images_root), "files": entries}
    (data_root / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_manifest(data_root: Path | str) -> dict:
    return json.loads((Path(data_root) / MANIFEST).read_text(encoding="utf-8"))


def run(data_root: Path | str, device: str = "cuda") -> None:
    data_root = Path(data_root)
    env = {**os.environ, **GPU_ENV, "PYTHONPATH": str(AUTOMORPH / "src")}
    cmd = [str(GPU_PY), "-m", "automorph_v2", "run", str(data_root), "--device", device, "--fresh", "--progress", "plain"]
    with open(data_root / LOG, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(AUTOMORPH), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"AutoMorph v2 failed (exit {proc.returncode}); see {data_root / LOG}")


@dataclass
class QualityRow:
    p_good: float
    p_usable: float
    p_reject: float
    prediction: int
    gradable: bool


@dataclass
class RunOutputs:
    data_root: Path
    quality: dict[str, QualityRow]  # stem -> M1 row
    features: dict[str, tuple[str, dict[str, float | None]]]  # stem -> (table, feature -> value)


def _stem(name: str) -> str:
    return Path(str(name).replace("\\", "/")).stem


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _number(text: str) -> float | None:
    # float() gives the correctly rounded double for every decimal in AutoMorph's CSV
    if text is None or text.strip() == "":
        return None
    v = float(text)
    return None if math.isnan(v) else v


def load(data_root: Path | str) -> RunOutputs:
    data_root = Path(data_root)
    results = data_root / "Results"
    quality = {
        _stem(r["Name"]): QualityRow(float(r["softmax_good"]), float(r["softmax_usable"]), float(r["softmax_bad"]),
                                     int(float(r["Prediction"])), r["Gradable"].strip().lower() in ("1", "true", "1.0"))
        for r in _rows(results / "M1" / "M1_quality_final.csv")
    }
    features: dict[str, tuple[str, dict[str, float | None]]] = {}
    for table in ("Macular_Features.csv", "Disc_Features.csv"):
        rows = _rows(results / "M3" / table)
        if rows and (missing := [f for f in FEATURES if f not in rows[0]]):
            raise ValueError(f"{table} lacks columns {missing[:3]}")
        for rec in rows:
            stem = _stem(rec["Name"])
            if stem in features:
                raise ValueError(f"{stem} appears in more than one AutoMorph feature row")
            features[stem] = (table, {f: _number(rec[f]) for f in FEATURES})
    return RunOutputs(data_root, quality, features)
