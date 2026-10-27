"""How close is the browser demo (one ensemble member) to official AutoMorph_v2?

Expects the official pipeline output under tools/official/Results (run_m0,
run_m1, run_m2 on the sample images) and tools/ref from make_reference.py, which
the browser matches pixel for pixel.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parent
OFFICIAL = TOOLS / "official" / "Results"
REF = TOOLS / "ref"
CLASSES = ["Good", "Usable", "Reject"]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def main() -> None:
    m1 = {}
    with (OFFICIAL / "M1" / "M1_quality_final.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m1[Path(row["Name"]).stem] = row
    print(f"{'sample':20s} {'M1 single':>17s} {'M1 official':>18s}  vessel dice  artery  vein   disc   cup")
    for stem in sorted(p.name for p in REF.iterdir() if p.is_dir()):
        meta = json.loads((REF / stem / "meta.json").read_text(encoding="utf-8"))
        q = meta["quality"]
        o = m1[stem]
        official = f"{CLASSES[int(o['Prediction'])]} {'keep' if o['Gradable'] == 'True' else 'drop'}"
        single = f"{q['prediction']} {'keep' if q['gradable'] else 'drop'}"
        line = f"{stem:20s} {single:>17s} {official:>18s}"
        vessel_png = OFFICIAL / "M2" / "binary_vessel" / "binary_process" / f"{stem}.png"
        if not vessel_png.exists():
            print(line + "  (official M2 skips photos M1 drops)")
            continue
        v_off = np.array(Image.open(vessel_png).convert("L")) > 0
        v_us = np.fromfile(REF / stem / "l_vessel.u8", dtype=np.uint8).reshape(912, 912) > 0
        av = np.array(Image.open(OFFICIAL / "M2" / "artery_vein" / "resized" / f"{stem}.png").convert("RGB"))
        av_us = np.fromfile(REF / stem / "l_artery_vein.u8", dtype=np.uint8).reshape(720, 720)
        dc = np.array(Image.open(OFFICIAL / "M2" / "optic_disc_cup" / "resized" / f"{stem}.png").convert("RGB"))
        dc_us = np.fromfile(REF / stem / "l_disc_cup.u8", dtype=np.uint8).reshape(512, 512)
        line += (
            f"  {dice(v_off, v_us):10.3f}  {dice(av[..., 0] > 0, av_us == 1):6.3f}  {dice(av[..., 2] > 0, av_us == 2):5.3f}"
            f"  {dice((dc[..., 0] > 0) | (dc[..., 2] > 0), dc_us > 0):5.3f}  {dice(dc[..., 2] > 0, dc_us == 2):5.3f}"
        )
        print(line)


if __name__ == "__main__":
    main()
