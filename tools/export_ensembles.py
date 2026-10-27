"""Export the complete AutoMorph v2 ensembles for the browser, one ONNX file per member,
with float16 weights (see fp16.py), and write models/manifest.json (version 2).

  M1 quality       8 EfficientNet-B4 members        quality_0..7.onnx
  M2 vessel       10 U-Net members                  vessel_0..9.onnx
  M2 artery/vein   8 BF-Net trios                   artery_vein_0..7.onnx
  M2 disc/cup      the 8 W-Nets in one file (mean)  disc_cup.onnx

Member order is AutoMorph v2's own order; the browser sums the members in that order and
divides by their number, as v2 does. Each member is checked against PyTorch on the sample.

  python tools/export_ensembles.py        (a Python with torch, onnx and onnxruntime)
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_models as E  # noqa: E402  (loaders, wrappers, reference inputs, export, compare)
from fp16 import to_fp16_weights  # noqa: E402

import onnx  # noqa: E402

LABELS = {
    "quality": "M1 image quality (EfficientNet-B4)",
    "vessel": "M2 binary vessel (U-Net)",
    "artery_vein": "M2 artery/vein (BF-Net trio)",
    "disc_cup": "M2 optic disc/cup (W-Net x8)",
}
SHAPES = {"quality": (1, 3, 512, 512), "vessel": (1, 3, 912, 912), "artery_vein": (1, 3, 720, 720),
          "disc_cup": (1, 3, 512, 512)}


def quality_members():
    root = E.AM / "runtime" / "M1_Retinal_Image_quality_EyePACS" / "Retinal_quality" / "EyePACS_quality" / "efficientnet"
    for directory in E.m1.CHECKPOINT_DIRS:
        model = E.m1.build_model()
        model.load_state_dict(E.m1._load_state(root / directory / "best_loss_checkpoint.pth"))
        model.set_swish(memory_efficient=False)
        yield directory, E.QualityNet(model.eval()).eval()


def vessel_members():
    legacy = E.AM / "runtime" / "M2_Vessel_seg"
    segmenter = E.m2v._import_legacy_segmenter(legacy)
    for seed in E.m2v.STRICT_SEEDS:
        path = (legacy / "Saved_model" / f"train_on_{E.m2v.STRICT_DATASET}"
                / f"{E.m2v.STRICT_JOB_NAME}_savebest_randomseed_{seed}" / E.m2v.CHECKPOINT_FILENAME)
        model = segmenter(input_channels=3, n_filters=32, n_classes=1, bilinear=False)
        model.load_state_dict(E.m2v._load_checkpoint(path))
        yield f"seed {seed}", E.VesselNet(model.eval()).eval()


def artery_vein_members():
    for trio in E.m2av.load_model_trios(E.AM / "runtime" / "M2_Artery_vein" / "ALL-AV", E.CPU):
        yield f"trio seed {trio.seed}", E.ArteryVeinNet(trio).eval()


FULL = "--float32" in sys.argv  # keep float32 weights (twice the download, no rounding of weights)


def write_member(name: str, module, x: torch.Tensor, tmp: Path) -> tuple[Path, dict]:
    """float32 export -> checked against PyTorch -> float16 weights (unless --float32) -> checked again."""
    E.OUT, out = tmp, E.OUT  # export() writes into E.OUT
    try:
        fp32 = E.export(name, module, x)
    finally:
        E.OUT = out
    check32 = E.compare(module, fp32, x)
    final = out / f"{name}.onnx"
    if FULL:
        final.write_bytes(fp32.read_bytes())
        return final, {"float32": check32}
    onnx.save(to_fp16_weights(onnx.load(str(fp32))), str(final))
    check16 = E.compare(module, final, x)
    return final, {"float32": check32, "float16_weights": check16}


def main() -> None:
    torch.set_grad_enabled(False)
    if "--out" in sys.argv:
        E.OUT = Path(sys.argv[sys.argv.index("--out") + 1])
        E.OUT.mkdir(parents=True, exist_ok=True)
    out = E.OUT
    inputs = E.reference_inputs()
    manifest = {"version": 2, "opset": E.OPSET, "precision": "float16 weights, float32 arithmetic",
                "source": "AutoMorph v2 runtime checkpoints (see automorph_v2/models/manifest.json)", "models": {}}
    tmp = Path(tempfile.mkdtemp(prefix="retivasc_export_"))
    for key, members in (("quality", quality_members), ("vessel", vessel_members), ("artery_vein", artery_vein_members)):
        parts = []
        for i, (member, module) in enumerate(members()):
            started = time.perf_counter()
            path, check = write_member(f"{key}_{i}", module, inputs[key], tmp)
            parts.append({"file": f"models/{path.name}", "bytes": path.stat().st_size, "sha256": E.sha256(path),
                          "member": member, "check": check})
            print(f"{key}_{i:<2d} {member:16s} {path.stat().st_size / 2**20:6.1f} MB {time.perf_counter() - started:5.1f}s "
                  f"{check.get('float16_weights', check['float32'])}", flush=True)
        manifest["models"][key] = {"label": LABELS[key], "input": list(SHAPES[key]), "parts": parts,
                                   "members": f"all {len(parts)} members (mean, AutoMorph v2 order)",
                                   "bytes": sum(p["bytes"] for p in parts)}
    # disc/cup already holds all eight W-Nets
    disc_module, _, disc_members = E.load_disc_cup()
    # disc/cup stays float32: it is small, and a one-pixel change in the disc moves M3's zone rings
    path = E.export("disc_cup", disc_module, inputs["disc_cup"])
    check = {"float32": E.compare(disc_module, path, inputs["disc_cup"])}
    manifest["models"]["disc_cup"] = {"label": LABELS["disc_cup"], "input": list(SHAPES["disc_cup"]), "members": disc_members,
                                      "bytes": path.stat().st_size,
                                      "parts": [{"file": "models/disc_cup.onnx", "bytes": path.stat().st_size, "sha256": E.sha256(path),
                                                 "member": "8 W-Nets, mean softmax", "precision": "float32", "check": check}]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(m["bytes"] for m in manifest["models"].values())
    print(f"wrote {out / 'manifest.json'}: {total / 2**20:.0f} MB in {sum(len(m['parts']) for m in manifest['models'].values())} files")


if __name__ == "__main__":
    main()
