"""Reference data for checking the browser port against AutoMorph_v2.

For every sample image this writes tools/ref/<stem>/ with
  input.rgb        decoded RGB (cv2, as M0 reads it)
  m0.rgb           M0 masked/cropped/padded square
  r_<task>.rgb     the PIL resize each model starts from
  x_<task>.f32     the exact model input tensor built by AutoMorph_v2's code
  y_<task>.f32     the exported ONNX member's raw output (CPU)
  l_<task>.u8      decoded labels (vessel binary / A-V labels / disc-cup labels)
  meta.json        sizes, M0 geometry, quality scores
Run: tools/.venv/Scripts/python.exe tools/make_reference.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from skimage.morphology import remove_small_objects

STUDIO = Path(__file__).resolve().parents[1]
AM = STUDIO / "automorph_v2"
SAMPLES = STUDIO / "webui" / "public" / "samples"
MODELS = STUDIO / "webui" / "public" / "models"
REF = STUDIO / "tools" / "ref"

sys.path.insert(0, str(AM / "src"))
from v2.M0 import core as m0  # noqa: E402
from v2.M1 import core as m1  # noqa: E402
from v2.M2 import artery_vein as m2av  # noqa: E402
from v2.M2 import disc_cup as m2dc  # noqa: E402
from v2.M2 import vessel as m2v  # noqa: E402


def save(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array).tobytes())


def decode_vessel(prob: np.ndarray) -> np.ndarray:
    binary = prob[0, 0] >= 0.5
    return remove_small_objects(binary, 30, connectivity=5).astype(np.uint8)


def decode_av(prob: np.ndarray) -> np.ndarray:
    labels = torch.max(torch.from_numpy(prob), 1)[1][0].numpy().astype(np.uint8)
    artery = remove_small_objects(labels == 1, 30, connectivity=5)
    vein = remove_small_objects(labels == 2, 30, connectivity=5)
    labels[(labels == 1) & ~artery] = 0
    labels[(labels == 2) & ~vein] = 0
    return labels


def decode_disc(prob: np.ndarray) -> np.ndarray:
    labels = torch.max(torch.from_numpy(prob), 1)[1][0].numpy()
    disc = remove_small_objects(labels == 1, 100)
    cup = remove_small_objects(labels == 2, 50)
    out = np.zeros(labels.shape, np.uint8)
    out[disc] = 1
    out[cup] = 2
    return out


def main() -> None:
    sessions = {
        name: ort.InferenceSession(str(MODELS / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        for name in ("quality", "vessel", "artery_vein", "disc_cup")
    }
    for sample in sorted(SAMPLES.glob("*.jpg")):
        out = REF / sample.stem
        out.mkdir(parents=True, exist_ok=True)
        image = m0.read_rgb(sample)
        mask, bbox, centre_w, centre_h, radius = m0.legacy_geometry(image)
        cropped = m0._mask_crop_pad_cpu(image, mask, bbox)
        save(out / "input.rgb", image)
        save(out / "m0.rgb", cropped)

        # the same intermediate numbers legacy_geometry derives, for debugging
        gray = m0.cv2.cvtColor(image, m0.cv2.COLOR_RGB2GRAY)
        normalized = m0.cv2.normalize(gray, None, 0, 255, m0.cv2.NORM_MINMAX)
        rough = m0._get_mask_bz(normalized)
        rows = rough.sum(axis=1)
        cols = rough.sum(axis=0)
        centre_row = float(np.where(rows > rows.max() * 0.95)[0].mean())
        centre_col = float(np.where(cols > cols.max() * 0.95)[0].mean())

        tmp = Path(tempfile.mkdtemp(prefix="studio_ref_"))
        png = tmp / f"{sample.stem}.png"
        Image.fromarray(cropped).save(png)
        with Image.open(png) as im:
            resized = {
                "quality": np.array(im.resize((512, 512))),
                "vessel": np.array(im.resize((912, 912))),
                "artery_vein": np.array(im.resize((720, 720))),
                "disc_cup": np.array(im.resize((512, 512), Image.BILINEAR)),
            }
        inputs = {
            "quality": m1.QualityDataset(tmp)[0][0].unsqueeze(0).float().numpy(),
            "vessel": m2v.preprocess_image(png).image.unsqueeze(0).float().numpy(),
            "artery_vein": m2av.AVImageDataset(tmp)[0].image.unsqueeze(0).float().numpy(),
            "disc_cup": m2dc.preprocess_legacy_image(m2dc.ImageEntry(stem=sample.stem, path=png)).tensor.unsqueeze(0).float().numpy(),
        }
        meta = {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "m0": {
                "centreRow": centre_row,
                "centreCol": centre_col,
                "radius": int(radius),
                "bbox": [int(v) for v in bbox],
                "side": int(cropped.shape[0]),
                "legacy_centre_w": int(centre_w),
                "legacy_centre_h": int(centre_h),
            },
        }
        for task, x in inputs.items():
            save(out / f"r_{task}.rgb", resized[task])
            save(out / f"x_{task}.f32", x.astype(np.float32))
            y = sessions[task].run(None, {"input": x.astype(np.float32)})[0]
            save(out / f"y_{task}.f32", y.astype(np.float32))
            if task == "quality":
                p = y[0]
                k = int(np.argmax(p))
                meta["quality"] = {
                    "probs": [float(v) for v in p],
                    "prediction": ["Good", "Usable", "Reject"][k],
                    "gradable": bool(k == 0 or (k == 1 and p[2] < 0.25)),
                }
            elif task == "vessel":
                save(out / "l_vessel.u8", decode_vessel(y))
            elif task == "artery_vein":
                save(out / "l_artery_vein.u8", decode_av(y))
            else:
                save(out / "l_disc_cup.u8", decode_disc(y))
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(sample.stem, meta["m0"], meta["quality"])


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
