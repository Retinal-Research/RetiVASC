"""Half-precision weights for the browser models: half the download, same arithmetic.

Every large float32 weight is stored as float16 and followed by a Cast back to float32, so the
model still computes in float32 on every GPU (no shader-f16 needed); onnxruntime folds the
casts when it loads the model. The only change is the rounding of each weight to float16.

  python tools/fp16.py convert <in.onnx> <out.onnx>
  python tools/fp16.py check   <models dir> <reference dir>    # fp32 vs fp16, on the sample inputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 64  # small tensors (shapes, scales) stay float32


def to_fp16_weights(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    casts, keep = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < MIN_ELEMENTS:
            keep.append(init)
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return model


def convert(src: Path, dst: Path) -> None:
    m = onnx.load(str(src))
    onnx.save(to_fp16_weights(m), str(dst))
    print(f"{src.name}: {src.stat().st_size / 2**20:.1f} MB -> {dst.stat().st_size / 2**20:.1f} MB")


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def check(models: Path, ref: Path, out_dir: Path) -> dict:
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for key, channels in (("quality", None), ("vessel", 1), ("artery_vein", 4), ("disc_cup", 3)):
        src = models / f"{key}.onnx"
        half = out_dir / f"{key}.onnx"
        if not half.exists():
            convert(src, half)
        s32 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        shape = [d if isinstance(d, int) else 1 for d in s32.get_inputs()[0].shape]
        rows = []
        for sample in sorted(p for p in ref.iterdir() if p.is_dir()):
            xf = sample / f"x_{key}.f32"
            if not xf.exists():
                continue
            x = np.fromfile(xf, dtype=np.float32).reshape(shape)
            a, b = _run(src, x), _run(half, x)
            if key == "quality":
                rows.append({"sample": sample.name, "max_prob_diff": float(np.abs(a - b).max()),
                             "same_class": bool(a.argmax() == b.argmax())})
            elif channels == 1:
                flips = int(((a >= 0.5) != (b >= 0.5)).sum())
                rows.append({"sample": sample.name, "pixels": int(a.size), "flipped": flips, "max_prob_diff": float(np.abs(a - b).max())})
            else:
                la, lb = a.reshape(channels, -1).argmax(0), b.reshape(channels, -1).argmax(0)
                rows.append({"sample": sample.name, "pixels": int(la.size), "flipped": int((la != lb).sum()),
                             "max_prob_diff": float(np.abs(a - b).max())})
        report[key] = {"size_mb": [round(src.stat().st_size / 2**20, 1), round(half.stat().st_size / 2**20, 1)], "samples": rows}
    tp = models / "tpot.onnx"
    if tp.exists():
        half = out_dir / "tpot.onnx"
        if not half.exists():
            convert(tp, half)
        rows = []
        s32 = ort.InferenceSession(str(tp), providers=["CPUExecutionProvider"])
        shape = [d if isinstance(d, int) else 1 for d in s32.get_inputs()[0].shape]
        rng = np.random.default_rng(0)
        for i in range(3):  # TPOT has no saved inputs here: random photo-like canvases
            x = rng.random(shape, dtype=np.float32)
            a, b = _run(tp, x), _run(half, x)
            rows.append({"sample": f"random_{i}", "max_level_diff_0_255": float(np.abs(a - b).max() * 255)})
        report["tpot"] = {"size_mb": [round(tp.stat().st_size / 2**20, 1), round(half.stat().st_size / 2**20, 1)], "samples": rows}
    return report


if __name__ == "__main__":
    if sys.argv[1] == "convert":
        convert(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        r = check(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        print(json.dumps(r, indent=1))
