"""Export one AutoMorph_v2 member per task to ONNX for the browser demo.

Run with the oct env (torch + onnxruntime, nothing to install):

    python tools/export_models.py   (a Python with torch and onnx)

Weights are read from AutoMorph_v2/runtime and never modified. The script writes
web/public/models/<name>.onnx plus manifest.json, then checks every exported
graph against PyTorch on a real sample image preprocessed by AutoMorph_v2's own
functions.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

STUDIO = Path(__file__).resolve().parents[1]
AM = STUDIO / "automorph_v2"
OUT = STUDIO / "webui" / "public" / "models"
SAMPLE = STUDIO / "webui" / "public" / "samples" / "cc0_left_3744.jpg"
OPSET = 17

sys.path.insert(0, str(AM / "src"))
from v2.M0 import core as m0  # noqa: E402
from v2.M1 import core as m1  # noqa: E402
from v2.M2 import artery_vein as m2av  # noqa: E402
from v2.M2 import disc_cup as m2dc  # noqa: E402
from v2.M2 import vessel as m2v  # noqa: E402

CPU = torch.device("cpu")


# ---------------------------------------------------------------- wrappers
class QualityNet(nn.Module):
    """EfficientNet-B4 member -> softmax over (good, usable, reject)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


class VesselNet(nn.Module):
    """Binary-vessel U-Net member -> vessel probability."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class ArteryVeinNet(nn.Module):
    """One main/artery/vein trio -> softmax over (background, artery, vein, overlap)."""

    def __init__(self, trio):
        super().__init__()
        self.main, self.artery, self.vein = trio.main, trio.artery, trio.vein
        self._trio = trio

    def forward(self, x):
        return m2av._member_probability(self._trio, x, skip_unused_heads=True)


def widen_input(conv: nn.Conv2d, channels: int) -> nn.Conv2d:
    """Same convolution with extra all-zero input channels appended."""
    wide = nn.Conv2d(channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    weight = torch.zeros(conv.out_channels, channels, *conv.kernel_size)
    weight[:, : conv.in_channels] = conv.weight.detach()
    wide.weight.data = weight
    if conv.bias is not None:
        wide.bias.data = conv.bias.detach().clone()
    return wide


class PaddedWNet(nn.Module):
    """A W-Net member whose second U-Net reads 8 channels instead of 6.

    The W-Net feeds cat([image, unet1(image)]) (3 + 3 channels) to its second
    U-Net. onnxruntime-web's WebGPU convolution returns wrong values for that
    6-channel input (CPU is fine; 3- and 8-channel inputs are fine), so two zero
    channels are appended and the two convolutions that read them get matching
    zero weights. The arithmetic is unchanged."""

    def __init__(self, wnet: nn.Module):
        super().__init__()
        self.unet1, self.unet2 = wnet.unet1, wnet.unet2
        first = self.unet2.first
        first.block[0] = widen_input(first.block[0], 8)
        first.shortcut[0] = widen_input(first.shortcut[0], 8)

    def forward(self, x):
        x1 = self.unet1(x)
        return self.unet2(torch.cat([x, x1, x[:, :2] * 0], dim=1))


class DiscCupNet(nn.Module):
    """All eight W-Net members (about 1 MB each) -> mean softmax over (bg, disc, cup)."""

    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self, x):
        total = None
        for model in self.members:
            out = model(x)
            logits = out[-1] if isinstance(out, (tuple, list)) else out
            p = F.softmax(logits, dim=1)
            total = p if total is None else total + p
        return total / len(self.members)


# ---------------------------------------------------------------- loading
def load_quality():
    directory = m1.CHECKPOINT_DIRS[0]
    path = AM / "runtime" / "M1_Retinal_Image_quality_EyePACS" / "Retinal_quality" / "EyePACS_quality" / "efficientnet" / directory / "best_loss_checkpoint.pth"
    model = m1.build_model()
    model.load_state_dict(m1._load_state(path))
    model.set_swish(memory_efficient=False)  # the memory-efficient swish is a custom autograd op
    model.eval()
    return QualityNet(model).eval(), path, f"{len(m1.CHECKPOINT_DIRS)}-member ensemble, member {directory}"


def load_vessel():
    seed = m2v.STRICT_SEEDS[0]
    legacy = AM / "runtime" / "M2_Vessel_seg"
    path = legacy / "Saved_model" / f"train_on_{m2v.STRICT_DATASET}" / f"{m2v.STRICT_JOB_NAME}_savebest_randomseed_{seed}" / m2v.CHECKPOINT_FILENAME
    segmenter = m2v._import_legacy_segmenter(legacy)
    model = segmenter(input_channels=3, n_filters=32, n_classes=1, bilinear=False)
    model.load_state_dict(m2v._load_checkpoint(path))
    model.eval()
    return VesselNet(model).eval(), path, f"{len(m2v.STRICT_SEEDS)}-member ensemble, member seed {seed}"


def load_artery_vein():
    weights = AM / "runtime" / "M2_Artery_vein" / "ALL-AV"
    trios = m2av.load_model_trios(weights, CPU)
    trio = trios[0]
    path = m2av.checkpoint_paths(weights)[0].parent
    return ArteryVeinNet(trio).eval(), path, f"{len(trios)}-trio ensemble, trio seed {trio.seed}"


def load_disc_cup():
    config = m2dc.DiscCupConfig(data_root=STUDIO)
    models = m2dc.load_ensemble_models(config, CPU)
    path = config.legacy_module_root / "experiments" / "wnet_All_three_1024_disc_cup"
    x = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        before = DiscCupNet(models).eval()(x)
        padded = DiscCupNet([PaddedWNet(m) for m in models]).eval()
        after = padded(x)
    diff = float((before - after).abs().max())
    assert diff < 1e-5, f"channel padding changed the output by {diff}"
    print(f"disc_cup: second U-Net input padded 6 -> 8 channels, max |diff| {diff:.2e}")
    return padded, path, f"all {len(models)} members (mean softmax)"


# ---------------------------------------------------------------- reference inputs
def reference_inputs() -> dict[str, torch.Tensor]:
    """Preprocess SAMPLE exactly as AutoMorph_v2 does, using its own functions."""

    image = m0.read_rgb(SAMPLE)
    mask, bbox, *_ = m0.legacy_geometry(image)
    cropped = m0._mask_crop_pad_cpu(image, mask, bbox)
    tmp = Path(tempfile.mkdtemp(prefix="studio_export_"))
    png = tmp / "sample.png"
    Image.fromarray(cropped).save(png)

    quality, _name = m1.QualityDataset(tmp)[0]
    vessel = m2v.preprocess_image(png).image
    av = m2av.AVImageDataset(tmp)[0].image
    disc = m2dc.preprocess_legacy_image(m2dc.ImageEntry(stem="sample", path=png)).tensor
    return {
        "quality": quality.unsqueeze(0).float(),
        "vessel": vessel.unsqueeze(0).float(),
        "artery_vein": av.unsqueeze(0).float(),
        "disc_cup": disc.unsqueeze(0).float(),
    }


# ---------------------------------------------------------------- export
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(name: str, module: nn.Module, example: torch.Tensor) -> Path:
    path = OUT / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            module,
            (example,),
            str(path),
            input_names=["input"],
            output_names=["output"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    return path


def compare(module: nn.Module, path: Path, x: torch.Tensor) -> dict:
    with torch.no_grad():
        expected = module(x).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": x.numpy()})[0]
    diff = np.abs(expected - got)
    report = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
    if expected.shape[1] > 1:  # class maps / class scores: argmax agreement
        report["argmax_agreement"] = float(np.mean(expected.argmax(1) == got.argmax(1)))
    else:  # vessel probability: agreement of the 0.5 threshold
        report["threshold_agreement"] = float(np.mean((expected >= 0.5) == (got >= 0.5)))
    return report


TASKS = {
    "quality": (load_quality, (1, 3, 512, 512), "M1 image quality (EfficientNet-B4)"),
    "vessel": (load_vessel, (1, 3, 912, 912), "M2 binary vessel (U-Net)"),
    "artery_vein": (load_artery_vein, (1, 3, 720, 720), "M2 artery/vein (BF-Net trio)"),
    "disc_cup": (load_disc_cup, (1, 3, 512, 512), "M2 optic disc/cup (W-Net x8)"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    inputs = reference_inputs()
    manifest = {"version": 1, "opset": OPSET, "source": str(AM), "models": {}}
    for name, (loader, shape, label) in TASKS.items():
        started = time.perf_counter()
        module, weights, members = loader()
        x = inputs[name]
        assert tuple(x.shape) == shape, (name, tuple(x.shape), shape)
        path = export(name, module, x)
        check = compare(module, path, x)
        manifest["models"][name] = {
            "label": label,
            "file": f"models/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "input": list(shape),
            "weights": str(weights),
            "members": members,
            "check": check,
        }
        print(f"{name:12s} {path.stat().st_size / 2**20:7.1f} MB  {time.perf_counter() - started:5.1f}s  {check}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
