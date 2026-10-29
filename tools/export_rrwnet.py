"""Export RRWNet (Morano, Aresta and Bogunovic, 2024) for the browser: the artery/vein/vessel segmentation
the RetiVASC pipeline uses, as in RetiFlow's examples (HRF weights, five refinement iterations).

  python tools/export_rrwnet.py <model.safetensors>

The weights are j-morano/rrwnet-hrf on Hugging Face (CC BY-NC 4.0; the code is MIT), at the revision in
REVISION. Nothing of the authors' code is imported: the network and its preprocessing are written out
here from model.py, preprocessing.py and utils.py of github.com/j-morano/rrwnet, and the weights are
safetensors (data, no code).

The network's last refined prediction, through a sigmoid, is the output: three probabilities per pixel,
arteries, veins and vessels (RRWNet's AV3 convention). The ONNX file is checked against PyTorch on
a sample preprocessed as RRWNet does it (tools/rrwnet_ref/), then stored with float16 weights (fp16.py)
and checked again. Writes webui/public/models/rrwnet/rrwnet.onnx and its manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from scipy import ndimage
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp16 import to_fp16_weights  # noqa: E402

OUT = ROOT / "webui" / "public" / "models" / "rrwnet"
REF = ROOT / "tools" / "rrwnet_ref"
OPSET = 17
SIDE = 584  # RetiFlow's examples: the photograph at 584 x 584, padded by RRWNet to 608
SOURCE = "https://huggingface.co/j-morano/rrwnet-hrf"
REVISION = "f64c297d81f05bcce2c40796c1a9bee4c72a1908"


# ---------------------------------------------------------------- the network (model.py), same parameter names

class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, 3, 1, 1)
        self.conv2 = nn.Conv2d(o, o, 3, 1, 1)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class UpConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.ConvTranspose2d(i, o, 2, 2)

    def forward(self, x):
        return self.conv(x)


class UNetModule(nn.Module):
    def __init__(self, i, o, b):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = ConvBlock(i, b), ConvBlock(b, 2 * b), ConvBlock(2 * b, 4 * b)
        self.conv4, self.conv5 = ConvBlock(4 * b, 8 * b), ConvBlock(8 * b, 16 * b)
        self.upconv1, self.conv6 = UpConv(16 * b, 8 * b), ConvBlock(16 * b, 8 * b)
        self.upconv2, self.conv7 = UpConv(8 * b, 4 * b), ConvBlock(8 * b, 4 * b)
        self.upconv3, self.conv8 = UpConv(4 * b, 2 * b), ConvBlock(4 * b, 2 * b)
        self.upconv4, self.conv9 = UpConv(2 * b, b), ConvBlock(2 * b, b)
        self.outconv = nn.Conv2d(b, o, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, 2, 2))
        x3 = self.conv3(F.max_pool2d(x2, 2, 2))
        x4 = self.conv4(F.max_pool2d(x3, 2, 2))
        x = self.upconv1(self.conv5(F.max_pool2d(x4, 2, 2)))
        x = self.upconv2(self.conv6(torch.cat((x4, x), 1)))
        x = self.upconv3(self.conv7(torch.cat((x3, x), 1)))
        x = self.upconv4(self.conv8(torch.cat((x2, x), 1)))
        return self.outconv(self.conv9(torch.cat((x1, x), 1)))


class RRWNet(nn.Module):
    def __init__(self, iterations=5):
        super().__init__()
        self.first_u = UNetModule(3, 3, 64)
        self.second_u = UNetModule(3, 2, 64)
        self.iterations = iterations

    def forward(self, x):
        """The last prediction of RRWNet.forward: refined A and V logits, and the first module's BV logits."""
        pred_1 = self.first_u(x)
        bv_logits = pred_1[:, 2:3]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3]
        pred_2 = self.second_u(pred_1)
        for _ in range(self.iterations):
            pred_2 = self.second_u(torch.cat((torch.sigmoid(pred_2), bv), 1))
        return torch.cat((pred_2, bv_logits), 1)


# ---------------------------------------------------------------- preprocessing (preprocessing.py, utils.py)

def enhance_image(img, mask, disk_size=5):
    """RRWNet's enhance_image: the field of view's edge filled from a 1.15x zoom, background removed
    with a Gaussian of sigma 10, divided by the standard deviation, scaled to [0, 1]."""
    img = img / 255
    mask = np.where(mask > 0.5, 1, 0)
    zoomed = Image.fromarray(np.uint8(img * 255)).resize((int(img.shape[1] * 1.15), int(img.shape[0] * 1.15)), Image.BICUBIC)
    zoomed = np.array(zoomed)
    h, w = img.shape[:2]
    y0, x0 = zoomed.shape[0] // 2 - h // 2, zoomed.shape[1] // 2 - w // 2
    zoomed = zoomed[y0:y0 + h, x0:x0 + w] / 255.0
    r = disk_size
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (yy ** 2 + xx ** 2 <= r * r).astype(np.uint8)  # skimage.morphology.disk(5)
    mask = ndimage.binary_erosion(mask, kernel) * 1.0
    img = img.copy()
    img[mask < 1.0] = 0.0
    mask = np.stack((mask, mask, mask), axis=2)
    composed = mask.copy()
    composed[mask == 1.0] = img[mask == 1.0]
    composed[mask < 1.0] = zoomed[mask < 1.0]
    sub = composed - ndimage.gaussian_filter(composed, sigma=(10, 10, 0))
    sub[mask < 1.0] = 0.0
    out = sub / np.std(sub)
    out = (out - out.min()) / (out.max() - out.min())  # to_0_1: interp1d([min, max], [0, 1])
    out[mask < 1.0] = 0.0
    return out, mask[:, :, 0]


def unet_padding(shape, n_down=5):
    n = 2 ** n_down
    h_pad, w_pad = n - shape[0] % n, n - shape[1] % n  # a full n when already divisible, as the authors' code does
    return (h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2)


class Probabilities(torch.nn.Module):
    """RRWNet's forward pass, ending in the probabilities its get_predictions.py saves."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(weights: Path) -> None:
    torch.set_grad_enabled(False)
    net = RRWNet(iterations=5)
    net.load_state_dict(load_file(str(weights)), strict=True)
    module = Probabilities(net.eval()).eval()

    # a reference input: a sample photograph's centre square at 584, its circular field of view,
    # preprocessed and padded as the authors' get_predictions.py does
    photo = Image.open(ROOT / "webui" / "public" / "samples" / "cc0_left_3744.jpg").convert("RGB")
    s = min(photo.size)
    left, top = (photo.width - s) // 2, (photo.height - s) // 2
    rgb = np.asarray(photo.crop((left, top, left + s, top + s)).resize((SIDE, SIDE), Image.BICUBIC))
    yy, xx = np.mgrid[:SIDE, :SIDE]
    mask = (((yy - (SIDE - 1) / 2) ** 2 + (xx - (SIDE - 1) / 2) ** 2) <= (SIDE / 2 - 1) ** 2).astype(np.uint8) * 255
    enhanced, eroded = enhance_image(rgb, mask)
    padding = unet_padding(enhanced.shape)
    padded = np.pad(enhanced, (*padding, (0, 0)))
    x = torch.from_numpy(padded.transpose(2, 0, 1).astype("float32"))[None]
    expected = module(x).numpy()

    REF.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(REF / "input.png")
    rgb.astype(np.uint8).tofile(REF / "input.rgb")             # (584, 584, 3), the same, raw
    Image.fromarray(mask).save(REF / "mask.png")
    enhanced.astype("<f8").tofile(REF / "enhanced.f64")       # (584, 584, 3), RRWNet's preprocessing
    eroded.astype("<f8").tofile(REF / "eroded.f64")           # (584, 584)
    x.numpy().astype("<f4").tofile(REF / "x.f32")              # (1, 3, 608, 608), padded
    expected.astype("<f4").tofile(REF / "probs.f32")           # (1, 3, 608, 608)
    json.dump({"padding": [list(p) for p in padding], "shape": list(x.shape)}, open(REF / "meta.json", "w"))

    OUT.mkdir(parents=True, exist_ok=True)
    fp32 = REF / "rrwnet_fp32.onnx"
    torch.onnx.export(module, x, str(fp32), opset_version=OPSET, input_names=["image"], output_names=["probabilities"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "probabilities": {2: "height", 3: "width"}}, dynamo=False)

    def check(path: Path) -> float:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = session.run(None, {"image": x.numpy()})[0]
        return float(np.abs(got - expected).max())

    print(f"float32 ONNX vs PyTorch: max |difference| {check(fp32):.2e}")
    final = OUT / "rrwnet.onnx"
    onnx.save(to_fp16_weights(onnx.load(str(fp32))), str(final))
    print(f"float16 weights vs PyTorch: max |difference| {check(final):.2e}  ({final.stat().st_size / 2**20:.0f} MB)")

    manifest = {
        "version": 1,
        "source": SOURCE,
        "revision": REVISION,
        "license": "CC BY-NC 4.0 (the weights on Hugging Face; the code is MIT). Non-commercial research use; cite RRWNet (Morano et al., Expert Systems with Applications 2024).",
        "precision": "float16 weights, float32 arithmetic",
        "models": {"rrwnet": {
            "label": "RRWNet artery/vein/vessel segmentation (HRF weights, 5 refinements)",
            "bytes": final.stat().st_size,
            "input": [1, 3, 608, 608],
            "members": "one network: a U-Net, then a second U-Net refining its own output 6 times",
            "parts": [{"file": "rrwnet.onnx", "bytes": final.stat().st_size, "sha256": sha256(final), "member": "rrwnet"}],
        }},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {final} and {OUT / 'manifest.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
