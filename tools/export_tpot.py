"""Export the lab's TPOT fundus enhancement (Retinal-Research/TPOT-SR) to ONNX for the browser.

    python tools/export_tpot.py WEIGHTS_DIR [ENHANCER]   (a Python with torch and Pillow; nothing else)

WEIGHTS_DIR holds the two checkpoints from TPOT-SR's weights/ folder:
  best_SSIM.pth          the TPOT generator (model/model_LC.py, _NetG), a plain state_dict
  fusion_head_best.pth   the full-resolution fusion head (model/fusion_head.py, FusionRefineHead)

ENHANCER, the default for the app, is the lab's EyePACS enhancer: TPOT fine-tuned on EyePACS with four
small adapters (DA_wrapper.py, GeneratorWithFullAdapter), eyepacs_step3_aggressive_best.pt, whose
"model" holds the generator under "base." and the adapters. It replaces best_SSIM.pth; the fusion head
stays TPOT-SR's. Without ENHANCER the script exports TPOT-SR's own generator.

They are read as tensors only (torch.load(weights_only=True)); TPOT-SR's own code is not run.
The two networks are written again below with TPOT-SR's parameter names, so the checkpoints load
with strict=True: a changed architecture fails loudly instead of loading half the weights.

For onnxruntime-web the graphs are rewritten in three exactly equivalent ways, and the script
checks the rewrite against the original form in PyTorch:
  - spectral norm is folded into the weights (eval-mode sigma from the stored u and v),
  - the single shared PReLU slope becomes a LeakyReLU with that slope,
  - ECA's Conv1d across the channels becomes a (k, 1) Conv2d.

Writes webui/public/models/tpot/{tpot.onnx, fusion.onnx}, the manifest next to them, and reference
inputs/outputs in tools/tpot_ref/ for webui/tests/tpot.ts, which runs the ONNX files in
onnxruntime-web and compares them with this PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import remove_spectral_norm, spectral_norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "webui" / "public" / "models" / "tpot"
REF = ROOT / "tools" / "tpot_ref"
SAMPLE = ROOT / "webui" / "public" / "samples" / "cc0_left_3744.jpg"
SOURCE = "Retinal-Research/TPOT-SR @ 0620afa (weights/best_SSIM.pth, weights/fusion_head_best.pth)"
EYEPACS = "lab EyePACS enhancer (TPOT + adapters, eyepacs_step3_aggressive_best.pt, epoch {epoch}); fusion head from TPOT-SR @ 0620afa (weights/fusion_head_best.pth)"
OPSET = 17
LOW = 256      # TPOT works at 256 x 256
HIGH = 2048    # the fusion head works at 2048 x 2048


# ---------------------------------------------------------------- TPOT generator, as in model_LC.py
def conv(cin, cout, k, bias=False, stride=1):
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias, stride=stride)


class ECA(nn.Module):
    """EfficientCALayer: channel attention from a 1-D convolution across the pooled channels."""

    def __init__(self, k):
        super().__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = spectral_norm(nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=True))
        self.sigmoid = nn.Sigmoid()
        self.browser = False

    def forward(self, x):
        if not self.browser:
            y = self.avg_pooling(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            return x * self.sigmoid(y)
        b, c = x.shape[0], x.shape[1]
        k = self.conv.weight.shape[-1]
        y = x.mean(dim=(2, 3)).reshape(b, 1, c, 1)
        y = F.conv2d(y, self.conv.weight.reshape(1, 1, k, 1), self.conv.bias, padding=((k - 1) // 2, 0))
        return x * torch.sigmoid(y.reshape(b, c, 1, 1))


class Act(nn.PReLU):
    """The generator's one PReLU (a single slope, shared by every block); LeakyReLU in the browser."""

    def __init__(self):
        super().__init__()
        self.browser = False

    def forward(self, x):
        return F.leaky_relu(x, float(self.weight)) if self.browser else super().forward(x)


class CAB(nn.Module):
    def __init__(self, n, k, reduction, act):
        super().__init__()
        self.body = nn.Sequential(spectral_norm(conv(n, n, k)), act, spectral_norm(conv(n, n, k)))
        self.CA = ECA(reduction)

    def forward(self, x):
        return self.CA(self.body(x)) + x


class DownSample(nn.Module):
    def __init__(self, cin, s):
        super().__init__()
        self.down = nn.Conv2d(cin, cin + s, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.down(x)


class SkipUpSample(nn.Module):
    def __init__(self, cin, s):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                nn.Conv2d(cin + s, cin, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        return self.up(x) + y


class Encoder(nn.Module):
    def __init__(self, n, k, red, act, s):
        super().__init__()
        self.encoder_level1 = nn.Sequential(*[CAB(n, k, red[0], act) for _ in range(2)])
        self.encoder_level2 = nn.Sequential(*[CAB(n + s, k, red[1], act) for _ in range(2)])
        self.encoder_level3 = nn.Sequential(*[CAB(n + 2 * s, k, red[2], act) for _ in range(2)])
        self.down12 = DownSample(n, s)
        self.down23 = DownSample(n + s, s)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        enc2 = self.encoder_level2(self.down12(enc1))
        enc3 = self.encoder_level3(self.down23(enc2))
        return enc1, enc2, enc3


class Decoder(nn.Module):
    def __init__(self, n, k, red, act, s):
        super().__init__()
        self.decoder_level1 = nn.Sequential(*[CAB(n, k, red[2], act) for _ in range(2)])
        self.decoder_level2 = nn.Sequential(*[CAB(n + s, k, red[1], act) for _ in range(2)])
        self.decoder_level3 = nn.Sequential(*[CAB(n + 2 * s, k, red[0], act) for _ in range(2)])
        self.skip_attn1 = CAB(n, k, red[2], act)
        self.skip_attn2 = CAB(n + s, k, red[1], act)
        self.up21 = SkipUpSample(n, s)
        self.up32 = SkipUpSample(n + s, s)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        dec2 = self.decoder_level2(self.up32(dec3, self.skip_attn2(enc2)))
        return self.decoder_level1(self.up21(dec2, self.skip_attn1(enc1)))


class SAM(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.conv1 = spectral_norm(conv(n, n, k))
        self.conv2 = spectral_norm(conv(n, 3, k))
        self.conv3 = spectral_norm(conv(3, n, k))

    def forward(self, x, x_img):
        return self.conv2(x) + x_img  # the image branch; the feature branch is not used by _NetG


class TPOT(nn.Module):
    """_NetG(n_feat=80, scale_unetfeats=48, kernel_size=3, reduction=[3, 5, 7], bias=False), clamped like enhance.py."""

    def __init__(self):
        super().__init__()
        n, s, k, red = 80, 48, 3, [3, 5, 7]
        act = Act()
        self.shallow_feat1 = nn.Sequential(spectral_norm(conv(3, n, k)), CAB(n, k, 3, act))
        self.stage1_encoder = Encoder(n, k, red, act, s)
        self.stage1_decoder = Decoder(n, k, red, act, s)
        self.sam12 = SAM(n, 1)

    def forward(self, x):
        fea = self.stage1_decoder(self.stage1_encoder(self.shallow_feat1(x)))
        return torch.clamp(self.sam12(fea, x), 0.0, 1.0)


# ---------------------------------------------------------------- the EyePACS enhancer, as in DA_wrapper.py
class Adapter(nn.Module):
    """x + scale * (3x3, ReLU, 3x3, ReLU, 1x1); the dropout between them does nothing in eval."""

    def __init__(self, c, scale=1.0):
        super().__init__()
        self.scale = scale
        self.adapter = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Identity(),
                                     nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Identity(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.scale * self.adapter(x)


class BottleneckAdapter(nn.Module):
    """BottleneckAdapter(c, reduction=4, scale=0.5, use_deep=True)."""

    def __init__(self, c, scale=0.5):
        super().__init__()
        h = max(4, c // 4)
        self.scale = scale
        self.adapter = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(), nn.Identity(), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                                     nn.Identity(), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, c, 1))

    def forward(self, x):
        return x + self.scale * self.adapter(x)


class EyePACS(nn.Module):
    """GeneratorWithFullAdapter(_NetG(), n_feat=80, scale_unetfeats=48): TPOT with adapters after encoder and
    decoder level 3, on the decoder's output and on the image."""

    def __init__(self):
        super().__init__()
        self.base = TPOT()
        self.adapter_enc3 = BottleneckAdapter(176)
        self.adapter_dec3 = BottleneckAdapter(176)
        self.adapter_fea3_0 = Adapter(80)
        self.adapter_image = Adapter(3)

    def forward(self, x):
        b, enc, dec = self.base, self.base.stage1_encoder, self.base.stage1_decoder
        enc1 = enc.encoder_level1(b.shallow_feat1(x))
        enc2 = enc.encoder_level2(enc.down12(enc1))
        enc3 = self.adapter_enc3(enc.encoder_level3(enc.down23(enc2)))
        dec3 = self.adapter_dec3(dec.decoder_level3(enc3))
        dec2 = dec.decoder_level2(dec.up32(dec3, dec.skip_attn2(enc2)))
        dec1 = self.adapter_fea3_0(dec.decoder_level1(dec.up21(dec2, dec.skip_attn1(enc1))))
        return torch.clamp(self.adapter_image(b.sam12(dec1, x)), 0.0, 1.0)


class Fusion(nn.Module):
    """FusionRefineHead: clamp(e + body(cat(x, e)), 0, 1) at 2048 x 2048; 5 layers of 3 x 3 convolutions."""

    def __init__(self, n=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, n, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(n, n, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(n, n, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(n, n, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(n, 3, 3, 1, 1),
        )

    def forward(self, x_hr, e_hr):
        return torch.clamp(e_hr + self.body(torch.cat([x_hr, e_hr], dim=1)), 0.0, 1.0)


def load(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return sd


def for_browser(model: nn.Module) -> None:
    """Fold spectral norm into the weights and switch to the browser-friendly operators."""
    for m in model.modules():
        if hasattr(m, "weight_orig"):
            remove_spectral_norm(m)
        if isinstance(m, (ECA, Act)):
            m.browser = True


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    weights = Path(sys.argv[1])
    enhancer = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)

    if enhancer:
        tpot = EyePACS().eval()
        tpot.load_state_dict(load(enhancer), strict=True)
        epoch = torch.load(enhancer, map_location="cpu", weights_only=True).get("epoch", "?")
        source, label = EYEPACS.format(epoch=epoch), "TPOT fundus enhancement, EyePACS enhancer (lab model)"
    else:
        tpot = TPOT().eval()
        tpot.load_state_dict(load(weights / "best_SSIM.pth"), strict=True)
        source, label = SOURCE, "TPOT fundus enhancement (lab model, ISBI 2025)"
    fusion = Fusion().eval()
    fusion.load_state_dict(load(weights / "fusion_head_best.pth"), strict=True)
    print(f"TPOT {sum(p.numel() for p in tpot.parameters()):,} parameters; fusion head {sum(p.numel() for p in fusion.parameters()):,}")

    # a real photograph, square-cropped and resized with PIL bicubic like TPOT-SR's enhance.py
    photo = Image.open(SAMPLE).convert("RGB")
    side = min(photo.size)
    photo = photo.crop(((photo.width - side) // 2, (photo.height - side) // 2, (photo.width + side) // 2, (photo.height + side) // 2))
    x_lr = to_tensor(photo.resize((LOW, LOW), Image.BICUBIC))
    x_hr = to_tensor(photo.resize((HIGH, HIGH), Image.BICUBIC))

    with torch.no_grad():
        original = tpot(x_lr)
        for_browser(tpot)
        rewritten = tpot(x_lr)
        print(f"rewrite check, TPOT: max |original - browser form| = {(original - rewritten).abs().max():.2e}")
        e_hr = F.interpolate(rewritten, size=(HIGH, HIGH), mode="bilinear", align_corners=False)
        # the reference for the fusion head is one 522 x 522 tile with 5 pixels of context on each side
        y0, x0 = 700, 900
        tile_x, tile_e = x_hr[..., y0:y0 + 522, x0:x0 + 522], e_hr[..., y0:y0 + 522, x0:x0 + 522]
        fused = fusion(tile_x, tile_e)
        full = fusion(x_hr, e_hr)
        print(f"tiling check: max |tile - full image| inside the 5-pixel margin = "
              f"{(fused[..., 5:-5, 5:-5] - full[..., y0 + 5:y0 + 517, x0 + 5:x0 + 517]).abs().max():.2e}")

    for name, model, args, names, dynamic in (
        ("tpot", tpot, (x_lr,), ["image"], None),
        ("fusion", fusion, (tile_x, tile_e), ["image", "enhanced"], {"image": {2: "h", 3: "w"}, "enhanced": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}),
    ):
        path = OUT / f"{name}.onnx"
        torch.onnx.export(model, args, str(path), opset_version=OPSET, input_names=names, output_names=["output"],
                          dynamic_axes=dynamic, do_constant_folding=True, dynamo=False)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")

    for name, t in (("x_lr", x_lr), ("e_lr", rewritten), ("tile_x", tile_x), ("tile_e", tile_e), ("tile_out", fused)):
        (REF / f"{name}.f32").write_bytes(t.contiguous().numpy().astype("<f4").tobytes())

    manifest = {
        "version": 1,
        "source": source,
        "precision": "float32 (19 MB; small enough to keep full precision)",
        "models": {
            "tpot": {
                "label": label,
                "input": [1, 3, LOW, LOW],
                "members": "one network, one pass at 256 x 256, input and output in [0, 1]",
                "float32": True,
            },
            "tpot_fusion": {
                "label": "TPOT-SR fusion head (full resolution)",
                "input": [1, 6, "h", "w"],
                "members": "5 convolutions at 2048 x 2048, run in tiles with 5 pixels of context",
                "float32": True,
            },
        },
    }
    for key, name in (("tpot", "tpot"), ("tpot_fusion", "fusion")):
        p = OUT / f"{name}.onnx"
        manifest["models"][key]["bytes"] = p.stat().st_size
        manifest["models"][key]["parts"] = [{"file": f"{name}.onnx", "bytes": p.stat().st_size, "sha256": sha256(p), "member": name}]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {(OUT / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
