"""Reference outputs of RetiFlow, the Python original, for the browser port (webui/src/retiflow).

  python tools/retiflow_reference.py <folder holding the RetiFlow package>

Reads tools/retiflow_ref/av3.rgb, an RRWNet AV3 map (584 x 584, RGB: arteries, veins, vessels) that
webui/tests/retiflow.ts writes, and runs each step of RetiFlow's infer.process_prob on it, keeping what
every step returns, so the test can check the port step by step:

  completion   a_mask.u8, v_mask.u8, conditional_a.f32
  centrelines  a_centerline.u8, v_centerline.u8
  two-pass     a_repaired.u8, v_repaired.u8 and, in result.json, the root and every bifurcation
  summary      result.json: process_prob's own summary, run separately

Needs RetiFlow's requirements (numpy, scipy, scikit-image, opencv-python, matplotlib, imageio).
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parent / "retiflow_ref"
SIDE = 584


def raw(name: str, array: np.ndarray, dtype) -> None:
    np.ascontiguousarray(array, dtype=dtype).tofile(REF / name)


def angles(found: list[dict]) -> list[dict]:
    return [{"x": a["points"][0], "y": a["points"][1], "angle": a["angle"], "children": a["children"]} for a in found]


def main(package_parent: Path) -> None:
    sys.path.insert(0, str(package_parent))
    from RetiFlow import infer
    from RetiFlow.completion.centerline import extract_centerline
    from RetiFlow.completion.completion import ProbabilityConfig, complete_probability_masks
    from RetiFlow.config import Config
    from RetiFlow.detect.endpoint_bridge import bridge_endpoints
    from RetiFlow.detect.two_pass import run_original

    img = np.fromfile(REF / "av3.rgb", np.uint8).reshape(SIDE, SIDE, 3)
    a, v, bv = (img[..., c].astype(np.float32) / 255.0 for c in range(3))
    cfg = Config()

    # infer.process_prob, step by step
    # the completion settings infer.py passes (config.py), not ProbabilityConfig's own defaults, which differ
    completion = ProbabilityConfig(**{k: x for k, x in asdict(cfg.completion).items() if k in ProbabilityConfig.__dataclass_fields__})
    am, vm, debug = complete_probability_masks(a, v, bv, config=completion)
    raw("a_mask.u8", am, np.uint8)
    raw("v_mask.u8", vm, np.uint8)
    raw("conditional_a.f32", debug["conditional_a"], np.float32)
    sa, _ = extract_centerline(bv * (0.65 + 0.35 * debug["conditional_a"]), am)
    sv, _ = extract_centerline(bv * (0.65 + 0.35 * (1 - debug["conditional_a"])), vm)
    raw("a_centerline.u8", sa, np.uint8)
    raw("v_centerline.u8", sv, np.uint8)
    root = [int(i) for i in infer._root_from_bv(bv)]
    bridge = infer._bridge_config(cfg)
    result = {"root": root}
    for name, skel in (("a", sa), ("v", sv)):
        before = run_original(skel, root_yx=root)
        repaired, _ = bridge_endpoints(skel, bv_probability=bv, config=bridge)
        after = run_original(repaired, root_yx=before["root_yx"])
        raw(f"{name}_repaired.u8", repaired, np.uint8)
        result[name] = {"before": angles(before["angles"]), "after": angles(after["angles"]),
                        "root": before["root_yx"]}

    # and process_prob itself, which the steps above must agree with
    with tempfile.TemporaryDirectory() as out:
        summary = infer.process_prob(a, v, bv, cfg, out)
    result["summary"] = summary
    (REF / "result.json").write_text(json.dumps(result, indent=1))
    for name in ("A", "V"):
        s = summary["RBAD"][name]
        print(f"{name}: {s['count']} bifurcations, mean angle {s['mean_angle']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
