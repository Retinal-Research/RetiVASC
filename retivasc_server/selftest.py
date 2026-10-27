"""Acceptance test for a new installation: run the full AutoMorph v2 on the bundled sample
photos and compare quality probabilities and the 72 features with the reference results
recorded on the development workstation (retivasc_server/reference.json).

Different GPUs, drivers or CPU builds can change the last digits of a result, so the check
allows a tolerance and reports the largest difference it saw. The grade and the gradable call
must match exactly. The reference comes from an NVIDIA GPU: other NVIDIA cards must agree within
0.1%. The CPU and Apple GPUs round differently inside the networks, which moves a few segmented
pixels and so the features slightly (the CPU: up to 0.14% on the samples), so they are held to 1%.
Keep one study on one kind of device.
"""

from __future__ import annotations

import json

import shutil
import tempfile
import time
from pathlib import Path

from . import automorph, config, devices
from .automorph import load_manifest, stage
from .settings import Settings

SAMPLES = config.ROOT / "webui" / "public" / "samples"
REFERENCE = Path(__file__).with_name("reference.json")
PROB_TOL = 1e-3   # absolute, on probabilities
FEAT_TOL = 1e-3   # relative, on features, on an NVIDIA GPU like the reference
FEAT_TOL_OTHER = 1e-2  # relative, on features, on the CPU or an Apple GPU


def _results(data_root: Path) -> dict:
    out = automorph.load(data_root)
    names = {e["stem"]: Path(e["rel_path"]).name for e in load_manifest(data_root)["files"]}
    res = {}
    for stem, name in names.items():
        q = out.quality.get(stem)
        table, feats = out.features.get(stem, (None, None))
        res[name] = {
            "quality": None if q is None else {"good": q.p_good, "usable": q.p_usable, "reject": q.p_reject,
                                               "prediction": q.prediction, "gradable": q.gradable},
            "table": table, "features": feats,
        }
    return res


def _compare(got: dict, ref: dict, feat_tol: float = FEAT_TOL) -> tuple[list[str], float, float]:
    problems, worst_p, worst_f, worst_where = [], 0.0, 0.0, ""
    for name, r in ref.items():
        g = got.get(name)
        if g is None:
            problems.append(f"{name}: no result")
            continue
        rq, gq = r["quality"], g["quality"]
        if (rq is None) != (gq is None):
            problems.append(f"{name}: quality result missing")
            continue
        if rq:
            if rq["prediction"] != gq["prediction"] or rq["gradable"] != gq["gradable"]:
                problems.append(f"{name}: grade differs ({gq['prediction']} vs {rq['prediction']})")
            for k in ("good", "usable", "reject"):
                worst_p = max(worst_p, abs(gq[k] - rq[k]))
        if r["table"] != g["table"]:
            problems.append(f"{name}: centred differently ({g['table']} vs {r['table']})")
        for k, rv in (r["features"] or {}).items():
            gv = (g["features"] or {}).get(k)
            if (rv is None) != (gv is None):
                problems.append(f"{name}: {k} missing on one side")
            elif rv is not None:
                d = abs(gv - rv) / max(abs(rv), 1e-9)
                if d > worst_f:
                    worst_f, worst_where = d, f"{k} of {name}"
    if worst_p > PROB_TOL:
        problems.append(f"quality probabilities differ by up to {worst_p:.2e}")
    if worst_f > feat_tol:
        problems.append(f"features differ by up to {worst_f:.2e} (relative; {worst_where}), allowed {feat_tol:.0e}")
    return problems, worst_p, worst_f


def run(settings: Settings, write_reference: bool = False) -> int:
    photos = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in (".jpg", ".png", ".tif", ".tiff"))
    tmp = Path(tempfile.mkdtemp(prefix="retivasc_selftest_"))
    try:
        (tmp / "in").mkdir()
        for p in photos:
            shutil.copyfile(p, tmp / "in" / p.name)
        data_root = tmp / "run"
        stage(tmp / "in", data_root)
        device = devices.resolve(settings.device)
        print(f"running AutoMorph v2 on {len(photos)} sample photos ({device}) …", flush=True)
        t = time.time()
        automorph.run(data_root, device)
        print(f"finished in {time.time() - t:.0f} s")
        got = _results(data_root)
        if write_reference:
            REFERENCE.write_text(json.dumps(got, indent=1), encoding="utf-8")
            print(f"reference written: {REFERENCE}")
            return 0
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
        tol = FEAT_TOL if device.startswith("cuda") else FEAT_TOL_OTHER
        problems, wp, wf = _compare(got, ref, tol)
        print(f"largest difference: quality {wp:.1e}, features {wf:.1e} (relative; allowed on {device}: {tol:.0e})")
        if problems:
            print("SELFTEST FAILED")
            for p in problems:
                print("  -", p)
            return 1
        exact = wp == 0 and wf == 0
        print(f"SELFTEST PASSED: {len(ref)} photos, grades identical, results {'bit-identical' if exact else 'within tolerance'}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



