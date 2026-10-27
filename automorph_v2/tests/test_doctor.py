from __future__ import annotations

import json

from scripts import doctor


def test_doctor_requires_the_release_tqdm_pin() -> None:
    assert doctor.EXPECTED_PACKAGES["tqdm"] == ("tqdm", "4.67.1")


def test_doctor_rejects_a_mismatched_tqdm_version(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor, "EXPECTED_PACKAGES", {"tqdm": ("tqdm", "4.67.1")}
    )
    monkeypatch.setattr(doctor, "_distribution_version", lambda name: "4.66.0")
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: object())
    audit = doctor.Audit()
    doctor._check_imports(audit)
    assert audit.records == [
        {
            "check": "import:tqdm",
            "passed": False,
            "detail": "distribution tqdm==4.66.0; expected 4.67.1",
        },
        {"check": "import:torch", "passed": True, "detail": "ok"},
        {"check": "import:torchvision", "passed": True, "detail": "ok"},
        {"check": "import:automorph_v2", "passed": True, "detail": "ok"},
        {"check": "import:automorph_v2.cli", "passed": True, "detail": "ok"},
    ]


def test_model_manifest_verifies_hashes_and_detects_tampering(
    tmp_path, monkeypatch
) -> None:
    weight = tmp_path / "runtime" / "weight.pth"
    support = tmp_path / "runtime" / "support.py"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"checkpoint")
    support.write_bytes(b"support")
    monkeypatch.setattr(doctor, "_expected_weights", lambda root: {weight})
    monkeypatch.setattr(
        doctor,
        "SUPPORT_ASSET_PATHS",
        ("runtime/support.py",),
    )
    files = []
    for path, kind in ((weight, "checkpoint"), (support, "model_source")):
        files.append(
            {
                "module": "fixture",
                "kind": kind,
                "path": path.relative_to(tmp_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": doctor._sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "checkpoint_count": 1,
        "support_asset_count": 1,
        "total_checkpoint_bytes": weight.stat().st_size,
        "total_manifested_bytes": weight.stat().st_size + support.stat().st_size,
        "files": files,
    }
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    accepted = doctor.Audit()
    doctor._check_model_manifest(accepted, tmp_path)
    assert accepted.passed

    support.write_bytes(b"tampered")
    rejected = doctor.Audit()
    doctor._check_model_manifest(rejected, tmp_path)
    assert not rejected.passed
    assert rejected.records[0]["check"] == "model-manifest"
