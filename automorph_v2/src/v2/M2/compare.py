"""Decoded-output comparison for strict M2 acceptance gates."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover(
    root: Path, includes: Sequence[str], excludes: Sequence[str]
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    selected = includes or (".",)
    for item in selected:
        base = (root / item).resolve()
        try:
            base.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"include escapes comparison root: {item}") from error
        if not base.exists():
            continue
        candidates = (base,) if base.is_file() else base.rglob("*")
        for path in candidates:
            if path.is_file() and not path.name.startswith("."):
                relative = path.relative_to(root).as_posix()
                if any(
                    fnmatch.fnmatch(relative, pattern)
                    or fnmatch.fnmatch(path.name, pattern)
                    for pattern in excludes
                ):
                    continue
                files[relative] = path
    return files


def _image_metrics(reference: Path, candidate: Path) -> dict[str, Any]:
    with Image.open(reference) as image:
        reference_mode = image.mode
        reference_array = np.asarray(image).copy()
    with Image.open(candidate) as image:
        candidate_mode = image.mode
        candidate_array = np.asarray(image).copy()

    result: dict[str, Any] = {
        "kind": "image",
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "reference_shape": "x".join(map(str, reference_array.shape)),
        "candidate_shape": "x".join(map(str, candidate_array.shape)),
        "reference_dtype": str(reference_array.dtype),
        "candidate_dtype": str(candidate_array.dtype),
    }
    if reference_array.shape != candidate_array.shape:
        result.update(
            exact=False,
            changed_elements=None,
            changed_pixels=None,
            max_abs=None,
            mean_abs=None,
            mse=None,
            psnr_db=None,
            foreground_iou=None,
        )
        return result

    difference = np.abs(
        reference_array.astype(np.float64) - candidate_array.astype(np.float64)
    )
    changed = difference != 0
    changed_pixels = (
        np.any(changed, axis=tuple(range(2, changed.ndim)))
        if changed.ndim > 2
        else changed
    )
    mse = float(np.mean(np.square(difference))) if difference.size else 0.0
    reference_foreground = (
        np.any(reference_array != 0, axis=tuple(range(2, reference_array.ndim)))
        if reference_array.ndim > 2
        else reference_array != 0
    )
    candidate_foreground = (
        np.any(candidate_array != 0, axis=tuple(range(2, candidate_array.ndim)))
        if candidate_array.ndim > 2
        else candidate_array != 0
    )
    union = np.logical_or(reference_foreground, candidate_foreground)
    intersection = np.logical_and(reference_foreground, candidate_foreground)
    iou = float(intersection.sum() / union.sum()) if union.any() else 1.0
    result.update(
        exact=bool(
            reference_mode == candidate_mode
            and reference_array.dtype == candidate_array.dtype
            and not changed.any()
        ),
        changed_elements=int(changed.sum()),
        changed_pixels=int(changed_pixels.sum()),
        max_abs=float(difference.max()) if difference.size else 0.0,
        mean_abs=float(difference.mean()) if difference.size else 0.0,
        mse=mse,
        psnr_db=math.inf if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
        foreground_iou=iou,
    )
    return result


def compare_trees(
    reference_root: Path,
    candidate_root: Path,
    *,
    includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
    decode_images: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_root = Path(reference_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    reference_files = _discover(reference_root, includes, excludes)
    candidate_files = _discover(candidate_root, includes, excludes)
    names = sorted(set(reference_files) | set(candidate_files))
    rows: list[dict[str, Any]] = []

    for name in names:
        reference = reference_files.get(name)
        candidate = candidate_files.get(name)
        row: dict[str, Any] = {
            "relative_path": name,
            "in_reference": reference is not None,
            "in_candidate": candidate is not None,
        }
        if reference is None or candidate is None:
            row.update(kind="missing", exact=False, byte_exact=False)
        else:
            row["reference_sha256"] = _sha256(reference)
            row["candidate_sha256"] = _sha256(candidate)
            row["byte_exact"] = row["reference_sha256"] == row["candidate_sha256"]
            if reference.suffix.lower() in IMAGE_SUFFIXES and decode_images:
                row.update(_image_metrics(reference, candidate))
            else:
                row.update(
                    kind="image_hash"
                    if reference.suffix.lower() in IMAGE_SUFFIXES
                    else "binary",
                    exact=row["byte_exact"],
                )
        rows.append(row)

    comparable_images = [row for row in rows if row.get("kind") == "image"]
    summary: dict[str, Any] = {
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "includes": list(includes),
        "excludes": list(excludes),
        "decode_images": decode_images,
        "reference_files": len(reference_files),
        "candidate_files": len(candidate_files),
        "union_files": len(names),
        "missing_from_candidate": sorted(set(reference_files) - set(candidate_files)),
        "extra_in_candidate": sorted(set(candidate_files) - set(reference_files)),
        "decoded_images": len(comparable_images),
        "decoded_exact_images": sum(bool(row.get("exact")) for row in comparable_images),
        "byte_exact_files": sum(bool(row.get("byte_exact")) for row in rows),
        "changed_elements": sum(
            int(row.get("changed_elements") or 0) for row in comparable_images
        ),
        "max_abs": max(
            (float(row.get("max_abs") or 0) for row in comparable_images), default=0.0
        ),
    }
    summary["pixel_exact"] = bool(
        len(reference_files) == len(candidate_files)
        and all(row.get("exact") is True for row in rows)
    )
    return summary, rows


def write_comparison(
    output_dir: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "per_file.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare decoded M2 outputs")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="compare file bytes without decoding images",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, rows = compare_trees(
        args.reference,
        args.candidate,
        includes=args.include,
        excludes=args.exclude,
        decode_images=not args.hash_only,
    )
    write_comparison(args.output_dir, summary, rows)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0 if summary["pixel_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
