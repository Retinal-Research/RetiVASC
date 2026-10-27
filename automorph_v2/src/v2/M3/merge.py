"""Deterministic, non-destructive assembly of the legacy M3 feature tables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import pandas as pd


DISC_KEYS = [
    "Name",
    "Disc_height",
    "Disc_width",
    "Cup_height",
    "Cup_width",
    "CDR_vertical",
    "CDR_horizontal",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required M3 table is missing: {path}")
    return pd.read_csv(path)


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_feature_tables(m3_root: str | Path) -> Mapping[str, Path]:
    """Reproduce ``csv_merge.py`` without deleting intermediate evidence."""

    root = Path(m3_root).expanduser().resolve()
    disc_dir = root / "Disc_centred"
    macular_dir = root / "Macular_centred"

    disc_whole = _read_csv(disc_dir / "Disc_Measurement.csv")
    disc_zone_b = _read_csv(disc_dir / "Disc_Zone_B_Measurement.csv")
    disc_zone_c = _read_csv(disc_dir / "Disc_Zone_C_Measurement.csv")
    macular_whole = _read_csv(macular_dir / "Macular_Measurement.csv")
    macular_zone_b = _read_csv(macular_dir / "Macular_Zone_B_Measurement.csv")
    macular_zone_c = _read_csv(macular_dir / "Macular_Zone_C_Measurement.csv")

    disc_zone = disc_zone_b.merge(
        disc_zone_c,
        how="outer",
        on=DISC_KEYS,
        suffixes=("_zone_b", "_zone_c"),
    )
    disc_all = disc_whole.merge(
        disc_zone,
        how="outer",
        on=DISC_KEYS,
    )

    macular_zone = macular_zone_b.merge(
        macular_zone_c,
        how="outer",
        on=DISC_KEYS,
        suffixes=("_zone_b", "_zone_c"),
    )
    macular_all = macular_whole.merge(
        macular_zone,
        how="outer",
        on=DISC_KEYS,
    )

    # Match the legacy public contract: failed per-image values become empty
    # CSV cells. Structural failures are rejected before this point.
    disc_all.replace(-1, "", inplace=True)
    macular_all.replace(-1, "", inplace=True)

    disc_output = root / "Disc_Features.csv"
    macular_output = root / "Macular_Features.csv"
    _write_csv_atomic(disc_all, disc_output)
    _write_csv_atomic(macular_all, macular_output)
    return {"disc": disc_output, "macular": macular_output}
