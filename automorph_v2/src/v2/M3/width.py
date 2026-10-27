# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact, allocation-efficient vessel-width measurement for AutoMorph M3.

The oracle in :func:`legacy_width_measurement` preserves the ``width_measurement``
algorithm bundled with Retipy/AutoMorph (Copyright 2017-2018 Alejandro Valdes
and later AutoMorph contributors, GPLv3).  The optimized implementation keeps
the same OpenCV filled-circle raster, boundary clipping, stopping condition,
zero-count correction, and final resolution multiplication order.

Only the storage strategy changes: a radius' OpenCV-generated circle offsets
are cached once and applied directly around each vessel point.  No distance
transform or approximate geometric substitute is used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import cv2
import numpy as np


# Keeping all radii up to 128 costs roughly 36 MiB on a 64-bit Python. Retinal
# widths are normally far below that value. Larger pathological radii remain
# exact but are generated transiently, bounding persistent cache memory.
MAX_CACHED_RADIUS = 128


def _inputs(
    retinal_or_map: Any,
    resolution: Any | None,
) -> tuple[np.ndarray, Any]:
    """Accept either the legacy Retina object or an explicit map/resolution."""

    if hasattr(retinal_or_map, "vessel_image"):
        vessel_map = np.asarray(retinal_or_map.vessel_image)
        if resolution is None:
            resolution = retinal_or_map.resolution
    else:
        vessel_map = np.asarray(retinal_or_map)

    if resolution is None:
        raise TypeError("resolution is required when vessel_map is passed directly")
    if vessel_map.ndim != 2:
        raise ValueError(f"vessel_map must be 2-D, got shape {vessel_map.shape!r}")
    return vessel_map, resolution


def _build_circle_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact cumulative legacy circle offsets for ``radius``.

    Retipy repeatedly draws filled circles onto the same float64 canvas.  The
    cache deliberately asks OpenCV to perform those same draws rather than
    replacing its integer rasterization with a Euclidean-distance formula.
    Offsets are emitted in row-major order, matching NumPy boolean indexing.
    """

    size = radius * 2 + 1
    circle = np.zeros((size, size))  # Legacy np.zeros dtype: float64.
    center = (radius, radius)
    # A filled OpenCV circle contains every smaller filled circle. The test
    # suite compares this one-draw raster against the literal cumulative oracle
    # for representative radii, so cache construction itself stays cheap.
    cv2.circle(
        circle,
        center,
        radius=radius,
        color=(255, 255, 255),
        thickness=-1,
    )

    rows, cols = np.nonzero(circle > 0)
    row_offsets = (rows - radius).astype(np.intp, copy=False)
    col_offsets = (cols - radius).astype(np.intp, copy=False)
    row_offsets.flags.writeable = False
    col_offsets.flags.writeable = False
    return row_offsets, col_offsets


@lru_cache(maxsize=MAX_CACHED_RADIUS)
def _cached_circle_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    return _build_circle_offsets(radius)


def _circle_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact offsets, caching common radii and bounding retained memory."""

    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be at least 1")
    if radius <= MAX_CACHED_RADIUS:
        return _cached_circle_offsets(radius)
    return _build_circle_offsets(radius)


def clear_circle_offset_cache() -> None:
    """Release all retained circle-offset arrays (mainly useful in tests)."""

    _cached_circle_offsets.cache_clear()


def legacy_width_measurement(
    x: Sequence[int],
    y: Sequence[int],
    vessel_map: np.ndarray | Any,
    resolution: float | None = None,
) -> list[Any]:
    """Literal Retipy width oracle, retained for equivalence tests.

    Like the original function, the last coordinate is intentionally skipped.
    A map containing no zero-valued pixel does not terminate in the original
    algorithm and is therefore outside this function's supported input domain.
    """

    vessel_map, resolution = _inputs(vessel_map, resolution)
    width_list: list[Any] = []

    for i in range(0, len(x) - 1):
        width = 0
        width_matrix: Any = 1
        width_mask = np.zeros(vessel_map.shape)
        width_cal = 0

        while width_matrix:
            width += 1
            cv2.circle(
                width_mask,
                (int(y[i]), int(x[i])),
                radius=width,
                color=(255, 255, 255),
                thickness=-1,
            )
            masked_vessel = vessel_map[width_mask > 0]
            width_matrix = np.all(masked_vessel > 0)

        zero_count = np.sum(masked_vessel == 0)
        if zero_count == 1:
            width_cal = width * 2
        elif zero_count == 2:
            width_cal = width * 2 - 1
        elif zero_count == 3:
            width_cal = width * 2 - 1
        else:
            width_cal = width * 2

        # Keep the legacy operation order and the resolution scalar's dtype.
        width_cal = width_cal * resolution
        width_list.append(width_cal)

    return width_list


def width_measurement(
    x: Sequence[int],
    y: Sequence[int],
    vessel_map: np.ndarray | Any,
    resolution: float | None = None,
) -> list[float]:
    """Measure vessel widths with exact cached OpenCV circle offsets.

    This is a drop-in replacement for Retipy's ``width_measurement(x, y,
    retinal)``.  Tests may instead supply a 2-D vessel map and ``resolution``.
    Values greater than zero are vessel pixels; only values exactly equal to
    zero contribute to the legacy diameter correction.
    """

    vessel_map, resolution = _inputs(vessel_map, resolution)
    width_list: list[float] = []

    if len(x) <= 1:
        return width_list
    if len(y) < len(x):
        raise IndexError("y must contain at least as many coordinates as x")
    if not np.any(vessel_map == 0):
        raise ValueError(
            "vessel_map contains no zero pixel; the legacy radius loop would not terminate"
        )

    height, width_px = vessel_map.shape
    for i in range(0, len(x) - 1):
        center_row = int(x[i])
        center_col = int(y[i])
        radius = 0

        while True:
            radius += 1
            row_offsets, col_offsets = _circle_offsets(radius)
            rows = row_offsets + center_row
            cols = col_offsets + center_col
            inside = (
                (rows >= 0)
                & (rows < height)
                & (cols >= 0)
                & (cols < width_px)
            )
            masked_vessel = vessel_map[rows[inside], cols[inside]]

            if np.all(masked_vessel > 0):
                continue

            zero_count = np.sum(masked_vessel == 0)
            if zero_count == 1:
                width_cal = radius * 2
            elif zero_count == 2:
                width_cal = radius * 2 - 1
            elif zero_count == 3:
                width_cal = radius * 2 - 1
            else:
                width_cal = radius * 2

            # Keep the legacy operation order and the resolution scalar's dtype.
            width_cal = width_cal * resolution
            width_list.append(width_cal)
            break

    return width_list


__all__ = [
    "MAX_CACHED_RADIUS",
    "clear_circle_offset_cache",
    "legacy_width_measurement",
    "width_measurement",
]
