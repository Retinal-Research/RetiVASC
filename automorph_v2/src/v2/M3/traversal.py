"""Exact, CPU-only vessel traversal for AutoMorph M3.

This module contains an optimized derivative of ``detect_vessel_border`` from
Retipy.  It preserves the legacy junction-removal and FIFO traversal order but
replaces both full-image Python scans with NumPy operations.

Retipy - Retinal Image Processing on Python
Copyright (C) 2017 Alejandro Valdes

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections import deque
from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vessel: TypeAlias = list[list[int]]
VesselList: TypeAlias = list[Vessel]

__all__ = ["detect_vessel_border", "extract_vessels_exact"]


def _validate_image(image: ArrayLike, ignored_pixels: int) -> NDArray[np.generic]:
    """Return a two-dimensional array and validate the legacy border option."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"expected a two-dimensional image, got shape {array.shape}")
    if isinstance(ignored_pixels, bool) or not isinstance(ignored_pixels, Integral):
        raise TypeError("ignored_pixels must be an integer")
    if ignored_pixels < 0:
        raise ValueError("ignored_pixels must be non-negative")
    return array


def _interior(shape: tuple[int, int], ignored_pixels: int) -> NDArray[np.bool_]:
    """Return the row/column seed domain used by the two legacy scans."""

    height, width = shape
    domain = np.zeros(shape, dtype=bool)
    if ignored_pixels < height - ignored_pixels and ignored_pixels < width - ignored_pixels:
        domain[
            ignored_pixels : height - ignored_pixels,
            ignored_pixels : width - ignored_pixels,
        ] = True
    return domain


def _legacy_neighbour_counts(active: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Count the eight legacy neighbours for every pixel at once.

    Retipy clamps neighbour coordinates at image boundaries.  ``mode='edge'``
    intentionally reproduces the resulting duplicate samples when
    ``ignored_pixels=0``; zero padding would not be equivalent there.
    """

    if active.size == 0:
        return np.zeros(active.shape, dtype=np.uint8)

    padded = np.pad(active.astype(np.uint8, copy=False), 1, mode="edge")
    height, width = active.shape
    counts = np.zeros(active.shape, dtype=np.uint8)
    for delta_x in (-1, 0, 1):
        for delta_y in (-1, 0, 1):
            if delta_x == 0 and delta_y == 0:
                continue
            counts += padded[
                1 + delta_x : 1 + delta_x + height,
                1 + delta_y : 1 + delta_y + width,
            ]
    return counts


def _opencv_radius_one_union(junctions: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Union filled OpenCV radius-one circles without calling OpenCV.

    For an integer centre, ``cv2.circle(..., radius=1, thickness=-1)`` covers
    exactly the centre and its four axial neighbours.  Explicit slices keep
    that cross footprint deterministic on Windows, Linux, and macOS.
    """

    removed = junctions.copy()
    if junctions.shape[0] > 1:
        removed[1:, :] |= junctions[:-1, :]
        removed[:-1, :] |= junctions[1:, :]
    if junctions.shape[1] > 1:
        removed[:, 1:] |= junctions[:, :-1]
        removed[:, :-1] |= junctions[:, 1:]
    return removed


def _append_active_neighbours(
    pending: deque[list[int]], work: NDArray[np.bool_], x: int, y: int
) -> None:
    """Append active neighbours in Retipy's original order."""

    x_less = max(0, x - 1)
    y_less = max(0, y - 1)
    x_more = min(work.shape[0] - 1, x + 1)
    y_more = min(work.shape[1] - 1, y + 1)

    # Do not deduplicate boundary coordinates: legacy Retipy appends every
    # occurrence, and FIFO order is part of the numerical output contract.
    positions = (
        (x_less, y_less),
        (x_less, y),
        (x_less, y_more),
        (x, y_less),
        (x, y_more),
        (x_more, y_less),
        (x_more, y),
        (x_more, y_more),
    )
    for neighbour_x, neighbour_y in positions:
        if work[neighbour_x, neighbour_y]:
            pending.append([neighbour_x, neighbour_y])


def extract_vessels_exact(image: ArrayLike, ignored_pixels: int = 1) -> VesselList:
    """Extract vessels with the exact ordering of legacy Retipy.

    Positive pixels are treated as vessel pixels, just as in the legacy
    ``> 0`` checks.  The input is never mutated.  Each returned vessel is
    ``[x_coordinates, y_coordinates]`` and vessels are ordered by the original
    row-major seed scan.

    The exact ordering matters: downstream tortuosity calculations use the
    first/last point and accumulate floating-point values in traversal order.

    Args:
        image: Two-dimensional skeleton image.
        ignored_pixels: Border width excluded from junction and seed scans.
            Connected traversal may still reach border pixels, matching
            Retipy.  The production M3 path uses the legacy default of one.

    Returns:
        A list of vessels represented by separate x- and y-coordinate lists.
    """

    array = _validate_image(image, ignored_pixels)
    if array.size == 0:
        return []

    active = np.asarray(array > 0, dtype=bool)
    seed_domain = _interior(active.shape, int(ignored_pixels))

    neighbour_counts = _legacy_neighbour_counts(active)
    junctions = active & seed_domain & (neighbour_counts > 2)
    work = active & ~_opencv_radius_one_union(junctions)

    # np.argwhere enumerates a C-contiguous 2-D mask in the same x-major,
    # y-minor order as the legacy nested range loops.  The list is captured
    # before traversal; stale seeds are skipped after earlier components clear
    # them, again matching the original code.
    seeds = np.argwhere(work & seed_domain)
    vessels: VesselList = []

    for seed_x, seed_y in seeds:
        x = int(seed_x)
        y = int(seed_y)
        if not work[x, y]:
            continue

        pending: deque[list[int]] = deque([[x, y]])
        vessel_pixels: list[list[int]] = []
        while pending:
            pixel_x, pixel_y = pending.popleft()
            if not work[pixel_x, pixel_y]:
                continue

            vessel_pixels.append([pixel_x, pixel_y])
            work[pixel_x, pixel_y] = False
            _append_active_neighbours(pending, work, pixel_x, pixel_y)

        vessels.append(
            [
                [pixel[0] for pixel in vessel_pixels],
                [pixel[1] for pixel in vessel_pixels],
            ]
        )

    return vessels


def detect_vessel_border(image: ArrayLike, ignored_pixels: int = 1) -> VesselList:
    """Compatibility entry point for the legacy Retipy function name.

    Unlike the legacy object-based implementation, ``image`` is the skeleton
    array itself and is not mutated.
    """

    return extract_vessels_exact(image, ignored_pixels=ignored_pixels)


def _legacy_extract_vessels_oracle(image: ArrayLike, ignored_pixels: int = 1) -> VesselList:
    """Slow, literal Retipy oracle retained solely for equivalence tests.

    This intentionally uses the original nested scans, OpenCV circle call, and
    ``list.pop(0)`` queue.  Production code must call
    :func:`extract_vessels_exact` instead.
    """

    array = _validate_image(image, ignored_pixels)
    if array.size == 0:
        return []

    # Import lazily so the optimized production path itself has no OpenCV
    # runtime dependency.
    import cv2  # pylint: disable=import-outside-toplevel

    work = array.copy()
    mask = np.ones(work.shape)
    height, width = work.shape

    def neighbours(pixel: list[int]) -> list[list[int]]:
        x, y = pixel
        x_less = max(0, x - 1)
        y_less = max(0, y - 1)
        x_more = min(height - 1, x + 1)
        y_more = min(width - 1, y + 1)
        result: list[list[int]] = []
        for neighbour_x, neighbour_y in (
            (x_less, y_less),
            (x_less, y),
            (x_less, y_more),
            (x, y_less),
            (x, y_more),
            (x_more, y_less),
            (x_more, y),
            (x_more, y_more),
        ):
            if work[neighbour_x, neighbour_y] > 0:
                result.append([neighbour_x, neighbour_y])
        return result

    for x in range(ignored_pixels, height - ignored_pixels):
        for y in range(ignored_pixels, width - ignored_pixels):
            if work[x, y] <= 0:
                continue
            x_less = max(0, x - 1)
            y_less = max(0, y - 1)
            x_more = min(height - 1, x + 1)
            y_more = min(width - 1, y + 1)
            active_neighbours = (
                (work[x_less, y_less] > 0).astype("float")
                + (work[x_less, y] > 0).astype("float")
                + (work[x_less, y_more] > 0).astype("float")
                + (work[x, y_less] > 0).astype("float")
                + (work[x, y_more] > 0).astype("float")
                + (work[x_more, y_less] > 0).astype("float")
                + (work[x_more, y] > 0).astype("float")
                + (work[x_more, y_more] > 0).astype("float")
            )
            if active_neighbours > 2:
                cv2.circle(mask, (y, x), radius=1, color=(0, 0, 0), thickness=-1)

    work = work * mask
    vessels: VesselList = []
    for x in range(ignored_pixels, height - ignored_pixels):
        for y in range(ignored_pixels, width - ignored_pixels):
            if work[x, y] <= 0:
                continue

            vessel_pixels: list[list[int]] = []
            pending_pixels = [[x, y]]
            while pending_pixels:
                pixel = pending_pixels.pop(0)
                if work[pixel[0], pixel[1]] > 0:
                    vessel_pixels.append(pixel)
                    work[pixel[0], pixel[1]] = 0
                    pending_pixels.extend(neighbours(pixel))

            vessels.append(
                [
                    [pixel[0] for pixel in vessel_pixels],
                    [pixel[1] for pixel in vessel_pixels],
                ]
            )

    return vessels
