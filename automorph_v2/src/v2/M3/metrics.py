# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict active Retipy metrics used by AutoMorph M3.

This is a focused derivative of Retipy's ``tortuosity_measures.py``.
Copyright (C) 2017-2018 Alejandro Valdes and later AutoMorph contributors.
Licensed under GPL-3.0-or-later.  Arithmetic and accumulation order are kept
deliberately close to the legacy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from skimage import io

from .traversal import extract_vessels_exact
from .width import width_measurement


VesselType = Literal["binary", "artery", "vein"]


@dataclass(frozen=True)
class MetricResult:
    fractal_dimension: object
    vessel_density: object
    average_width: object
    distance_tortuosity: object
    squared_curvature_tortuosity: object
    tortuosity_density: object
    crae_hubbard: object = -1
    crve_hubbard: object = -1
    crae_knudtson: object = -1
    crve_knudtson: object = -1

    @property
    def base_values(self) -> tuple[object, ...]:
        return (
            self.fractal_dimension,
            self.vessel_density,
            self.average_width,
            self.distance_tortuosity,
            self.squared_curvature_tortuosity,
            self.tortuosity_density,
        )


def load_legacy_image(path: str | Path) -> np.ndarray:
    """Match Retipy's active ``skimage.imread`` plus cubic resize path."""

    return cv2.resize(
        io.imread(str(path)),
        dsize=(912, 912),
        interpolation=cv2.INTER_CUBIC,
    )


def fractal_dimension(image: np.ndarray) -> object:
    assert len(image.shape) == 2

    def boxcount(array: np.ndarray, size: object) -> int:
        sums = np.add.reduceat(
            np.add.reduceat(array, np.arange(0, array.shape[0], size), axis=0),
            np.arange(0, array.shape[1], size),
            axis=1,
        )
        return len(np.where((sums > 0) & (sums < size * size))[0])

    side = min(image.shape)
    exponent = 2 ** np.floor(np.log(side) / np.log(2))
    exponent = int(np.log(exponent) / np.log(2))
    sizes = 2 ** np.arange(exponent, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(image, size))
    coefficients = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coefficients[0]


def global_metrics(
    skeleton_image: np.ndarray,
    vessel_image: np.ndarray,
    resolution: object,
) -> tuple[object, object, object]:
    vessel = vessel_image
    skeleton = skeleton_image
    if np.max(vessel) > 1:
        vessel = vessel / 255
    if np.max(skeleton) > 1:
        skeleton = skeleton / 255
    fd_value = fractal_dimension(vessel)
    vd_value = np.sum(vessel == 1) / (vessel.shape[0] * vessel.shape[1])
    width_value = np.sum(vessel) / np.sum(skeleton) * resolution
    return fd_value, vd_value, width_value


def _distance_2p(x1: object, y1: object, x2: object, y2: object) -> object:
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _curve_length(x: Sequence[int], y: Sequence[int]) -> object:
    distance: object = 0
    for index in range(0, len(x) - 1):
        distance += _distance_2p(x[index], y[index], x[index + 1], y[index + 1])
    return distance


def _chord_length(x: Sequence[int], y: Sequence[int]) -> object:
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])


def _detect_inflection_points(x: Sequence[int], y: Sequence[int]) -> list[int]:
    convolution = np.convolve(y, [1, -1])
    points: list[int] = []
    for iterator in range(2, len(x)):
        if np.sign(convolution[iterator]) != np.sign(convolution[iterator - 1]):
            points.append(iterator - 1)
    return points


def linear_regression_tortuosity(
    x: Sequence[int],
    y: Sequence[int],
    sampling_size: int = 6,
    retry: bool = True,
) -> object:
    if len(x) < 4:
        raise ValueError("Given curve must have more than 4 elements")
    try:
        min_point_x = x[0]
        min_point_y = y[0]
        slope = (y[len(y) - 1] - min_point_y) / (x[len(x) - 1] - min_point_x)
        y_intercept = min_point_y - slope * min_point_x
        sample_distance = max(round(len(x) / sampling_size), 1)

        def fitted(value: object) -> object:
            return value * slope + y_intercept

        y_average: object = 0
        item_count = 0
        for index in range(1, len(x) - 1, sample_distance):
            y_average += y[index]
            item_count += 1
        y_average /= item_count
        top_sum: object = 0
        bottom_sum: object = 0
        for index in range(1, len(x) - 1, sample_distance):
            top_sum += (fitted(x[index]) - y_average) ** 2
            bottom_sum += (y[index] - y_average) ** 2
        r_squared = top_sum / bottom_sum
    except ZeroDivisionError:
        if retry:
            r_squared = linear_regression_tortuosity(y, x, retry=False)
        else:
            r_squared = 1
    if math.isnan(r_squared):
        r_squared = 0
    return r_squared


def distance_measure_tortuosity(x: Sequence[int], y: Sequence[int]) -> object:
    if len(x) < 2:
        raise ValueError("Given curve must have at least 2 elements")
    return _curve_length(x, y) / _chord_length(x, y)


def distance_inflection_count_tortuosity(
    x: Sequence[int], y: Sequence[int]
) -> tuple[object, int]:
    return (
        distance_measure_tortuosity(x, y) * (len(_detect_inflection_points(x, y)) + 1),
        len(_detect_inflection_points(x, y)),
    )


def tortuosity_density(x: Sequence[int], y: Sequence[int]) -> object:
    inflection_points = _detect_inflection_points(x, y)
    count = len(inflection_points)
    if not count:
        return 0
    starting_position = 0
    sum_segments: object = 0
    for point in inflection_points:
        segment_x = x[starting_position:point]
        segment_y = y[starting_position:point]
        chord = _chord_length(segment_x, segment_y)
        if chord:
            sum_segments += (
                _curve_length(segment_x, segment_y)
                / _chord_length(segment_x, segment_y)
                - 1
            )
        starting_position = point
    return (count - 1) / count + (1 / _curve_length(x, y)) * sum_segments


def _derivative1_centered_h1(target: int, values: Sequence[int]) -> object:
    if len(values) - 1 <= target <= 0:
        raise ValueError(f"Invalid target, array size {len(values)}, given {target}")
    return (values[target + 1] - values[target - 1]) / 2


def _derivative2_centered_h1(target: int, values: Sequence[int]) -> object:
    if len(values) - 1 <= target <= 0:
        raise ValueError(f"Invalid target, array size {len(values)}, given {target}")
    return (values[target + 1] - 2 * values[target] + values[target - 1]) / 4


def squared_curvature_tortuosity(x: Sequence[int], y: Sequence[int]) -> object:
    curvatures = []
    x_values = range(1, len(x) - 1)
    for index in x_values:
        x_1 = _derivative1_centered_h1(index, x)
        x_2 = _derivative2_centered_h1(index, x)
        y_1 = _derivative1_centered_h1(index, y)
        y_2 = _derivative2_centered_h1(index, y)
        curvatures.append((x_1 * y_2 - x_2 * y_1) / (y_1**2 + x_1**2) ** 1.5)
    return abs(np.trapz(curvatures, x_values))


def hubbard_cal(width_1: object, width_2: object) -> tuple[object, object]:
    artery = np.sqrt(
        0.87 * np.square(width_1)
        + 1.01 * np.square(width_2)
        - 0.22 * width_1 * width_2
        - 10.76
    )
    vein = np.sqrt(0.72 * np.square(width_1) + 0.91 * np.square(width_2) + 450.05)
    return artery, vein


def knudtson_cal(width_1: object, width_2: object) -> tuple[object, object]:
    artery = 0.88 * np.sqrt(np.square(width_1) + np.square(width_2))
    vein = 0.95 * np.sqrt(np.square(width_1) + np.square(width_2))
    return artery, vein


def _calibre_values(
    width_averages: list[object], vessel_type: VesselType
) -> tuple[object, object, object, object]:
    try:
        selected = sorted(width_averages)[-6:]
        first_hubbard_artery = []
        first_hubbard_vein = []
        for left, right in ((0, 5), (1, 4), (2, 3)):
            artery, vein = hubbard_cal(selected[left], selected[right])
            first_hubbard_artery.append(artery)
            first_hubbard_vein.append(vein)
        first_hubbard_artery = sorted(first_hubbard_artery)
        first_hubbard_vein = sorted(first_hubbard_vein)
        if vessel_type == "artery":
            second, _ = hubbard_cal(first_hubbard_artery[0], first_hubbard_artery[2])
            second_values = sorted([second, first_hubbard_artery[1]])
            crae_hubbard, crve_hubbard = hubbard_cal(second_values[0], second_values[1])
        else:
            _, second = hubbard_cal(first_hubbard_vein[0], first_hubbard_vein[2])
            second_values = sorted([second, first_hubbard_vein[1]])
            crae_hubbard, crve_hubbard = hubbard_cal(second_values[0], second_values[1])

        first_knudtson_artery = []
        first_knudtson_vein = []
        for left, right in ((0, 5), (1, 4), (2, 3)):
            artery, vein = knudtson_cal(selected[left], selected[right])
            first_knudtson_artery.append(artery)
            first_knudtson_vein.append(vein)
        first_knudtson_artery = sorted(first_knudtson_artery)
        first_knudtson_vein = sorted(first_knudtson_vein)
        if vessel_type == "artery":
            second, _ = knudtson_cal(first_knudtson_artery[0], first_knudtson_artery[2])
            second_values = sorted([second, first_knudtson_artery[1]])
            crae_knudtson, crve_knudtson = knudtson_cal(second_values[0], second_values[1])
        else:
            _, second = knudtson_cal(first_knudtson_vein[0], first_knudtson_vein[2])
            second_values = sorted([second, first_knudtson_vein[1]])
            crae_knudtson, crve_knudtson = knudtson_cal(second_values[0], second_values[1])
        return crae_hubbard, crve_hubbard, crae_knudtson, crve_knudtson
    except Exception:
        return -1, -1, -1, -1


def evaluate_arrays(
    skeleton_image: np.ndarray,
    vessel_image: np.ndarray,
    resolution: object,
    *,
    is_zone: bool,
    vessel_type: VesselType,
    min_pixels_per_vessel: int = 15,
    sampling_size: int = 6,
    r2_threshold: float = 0.96,
) -> MetricResult:
    """Evaluate one paired M2 skeleton/process image in legacy order."""

    # The legacy driver first wraps the 912 x 912 skeleton in one Retipy
    # Window.  Window.create_windows rejects it when the raw (not binarised)
    # pixel sum is below PixelsPerWindow, and the driver then emits an all--1
    # row.  Keep that admission rule before calculating any partial metrics.
    if skeleton_image.sum() < min_pixels_per_vessel:
        raise ValueError("No windows were created for the given retinal image")

    fd_value, vd_value, average_width = global_metrics(
        skeleton_image, vessel_image, resolution
    )
    vessels = extract_vessels_exact(skeleton_image)
    vessel_count = 0
    t2: object = 0
    t4: object = 0
    td: object = 0
    width_averages: list[object] = []

    for vessel in vessels:
        x, y = vessel
        if len(x) > min_pixels_per_vessel:
            vessel_count += 1
            if is_zone:
                # Retain legacy calls even though t1/t3 are not exported.  This
                # preserves valid-input failure behavior while the strict oracle
                # is established.
                linear_regression_tortuosity(x, y, sampling_size)
            t2 += distance_measure_tortuosity(x, y)
            if is_zone:
                _curve_length(x, y)
                distance_inflection_count_tortuosity(x, y)
            t4 += squared_curvature_tortuosity(x, y)
            td += tortuosity_density(x, y)
            if is_zone:
                widths = width_measurement(x, y, vessel_image, resolution)
                width_averages.append(sum(widths) / len(widths))

    if vessel_count > 0:
        t2 = t2 / vessel_count
        t4 = t4 / vessel_count
        td = td / vessel_count

    calibre = (-1, -1, -1, -1)
    if is_zone:
        calibre = _calibre_values(width_averages, vessel_type)
    return MetricResult(
        fd_value,
        vd_value,
        average_width,
        t2,
        t4,
        td,
        *calibre,
    )


def evaluate_paths(
    skeleton_path: str | Path,
    process_path: str | Path,
    resolution: object,
    *,
    is_zone: bool,
    vessel_type: VesselType,
    min_pixels_per_vessel: int = 15,
    sampling_size: int = 6,
    r2_threshold: float = 0.96,
) -> MetricResult:
    skeleton = load_legacy_image(skeleton_path)
    vessel = load_legacy_image(process_path)
    # Mirror Window.create_windows at the path API boundary too.  Keeping the
    # same check in evaluate_arrays protects callers that already loaded the
    # images, while this check guarantees no metric work starts for a window
    # the original driver would reject.
    if skeleton.sum() < min_pixels_per_vessel:
        raise ValueError("No windows were created for the given retinal image")
    return evaluate_arrays(
        skeleton,
        vessel,
        resolution,
        is_zone=is_zone,
        vessel_type=vessel_type,
        min_pixels_per_vessel=min_pixels_per_vessel,
        sampling_size=sampling_size,
        r2_threshold=r2_threshold,
    )
