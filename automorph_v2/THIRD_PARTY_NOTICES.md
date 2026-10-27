# Third-party notices and provenance

This downstream AutoMorph v2 distribution combines original optimization and
packaging work with source, algorithms, and model artifacts originating in
other projects. This file records the provenance visible in the source tree;
it is not legal advice and does not replace the referenced license texts.

## AutoMorph

AutoMorph v2 is based on the AutoMorph project:

- Upstream source: <https://github.com/rmaphoh/AutoMorph>
- Project site: <https://rmaphoh.github.io/projects/automorph.html>
- Upstream code license presented by that repository: Apache License 2.0
- Included license text: [`LICENSE`](LICENSE)

The scientific method and original pipeline are described in:

> Yukun Zhou, Siegfried K. Wagner, Mark A. Chia, An Zhao,
> Peter Woodward-Court, Moucheng Xu, Robbert Struyven,
> Daniel C. Alexander, and Pearse A. Keane. "AutoMorph: Automated Retinal
> Vascular Morphology Quantification Via a Deep Learning Pipeline."
> *Translational Vision Science & Technology* 11(7), article 12, 2022.
> <https://doi.org/10.1167/tvst.11.7.12>

The Apache-2.0 notice above describes the upstream repository's stated code
license. It should not be read as a determination that every separately
originating component or trained checkpoint is covered by that license.

## Retipy-derived M3 feature code

The optimized M3 implementation contains modified derivatives of Retipy
algorithms, including code in:

- `src/v2/M3/metrics.py`
- `src/v2/M3/traversal.py`
- `src/v2/M3/width.py`

Retipy identifies Alejandro Valdes as its author and licenses its code under
the GNU General Public License, version 3 or, according to its source headers,
any later version.

- Retipy source: <https://github.com/alevalv/retipy>
- Included license text: [`LICENSES/RETIPY_GPL-3.0.md`](LICENSES/RETIPY_GPL-3.0.md)
- Applicable SPDX identifier for the derivative files: `GPL-3.0-or-later`

Those files retain explicit provenance and license headers. The presence of an
Apache-2.0 license at the release root does not override their GPL terms.

## Model architecture provenance

The release retains the exact model definitions needed to read the tested
AutoMorph checkpoints. The upstream AutoMorph documentation attributes the
major components to the following projects:

- EyeQ image-quality preprocessing: <https://github.com/HzFu/EyeQ>
- BF-Net / artery-vein segmentation:
  <https://github.com/rmaphoh/Learning-AVSegmentation>
- lwnet optic-disc/cup segmentation: <https://github.com/agaldran/lwnet>
- Retipy feature measurement: <https://github.com/alevalv/retipy>

The exact retained model-source files and their SHA-256 digests are listed in
[`models/manifest.json`](models/manifest.json). Attribution here does not alter
the license of any upstream project, nor does it imply endorsement by its
authors.

## Trained checkpoints

The manifest inventories 50 checkpoint files required for strict compatibility
with the tested AutoMorph ensembles. A checksum proves only which bytes were
tested. It is not a copyright license, patent license, trademark license, or
grant of permission to redistribute a checkpoint.

No separate checkpoint redistribution grant was identified in the packaged
model directories during preparation of this release. Before publishing,
mirroring, or attaching the checkpoint files to a GitHub release, the release
maintainer must independently confirm authorization from the relevant rights
holders and compliance with any training-data or model-specific terms. If that
authorization is unavailable, distribute only the source and manifest and have
users obtain the checkpoints from an authorized upstream location.

## Installed dependencies

Python, PyTorch, torchvision, EfficientNet-PyTorch, NumPy, pandas, Pillow,
OpenCV, SciPy, and scikit-image are installed as external dependencies rather
than vendored into this source tree. Each is governed by its own license. Refer
to the installed package metadata and upstream project for the version selected
for a particular platform.
