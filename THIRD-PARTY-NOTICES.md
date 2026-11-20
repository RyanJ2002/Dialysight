# Third-Party Notices

This project incorporates or depends on the third-party components listed below.
Each component remains subject to its own licence.

本專案使用下列第三方元件，各元件仍受其原授權條款規範。

---

## ⚠️ Copyleft component — please read first

| Component | Version | Licence | Used for |
|---|---|---|---|
| **Ultralytics YOLOv8** (`ultralytics`) | >= 8.1 | **AGPL-3.0-or-later** | Development-time pose-estimation backend (`src/inference/ultralytics_backend.py`) |
| **`yolov8s-pose.pt`** model weights | — | **AGPL-3.0-or-later** | Pose estimation during Stage I evaluation. Not redistributed in this repository; downloaded at runtime by `ultralytics`. |

**AGPL-3.0 is a strong copyleft licence.** Anyone who distributes this software, or
makes it available to users over a network, must provide the complete corresponding
source under AGPL-3.0. AGPL-3.0 terms cannot be sublicensed under different terms.

This repository is therefore licensed under **AGPL-3.0** (see [`LICENSE`](LICENSE)).

`ultralytics` is **one of three interchangeable inference backends**, selected at
runtime by `inference.backend` in the configuration and imported lazily inside
`make_backend()` (`src/inference/base.py`). The other two backends — `replay`
(reads pre-extracted keypoints) and `hailo` (Hailo-10H HEF) — do not depend on
`ultralytics`.

**Open item:** the competition terms request a perpetual, royalty-free,
sublicensable commercial licence. AGPL-3.0 may not permit granting those rights.
This is disclosed to the organisers and is being resolved — see the
"Licensing" section of [`README.md`](README.md). The planned Stage II path is to
replace the pose model with a permissively licensed alternative
(e.g. `centerpose_regnetx_800mf` from the Hailo Model Zoo).

---

## Permissive components

| Component | Version | Licence | Used for |
|---|---|---|---|
| NumPy | >= 1.24 | BSD-3-Clause | Array maths throughout `src/temporal/` |
| OpenCV (`opencv-python`) | >= 4.8 | Apache-2.0 (library); MIT (Python wrapper) | Video decoding, colour conversion, rendering |
| PyYAML | >= 6.0 | MIT | Configuration loading |
| Matplotlib | >= 3.7 | Matplotlib Licence (BSD-style, PSF-derived) | Experiment plots (`scripts/plot_feature.py`) |
| psutil | — | BSD-3-Clause | Resource measurement (`scripts/benchmark.py`) |

**Note on `opencv-python` wheels:** the prebuilt wheels published on PyPI bundle
FFmpeg, parts of which are distributed under **LGPL-2.1-or-later**. Builds that
enable additional codecs may carry further terms. Verify against the exact wheel
in use before any redistribution.

---

## Not yet used (Stage II)

| Component | Licence | Status |
|---|---|---|
| HailoRT / `hailo-platform` | Hailo proprietary SDK terms | Referenced in `src/inference/hailo_backend.py`; hardware not yet received. Terms to be reviewed on receipt of the UGen300 module. |

---

## Fonts and assets

The nurse-station dashboard (`src/station/dashboard.html`) and the OpenCV
renderer use **system fonts only**. No font files are bundled or redistributed.

No third-party images, video, audio, icons, or datasets are included in this
repository. All footage used for evaluation was recorded by the team;
**no real patient footage is used at any stage.**

---

## How this list was produced

Compiled from `requirements.txt` and from the imports found in `src/` and
`scripts/`. To regenerate authoritatively from the versions actually installed:

```bash
pip install pip-licenses
pip-licenses --format=markdown \
             --with-urls --with-license-file --no-license-path \
             --order=license
```

Licence identifiers follow [SPDX](https://spdx.org/licenses/). Where this file
and a component's own licence text disagree, **the component's own licence text
governs.**

---

*Last reviewed: 2026-10-12. This notice is informational and is not legal advice.*
