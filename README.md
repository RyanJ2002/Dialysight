# DialySight

**Bedside temporal risk monitoring for hemodialysis — on-device, camera-based, privacy-preserving.**

ASUS UGen AI League 2026 · Battlefield Lightning · Workplace AI
Target hardware: **ASUS UGen300 M.2 AI Module 8GB** (Hailo-10H)

---

## The claim

> **A single frame cannot see gradual risk, and a single threshold cannot survive false positives.
> We solve both with a temporal layer — and a temporal layer needs continuous high-frequency sampling.
> That is why this workload needs the accelerator.**

Venous needle dislodgement during dialysis is not an instantaneous event. It is preceded by a
period of slow arm drift — and **that drift looks completely normal in any individual frame**.
Existing detection systems (Redsense, AcuSense) are contact-based and fire *after* blood has
already leaked. We operate upstream of them.

```
arm begins to drift ──→ needle loosens ──→ blood leaks ──→ major bleed
         ▲                                      ▲
    DialySight                        Redsense / AcuSense
```

**We do not replace leak detection. We intervene before it.**

---

## Why an edge accelerator rather than a central GPU

We are explicit about this: **on raw throughput, a central GPU wins.** 30 beds × 10 FPS = 300 FPS,
which a Jetson Orin Nano handles. Compute is not the argument. Deployment topology is.

| | Central GPU / Jetson | One accelerator per bed |
|---|---|---|
| Cabling | 30 runs to the nurse station — construction-grade work in a live clinical unit | One short run to the bedside module |
| Network exposure | 30 × 1080p ≈ 120 Mbps of **patient video** crossing the hospital network | **Video never leaves the bedside**; only event text is sent |
| Single point of failure | One box down = 30 beds blind | Swap one module |
| Pilot deployment | Must build the whole stack to trial 2 beds | Buy two modules |
| 30-bed power | 575 W + server + HVAC | **75 W** |
| Server room | Required | Not required |

**This is a device, not a system.** It plugs in and works — the nursing unit adopts it without
involving IT. In a hospital, anything that touches the network has a deployment cycle measured
in years.

---

## Architecture

```
camera
  │
  ▼
┌──────────────────────────────────────────────┐
│  Hailo-10H                                   │
│  per-frame inference: pose keypoints         │
│  (Stage II: + depth, + low-light enhance)    │
└──────────────────────────────────────────────┘
  │  structured observation per frame (numbers, not pixels)
  ▼
┌──────────────────────────────────────────────┐
│  Host CPU — temporal layer                   │
│   0. confidence filter                       │
│   1. feature extraction                      │
│   2. smoothing                               │
│   3. sliding-window statistics (O(1))        │
│   4. hysteresis state machine                │
│   5. multi-signal consensus                  │
└──────────────────────────────────────────────┘
  │  events
  ▼
template-based session record  ──→  nurse station (text only)
```

**The temporal layer runs entirely on the host CPU.** A 15-minute window holds ~216 KB; all
statistics use O(1) incremental updates. There is no GenAI model anywhere in the pipeline —
clinical records require auditable, reproducible, deterministic text, which a template produces
better than an LLM.

### Dual channel

| | Fast channel | Slow channel |
|---|---|---|
| Event type | Acute, visually obvious | Gradual, invisible in a single frame |
| Basis | Single frame + 1–2 s confirmation | Sliding windows + state machine |
| Latency | < 2 s | 5–30 s |
| Example | Major bleed, patient leaves bed | Arm drift, rising agitation |

The temporal layer actively *hurts* acute events — a 10-second confirmation window is a
10-second delay. Hence two channels.

---

## Quick start

```bash
git clone <repo-url> && cd dialysight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Pre-compute keypoints once (much faster iteration afterwards)
python scripts/dump_keypoints.py \
    --video data/clips/clip_03_gradual_drift.mp4 \
    --out   data/clips/clip_03.json

# 2. Run the pipeline and render the demo video
python scripts/run_demo.py \
    --video data/clips/clip_03_gradual_drift.mp4 \
    --out   results/clip_03_demo.mp4 \
    --report results/clip_03_session.md

# 3. Reproduce the false-positive comparison
python scripts/run_experiment.py \
    --videos data/clips/clip_01_baseline.mp4 data/clips/clip_02_interference.mp4 \
    --out    results/false_positive_comparison
```

No accelerator is required for any of the above. Stage I runs entirely on a laptop.

---

## Repository layout

```
src/
  inference/
    base.py                 ★ the contract: FrameObservation + InferenceBackend
    ultralytics_backend.py    development backend (laptop)
    replay_backend.py         replay from pre-dumped JSON — fastest iteration
    hailo_backend.py          UGen300 backend (implemented after hardware arrives)
  temporal/
    filter.py               layer 0 — confidence filtering
    features.py             layer 1 — keypoints → clinically meaningful scalars
    smooth.py               layer 2 — median / moving average / EWMA
    window.py               layer 3 — sliding windows, O(1) mean/std/slope
    state_machine.py        layer 4 — hysteresis state machine ★
  pipeline.py               glue — no algorithm logic lives here
  report.py                 template-based session record (no LLM)
  visualize.py              OpenCV compositing — this is the UI
config/
  default.yaml              experiment group
  baseline.yaml             control group: single threshold, no hysteresis
scripts/
  dump_keypoints.py
  run_demo.py
  run_experiment.py         ★ the false-positive comparison
data/
  annotations.md            ground truth for every clip
```

### The one interface that matters

Every module downstream of inference speaks only `FrameObservation`. Swapping the backend from
`ultralytics` to `hailo` is a one-line config change; **no downstream code changes**.

```python
@dataclass
class FrameObservation:
    frame_idx: int
    timestamp: float
    keypoints: list[Keypoint]     # always 17, missing ones have confidence=0.0
    depth_map: np.ndarray | None = None
    raw_frame: np.ndarray | None = None
```

Coordinates are normalised to 0–1 so thresholds survive a change of camera or resolution.

---

## Models

All models are **pre-compiled HEFs from the Hailo Model Zoo**. We do not train and we do not
compile. Figures below are Hailo's own measurements (Intel i5-9400, DFC v5.4.0, PCIe Gen3 x4).

| Role | Model | float | hardware | quantisation loss | FPS (B=1) | params |
|---|---|---|---|---|---|---|
| Pose | `yolov8s_pose` ⭐ | AP 59.2 | AP 56.8 | −2.4 | 161 | 11.6M |
| Low-light (Stage II) | `zero_dce` | PSNR 16.2 | PSNR 16.2 | **0** | 200 | **0.21M** |
| Depth (Stage II) | `scdepthv3` ⭐ | RMSE 0.48 | RMSE 0.48 | **0** | 145 | 14.8M |
| Bed binding (optional) | `repvgg_a0_person_reid_512` | rank1 89.9 | rank1 89.5 | −0.4 | 4580 | — |

⭐ = used by the official `hailo-apps` framework.

> **On depth:** Hailo's documentation states that depth values may be relative, normalised and
> unitless, and need not correspond to real-world distance. We use only the *relative change* in
> the depth direction — we make no distance-measurement claim. We use monocular `scdepthv3`;
> `stereonet` has reported output anomalies on Hailo-10H.

> **HEF versioning:** a HEF must match your HailoRT version (`hailortcli --version`).
> Hailo-8 HEFs do not run on Hailo-10H.

---

## Results

<!-- Populate after running scripts/run_experiment.py on real footage -->

**False-positive comparison** — same footage, two configurations:

| Clip | Single threshold | DialySight | Reduction |
|---|---|---|---|
| _pending real footage_ | | | |

Synthetic-signal sanity check (140 s, 10 FPS, one genuine 35-second drift event plus 2%-probability
noise spikes): **19 alerts → 1 alert, 94.7 % reduction, genuine event detected.**
Reproduce with `python scripts/run_experiment.py`.

---

## What we claim, and what we do not

**We claim**

1. Single-frame analysis cannot observe gradual risk — an architectural argument, not a clinical one
2. The temporal layer reduces false positives — measurable against interference footage
3. Video never leaves the bedside — verifiable from the system design
4. This workload requires continuous high-frequency sampling — derivable from the temporal resolution needed

**We do not claim**

1. **No clinical efficacy claim.** Three weeks, simulated footage, no real patients — we cannot
   produce evidence that this reduces dislodgement incidence
2. **No replacement of nursing judgement**
3. **Not a medical device.** This is prototype validation, not a certified product
4. **No replacement of leak-detection systems.** Redsense and AcuSense are better at the
   post-leak stage and already carry clearance. We operate at a different point in time
5. **Degraded performance under occlusion.** Blankets and staff passing in front reduce keypoint
   visibility. The system down-weights via `keypoint_visibility` rather than firing blindly,
   but performance does drop

---

## Data

**No real patient footage is used at any stage.** All validation uses simulated scenes recorded
in our lab with team members acting. Clinical collaborators contribute domain knowledge — event
definitions, risk criteria, threshold guidance — and no images or records.

This is deliberate: it avoids IRB scheduling risk, keeps the repository complete and shareable,
and keeps the licensing position clean.

---

## ⚠️ Licensing — unresolved

**Ultralytics YOLOv8 is AGPL-3.0.** The competition rules require winning entries to grant a
perpetual, royalty-free licence to ASUS / Intel / Hailo for commercial use, with the right to
sublicense. These may conflict.

Two paths under evaluation:
1. Confirm whether the Hailo pre-compiled HEF carries different terms
2. Switch to a more permissively licensed pose model (e.g. `centerpose_regnetx_800mf`,
   AP 43.1, 125 FPS)

**This repository's own licence is not yet selected** pending the above. Until a `LICENSE` file
is added, default copyright applies.

---

## References

- Hailo Model Zoo — https://github.com/hailo-ai/hailo_model_zoo (use the `master` branch for Hailo-10H)
- Hailo-10H pose estimation — [`HAILO10H_pose_estimation.rst`](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_pose_estimation.rst)
- HailoRT — https://github.com/hailo-ai/hailort
- `hailo-apps` — https://github.com/hailo-ai/hailo-apps
- ASUS UGen series — https://www.asus.com/motherboards-components/ai-accelerator/ugen/
- AcuSense concentric-ring blood-leakage sensor with IoMT multi-bed monitoring — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8914681/
- Venous pressure simulation for VND detection (incl. Redsense FDA clearance) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10278846/
- VND monitoring patent (source for the blanket-occlusion and roll-over descriptions) — https://patents.google.com/patent/US8348850

---

## Team notes

Code comments are written in Traditional Chinese for the team; this README is in English for
review. Development guide: [`docs/開發指南.md`](docs/開發指南.md).
