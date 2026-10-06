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
│   4a. slow channel — hysteresis state machine│
│   4b. fast channel — amplitude + persistence │
│   5. multi-signal consensus + confidence     │
└──────────────────────────────────────────────┘
  │  events — 176 bytes each, flat scalars only
  ▼
template-based session record  ──→  nurse station (text only)
```

30 beds of event traffic is roughly **8.5 kbps**, against **120,000 kbps** for 30
streams of 1080p video — a factor of 11,415. That claim is enforced in code, not
by convention: `sink.validate_payload()` rejects any nested structure, so an
attempt to smuggle a keypoint array into an outbound payload raises rather than
quietly putting patient imagery on the network.

**The temporal layer runs entirely on the host CPU.** A 15-minute window holds ~216 KB; all
statistics use O(1) incremental updates. There is no GenAI model anywhere in the pipeline —
clinical records require auditable, reproducible, deterministic text, which a template produces
better than an LLM.

### Dual channel

| | Fast channel | Slow channel |
|---|---|---|
| Event type | Acute, visually obvious | Gradual, invisible in a single frame |
| Watches | Amplitude within a 1 s window | Deviation from baseline, over minutes |
| Confirms by | The displacement **holding** for 0.8 s | Sustained exceedance for 5 s |
| Latency | 0.8 s | ~6 s |
| Example | Patient sits up, arm yanked away | Arm drift, rising agitation |
| Reads | `feats` (absolute) | `dev` (relative to baseline) |

The temporal layer actively *hurts* acute events — a confirmation window is, by
construction, a delay. Lowering the slow channel's confirmation time to cover
them would undo the very mechanism that suppresses 94.7% of false positives.
Hence two channels over one shared inference pass.

**The hard part of the fast channel is not detecting large movements.** It is
telling a real movement apart from a pose-estimation glitch — a frame where the
model puts the wrist on a blanket fold or an IV pole. Both look identical in
amplitude. The difference shows up in the following second: a real movement
*stays*, a glitch snaps back. So the fast channel verifies persistence before
alerting, and reports how many glitches it rejected (`glitch_rejected`).

### Saying "unknown" instead of "normal"

The fusion layer returns a risk score **and** a confidence in that score. When
occlusion leaves too little signal, the state is `UNRELIABLE`, and the nurse
station shows **"無法判定"** — not a green light.

A naive consensus vote would report "no signal voted abnormal", i.e. *normal*,
which is the most dangerous failure mode a monitoring system has: it makes staff
believe someone is watching when nobody is. Our answer to "what if a blanket
covers the arm?" is therefore not "we can see through blankets" — we cannot —
but **"we know when we cannot see, and we say so."**

Reliability gating is per-signal, keyed on the keypoints each feature actually
depends on, not on the fraction of all 17 keypoints visible. A patient lying
under a blanket never shows knees or ankles; that is normal, not occlusion. So a
covered arm down-weights `arm_displacement` while `torso_angle` keeps voting.

---

## Quick start

```bash
git clone <repo-url> && cd dialysight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0. Verify everything without footage or hardware (4 seconds)
python scripts/selftest.py

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

# 4. Nurse-station overview (open http://localhost:8420)
python scripts/nurse_station.py --port 8420
```

Step 2 writes `results/events_<bed>.jsonl`. Open it: one JSON object per line,
flat scalars only — no pixels, no coordinates, nothing from which a frame could
be reconstructed. That file *is* the privacy argument.

No accelerator is required for any of the above. Stage I runs entirely on a laptop.
Nothing in the repo needs a camera either — `config/webcam.yaml` uses one if you
have it, and `scripts/selftest.py` needs neither.

---

## Repository layout

```
src/
  inference/
    base.py                 ★ the contract: FrameObservation + InferenceBackend
    subject.py              ★ pick the patient out of a multi-person frame
    ultralytics_backend.py    development backend (laptop)
    replay_backend.py         replay from pre-dumped JSON — fastest iteration
    hailo_backend.py          UGen300 backend (implemented after hardware arrives)
  temporal/
    filter.py               layer 0 — confidence filtering
    features.py             layer 1 — keypoints → clinically meaningful scalars
    color.py                layer 1 — pose-guided bleeding detection (optional)
    smooth.py               layer 2 — median / moving average / EWMA
    window.py               layer 3 — sliding windows, O(1) mean/std/slope
    state_machine.py        layer 4 — slow channel: hysteresis state machine ★
    fast_channel.py         layer 4 — fast channel: amplitude + persistence ★
  fusion.py                 layer 5 — weighted consensus + explicit confidence ★
  pipeline.py               glue — no algorithm logic lives here
  sink.py                   ★ event output; payload validation enforces
                              "no imagery leaves the bedside" in code
  report.py                 template-based session record (no LLM)
  visualize.py              OpenCV compositing — this is the bedside UI
  station/dashboard.html    nurse-station overview (zero imagery)
config/
  default.yaml              experiment group
  baseline.yaml             control group: single threshold, no hysteresis,
                            no fast channel, no fusion
  webcam.yaml               try it right now with a laptop camera
scripts/
  selftest.py               ★ 51 checks, no footage and no hardware required
  dump_keypoints.py
  plot_feature.py           feature diagnostics + three go/no-go checks
  calibrate_roi.py
  run_demo.py
  run_experiment.py         ★ the false-positive comparison
  nurse_station.py          HTTP collector + overview page
  benchmark.py
data/
  annotations.md            ground truth for every clip
```

### Verify the whole system in four seconds

```bash
python scripts/selftest.py
```

No footage, no accelerator, no model weights. It exercises the sliding windows,
both channels, the fusion layer, subject selection, bleeding discrimination,
payload validation, an end-to-end pipeline run on synthetic keypoints, the
renderer and the session record — and prints the headline numbers
(19 → 1 alerts, 176-byte payloads, 0.91 vs 0.50 monotonicity) so they are
reproducible on any machine rather than only on ours.

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

## Documentation

| Document | Audience | Content |
|---|---|---|
| [`docs/程式碼導讀.md`](docs/程式碼導讀.md) | reviewers & team | **Full code walkthrough** — architecture, data flow, every module, variable conventions |
| [`docs/開發指南.md`](docs/開發指南.md) | team | Setup, task ownership, FAQ |
| [`docs/architecture.md`](docs/architecture.md) | reviewers | System architecture diagram (text form) |

Code comments are written in Traditional Chinese for the team; this README is in English for review.
