#!/usr/bin/env python3
"""Render a frame-aligned, same-video comparison of two public Pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Pipeline
from src.visualize import (ComparisonRenderer, DemoRenderer, LayoutConfig,
                           PreviewWindow)


def probe_fps(video: str) -> float:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps) if 1.0 <= fps <= 240.0 else 30.0


def renderer_for(cfg: dict) -> DemoRenderer:
    layout = LayoutConfig(**{
        key: value for key, value in cfg.get("visualize", {}).items()
        if key in LayoutConfig.__dataclass_fields__
    })
    return DemoRenderer(
        layout=layout,
        thresholds={
            "enter": cfg["state_machine"]["enter_threshold"],
            "exit": cfg["state_machine"]["exit_threshold"],
        },
        primary_feature=cfg["features"]["enabled"][0],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--baseline-config", default="config/baseline.yaml")
    parser.add_argument("--ours-config", default="config/default.yaml")
    parser.add_argument("--out", help="Output MP4 path")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--keypoints", help=(
        "Replay both pipelines from one keypoint dump "
        "(scripts/dump_keypoints.py). Strongly preferred: without it this "
        "script runs pose inference twice over the clip, which is slow and "
        "lets the two sides see different keypoints -- the comparison is "
        "about the temporal layer, so both sides must get identical input."))
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale the 2560px-wide output, e.g. 0.75 -> 1920")
    args = parser.parse_args()
    if not args.out and not args.show:
        parser.error("Use --out, --show, or both")

    baseline_cfg = yaml.safe_load(
        Path(args.baseline_config).read_text(encoding="utf-8"))
    ours_cfg = yaml.safe_load(
        Path(args.ours_config).read_text(encoding="utf-8"))

    if args.keypoints:
        if not Path(args.keypoints).is_file():
            raise SystemExit(f"Keypoint dump not found: {args.keypoints}")
        for cfg in (baseline_cfg, ours_cfg):
            cfg["inference"] = {"backend": "replay",
                                "keypoints_json": args.keypoints}
        print(f"Replaying both sides from {args.keypoints}", flush=True)

    fps = probe_fps(args.video)
    baseline = Pipeline(baseline_cfg, use_baseline_machine=True,
                        enable_sink=False, fps_hint=fps)
    ours = Pipeline(ours_cfg, use_baseline_machine=False,
                    enable_sink=False, fps_hint=fps)
    comparison = ComparisonRenderer(renderer_for(baseline_cfg),
                                    renderer_for(ours_cfg),
                                    scale=args.scale)
    print(f"Output {comparison.width}x{comparison.height} @ {fps:.2f} fps",
          flush=True)
    preview = (PreviewWindow("DialySight comparison",
                             comparison.width, comparison.height)
               if args.show else None)
    writer = None
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (comparison.width, comparison.height))
        if not writer.isOpened():
            baseline.close()
            ours.close()
            raise RuntimeError(f"Cannot write video: {out}")

    try:
        for base_result, ours_result in zip_longest(
                baseline.run(args.video), ours.run(args.video)):
            if base_result is None or ours_result is None:
                raise RuntimeError("Pipeline streams have different frame counts")
            if base_result.obs.frame_idx != ours_result.obs.frame_idx:
                raise RuntimeError("Pipeline frame indices are not aligned")
            if abs(base_result.obs.timestamp - ours_result.obs.timestamp) > 1e-3:
                raise RuntimeError("Pipeline timestamps are not aligned")
            frame = comparison.render(base_result, ours_result)
            if writer is not None:
                writer.write(frame)
            if preview is not None:
                preview.show(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        base_summary = baseline.summary()
        ours_summary = ours.summary()
        from_summary = {
            "baseline_alerts": base_summary["alerts"],
            "ours_alerts": ours_summary["alerts"],
            "ours_alerts_slow": ours_summary["alerts_slow"],
            "ours_alerts_fast": ours_summary["alerts_fast"],
        }
        print(json.dumps(from_summary, indent=2))

        # The counters burnt into the video and the pipeline summaries are
        # computed independently, so a mismatch means the video shows numbers
        # the experiment does not support. Catch that here, not in review.
        on_screen = comparison.tally()
        if on_screen != from_summary:
            print("\nWARNING: on-screen counters disagree with the pipeline "
                  "summaries -- do not use this video.")
            print(f"  on screen: {on_screen}")
            print(f"  summaries: {from_summary}")
        else:
            reduction = (100.0 * (from_summary["baseline_alerts"]
                                  - from_summary["ours_alerts"])
                         / from_summary["baseline_alerts"]
                         if from_summary["baseline_alerts"] else 0.0)
            print(f"\nOn-screen counters match the summaries. "
                  f"Reduction {reduction:.1f}%.")
    finally:
        if writer is not None:
            writer.release()
        if preview is not None:
            preview.close()
        baseline.close()
        ours.close()


if __name__ == "__main__":
    main()
