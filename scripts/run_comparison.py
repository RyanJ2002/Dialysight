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
from src.visualize import ComparisonRenderer, DemoRenderer, LayoutConfig


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
    args = parser.parse_args()
    if not args.out and not args.show:
        parser.error("Use --out, --show, or both")

    baseline_cfg = yaml.safe_load(
        Path(args.baseline_config).read_text(encoding="utf-8"))
    ours_cfg = yaml.safe_load(
        Path(args.ours_config).read_text(encoding="utf-8"))
    fps = probe_fps(args.video)
    baseline = Pipeline(baseline_cfg, use_baseline_machine=True,
                        enable_sink=False, fps_hint=fps)
    ours = Pipeline(ours_cfg, use_baseline_machine=False,
                    enable_sink=False, fps_hint=fps)
    comparison = ComparisonRenderer(renderer_for(baseline_cfg),
                                    renderer_for(ours_cfg))
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
            if args.show:
                cv2.imshow("DialySight comparison", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        base_summary = baseline.summary()
        ours_summary = ours.summary()
        print(json.dumps({
            "baseline_alerts": base_summary["alerts"],
            "ours_alerts": ours_summary["alerts"],
            "ours_alerts_slow": ours_summary["alerts_slow"],
            "ours_alerts_fast": ours_summary["alerts_fast"],
        }, indent=2))
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        baseline.close()
        ours.close()


if __name__ == "__main__":
    main()
