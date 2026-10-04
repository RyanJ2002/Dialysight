#!/usr/bin/env python3
"""
把影片離線轉成關鍵點 JSON。

★ 開發時優先做這一步。之後調時序層參數時不用重跑推論，
  迭代速度差一個量級。

用法：
    python scripts/dump_keypoints.py --video data/clips/clip_03.mp4 \
                                     --out data/clips/clip_03.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.base import make_backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    frames = []
    with make_backend(cfg) as backend:
        for obs in backend.stream(args.video):
            frames.append({
                "frame_idx": obs.frame_idx,
                "timestamp": round(obs.timestamp, 4),
                "keypoints": [
                    [round(k.x, 5), round(k.y, 5), round(k.confidence, 4)]
                    for k in obs.keypoints
                ],
            })
            if obs.frame_idx % 100 == 0:
                print(f"  {obs.frame_idx} frames...", flush=True)

    Path(args.out).write_text(
        json.dumps({"video_path": args.video, "fps": fps, "frames": frames},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✓ {len(frames)} frames → {args.out}")


if __name__ == "__main__":
    main()
