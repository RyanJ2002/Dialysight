"""
重播後端：從事先 dump 好的 JSON 讀關鍵點。

★ 開發時優先用這個。不用每次重跑推論，迭代速度差一個量級。

流程：
    1. python scripts/dump_keypoints.py --video clip.mp4 --out clip.json
    2. config 的 backend 設成 replay，指向那個 JSON
    3. 之後調時序層參數時，一秒就跑完整支影片
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import FrameObservation, InferenceBackend, Keypoint


class ReplayBackend(InferenceBackend):
    def __init__(self, cfg: dict):
        path = Path(cfg["inference"]["keypoints_json"])
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        self.fps: float = payload.get("fps", 30.0)
        self.video_path = payload.get("video_path")
        self._frames: list[dict] = payload["frames"]
        self._index = {fr["frame_idx"]: fr for fr in self._frames}

    def infer(self, frame, frame_idx: int, timestamp: float) -> FrameObservation:
        rec = self._index.get(frame_idx)
        kps = (
            [Keypoint(*k) for k in rec["keypoints"]] if rec
            else [Keypoint(0.0, 0.0, 0.0) for _ in range(17)]
        )
        return FrameObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=kps,
            raw_frame=frame,
            meta={"backend": "replay"},
        )

    def stream(self, source=None):
        """不需要影片檔也能跑（純時序層測試）。

        若要同時顯示影片，傳入影片路徑即可；否則 raw_frame 為 None。
        """
        import cv2

        cap = cv2.VideoCapture(str(source)) if source else None
        try:
            for rec in self._frames:
                frame = None
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                yield FrameObservation(
                    frame_idx=rec["frame_idx"],
                    timestamp=rec["timestamp"],
                    keypoints=[Keypoint(*k) for k in rec["keypoints"]],
                    raw_frame=frame,
                    meta={"backend": "replay"},
                )
        finally:
            if cap is not None:
                cap.release()
