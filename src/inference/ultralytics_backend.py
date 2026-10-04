"""
開發期推論後端：ultralytics YOLOv8-pose，跑在筆電 CPU/GPU 上。

★ 這個後端只在 Stage I 用。10/26 拿到 UGen300 後換成 HailoBackend，
  下游程式碼一行都不用改。

⚠️ 授權提醒：
  Ultralytics YOLOv8 採 AGPL-3.0。競賽規則要求得獎作品永久無償
  授權給 ASUS / Intel / Hailo 做商業使用，兩者可能衝突。
  見提案討論稿待決事項 #1，10/7 前要有結論。
  備案：改用 centerpose_regnetx_800mf（AP 43.1、125 FPS）。
"""

from __future__ import annotations

import time

import numpy as np

from .base import NUM_KEYPOINTS, FrameObservation, InferenceBackend, Keypoint


class UltralyticsBackend(InferenceBackend):
    def __init__(self, cfg: dict):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "需要 ultralytics：pip install ultralytics"
            ) from e

        ic = cfg["inference"]
        self.model = YOLO(ic.get("model", "yolov8s-pose.pt"))
        self.device = ic.get("device", "cpu")
        self.imgsz = int(ic.get("imgsz", 640))
        self.conf = float(ic.get("conf", 0.25))

    def infer(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameObservation:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf,
            device=self.device, verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        keypoints = [Keypoint(0.0, 0.0, 0.0) for _ in range(NUM_KEYPOINTS)]

        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # 多人時取面積最大的那個 bbox，假設病人是畫面主體
            idx = 0
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 1:
                areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).tolist()
                idx = int(np.argmax(areas))

            kp = results[0].keypoints
            xy = kp.xy[idx].cpu().numpy()                       # (17, 2) 像素座標
            conf = (kp.conf[idx].cpu().numpy() if kp.conf is not None
                    else np.ones(len(xy), dtype=np.float32))

            for i in range(min(NUM_KEYPOINTS, len(xy))):
                keypoints[i] = Keypoint(
                    x=float(xy[i][0]) / w,      # 正規化到 0~1
                    y=float(xy[i][1]) / h,
                    confidence=float(conf[i]),
                )

        return FrameObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=keypoints,
            raw_frame=frame,
            meta={"backend": "ultralytics", "infer_ms": round(elapsed_ms, 2)},
        )
