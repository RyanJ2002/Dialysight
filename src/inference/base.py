"""
推論後端的抽象介面。

這個檔案是整個專案的契約。所有下游模組（特徵抽取、平滑、滑動窗、
狀態機、視覺化）都只認識 FrameObservation，不認識任何特定的推論後端。

因此：
  - 開發期用 UltralyticsBackend（筆電，不需要硬體）
  - 10/26 拿到 UGen300 後實作 HailoBackend
  - 切換只需要改 config 一行，下游程式碼一行都不用動

★ 這個檔案的介面一旦定稿就不要輕易改，改了所有人都要跟著改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# COCO 17 點關鍵點索引
# --------------------------------------------------------------------------

class KP(IntEnum):
    """COCO 格式的 17 個關鍵點索引。

    用 KP.RIGHT_WRIST 而不是魔術數字 10，讓程式碼讀得懂。
    """
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


NUM_KEYPOINTS = 17


# --------------------------------------------------------------------------
# 資料結構
# --------------------------------------------------------------------------

@dataclass
class Keypoint:
    """單一關鍵點。

    座標一律使用「正規化座標」：x, y 皆為 0.0 ~ 1.0，相對於影像寬高。

    為什麼要正規化？
      - 不同攝影機解析度不同
      - Hailo 的模型輸入是 640x640，跟原始影像解析度未必相同
      - 正規化後，閾值參數不會因為換相機就全部失效
    """
    x: float
    y: float
    confidence: float

    @property
    def is_valid(self) -> bool:
        """信心值為 0 代表這個點是缺失的（見 filter.py）。"""
        return self.confidence > 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class FrameObservation:
    """單一影格的推論結果。

    這是推論層與時序層之間唯一的資料交換格式。

    注意 keypoints 的長度永遠是 17。偵測不到的點不會被刪除，
    而是以 confidence=0.0 標記為缺失 —— 這樣下游的索引才不會錯位。
    """
    frame_idx: int
    timestamp: float                      # 秒，從影片開頭起算
    keypoints: list[Keypoint]             # 固定 17 個

    # 以下為選用欄位
    depth_map: Optional[np.ndarray] = None    # Stage II：scdepthv3 輸出，320x256
    raw_frame: Optional[np.ndarray] = None    # BGR，僅視覺化用，正式跑可設 None
    meta: dict = field(default_factory=dict)  # 後端專屬資訊（推論耗時等）

    def __post_init__(self) -> None:
        if len(self.keypoints) != NUM_KEYPOINTS:
            raise ValueError(
                f"keypoints 長度必須是 {NUM_KEYPOINTS}，收到 {len(self.keypoints)}。"
                "偵測不到的點請以 confidence=0.0 填補，不要省略。"
            )

    def kp(self, which: KP) -> Keypoint:
        """以語意索引取得關鍵點，例如 obs.kp(KP.RIGHT_WRIST)。"""
        return self.keypoints[int(which)]

    def visibility_ratio(self) -> float:
        """可見關鍵點比例，0.0 ~ 1.0。

        這是 keypoint_visibility 特徵的來源，用來在遮蔽嚴重時
        降低其他特徵的權重（見 fusion.py）。
        """
        return sum(1 for k in self.keypoints if k.is_valid) / NUM_KEYPOINTS


# --------------------------------------------------------------------------
# 後端介面
# --------------------------------------------------------------------------

class InferenceBackend(ABC):
    """所有推論後端都要實作這個介面。

    目前的實作：
      - UltralyticsBackend  (src/inference/ultralytics_backend.py)  開發期用
      - HailoBackend        (src/inference/hailo_backend.py)        10/26 後

    用法：
        backend = make_backend(cfg)
        with backend:
            for obs in backend.stream(video_path):
                ...
    """

    @abstractmethod
    def infer(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameObservation:
        """對單一影格推論。

        Args:
            frame: BGR 影像，shape (H, W, 3)，dtype uint8
            frame_idx: 影格編號，從 0 起算
            timestamp: 該影格的時間戳，單位秒

        Returns:
            FrameObservation，keypoints 長度固定 17
        """
        ...

    def stream(self, source):
        """對影片或攝影機串流逐格推論，產生 FrameObservation。

        子類別通常不需要覆寫這個方法。
        source 可以是影片檔路徑（str / Path）或攝影機編號（int）。
        """
        import cv2

        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片來源：{source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.infer(frame, idx, idx / fps)
                idx += 1
        finally:
            cap.release()

    def close(self) -> None:
        """釋放資源。HailoBackend 需要覆寫以關閉 VDevice。"""
        return None

    def __enter__(self) -> "InferenceBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# 工廠函式
# --------------------------------------------------------------------------

def make_backend(cfg: dict) -> InferenceBackend:
    """依 config 建立推論後端。

    config/default.yaml:
        inference:
          backend: ultralytics    # 10/26 之後改成 hailo
          model: yolov8s-pose.pt

    ★ 切換硬體只需要改這一行。
    """
    name = cfg["inference"]["backend"].lower()

    if name == "ultralytics":
        from .ultralytics_backend import UltralyticsBackend
        return UltralyticsBackend(cfg)

    if name == "hailo":
        from .hailo_backend import HailoBackend
        return HailoBackend(cfg)

    if name == "replay":
        # 從事先 dump 好的 JSON 讀取，開發時迭代最快
        from .replay_backend import ReplayBackend
        return ReplayBackend(cfg)

    raise ValueError(f"未知的推論後端：{name}")
