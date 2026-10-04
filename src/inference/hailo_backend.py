"""
Hailo-10H 推論後端。10/26 拿到 UGen300 M.2 8GB 後實作。

目標模型（全部使用官方預編譯 HEF，不自行編譯）：
    yolov8s_pose    AP 56.8（float 59.2）  161 FPS   640x640
    zero_dce        PSNR 16.2（量化零損失） 200 FPS   400x600   Stage II
    scdepthv3       RMSE 0.48（量化零損失） 145 FPS   256x320   Stage II

⚠️ 實作前必讀：

1. HEF 版本必須對上 HailoRT 版本
   路徑格式：.../ModelZoo/Compiled/<version>/hailo10h/<model>.hef
   用 `hailortcli --version` 查版本。Hailo-8 的 HEF 不能用。

2. 多模型併行
   HailoRT 支援在單一 VDevice 上用 round-robin scheduler 配置多個模型。
   拿到硬體第一週要實測切換開銷 —— 這是提案裡標記的最大技術未知數。

3. 輸入格式
   HEF 的輸入解析度、通道順序、normalize 在編譯時就固定了。
   前處理必須完全符合，否則輸出是垃圾。

4. 座標正規化
   ★ 一定要把輸出座標正規化回 0~1，與 UltralyticsBackend 一致。
     不然所有閾值參數都要重調。

實作骨架：

    from hailo_platform import (
        VDevice, HEF, ConfigureParams, HailoStreamInterface,
        InputVStreamParams, OutputVStreamParams, InferVStreams,
    )

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    self.device = VDevice(params)
    hef = HEF(hef_path)
    ...
"""

from __future__ import annotations

import numpy as np

from .base import FrameObservation, InferenceBackend


class HailoBackend(InferenceBackend):
    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "HailoBackend 於 10/26 收到 UGen300 後實作。"
            "Stage I 請使用 backend: ultralytics 或 replay。"
        )

    def infer(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameObservation:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
