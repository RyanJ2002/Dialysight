"""
平滑 / 去抖動（第 2 層，B3）。

原始特徵有大量高頻雜訊 —— 關鍵點在相鄰影格間會跳動好幾個像素，
即使人完全沒動。直接丟進狀態機會造成大量偽陽性。

選用建議：
    位置類特徵（arm_displacement, torso_angle）→ 中位數濾波
        對離群值穩健。關鍵點偶爾跳飛時，中位數不受影響，
        移動平均會被拉走。

    速度類特徵（*_velocity）→ 移動平均
        速度本身就是差分，已經放大了雜訊，
        需要的是整體趨勢而非抗離群。

    長時間常駐 → EWMA
        記憶體佔用最小（只存一個數字），不需要 buffer。

★ 窗長的取捨：
    太短 → 濾不掉雜訊
    太長 → 真事件被延遲，快通道會失效
    建議 0.5–1 秒（10 FPS 下是 5–10 影格）
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NAN = float("nan")


class Smoother(ABC):
    """平滑器介面。NaN 一律直接穿透，不嘗試補值。"""

    @abstractmethod
    def update(self, value: float) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class MedianFilter(Smoother):
    """中位數濾波。位置類特徵用這個。

    window=7 時，最多可以容忍 3 個離群值而不影響輸出。
    """

    def __init__(self, window: int = 7):
        if window < 1:
            raise ValueError("window 必須 >= 1")
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        if value != value:
            return NAN
        self._buf.append(value)
        return float(np.median(self._buf))

    def reset(self) -> None:
        self._buf.clear()


class MovingAverage(Smoother):
    """簡單移動平均。速度類特徵用這個。O(1) 增量更新。"""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window 必須 >= 1")
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum = 0.0

    def update(self, value: float) -> float:
        if value != value:
            return NAN
        if len(self._buf) == self.window:
            self._sum -= self._buf[0]
        self._buf.append(value)
        self._sum += value
        return self._sum / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
        self._sum = 0.0


class EWMA(Smoother):
    """指數加權移動平均。記憶體最小，適合長時間常駐。

        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    alpha 越小越平滑但延遲越大。
    alpha = 2 / (N + 1) 大致等效於窗長 N 的移動平均。
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha 必須在 (0, 1]")
        self.alpha = alpha
        self._state: Optional[float] = None

    @classmethod
    def from_window(cls, window: int) -> "EWMA":
        return cls(alpha=2.0 / (window + 1))

    def update(self, value: float) -> float:
        if value != value:
            return NAN
        if self._state is None:
            self._state = value
        else:
            self._state = self.alpha * value + (1 - self.alpha) * self._state
        return self._state

    def reset(self) -> None:
        self._state = None


class PassThrough(Smoother):
    """不做平滑。用於對照實驗，或某些本來就不該平滑的特徵。"""

    def update(self, value: float) -> float:
        return value

    def reset(self) -> None:
        return None


# --------------------------------------------------------------------------

SMOOTHERS = {
    "median": MedianFilter,
    "moving_average": MovingAverage,
    "ewma": EWMA,
    "none": PassThrough,
}


@dataclass
class SmoothConfig:
    """config/default.yaml 的 smooth 區塊。

        smooth:
          default: {method: median, window: 7}
          per_feature:
            arm_displacement:          {method: median, window: 7}
            arm_displacement_velocity: {method: moving_average, window: 5}
    """
    default_method: str = "median"
    default_window: int = 7
    per_feature: dict[str, dict] = field(default_factory=dict)


def make_smoother(spec: dict) -> Smoother:
    method = spec.get("method", "median")
    cls = SMOOTHERS.get(method)
    if cls is None:
        raise ValueError(f"未知的平滑方法：{method}。可用：{sorted(SMOOTHERS)}")

    if method == "ewma":
        if "alpha" in spec:
            return EWMA(alpha=float(spec["alpha"]))
        return EWMA.from_window(int(spec.get("window", 7)))
    if method == "none":
        return PassThrough()
    return cls(window=int(spec.get("window", 7)))


class SmoothingBank:
    """每個特徵各自一個平滑器。

    用法：
        bank = SmoothingBank(cfg, feature_names)
        smoothed = bank.update(raw_features)
    """

    def __init__(self, cfg: SmoothConfig, feature_names: list[str]):
        self.cfg = cfg
        self._smoothers: dict[str, Smoother] = {}
        for name in feature_names:
            spec = cfg.per_feature.get(name, {
                "method": cfg.default_method,
                "window": cfg.default_window,
            })
            self._smoothers[name] = make_smoother(spec)

    def update(self, features: dict[str, float]) -> dict[str, float]:
        out = {}
        for name, value in features.items():
            sm = self._smoothers.get(name)
            out[name] = sm.update(value) if sm is not None else value
        return out

    def reset(self) -> None:
        for sm in self._smoothers.values():
            sm.reset()
