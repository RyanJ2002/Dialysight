"""
特徵抽取（第 1 層，B2）。

把 17 個關鍵點壓縮成少數幾個有臨床意義的純量。

★ 不要把 17 個點全丟進時序層。34 個座標的時序統計沒有意義，
  而且會讓後面的閾值完全沒辦法調。

★ 第一階段只做 arm_displacement 和 torso_angle 兩個。
  把整條管線打通再加其他。貪多會卡在特徵工程。

關於基準線（這是最重要的設計決定）：

    arm_displacement 必須是「相對於治療開始時的基準位置」，
    不是絕對座標。

    因為每個病人的體型、躺的位置、床位角度、攝影機架設位置
    都不同。絕對座標沒有跨床、跨次治療的可比性 ——
    同一個閾值在 A 床有效，在 B 床就全錯。

    建立基準線有兩種方式（見提案討論稿待決事項 #4）：
      (a) 自動：治療開始前 N 影格的中位數
      (b) 手動：護理師在介面上按一次「建立基準」

    目前實作 (a)，用中位數而非均值，對開頭的雜訊較穩健。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..inference.base import KP, FrameObservation

NAN = float("nan")


# --------------------------------------------------------------------------
# 基準線
# --------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    mode: str = "auto"          # "auto" | "manual"
    n_frames: int = 300         # auto 模式：收集前幾影格（10 FPS = 30 秒）
    min_valid_frames: int = 50  # 至少要這麼多有效影格才算建立成功


class BaselineTracker:
    """收集治療開頭的影格，建立各特徵的基準值。

    用法：
        bt = BaselineTracker(cfg)
        for obs in stream:
            feats = extractor(obs)
            if not bt.is_ready:
                bt.observe(feats)
                continue        # 建立基準期間不做事件判定
            dev = bt.deviation(feats)
    """

    def __init__(self, cfg: Optional[BaselineConfig] = None):
        self.cfg = cfg or BaselineConfig()
        self._samples: dict[str, list[float]] = {}
        self._baseline: dict[str, float] = {}
        self._count = 0

    @property
    def is_ready(self) -> bool:
        return bool(self._baseline)

    def observe(self, features: dict[str, float]) -> None:
        """在基準建立期收集樣本。"""
        if self.is_ready:
            return

        self._count += 1
        for name, value in features.items():
            if value == value:              # 非 NaN
                self._samples.setdefault(name, []).append(value)

        if self._count >= self.cfg.n_frames:
            self._finalize()

    def _finalize(self) -> None:
        for name, values in self._samples.items():
            if len(values) >= self.cfg.min_valid_frames:
                # 用中位數，對開頭的雜訊較穩健
                self._baseline[name] = float(np.median(values))
        self._samples.clear()

    def set_manual(self, features: dict[str, float]) -> None:
        """手動建立基準（護理師按鈕時呼叫）。"""
        self._baseline = {k: v for k, v in features.items() if v == v}

    def get(self, name: str) -> Optional[float]:
        return self._baseline.get(name)

    def deviation(self, features: dict[str, float]) -> dict[str, float]:
        """回傳各特徵相對基準的偏離量。基準未建立的特徵回傳 NaN。"""
        out = {}
        for name, value in features.items():
            base = self._baseline.get(name)
            if base is None or value != value:
                out[name] = NAN
            else:
                out[name] = value - base
        return out

    def reset(self) -> None:
        self._samples.clear()
        self._baseline.clear()
        self._count = 0


# --------------------------------------------------------------------------
# 各特徵的計算
# --------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """特徵抽取參數。"""
    wrist_side: str = "right"           # 穿刺側："left" 或 "right"
    enabled: list[str] = field(default_factory=lambda: [
        "arm_displacement",
        "torso_angle",
    ])
    # 第二階段再開啟：arm_velocity, bed_occupancy, keypoint_visibility
    # Stage II 再開啟：arm_depth_delta


def _wrist_index(side: str) -> KP:
    return KP.RIGHT_WRIST if side == "right" else KP.LEFT_WRIST


def _shoulder_index(side: str) -> KP:
    return KP.RIGHT_SHOULDER if side == "right" else KP.LEFT_SHOULDER


def arm_displacement(obs: FrameObservation, cfg: FeatureConfig) -> float:
    """穿刺側手腕的位置（正規化座標下的向量長度）。

    注意這裡回傳的是「當前位置」的純量表示，
    相對基準的偏離由 BaselineTracker 計算。

    用「手腕相對同側肩膀」的向量，而非手腕的絕對座標 ——
    這樣病人整個人往下滑時不會誤判成手臂移動。
    """
    wrist = obs.kp(_wrist_index(cfg.wrist_side))
    shoulder = obs.kp(_shoulder_index(cfg.wrist_side))

    if not (wrist.is_valid and shoulder.is_valid):
        return NAN

    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    return math.hypot(dx, dy)


def torso_angle(obs: FrameObservation, cfg: FeatureConfig) -> float:
    """肩線與髖線的夾角（度）。

    用來偵測坐起、翻身。平躺時兩條線接近平行，夾角接近 0。
    """
    ls, rs = obs.kp(KP.LEFT_SHOULDER), obs.kp(KP.RIGHT_SHOULDER)
    lh, rh = obs.kp(KP.LEFT_HIP), obs.kp(KP.RIGHT_HIP)

    if not all(k.is_valid for k in (ls, rs, lh, rh)):
        return NAN

    shoulder_vec = np.array([rs.x - ls.x, rs.y - ls.y])
    hip_vec = np.array([rh.x - lh.x, rh.y - lh.y])

    n1, n2 = np.linalg.norm(shoulder_vec), np.linalg.norm(hip_vec)
    if n1 < 1e-6 or n2 < 1e-6:
        return NAN

    cos = float(np.clip(np.dot(shoulder_vec, hip_vec) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def bed_occupancy(obs: FrameObservation, cfg: FeatureConfig) -> float:
    """軀幹關鍵點的可見比例，當作「人還在床上」的粗略指標。

    TODO: Stage II 改用床面分割遮罩做精確判定。
          目前用軀幹四點的可見性近似。
    """
    torso = [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP]
    valid = sum(1 for k in torso if obs.kp(k).is_valid)
    return valid / len(torso)


def keypoint_visibility(obs: FrameObservation, cfg: FeatureConfig) -> float:
    """整格的可見關鍵點比例。

    ★ 這個特徵不是用來告警的，是用來「降低其他特徵權重」的。
      遮蔽嚴重時，arm_displacement 的可信度下降，
      fusion.py 會據此調整共識投票的權重。

      這也是我們回答評審「被子遮住怎麼辦」的技術依據。
    """
    return obs.visibility_ratio()


def arm_depth_delta(obs: FrameObservation, cfg: FeatureConfig) -> float:
    """手腕位置的深度值（Stage II，需要 scdepthv3）。

    ★ 這解掉 2D 姿態估計的一個真實盲點：
      病人手臂往鏡頭方向移動時，2D 座標幾乎不變，
      但實際上手臂已經動了。

    ⚠️ 注意：scdepthv3 的輸出是相對、正規化、無單位的深度，
       不代表真實距離。我們只用它的「變化量」，不聲稱量測距離。

    TODO: 10/26 拿到硬體後實作。
          深度圖是 320x256，要把正規化的手腕座標映射過去取值。
    """
    if obs.depth_map is None:
        return NAN

    wrist = obs.kp(_wrist_index(cfg.wrist_side))
    if not wrist.is_valid:
        return NAN

    h, w = obs.depth_map.shape[:2]
    px = int(np.clip(wrist.x * w, 0, w - 1))
    py = int(np.clip(wrist.y * h, 0, h - 1))

    # 取 3x3 鄰域中位數，避免單點雜訊
    y0, y1 = max(0, py - 1), min(h, py + 2)
    x0, x1 = max(0, px - 1), min(w, px + 2)
    return float(np.median(obs.depth_map[y0:y1, x0:x1]))


# --------------------------------------------------------------------------
# 抽取器
# --------------------------------------------------------------------------

EXTRACTORS: dict[str, Callable[[FrameObservation, FeatureConfig], float]] = {
    "arm_displacement": arm_displacement,
    "torso_angle": torso_angle,
    "bed_occupancy": bed_occupancy,
    "keypoint_visibility": keypoint_visibility,
    "arm_depth_delta": arm_depth_delta,
}


class FeatureExtractor:
    """依 config 抽取啟用的特徵。

    用法：
        ex = FeatureExtractor(cfg)
        feats = ex(obs)     # {"arm_displacement": 0.42, "torso_angle": 3.1}
    """

    def __init__(self, cfg: Optional[FeatureConfig] = None):
        self.cfg = cfg or FeatureConfig()
        unknown = set(self.cfg.enabled) - set(EXTRACTORS)
        if unknown:
            raise ValueError(f"未知的特徵名稱：{unknown}。可用：{sorted(EXTRACTORS)}")

    def __call__(self, obs: FrameObservation) -> dict[str, float]:
        return {
            name: EXTRACTORS[name](obs, self.cfg)
            for name in self.cfg.enabled
        }

    @property
    def names(self) -> list[str]:
        return list(self.cfg.enabled)


class VelocityTracker:
    """從位置特徵算一階差分，得到速度。

    獨立成一個類別是因為它需要記住上一格的值。

    用法：
        vt = VelocityTracker(["arm_displacement"])
        vel = vt.update(timestamp, feats)   # {"arm_displacement_velocity": ...}
    """

    def __init__(self, feature_names: list[str]):
        self.names = feature_names
        self._prev: dict[str, tuple[float, float]] = {}   # name -> (ts, value)

    def update(self, timestamp: float, features: dict[str, float]) -> dict[str, float]:
        out = {}
        for name in self.names:
            value = features.get(name, NAN)
            key = f"{name}_velocity"

            if value != value or name not in self._prev:
                out[key] = NAN
            else:
                prev_ts, prev_val = self._prev[name]
                dt = timestamp - prev_ts
                out[key] = (value - prev_val) / dt if dt > 1e-6 else NAN

            if value == value:
                self._prev[name] = (timestamp, value)
        return out

    def reset(self) -> None:
        self._prev.clear()
