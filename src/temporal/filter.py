"""
關鍵點信心值過濾（第 0 層，B1）。

★ 這一層看起來很小，但省略它會讓整條曲線被污染。

姿態模型在遮蔽、低光、罕見姿勢下會吐出低信心的關鍵點，座標往往
是亂猜的。如果直接丟進特徵抽取，arm_displacement 會出現大幅跳動，
而那些跳動會一路傳到狀態機，變成偽陽性。

處理原則：
    低於閾值的點 → 標記為缺失（confidence=0.0）
    ★ 不是把座標設成 0！座標 (0,0) 是畫面左上角，
      會被誤判成「手臂移到畫面角落」這種巨大位移。

缺失的點由下游決定怎麼處理：
    - features.py 回傳 NaN
    - window.py 忽略 NaN，不納入統計
    - state_machine.py 遇到 NaN 不推進也不重置狀態
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..inference.base import FrameObservation, Keypoint


@dataclass
class FilterConfig:
    """過濾參數。

    min_confidence 的取值建議：
        0.3  寬鬆，保留較多點，適合遮蔽嚴重的場景
        0.5  預設
        0.7  嚴格，只信高信心的點，曲線最乾淨但缺失率高

    ★ 這個值要用實拍影片調。拍完素材後，統計各關鍵點的信心值
      分布，再決定閾值。不要憑感覺設。
    """
    min_confidence: float = 0.5

    # 遮蔽判定：可見點比例低於此值時，整格視為不可靠
    min_visibility_ratio: float = 0.4


MISSING = Keypoint(x=0.0, y=0.0, confidence=0.0)


def filter_keypoints(obs: FrameObservation, cfg: FilterConfig) -> FrameObservation:
    """過濾單一影格的關鍵點。

    回傳新的 FrameObservation，不修改原物件。

    被過濾掉的點以 confidence=0.0 標記，長度仍為 17。
    """
    filtered = [
        kp if kp.confidence >= cfg.min_confidence else MISSING
        for kp in obs.keypoints
    ]

    new_obs = replace(obs, keypoints=filtered)
    new_obs.meta = dict(obs.meta)
    new_obs.meta["visibility_ratio"] = new_obs.visibility_ratio()
    new_obs.meta["is_occluded"] = (
        new_obs.visibility_ratio() < cfg.min_visibility_ratio
    )
    return new_obs


class KeypointFilter:
    """有狀態的過濾器，額外提供缺失率統計。

    用法：
        f = KeypointFilter(cfg)
        obs = f(raw_obs)
        ...
        print(f.report())   # 哪些關鍵點最常缺失

    缺失率統計的用途：
      1. 調 min_confidence 的依據
      2. 寫進提案 —— 「穿刺側手腕的可見率是 X%」是真實數據
      3. 如果某個點缺失率過高，代表攝影機角度要調整
    """

    def __init__(self, cfg: FilterConfig | None = None):
        self.cfg = cfg or FilterConfig()
        self._total = 0
        self._missing = [0] * 17
        self._occluded_frames = 0

    def __call__(self, obs: FrameObservation) -> FrameObservation:
        out = filter_keypoints(obs, self.cfg)

        self._total += 1
        for i, kp in enumerate(out.keypoints):
            if not kp.is_valid:
                self._missing[i] += 1
        if out.meta.get("is_occluded"):
            self._occluded_frames += 1

        return out

    def missing_rate(self, index: int) -> float:
        if self._total == 0:
            return 0.0
        return self._missing[index] / self._total

    def occlusion_rate(self) -> float:
        if self._total == 0:
            return 0.0
        return self._occluded_frames / self._total

    def report(self) -> dict:
        from ..inference.base import KP
        return {
            "frames": self._total,
            "occlusion_rate": round(self.occlusion_rate(), 4),
            "missing_rate_by_keypoint": {
                KP(i).name: round(self.missing_rate(i), 4)
                for i in range(17)
            },
        }

    def reset(self) -> None:
        self._total = 0
        self._missing = [0] * 17
        self._occluded_frames = 0
