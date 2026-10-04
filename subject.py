"""
受試者選擇 —— 在多人畫面中認出哪一個是病人。

★ 為什麼需要這一層：

  護理師走到床邊時，畫面上會有兩個人。如果取錯人，
  arm_displacement 會瞬間跳到護理師的手臂位置，
  狀態機誤判成「病人手臂大幅移動」。

  更糟的是：如果在基準建立期取錯人，整次治療的基準線都是錯的。

★ 原本的作法（取 bbox 面積最大）是錯的：
  護理師站著、離鏡頭近，框可能比躺著的病人還大。

採用的判別條件（兩個 AND）：

  1. 床位 ROI —— 髖部中心必須落在標定的床區內
     病床固定、攝影機固定，所以床在畫面上的位置永遠不變。
     一次性標定，之後極穩定。

  2. 躺姿判定 —— 肩→髖向量必須接近水平
     補 ROI 的漏洞：護理師彎腰靠近時髖部可能進入床區，
     但她的軀幹仍然接近垂直。

兩個條件都滿足的人才是病人。若有多個，取髖部中心最接近床區中心的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .base import NUM_KEYPOINTS, KP, Keypoint


@dataclass
class BedROI:
    """床位區域，正規化座標（0–1）。

    標定方式（C 拍完正式素材後做一次）：
        1. 從影片抓一格有病人躺好的畫面
        2. 用小工具（scripts/calibrate_roi.py）框出床面
        3. 把四個數字寫進 config

    ⚠️ 攝影機一動，這個標定就失效。這也是「相機固定」不可妥協的另一個理由。
    """
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 1.0
    y_max: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.x_min < self.x_max <= 1.0):
            raise ValueError(f"x 範圍不合法：{self.x_min}–{self.x_max}")
        if not (0.0 <= self.y_min < self.y_max <= 1.0):
            raise ValueError(f"y 範圍不合法：{self.y_min}–{self.y_max}")

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class SubjectConfig:
    """受試者選擇參數。"""

    roi: BedROI = None                      # None 代表不做 ROI 篩選
    require_lying: bool = True              # 是否啟用躺姿判定
    max_torso_angle_deg: float = 45.0       # 肩→髖向量與水平線的最大夾角
    min_torso_keypoints: int = 3            # 軀幹四點至少要看到幾個

    def __post_init__(self) -> None:
        if self.roi is None:
            self.roi = BedROI()              # 全畫面


# --------------------------------------------------------------------------

def _mid(a: Keypoint, b: Keypoint) -> Optional[tuple[float, float]]:
    """兩點中點。任一缺失則用另一點；都缺失回 None。"""
    if a.is_valid and b.is_valid:
        return ((a.x + b.x) / 2, (a.y + b.y) / 2)
    if a.is_valid:
        return (a.x, a.y)
    if b.is_valid:
        return (b.x, b.y)
    return None


def hip_center(kps: Sequence[Keypoint]) -> Optional[tuple[float, float]]:
    return _mid(kps[int(KP.LEFT_HIP)], kps[int(KP.RIGHT_HIP)])


def shoulder_center(kps: Sequence[Keypoint]) -> Optional[tuple[float, float]]:
    return _mid(kps[int(KP.LEFT_SHOULDER)], kps[int(KP.RIGHT_SHOULDER)])


def torso_tilt_deg(kps: Sequence[Keypoint]) -> Optional[float]:
    """肩→髖向量與水平線的夾角（度），0–90。

        躺姿：軀幹橫躺，向量接近水平 → 角度小
        站姿：軀幹直立，向量接近垂直 → 角度接近 90

    回傳 None 代表關鍵點不足以判斷。
    """
    sc = shoulder_center(kps)
    hc = hip_center(kps)
    if sc is None or hc is None:
        return None

    dx = hc[0] - sc[0]
    dy = hc[1] - sc[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    # atan2(|dy|, |dx|)：水平 → 0 度，垂直 → 90 度
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def torso_visible_count(kps: Sequence[Keypoint]) -> int:
    torso = (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP)
    return sum(1 for k in torso if kps[int(k)].is_valid)


# --------------------------------------------------------------------------

@dataclass
class Candidate:
    """一個偵測到的人，以及他通過了哪些判別條件。"""
    index: int                              # 在模型輸出裡的索引
    keypoints: list[Keypoint]
    hip: Optional[tuple[float, float]]
    tilt_deg: Optional[float]
    in_roi: bool
    is_lying: bool
    torso_visible: int

    @property
    def is_patient(self) -> bool:
        return self.in_roi and self.is_lying


def evaluate_candidates(
    people: Sequence[Sequence[Keypoint]],
    cfg: SubjectConfig,
) -> list[Candidate]:
    """對每個偵測到的人做判別，回傳帶判別結果的候選清單。"""
    out: list[Candidate] = []

    for i, kps in enumerate(people):
        hip = hip_center(kps)
        tilt = torso_tilt_deg(kps)
        visible = torso_visible_count(kps)

        in_roi = hip is not None and cfg.roi.contains(*hip)

        if not cfg.require_lying:
            is_lying = True
        elif tilt is None or visible < cfg.min_torso_keypoints:
            is_lying = False                 # 資訊不足，保守判定為非病人
        else:
            is_lying = tilt <= cfg.max_torso_angle_deg

        out.append(Candidate(
            index=i, keypoints=list(kps), hip=hip, tilt_deg=tilt,
            in_roi=in_roi, is_lying=is_lying, torso_visible=visible,
        ))

    return out


def select_patient(
    people: Sequence[Sequence[Keypoint]],
    cfg: SubjectConfig,
) -> tuple[Optional[list[Keypoint]], dict]:
    """從多人中選出病人。

    Returns:
        (病人的 17 個關鍵點, 判別摘要)
        選不出來時第一個元素是 None —— 呼叫端應回傳全缺失的 FrameObservation，
        讓下游自然產生 NaN，而不是硬選一個人。

    ★ 選不出來不是錯誤。病人被完全遮擋、或畫面上只有護理師時，
      正確行為就是「這一格沒有觀測」，不是亂猜。
    """
    cands = evaluate_candidates(people, cfg)
    passed = [c for c in cands if c.is_patient]

    info = {
        "n_detected": len(cands),
        "n_passed": len(passed),
        "rejected": [
            {"index": c.index, "in_roi": c.in_roi, "is_lying": c.is_lying,
             "tilt_deg": round(c.tilt_deg, 1) if c.tilt_deg is not None else None}
            for c in cands if not c.is_patient
        ],
    }

    if not passed:
        info["selected"] = None
        return None, info

    if len(passed) == 1:
        info["selected"] = passed[0].index
        return passed[0].keypoints, info

    # 多人都通過：取髖部中心最接近床區中心的
    cx, cy = cfg.roi.center
    best = min(passed, key=lambda c: math.hypot(c.hip[0] - cx, c.hip[1] - cy))
    info["selected"] = best.index
    info["note"] = "multiple candidates passed; chose nearest to ROI centre"
    return best.keypoints, info


# --------------------------------------------------------------------------

class SubjectSelector:
    """有狀態的選擇器，額外提供統計。

    用法（在 backend 的 infer() 裡）：
        kps, info = self.selector.select(all_people)
        if kps is None:
            kps = [Keypoint(0.0, 0.0, 0.0)] * 17      # 全缺失

    統計的用途：
      - 「選不出病人」的比例如果偏高，代表 ROI 標定有問題或攝影機角度不佳
      - 這個數字可以寫進提案，證明我們驗證過多人情境
    """

    def __init__(self, cfg: Optional[SubjectConfig] = None):
        self.cfg = cfg or SubjectConfig()
        self._frames = 0
        self._no_patient = 0
        self._multi_detected = 0
        self._rejected_not_lying = 0
        self._rejected_out_of_roi = 0

    def select(
        self, people: Sequence[Sequence[Keypoint]]
    ) -> tuple[Optional[list[Keypoint]], dict]:
        kps, info = select_patient(people, self.cfg)

        self._frames += 1
        if info["n_detected"] > 1:
            self._multi_detected += 1
        if kps is None:
            self._no_patient += 1
        for r in info["rejected"]:
            if not r["in_roi"]:
                self._rejected_out_of_roi += 1
            elif not r["is_lying"]:
                self._rejected_not_lying += 1

        return kps, info

    def report(self) -> dict:
        n = max(self._frames, 1)
        return {
            "frames": self._frames,
            "no_patient_rate": round(self._no_patient / n, 4),
            "multi_person_rate": round(self._multi_detected / n, 4),
            "rejected_out_of_roi": self._rejected_out_of_roi,
            "rejected_not_lying": self._rejected_not_lying,
        }

    def reset(self) -> None:
        self._frames = self._no_patient = self._multi_detected = 0
        self._rejected_not_lying = self._rejected_out_of_roi = 0
