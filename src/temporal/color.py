"""
姿態導引的顏色分析 —— 偵測穿刺處滲血。

★ 核心想法：用關鍵點定位手臂，ROI 跟著手臂走。
  病人移動時 ROI 自動跟上，不會像固定 ROI 那樣失準。

★ 三層設計，每層解掉一個會讓天真做法失敗的問題：

  1. 姿態導引的動態 ROI
     前臂（肘→腕）是穿刺處所在。ROI 由關鍵點算出，跟著手臂走。
     解掉：病人移動導致固定 ROI 失準

  2. CIELab 的 a* 通道
     a* 是紅綠軸，血液讓它上升。
     不用 HSV 是因為紅色的 Hue 會在 0/180 處環繞，處理麻煩。
     不用 RGB 是因為三個通道會同時受亮度影響。
     解掉：色彩空間的糾纏

  3. ★ 用上臂當參考區，算相對值
     上臂（肩→肘）：同樣的皮膚、同樣的光線，但不會出血
     前臂（肘→腕）：穿刺處在這裡

         signal = mean_a*(前臂) − mean_a*(上臂)

     這一減，三個問題同時消失：
       - 光線變化        兩區同步變，相減抵銷
       - 相機自動白平衡  兩區同步變，相減抵銷
       - 個人膚色差異    同一個人的皮膚，基準相同
     解掉：環境干擾

★ 時序性在哪：

  |                  | 真實滲血   | 紅色被子移入 | 光線變化 |
  |------------------|-----------|------------|---------|
  | 變化形狀          | 單調累積   | 階梯式跳變  | 階梯或緩變 |
  | 會回到基準嗎      | 不會      | 會          | 會       |
  | 參考區同步變嗎    | 不會      | 不一定      | 會       |

  血不會自己消失。所以慢通道看「持續單調上升且不回頭」，
  這是被子和光線做不到的。快通道則看「大幅驟升」，對應大量出血。

⚠️ 已知限制（要寫進提案）：
  - 被子蓋住前臂時完全失效 → 由 roi_coverage 回報，權重自動降低
  - 低光下 a* 的訊噪比下降 → Stage II 的 zero_dce 可改善，但需同步重建基準
  - 不適用於非外露的滲血
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..inference.base import KP, FrameObservation, Keypoint

NAN = float("nan")


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

@dataclass
class ColorConfig:
    """顏色分析參數。"""

    wrist_side: str = "right"           # 穿刺側，與 FeatureConfig 一致

    # ROI 幾何（相對於該段肢體長度的比例）
    band_width_ratio: float = 0.55      # ROI 寬度 ÷ 前臂長度
    extend_beyond_wrist: float = 0.15   # 往手掌方向延伸，穿刺處可能靠近腕部
    extend_beyond_elbow: float = 0.05   # 往肘部外側延伸一點

    # 有效性門檻
    min_keypoint_conf: float = 0.4      # 低於此信心的關鍵點視為不可用
    min_roi_pixels: int = 200           # ROI 太小則不計算

    # 統計方式
    percentile: float = 90.0            # 取 a* 的高分位數而非均值
                                        # 理由：滲血初期只佔 ROI 一小部分，
                                        #      均值會被大面積正常皮膚稀釋

    use_reference: bool = True          # 是否用上臂做參考區正規化


@dataclass
class ColorReading:
    """單格的顏色量測結果。"""
    forearm_a: float = NAN              # 前臂 ROI 的 a* 分位數
    upperarm_a: float = NAN             # 上臂參考區的 a*
    signal: float = NAN                 # 相對訊號（前臂 − 上臂）
    roi_pixels: int = 0
    roi_coverage: float = 0.0           # ROI 有效像素比例，遮蔽時下降
    valid: bool = False


# --------------------------------------------------------------------------
# ROI 幾何
# --------------------------------------------------------------------------

def _limb_quad(
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    extend_start: float = 0.0,
    extend_end: float = 0.0,
) -> np.ndarray:
    """沿 p0→p1 這段肢體建立一個旋轉矩形的四個角點。

    Args:
        p0, p1: 肢體兩端（像素座標）
        width: 矩形寬度（像素）
        extend_start / extend_end: 往兩端延伸的長度（像素）

    Returns:
        (4, 2) int32 的角點陣列，給 cv2.fillConvexPoly 用
    """
    v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float64)
    length = np.linalg.norm(v)
    if length < 1e-6:
        return np.zeros((4, 2), dtype=np.int32)

    u = v / length                              # 單位向量（沿肢體）
    perp = np.array([-u[1], u[0]])              # 垂直向量

    a = np.array(p0, dtype=np.float64) - u * extend_start
    b = np.array(p1, dtype=np.float64) + u * extend_end
    half = perp * (width / 2.0)

    return np.array([a + half, b + half, b - half, a - half], dtype=np.int32)


def _kp_px(kp: Keypoint, w: int, h: int) -> Optional[tuple[float, float]]:
    """正規化座標 → 像素座標。缺失回 None。"""
    if not kp.is_valid:
        return None
    return (kp.x * w, kp.y * h)


# --------------------------------------------------------------------------
# 量測
# --------------------------------------------------------------------------

def measure(obs: FrameObservation, cfg: ColorConfig) -> ColorReading:
    """量測單格的前臂顏色訊號。

    需要 obs.raw_frame（BGR）。沒有影像時回傳 valid=False。
    """
    import cv2

    if obs.raw_frame is None:
        return ColorReading()

    frame = obs.raw_frame
    h, w = frame.shape[:2]

    side = cfg.wrist_side
    shoulder = obs.kp(KP.RIGHT_SHOULDER if side == "right" else KP.LEFT_SHOULDER)
    elbow = obs.kp(KP.RIGHT_ELBOW if side == "right" else KP.LEFT_ELBOW)
    wrist = obs.kp(KP.RIGHT_WRIST if side == "right" else KP.LEFT_WRIST)

    # 信心值門檻
    for kp in (elbow, wrist):
        if kp.confidence < cfg.min_keypoint_conf:
            return ColorReading()

    p_elbow = _kp_px(elbow, w, h)
    p_wrist = _kp_px(wrist, w, h)
    if p_elbow is None or p_wrist is None:
        return ColorReading()

    forearm_len = math.hypot(p_wrist[0] - p_elbow[0], p_wrist[1] - p_elbow[1])
    if forearm_len < 10.0:                      # 太短，多半是誤偵測
        return ColorReading()

    band_w = forearm_len * cfg.band_width_ratio

    # 轉 Lab：L=亮度, a=紅綠軸, b=藍黃軸
    # OpenCV 的 uint8 Lab 中 a 以 128 為中性，>128 偏紅
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1].astype(np.float32) - 128.0

    # --- 前臂 ROI ---
    quad = _limb_quad(
        p_elbow, p_wrist, band_w,
        extend_start=forearm_len * cfg.extend_beyond_elbow,
        extend_end=forearm_len * cfg.extend_beyond_wrist,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)

    quad_area = max(cv2.contourArea(quad.astype(np.float32)), 1.0)
    pixels = a_channel[mask > 0]
    n_px = int(pixels.size)

    if n_px < cfg.min_roi_pixels:
        return ColorReading(roi_pixels=n_px,
                            roi_coverage=n_px / quad_area)

    forearm_a = float(np.percentile(pixels, cfg.percentile))

    reading = ColorReading(
        forearm_a=forearm_a,
        roi_pixels=n_px,
        roi_coverage=min(1.0, n_px / quad_area),
        valid=True,
    )

    # --- 上臂參考區 ---
    if cfg.use_reference and shoulder.confidence >= cfg.min_keypoint_conf:
        p_shoulder = _kp_px(shoulder, w, h)
        if p_shoulder is not None:
            upper_len = math.hypot(p_elbow[0] - p_shoulder[0],
                                   p_elbow[1] - p_shoulder[1])
            if upper_len >= 10.0:
                ref_quad = _limb_quad(p_shoulder, p_elbow,
                                      upper_len * cfg.band_width_ratio)
                ref_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(ref_mask, ref_quad, 255)
                ref_px = a_channel[ref_mask > 0]
                if ref_px.size >= cfg.min_roi_pixels:
                    reading.upperarm_a = float(np.percentile(ref_px, cfg.percentile))

    # --- 相對訊號 ---
    if cfg.use_reference and reading.upperarm_a == reading.upperarm_a:
        reading.signal = reading.forearm_a - reading.upperarm_a
    else:
        # 沒有參考區時退回絕對值。
        # ⚠️ 這條路徑容易受光線影響，報告裡要標註比例。
        reading.signal = reading.forearm_a

    return reading


def roi_overlay(obs: FrameObservation, cfg: ColorConfig) -> Optional[np.ndarray]:
    """在影格上畫出兩個 ROI，供視覺化與標定檢查用。

    前臂 ROI 畫紅框，上臂參考區畫藍框。
    """
    import cv2

    if obs.raw_frame is None:
        return None

    frame = obs.raw_frame.copy()
    h, w = frame.shape[:2]
    side = cfg.wrist_side

    shoulder = obs.kp(KP.RIGHT_SHOULDER if side == "right" else KP.LEFT_SHOULDER)
    elbow = obs.kp(KP.RIGHT_ELBOW if side == "right" else KP.LEFT_ELBOW)
    wrist = obs.kp(KP.RIGHT_WRIST if side == "right" else KP.LEFT_WRIST)

    p_e, p_w_ = _kp_px(elbow, w, h), _kp_px(wrist, w, h)
    if p_e and p_w_:
        flen = math.hypot(p_w_[0] - p_e[0], p_w_[1] - p_e[1])
        if flen >= 10.0:
            q = _limb_quad(p_e, p_w_, flen * cfg.band_width_ratio,
                           flen * cfg.extend_beyond_elbow,
                           flen * cfg.extend_beyond_wrist)
            cv2.polylines(frame, [q], True, (60, 60, 240), 2, cv2.LINE_AA)

    p_s = _kp_px(shoulder, w, h)
    if p_s and p_e:
        ulen = math.hypot(p_e[0] - p_s[0], p_e[1] - p_s[1])
        if ulen >= 10.0:
            q = _limb_quad(p_s, p_e, ulen * cfg.band_width_ratio)
            cv2.polylines(frame, [q], True, (240, 160, 60), 2, cv2.LINE_AA)

    return frame


# --------------------------------------------------------------------------
# 時序特徵
# --------------------------------------------------------------------------

class BleedingMonitor:
    """把顏色訊號變成可供狀態機使用的特徵。

    輸出三個特徵：

        roi_redness        相對基準的 a* 上升量（主訊號）
        roi_coverage       ROI 有效像素比例，遮蔽時下降 → 用來降權重
        redness_monotonic  單調性分數 0–1，區分「累積」與「跳變」

    ★ redness_monotonic 是這個模組最有鑑別力的輸出。

      血會累積、不會自己消失 → 訊號持續上升，單調性高
      紅色被子移入再移走     → 階梯式跳變後回落，單調性低
      光線變化              → 參考區同步變，相減後幾乎沒訊號

    用法：
        mon = BleedingMonitor(cfg)
        feats = mon.update(obs)     # {"roi_redness": .., "roi_coverage": ..}
    """

    def __init__(self, cfg: Optional[ColorConfig] = None,
                 baseline_frames: int = 300,
                 monotonic_window_s: float = 60.0):
        self.cfg = cfg or ColorConfig()
        self.baseline_frames = baseline_frames
        self.monotonic_window_s = monotonic_window_s

        self._baseline: Optional[float] = None
        self._samples: list[float] = []
        self._history: list[tuple[float, float]] = []   # (ts, signal)

        self._frames = 0
        self._invalid = 0

    # ------------------------------------------------------------------

    def update(self, obs: FrameObservation) -> dict[str, float]:
        reading = measure(obs, self.cfg)
        self._frames += 1

        if not reading.valid:
            self._invalid += 1
            return {
                "roi_redness": NAN,
                "roi_coverage": reading.roi_coverage,
                "redness_monotonic": NAN,
            }

        sig = reading.signal

        # 基準建立期
        if self._baseline is None:
            self._samples.append(sig)
            if len(self._samples) >= self.baseline_frames:
                self._baseline = float(np.median(self._samples))
                self._samples.clear()
            return {
                "roi_redness": NAN,
                "roi_coverage": reading.roi_coverage,
                "redness_monotonic": NAN,
            }

        redness = sig - self._baseline       # 只看上升方向，不取絕對值
        redness = max(0.0, redness)          # 比基準還白不是異常

        self._history.append((obs.timestamp, redness))
        cutoff = obs.timestamp - self.monotonic_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.pop(0)

        return {
            "roi_redness": redness,
            "roi_coverage": reading.roi_coverage,
            "redness_monotonic": self._monotonic_score(),
        }

    def _monotonic_score(self) -> float:
        """窗內訊號的單調上升程度，0–1。

        做法：數相鄰差分為正的比例。
            純累積   → 接近 1.0
            來回震盪 → 接近 0.5
            持續下降 → 接近 0.0

        ★ 這個分數乘上 roi_redness 就是慢通道的輸入：
          「紅得夠多」且「是累積上去的」才算滲血。
        """
        if len(self._history) < 10:
            return NAN

        vals = [v for _, v in self._history]
        diffs = np.diff(vals)
        nonzero = diffs[np.abs(diffs) > 1e-6]
        if nonzero.size < 5:
            return 0.5                       # 幾乎沒變化，視為中性
        return float(np.sum(nonzero > 0) / nonzero.size)

    # ------------------------------------------------------------------

    @property
    def is_baseline_ready(self) -> bool:
        return self._baseline is not None

    def report(self) -> dict:
        n = max(self._frames, 1)
        return {
            "frames": self._frames,
            "invalid_rate": round(self._invalid / n, 4),
            "baseline_a": round(self._baseline, 3) if self._baseline else None,
            "reference_used": self.cfg.use_reference,
        }

    def reset(self) -> None:
        self._baseline = None
        self._samples.clear()
        self._history.clear()
        self._frames = self._invalid = 0
