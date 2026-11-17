"""
Demo 視覺化（B6）—— 這就是這個專案的 UI。

★ 不要做網頁介面。Stage I 需要的是「影片裡看得到東西在跑」，
  不是產品級的護理站看板。網頁 UI 做了不會加分，而且會吃掉
  做對照實驗的時間。

畫面佈局（這是三分鐘影片的主體）：

    ┌─────────────────────┬──────────────────────┐
    │                     │  arm_displacement    │
    │   影片 + 骨架疊圖    │  ～～～～／‾‾‾‾‾‾   │← 曲線即時右捲
    │                     │  ---- enter 閾值 ----│
    │   [第 7 床]         │                      │
    │                     │  狀態: WATCHING      │
    ├─────────────────────┴──────────────────────┤
    │  14:32  上肢姿態異常（持續 18 秒）          │
    │  14:47  深度方向位移異常                    │← 事件即時跳出
    └────────────────────────────────────────────┘

三個東西必須同時在畫面上：影片、特徵曲線、狀態。
觀眾要能看到「畫面上人幾乎沒動，但曲線在緩慢爬升」——
那一刻我們的核心主張就成立了。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .inference.base import KP, FrameObservation
from .temporal.state_machine import Event, State


# COCO 骨架連線。畫骨架用。
SKELETON: list[tuple[int, int]] = [
    (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER),
    (KP.LEFT_SHOULDER, KP.LEFT_ELBOW),
    (KP.LEFT_ELBOW, KP.LEFT_WRIST),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW),
    (KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
    (KP.LEFT_SHOULDER, KP.LEFT_HIP),
    (KP.RIGHT_SHOULDER, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.LEFT_KNEE),
    (KP.LEFT_KNEE, KP.LEFT_ANKLE),
    (KP.RIGHT_HIP, KP.RIGHT_KNEE),
    (KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
]

# BGR 配色。深色底，狀態用顏色區分。
COLORS = {
    "bg":        (24, 24, 28),
    "panel":     (38, 38, 44),
    "grid":      (60, 60, 68),
    "text":      (235, 235, 240),
    "text_dim":  (150, 150, 160),
    "skeleton":  (180, 220, 120),
    "keypoint":  (120, 200, 255),
    "curve":     (120, 200, 255),
    "threshold": (90, 90, 180),
    "redness":   (80, 80, 230),      # 前臂 ROI：紅
    "reference": (230, 170, 80),     # 上臂參考區：藍
    State.NORMAL:   (120, 200, 120),
    State.WATCHING: (60, 190, 240),
    State.ALERT:    (60, 60, 240),
    State.COOLDOWN: (160, 160, 160),
}

STATE_LABEL = {
    State.NORMAL: "NORMAL",
    State.WATCHING: "WATCHING",
    State.ALERT: "ALERT",
    State.COOLDOWN: "COOLDOWN",
}


@dataclass
class LayoutConfig:
    """畫面尺寸。1280x720 輸出，上傳 YouTube 剛好。"""
    width: int = 1280
    height: int = 720
    video_w: int = 720          # 左側影片區寬度
    curve_h: int = 440          # 右側曲線區高度
    event_h: int = 160          # 下方事件列高度
    curve_span_s: float = 90.0  # 曲線顯示最近幾秒
    bed_label: str = "Bed 07"

    # 滲血偵測的視覺化（config 的 color.enabled 開啟時才有意義）
    show_color_panel: bool = False   # 右側多一條 roi_redness 曲線
    show_roi_overlay: bool = False   # 影片上畫出前臂 ROI 與上臂參考區


class DemoRenderer:
    """把影格、特徵曲線、狀態合成為一張畫面。

    用法：
        renderer = DemoRenderer(layout, thresholds={"enter": 0.15, "exit": 0.10})
        canvas = renderer.render(obs, features, state, events, timestamp)
        writer.write(canvas)

    輸出的 canvas 是 BGR，可以直接丟給 cv2.VideoWriter 或 cv2.imshow。
    """

    def __init__(
        self,
        layout: Optional[LayoutConfig] = None,
        thresholds: Optional[dict[str, float]] = None,
        primary_feature: str = "arm_displacement",
    ):
        self.L = layout or LayoutConfig()
        self.thresholds = thresholds or {"enter": 0.15, "exit": 0.10}
        self.primary = primary_feature

        # 曲線歷史。deque 自動淘汰，長度抓 curve_span × 60fps 保險
        self._history: deque[tuple[float, float]] = deque(
            maxlen=int(self.L.curve_span_s * 60)
        )
        # 滲血訊號的歷史（第二條曲線）
        self._redness: deque[tuple[float, float]] = deque(
            maxlen=int(self.L.curve_span_s * 60)
        )
        self._monotonic: float = float("nan")
        self._event_log: deque[str] = deque(maxlen=4)

    # ------------------------------------------------------------------

    def render(
        self,
        obs: FrameObservation,
        features: dict[str, float],
        state: State,
        new_event: Optional[Event] = None,
        timestamp: Optional[float] = None,
        color_cfg=None,
        baseline_progress: Optional[tuple[float, float]] = None,   # (已過秒, 需要秒)
    ) -> np.ndarray:
        """合成單一影格。

        Args:
            color_cfg: ColorConfig。傳入且 show_roi_overlay 為 True 時，
                       會在影片上畫出前臂 ROI（紅）與上臂參考區（藍）。
        """
        ts = timestamp if timestamp is not None else obs.timestamp

        value = features.get(self.primary, float("nan"))
        if value == value:                       # 非 NaN
            self._history.append((ts, value))

        red = features.get("roi_redness", float("nan"))
        if red == red:
            self._redness.append((ts, red))
        mono = features.get("redness_monotonic", float("nan"))
        if mono == mono:
            self._monotonic = mono

        if new_event is not None:
            self._event_log.append(self._format_event(new_event))

        canvas = np.full((self.L.height, self.L.width, 3), COLORS["bg"], np.uint8)

        self._draw_video(canvas, obs, state, color_cfg)

        building = (baseline_progress is not None
                    and baseline_progress[1] > 0
                    and baseline_progress[0] < baseline_progress[1])
        if building:
            # ★ 基準建立期：右側顯示進度，不要留一片空白
            self._draw_baseline_progress(canvas, *baseline_progress)
        else:
            self._draw_curve(canvas, ts, state)
            self._draw_state_badge(canvas, state, value)

        self._draw_events(canvas)
        return canvas

    def _draw_baseline_progress(self, canvas: np.ndarray,
                                done: float, need: float) -> None:
        """基準線建立中的提示畫面。

        沒有這個的話右側會是一片空白，觀眾（和評審）不知道系統在幹嘛。

        單位是秒。影格數會隨 FPS 變動，秒數不會 ——
        demo 時觀眾看到「12.4 / 30.0 s」知道還要等多久，
        看到「124 / 300 frames」只會一頭霧水。
        """
        x0 = self.L.video_w + 16
        x1 = self.L.width - 16
        y0 = 56
        y1 = self.L.height - self.L.event_h - 8

        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS["panel"], -1)

        cy = (y0 + y1) // 2
        self._text(canvas, "Establishing baseline", (x0 + 28, cy - 46),
                   COLORS["text"], scale=0.8, thickness=2)
        self._text(canvas, "per-session reference for this patient",
                   (x0 + 28, cy - 20), COLORS["text_dim"], scale=0.5)

        # 進度條
        bw = (x1 - x0) - 56
        bx, by = x0 + 28, cy + 8
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + 14), COLORS["grid"], -1)
        frac = min(1.0, done / max(need, 1e-6))
        cv2.rectangle(canvas, (bx, by), (bx + int(bw * frac), by + 14),
                      COLORS[State.WATCHING], -1)

        self._text(canvas, f"{done:.1f} / {need:.1f} s", (bx, by + 40),
                   COLORS["text"], scale=0.6)
        self._text(canvas, f"{frac * 100:.0f}%", (x1 - 76, by + 40),
                   COLORS[State.WATCHING], scale=0.6)

    # -- 左側：影片 + 骨架 ----------------------------------------------

    def _draw_video(self, canvas: np.ndarray, obs: FrameObservation,
                    state: State, color_cfg=None) -> None:
        h = self.L.height - self.L.event_h
        w = self.L.video_w

        if obs.raw_frame is None:
            cv2.rectangle(canvas, (0, 0), (w, h), COLORS["panel"], -1)
            self._text(canvas, "no frame", (20, 40), COLORS["text_dim"])
            return

        src = obs.raw_frame
        # 滲血偵測的 ROI 疊圖：前臂紅框、上臂參考區藍框
        if self.L.show_roi_overlay and color_cfg is not None:
            from .temporal.color import roi_overlay
            overlaid = roi_overlay(obs, color_cfg)
            if overlaid is not None:
                src = overlaid

        # ★ 保持長寬比，置中貼上。直接 resize 會把直立構圖拉扁
        sh, sw = src.shape[:2]
        scale = min(w / sw, h / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        fitted = cv2.resize(src, (nw, nh))

        frame = np.full((h, w, 3), COLORS["bg"], np.uint8)
        ox, oy = (w - nw) // 2, (h - nh) // 2
        frame[oy:oy + nh, ox:ox + nw] = fitted

        self._draw_skeleton(frame, obs, (ox, oy, nw, nh))

        # 狀態色的外框，讓觀眾一眼看到當前狀態
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLORS[state], 4)

        canvas[0:h, 0:w] = frame
        self._text(canvas, self.L.bed_label, (18, 36), COLORS["text"], scale=0.8)
        self._text(canvas, f"t = {obs.timestamp:6.1f}s", (18, h - 18),
                   COLORS["text_dim"], scale=0.55)

    def _draw_skeleton(self, frame: np.ndarray, obs: FrameObservation,
                       box: Optional[tuple[int, int, int, int]] = None) -> None:
        """box = (ox, oy, nw, nh)：影片在畫布上的實際位置與尺寸。

        正規化座標要映射到這個範圍，而不是整張畫布 ——
        否則保持長寬比之後骨架會偏移。
        """
        if box is None:
            h, w = frame.shape[:2]
            ox = oy = 0
        else:
            ox, oy, w, h = box

        def px(kp) -> tuple[int, int]:
            return int(ox + kp.x * w), int(oy + kp.y * h)

        for a, b in SKELETON:
            ka, kb = obs.keypoints[int(a)], obs.keypoints[int(b)]
            if ka.is_valid and kb.is_valid:
                # 先畫深色外框再畫亮色線，確保在明亮背景上也看得清楚
                cv2.line(frame, px(ka), px(kb), (20, 20, 24), 5, cv2.LINE_AA)
                cv2.line(frame, px(ka), px(kb), COLORS["skeleton"], 2, cv2.LINE_AA)

        for kp in obs.keypoints:
            if kp.is_valid:
                cv2.circle(frame, px(kp), 4, COLORS["keypoint"], -1, cv2.LINE_AA)

        # 穿刺側手腕畫大一點 —— 那是我們真正在追蹤的點
        wrist = obs.kp(KP.RIGHT_WRIST)
        if wrist.is_valid:
            cv2.circle(frame, px(wrist), 10, (60, 190, 240), 2, cv2.LINE_AA)

    # -- 右側：特徵曲線 --------------------------------------------------

    def _draw_curve(self, canvas: np.ndarray, now: float, state: State) -> None:
        """主訊號曲線。開啟 show_color_panel 時上半畫主訊號、下半畫滲血訊號。"""
        if self.L.show_color_panel:
            gap = 14
            h_each = (self.L.curve_h - gap) // 2
            self._draw_one_curve(canvas, now, state, self._history, 56, h_each,
                                 self.primary, COLORS["curve"], self.thresholds)
            self._draw_one_curve(canvas, now, state, self._redness,
                                 56 + h_each + gap, h_each,
                                 "roi_redness", COLORS["redness"], None,
                                 badge=self._monotonic)
            return

        self._draw_one_curve(canvas, now, state, self._history, 56,
                             self.L.curve_h, self.primary,
                             COLORS["curve"], self.thresholds)

    def _draw_one_curve(self, canvas, now, state, history, top, height,
                        label, color, thresholds, badge=None) -> None:
        x0 = self.L.video_w + 16
        y0 = top
        x1 = self.L.width - 16
        y1 = y0 + height

        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS["panel"], -1)
        self._text(canvas, label, (x0, y0 - 14), COLORS["text"], scale=0.6)

        # 單調性分數：滲血曲線才有。>0.8 代表「持續累積」而非「跳變」
        if badge is not None and badge == badge:
            tag = f"monotonic {badge:.2f}"
            col = COLORS["redness"] if badge > 0.8 else COLORS["text_dim"]
            self._text(canvas, tag, (x1 - 150, y0 - 14), col, scale=0.5)

        if len(history) < 2:
            return

        floor = thresholds["enter"] if thresholds else 1e-6
        vmax = max(max(v for _, v in history), floor) * 1.25
        vmax = max(vmax, 1e-6)

        def to_px(t: float, v: float) -> tuple[int, int]:
            fx = (t - (now - self.L.curve_span_s)) / self.L.curve_span_s
            fx = min(max(fx, 0.0), 1.0)
            return (
                int(x0 + fx * (x1 - x0)),
                int(y1 - min(v / vmax, 1.0) * (y1 - y0)),
            )

        # 閾值線（滲血曲線沒有閾值，跳過）
        if thresholds:
            for name, c in (("enter", COLORS["threshold"]), ("exit", COLORS["grid"])):
                yv = to_px(now, thresholds[name])[1]
                cv2.line(canvas, (x0, yv), (x1, yv), c, 1, cv2.LINE_AA)
                self._text(canvas, name, (x1 - 54, yv - 6), c, scale=0.42)

        # 曲線本體
        pts = [to_px(t, v) for t, v in history if t >= now - self.L.curve_span_s]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False,
                          color, 2, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 5,
                       COLORS[state] if thresholds else color, -1, cv2.LINE_AA)

    # -- 右下：狀態徽章 --------------------------------------------------

    def _draw_state_badge(self, canvas: np.ndarray, state: State, value: float) -> None:
        # 從事件列往上貼齊，避免 curve_h 改動時溢出
        x0 = self.L.video_w + 16
        x1 = self.L.width - 16
        y1 = self.L.height - self.L.event_h - 8
        y0 = y1 - 64

        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS["panel"], -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + 8, y1), COLORS[state], -1)

        self._text(canvas, STATE_LABEL[state], (x0 + 22, y0 + 28),
                   COLORS[state], scale=0.85, thickness=2)
        shown = f"{value:.4f}" if value == value else "--"
        self._text(canvas, f"{self.primary} = {shown}", (x0 + 22, y0 + 52),
                   COLORS["text_dim"], scale=0.5)

    # -- 下方：事件列 ----------------------------------------------------

    def _draw_events(self, canvas: np.ndarray) -> None:
        y0 = self.L.height - self.L.event_h
        cv2.rectangle(canvas, (0, y0), (self.L.width, self.L.height),
                      COLORS["panel"], -1)
        self._text(canvas, "EVENTS", (18, y0 + 26), COLORS["text_dim"], scale=0.5)

        if not self._event_log:
            self._text(canvas, "no events yet", (18, y0 + 56),
                       COLORS["text_dim"], scale=0.55)
            return

        for i, line in enumerate(self._event_log):
            self._text(canvas, line, (18, y0 + 54 + i * 26),
                       COLORS["text"], scale=0.55)

    # -- 小工具 ----------------------------------------------------------

    @staticmethod
    def _format_event(ev: Event) -> str:
        mm, ss = divmod(int(ev.start_time), 60)
        return f"{mm:02d}:{ss:02d}  {ev.feature}  (peak {ev.peak_value:.3f})"

    @staticmethod
    def _text(img, s, org, color, scale=0.6, thickness=1) -> None:
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)


class ComparisonRenderer:
    """對照實驗用：左右並排顯示單閾值版 vs 時序版。

    ★ 這是三分鐘影片 0:20–1:00 那一段的畫面，
      也是整支影片最有說服力的地方。

    左邊的告警數字會一路飆高，右邊維持個位數。

    用法（見 scripts/run_comparison.py）：
        cmp = ComparisonRenderer(renderer_for(baseline_cfg),
                                 renderer_for(ours_cfg))
        writer = cv2.VideoWriter(..., (cmp.width, cmp.height))
        canvas = cmp.render(baseline_result, ours_result)

    ⚠️ 右邊的計數含快通道。
       現在系統有兩條通道，只數慢通道會讓數字好看，但那不是護理師
       實際會收到的量。run_experiment.py 的降幅也是用總和算的 ——
       兩邊要一致，不然簡報上的數字會對不起來。

       所以這裡每格累加 new_event 與 fast_event **各自**計數，
       不是用 any_event：兩條通道同一格都觸發時 any_event 只給一個，
       會少算。跑完的總數與 pipeline.summary()["alerts"] 一致。

    ⚠️ 文字一律用 ASCII。cv2.putText 畫不出中文，會變成一排問號，
       而且不會報錯 —— 存成影片才發現就來不及了。
    """

    BAND_H = 84                      # 上方計數列高度（縮放前）
    FLASH_FRAMES = 10                # 新警報觸發後閃幾格

    def __init__(self, baseline_renderer: DemoRenderer, ours_renderer: DemoRenderer,
                 scale: float = 1.0):
        self.baseline = baseline_renderer
        self.ours = ours_renderer
        self.scale = float(scale)

        bl, ol = baseline_renderer.L, ours_renderer.L
        if (bl.width, bl.height) != (ol.width, ol.height):
            raise ValueError(
                f"兩邊的畫面尺寸必須一樣才能並排："
                f"{bl.width}x{bl.height} vs {ol.width}x{ol.height}。"
                "把兩份 config 的 visualize.width/height 改成相同。")

        self._panel_w, self._panel_h = bl.width, bl.height
        self._raw_w = self._panel_w * 2
        self._raw_h = self._panel_h + self.BAND_H
        self.width = int(round(self._raw_w * self.scale))
        self.height = int(round(self._raw_h * self.scale))

        self.n_base = 0
        self.n_ours = 0
        self.n_ours_slow = 0
        self.n_ours_fast = 0
        self._flash_base = 0
        self._flash_ours = 0
        self._ts = 0.0

    # -- 計數 ------------------------------------------------------------

    @staticmethod
    def _fired(result) -> int:
        """這一格觸發了幾個事件。兩條通道同格都觸發時要算 2。"""
        return ((1 if getattr(result, "new_event", None) else 0)
                + (1 if getattr(result, "fast_event", None) else 0))

    # -- 繪製 ------------------------------------------------------------

    def render(self, baseline_result, ours_result) -> np.ndarray:
        """合成一格並排畫面。兩個 result 都是 pipeline.FrameResult。"""
        self._ts = ours_result.obs.timestamp

        nb = self._fired(baseline_result)
        if nb:
            self.n_base += nb
            self._flash_base = self.FLASH_FRAMES
        if getattr(ours_result, "new_event", None):
            self.n_ours += 1
            self.n_ours_slow += 1
            self._flash_ours = self.FLASH_FRAMES
        if getattr(ours_result, "fast_event", None):
            self.n_ours += 1
            self.n_ours_fast += 1
            self._flash_ours = self.FLASH_FRAMES

        left = self._panel(self.baseline, baseline_result)
        right = self._panel(self.ours, ours_result)

        canvas = np.empty((self._raw_h, self._raw_w, 3), dtype=np.uint8)
        canvas[: self.BAND_H] = COLORS["bg"]
        canvas[self.BAND_H:, : self._panel_w] = left
        canvas[self.BAND_H:, self._panel_w:] = right

        self._band(canvas)

        # 中間分隔線，讓兩邊一眼分得開
        cv2.line(canvas, (self._panel_w, 0), (self._panel_w, self._raw_h),
                 COLORS["grid"], 2)

        self._flash_base = max(0, self._flash_base - 1)
        self._flash_ours = max(0, self._flash_ours - 1)

        if self.scale != 1.0:
            canvas = cv2.resize(canvas, (self.width, self.height),
                                interpolation=cv2.INTER_AREA)
        return canvas

    def _panel(self, renderer: DemoRenderer, result) -> np.ndarray:
        """呼叫既有的 DemoRenderer。display_features 是偏離量，與閾值線同量綱
        —— 跟 run_demo.py 一致，不要換成 features，曲線會對不上閾值。"""
        return renderer.render(
            result.obs,
            getattr(result, "display_features", None) or result.features,
            result.state,
            result.any_event,
            result.obs.timestamp,
            baseline_progress=getattr(result, "baseline_progress", None),
        )

    def _band(self, canvas: np.ndarray) -> None:
        seam = self._panel_w

        # 降幅先畫，跨在分隔線上 —— 這是整段影片要觀眾記住的數字。
        # 先畫是為了拿到它的右緣，右半邊的文字要從那之後開始，
        # 否則數字會疊在 "DIALYSIGHT / ALERTS n" 上面。
        badge_right = seam
        if self.n_base > 0:
            cut = (self.n_base - self.n_ours) / self.n_base * 100.0
            txt = f"-{cut:.1f}%"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            x = seam - tw // 2
            cv2.rectangle(canvas, (x - 20, 12), (x + tw + 20, 72),
                          COLORS["panel"], -1)
            DemoRenderer._text(canvas, txt, (x, 57),
                               COLORS[State.NORMAL], scale=1.1, thickness=2)
            badge_right = x + tw + 20

        self._side(canvas, 0, "SINGLE THRESHOLD", self.n_base,
                   None, self._flash_base, 24)
        self._side(canvas, seam, "DIALYSIGHT", self.n_ours,
                   f"slow {self.n_ours_slow} / fast {self.n_ours_fast}",
                   self._flash_ours, max(24, badge_right - seam + 24))

        mm, ss = divmod(int(self._ts), 60)
        DemoRenderer._text(canvas, f"{mm:02d}:{ss:02d}",
                           (self._raw_w - 96, self.BAND_H - 14),
                           COLORS["text_dim"], scale=0.6)

    def _side(self, canvas, x0: int, label: str, count: int,
              sub: Optional[str], flash: int, pad: int) -> None:
        colour = COLORS[State.ALERT] if flash else COLORS["text"]
        count_txt = f"ALERTS {count}"
        (cw, _), _ = cv2.getTextSize(count_txt, cv2.FONT_HERSHEY_SIMPLEX,
                                     1.0, 2)
        if flash:
            cv2.rectangle(canvas, (x0 + pad - 10, 10),
                          (x0 + pad + cw + 10, self.BAND_H - 10),
                          COLORS["panel"], -1)
        DemoRenderer._text(canvas, label, (x0 + pad, 34),
                           COLORS["text_dim"], scale=0.56)
        DemoRenderer._text(canvas, count_txt, (x0 + pad, 70),
                           colour, scale=1.0, thickness=2)
        if sub:
            DemoRenderer._text(canvas, sub, (x0 + pad + cw + 24, 70),
                               COLORS["text_dim"], scale=0.6)

    # -- 跑完之後給 run_comparison.py 對帳用 -------------------------------

    def tally(self) -> dict:
        """渲染過程累加的計數。應與 pipeline.summary()['alerts'] 一致。"""
        return {"baseline_alerts": self.n_base, "ours_alerts": self.n_ours,
                "ours_alerts_slow": self.n_ours_slow,
                "ours_alerts_fast": self.n_ours_fast}


class PreviewWindow:
    """即時預覽視窗。--show 用。

    並排畫面是 2560 px 寬，大多數螢幕放不下，所以用 WINDOW_NORMAL
    加 resizeWindow 縮到看得完的大小。縮的是視窗，不是寫進檔案的影格。
    """

    def __init__(self, title: str, width: int, height: int,
                 max_width: int = 1600):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        k = min(1.0, max_width / max(1, width))
        cv2.resizeWindow(title, int(width * k), int(height * k))

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.title, frame)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass


def make_writer(path: str, fps: float, layout: LayoutConfig) -> "cv2.VideoWriter":
    """建立影片輸出器。mp4v 在大多數平台都能用。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (layout.width, layout.height))
 
