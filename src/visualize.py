"""OpenCV views for the bedside demo and the same-video comparison."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # English remains usable in lean deployments.
    Image = ImageDraw = ImageFont = None

from .fusion import FusionResult, Reliability
from .inference.base import KP, FrameObservation
from .temporal.state_machine import Event, State

if TYPE_CHECKING:
    from .pipeline import FrameResult

SKELETON = [
    (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER),
    (KP.LEFT_SHOULDER, KP.LEFT_ELBOW), (KP.LEFT_ELBOW, KP.LEFT_WRIST),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW), (KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
    (KP.LEFT_SHOULDER, KP.LEFT_HIP), (KP.RIGHT_SHOULDER, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.RIGHT_HIP), (KP.LEFT_HIP, KP.LEFT_KNEE),
    (KP.LEFT_KNEE, KP.LEFT_ANKLE), (KP.RIGHT_HIP, KP.RIGHT_KNEE),
    (KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
]

# BGR colors. Lime is a brand accent; alert remains red.
C = {
    "bg": (246, 246, 242), "paper": (255, 255, 255),
    "navy": (54, 43, 29), "navy_soft": (78, 64, 47),
    "line": (225, 221, 214), "text": (68, 59, 46),
    "muted": (143, 132, 116), "white": (255, 255, 255),
    "lime": (126, 216, 191), "curve": (165, 121, 52),
    "skeleton": (137, 232, 222), "keypoint": (166, 245, 234),
    "stable": (117, 140, 29), "watch": (42, 156, 216),
    "alert": (69, 69, 216), "limited": (144, 125, 113),
    "warm_tint": (242, 249, 255),
}

# English source strings serve as keys; all display translations live here.
TEXT = {
    "en": {
        "treatment_clock": "TREATMENT {clock}",
        "trend_span": "{feature}  |  last {span:g} sec",
        "now_value": "NOW  {value:.3f}",
        "events_count": "Events this session: {count}",
        "event_time": "T+{time}",
        "progress": "{done:.1f} / {need:.1f} sec",
    },
    "zh": {
        "CURRENT STATUS": "目前狀態",
        "STABLE": "姿態穩定",
        "No persistent posture change detected": "未偵測到持續姿態偏移",
        "WATCHING": "觀察中",
        "Gradual arm movement detected": "偵測到漸進式手臂位移",
        "Monitoring for persistence...": "持續監測中...",
        "ATTENTION NEEDED": "需要注意",
        "Persistent arm displacement detected": "偵測到持續手臂位移",
        "SIGNAL LIMITED": "訊號受限",
        "Unable to assess reliably": "目前無法可靠判讀",
        "Visibility is reduced": "可視範圍受限",
        "Patient or arm not clearly visible": "病人或手臂影像不清晰",
        "Recently alerted - monitoring": "近期曾警示，持續監測中",
        "SIGNAL QUALITY": "訊號品質",
        "GOOD": "良好",
        "LIMITED": "受限",
        "UNAVAILABLE": "無法判讀",
        "ARM POSITION TREND": "手臂位置趨勢",
        "trend_span": "{feature}  |  最近 {span:g} 秒",
        "now_value": "現在  {value:.3f}",
        "RECENT EVENTS": "最近事件",
        "events_count": "本次治療事件：{count}",
        "No events yet": "尚無事件",
        "GRADUAL": "漸進",
        "SUDDEN": "突發",
        "THRESHOLD": "閾值",
        "Gradual arm drift": "漸進式手臂位移",
        "Sudden movement": "突然移動",
        "Arm threshold crossed": "手臂位置超過閾值",
        "event_time": "治療後 {time}",
        "ACKNOWLEDGE": "已查看",
        "VIEW EVENT": "查看事件",
        "ESTABLISHING BASELINE": "建立個人基準中",
        "Learning patient's resting posture": "正在學習目前靜止姿態",
        "Monitoring will begin automatically": "完成後將自動開始監測",
        "Waiting for a clear view to finish setup": "等待清晰畫面以完成設定",
        "SETTING UP MONITORING": "監測設定中",
        "CAMERA VIEW": "攝影機畫面",
        "POSE TRACKED": "姿態追蹤中",
        "Video unavailable": "無法取得影像",
        "TREATMENT": "治療時間",
        "treatment_clock": "治療時間 {clock}",
        "LIVE": "即時",
        "BASELINE": "基準",
        "ENTER": "進入",
        "EXIT": "退出",
        "NOW": "現在",
        "progress": "{done:.1f} / {need:.1f} 秒",
    },
}

_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msjh.ttc"),
    Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
    Path("C:/Windows/Fonts/mingliu.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)
_CJK_FONT = (next((path for path in _FONT_CANDIDATES if path.is_file()), None)
             if ImageFont is not None else None)
_CJK_BOLD_FONT = (Path("C:/Windows/Fonts/msjhbd.ttc")
                  if Path("C:/Windows/Fonts/msjhbd.ttc").is_file()
                  else _CJK_FONT)


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False):
    return ImageFont.truetype(
        str(_CJK_BOLD_FONT if bold else _CJK_FONT), size)


def fit_preview(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale the complete canonical frame into a dark preview viewport."""
    width, height = size
    if width <= 0 or height <= 0:
        return frame
    src_h, src_w = frame.shape[:2]
    if (width, height) == (src_w, src_h):
        return frame
    scale = min(width / src_w, height / src_h)
    fitted_w = max(1, min(width, int(round(src_w * scale))))
    fitted_h = max(1, min(height, int(round(src_h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    fitted = cv2.resize(frame, (fitted_w, fitted_h),
                        interpolation=interpolation)
    preview = np.full((height, width, 3), C["navy"], np.uint8)
    x, y = (width - fitted_w) // 2, (height - fitted_h) // 2
    preview[y:y + fitted_h, x:x + fitted_w] = fitted
    return preview


class PreviewWindow:
    """Resizable OpenCV preview. VideoWriter still receives the source frame."""

    def __init__(self, name: str, width: int = 1280, height: int = 720):
        self.name = name
        self.fallback_size = (width, height)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL |
                        getattr(cv2, "WINDOW_FREERATIO", 0))
        cv2.resizeWindow(name, width, height)

    def show(self, frame: np.ndarray) -> None:
        size = self.fallback_size
        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
            if width > 0 and height > 0:
                size = (width, height)
        except (AttributeError, cv2.error):
            pass
        cv2.imshow(self.name, fit_preview(frame, size))


@dataclass
class LayoutConfig:
    width: int = 1280
    height: int = 720
    video_w: int = 720
    curve_h: int = 440  # Retained for existing YAML compatibility.
    event_h: int = 160  # Retained for existing YAML compatibility.
    curve_span_s: float = 90.0
    bed_label: str = "Bed 07"
    show_color_panel: bool = False
    show_roi_overlay: bool = False
    debug: bool = False
    language: str = "en"


class DemoRenderer:
    """Draw a clinical-facing bedside frame from existing pipeline fields."""

    def __init__(self, layout: Optional[LayoutConfig] = None,
                 thresholds: Optional[dict[str, float]] = None,
                 primary_feature: str = "arm_displacement"):
        self.L = layout or LayoutConfig()
        self.thresholds = thresholds or {"enter": 0.15, "exit": 0.10}
        self.primary = primary_feature
        self._history: deque[tuple[float, float]] = deque(
            maxlen=max(2, int(self.L.curve_span_s * 60)))
        self._redness: deque[tuple[float, float]] = deque(
            maxlen=max(2, int(self.L.curve_span_s * 60)))
        self._events: deque[Event] = deque(maxlen=20)
        self._event_count = 0
        self._start_time: Optional[float] = None
        self.language = "en"
        self.set_language(self.L.language)
        self._cjk_labels: list[tuple[str, tuple[int, int], tuple[int, int, int],
                                     float, int]] = []

    def set_language(self, language: str) -> str:
        """Use English if the optional CJK font stack is unavailable."""
        if language not in ("en", "zh"):
            raise ValueError("language must be 'en' or 'zh'")
        self.language = language if language == "en" or _CJK_FONT else "en"
        return self.language

    def toggle_language(self) -> str:
        return self.set_language("zh" if self.language == "en" else "en")

    def render(self, obs: FrameObservation, features: dict[str, float],
               state: State, new_event: Optional[Event] = None,
               timestamp: Optional[float] = None, color_cfg=None,
               baseline_progress: Optional[tuple[float, float]] = None,
               fusion: Optional[FusionResult] = None,
               fast_event: Optional[Event] = None,
               baseline_ready: Optional[bool] = None) -> np.ndarray:
        ts = obs.timestamp if timestamp is None else timestamp
        if self._start_time is None:
            self._start_time = ts
        value = features.get(self.primary, float("nan"))
        if np.isfinite(value):
            self._history.append((ts, value))
        red = features.get("roi_redness", float("nan"))
        if np.isfinite(red):
            self._redness.append((ts, red))
        for event in (new_event, fast_event):
            if event is not None:
                self._events.append(event)
                self._event_count += 1

        self._cjk_labels.clear()
        canvas = np.full((self.L.height, self.L.width, 3), C["bg"], np.uint8)
        self._draw_header(canvas, max(0.0, ts - self._start_time))
        top, bottom = 82, self.L.height - 16
        left, left_end = 20, min(self.L.video_w, self.L.width - 380)
        right, right_end = left_end + 16, self.L.width - 20
        video_end = bottom - 56
        building = (baseline_progress is not None
                    and (not baseline_ready if baseline_ready is not None
                         else baseline_progress[1] > 0
                         and baseline_progress[0] < baseline_progress[1]))
        self._draw_video(canvas, obs, (left, top, left_end, video_end),
                         state, fusion, color_cfg, baseline_building=building)
        self._draw_video_footer(canvas, (left, video_end + 12, left_end, bottom),
                                any(kp.is_valid for kp in obs.keypoints))

        status_end, chart_end = top + 174, bottom - 198
        if building:
            self._draw_baseline_progress(
                canvas, (right, top, right_end, chart_end), *baseline_progress)
        else:
            self._draw_status(canvas, (right, top, right_end, status_end),
                              state, fusion, value)
            self._draw_curve(canvas, (right, status_end + 12, right_end,
                                      chart_end), ts, value, features)
        self._draw_events(canvas, (right, chart_end + 12, right_end, bottom))
        self._flush_cjk(canvas)
        return canvas

    @staticmethod
    def _text(img, s, org, color, scale=.6, thickness=1) -> None:
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)

    def _draw_text(self, img, s, org, color, scale=.6, thickness=1) -> None:
        if any(ord(char) > 127 for char in s) and _CJK_FONT is not None:
            self._cjk_labels.append((s, org, color, scale, thickness))
        else:
            self._text(img, s, org, color, scale, thickness)

    def _label(self, img, key, org, color, scale=.6, thickness=1,
               **values) -> None:
        template = TEXT[self.language].get(
            key, TEXT["en"].get(key, key))
        self._draw_text(img, template.format(**values), org, color,
                        scale, thickness)

    def _flush_cjk(self, canvas: np.ndarray) -> None:
        if not self._cjk_labels:
            return
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for label, origin, color, scale, thickness in self._cjk_labels:
            size = max(11, int(round(scale * 32)))
            draw.text(origin, label, font=_font(size, thickness > 1),
                      fill=color[::-1],
                      anchor="ls")
        canvas[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _box(canvas, box, color) -> None:
        x0, y0, x1, y1 = box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)

    def _draw_header(self, canvas: np.ndarray, elapsed: float) -> None:
        cv2.rectangle(canvas, (0, 0), (self.L.width, 64), C["navy"], -1)
        self._text(canvas, "DIALYSIGHT", (22, 40), C["white"], .82, 2)
        cv2.line(canvas, (226, 18), (226, 48), C["navy_soft"], 1)
        bed = self.L.bed_label.upper() if self.language == "en" else (
            self.L.bed_label.replace("Bed", "床位").replace("bed", "床位"))
        self._draw_text(canvas, bed, (246, 39), C["white"], .63, 2)
        cv2.circle(canvas, (419, 33), 5, C["lime"], -1, cv2.LINE_AA)
        self._label(canvas, "LIVE", (433, 39), C["white"], .52)
        seconds = int(elapsed)
        clock = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        self._label(canvas, "treatment_clock", (505, 39), C["white"],
                    .54, clock=clock)
        right = self.L.width - 20
        zh_box = (right - 130, 16, right - 70, 50)
        en_box = (right - 64, 16, right, 50)
        for box, active in ((zh_box, self.language == "zh"),
                            (en_box, self.language == "en")):
            cv2.rectangle(canvas, box[:2], box[2:],
                          C["white"] if active else C["navy_soft"], -1)
        self._draw_text(canvas, "中文", (zh_box[0] + 11, 40),
                        C["navy"] if self.language == "zh" else C["white"], .47)
        self._text(canvas, "EN", (en_box[0] + 17, 40),
                   C["navy"] if self.language == "en" else C["white"], .47)

    def _draw_video(self, canvas: np.ndarray, obs: FrameObservation, box,
                    state: State, fusion: Optional[FusionResult],
                    color_cfg=None, baseline_building=False) -> None:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        frame = np.full((h, w, 3), C["navy"], np.uint8)
        if obs.raw_frame is not None:
            src = obs.raw_frame
            if self.L.show_roi_overlay and color_cfg is not None:
                from .temporal.color import roi_overlay
                overlaid = roi_overlay(obs, color_cfg)
                if overlaid is not None:
                    src = overlaid
            sh, sw = src.shape[:2]
            scale = min(w / sw, h / sh)
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            ox, oy = (w - nw) // 2, (h - nh) // 2
            frame[oy:oy + nh, ox:ox + nw] = cv2.resize(src, (nw, nh))
            self._draw_skeleton(frame, obs, (ox, oy, nw, nh))
        border = (C["limited"] if baseline_building
                  else self._status(state, fusion)[2])
        cv2.rectangle(frame, (1, 1), (w - 2, h - 2), border, 2)
        canvas[y0:y1, x0:x1] = frame
        if obs.raw_frame is None:
            self._label(canvas, "Video unavailable", (x0 + 22, y0 + 42),
                        C["white"])

    def _draw_skeleton(self, frame: np.ndarray, obs: FrameObservation,
                       box: tuple[int, int, int, int]) -> None:
        ox, oy, w, h = box

        def px(kp):
            return int(ox + kp.x * w), int(oy + kp.y * h)

        for a, b in SKELETON:
            ka, kb = obs.keypoints[int(a)], obs.keypoints[int(b)]
            if ka.is_valid and kb.is_valid:
                cv2.line(frame, px(ka), px(kb), (20, 20, 24), 5, cv2.LINE_AA)
                cv2.line(frame, px(ka), px(kb), C["skeleton"], 2, cv2.LINE_AA)
        for kp in obs.keypoints:
            if kp.is_valid:
                cv2.circle(frame, px(kp), 4, C["keypoint"], -1, cv2.LINE_AA)
        wrist = obs.kp(KP.RIGHT_WRIST)
        if wrist.is_valid:
            cv2.circle(frame, px(wrist), 10, C["watch"], 2, cv2.LINE_AA)

    @staticmethod
    def _status(state: State, fusion: Optional[FusionResult]):
        if fusion is not None and fusion.reliability is Reliability.UNRELIABLE:
            return "SIGNAL LIMITED", "Unable to assess reliably", C["limited"]
        if fusion is not None and fusion.reliability is Reliability.DEGRADED:
            return "SIGNAL LIMITED", "Visibility is reduced", C["limited"]
        if state is State.ALERT:
            return ("ATTENTION NEEDED", "Persistent arm displacement detected",
                    C["alert"])
        if state is State.WATCHING:
            return "WATCHING", "Gradual arm movement detected", C["watch"]
        if state is State.COOLDOWN:
            return "STABLE", "Recently alerted - monitoring", C["stable"]
        return "STABLE", "No persistent posture change detected", C["stable"]

    @staticmethod
    def _quality(fusion: Optional[FusionResult]) -> str:
        if fusion is None or fusion.reliability is Reliability.UNRELIABLE:
            return "UNAVAILABLE"
        if fusion.reliability is Reliability.DEGRADED:
            return "LIMITED"
        return "GOOD"

    def _draw_video_footer(self, canvas, box, pose_visible: bool) -> None:
        self._box(canvas, box, C["paper"])
        x0, y0, _, _ = box
        self._label(canvas, "CAMERA VIEW", (x0 + 18, y0 + 29),
                    C["text"], .50, 2)
        if pose_visible:
            cv2.circle(canvas, (x0 + 239, y0 + 23), 4,
                       C["stable"], -1, cv2.LINE_AA)
            self._label(canvas, "POSE TRACKED", (x0 + 253, y0 + 29),
                        C["muted"], .43)

    def _draw_status(self, canvas, box, state, fusion, value) -> None:
        self._box(canvas, box, C["navy"])
        x0, y0, x1, _ = box
        title, reason, color = self._status(state, fusion)
        self._label(canvas, "CURRENT STATUS", (x0 + 22, y0 + 29),
                    (194, 192, 184), .46)
        cv2.circle(canvas, (x1 - 36, y0 + 24), 7, color, -1, cv2.LINE_AA)
        self._label(canvas, title, (x0 + 22, y0 + 76), C["white"], .90, 2)
        self._label(canvas, reason, (x0 + 22, y0 + 108),
                    (220, 218, 213), .49)
        quality = self._quality(fusion)
        self._label(canvas, "SIGNAL QUALITY", (x1 - 180, y0 + 28),
                    (194, 192, 184), .40)
        self._label(canvas, quality, (x1 - 180, y0 + 54),
                    C["stable"] if quality == "GOOD" else C["limited"],
                    .51, 2)
        if state is State.WATCHING and title == "WATCHING":
            self._label(canvas, "Monitoring for persistence...",
                        (x0 + 22, y0 + 132), (220, 218, 213), .46)
        if fusion is not None and fusion.reliability is Reliability.UNRELIABLE:
            self._label(canvas, "Patient or arm not clearly visible",
                        (x0 + 22, y0 + 132), (220, 218, 213), .46)
        if title == "ATTENTION NEEDED":
            cv2.rectangle(canvas, (x0 + 22, y0 + 125),
                          (x0 + 181, y0 + 162), C["white"], -1)
            cv2.rectangle(canvas, (x0 + 191, y0 + 125),
                          (x0 + 328, y0 + 162), C["navy_soft"], -1)
            self._label(canvas, "ACKNOWLEDGE", (x0 + 32, y0 + 149),
                        C["navy"], .40, 2)
            self._label(canvas, "VIEW EVENT", (x0 + 204, y0 + 149),
                        C["white"], .40, 2)
        if self.L.debug and title != "ATTENTION NEEDED":
            shown = f"{value:.3f}" if np.isfinite(value) else "--"
            confidence = (f"  conf {fusion.confidence:.2f}"
                          if fusion is not None else "")
            self._text(canvas, f"{state.value}  {shown}{confidence}",
                       (x1 - 235, y0 + 158), (194, 192, 184), .35)

    def _draw_baseline_progress(self, canvas, box, done, need) -> None:
        self._box(canvas, box, C["navy"])
        x0, y0, x1, y1 = box
        self._label(canvas, "SETTING UP MONITORING", (x0 + 24, y0 + 34),
                    (194, 192, 184), .46)
        self._label(canvas, "ESTABLISHING BASELINE", (x0 + 24, y0 + 105),
                    C["white"], .76, 2)
        self._label(canvas, "Learning patient's resting posture",
                    (x0 + 24, y0 + 168), (220, 218, 213), .48)
        bx, by, bw = x0 + 24, y0 + 208, x1 - x0 - 48
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + 12),
                      C["navy_soft"], -1)
        frac = max(0.0, min(1.0, done / need))
        cv2.rectangle(canvas, (bx, by), (bx + int(bw * frac), by + 12),
                      C["lime"], -1)
        self._label(canvas, "progress", (bx, by + 40), C["white"], .55,
                    done=done, need=need)
        note = ("Waiting for a clear view to finish setup" if frac >= 1.0
                else "Monitoring will begin automatically")
        self._label(canvas, note, (bx, min(y1 - 24, by + 88)),
                    (220, 218, 213), .45)

    def _draw_curve(self, canvas, box, now, value, features) -> None:
        self._box(canvas, box, C["paper"])
        x0, y0, x1, y1 = box
        self._label(canvas, "ARM POSITION TREND", (x0 + 20, y0 + 28),
                    C["text"], .55, 2)
        self._label(canvas, "trend_span", (x0 + 20, y0 + 49),
                    C["muted"], .40, feature=self.primary,
                    span=self.L.curve_span_s)
        if np.isfinite(value):
            self._label(canvas, "now_value", (x1 - 145, y0 + 29),
                        C["curve"], .46, 2, value=value)
        px0, py0, px1, py1 = x0 + 26, y0 + 70, x1 - 26, y1 - 28
        cv2.line(canvas, (px0, py1), (px1, py1), C["line"], 1)
        self._label(canvas, "BASELINE", (px0 + 4, py1 - 5),
                    C["muted"], .34)
        visible = [(t, v) for t, v in self._history
                   if t >= now - self.L.curve_span_s]
        ceiling = max([v for _, v in visible] +
                      list(self.thresholds.values()) + [0.01]) * 1.2

        def xy(t, v):
            fx = min(1.0, max(0.0, (t - now + self.L.curve_span_s)
                               / self.L.curve_span_s))
            return (int(px0 + fx * (px1 - px0)),
                    int(py1 - min(1.0, max(0.0, v / ceiling)) * (py1 - py0)))

        if "enter" in self.thresholds:
            enter_y = xy(now, self.thresholds["enter"])[1]
            if enter_y > py0:
                cv2.rectangle(canvas, (px0, py0), (px1, enter_y - 1),
                              C["warm_tint"], -1)
        for label, color in (("enter", C["watch"]), ("exit", C["muted"])):
            if label in self.thresholds:
                yy = xy(now, self.thresholds[label])[1]
                cv2.line(canvas, (px0, yy), (px1, yy), color, 1, cv2.LINE_AA)
                self._label(canvas, label.upper(), (px0 + 4, yy - 5),
                            color, .35)
        if len(visible) >= 2:
            points = np.array([xy(t, v) for t, v in visible], np.int32)
            cv2.polylines(canvas, [points], False, C["curve"], 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(points[-1]), 4, C["curve"], -1,
                       cv2.LINE_AA)
        cv2.line(canvas, (px1, py0), (px1, py1), C["line"], 1)
        self._label(canvas, "NOW", (px1 - 32, py1 + 18), C["muted"], .36)
        if self.L.debug:
            torso = features.get("torso_angle", float("nan"))
            if np.isfinite(torso):
                self._text(canvas, f"torso_angle {torso:.2f}",
                           (x0 + 225, y0 + 49), C["muted"], .38)

    def _draw_events(self, canvas, box) -> None:
        self._box(canvas, box, C["paper"])
        x0, y0, x1, y1 = box
        self._label(canvas, "RECENT EVENTS", (x0 + 20, y0 + 27),
                    C["text"], .54, 2)
        self._label(canvas, "events_count", (x1 - 210, y0 + 27),
                    C["muted"], .40, count=self._event_count)
        cv2.line(canvas, (x0 + 20, y0 + 40), (x1 - 20, y0 + 40),
                 C["line"], 1)
        if not self._events:
            self._label(canvas, "No events yet", (x0 + 20, y0 + 78),
                        C["muted"], .57)
            return
        for i, event in enumerate(reversed(self._events)):
            yy = y0 + 65 + i * 50
            if yy + 32 > y1:
                break
            tag, title = self._event_label(event)
            color = C["alert"] if event.channel == "fast" else C["watch"]
            self._label(canvas, tag, (x0 + 20, yy), color, .43, 2)
            self._label(canvas, title, (x0 + 124, yy), C["text"], .48)
            mm, ss = divmod(int(event.start_time), 60)
            self._label(canvas, "event_time", (x0 + 124, yy + 19),
                        C["muted"], .37, time=f"{mm:02d}:{ss:02d}")
            if self.L.debug:
                self._text(canvas, f"{event.feature} peak {event.peak_value:.3f}",
                           (x0 + 260, yy + 19), C["muted"], .33)

    @staticmethod
    def _event_label(event: Event) -> tuple[str, str]:
        if event.channel == "fast":
            return "SUDDEN", "Sudden movement"
        if event.channel == "baseline":
            return "THRESHOLD", "Arm threshold crossed"
        return "GRADUAL", "Gradual arm drift"


class ComparisonRenderer:
    """A 1280x720 frame comparing separate pipelines on the same source."""

    def __init__(self, baseline_renderer: DemoRenderer,
                 ours_renderer: DemoRenderer):
        self.baseline = baseline_renderer
        self.ours = ours_renderer
        self.width, self.height = 1280, 720
        self.baseline_alerts = 0
        self.ours_alerts = 0
        self._baseline_open_event: Optional[Event] = None

    def render(self, baseline_result: FrameResult,
               ours_result: FrameResult) -> np.ndarray:
        self.baseline_alerts += int(baseline_result.new_event is not None)
        self.baseline_alerts += int(baseline_result.fast_event is not None)
        self.ours_alerts += int(ours_result.new_event is not None)
        self.ours_alerts += int(ours_result.fast_event is not None)
        if baseline_result.new_event is not None:
            self._baseline_open_event = baseline_result.new_event
        baseline_open = (self._baseline_open_event is not None
                         and self._baseline_open_event.is_open)
        canvas = np.full((self.height, self.width, 3), C["bg"], np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.width, 68), C["navy"], -1)
        DemoRenderer._text(canvas, "SAME VIDEO  /  SAME INPUT", (24, 43),
                           C["white"], .85, 2)
        self._side(canvas, (20, 86, 632, 700), "SINGLE THRESHOLD",
                   baseline_result, self.baseline_alerts, self.baseline,
                   baseline_open=baseline_open)
        self._side(canvas, (648, 86, 1260, 700), "DIALYSIGHT TEMPORAL",
                   ours_result, self.ours_alerts, self.ours)
        return canvas

    def _side(self, canvas, box, label, result, count, renderer,
              baseline_open=False) -> None:
        x0, y0, x1, y1 = box
        DemoRenderer._box(canvas, box, C["paper"])
        DemoRenderer._text(canvas, label, (x0 + 20, y0 + 32),
                           C["text"], .65, 2)
        DemoRenderer._text(canvas, f"ALERTS: {count}", (x0 + 20, y0 + 89),
                           C["alert"], 1.12, 3)
        renderer._draw_video(canvas, result.obs,
                             (x0 + 14, y0 + 112, x1 - 14, y1 - 88),
                             result.state, result.fusion,
                             baseline_building=not result.baseline_ready)
        if not result.baseline_ready:
            title, reason, color = ("ESTABLISHING BASELINE",
                                    "Monitoring will begin automatically",
                                    C["limited"])
        elif baseline_open:
            title, reason, color = ("ATTENTION NEEDED",
                                    "Arm position threshold crossed",
                                    C["alert"])
        elif result.fast_event is not None:
            title, reason, color = ("SUDDEN MOVEMENT",
                                    "Fast channel event confirmed",
                                    C["alert"])
        else:
            title, reason, color = renderer._status(result.state, result.fusion)
        DemoRenderer._text(canvas, title, (x0 + 20, y1 - 48), color, .69, 2)
        DemoRenderer._text(canvas, reason, (x0 + 20, y1 - 22),
                           C["muted"], .42)


def make_writer(path: str, fps: float, layout: LayoutConfig) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (layout.width, layout.height))
