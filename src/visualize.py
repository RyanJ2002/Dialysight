"""
床邊畫面與同片對照畫面（OpenCV 合成，PIL 排字）。

床邊畫面版面（預設 1280×720，深色）：

    ┌ 頁首 ─ 床號 · 穿刺側 · 治療時間 │ NORMAL ─ WATCHING ─ ALERT │ 相機 · 光線 · 本機處理 │ 時鐘
    ├ 左 ── 影像 + 骨架。虛線 = 治療開始時的手臂位置，實線 = 現在；偏離標籤跟著手腕走
    ├ 右 ── 狀態卡：現在是什麼狀態、為什麼還沒通知、還差多久
    │       趨勢圖：門檻、基準範圍、趨勢外推
    │       訊號投票 │ 雙通道
    └ 頁尾 ─ 本次治療時間軸（狀態色帶 + 事件標記）· 最近事件 · 裝置效能

★ 畫面上的數字都來自管線既有的欄位，這裡不做任何判斷。
  唯一「自己算」的是趨勢外推（最近 10 秒的線性斜率）——
  它只用來顯示，不會觸發任何東西，畫面上也標成「預估」。

★ 治療開始時的手臂位置（虛線）是在基準建立期間，從這台機器上的關鍵點
  平均出來的，只存在記憶體裡，不會送出床邊。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # 精簡部署時英文介面仍可用
    Image = ImageDraw = ImageFont = None

try:
    import psutil
except ImportError:
    psutil = None

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

ARM = {
    "right": (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
    "left": (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST),
}


# --------------------------------------------------------------------------
# 色票（BGR）
# --------------------------------------------------------------------------

def _bgr(hex_rgb: str) -> tuple[int, int, int]:
    v = int(hex_rgb.lstrip("#"), 16)
    return (v & 255, (v >> 8) & 255, v >> 16)


def _mix(a, b, t: float) -> tuple[int, int, int]:
    """a 往 b 混 t（0 = a，1 = b）。"""
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


C = {
    "bg": _bgr("18202B"),
    "panel": _bgr("212B38"),
    "raised": _bgr("2A3544"),
    "line": _bgr("2E3A48"),
    "text": _bgr("E8ECF1"),
    "muted": _bgr("9AA6B4"),
    "dim": _bgr("7D8896"),
    "teal": _bgr("5FBF8F"),
    "teal_dim": _bgr("2C4A3B"),
    "amber": _bgr("F5B547"),
    "red": _bgr("E5534B"),
    "unrel": _bgr("A3ACB8"),
    "cool": _bgr("8FD3AE"),
    "base": _bgr("7FA6D6"),
    "ghost": _bgr("8D98A6"),
    "white": (255, 255, 255),
}

# ★ unreliable 用灰色加斜線，不用綠色 —— 它的意思是「看不到」，不是「沒事」。
#   跟護理站的「無法判定 = 灰色虛線框」一致。
MODE_COLOR = {
    "normal": C["teal"], "watching": C["amber"], "alert": C["red"],
    "cooldown": C["cool"], "unreliable": C["unrel"], "baseline": C["base"],
}

# 時間軸色帶：正常的時段要安靜，有狀況的時段才跳出來
RUN_COLOR = {
    "normal": C["teal_dim"], "watching": C["amber"], "alert": C["red"],
    "cooldown": _mix(C["cool"], C["panel"], .45),
    "unreliable": _mix(C["unrel"], C["panel"], .6),
    "baseline": _mix(C["base"], C["panel"], .55),
}


# --------------------------------------------------------------------------
# 文字
# --------------------------------------------------------------------------

# key → (English, 中文)。{arm} 會自動代入「右手 / R-arm」。
T = {
    "site": ("{arm} access · treatment", "{arm}穿刺 · 治療"),
    "cam_ok": ("Camera OK", "相機位置正常"),
    "cam_lost": ("Patient not found", "找不到病人"),
    "light_ok": ("Good light", "光線充足"),
    "light_low": ("Low light", "光線不足"),
    "light_high": ("Overexposed", "畫面過亮"),
    "no_video": ("No video", "無影像"),
    "local": ("Video stays on device", "影像僅本機處理"),
    "live": ("LIVE", "LIVE"),
    "pose_conf": ("Pose conf {v:.2f}", "姿態信心 {v:.2f}"),
    "roi": ("Monitoring ROI", "監測範圍 ROI"),
    "legend_base": ("At treatment start", "治療開始時的位置"),
    "legend_now": ("Current", "目前位置"),
    "callout": ("{arm} vs. baseline", "{arm}偏離基準"),
    "video_unavailable": ("Video unavailable", "無法取得影像"),
    "badge_occluded": ("{arm} wrist not visible · cannot assess",
                       "看不到{arm}腕 · 無法判定"),
    "badge_baseline": ("Learning resting posture · please stay still",
                       "正在學習靜止姿態 · 請保持不動"),
    "word_normal": ("NORMAL", "NORMAL"),
    "word_watching": ("WATCHING", "WATCHING"),
    "word_alert": ("ALERT", "ALERT"),
    "word_cooldown": ("COOLDOWN", "COOLDOWN"),
    "word_unreliable": ("NO READING", "無法判定"),
    "word_baseline": ("BASELINE", "建立基準"),
    "desc_normal": ("Monitoring", "持續監測中"),
    "desc_watching": ("Logged only · not notified", "只記錄，尚未通知"),
    "desc_alert": ("Nurse station notified", "已通知護理站"),
    "desc_cooldown": ("Event resolved", "事件已解除"),
    "desc_unreliable": ("Not “normal” — not visible", "不是「正常」，是「看不到」"),
    "desc_baseline": ("Learning resting posture", "學習目前的靜止姿態"),
    "degraded": (" · signal limited", " · 訊號受限"),
    "lbl_dev": ("Deviation", "偏離基準"),
    "lbl_confirm": ("Confirming", "確認中"),
    "lbl_elapsed": ("Duration", "已持續"),
    "lbl_cool": ("Cooldown", "冷卻"),
    "lbl_conf": ("Confidence", "可信度"),
    "lbl_progress": ("Progress", "進度"),
    "peak": ("peak {v:.2f}", "峰值 {v:.2f}"),
    "note_normal": ("Escalates only after > {enter:.2f} for {t:g} s",
                    "越過門檻 {enter:.2f} 並持續 {t:g} 秒，才會升級為 ALERT"),
    "note_watching": ("Must stay > {enter:.2f} for {t:g} s before the station is notified",
                      "越過門檻 {enter:.2f} 並持續 {t:g} 秒，才會升級為 ALERT 並通知護理站"),
    "note_falling": ("Back under {enter:.2f} · confirmation paused",
                     "已回到門檻 {enter:.2f} 以下 · 確認計時暫停"),
    "note_alert": ("Clears after < {exit:.2f} for {t:g} s",
                   "回到 {exit:.2f} 以下並維持 {t:g} 秒才會解除"),
    "note_cooldown": ("No repeat notification during cooldown · detection continues",
                      "冷卻期內不重複通知，偵測仍持續"),
    "note_unreliable": ("Check the access site visually on your next round",
                        "建議巡視時目視確認穿刺處"),
    "note_baseline": ("Monitoring starts automatically when done",
                      "完成後自動開始監測"),
    "note_baseline_wait": ("Waiting for a clear view to finish setup",
                           "等待清晰畫面以完成設定"),
    "chart_title": ("{arm} deviation · last {s:g} s", "{arm}偏離量 · 最近 {s:g} 秒"),
    "predict": ("Est. ~{s:.0f} s to threshold", "預估 ~{s:.0f} 秒後越過門檻"),
    "now_val": ("now {v:.3f}", "現在 {v:.3f}"),
    "band": ("Baseline range", "基準範圍"),
    "votes_title": ("Signal vote", "訊號投票"),
    "conf": ("conf {v}", "信心 {v}"),
    "sig_arm": ("{arm} shift", "{arm}位移"),
    "sig_torso": ("Torso angle", "軀幹角度"),
    "sig_vis": ("Visibility", "可見度"),
    "ch_title": ("Dual channel", "雙通道"),
    "ch_slow": ("Slow · gradual", "慢通道 · 漸進"),
    "ch_fast": ("Fast · acute", "快通道 · 急性"),
    "st_idle": ("Standby", "待命"),
    "st_watch": ("Watching", "觀察中"),
    "st_alert": ("Alerting", "告警中"),
    "st_cool": ("Cooldown", "冷卻中"),
    "st_hold": ("Paused · no view", "暫停 · 看不到"),
    "st_base": ("Learning", "建立基準"),
    "st_fast_idle": ("Ready < 1 s", "待命 < 1 s"),
    "st_verify": ("Verifying", "驗證中"),
    "st_fired": ("Triggered", "已觸發"),
    "glitches": ("Glitches rejected: {n}", "已擋下偵測跳點 {n} 次"),
    "session": ("Session", "本次治療"),
    "events_n": ("Events: {n}", "事件 {n} 筆"),
    "infer": ("infer {v:.0f} ms", "推論 {v:.0f} ms"),
    "keys": ("L language · Q quit", "L 切換語言 · Q 離開"),
    "cmp_title": ("Same video · same input", "同一段影片 · 同一份輸入"),
    "cmp_base": ("Single threshold (control)", "單一閾值（對照組）"),
    "cmp_ours": ("DialySight temporal layer", "DialySight 時序層"),
    "cmp_base_sub": ("Notifies on every crossing", "每越線一次就通知一次"),
    "cmp_ours_sub": ("Hysteresis + confirmation + dual channel",
                     "遲滯 + 確認時間 + 雙通道"),
    "cmp_alerts": ("alerts", "次告警"),
    "cmp_fired": ("Threshold crossed · notified", "超過閾值，已通知"),
    "cmp_sudden": ("Sudden movement confirmed", "突發位移已確認"),
}

ARM_WORD = {("right", "en"): "R-arm", ("right", "zh"): "右手",
            ("left", "en"): "L-arm", ("left", "zh"): "左手"}


# --------------------------------------------------------------------------
# 字型
# --------------------------------------------------------------------------

def _first(*paths: Path) -> Optional[str]:
    return next((str(p) for p in paths if p.is_file()), None)


_WIN = Path("C:/Windows/Fonts")
_NOTO = Path("/usr/share/fonts/opentype/noto")
_NOTO_TT = Path("/usr/share/fonts/truetype/noto")
_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
_MAC = Path("/System/Library/Fonts")

if ImageFont is not None:
    _cjk = _first(_WIN / "msjh.ttc", _WIN / "NotoSansTC-VF.ttf",
                  _NOTO / "NotoSansCJK-Regular.ttc", _NOTO_TT / "NotoSansCJK-Regular.ttc",
                  _MAC / "PingFang.ttc", _WIN / "mingliu.ttc")
    _cjk_bold = _first(_WIN / "msjhbd.ttc", _NOTO / "NotoSansCJK-Bold.ttc",
                       _NOTO_TT / "NotoSansCJK-Bold.ttc") or _cjk
    _sans = _first(_WIN / "segoeui.ttf", _DEJAVU / "DejaVuSans.ttf",
                   _MAC / "Helvetica.ttc") or _cjk
    _sans_bold = _first(_WIN / "segoeuib.ttf", _DEJAVU / "DejaVuSans-Bold.ttf") or _sans
    _display = _first(_WIN / "bahnschrift.ttf",
                      _DEJAVU / "DejaVuSansCondensed-Bold.ttf") or _sans_bold
    _mono = _first(_WIN / "consola.ttf", _DEJAVU / "DejaVuSansMono.ttf",
                   _MAC / "Menlo.ttc") or _sans
    _mono_bold = _first(_WIN / "consolab.ttf", _DEJAVU / "DejaVuSansMono-Bold.ttf") or _mono
else:
    _cjk = _cjk_bold = _sans = _sans_bold = _display = _mono = _mono_bold = None

_FONTS = {"sans": _sans, "bold": _sans_bold, "display": _display,
          "mono": _mono, "mono_bold": _mono_bold, "cjk": _cjk, "cjk_bold": _cjk_bold}
_CJK_FONT = _cjk     # 沒有中文字型時介面自動退回英文


def _font_kind(kind: str, s: str) -> str:
    """含中日韓字元的字串一律改用中文字型（拉丁字型沒有那些字）。"""
    if any(ord(ch) >= 0x2E80 for ch in s):
        return "cjk_bold" if kind in ("bold", "display", "mono_bold") else "cjk"
    return kind


@lru_cache(maxsize=256)
def _font(kind: str, size: int):
    path = _FONTS.get(kind)
    if path is None or ImageFont is None:
        return None
    try:
        font = ImageFont.truetype(path, size)
    except OSError:
        return None
    if kind == "display" and path.lower().endswith("bahnschrift.ttf"):
        try:
            font.set_variation_by_name("Bold")
        except Exception:
            pass
    return font


_ASCII = str.maketrans({"·": "-", "→": "->", "↑": "^", "↓": "v", "°": "",
                        "—": "-", "“": '"', "”": '"', "~": "~"})


@lru_cache(maxsize=2048)
def _glyph_mask(s: str, kind: str, size: int, anchor: str):
    """把一段字排成 alpha 遮罩並快取。

    畫面上大部分文字（標籤、標題）每格都一樣，快取起來之後每格只剩
    小塊的 alpha 混合，不必每格重新點陣化、也不必整張畫布轉 PIL 再轉回來。
    """
    font = _font(_font_kind(kind, s), size)
    if font is None:
        return None
    left, top, right, bottom = font.getbbox(s, anchor=anchor)
    w, h = max(1, right - left), max(1, bottom - top)
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-left, -top), s, font=font, fill=255, anchor=anchor)
    return (np.asarray(im, dtype=np.float32)[..., None] / 255.0, left, top)


@lru_cache(maxsize=4096)
def _text_width(s: str, size: int, kind: str) -> int:
    font = _font(_font_kind(kind, s), size)
    if font is not None:
        return int(math.ceil(font.getlength(s)))
    (w, _), _ = cv2.getTextSize(s.translate(_ASCII), cv2.FONT_HERSHEY_SIMPLEX,
                                size / 30, 1)
    return w


@lru_cache(maxsize=64)
def _solid(h: int, w: int, color: tuple) -> np.ndarray:
    """單色底圖。np.full 填三通道顏色很慢，快取後每格只需 copy。"""
    out = np.empty((h, w, 3), np.uint8)
    out[:] = color
    out.flags.writeable = False
    return out


class Painter:
    """形狀直接畫在畫布上；文字先排隊，最後統一貼上。

    文字由 PIL 點陣化成 alpha 遮罩（有快取），再直接混合進 BGR 畫布。
    沒有 PIL 時退回 OpenCV 內建字型（只剩英文）。
    """

    def __init__(self, img: np.ndarray):
        self.img = img
        self._texts: list = []

    def text(self, s: str, org, color, size: int = 14, kind: str = "sans",
             anchor: str = "ls") -> None:
        if s:
            self._texts.append((s, (int(org[0]), int(org[1])), color,
                                int(size), kind, anchor))

    @staticmethod
    def width(s: str, size: int, kind: str = "sans") -> int:
        return _text_width(s, int(size), kind)

    def flush(self) -> None:
        img = self.img
        H, W = img.shape[:2]
        for item in self._texts:
            s, org, color, size, kind, anchor = item
            glyph = _glyph_mask(s, kind, size, anchor) if Image is not None else None
            if glyph is None:
                self._cv_text(*item)
                continue
            a, left, top = glyph
            x, y = org[0] + left, org[1] + top
            h, w = a.shape[:2]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            a = a[y0 - y:y1 - y, x0 - x:x1 - x]
            region = img[y0:y1, x0:x1].astype(np.float32)
            region += (np.array(color, np.float32) - region) * a
            img[y0:y1, x0:x1] = (region + .5).astype(np.uint8)
        self._texts = []

    def _cv_text(self, s, org, color, size, kind, anchor) -> None:
        s = s.translate(_ASCII).encode("ascii", "replace").decode()
        scale = size / 30
        thick = 2 if kind in ("bold", "display", "mono_bold") else 1
        (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        x -= {"m": w // 2, "r": w}.get(anchor[0], 0)
        y += {"m": h // 2, "t": h, "a": h}.get(anchor[1], 0)
        cv2.putText(self.img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)


# --------------------------------------------------------------------------
# 形狀
# --------------------------------------------------------------------------

def rrect(img, box, r, color, thickness: int = -1) -> None:
    """圓角矩形。直邊不需要反鋸齒，圓角用 LINE_AA 畫。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if r == 0:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        return
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    t = thickness
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, t, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, t, cv2.LINE_AA)


def rrect_alpha(img, box, r, color, alpha: float, thickness: int = -1) -> None:
    """半透明圓角矩形（填色或框線）。只混合那一小塊，不動整張畫布。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    pad = max(1, thickness) + 1
    h, w = img.shape[:2]
    ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
    ax1, ay1 = min(w, x1 + pad + 1), min(h, y1 + pad + 1)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    roi = img[ay0:ay1, ax0:ax1]
    over = roi.copy()
    rrect(over, (x0 - ax0, y0 - ay0, x1 - ax0, y1 - ay0), r, color, thickness)
    roi[:] = cv2.addWeighted(over, alpha, roi, 1 - alpha, 0)


def blend_rect(img, box, color, alpha: float) -> None:
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.addWeighted(_solid(y1 - y0, x1 - x0, tuple(color)), alpha, roi, 1 - alpha, 0)


def dashed_line(img, p0, p1, color, thickness: int = 1,
                dash: float = 7, gap: float = 5) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        cv2.line(img, (int(round(x0 + dx * pos)), int(round(y0 + dy * pos))),
                 (int(round(x0 + dx * end)), int(round(y0 + dy * end))),
                 color, thickness, cv2.LINE_AA)
        pos += dash + gap


def hatch(img, box, color, step: int = 6) -> None:
    """斜線填充 —— 「看不到」的時段用這個，跟實心的「有狀況」區分。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    sub = img[y0:y1, x0:x1].copy()
    for k in range(-h, w, step):
        cv2.line(sub, (k, h), (k + h, 0), color, 1, cv2.LINE_AA)
    img[y0:y1, x0:x1] = sub


@lru_cache(maxsize=16)
def _corner_alpha(r: int) -> np.ndarray:
    m = np.zeros((2 * r, 2 * r), np.uint8)
    cv2.circle(m, (r, r), r, 255, -1, cv2.LINE_AA)
    return (m.astype(np.float32) / 255.0)[..., None]


def round_corners(img, box, r: int, bg) -> None:
    """把一塊已經畫好的矩形內容修成圓角（用背景色蓋掉四個角）。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    r = int(min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if r <= 0:
        return
    a = _corner_alpha(r)
    bg_arr = np.array(bg, np.float32)
    for sy, sx, ay, ax in ((y0, x0, slice(0, r), slice(0, r)),
                           (y0, x1 - r, slice(0, r), slice(r, 2 * r)),
                           (y1 - r, x0, slice(r, 2 * r), slice(0, r)),
                           (y1 - r, x1 - r, slice(r, 2 * r), slice(r, 2 * r))):
        region = img[sy:sy + r, sx:sx + r].astype(np.float32)
        al = a[ay, ax]
        img[sy:sy + r, sx:sx + r] = (region * al + bg_arr * (1 - al)).astype(np.uint8)


def fill_under(img, pts: np.ndarray, base_y: int, color, alpha_top: float) -> None:
    """曲線下方的漸層填色，上濃下淡。"""
    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    y_min = int(pts[:, 1].min())
    if x_max - x_min < 2 or base_y - y_min < 2:
        return
    h, w = base_y - y_min + 1, x_max - x_min + 1
    poly = np.vstack([pts, [[x_max, base_y], [x_min, base_y]]]) - [x_min, y_min]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255, cv2.LINE_AA)
    ramp = np.linspace(alpha_top, 0.0, h, dtype=np.float32)[:, None]
    a = ((mask.astype(np.float32) / 255.0) * ramp)[..., None]
    region = img[y_min:base_y + 1, x_min:x_max + 1].astype(np.float32)
    img[y_min:base_y + 1, x_min:x_max + 1] = (
        region * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def progress_bar(img, box, frac: float, color, track=None) -> None:
    x0, y0, x1, y1 = (int(v) for v in box)
    r = (y1 - y0) // 2
    rrect(img, box, r, track or C["raised"])
    frac = max(0.0, min(1.0, frac if frac == frac else 0.0))
    if frac > 0:
        rrect(img, (x0, y0, max(x0 + 2 * r, int(x0 + (x1 - x0) * frac)), y1), r, color)


def _clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def _mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    if s >= 3600:
        return _clock(s)
    return f"{s // 60:02d}:{s % 60:02d}"


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
    fitted = cv2.resize(frame, (fitted_w, fitted_h), interpolation=interpolation)
    preview = np.full((height, width, 3), C["bg"], np.uint8)
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


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

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
    language: str = "zh"               # zh | en，執行中按 L 切換
    # 以下由 layout_from_config() 從 config 其他區塊帶入
    wrist_side: str = "right"
    roi: Optional[Sequence[float]] = None        # (x_min, y_min, x_max, y_max)，0–1
    device_label: str = ""


def _device_label(cfg: dict) -> str:
    inf = cfg.get("inference", {})
    backend = str(inf.get("backend", ""))
    if backend == "hailo":
        return "Hailo-10H · UGen300 M.2"
    if backend == "ultralytics":
        return f"Ultralytics · {str(inf.get('device', 'cpu')).upper()}"
    if backend == "replay":
        return "Replay · keypoints JSON"
    return backend


def layout_from_config(cfg: dict) -> LayoutConfig:
    """從整份 config 組出 LayoutConfig（visualize 區塊 + 穿刺側 + ROI + 裝置）。"""
    layout = LayoutConfig(**{k: v for k, v in cfg.get("visualize", {}).items()
                             if k in LayoutConfig.__dataclass_fields__})
    layout.wrist_side = cfg.get("features", {}).get("wrist_side", layout.wrist_side)
    roi = cfg.get("subject", {}).get("roi")
    if isinstance(roi, dict) and layout.roi is None:
        try:
            layout.roi = (float(roi["x_min"]), float(roi["y_min"]),
                          float(roi["x_max"]), float(roi["y_max"]))
        except (KeyError, TypeError, ValueError):
            pass
    if not layout.device_label:
        layout.device_label = _device_label(cfg)
    return layout


def thresholds_from_config(cfg: dict) -> dict[str, float]:
    sm = cfg.get("state_machine", {})
    out = {"enter": sm.get("enter_threshold", 0.15),
           "exit": sm.get("exit_threshold", 0.10)}
    for key in ("t_enter", "t_confirm", "t_exit", "cooldown"):
        if key in sm:
            out[key] = sm[key]
    return out


# --------------------------------------------------------------------------
# 床邊畫面
# --------------------------------------------------------------------------

@dataclass
class _Ctx:
    """一格的顯示脈絡。由 _track() 算好，各區塊共用。"""
    ts: float
    elapsed: float
    mode: str
    value: float
    slope: float
    building: bool
    degraded: bool


class DemoRenderer:
    """Draw a clinical-facing bedside frame from existing pipeline fields."""

    def __init__(self, layout: Optional[LayoutConfig] = None,
                 thresholds: Optional[dict[str, float]] = None,
                 primary_feature: str = "arm_displacement"):
        self.L = layout or LayoutConfig()
        self.thresholds = {"enter": 0.15, "exit": 0.10, "t_enter": 1.0,
                           "t_confirm": 5.0, "t_exit": 3.0, "cooldown": 30.0}
        self.thresholds.update(thresholds or {})
        self.primary = primary_feature
        self.side = self.L.wrist_side if self.L.wrist_side in ARM else "right"
        self._history: deque[tuple[float, float]] = deque(
            maxlen=max(2, int(self.L.curve_span_s * 60)))
        self._timeline_events: list[tuple[float, str]] = []
        self._event_count = 0
        self._start_time: Optional[float] = None
        self._above_since: Optional[float] = None
        self._mode = "normal"
        self._mode_since = 0.0
        self._alert_peak = 0.0
        self._runs: list[list] = []                       # [起, 迄, mode]
        self._fast_flash_until = -1.0
        self._fast_was_verifying = False
        self._glitches = 0
        self._base_acc = {int(k): [0.0, 0.0, 0] for k in ARM[self.side]}
        self._base_arm: Optional[dict[int, tuple[float, float]]] = None
        self._brightness: Optional[float] = None
        self._arm_vis = 0.0
        self._fps: Optional[float] = None
        self._infer_ms: Optional[float] = None
        self._last_wall: Optional[float] = None
        self._cpu: Optional[float] = None
        self._cpu_t = 0.0
        self.language = "en"
        self.set_language(self.L.language)

    # -- 語言 -----------------------------------------------------------

    def set_language(self, language: str) -> str:
        """Use English if the optional CJK font stack is unavailable."""
        if language not in ("en", "zh"):
            raise ValueError("language must be 'en' or 'zh'")
        self.language = language if language == "en" or _CJK_FONT else "en"
        return self.language

    def toggle_language(self) -> str:
        return self.set_language("zh" if self.language == "en" else "en")

    def _t(self, key: str, **values) -> str:
        en, zh = T[key]
        values.setdefault("arm", ARM_WORD[(self.side, self.language)])
        return (zh if self.language == "zh" else en).format(**values)

    # -- 主流程 ---------------------------------------------------------

    def render(self, obs: FrameObservation, features: dict[str, float],
               state: State, new_event: Optional[Event] = None,
               timestamp: Optional[float] = None, color_cfg=None,
               baseline_progress: Optional[tuple[float, float]] = None,
               fusion: Optional[FusionResult] = None,
               fast_event: Optional[Event] = None,
               baseline_ready: Optional[bool] = None,
               fast_verifying: Optional[bool] = None) -> np.ndarray:
        ctx = self._track(obs, features, state, fusion, timestamp, new_event,
                          fast_event, baseline_progress, baseline_ready,
                          fast_verifying)
        W, H = self.L.width, self.L.height
        canvas = _solid(H, W, C["bg"]).copy()
        p = Painter(canvas)

        pad, gap, head_h, foot_h = 16, 10, 56, 74
        foot = (pad, H - 12 - foot_h, W - pad, H - 12)
        top, bot = head_h + gap, foot[1] - gap
        video_w = max(320, min(self.L.video_w, W - 2 * pad - gap - 380))
        left = (pad, top, pad + video_w, bot)
        rx0, rx1 = left[2] + gap, W - pad
        status = (rx0, top, rx1, top + 128)
        cards_y0 = bot - 150
        mid = (rx0 + rx1 - gap) // 2
        chart = (rx0, status[3] + gap, rx1, cards_y0 - gap)

        self._draw_header(p, ctx, obs)
        self._draw_video(p, obs, left, ctx, fusion, color_cfg)
        self._draw_status(p, status, ctx, fusion, baseline_progress)
        self._draw_chart(p, chart, ctx, features)
        self._draw_votes(p, (rx0, cards_y0, mid, bot), ctx, fusion, features)
        self._draw_channels(p, (mid + gap, cards_y0, rx1, bot), ctx, fast_verifying)
        self._draw_footer(p, foot, ctx, obs)
        p.flush()
        return canvas

    def _track(self, obs, features, state, fusion, timestamp, new_event,
               fast_event, baseline_progress, baseline_ready,
               fast_verifying) -> _Ctx:
        """更新跨格的記憶（歷史曲線、狀態時段、事件、基準手臂），回傳這一格的脈絡。"""
        ts = obs.timestamp if timestamp is None else timestamp
        if self._start_time is None:
            self._start_time = ts
            self._mode_since = ts
        elapsed = max(0.0, ts - self._start_time)
        value = features.get(self.primary, float("nan"))

        building = (baseline_progress is not None
                    and (not baseline_ready if baseline_ready is not None
                         else baseline_progress[1] > 0
                         and baseline_progress[0] < baseline_progress[1]))
        if baseline_ready is False:
            building = True

        if np.isfinite(value) and not building:
            self._history.append((ts, value))
            self._above_since = ((self._above_since if self._above_since is not None else ts)
                                 if value > self.thresholds["enter"] else None)

        mode = self._resolve_mode(state, fusion, building)
        if mode != self._mode:
            self._mode, self._mode_since = mode, ts
            self._alert_peak = value if np.isfinite(value) else 0.0
        if mode == "alert" and np.isfinite(value):
            self._alert_peak = max(self._alert_peak, value)
        if self._runs and self._runs[-1][2] == mode:
            self._runs[-1][1] = ts
        else:
            if self._runs:
                self._runs[-1][1] = ts
            self._runs.append([ts, ts, mode])

        for event in (new_event, fast_event):
            if event is not None:
                self._event_count += 1
                self._timeline_events.append((event.start_time, event.channel))
        if fast_event is not None:
            self._fast_flash_until = ts + 4.0
        if fast_verifying is not None:
            if self._fast_was_verifying and not fast_verifying and fast_event is None:
                self._glitches += 1           # 驗證中途放棄 = 擋下一次偵測跳點
            self._fast_was_verifying = bool(fast_verifying)

        # 治療開始時的手臂位置：基準建立期間平均；沒有基準資訊時取前 3 秒
        arm = ARM[self.side]
        capturing = building or (baseline_ready is None and baseline_progress is None
                                 and elapsed < 3.0)
        if capturing and self._base_arm is None:
            for k in arm:
                kp = obs.keypoints[int(k)]
                if kp.is_valid:
                    acc = self._base_acc[int(k)]
                    acc[0] += kp.x
                    acc[1] += kp.y
                    acc[2] += 1
        elif self._base_arm is None and all(a[2] >= 5 for a in self._base_acc.values()):
            self._base_arm = {k: (a[0] / a[2], a[1] / a[2])
                              for k, a in self._base_acc.items()}

        kps = [obs.keypoints[int(k)] for k in arm]
        self._arm_vis = sum(kp.confidence if kp.is_valid else 0.0 for kp in kps) / len(kps)

        if obs.raw_frame is not None:
            small = cv2.resize(obs.raw_frame, (24, 24), interpolation=cv2.INTER_AREA)
            b = float(small.mean())
            self._brightness = b if self._brightness is None else .9 * self._brightness + .1 * b
        else:
            self._brightness = None

        now = time.perf_counter()
        if self._last_wall is not None and now > self._last_wall:
            inst = 1.0 / (now - self._last_wall)
            self._fps = inst if self._fps is None else .9 * self._fps + .1 * inst
        self._last_wall = now
        infer = obs.meta.get("infer_ms") if obs.meta else None
        if isinstance(infer, (int, float)):
            self._infer_ms = (infer if self._infer_ms is None
                              else .9 * self._infer_ms + .1 * infer)
        if psutil is not None and now - self._cpu_t > 1.0:
            self._cpu, self._cpu_t = psutil.cpu_percent(None), now

        return _Ctx(ts=ts, elapsed=elapsed, mode=mode, value=value,
                    slope=self._slope(ts), building=building,
                    degraded=(fusion is not None
                              and fusion.reliability is Reliability.DEGRADED))

    def _track_result(self, result: "FrameResult") -> _Ctx:
        return self._track(result.obs, result.display_features, result.state,
                           result.fusion, result.obs.timestamp, result.new_event,
                           result.fast_event, result.baseline_progress,
                           result.baseline_ready,
                           getattr(result, "fast_verifying", None))

    @staticmethod
    def _resolve_mode(state: State, fusion: Optional[FusionResult],
                      building: bool) -> str:
        # ★ 已確認的告警優先於「看不到」—— 開著的告警不能被遮蔽蓋掉。
        if building:
            return "baseline"
        if state is State.ALERT:
            return "alert"
        if fusion is not None and fusion.reliability is Reliability.UNRELIABLE:
            return "unreliable"
        return {State.WATCHING: "watching",
                State.COOLDOWN: "cooldown"}.get(state, "normal")

    def _slope(self, ts: float, window: float = 10.0) -> float:
        """最近 window 秒的線性斜率（每秒）。資料不足回傳 0。"""
        pts = [(t, v) for t, v in self._history if t >= ts - window]
        if len(pts) < 8 or pts[-1][0] - pts[0][0] < 2.0:
            return 0.0
        t = np.array([q[0] for q in pts])
        v = np.array([q[1] for q in pts])
        t = t - t.mean()
        denom = float((t * t).sum())
        return float((t * (v - v.mean())).sum() / denom) if denom > 0 else 0.0

    def _eta(self, ctx: _Ctx) -> Optional[float]:
        """照目前斜率，幾秒後越過 enter 門檻。只是顯示用的外推。"""
        enter = self.thresholds["enter"]
        if (ctx.mode not in ("normal", "watching") or not np.isfinite(ctx.value)
                or ctx.value >= enter or ctx.slope <= 1e-4
                or ctx.value < self.thresholds["exit"] * .5):
            return None
        eta = (enter - ctx.value) / ctx.slope
        return eta if 0 < eta < 120 else None

    # -- 頁首 -----------------------------------------------------------

    def _draw_header(self, p: Painter, ctx: _Ctx, obs: FrameObservation) -> None:
        img, W = p.img, self.L.width
        cv2.line(img, (0, 56), (W, 56), C["line"], 1)
        x = 20
        bed = self.L.bed_label.upper()
        p.text(bed, (x, 38), C["text"], 26, "display")
        x += p.width(bed, 26, "display") + 14
        site = self._t("site")
        p.text(site, (x, 36), C["muted"], 14)
        x += p.width(site, 14) + 10
        clock = _clock(ctx.elapsed)
        p.text(clock, (x, 36), C["text"], 16, "mono")
        x += p.width(clock, 16, "mono") + 26

        for i, (key, color) in enumerate((("normal", C["teal"]),
                                          ("watching", C["amber"]),
                                          ("alert", C["red"]))):
            label = key.upper()
            pw = p.width(label, 12, "bold") + 28
            box = (x, 15, x + pw, 41)
            active = ctx.mode == key or (key == "normal" and ctx.mode == "cooldown")
            if active:
                rrect(img, box, 13, color)
                p.text(label, (x + pw // 2, 28), C["bg"], 12, "bold", "mm")
            else:
                rrect_alpha(img, box, 13, color, .40, 1)
                p.text(label, (x + pw // 2, 28), _mix(color, C["bg"], .45), 12, "bold", "mm")
            x += pw
            if i < 2:
                cv2.line(img, (x + 5, 28), (x + 17, 28), C["line"], 1, cv2.LINE_AA)
                x += 22

        rx = W - 20
        wall = datetime.now().strftime("%H:%M:%S")
        p.text(wall, (rx, 36), C["text"], 18, "mono", "rs")
        rx -= p.width(wall, 18, "mono") + 18

        if obs.raw_frame is None:
            light = ("no_video", C["red"])
        elif self._brightness is not None and self._brightness < 45:
            light = ("light_low", C["amber"])
        elif self._brightness is not None and self._brightness > 225:
            light = ("light_high", C["amber"])
        else:
            light = ("light_ok", C["teal"])
        found = (obs.meta.get("subject_found", True) if obs.meta else True) and any(
            kp.is_valid for kp in obs.keypoints)
        cam = ("cam_ok", C["teal"]) if found else ("cam_lost", C["amber"])
        for key, color in (("local", C["teal"]), light, cam):
            text = self._t(key)
            cw = p.width(text, 12) + 32
            if rx - cw < x + 12:
                break
            box = (rx - cw, 15, rx, 41)
            rrect(img, box, 13, C["raised"])
            rrect(img, box, 13, C["line"], 1)
            cv2.circle(img, (box[0] + 14, 28), 3, color, -1, cv2.LINE_AA)
            p.text(text, (box[0] + 23, 28), C["text"], 12, "sans", "lm")
            rx -= cw + 8

    # -- 影像 -----------------------------------------------------------

    def _draw_video(self, p: Painter, obs: FrameObservation, box, ctx: _Ctx,
                    fusion: Optional[FusionResult], color_cfg=None,
                    compact: bool = False) -> None:
        img = p.img
        x0, y0, x1, y1 = (int(v) for v in box)
        w, h = x1 - x0, y1 - y0
        view = _solid(h, w, C["panel"]).copy()
        fit = None
        if obs.raw_frame is not None:
            src = obs.raw_frame
            if self.L.show_roi_overlay and color_cfg is not None:
                from .temporal.color import roi_overlay
                overlaid = roi_overlay(obs, color_cfg)
                if overlaid is not None:
                    src = overlaid
            sh, sw = src.shape[:2]
            # 背景：同一張畫面裁成面板比例、縮到極小再放大（等於重度模糊）、壓暗，
            # 填滿直式影片兩側的空白
            cover = max(w / sw, h / sh)
            cw, ch = min(sw, int(w / cover)), min(sh, int(h / cover))
            cx0, cy0 = (sw - cw) // 2, (sh - ch) // 2
            tiny = cv2.resize(src[cy0:cy0 + ch, cx0:cx0 + cw],
                              (max(1, w // 24), max(1, h // 24)),
                              interpolation=cv2.INTER_AREA)
            back = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
            view = cv2.addWeighted(back, .20, view, .80, 0)
            scale = min(w / sw, h / sh)
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            ox, oy = (w - nw) // 2, (h - nh) // 2
            view[oy:oy + nh, ox:ox + nw] = cv2.resize(
                src, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            fit = (x0 + ox, y0 + oy, nw, nh)
        img[y0:y1, x0:x1] = view

        color = MODE_COLOR[ctx.mode]
        if fit is not None:
            fx, fy, fw, fh = fit

            def px(x, y):
                return int(fx + x * fw), int(fy + y * fh)

            if self.L.roi is not None and not compact:
                rx0, ry0 = px(self.L.roi[0], self.L.roi[1])
                rx1, ry1 = px(self.L.roi[2], self.L.roi[3])
                roi_col = _mix(C["teal"], C["panel"], .35)
                for a, b in (((rx0, ry0), (rx1, ry0)), ((rx1, ry0), (rx1, ry1)),
                             ((rx1, ry1), (rx0, ry1)), ((rx0, ry1), (rx0, ry0))):
                    dashed_line(img, a, b, roi_col, 1, 8, 6)
                p.text(self._t("roi"), (rx0 + 2, ry0 - 8), roi_col, 12)

            self._draw_skeleton(img, obs, px, color, ctx)
            if not compact:
                self._draw_callout(p, obs, px, color, ctx, (x0, y0, x1, y1))

        if obs.raw_frame is None:
            p.text(self._t("video_unavailable"), ((x0 + x1) // 2, (y0 + y1) // 2),
                   C["muted"], 16, "sans", "mm")

        if ctx.mode == "unreliable":
            self._badge(p, ((x0 + x1) // 2, y0 + h // 2), self._t("badge_occluded"),
                        C["unrel"])
        elif ctx.building and not compact:
            self._badge(p, ((x0 + x1) // 2, y0 + 34), self._t("badge_baseline"),
                        C["base"])

        if not compact:
            # 左上：LIVE 與姿態信心
            cx = x0 + 12
            for text, dot in ((self._t("live"), C["red"]),
                              (self._t("pose_conf", v=self._arm_vis), None)):
                cw = p.width(text, 12, "bold" if dot else "sans") + (34 if dot else 22)
                chip = (cx, y0 + 12, cx + cw, y0 + 38)
                rrect_alpha(img, chip, 13, C["bg"], .72)
                if dot:
                    cv2.circle(img, (cx + 14, y0 + 25), 4, dot, -1, cv2.LINE_AA)
                    p.text(text, (cx + 24, y0 + 25), C["text"], 12, "bold", "lm")
                else:
                    p.text(text, (cx + 11, y0 + 25), C["text"], 12, "sans", "lm")
                cx += cw + 8
            # 左下：圖例
            if self._base_arm is not None and fit is not None:
                a, b = self._t("legend_base"), self._t("legend_now")
                lw = 26 + p.width(a, 12) + 22 + 26 + p.width(b, 12) + 16
                chip = (x0 + 12, y1 - 40, x0 + 12 + lw, y1 - 14)
                rrect_alpha(img, chip, 13, C["bg"], .72)
                lx, ly = chip[0] + 12, (chip[1] + chip[3]) // 2
                dashed_line(img, (lx, ly), (lx + 16, ly), C["ghost"], 2, 4, 3)
                p.text(a, (lx + 24, ly), C["muted"], 12, "sans", "lm")
                lx += 24 + p.width(a, 12) + 22
                cv2.line(img, (lx, ly), (lx + 16, ly), color, 3, cv2.LINE_AA)
                p.text(b, (lx + 24, ly), C["muted"], 12, "sans", "lm")

        round_corners(img, (x0, y0, x1, y1), 14, C["bg"])
        accented = ctx.mode in ("watching", "alert", "unreliable")
        if ctx.mode == "alert":
            pulse = .5 + .5 * math.sin(ctx.ts * 2 * math.pi / 1.4)
            rrect_alpha(img, (x0 - 3, y0 - 3, x1 + 2, y1 + 2), 17, color,
                        .25 + .35 * pulse, 2)
        rrect(img, (x0, y0, x1 - 1, y1 - 1), 14, color if accented else C["line"],
              2 if accented else 1)

    def _draw_skeleton(self, img, obs: FrameObservation, px, color, ctx: _Ctx) -> None:
        kps = obs.keypoints
        arm = [int(k) for k in ARM[self.side]]
        arm_pairs = {(arm[0], arm[1]), (arm[1], arm[2])}
        bone = C["teal"]

        # 治療開始時的手臂（虛線）
        wrist_px = None
        if self._base_arm is not None and not ctx.building:
            pts = [px(*self._base_arm[k]) for k in arm]
            for a, b in zip(pts, pts[1:]):
                dashed_line(img, a, b, C["ghost"], 2, 6, 5)
            cv2.circle(img, pts[2], 8, C["ghost"], 2, cv2.LINE_AA)
            w = kps[arm[2]]
            if w.is_valid:
                wrist_px = px(w.x, w.y)
                dashed_line(img, pts[2], wrist_px, color, 1, 3, 4)

        # 頭：鼻子為圓心，兩耳距離決定大小；再連到肩膀中點
        nose = kps[int(KP.NOSE)]
        ls, rs = kps[int(KP.LEFT_SHOULDER)], kps[int(KP.RIGHT_SHOULDER)]
        if nose.is_valid:
            le, re = kps[int(KP.LEFT_EAR)], kps[int(KP.RIGHT_EAR)]
            c = px(nose.x, nose.y)
            r = 12
            if le.is_valid and re.is_valid:
                a, b = px(le.x, le.y), px(re.x, re.y)
                r = int(max(9, min(40, math.hypot(a[0] - b[0], a[1] - b[1]) * .62)))
            cv2.circle(img, c, r, C["bg"], 5, cv2.LINE_AA)
            cv2.circle(img, c, r, bone, 2, cv2.LINE_AA)
            if ls.is_valid and rs.is_valid:
                m = px((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
                cv2.line(img, c, m, C["bg"], 5, cv2.LINE_AA)
                cv2.line(img, c, m, bone, 2, cv2.LINE_AA)

        for a, b in SKELETON:
            ka, kb = kps[int(a)], kps[int(b)]
            if not (ka.is_valid and kb.is_valid):
                continue
            pa, pb = px(ka.x, ka.y), px(kb.x, kb.y)
            on_arm = (int(a), int(b)) in arm_pairs or (int(b), int(a)) in arm_pairs
            cv2.line(img, pa, pb, C["bg"], 7 if on_arm else 5, cv2.LINE_AA)
            cv2.line(img, pa, pb, color if on_arm else bone, 4 if on_arm else 2,
                     cv2.LINE_AA)
        for i, kp in enumerate(kps):
            if kp.is_valid and i not in (int(KP.LEFT_EYE), int(KP.RIGHT_EYE),
                                         int(KP.LEFT_EAR), int(KP.RIGHT_EAR), int(KP.NOSE)):
                on_arm = i in arm
                cv2.circle(img, px(kp.x, kp.y), 5 if on_arm else 3,
                           color if on_arm else bone, -1, cv2.LINE_AA)
        w = kps[arm[2]]
        if w.is_valid:
            c = px(w.x, w.y)
            rrect_alpha(img, (c[0] - 15, c[1] - 15, c[0] + 15, c[1] + 15), 15, color, .30, 2)
            cv2.circle(img, c, 8, color, -1, cv2.LINE_AA)

    def _draw_callout(self, p: Painter, obs, px, color, ctx: _Ctx, bounds) -> None:
        """手腕旁的偏離量標籤。基準還沒建好時不顯示 —— 那時沒有「偏離」可言。"""
        w = obs.keypoints[int(ARM[self.side][2])]
        # 看不到的時候也不顯示 —— 那個數字不可信，標在畫面上反而誤導
        if (ctx.building or ctx.mode == "unreliable" or not w.is_valid
                or not np.isfinite(ctx.value)):
            return
        cx, cy = px(w.x, w.y)
        arrow = "↑" if ctx.slope > 0.002 else "↓" if ctx.slope < -0.002 else "→"
        label = self._t("callout")
        val = f"{ctx.value:.2f} {arrow}"
        bw = max(p.width(label, 12), p.width(val, 22, "mono_bold")) + 26
        bh = 56
        x0, y0, x1, y1 = bounds
        bx = cx + 26 if cx + 26 + bw < x1 - 10 else cx - 26 - bw
        by = min(max(cy + 6, y0 + 50), y1 - bh - 48)
        box = (bx, by, bx + bw, by + bh)
        rrect_alpha(p.img, box, 10, C["bg"], .84)
        rrect(p.img, box, 10, _mix(color, C["bg"], .25), 1)
        p.text(label, (bx + 13, by + 21), C["muted"], 12)
        p.text(val, (bx + 13, by + 46), color, 22, "mono_bold")

    def _badge(self, p: Painter, center, text: str, color) -> None:
        tw = p.width(text, 14, "bold")
        cx, cy = center
        box = (cx - tw // 2 - 18, cy - 18, cx + tw // 2 + 18, cy + 18)
        rrect_alpha(p.img, box, 18, C["bg"], .85)
        rrect(p.img, box, 18, color, 1)
        p.text(text, (cx, cy), color, 14, "bold", "mm")

    # -- 狀態卡 ---------------------------------------------------------

    def _card(self, img, box, accent=None, pulse: float = 0.0) -> None:
        rrect(img, box, 14, C["panel"])
        if accent is None:
            rrect(img, box, 14, C["line"], 1)
        else:
            rrect_alpha(img, box, 14, accent, .07)
            rrect_alpha(img, box, 14, accent, .45 + .35 * pulse, 1)

    def _draw_status(self, p: Painter, box, ctx: _Ctx,
                     fusion: Optional[FusionResult], baseline_progress) -> None:
        img = p.img
        x0, y0, x1, y1 = box
        mode, color = ctx.mode, MODE_COLOR[ctx.mode]
        th = self.thresholds
        pulse = (.5 + .5 * math.sin(ctx.ts * 2 * math.pi / 1.4)) if mode == "alert" else 0
        self._card(img, box, color if mode in ("watching", "alert", "unreliable") else None,
                   pulse)

        p.text(self._t("word_" + mode), (x0 + 22, y0 + 50), color, 36, "display")
        desc = self._t("desc_" + mode)
        if ctx.degraded and mode not in ("unreliable", "baseline"):
            desc += self._t("degraded")
        p.text(desc, (x1 - 22, y0 + 42),
               C["text"] if mode in ("watching", "alert", "unreliable") else C["muted"],
               15, "sans", "rs")

        ry = y0 + 82
        value = ctx.value
        frac, right, label = None, "", ""
        note = ""
        extra = ""
        if mode == "baseline":
            done, need = baseline_progress or (0.0, 0.0)
            label = self._t("lbl_progress")
            frac = done / need if need > 0 else 0.0
            right = f"{done:.1f} / {need:.1f} s"
            note = self._t("note_baseline_wait" if frac >= 1.0 else "note_baseline")
        elif mode == "watching":
            label = self._t("lbl_confirm")
            if self._above_since is not None:
                done = max(0.0, ctx.ts - self._above_since - th["t_enter"])
                frac = done / th["t_confirm"] if th["t_confirm"] > 0 else 1.0
                note = self._t("note_watching", enter=th["enter"], t=th["t_confirm"])
            else:
                done, frac = 0.0, 0.0
                note = self._t("note_falling", enter=th["enter"])
            right = f"{min(done, th['t_confirm']):.1f} / {th['t_confirm']:.1f} s"
        elif mode == "alert":
            label = self._t("lbl_elapsed")
            right = f"{value:.2f}" if np.isfinite(value) else "—"
            extra = (f"{_mmss(ctx.ts - self._mode_since)}   ·   "
                     + self._t("peak", v=self._alert_peak))
            note = self._t("note_alert", exit=th["exit"], t=th["t_exit"])
        elif mode == "cooldown":
            label = self._t("lbl_cool")
            done = ctx.ts - self._mode_since
            frac = done / th["cooldown"] if th["cooldown"] > 0 else 1.0
            right = f"{min(done, th['cooldown']):.0f} / {th['cooldown']:.0f} s"
            note = self._t("note_cooldown")
        elif mode == "unreliable":
            label = self._t("lbl_conf")
            conf = fusion.confidence if fusion is not None else 0.0
            frac = conf
            right = f"{conf:.2f}"
            note = self._t("note_unreliable")
        else:
            label = self._t("lbl_dev")
            frac = value / th["enter"] if np.isfinite(value) and th["enter"] > 0 else 0.0
            right = (f"{value:.2f} / {th['enter']:.2f}" if np.isfinite(value) else "—")
            note = self._t("note_normal", enter=th["enter"], t=th["t_confirm"])

        p.text(label, (x0 + 22, ry), C["muted"], 13, "sans", "lm")
        lw = p.width(label, 13) + 16
        p.text(right, (x1 - 22, ry), color, 16, "mono_bold", "rm")
        rw = p.width(right, 16, "mono_bold") + 18
        if frac is not None:
            bar_col = color
            if mode == "normal" and frac > .66:
                bar_col = C["amber"]
            progress_bar(img, (x0 + 22 + lw, ry - 4, x1 - 22 - rw, ry + 4), frac, bar_col)
        elif extra:
            p.text(extra, (x0 + 22 + lw, ry), C["text"], 15, "mono", "lm")
        p.text(note, (x0 + 22, y1 - 17), C["muted"], 13)

    # -- 趨勢圖 ---------------------------------------------------------

    def _draw_chart(self, p: Painter, box, ctx: _Ctx, features) -> None:
        img = p.img
        x0, y0, x1, y1 = box
        self._card(img, box)
        span = self.L.curve_span_s
        p.text(self._t("chart_title", s=span), (x0 + 20, y0 + 30), C["text"], 14, "bold")
        eta = self._eta(ctx)
        if eta is not None:
            p.text(self._t("predict", s=eta), (x1 - 20, y0 + 30), C["amber"], 13, "sans", "rs")
        elif np.isfinite(ctx.value) and not ctx.building:
            p.text(self._t("now_val", v=ctx.value), (x1 - 20, y0 + 30), C["muted"], 13,
                   "mono", "rs")

        th = self.thresholds
        px0, px1 = x0 + 18, x1 - 82
        py0, py1 = y0 + 50, y1 - 30
        future = span * .22
        t_lo, t_hi = ctx.ts - span, ctx.ts + future
        visible = [(t, v) for t, v in self._history if t >= t_lo]
        vmax = max([th["enter"] * 1.6] + [v * 1.15 for _, v in visible])

        def X(t):
            return int(px0 + (t - t_lo) / (t_hi - t_lo) * (px1 - px0))

        def Y(v):
            return int(py1 - max(0.0, min(1.0, v / vmax)) * (py1 - py0))

        now_x = X(ctx.ts)
        blend_rect(img, (now_x, py0, px1, py1), C["raised"], .55)
        band_top = Y(th["exit"])
        blend_rect(img, (px0, band_top, now_x, py1), C["teal"], .07)
        p.text(self._t("band"), (px0 + 8, py1 - 8), _mix(C["teal"], C["panel"], .35), 11)
        cv2.line(img, (px0, py1), (px1, py1), C["line"], 1)
        for key, color in (("enter", C["red"]), ("exit", C["amber"])):
            yy = Y(th[key])
            dashed_line(img, (px0, yy), (px1, yy), _mix(color, C["panel"], .25), 1, 6, 5)
            p.text(f"{key} {th[key]:.2f}", (x1 - 16, yy), color, 11, "mono", "rm")
        cv2.line(img, (now_x, py0), (now_x, py1), C["line"], 1)

        if len(visible) >= 2 and not ctx.building:
            pts = np.array([[X(t), Y(v)] for t, v in visible], np.int32)
            fill_under(img, pts, py1, C["text"], .10)
            cv2.polylines(img, [pts], False, C["text"], 2, cv2.LINE_AA)
            last = tuple(int(c) for c in pts[-1])
            if eta is not None:
                cross_t = ctx.ts + eta
                if cross_t <= t_hi:
                    end = (X(cross_t), Y(th["enter"]))
                else:
                    end = (px1, Y(ctx.value + ctx.slope * future))
                dashed_line(img, last, end, C["amber"], 2, 4, 4)
                if cross_t <= t_hi:
                    cv2.circle(img, end, 5, C["amber"], 2, cv2.LINE_AA)
            color = MODE_COLOR[ctx.mode]
            rrect_alpha(img, (last[0] - 9, last[1] - 9, last[0] + 9, last[1] + 9), 9,
                        color, .35)
            cv2.circle(img, last, 5, color, -1, cv2.LINE_AA)

        for t, label in ((t_lo, f"-{span:g}s"), (ctx.ts - span / 2, f"-{span / 2:g}s"),
                         (ctx.ts, "now")):
            p.text(label, (X(t), y1 - 11), C["dim"], 11, "mono",
                   "ls" if t == t_lo else "ms")
        if self.L.debug:
            torso = features.get("torso_angle", float("nan"))
            if np.isfinite(torso):
                p.text(f"torso {torso:.1f}", (x0 + 20, y0 + 46), C["dim"], 11, "mono")

    # -- 訊號投票 ／ 雙通道 ---------------------------------------------

    def _draw_votes(self, p: Painter, box, ctx: _Ctx,
                    fusion: Optional[FusionResult], features) -> None:
        img = p.img
        x0, y0, x1, y1 = box
        self._card(img, box)
        p.text(self._t("votes_title"), (x0 + 18, y0 + 30), C["text"], 14, "bold")
        conf = ("—" if fusion is None or ctx.building else f"{fusion.confidence:.2f}")
        p.text(self._t("conf", v=conf), (x1 - 18, y0 + 30), C["muted"], 13, "sans", "rs")

        def ramp(v, lo, hi):
            return 0.0 if not np.isfinite(v) else max(0.0, min(1.0, (v - lo) / (hi - lo)))

        arm_v = features.get(self.primary, float("nan"))
        torso_v = features.get("torso_angle", float("nan"))
        votes = fusion.votes if fusion is not None else {}
        rows = [
            (self._t("sig_arm"),
             votes.get(self.primary, ramp(arm_v, self.thresholds["exit"],
                                          self.thresholds["enter"] * 1.67)),
             f"{arm_v:.2f}" if np.isfinite(arm_v) else "—", None),
            (self._t("sig_torso"), votes.get("torso_angle", ramp(torso_v, 8.0, 30.0)),
             f"{torso_v:.0f}°" if np.isfinite(torso_v) else "—", None),
            (self._t("sig_vis"), self._arm_vis, f"{self._arm_vis:.2f}",
             C["teal"] if self._arm_vis >= .5 else C["unrel"]),
        ]
        if ctx.building:
            rows = [(r[0], 0.0, "—", r[3]) for r in rows[:2]] + rows[2:]
        lw = max(p.width(r[0], 13) for r in rows) + 12
        for i, (label, frac, text, col) in enumerate(rows):
            ry = y0 + 64 + i * 29
            if col is None:
                col = C["red"] if frac >= .66 else C["amber"] if frac >= .33 else C["teal"]
            p.text(label, (x0 + 18, ry), C["text"], 13, "sans", "lm")
            progress_bar(img, (x0 + 18 + lw, ry - 3, x1 - 66, ry + 3), frac, col)
            p.text(text, (x1 - 18, ry), col, 13, "mono", "rm")

    def _draw_channels(self, p: Painter, box, ctx: _Ctx,
                       fast_verifying: Optional[bool]) -> None:
        img = p.img
        x0, y0, x1, y1 = box
        self._card(img, box)
        p.text(self._t("ch_title"), (x0 + 18, y0 + 30), C["text"], 14, "bold")

        slow = {"normal": ("st_idle", None), "watching": ("st_watch", C["amber"]),
                "alert": ("st_alert", C["red"]), "cooldown": ("st_cool", C["cool"]),
                "unreliable": ("st_hold", C["unrel"]),
                "baseline": ("st_base", C["base"])}[ctx.mode]
        if ctx.ts < self._fast_flash_until:
            fast = ("st_fired", C["red"])
        elif fast_verifying:
            fast = ("st_verify", C["amber"])
        else:
            fast = ("st_fast_idle", None)

        for i, (label, (key, accent)) in enumerate(((self._t("ch_slow"), slow),
                                                    (self._t("ch_fast"), fast))):
            ry0 = y0 + 46 + i * 42
            row = (x0 + 12, ry0, x1 - 12, ry0 + 34)
            if accent is None:
                rrect(img, row, 9, C["raised"])
            else:
                rrect_alpha(img, row, 9, accent, .14)
                rrect_alpha(img, row, 9, accent, .55, 1)
            cy = ry0 + 17
            p.text(label, (row[0] + 12, cy), C["text"], 13, "sans", "lm")
            p.text(self._t(key), (row[2] - 12, cy), accent or C["teal"], 13, "bold", "rm")
        if self._glitches:
            p.text(self._t("glitches", n=self._glitches), (x0 + 18, y1 - 10),
                   C["muted"], 11)

    # -- 頁尾 -----------------------------------------------------------

    def _draw_footer(self, p: Painter, box, ctx: _Ctx, obs: FrameObservation) -> None:
        img = p.img
        x0, y0, x1, y1 = box
        self._card(img, box)

        # 第一列：本次治療時間軸
        ly = y0 + 24
        label = self._t("session")
        p.text(label, (x0 + 18, ly), C["muted"], 13, "sans", "lm")
        bx0 = x0 + 18 + max(p.width(label, 13), 56) + 14
        span_txt = f"00:00:00 → {_clock(ctx.elapsed)}"
        p.text(span_txt, (x1 - 18, ly), C["muted"], 12, "mono", "rm")
        bx1 = x1 - 18 - p.width(span_txt, 12, "mono") - 16
        bw = max(1, bx1 - bx0)
        strip = _solid(10, bw, C["raised"]).copy()
        t0, span = self._start_time or 0.0, max(ctx.elapsed, 1.0)

        def sx(t):
            return int(max(0, min(bw, (t - t0) / span * bw)))

        for s, e, mode in self._runs:
            a = min(sx(s), bw - 1)
            b = max(a + 1, sx(e))
            strip[:, a:b] = RUN_COLOR[mode]
            if mode == "unreliable":
                hatch(strip, (a, 0, b, 10), C["unrel"], 4)
        img[ly - 5:ly + 5, bx0:bx0 + bw] = strip
        round_corners(img, (bx0, ly - 5, bx0 + bw, ly + 5), 5, C["panel"])
        for t, channel in self._timeline_events:
            ex = bx0 + sx(t)
            col = C["red"] if channel == "fast" else C["amber"]
            cv2.line(img, (ex, ly - 10), (ex, ly + 10), C["panel"], 4)
            cv2.line(img, (ex, ly - 10), (ex, ly + 10), col, 2, cv2.LINE_AA)

        # 第二列：裝置與效能（事件細節看時間軸上的刻度與右側雙通道卡）
        dy = y0 + 52
        ix = x0 + 18
        rrect(img, (ix, dy - 6, ix + 12, dy + 6), 2, C["teal"], 1)
        for k in (3, 6, 9):
            cv2.line(img, (ix + k, dy - 9), (ix + k, dy - 7), C["teal"], 1)
            cv2.line(img, (ix + k, dy + 7), (ix + k, dy + 9), C["teal"], 1)
        device = self.L.device_label or str((obs.meta or {}).get("backend", "—"))
        p.text(device, (ix + 22, dy), C["text"], 12, "bold", "lm")
        cx = ix + 22 + p.width(device, 12, "bold") + 18
        stats = []
        if self._fps is not None:
            stats.append(f"{self._fps:.0f} FPS")
        if self._infer_ms is not None:
            stats.append(self._t("infer", v=self._infer_ms))
        if self._cpu is not None:
            stats.append(f"CPU {self._cpu:.0f}%")
        if self.L.debug:
            stats.append(f"{ctx.mode} {ctx.value:.3f}" if np.isfinite(ctx.value)
                         else ctx.mode)
        for s in stats:
            p.text(s, (cx, dy), C["muted"], 12, "mono", "lm")
            cx += p.width(s, 12, "mono") + 18
        keys = self._t("keys")
        p.text(keys, (x1 - 18, dy), C["dim"], 12, "sans", "rm")
        p.text(self._t("events_n", n=self._event_count),
               (x1 - 18 - p.width(keys, 12) - 24, dy), C["muted"], 12, "sans", "rm")


# --------------------------------------------------------------------------
# 同片對照
# --------------------------------------------------------------------------

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
        base_ctx = self.baseline._track_result(baseline_result)
        ours_ctx = self.ours._track_result(ours_result)

        canvas = _solid(self.height, self.width, C["bg"]).copy()
        p = Painter(canvas)
        t = self.ours._t
        cv2.line(canvas, (0, 56), (self.width, 56), C["line"], 1)
        p.text("DIALYSIGHT", (20, 37), C["text"], 22, "display")
        p.text(t("cmp_title"), (20 + p.width("DIALYSIGHT", 22, "display") + 16, 36),
               C["muted"], 15)
        p.text(_clock(ours_ctx.elapsed), (self.width - 20, 36), C["text"], 18, "mono", "rs")

        self._side(p, (16, 68, 632, 708), self.baseline, baseline_result, base_ctx,
                   self.baseline_alerts, "cmp_base", "cmp_base_sub", C["red"],
                   baseline_open=baseline_open)
        self._side(p, (648, 68, 1264, 708), self.ours, ours_result, ours_ctx,
                   self.ours_alerts, "cmp_ours", "cmp_ours_sub", C["teal"])
        p.flush()
        return canvas

    def _side(self, p: Painter, box, renderer: DemoRenderer, result, ctx: _Ctx,
              count: int, title_key: str, sub_key: str, accent,
              baseline_open: bool = False) -> None:
        img = p.img
        x0, y0, x1, y1 = box
        t = self.ours._t
        rrect(img, box, 16, C["panel"])
        rrect(img, box, 16, C["line"], 1)
        p.text(t(title_key), (x0 + 20, y0 + 34), C["text"], 17, "bold")
        p.text(t(sub_key), (x0 + 20, y0 + 57), C["muted"], 13)
        unit = t("cmp_alerts")
        p.text(unit, (x1 - 20, y0 + 52), C["muted"], 14, "sans", "rs")
        p.text(str(count), (x1 - 20 - p.width(unit, 14) - 8, y0 + 54), accent, 48,
               "display", "rs")

        renderer._draw_video(p, result.obs, (x0 + 12, y0 + 74, x1 - 12, y1 - 62),
                             ctx, result.fusion, compact=True)

        if ctx.building:
            word, reason, color = (t("word_baseline"), t("desc_baseline"), C["base"])
        elif baseline_open:
            word, reason, color = (t("word_alert"), t("cmp_fired"), C["red"])
        elif ctx.ts < renderer._fast_flash_until:
            word, reason, color = (t("word_alert"), t("cmp_sudden"), C["red"])
        else:
            word, reason, color = (t("word_" + ctx.mode), t("desc_" + ctx.mode),
                                   MODE_COLOR[ctx.mode])
        sy = y1 - 30
        cv2.circle(img, (x0 + 26, sy), 6, color, -1, cv2.LINE_AA)
        p.text(word, (x0 + 42, sy), color, 22, "display", "lm")
        p.text(reason, (x0 + 42 + p.width(word, 22, "display") + 14, sy), C["muted"],
               14, "sans", "lm")


def make_writer(path: str, fps: float, layout: LayoutConfig) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (layout.width, layout.height))
