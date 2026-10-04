"""
事件紀錄輸出（B7）—— 範本填空，不用 LLM。

★ 為什麼不用 LLM：
  臨床紀錄要的是可稽核、可重現、每次都一樣的文字。
  LLM 的自然語言變異在這裡是缺點不是優點。
  而且計數、排序、時間換算、分組全部是確定性邏輯，範本就能做。

  這也是我們不需要 GenAI 模型的原因 —— 床邊只跑 vision，
  完全避開「LLM 服務獨佔鎖定裝置」的限制。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from .temporal.state_machine import Event

FEATURE_LABEL = {
    "arm_displacement": "上肢姿態異常",
    "torso_angle": "軀幹角度變化",
    "arm_depth_delta": "深度方向位移異常",
    "bed_occupancy": "離床偵測",
}


def _clock(start: Optional[datetime], offset_s: float) -> str:
    if start is None:
        mm, ss = divmod(int(offset_s), 60)
        return f"{mm:02d}:{ss:02d}"
    return (start + timedelta(seconds=offset_s)).strftime("%H:%M")


def format_event(ev: Event, session_start: Optional[datetime] = None) -> str:
    label = FEATURE_LABEL.get(ev.feature, ev.feature)
    when = _clock(session_start, ev.start_time)
    if ev.duration is not None:
        return f"{when}  {label}（持續 {int(ev.duration)} 秒）"
    return f"{when}  {label}"


def session_report(
    events: list[Event],
    session_duration_s: float,
    bed_label: str = "Bed 07",
    session_start: Optional[datetime] = None,
) -> str:
    """產生一次治療的事件紀錄。

    輸出是護理師交班時直接可用的文字，不是框框座標。
    """
    lines = [
        f"# 透析治療事件紀錄 — {bed_label}",
        "",
        f"治療時長：{session_duration_s / 3600:.1f} 小時",
        f"事件總數：{len(events)}",
        "",
    ]

    if not events:
        lines.append("本次治療未偵測到異常事件。")
        return "\n".join(lines)

    lines.append("## 事件清單")
    lines.append("")
    for ev in sorted(events, key=lambda e: e.start_time):
        lines.append(f"- {format_event(ev, session_start)}")

    lines.append("")
    lines.append("## 趨勢摘要")
    lines.append("")
    lines.append(_trend_summary(events, session_duration_s))
    return "\n".join(lines)


def _trend_summary(events: list[Event], duration_s: float) -> str:
    """分布分析。純規則，沒有 LLM。"""
    if not events:
        return "無事件。"

    half = duration_s / 2
    first = sum(1 for e in events if e.start_time < half)
    second = len(events) - first

    if second > first * 2 and second >= 2:
        where = "集中於治療後半段"
    elif first > second * 2 and first >= 2:
        where = "集中於治療前半段"
    else:
        where = "分布於整段治療"

    total = sum(e.duration for e in events if e.duration is not None)
    return (
        f"本次治療共 {len(events)} 次事件，{where}"
        f"（前半 {first} 次、後半 {second} 次），"
        f"異常狀態累計約 {int(total)} 秒。"
    )
