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
    "roi_redness": "穿刺處滲血疑慮",
}

# ★ 兩條通道在臨床上的意義不同，紀錄要分得出來。
#   「漸進」是慢慢偏移累積出來的，「突發」是一瞬間的大動作。
#   交班時護理師需要知道是哪一種。
CHANNEL_LABEL = {
    "slow": "漸進",
    "fast": "突發",
    "baseline": "單閾值",
}


def _clock(start: Optional[datetime], offset_s: float) -> str:
    """事件發生時間。

    有治療開始時間就印實際鐘面時間（交班時用的是這個）。
    沒有的話印「治療開始後第幾時幾分幾秒」—— 超過一小時要進位成
    h:mm:ss，不然四小時的治療會印出「153:20」，看起來像壞掉。
    """
    if start is not None:
        return (start + timedelta(seconds=offset_s)).strftime("%H:%M")

    total = int(offset_s)
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    if hh:
        return f"{hh}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _duration_text(seconds: float) -> str:
    """持續時間的文字。

    不要對不足一秒的事件寫「持續 0 秒」—— 那看起來像沒發生。
    快通道的事件本來就很短，那是它的設計，不是錯誤。
    """
    if seconds < 1.0:
        return "不足 1 秒"
    if seconds < 90:
        return f"{int(seconds)} 秒"
    return f"{seconds / 60:.1f} 分鐘"


def format_event(ev: Event, session_start: Optional[datetime] = None) -> str:
    label = FEATURE_LABEL.get(ev.feature, ev.feature)
    when = _clock(session_start, ev.start_time)
    chan = CHANNEL_LABEL.get(ev.channel, ev.channel)

    parts = [f"{when}  {label}（{chan}"]
    if ev.duration is not None:
        d = _duration_text(ev.duration)
        parts.append(f"，{d}" if d.startswith("不足") else f"，持續 {d}")
    parts.append("）")

    # ★ 治療結束時還在進行中的事件要標出來。
    #   它不是「解除」，只是我們不再看了 —— 這兩件事不能混在一起。
    if ev.meta.get("outcome") == "open_at_session_end":
        parts.append("　⚠ 治療結束時仍持續")

    return "".join(parts)


def _session_length(seconds: float) -> str:
    """治療時長。

    一次透析通常 4 小時，但測試用的影片只有幾分鐘 ——
    用「0.0 小時」顯示兩分鐘的測試片段會讓人以為程式壞了。
    """
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分鐘"
    return f"{seconds / 3600:.1f} 小時"


def session_report(
    events: list[Event],
    session_duration_s: float,
    bed_label: str = "Bed 07",
    session_start: Optional[datetime] = None,
) -> str:
    """產生一次治療的事件紀錄。

    輸出是護理師交班時直接可用的文字，不是框框座標。
    """
    n_slow = sum(1 for e in events if e.channel == "slow")
    n_fast = sum(1 for e in events if e.channel == "fast")

    lines = [
        f"# 透析治療事件紀錄 — {bed_label}",
        "",
        f"治療時長：{_session_length(session_duration_s)}",
        f"事件總數：{len(events)}"
        + (f"（漸進 {n_slow}、突發 {n_fast}）" if n_fast else ""),
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
