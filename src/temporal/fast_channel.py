"""
快通道（雙通道設計的另一半）—— 偵測「突發」而非「漸進」。

★ 為什麼需要兩條通道，而不是把閾值調低就好？

  慢通道（state_machine.py）要求超標持續 t_confirm = 5 秒才告警。
  這個設計是對的 —— 它就是我們壓掉 94.7% 偽陽性的原因。

  但它對一種真實風險是瞎的：**瞬間的大動作**。

      病人突然坐起、手臂猛然抽離、整個人往床緣滑

  這類事件的危險不在於「持續」，而在於「幅度」。
  等 5 秒確認完，針可能已經脫落了。

  把慢通道的 t_confirm 調短來涵蓋它？那偽陽性會整個回來，
  因為 5 秒確認正是擋住雜訊尖峰的機制。

  所以答案是兩條通道，各自用不同的物理量判斷：

      慢通道    看「偏離基準的量」持續多久        → 漸進性風險
      快通道    看「短時間內變化的幅度」是否站住   → 突發性風險

  兩者共用同一份特徵流，互不干擾，各自產生 channel 標記不同的 Event。

★ 快通道真正難的地方，不是偵測大動作，是分辨「大動作」與「偵測跳點」。

  YOLOv8-pose 偶爾會把手腕認到別的地方 —— 被子的皺摺、旁邊的點滴架。
  這在數值上就是一格之內座標暴衝，跟「病人猛然抽手」長得一模一樣。

  單看振幅，兩者無法區分。唯一的差別在之後那一秒：

      真實動作    手臂移到新位置後，**停在那裡**
      偵測跳點    下一格就跳回原位

  所以快通道偵測到振幅後不立刻告警，而是進入「驗證位移」階段：
  要求新位置維持 t_persist 秒。撐住的是真動作，彈回去的是跳點。

  被擋下的次數記在 glitch_rejected 裡 —— 這個數字可以直接給評審看，
  它是「我們知道單幀推論會出錯，而且有對策」的證據。

時間軸示意：

    value   ────────╱‾‾‾‾‾‾‾‾‾‾‾‾   真動作：跳上去，停住 → 告警
                   ╱
                   ↑ 振幅超標，開始驗證
                   └─ t_persist ─┘

    value   ────────╱╲──────────    跳點：跳上去，彈回來 → 丟棄
                   ╱  ╲
                   ↑ 振幅超標，開始驗證
                       ↑ 位移沒撐住，glitch_rejected += 1
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .state_machine import Event

NAN = float("nan")


@dataclass
class FastChannelConfig:
    """快通道參數。

    ★ 跟慢通道的參數不是同一個量綱，不要互相比較：
      慢通道的 enter_threshold 是「偏離基準多少」，
      快通道的 amplitude_threshold 是「window_s 秒內變化多少」。
    """
    enabled: bool = True

    window_s: float = 1.0             # 振幅觀察窗。太長會把慢速移動也算進來
    amplitude_threshold: float = 0.12 # 窗內 (max − min) 超過此值就啟動驗證
    min_samples: int = 5              # 窗內有效樣本數下限，太少不判定

    # ★ 參考值（跳變前的位置）取自振幅窗「之前」的這一段，不是窗內前半段。
    #
    #   原本取窗內前半段的中位數。那會出錯，而且錯的方式很隱密：
    #   觸發時機偏晚時（例如前一次事件的冷卻期剛結束），
    #   振幅窗已經橫跨跳變點，前半段的中位數就落在跳變前後之間 ——
    #   參考值被拉高 → signed 位移算小 → **真實的持續位移被判成偵測跳點**。
    #
    #   實測：手臂在 t=55 跳到新位置並一路維持到 t=70，
    #   卻因為參考值被污染成 0.2269（真值 0.13）而進了 glitch_rejected。
    #
    #   這個數字我們要給評審看，裡面摻真實事件是不能接受的。
    #   改成從振幅窗之前取樣，參考值在結構上就不可能橫跨這次跳變。
    ref_window_s: float = 2.0         # 參考區長度，緊接在振幅窗之前
    min_ref_samples: int = 4          # 參考區樣本數下限，不足就不觸發

    t_persist: float = 0.8            # 位移要維持這麼久才算真事件
    persist_ratio: float = 0.6        # 維持期間位移至少要保住原振幅的幾成
    cooldown: float = 15.0            # 告警後的靜默期，比慢通道短

    # 快通道只看這一個特徵。多特徵融合是 fusion.py 的事，不要混在這裡。
    feature: str = "arm_displacement"


@dataclass
class _Pending:
    """正在驗證位移的候選事件。"""
    triggered_at: float
    reference: float       # 跳變前的位置（窗內前半段中位數）
    target: float          # 跳變後的位置
    amplitude: float
    peak: float
    direction: float       # ★ +1 往外跳、−1 往內跳。見 _check_persist()


class FastChannel:
    """突發性變化偵測。

    用法（在主迴圈裡每格呼叫一次）：
        fc = FastChannel(cfg)
        event = fc.update(timestamp, value)     # value 是平滑後的特徵值
        if event is not None:
            alert(event)      # event.channel == "fast"

    ⚠️ 餵進來的 value 應該是**特徵本身**（例如 arm_displacement），
       不是「相對基準的偏離量」。快通道看的是短期變化，
       跟基準線無關，基準還沒建立好的時候它就可以開始工作 ——
       這也是它比慢通道早上線的原因。
    """

    def __init__(self, cfg: Optional[FastChannelConfig] = None):
        self.cfg = cfg or FastChannelConfig()

        self._buf: deque[tuple[float, float]] = deque()
        self._pending: Optional[_Pending] = None
        self._cooldown_until: float = -1e9
        # 丟棄一個候選後的短暫抑制期。見 _check_persist() 的說明。
        self._suppress_until: float = -1e9

        self.events: list[Event] = []
        self._glitch_rejected = 0
        self._triggers = 0

    # ------------------------------------------------------------------

    def update(self, timestamp: float, value: float) -> Optional[Event]:
        """餵入一筆特徵值。

        Returns:
            剛確認的突發事件（只在確認的那一格回傳），其他時候 None。
        """
        if not self.cfg.enabled:
            return None

        if value is None or value != value:      # NaN：關鍵點缺失
            # 不推進也不重置。驗證中的候選留著 ——
            # 短暫遮蔽不代表位移沒發生，但也不能當作撐住了，
            # 所以下面的 _check_persist 用「有效樣本」判斷而非單純時間。
            return None

        self._buf.append((timestamp, value))
        # 緩衝區要同時裝得下參考區與振幅窗
        cutoff = timestamp - (self.cfg.window_s + self.cfg.ref_window_s)
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if self._pending is not None:
            return self._check_persist(timestamp, value)

        if timestamp < self._cooldown_until or timestamp < self._suppress_until:
            return None

        return self._check_trigger(timestamp)

    # ------------------------------------------------------------------

    def _check_trigger(self, now: float) -> None:
        """看振幅窗內的振幅有沒有超標。超標就開始驗證，不直接告警。

        緩衝區切成兩段：

            [now − window_s − ref_window_s, now − window_s)   參考區
            [now − window_s,                now          ]   振幅窗

        參考值只從參考區取 —— 它整段都在這次跳變之前，不可能被污染。

        ⚠️ 已知限制：如果跳變發生在 window_s 之前（振幅窗已經整段都是
           跳變後的值），窗內振幅會趨近 0，這裡就不會觸發。

           這是刻意不處理的。快通道的價值在於 0.8 秒內反應；
           超過 window_s 才發現就已經失去那個優勢，而「位置改變且維持」
           本來就是慢通道的職責，它會在約 6 秒後確認。
           兩條通道都漏掉的情況不存在。
        """
        split = now - self.cfg.window_s
        ref_vals = [v for t, v in self._buf if t < split]
        amp_vals = [v for t, v in self._buf if t >= split]

        if len(amp_vals) < self.cfg.min_samples:
            return None
        # 參考區樣本不足（剛開始跑、或剛從遮蔽恢復）→ 沒有乾淨的基準可比，不觸發
        if len(ref_vals) < self.cfg.min_ref_samples:
            return None

        amplitude = max(amp_vals) - min(amp_vals)
        if amplitude < self.cfg.amplitude_threshold:
            return None

        reference = float(np.median(ref_vals))
        target = amp_vals[-1]

        # 相對乾淨參考值的位移。太小代表振幅是窗內來回抖動而非位置改變，
        # 那種東西交給慢通道的遲滯處理，快通道不該介入。
        if abs(target - reference) < self.cfg.amplitude_threshold * self.cfg.persist_ratio:
            return None

        vals = amp_vals

        self._pending = _Pending(
            triggered_at=now,
            reference=reference,
            target=target,
            amplitude=amplitude,
            peak=max(abs(v - reference) for v in vals),
            # ★ 記住跳的方向。驗證時要求位移維持在**同一個方向**。
            direction=1.0 if target >= reference else -1.0,
        )
        self._triggers += 1
        return None

    def _check_persist(self, now: float, value: float) -> Optional[Event]:
        """驗證階段：位移有沒有站住。

        ★ 這裡比的是**有號**位移，不是絕對值。原本只比大小，有個 bug：

          跳點結束、手臂回到原位之後，振幅窗的前半段裝的是跳點本身，
          reference 被污染成「高位」。此時當前值（已回原位）離 reference
          很遠 —— 絕對值一看就過關，於是「回到原位」被讀成
          「位移站住了」，跳點反而被確認成真事件。

          加上方向判斷後，回原位是反方向，signed 會變成負值，直接丟棄。

        ★ 丟棄之後還要抑制 window_s 秒。
          因為振幅窗裡還留著那段跳點，不抑制的話同一個擾動會從另一側
          再觸發一次驗證。一次擾動只該產生一個判斷。
        """
        p = self._pending
        assert p is not None

        # 有號位移：往原本跳的方向算正
        signed = (value - p.reference) * p.direction
        p.peak = max(p.peak, abs(value - p.reference))

        # 彈回去了（或反向跑了）→ 偵測跳點，丟棄
        if signed < p.amplitude * self.cfg.persist_ratio:
            self._pending = None
            self._glitch_rejected += 1
            self._suppress_until = now + self.cfg.window_s
            return None

        # 撐住夠久 → 確認
        if now - p.triggered_at >= self.cfg.t_persist:
            ev = Event(
                feature=self.cfg.feature,
                start_time=p.triggered_at,
                end_time=now,
                peak_value=p.peak,
                confirmed_at=now,
                channel="fast",
                meta={
                    "outcome": "confirmed",
                    "amplitude": round(p.amplitude, 4),
                    "latency_s": round(now - p.triggered_at, 2),
                },
            )
            self.events.append(ev)
            self._pending = None
            self._cooldown_until = now + self.cfg.cooldown
            return ev

        return None

    # -- 查詢 -----------------------------------------------------------

    @property
    def is_verifying(self) -> bool:
        """正在驗證位移。UI 可以用這個顯示一個短暫的提示。"""
        return self._pending is not None

    def confirmed_events(self) -> list[Event]:
        return list(self.events)

    def report(self) -> dict:
        """★ glitch_rejected 是這個模組最值得給評審看的數字。

        它量化了「單幀推論會出錯」這件事在我們的資料上發生了幾次，
        以及我們擋掉了幾次。沒有這個數字，「我們處理了偵測跳點」
        只是一句話。
        """
        return {
            "channel": "fast",
            "triggers": self._triggers,
            "confirmed": len(self.events),
            "glitch_rejected": self._glitch_rejected,
            "reject_rate": round(self._glitch_rejected / self._triggers, 3)
            if self._triggers else 0.0,
            "mean_latency_s": round(
                float(np.mean([e.meta["latency_s"] for e in self.events])), 2
            ) if self.events else None,
        }

    def reset(self) -> None:
        self._buf.clear()
        self._pending = None
        self._cooldown_until = -1e9
        self._suppress_until = -1e9
        self.events.clear()
        self._glitch_rejected = 0
        self._triggers = 0
