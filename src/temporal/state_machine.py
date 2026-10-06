"""
狀態機（第 4 層）—— 把連續的數字變成離散的事件。

★ 這是整個系統偽陽性率的決定層，也是我們跟官方 demo 的分界線。

官方 demo 的作法：
    if value > threshold:  alert()

這會發生什麼？病人翻個身、護理師走過、調整枕頭，全部觸發。
一個會亂叫的系統，護理師三天後就會把它關掉 ——
這是所有臨床監測系統的真實死因。

我們的作法，四個機制疊加：

  1. 遲滯（hysteresis）
     進入與離開用不同閾值。避免特徵在閾值邊界附近來回跳動時，
     狀態跟著反覆切換。

  2. 持續時間要求
     超標必須持續 T_confirm 秒才算真事件。濾掉瞬時雜訊。

  3. WATCHING 中間狀態
     已經超標但還沒確認的期間，記錄但不告警。
     事後可以分析「哪些差點觸發」—— 這是調參數的重要依據。

  4. 冷卻期
     告警後一段時間內不重複告警，避免同一事件轟炸護理站。

時間軸示意：

    value  ──────╱‾‾‾‾‾‾‾‾╲────────
                ╱          ╲
    enter  ────┼───────────┼───────
    exit   ──┼─────────────┼─┼─────
             │ │           │ │
    state  NORMAL│WATCHING │ALERT│COOLDOWN│NORMAL
                 └T_confirm┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    NORMAL = "normal"        # 一切正常
    WATCHING = "watching"    # 已超標，等待確認（不告警）
    ALERT = "alert"          # 已確認的事件（告警）
    COOLDOWN = "cooldown"    # 事件結束後的靜默期


@dataclass
class Event:
    """一次完整的事件紀錄。

    這個物件會被 report.py 拿去填進範本，產生床次紀錄。
    不要在這裡做文字格式化 —— 狀態機只負責判斷，文字交給 report.py。
    """
    feature: str
    start_time: float
    end_time: Optional[float] = None
    peak_value: float = 0.0
    confirmed_at: Optional[float] = None    # 從 WATCHING 升級到 ALERT 的時間
    channel: str = "slow"                   # "fast" 或 "slow"
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_open(self) -> bool:
        return self.end_time is None


@dataclass
class StateMachineConfig:
    """狀態機參數。

    ★ 這些值不是工程能自己定的，要跟臨床端討論。
      見提案討論稿「需要跟臨床端討論的參數」。

    調參原則：先用保守值，寧可漏報也不要洗版。
    """
    enter_threshold: float = 0.15   # 進入 WATCHING 的閾值
    exit_threshold: float = 0.10    # 回到 NORMAL 的閾值（必須 < enter）
    t_enter: float = 1.0            # 超過 enter 多久後進 WATCHING（秒）
    t_confirm: float = 5.0          # WATCHING 持續多久後升級 ALERT（秒）
    t_exit: float = 3.0             # 低於 exit 多久後離開 ALERT（秒）
    cooldown: float = 30.0          # 告警後的靜默期（秒）

    def __post_init__(self) -> None:
        if self.exit_threshold >= self.enter_threshold:
            raise ValueError(
                "exit_threshold 必須小於 enter_threshold，否則遲滯失效。"
                f"收到 enter={self.enter_threshold}, exit={self.exit_threshold}"
            )


class HysteresisStateMachine:
    """單一特徵的遲滯狀態機。

    用法：
        sm = HysteresisStateMachine("arm_displacement", cfg)
        for each frame:
            event = sm.update(timestamp, value)
            if event is not None:
                # 新確認的事件，送去告警
                alert(event)

    update() 只在「事件剛被確認」的那一格回傳 Event，其他時候回傳 None。
    想知道當前狀態用 sm.state，想拿所有事件用 sm.events。
    """

    def __init__(self, feature: str, config: StateMachineConfig, channel: str = "slow"):
        self.feature = feature
        self.cfg = config
        self.channel = channel

        self.state = State.NORMAL
        self.events: list[Event] = []

        self._state_entered_at: Optional[float] = None   # 進入當前狀態的時間
        self._above_since: Optional[float] = None        # 連續超過 enter 的起點
        self._below_since: Optional[float] = None        # 連續低於 exit 的起點
        self._current: Optional[Event] = None
        self._cooldown_until: float = -1.0

    # ------------------------------------------------------------------

    def update(self, timestamp: float, value: float) -> Optional[Event]:
        """餵入一筆特徵值，推進狀態機。

        Returns:
            剛被確認的 Event（只在 WATCHING → ALERT 那一格回傳），
            其他時候回傳 None。
        """
        if value is None or value != value:   # NaN 檢查
            # 關鍵點缺失。不推進狀態，但也不重置 ——
            # 短暫遮蔽不應該讓已經累積的 WATCHING 狀態歸零。
            return None

        above_enter = value > self.cfg.enter_threshold
        below_exit = value < self.cfg.exit_threshold

        # 追蹤連續超標／連續低標的起點
        self._above_since = timestamp if (above_enter and self._above_since is None) else (
            None if not above_enter else self._above_since
        )
        self._below_since = timestamp if (below_exit and self._below_since is None) else (
            None if not below_exit else self._below_since
        )

        if self.state is State.NORMAL:
            return self._tick_normal(timestamp, value)
        if self.state is State.WATCHING:
            return self._tick_watching(timestamp, value)
        if self.state is State.ALERT:
            return self._tick_alert(timestamp, value)
        if self.state is State.COOLDOWN:
            return self._tick_cooldown(timestamp, value)
        return None

    # -- 各狀態的處理 ---------------------------------------------------

    def _tick_normal(self, ts: float, value: float) -> None:
        if self._above_since is None:
            return None
        if ts - self._above_since >= self.cfg.t_enter:
            self._transition(State.WATCHING, ts)
            self._current = Event(
                feature=self.feature,
                start_time=self._above_since,
                peak_value=value,
                channel=self.channel,
            )
        return None

    def _tick_watching(self, ts: float, value: float) -> Optional[Event]:
        assert self._current is not None
        self._current.peak_value = max(self._current.peak_value, value)

        # 掉回 exit 以下且持續夠久 → 誤報，取消。不產生告警。
        if self._below_since is not None and ts - self._below_since >= self.cfg.t_exit:
            self._current.end_time = ts
            self._current.meta["outcome"] = "dismissed_in_watching"
            self.events.append(self._current)     # 仍然記錄，供事後調參分析
            self._current = None
            self._transition(State.NORMAL, ts)
            return None

        # 持續超標夠久 → 升級為正式事件
        if self._above_since is not None and ts - self._above_since >= (
            self.cfg.t_enter + self.cfg.t_confirm
        ):
            self._current.confirmed_at = ts
            # ★ 在「確認的當下」就入列，不要等到事件結束才入列。
            #
            #   原本的寫法是在 _tick_alert 解除時才 append，結果是：
            #   影片／治療結束時狀態還停在 ALERT 的那一筆，永遠不會進 events，
            #   confirmed_events() 看不到它，床次紀錄也不會有它。
            #
            #   而那恰好是最不能漏的一筆 —— 監測結束時還在告警中的事件。
            #
            #   events 裡放的是同一個物件參照，之後 _tick_alert 設 end_time
            #   會直接反映到清單裡這一筆上，不需要重新 append。
            self.events.append(self._current)
            self._transition(State.ALERT, ts)
            return self._current      # ★ 唯一回傳 Event 的地方

        return None

    def _tick_alert(self, ts: float, value: float) -> None:
        assert self._current is not None
        self._current.peak_value = max(self._current.peak_value, value)

        if self._below_since is not None and ts - self._below_since >= self.cfg.t_exit:
            # 這一筆已經在 events 裡了（確認時就入列），直接改它的欄位
            self._current.end_time = ts
            self._current.meta["outcome"] = "resolved"
            self._current = None
            self._cooldown_until = ts + self.cfg.cooldown
            self._transition(State.COOLDOWN, ts)
        return None

    def finalize(self, now: float) -> Optional[Event]:
        """治療／影片結束時呼叫，收掉還開著的事件。

        ★ 結束時仍在 ALERT 的事件要標記成 open_at_session_end，
          不要標成 resolved —— 它沒有被解除，只是我們不再看了。
          這兩件事在臨床紀錄上意義完全不同。

        Returns:
            被收掉的那筆事件，沒有的話 None。
        """
        if self._current is None:
            return None

        ev = self._current
        ev.end_time = now
        if ev.confirmed_at is not None:
            ev.meta["outcome"] = "open_at_session_end"
        else:
            # 還在 WATCHING 就結束了，沒有升級成告警
            ev.meta["outcome"] = "watching_at_session_end"
            self.events.append(ev)
        self._current = None
        return ev

    def _tick_cooldown(self, ts: float, value: float) -> None:
        if ts >= self._cooldown_until:
            self._transition(State.NORMAL, ts)
        return None

    def _transition(self, new_state: State, ts: float) -> None:
        self.state = new_state
        self._state_entered_at = ts

    # -- 查詢 -----------------------------------------------------------

    @property
    def is_alerting(self) -> bool:
        return self.state is State.ALERT

    def time_in_state(self, now: float) -> float:
        if self._state_entered_at is None:
            return 0.0
        return now - self._state_entered_at

    def confirmed_events(self) -> list[Event]:
        """只回傳真正告警過的事件，不含 WATCHING 階段被取消的。

        包含「結束時還開著」的那一筆 —— 它確認過，只是沒等到解除。
        """
        return [e for e in self.events if e.confirmed_at is not None]

    def open_event(self) -> Optional[Event]:
        """當前還沒結束的事件（WATCHING 或 ALERT 中）。"""
        return self._current

    def dismissed_count(self) -> int:
        """WATCHING 階段被取消的次數。

        這個數字很有價值 —— 它就是「如果用單閾值，會多叫幾次」的下界。
        對照實驗可以直接引用。
        """
        return sum(
            1 for e in self.events
            if e.meta.get("outcome") == "dismissed_in_watching"
        )

    def reset(self) -> None:
        self.state = State.NORMAL
        self.events.clear()
        self._state_entered_at = None
        self._above_since = None
        self._below_since = None
        self._current = None
        self._cooldown_until = -1.0


class SingleThresholdStateMachine:
    """對照組：單一閾值，無遲滯、無確認時間、無冷卻。

    ★ 這就是 baseline.py 要用的東西，模擬官方 demo 的作法。

    偽陽性對照實驗就是跑同一段影片，比較這個跟 HysteresisStateMachine
    各自告警幾次。如果數字是 30 對 3，那張圖比任何論述都有力。
    """

    def __init__(self, feature: str, threshold: float):
        self.feature = feature
        self.threshold = threshold
        self.events: list[Event] = []
        self._was_above = False

    def update(self, timestamp: float, value: float) -> Optional[Event]:
        if value is None or value != value:
            return None

        is_above = value > self.threshold

        # 上升緣觸發（不然每一格都會叫）
        if is_above and not self._was_above:
            ev = Event(
                feature=self.feature,
                start_time=timestamp,
                confirmed_at=timestamp,
                peak_value=value,
                channel="baseline",
            )
            self.events.append(ev)
            self._was_above = True
            return ev

        if not is_above:
            self._was_above = False
            if self.events and self.events[-1].is_open:
                self.events[-1].end_time = timestamp

        return None

    def finalize(self, now: float) -> Optional[Event]:
        """對照組的事件在觸發當下就入列了，這裡只補上結束時間。"""
        if self.events and self.events[-1].is_open:
            self.events[-1].end_time = now
            self.events[-1].meta["outcome"] = "open_at_session_end"
            return self.events[-1]
        return None

    def confirmed_events(self) -> list[Event]:
        return list(self.events)

    def reset(self) -> None:
        self.events.clear()
        self._was_above = False
