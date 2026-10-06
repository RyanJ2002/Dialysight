"""
把各層串起來的主管線。

    影格 → 推論 → 過濾 → 特徵 ─┬→ 平滑 → 基準偏離 → 滑動窗 ─┐
                               │                              ├→ 融合 → 慢通道 ─┐
                               ├→ 顏色（滲血）────────────────┘                 ├→ 事件 → sink
                               └→ 快通道（突發）───────────────────────────────┘

★ 這個檔案是「膠水」，不要在這裡放演算法邏輯。
  每一層的邏輯都在各自的模組裡，這裡只負責串接與資料流。

★ 兩條通道共用前面的推論與特徵抽取，只在第 4 層分岔：

    慢通道  dev → windows → fusion → HysteresisStateMachine   漸進性風險
    快通道  feats →────────────────→ FastChannel              突發性風險

  共用前段是硬體層面的關鍵 —— 推論只跑一次。
  兩條通道加起來的額外成本是幾十微秒的純量運算，
  不是第二次模型推論。這就是「時序層幾乎免費」的來源。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional

from .fusion import ConsensusFusion, FusionConfig, FusionResult, Reliability
from .inference.base import FrameObservation, make_backend
from .sink import BedStatus, EventSink, StatusReporter, event_payload, make_sink
from .temporal.color import BleedingMonitor, ColorConfig
from .temporal.fast_channel import FastChannel, FastChannelConfig
from .temporal.features import (
    BaselineConfig, BaselineTracker, FeatureConfig, FeatureExtractor, VelocityTracker,
)
from .temporal.filter import FilterConfig, KeypointFilter
from .temporal.smooth import SmoothConfig, SmoothingBank
from .temporal.state_machine import (
    Event, HysteresisStateMachine, SingleThresholdStateMachine,
    State, StateMachineConfig,
)
from .temporal.window import FeatureWindows

NAN = float("nan")


def _check_module_versions() -> None:
    """確認各模組都是同一批的版本。

    ★ 存在的理由：五個人用檔案互傳同步 repo 時，很容易只更新一部分。
      少了某個檔案的話原本會噴 AttributeError，看不出是檔案沒更新 ——
      有人會以為程式壞了，去改不該改的地方。

      這裡用「需要的屬性在不在」判斷，直接講清楚是哪個檔案舊了。
    """
    need = [
        (BaselineConfig, "from_dict", "src/temporal/features.py"),
        (FeatureExtractor, "visibilities", "src/temporal/features.py"),
        (HysteresisStateMachine, "finalize", "src/temporal/state_machine.py"),
    ]
    stale = sorted({path for obj, attr, path in need if not hasattr(obj, attr)})
    if stale:
        raise RuntimeError(
            "以下檔案是舊版，跟 pipeline.py 不同批：\n  "
            + "\n  ".join(stale)
            + "\n\n整組改動請一起更新（或直接 git pull）。"
              "更新完跑 `python scripts/selftest.py` 確認，應該是 54 項全綠。"
        )


@dataclass
class FrameResult:
    """單一影格跑完整條管線後的結果。視覺化與報告都吃這個。"""
    obs: FrameObservation
    features: dict[str, float]          # 平滑後的特徵值（絕對值）
    deviations: dict[str, float]        # 相對基準的偏離量 ★ 狀態機吃的是這個
    state: State
    new_event: Optional[Event] = None           # 慢通道剛確認的事件
    fast_event: Optional[Event] = None          # ★ 快通道剛確認的事件
    baseline_ready: bool = True
    baseline_progress: tuple[float, float] = (0.0, 0.0)   # (已過秒, 需要秒)
    window_stats: dict = field(default_factory=dict)
    color: dict[str, float] = field(default_factory=dict) # 滲血訊號
    fusion: Optional[FusionResult] = None                 # 融合結果

    @property
    def any_event(self) -> Optional[Event]:
        """兩條通道任一剛觸發的事件。視覺化用這個就好。"""
        return self.new_event or self.fast_event

    @property
    def display_features(self) -> dict[str, float]:
        """給 visualize 的合併字典（偏離量 + 顏色訊號）。"""
        return {**self.deviations, **self.color}


class Pipeline:
    """完整的處理管線。

    用法：
        pipe = Pipeline(cfg)
        for result in pipe.run("data/clips/clip_03.mp4"):
            canvas = renderer.render(result.obs, result.display_features,
                                     result.state, result.any_event)
        pipe.close()            # ★ 一定要呼叫，sink 要 flush

    Args:
        cfg: 讀進來的 yaml
        use_baseline_machine: 對照組模式，用單閾值狀態機
        enable_sink: 是否輸出事件。對照實驗與單元測試傳 False，
                     免得在 results/ 下寫出一堆檔案
        fps_hint: 影片的實際 FPS，只用於換算舊版 config 的 baseline.n_frames
    """

    def __init__(
        self,
        cfg: dict,
        use_baseline_machine: bool = False,
        enable_sink: bool = True,
        fps_hint: float = 10.0,
    ):
        _check_module_versions()

        self.cfg = cfg
        self.backend = make_backend(cfg)

        self.filter = KeypointFilter(FilterConfig(**cfg.get("filter", {})))

        fcfg = FeatureConfig(**cfg.get("features", {}))
        self.extractor = FeatureExtractor(fcfg)
        self.velocity = VelocityTracker(
            [n for n in fcfg.enabled if n in ("arm_displacement", "torso_angle")]
        )

        self.baseline = BaselineTracker(
            BaselineConfig.from_dict(cfg.get("baseline", {}), fps_hint=fps_hint)
        )

        scfg = SmoothConfig(
            default_method=cfg.get("smooth", {}).get("default", {}).get("method", "median"),
            default_window=cfg.get("smooth", {}).get("default", {}).get("window", 7),
            per_feature=cfg.get("smooth", {}).get("per_feature", {}),
        )
        self.smoother = SmoothingBank(scfg, self.extractor.names)

        self.windows = FeatureWindows(
            feature_names=self.extractor.names,
            durations=cfg.get("windows", {"short": 5.0, "medium": 60.0, "long": 900.0}),
        )

        # 主特徵：第一個啟用的特徵。對照實驗只比這一個。
        self.primary = self.extractor.names[0]

        # -- 第 4 層：慢通道 --------------------------------------------
        if use_baseline_machine:
            # 對照組：單閾值，模擬官方 demo 的作法
            th = cfg["state_machine"]["enter_threshold"]
            self.machine = SingleThresholdStateMachine(self.primary, th)
        else:
            self.machine = HysteresisStateMachine(
                self.primary, StateMachineConfig(**cfg["state_machine"])
            )

        # -- 第 4 層：快通道 --------------------------------------------
        # 對照組不開快通道 —— 對照組要模擬的就是「只有單閾值」
        fast_raw = dict(cfg.get("fast_channel", {}))
        fast_raw.setdefault("feature", self.primary)
        self.fast = (None if use_baseline_machine
                     else FastChannel(FastChannelConfig(**fast_raw)))

        # -- 顏色（滲血）------------------------------------------------
        ccfg_raw = dict(cfg.get("color", {}))
        self.color_enabled = bool(ccfg_raw.pop("enabled", False))
        self.color_cfg = ColorConfig(**ccfg_raw)
        self.bleeding = (
            BleedingMonitor(self.color_cfg,
                            baseline_frames=int(self.baseline.cfg.duration_s * fps_hint))
            if self.color_enabled else None
        )

        # -- 第 5 層：融合 ----------------------------------------------
        fusion_raw = dict(cfg.get("fusion", {}))
        self.fusion_cfg = FusionConfig.from_dict(fusion_raw)
        if self.fusion_cfg.enabled and not self.fusion_cfg.signals:
            from .fusion import default_signals
            self.fusion_cfg.signals = default_signals(self.color_enabled)
        # 對照組不做融合
        self.fusion = (None if use_baseline_machine or not self.fusion_cfg.enabled
                       else ConsensusFusion(self.fusion_cfg))

        # -- 輸出層 ------------------------------------------------------
        self.bed_id = cfg.get("sink", {}).get("bed_id", "Bed00")
        self.session_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]

        self.sink: Optional[EventSink] = None
        self.status: Optional[StatusReporter] = None
        if enable_sink and cfg.get("sink"):
            self.sink = make_sink(cfg, self.bed_id)
            self.status = StatusReporter(
                self.sink,
                interval_s=float(cfg["sink"].get("status_interval_s", 5.0)),
            )

        self._t_start: Optional[float] = None
        self._t_last: float = 0.0
        self._emitted = 0
        self._finalized = False
        self._open_at_end = False
        self._prev_state: State = State.NORMAL

    # ------------------------------------------------------------------

    def run(self, source) -> Iterator[FrameResult]:
        for obs in self.backend.stream(source):
            yield self.step(obs)

    def step(self, obs: FrameObservation) -> FrameResult:
        obs = self.filter(obs)

        if self._t_start is None:
            self._t_start = obs.timestamp
        self._t_last = obs.timestamp

        # -- 層 1–2：特徵與平滑 -----------------------------------------
        raw = self.extractor(obs)
        raw.update(self.velocity.update(obs.timestamp, raw))
        feats = self.smoother.update(raw)

        # -- 顏色：獨立於姿態特徵，有自己的基準線 ------------------------
        # ★ 不要把它混進 feats —— BleedingMonitor 已經做過 baseline，
        #   再丟給 BaselineTracker 會變成減兩次基準。
        color = self.bleeding.update(obs) if self.bleeding is not None else {}

        # -- 快通道：吃特徵本身，不需要基準線 ----------------------------
        # 所以它在基準建立期就已經在工作了。突發事件不該等 30 秒才開始防。
        fast_event = None
        if self.fast is not None:
            fast_event = self.fast.update(obs.timestamp, feats.get(self.primary, NAN))

        # -- 基準建立期：慢通道還不能判定 --------------------------------
        if not self.baseline.is_ready:
            self.baseline.observe(feats, obs.timestamp)
            result = FrameResult(
                obs=obs, features=feats, deviations={},
                state=State.NORMAL, fast_event=fast_event,
                baseline_ready=False,
                baseline_progress=self.baseline.progress(),
                color=color,
            )
            self._emit(result)
            return result

        # -- 層 3：相對基準的偏離量 --------------------------------------
        # 取絕對值 —— 往哪個方向偏離都算異常
        dev = {k: abs(v) if v == v else v
               for k, v in self.baseline.deviation(feats).items()}

        self.windows.push(obs.timestamp, dev)

        # -- 層 5：融合 --------------------------------------------------
        fusion_result = None
        if self.fusion is not None:
            # ★ 閘門用「每個特徵自己依賴的關鍵點」的可見率，
            #   不是全身 17 點的可見率 —— 見 features.feature_keypoints()。
            merged = {
                **dev, **color,
                **self.extractor.visibilities(obs),
                "keypoint_visibility": obs.visibility_ratio(),   # 僅供診斷
            }
            fusion_result = self.fusion(merged)

        # -- 層 4：慢通道狀態機 ------------------------------------------
        if (fusion_result is not None and self.fusion_cfg.drive_state_machine):
            # 融合分數驅動狀態機。不可信的那幾格餵 NaN ——
            # 狀態機看到 NaN 不推進也不重置，正好是「暫時看不到」該有的行為。
            value = fusion_result.risk if fusion_result.usable else NAN
        else:
            value = dev.get(self.primary, NAN)

        event = self.machine.update(obs.timestamp, value)

        state = getattr(self.machine, "state", State.NORMAL)
        result = FrameResult(
            obs=obs, features=feats, deviations=dev,
            state=state, new_event=event, fast_event=fast_event,
            baseline_ready=True,
            baseline_progress=self.baseline.progress(),
            window_stats={
                name: self.windows.stats(self.primary, name)
                for name in self.windows.durations
            },
            color=color,
            fusion=fusion_result,
        )
        self._emit(result)
        return result

    # ------------------------------------------------------------------
    # 輸出
    # ------------------------------------------------------------------

    def _emit(self, r: FrameResult) -> None:
        """把事件與心跳送出去。

        ★ 這裡是「影像不出床邊」這句話真正被執行的地方。
          送出去的只有 event_payload() 組出來的扁平字典，
          而 sink.validate_payload() 會在寫出前拒絕任何巢狀結構。
          想偷渡關鍵點座標，程式會直接拋例外。
        """
        if self.sink is None:
            return

        for ev in (r.new_event, r.fast_event):
            if ev is None:
                continue
            extra = {}
            if r.fusion is not None:
                extra = r.fusion.to_payload()
            self.sink.emit(event_payload(ev, self.bed_id, self.session_id, extra))
            self._emitted += 1

        # ★ 事件解除時補送一筆收尾。
        #
        #   事件是在「確認」的當下送出去的，那時 duration 還是 null、
        #   peak 還只是當下的值。如果不補送，護理站的紀錄裡這筆事件
        #   永遠不知道持續了多久、峰值多少 —— 稽核時那是不能接受的空白。
        #
        #   成本：每個事件多一筆約 180 bytes。相對於每小時 720 筆心跳，
        #   可以忽略，但換到的是完整的事件紀錄。
        if (self._prev_state is State.ALERT and r.state is not State.ALERT):
            done = [e for e in self.machine.confirmed_events()
                    if e.end_time is not None]
            if done:
                self.sink.emit(event_payload(
                    done[-1], self.bed_id, self.session_id, {"final": True}))
        self._prev_state = r.state

        if self.status is not None:
            self.status.maybe_report(r.obs.timestamp, self._bed_status(r))

    def _bed_status(self, r: FrameResult) -> BedStatus:
        if r.fusion is not None and r.fusion.reliability is Reliability.UNRELIABLE:
            # ★ 不可信時回報 unreliable，不要回報 normal。
            #   護理站總覽上這一床會變成灰色而不是綠色。
            state_str = "unreliable"
        elif not r.baseline_ready:
            state_str = "baseline"
        else:
            state_str = r.state.value

        return BedStatus(
            bed=self.bed_id,
            session=self.session_id,
            state=state_str,
            baseline_ready=r.baseline_ready,
            subject_found=r.obs.meta.get("subject_found", True),
            visibility=r.obs.visibility_ratio(),
            primary_value=r.deviations.get(self.primary, NAN),
            events_total=self._emitted,
            # ★ 用 `is None` 不要用 `or` —— 第一格的 timestamp 是 0.0，
            #   `self._t_start or ...` 會把它當成假值，uptime 永遠是 0。
            uptime_s=r.obs.timestamp - (
                self._t_start if self._t_start is not None else r.obs.timestamp),
        )

    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """收掉還開著的事件。summary() 與 close() 都會先呼叫它。

        ★ 沒有這一步，「影片結束時還在告警中」的那一筆會從紀錄裡消失。
          那恰好是最重要的一筆。
        """
        if self._finalized:
            return
        self._finalized = True
        ev = self.machine.finalize(self._t_last)
        # 不在這裡補送 —— 已確認的事件在確認當下就送出去了。
        # 這裡只是把本地紀錄補完整，並記下它結束時還開著。
        self._open_at_end = ev is not None and ev.confirmed_at is not None

    def summary(self) -> dict:
        """跑完一支影片後的統計。對照實驗與報告都用這個。"""
        self.finalize()
        slow_events = self.machine.confirmed_events()
        fast_events = self.fast.confirmed_events() if self.fast else []

        out = {
            "session": self.session_id,
            "bed": self.bed_id,
            "alerts": len(slow_events) + len(fast_events),
            "alerts_slow": len(slow_events),
            "alerts_fast": len(fast_events),
            "dismissed_in_watching": getattr(
                self.machine, "dismissed_count", lambda: 0)(),
            "events": [
                {
                    "channel": e.channel,
                    "feature": e.feature,
                    "start": round(e.start_time, 2),
                    "end": round(e.end_time, 2) if e.end_time else None,
                    "duration": round(e.duration, 2) if e.duration else None,
                    "peak": round(e.peak_value, 4),
                }
                for e in sorted(slow_events + fast_events, key=lambda x: x.start_time)
            ],
            "filter": self.filter.report(),
            "baseline": {
                "ready": self.baseline.is_ready,
                "duration_s": self.baseline.cfg.duration_s,
                "values": {k: round(v, 4) for k, v in self.baseline._baseline.items()},
            },
        }
        if self.fast is not None:
            out["fast_channel"] = self.fast.report()
        if self.fusion is not None:
            out["fusion"] = self.fusion.report()
        if self.bleeding is not None:
            out["bleeding"] = self.bleeding.report()
        if self.sink is not None:
            out["sink"] = self.sink.stats()
        return out

    def close(self) -> None:
        """收尾。★ 一定要呼叫 —— sink 要 flush，HTTP 執行緒要停。"""
        self.finalize()
        if self.sink is not None:
            try:
                # 床次結束的摘要，一筆。護理站據此知道這床下線了。
                self.sink.emit({
                    "v": 1, "type": "session_end",
                    "bed": self.bed_id, "session": self.session_id,
                    "duration": round(self._t_last - (self._t_start or 0.0), 1),
                    "alerts_slow": len(self.machine.confirmed_events()),
                    "alerts_fast": len(self.fast.confirmed_events()) if self.fast else 0,
                    "dismissed": getattr(self.machine, "dismissed_count",
                                         lambda: 0)(),
                    "glitch_rejected": self.fast.report()["glitch_rejected"]
                    if self.fast else 0,
                    # ★ 結束時還在告警中。護理站要知道這床不是「平安結束」
                    "open_at_end": self._open_at_end,
                })
            except Exception as e:
                print(f"[pipeline] 結束摘要送出失敗：{e}")
            self.sink.close()
            self.sink = None
        self.backend.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
