"""
把各層串起來的主管線。

    影格 → 推論 → 過濾 → 特徵 → 平滑 → 基準偏離 → 滑動窗 → 狀態機 → 事件

★ 這個檔案是「膠水」，不要在這裡放演算法邏輯。
  每一層的邏輯都在各自的模組裡，這裡只負責串接與資料流。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .inference.base import FrameObservation, make_backend
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


@dataclass
class FrameResult:
    """單一影格跑完整條管線後的結果。視覺化與報告都吃這個。"""
    obs: FrameObservation
    features: dict[str, float]          # 平滑後
    deviations: dict[str, float]        # 相對基準
    state: State
    new_event: Optional[Event] = None
    baseline_ready: bool = True
    baseline_progress: tuple[int, int] = (0, 0)   # (已收集, 需要) 給 UI 顯示進度
    window_stats: dict = field(default_factory=dict)


class Pipeline:
    """完整的處理管線。

    用法：
        pipe = Pipeline(cfg)
        for result in pipe.run("data/clips/clip_03.mp4"):
            canvas = renderer.render(result.obs, result.deviations,
                                     result.state, result.new_event)
    """

    def __init__(self, cfg: dict, use_baseline_machine: bool = False):
        self.cfg = cfg
        self.backend = make_backend(cfg)

        self.filter = KeypointFilter(FilterConfig(**cfg.get("filter", {})))

        fcfg = FeatureConfig(**cfg.get("features", {}))
        self.extractor = FeatureExtractor(fcfg)
        self.velocity = VelocityTracker(
            [n for n in fcfg.enabled if n in ("arm_displacement", "torso_angle")]
        )

        self.baseline = BaselineTracker(BaselineConfig(**cfg.get("baseline", {})))

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

        if use_baseline_machine:
            # 對照組：單閾值，模擬官方 demo 的作法
            th = cfg["state_machine"]["enter_threshold"]
            self.machine = SingleThresholdStateMachine(self.primary, th)
        else:
            self.machine = HysteresisStateMachine(
                self.primary, StateMachineConfig(**cfg["state_machine"])
            )

    # ------------------------------------------------------------------

    def run(self, source) -> Iterator[FrameResult]:
        for obs in self.backend.stream(source):
            yield self.step(obs)

    def step(self, obs: FrameObservation) -> FrameResult:
        obs = self.filter(obs)

        raw = self.extractor(obs)
        raw.update(self.velocity.update(obs.timestamp, raw))
        feats = self.smoother.update(raw)

        # 基準建立期：收集樣本，不做事件判定
        if not self.baseline.is_ready:
            self.baseline.observe(feats)
            return FrameResult(
                obs=obs, features=feats, deviations={},
                state=State.NORMAL, baseline_ready=False,
                baseline_progress=(self.baseline._count, self.baseline.cfg.n_frames),
            )

        # 取絕對值 —— 往哪個方向偏離都算異常
        dev = {k: abs(v) if v == v else v
               for k, v in self.baseline.deviation(feats).items()}

        self.windows.push(obs.timestamp, dev)

        value = dev.get(self.primary, float("nan"))
        event = self.machine.update(obs.timestamp, value)

        state = getattr(self.machine, "state", State.NORMAL)
        return FrameResult(
            obs=obs, features=feats, deviations=dev,
            state=state, new_event=event, baseline_ready=True,
            window_stats={
                name: self.windows.stats(self.primary, name)
                for name in self.windows.durations
            },
        )

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """跑完一支影片後的統計。對照實驗與報告都用這個。"""
        events = self.machine.confirmed_events()
        return {
            "alerts": len(events),
            "dismissed_in_watching": getattr(self.machine, "dismissed_count", lambda: 0)(),
            "events": [
                {
                    "feature": e.feature,
                    "start": round(e.start_time, 2),
                    "end": round(e.end_time, 2) if e.end_time else None,
                    "duration": round(e.duration, 2) if e.duration else None,
                    "peak": round(e.peak_value, 4),
                }
                for e in events
            ],
            "filter": self.filter.report(),
            "baseline": {
                "ready": self.baseline.is_ready,
                "values": {k: round(v, 4) for k, v in self.baseline._baseline.items()},
            },
        }

    def close(self) -> None:
        self.backend.close()