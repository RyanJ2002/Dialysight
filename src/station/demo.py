"""
示範模式 —— 沒有床邊主機時，模擬 8 床送資料進護理站。

    python scripts/nurse_station.py --demo

★ 送進護理站的 payload 跟真的床邊一模一樣：
  心跳用 sink.BedStatus.to_payload()，事件用 sink.event_payload()，
  而且都先過 sink.validate_payload()。所以頁面上的流量（bytes、kbps）
  是真實格式下的數字，不是另外編的。

  慢通道告警由 HysteresisStateMachine 產生 —— 跟床邊同一個類別，
  門檻、確認時間、冷卻都照 StateMachineConfig 預設值走，不是寫死的劇本。

★ 每一床的情境（數值曲線）是模擬的，畫面上會標「示範資料」。
  另外預先放幾份「已結束的床次」，讓紀錄整理頁一打開就有東西可看。
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from datetime import datetime, timezone

from ..sink import BedStatus, event_payload, validate_payload
from ..temporal.state_machine import (Event, HysteresisStateMachine, State,
                                      StateMachineConfig)

FEATURE = "arm_displacement"


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def _sid(bed: str, start: float) -> str:
    return datetime.fromtimestamp(start).strftime("%Y%m%dT%H%M%S") + "-" + bed.lower()


class SimBed:
    """一床的模擬：依情境產生偏離量，交給真的狀態機判斷。"""

    def __init__(self, bed: str, scenario: str, started_ago_s: float, phase: float,
                 seed: int):
        self.bed, self.scenario, self.phase = bed, scenario, phase
        now = time.time()
        self.t0 = now - started_ago_s              # 治療開始
        self.demo_t0 = now                         # 示範開始
        self.session = _sid(bed, self.t0)
        self.machine = HysteresisStateMachine(FEATURE, StateMachineConfig())
        self.rng = random.Random(seed)
        self.prev_state = State.NORMAL
        self.next_hb = now + (seed % 5)            # 心跳錯開，不要全部擠在同一秒
        self.emitted = 0
        self.fast_fired: set[int] = set()

    def sample(self, t: float) -> tuple[float | None, float, str | None]:
        """回傳 (偏離量 | None=看不到, 可見度, 覆寫狀態)。t = 示範開始後秒數。"""
        noise = self.rng.gauss(0, 0.006)
        s = self.scenario
        if s == "drift":                           # 漸進偏移 → 告警 → 解除
            p = (t + self.phase) % 200
            if p < 40:
                v = 0.03
            elif p < 80:
                v = 0.03 + (p - 40) / 40 * 0.21
            elif p < 130:
                v = 0.235 + 0.01 * math.sin(p)
            elif p < 150:
                v = 0.24 - (p - 130) / 20 * 0.20
            else:
                v = 0.04
            return max(0.0, v + noise), 0.93, None
        if s == "hover":                           # 接近門檻、觀察後自行回復
            p = (t + self.phase) % 70
            if 20 <= p < 21:
                v = 0.06 + (p - 20) * 0.10
            elif 21 <= p < 26:
                v = 0.162
            else:
                v = 0.06 + 0.012 * math.sin(p / 3)
            return max(0.0, v + noise), 0.9, None
        if s == "occluded":                        # 被子蓋住手臂 → 無法判定
            p = (t + self.phase) % 180
            if 30 <= p < 150:
                return None, 0.22, "unreliable"
            return max(0.0, 0.04 + noise), 0.88, None
        if s == "sudden":                          # 偶爾突然坐起（快通道）
            p = (t + self.phase) % 150
            v = 0.28 if 40 <= p < 41.5 else 0.035
            return max(0.0, v + noise), 0.9, None
        if s == "baseline":                        # 剛上機，先建立個人基準
            if t < 35:
                return None, 0.95, "baseline"
            return max(0.0, 0.03 + noise), 0.95, None
        return max(0.0, 0.035 + 0.01 * math.sin(t / 9) + noise), 0.92, None

    def tick(self, now: float) -> list[dict]:
        if self.scenario == "offline" and now - self.demo_t0 > 8:
            return []                              # 床邊主機掉線：心跳停了
        t = now - self.demo_t0
        elapsed = now - self.t0
        value, vis, override = self.sample(t)
        out: list[dict] = []

        if value is not None and override is None:
            ev = self.machine.update(elapsed, value)
            if ev is not None:
                out.append(event_payload(ev, self.bed, self.session,
                                         {"risk": round(min(1.0, value / .25), 3),
                                          "conf": round(vis, 3), "reliability": "ok"}))
                self.emitted += 1
        state = self.machine.state
        if self.prev_state is State.ALERT and state is not State.ALERT:
            done = [e for e in self.machine.confirmed_events() if e.end_time is not None]
            if done:
                out.append(event_payload(done[-1], self.bed, self.session, {"final": True}))
        self.prev_state = state

        if self.scenario == "sudden":
            k = int((t + self.phase) // 150)
            if (t + self.phase) % 150 >= 41.0 and k not in self.fast_fired:
                self.fast_fired.add(k)
                ev = Event(FEATURE, elapsed - 0.8, elapsed, 0.31, elapsed, "fast",
                           {"outcome": "confirmed"})
                out.append(event_payload(ev, self.bed, self.session,
                                         {"risk": 1.0, "conf": 0.9, "reliability": "ok"}))
                self.emitted += 1

        if now >= self.next_hb:
            self.next_hb = now + 5.0
            out.append(BedStatus(
                bed=self.bed, session=self.session,
                state=override or state.value,
                baseline_ready=override != "baseline",
                subject_found=True, visibility=vis,
                primary_value=value if value is not None else float("nan"),
                events_total=self.emitted, uptime_s=elapsed,
            ).to_payload())
        return out


BEDS = [
    ("Bed01", "normal"), ("Bed02", "offline"), ("Bed03", "hover"),
    ("Bed04", "sudden"), ("Bed05", "normal"), ("Bed06", "occluded"),
    ("Bed07", "drift"), ("Bed08", "baseline"),
]


def _seed_sessions(store) -> None:
    """預先放幾份已結束的床次（紀錄整理頁用），含 Bed07 之前三次治療供比較。"""
    now = time.time()

    def make(bed, end_ago_min, dur_s, alerts=(), watch=(), unrel=(), dismissed=0,
             glitch=0, pauses=0, signed_by=None):
        end = now - end_ago_min * 60
        start = end - dur_s
        sid = _sid(bed, start)
        events = []
        for off, ch, peak, dur, ack_by, ack_after in alerts:
            store._eid += 1
            events.append({
                "v": 1, "bed": bed, "session": sid, "ts": _iso(start + off),
                "t_offset": float(off), "feature": FEATURE, "channel": ch,
                "duration": dur, "peak": peak,
                "outcome": "confirmed" if ch == "fast" else "resolved",
                "risk": 0.8, "conf": 0.9, "reliability": "ok",
                "_seen": start + off, "eid": store._eid,
                **({"ack_by": ack_by, "ack_after_s": ack_after} if ack_by else {}),
            })
        store.sessions[sid] = {
            "session": sid, "bed": bed, "first": start, "last": end, "events": events,
            "end": {"v": 1, "type": "session_end", "bed": bed, "session": sid,
                    "duration": float(dur_s),
                    "alerts_slow": sum(1 for a in alerts if a[1] == "slow"),
                    "alerts_fast": sum(1 for a in alerts if a[1] == "fast"),
                    "dismissed": dismissed, "glitch_rejected": glitch,
                    "open_at_end": False},
            "status_n": int(dur_s / 5),
            "unreliable_n": int(sum(d for _, d in unrel) / 5), "paused_n": 0,
            "watch_eps": [[start + o, start + o + d, p] for o, d, p in watch],
            "unrel_eps": [[start + o, start + o + d] for o, d in unrel],
            "pauses": pauses, "_prev": None,
        }
        if signed_by:
            draft = store.draft(sid, start_llm=False)
            store.signed[sid] = {
                "type": "nursing_record", "session": sid, "bed": bed,
                "signed_at": _iso(end + 600), "nurse": signed_by,
                "summary": draft["headline"], "actions": "", "complaint": "",
                "fact_check": {"ok": True, "unmatched": []},
                "generated_by": draft["meta"]["method"],
            }

    day = 86400
    make("Bed07", 14 * day / 60, 14400, watch=[(5400, 6, 0.12)], dismissed=1,
         signed_by="護理師 A")
    make("Bed07", 9 * day / 60, 14460, alerts=[(9100, "slow", 0.19, 25.0, "護理師 C", 35)],
         dismissed=1, signed_by="護理師 C")
    make("Bed07", 7 * day / 60, 14280, signed_by="護理師 A")
    make("Bed05", 50, 14520, pauses=2, signed_by="護理師 B")
    make("Bed03", 32, 14700, watch=[(2400, 6, 0.12), (6900, 6, 0.13), (11200, 5, 0.11)],
         dismissed=3)
    make("Bed02", 26, 14100, unrel=[(5200, 1320)])
    make("Bed07", 18, 14520,
         alerts=[(7860, "slow", 0.21, 18.0, "護理師 A", 42)],
         watch=[(1320, 4, 0.11), (6300, 6, 0.12)], unrel=[(4260, 180)],
         dismissed=2, glitch=1)


def start_demo(store) -> threading.Thread:
    store.demo = True
    _seed_sessions(store)
    rng = random.Random(7)
    beds = [SimBed(bed, sc, started_ago_s=rng.uniform(3600, 9000),
                   phase=rng.uniform(0, 60), seed=i)
            for i, (bed, sc) in enumerate(BEDS)]

    def run() -> None:
        while True:
            now = time.time()
            for b in beds:
                for payload in b.tick(now):
                    validate_payload(payload)         # 跟床邊同一道關卡
                    raw = json.dumps(payload, ensure_ascii=False).encode()
                    store.ingest(payload, len(raw))
            time.sleep(0.5)

    th = threading.Thread(target=run, daemon=True, name="demo-beds")
    th.start()
    return th
