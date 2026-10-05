"""
事件輸出層 —— 把事件寫到本地、送到護理站。

★ 這個模組存在的理由是提案的核心主張：
  「影像不出床邊，網路上只跑事件文字。」

  沒有這一層，那句話只是口號。有了這一層，我們可以量給評審看：
  一筆事件的 JSON payload 是多少 bytes，30 床一小時總共多少。

  對照：30 路 1080p 串流約 120 Mbps 持續佔用。

三種 sink，可以疊加：

    JSONLSink     寫本地檔案，一行一個 JSON。離線可查、進 git 可重現
    HTTPSink      POST 到護理站。失敗自動排隊重送，不阻塞推論
    CompositeSink 同時寫多個目的地

設計原則：
  1. **絕對不送影像。** payload 只有文字與數字，上限 1 KB
  2. **送出失敗不能影響推論。** HTTPSink 用背景執行緒 + 佇列
  3. **本地永遠先寫一份。** 網路斷掉時事件不會遺失
"""

from __future__ import annotations

import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .temporal.state_machine import Event

# payload 大小上限。超過代表有人把不該送的東西塞進來了
MAX_PAYLOAD_BYTES = 1024


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------

def event_payload(
    ev: Event,
    bed_id: str,
    session_id: str,
    extra: Optional[dict] = None,
) -> dict:
    """把 Event 轉成要送出去的 payload。

    ★ 這裡決定了「網路上到底跑什麼」。
      只有文字與數字 —— 沒有影像、沒有關鍵點座標、沒有任何可還原畫面的東西。

    典型大小約 180 bytes。
    """
    payload = {
        "v": 1,                                  # schema 版本
        "bed": bed_id,
        "session": session_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "t_offset": round(ev.start_time, 1),     # 治療開始後第幾秒
        "feature": ev.feature,
        "channel": ev.channel,                   # fast / slow
        "duration": round(ev.duration, 1) if ev.duration else None,
        "peak": round(ev.peak_value, 4),
        "outcome": ev.meta.get("outcome"),
    }
    if extra:
        payload.update(extra)
    return payload


def payload_size(payload: dict) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


class PayloadTooLarge(ValueError):
    pass


def validate_payload(payload: dict) -> int:
    """確認 payload 沒有夾帶不該送的東西。

    這個檢查是刻意的 —— 它讓「影像不出床邊」變成程式碼強制的規則，
    而不是靠開發者自律。
    """
    size = payload_size(payload)
    if size > MAX_PAYLOAD_BYTES:
        raise PayloadTooLarge(
            f"payload {size} bytes 超過上限 {MAX_PAYLOAD_BYTES}。"
            "檢查是否誤把影像或關鍵點陣列塞進去了。"
        )
    for key, value in payload.items():
        if isinstance(value, (bytes, bytearray)):
            raise PayloadTooLarge(f"欄位 {key} 是二進位資料，不允許送出")
        if isinstance(value, list) and len(value) > 32:
            raise PayloadTooLarge(f"欄位 {key} 是長度 {len(value)} 的陣列，疑似座標資料")
    return size


# --------------------------------------------------------------------------
# sink
# --------------------------------------------------------------------------

class EventSink(ABC):
    @abstractmethod
    def emit(self, payload: dict) -> None:
        ...

    def close(self) -> None:
        return None

    def stats(self) -> dict:
        return {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class JSONLSink(EventSink):
    """寫本地 JSONL，一行一個事件。

    為什麼用 JSONL 而非資料庫：
      - 床邊主機不該跑資料庫，多一個會壞的東西
      - 一行一筆，追加寫入，斷電最多掉最後一筆
      - 可以直接 grep、可以進 git、評審可重現
    """

    def __init__(self, path: str | Path, flush_each: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._flush_each = flush_each
        self._count = 0
        self._bytes = 0

    def emit(self, payload: dict) -> None:
        size = validate_payload(payload)
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if self._flush_each:
            self._fh.flush()
        self._count += 1
        self._bytes += size

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def stats(self) -> dict:
        return {
            "sink": "jsonl",
            "path": str(self.path),
            "events": self._count,
            "payload_bytes_total": self._bytes,
            "payload_bytes_mean": round(self._bytes / self._count, 1) if self._count else 0,
        }


class HTTPSink(EventSink):
    """POST 到護理站的收集器。

    ★ 背景執行緒 + 佇列。網路慢或斷掉時推論不會被拖住 ——
      床邊的即時偵測比送出去重要。

    失敗的事件會留在佇列裡重試。佇列滿了就丟掉最舊的，
    因為 JSONLSink 已經有完整紀錄，HTTP 只是即時通報。
    """

    def __init__(
        self,
        url: str,
        timeout: float = 2.0,
        max_queue: int = 500,
        retry_interval: float = 5.0,
    ):
        self.url = url
        self.timeout = timeout
        self.retry_interval = retry_interval

        self._q: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._sent = 0
        self._failed = 0
        self._dropped = 0
        self._bytes = 0

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def emit(self, payload: dict) -> None:
        self._bytes += validate_payload(payload)
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()          # 丟最舊的
                self._dropped += 1
                self._q.put_nowait(payload)
            except queue.Empty:
                pass

    def _run(self) -> None:
        import urllib.error
        import urllib.request

        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            data = json.dumps(payload, ensure_ascii=False).encode()
            req = urllib.request.Request(
                self.url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout):
                    self._sent += 1
            except (urllib.error.URLError, OSError, TimeoutError):
                self._failed += 1
                try:
                    self._q.put_nowait(payload)      # 放回去重試
                except queue.Full:
                    self._dropped += 1
                self._stop.wait(self.retry_interval)

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=3.0)

    def stats(self) -> dict:
        return {
            "sink": "http",
            "url": self.url,
            "sent": self._sent,
            "failed": self._failed,
            "dropped": self._dropped,
            "pending": self._q.qsize(),
            "payload_bytes_total": self._bytes,
        }


class CompositeSink(EventSink):
    """同時寫多個目的地。任一失敗不影響其他。"""

    def __init__(self, sinks: list[EventSink]):
        self.sinks = sinks

    def emit(self, payload: dict) -> None:
        for s in self.sinks:
            try:
                s.emit(payload)
            except Exception as e:                # 一個 sink 掛掉不能拖垮推論
                print(f"[sink] {type(s).__name__} 失敗：{e}")

    def close(self) -> None:
        for s in self.sinks:
            s.close()

    def stats(self) -> dict:
        return {"sinks": [s.stats() for s in self.sinks]}


# --------------------------------------------------------------------------
# 床邊狀態心跳
# --------------------------------------------------------------------------

@dataclass
class BedStatus:
    """床邊主機定期回報的狀態。

    這不是事件，是心跳 —— 護理站需要知道每一床的系統是不是活著。
    沒有心跳的床在總覽上會變灰，代表「這床現在沒在監測」。
    """
    bed: str
    session: str
    state: str                       # normal / watching / alert / cooldown
    baseline_ready: bool
    subject_found: bool              # 這一格有沒有認出病人
    visibility: float                # 關鍵點可見率
    primary_value: float             # 主特徵的當前偏離量
    events_total: int
    uptime_s: float
    ts: str = field(default_factory=lambda:
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_payload(self) -> dict:
        return {
            "v": 1, "type": "status",
            "bed": self.bed, "session": self.session, "ts": self.ts,
            "state": self.state,
            "baseline_ready": self.baseline_ready,
            "subject_found": self.subject_found,
            "visibility": round(self.visibility, 3),
            "value": round(self.primary_value, 4) if self.primary_value == self.primary_value else None,
            "events": self.events_total,
            "uptime": round(self.uptime_s, 1),
        }


class StatusReporter:
    """按固定間隔送出床邊狀態心跳。

    用法（在主迴圈裡每格呼叫，內部自己節流）：
        reporter.maybe_report(timestamp, status)
    """

    def __init__(self, sink: EventSink, interval_s: float = 5.0):
        self.sink = sink
        self.interval_s = interval_s
        self._last = -1e9

    def maybe_report(self, now: float, status: BedStatus) -> bool:
        if now - self._last < self.interval_s:
            return False
        self._last = now
        self.sink.emit(status.to_payload())
        return True


# --------------------------------------------------------------------------
# 工廠
# --------------------------------------------------------------------------

def make_sink(cfg: dict, bed_id: str) -> EventSink:
    """依 config 建立 sink。

    config/default.yaml:
        sink:
          jsonl: results/events_{bed}.jsonl    # {bed} 會被替換
          http: http://nurse-station.local:8420/ingest   # 可省略
          status_interval_s: 5.0
    """
    scfg = cfg.get("sink", {})
    sinks: list[EventSink] = []

    jsonl = scfg.get("jsonl")
    if jsonl:
        sinks.append(JSONLSink(str(jsonl).replace("{bed}", bed_id)))

    http = scfg.get("http")
    if http:
        sinks.append(HTTPSink(http))

    if not sinks:
        sinks.append(JSONLSink(f"results/events_{bed_id}.jsonl"))

    return CompositeSink(sinks) if len(sinks) > 1 else sinks[0]
