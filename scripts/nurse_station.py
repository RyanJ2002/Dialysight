#!/usr/bin/env python3
"""
護理站收集器 —— 接收各床送來的事件文字，提供多床總覽。

★ 這支程式驗證提案的核心主張：
  床邊送過來的只有文字。這裡收到什麼，就是網路上跑的全部。
  啟動後它會持續統計「每筆 payload 幾 bytes、總共多少」——
  那組數字可以直接寫進簡報。

只用 Python 標準函式庫，不需要安裝任何東西。

用法：
    # 護理站電腦上啟動
    python scripts/nurse_station.py --port 8420

    # 床邊主機的 config 指向它
    #   sink:
    #     http: http://<護理站IP>:8420/ingest

    # 瀏覽器開 http://localhost:8420 看總覽

端點：
    POST /ingest     床邊送事件或心跳進來
    GET  /           多床總覽網頁
    GET  /api/state  目前所有床的狀態（JSON，給網頁輪詢）
    GET  /api/events 最近的事件列表
    GET  /api/stats  流量統計 ★ 簡報要用的數字
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "src" / "station" / "dashboard.html"

# 多久沒心跳就當作離線
OFFLINE_AFTER_S = 20.0


class Store:
    """記憶體狀態 + JSONL 落地。"""

    def __init__(self, log_path: Path, max_events: int = 500):
        self.lock = threading.Lock()
        self.beds: dict[str, dict] = {}
        self.events: deque[dict] = deque(maxlen=max_events)
        self.started = time.time()

        self.n_events = 0
        self.n_status = 0
        self.bytes_total = 0

        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = log_path.open("a", encoding="utf-8")

    def ingest(self, payload: dict, raw_size: int) -> None:
        with self.lock:
            self.bytes_total += raw_size
            bed = payload.get("bed", "unknown")
            now = time.time()

            kind = payload.get("type")

            if kind == "status":
                self.n_status += 1
                # 保留 events 累計值，心跳的 events 欄位是床邊自己數的
                prev = self.beds.get(bed, {})
                self.beds[bed] = {**prev, **payload, "_seen": now}

            elif kind == "session_end":
                # ★ 床次結束不是事件，不要計進 n_events，也不要進事件列表。
                #   但要標記這床已經下線，以及它是不是「還在告警中就結束」。
                b = self.beds.setdefault(bed, {"bed": bed})
                b["_seen"] = now
                b["state"] = "ended"
                b["session_end"] = payload
                b["open_at_end"] = payload.get("open_at_end", False)

            elif payload.get("final"):
                # ★ 事件解除的補送。更新既有那一筆，不要當成新事件 ——
                #   不然護理站的事件數會是實際的兩倍。
                #   用 (床, 事件起始秒, 通道) 配對。
                key = (payload.get("t_offset"), payload.get("channel"))
                for e in self.events:
                    if (e.get("bed") == bed
                            and (e.get("t_offset"), e.get("channel")) == key):
                        e.update({k: v for k, v in payload.items()
                                  if k != "final"})
                        break
                else:
                    # 沒配對到（護理站中途才上線），當成新事件收下
                    self.n_events += 1
                    self.events.appendleft({**payload, "_seen": now})
                b = self.beds.setdefault(bed, {"bed": bed})
                b["_seen"] = now

            else:
                self.n_events += 1
                self.events.appendleft({**payload, "_seen": now})
                b = self.beds.setdefault(bed, {"bed": bed})
                b["_seen"] = now
                b["last_event"] = payload
                b["events"] = b.get("events", 0) + 1

            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._fh.flush()

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            beds = []
            for bed, b in sorted(self.beds.items()):
                age = now - b.get("_seen", 0)
                beds.append({
                    "bed": bed,
                    "online": age < OFFLINE_AFTER_S,
                    "age_s": round(age, 1),
                    "state": b.get("state", "unknown"),
                    "baseline_ready": b.get("baseline_ready"),
                    "subject_found": b.get("subject_found"),
                    "visibility": b.get("visibility"),
                    "value": b.get("value"),
                    "events": b.get("events", 0),
                    "last_event": b.get("last_event"),
                    # ★ 結束時還在告警中 —— 這床不是平安結束，要看得出來
                    "open_at_end": b.get("open_at_end", False),
                })
            return {"beds": beds, "server_time": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")}

    def recent_events(self, n: int = 50) -> list[dict]:
        with self.lock:
            return list(self.events)[:n]

    def stats(self) -> dict:
        """★ 這組數字直接寫進簡報。"""
        with self.lock:
            total = self.n_events + self.n_status
            uptime = max(time.time() - self.started, 1.0)
            mean = self.bytes_total / total if total else 0
            return {
                "uptime_s": round(uptime, 1),
                "beds_seen": len(self.beds),
                "events": self.n_events,
                "status_heartbeats": self.n_status,
                "bytes_total": self.bytes_total,
                "bytes_mean_per_message": round(mean, 1),
                "bytes_per_hour": round(self.bytes_total / uptime * 3600, 1),
                "kbps": round(self.bytes_total * 8 / uptime / 1000, 4),
                "note": "對照：單路 1080p 串流約 4 Mbps；30 床約 120 Mbps",
            }


class Handler(BaseHTTPRequestHandler):
    store: Store = None                      # 由 main 注入

    def log_message(self, *args):            # 關掉預設的逐筆 log
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_POST(self):
        if self.path.rstrip("/") != "/ingest":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            payload = json.loads(raw)
        except Exception as e:
            self._json({"error": f"bad payload: {e}"}, 400)
            return

        self.store.ingest(payload, len(raw))
        self._json({"ok": True, "bytes": len(raw)})

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/":
            if DASHBOARD.exists():
                self._send(200, DASHBOARD.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"dashboard.html not found", "text/plain")
        elif path == "/api/state":
            self._json(self.store.snapshot())
        elif path == "/api/events":
            self._json(self.store.recent_events())
        elif path == "/api/stats":
            self._json(self.store.stats())
        else:
            self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--log", default="results/nurse_station.jsonl")
    args = ap.parse_args()

    Handler.store = Store(Path(args.log))
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"護理站收集器啟動")
    print(f"  總覽網頁   http://localhost:{args.port}")
    print(f"  床邊 POST  http://<本機IP>:{args.port}/ingest")
    print(f"  流量統計   http://localhost:{args.port}/api/stats")
    print(f"  事件紀錄   {args.log}")
    print("\nCtrl-C 離開")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n" + json.dumps(Handler.store.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
