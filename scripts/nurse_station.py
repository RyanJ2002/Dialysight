#!/usr/bin/env python3
"""
護理站收集器 —— 接收各床送來的事件文字，提供多床總覽、事件、紀錄整理與報表。

★ 這支程式驗證提案的核心主張：
  床邊送過來的只有文字。這裡收到什麼，就是網路上跑的全部。
  啟動後它會持續統計「每筆 payload 幾 bytes、總共多少」——
  那組數字可以直接寫進簡報。

只用 Python 標準函式庫，不需要安裝任何東西。完整操作說明：docs/USAGE.md

用法：
    python scripts/nurse_station.py                 # 正式：等床邊主機送資料進來
    python scripts/nurse_station.py --demo          # 沒有床邊主機：模擬 8 床
    python scripts/nurse_station.py --llm http://localhost:8000
                                                    # 選用：紀錄草稿交給 hailo-ollama 改寫

    瀏覽器開 http://localhost:8420（啟動時會印出同網路其他電腦可用的網址）

網頁（同一頁，左側選單切換）：
    床位總覽 · 事件與告警 · 紀錄整理 · 報表與裝置

端點：
    POST /ingest              床邊送事件或心跳進來
    GET  /api/state           各床狀態（含待確認告警、照護暫停、最近 15 分鐘趨勢）
    GET  /api/events          本班事件（告警、觀察、無法判定、照護暫停）
    GET  /api/report          本班統計、各床連線、網路上實際傳輸的內容
    GET  /api/stats           流量統計 ★ 簡報要用的數字
    POST /api/ack             {"bed", "nurse"}               確認告警
    POST /api/pause           {"bed", "minutes", "nurse"}    照護中，暫停通知
    POST /api/resume          {"bed"}                        提前恢復
    GET  /api/records         已結束的床次
    GET  /api/records/<id>    單一床次的紀錄草稿
    POST /api/records/regen   {"session"}                    重新產生草稿
    POST /api/records/sign    {"session", "text", "actions", "nurse"}

★ 護理站端的操作（確認、照護暫停、簽核）只存在護理站，
  不會回送床邊 —— 床邊的偵測永遠照常進行。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.station.records import (HEARTBEAT_S, LLMClient, build_draft,  # noqa: E402
                                 check_claims, fact_check, list_item,
                                 llm_rewrite, session_window)

PAGE = ROOT / "src" / "station" / "dashboard.html"

OFFLINE_AFTER_S = 20.0      # 多久沒心跳就當作離線
ESCALATE_AFTER_S = 60.0     # 告警多久沒人確認就升級提醒
SPARK_LEN = 180             # 趨勢圖：180 筆心跳 ≈ 15 分鐘
ACK_SHOW_S = 600.0          # 「已確認」在卡片上留多久


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Store:
    """記憶體狀態 + JSONL 落地。"""

    def __init__(self, log_path: Path, records_path: Path, max_events: int = 500,
                 enter: float = 0.15, llm: LLMClient | None = None):
        self.lock = threading.RLock()
        self.beds: dict[str, dict] = {}
        self.events: deque[dict] = deque(maxlen=max_events)
        self.sessions: dict[str, dict] = {}
        self.signed: dict[str, dict] = {}
        self.care_log: list[dict] = []
        self.started = time.time()
        self.enter = enter
        self.demo = False
        self.llm = llm
        self._llm_cache: dict[str, dict] = {}

        self.n_events = 0
        self.n_status = 0
        self.bytes_total = 0
        self._eid = 0
        self.last_status: dict | None = None
        self.last_event: dict | None = None

        self.log_path = log_path
        self.records_path = records_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = log_path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------
    # 收資料
    # ------------------------------------------------------------------

    def _session(self, sid: str, bed: str, now: float) -> dict:
        s = self.sessions.get(sid)
        if s is None:
            s = self.sessions[sid] = {
                "session": sid, "bed": bed, "first": now, "last": now,
                "events": [], "end": None, "status_n": 0, "unreliable_n": 0,
                "paused_n": 0, "watch_eps": [], "unrel_eps": [], "pauses": 0,
                "_prev": None,
            }
        return s

    def _log(self, payload: dict) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def ingest(self, payload: dict, raw_size: int, at: float | None = None) -> None:
        with self.lock:
            now = time.time() if at is None else at
            self.bytes_total += raw_size
            bed = payload.get("bed", "unknown")
            sid = payload.get("session")
            sess = self._session(sid, bed, now) if sid else None
            b = self.beds.setdefault(bed, {"bed": bed})
            kind = payload.get("type")

            if kind == "status":
                self.n_status += 1
                self.last_status = payload
                if b.get("session") not in (None, sid):
                    # 同一床換了新床次：上一床次的告警、確認、趨勢都不屬於這一次
                    for k in ("pending", "ack", "spark", "session_end", "open_at_end"):
                        b.pop(k, None)
                    b["events"] = 0
                # 保留 events 累計值，心跳的 events 欄位是床邊自己數的
                state = payload.get("state")
                if state != b.get("state"):
                    b["_state_since"] = now
                b.update({k: v for k, v in payload.items() if k != "events"})
                b["_seen"] = now
                if _num(payload.get("uptime")):
                    b["_start"] = now - float(payload["uptime"])
                v = payload.get("value")
                b.setdefault("spark", deque(maxlen=SPARK_LEN)).append(v if _num(v) else None)
                pend = b.get("pending")
                if pend and state == "alert" and _num(v):
                    pend["peak"] = max(pend.get("peak") or 0, v)   # 告警還在擴大
                if sess is not None:
                    self._track_episodes(sess, state, v, now)
                    if self._pause_active(b, now):
                        sess["paused_n"] += 1

            elif kind == "session_end":
                # ★ 床次結束不是事件，不要計進 n_events，也不要進事件列表。
                #   但要標記這床已經下線，以及它是不是「還在告警中就結束」。
                b["_seen"] = now
                b["state"] = "ended"
                b["session_end"] = payload
                b["open_at_end"] = payload.get("open_at_end", False)
                if sess is not None:
                    sess["end"] = payload
                    if self.llm is not None:          # 先在背景起草，護理師打開時就好了
                        threading.Thread(target=self.draft, args=(sid,), daemon=True).start()

            elif payload.get("final"):
                # ★ 事件解除的補送。更新既有那一筆，不要當成新事件 ——
                #   不然護理站的事件數會是實際的兩倍。
                #   用 (床, 事件起始秒, 通道) 配對。
                key = (payload.get("t_offset"), payload.get("channel"))
                for e in self.events:
                    if (e.get("bed") == bed
                            and (e.get("t_offset"), e.get("channel")) == key):
                        e.update({k: v for k, v in payload.items() if k != "final"})
                        break
                else:
                    # 沒配對到（護理站中途才上線），當成新事件收下
                    self._new_event(payload, b, sess, now)
                b["_seen"] = now

            else:
                self._new_event(payload, b, sess, now)
                self.last_event = payload
                b["_seen"] = now

            if sess is not None:
                sess["last"] = max(sess["last"], now)
            self._log(payload)

    def _new_event(self, payload: dict, b: dict, sess: dict | None, now: float) -> None:
        self.n_events += 1
        self._eid += 1
        rec = {**payload, "_seen": now, "eid": self._eid}
        rec.pop("final", None)
        self.events.appendleft(rec)
        if sess is not None:
            sess["events"].append(rec)              # 同一個物件，確認時兩邊一起更新
        b["events"] = b.get("events", 0) + 1
        if self._pause_active(b, now):
            rec["paused"] = True                    # 照護中：記錄，但不通知
            return
        pend = b.get("pending")
        if pend is None:
            b["pending"] = {"since": now, "eids": [rec["eid"]],
                            "channel": rec.get("channel"), "peak": rec.get("peak")}
        else:
            pend["eids"].append(rec["eid"])
            if _num(rec.get("peak")):
                pend["peak"] = max(pend.get("peak") or 0, rec["peak"])
        b.pop("ack", None)

    @staticmethod
    def _track_episodes(sess: dict, state: str | None, value, now: float) -> None:
        """從心跳推出「觀察中」與「無法判定」的時段（給事件列表與紀錄草稿用）。"""
        sess["status_n"] += 1
        prev = sess["_prev"]
        if state == "unreliable":
            sess["unreliable_n"] += 1
            if prev != "unreliable":
                sess["unrel_eps"].append([now, now])
            else:
                sess["unrel_eps"][-1][1] = now
        if state == "watching":
            v = value if _num(value) else None
            if prev != "watching":
                sess["watch_eps"].append([now, now, v])
            else:
                ep = sess["watch_eps"][-1]
                ep[1] = now
                if v is not None:
                    ep[2] = v if ep[2] is None else max(ep[2], v)
        sess["_prev"] = state

    # ------------------------------------------------------------------
    # 護理站端操作
    # ------------------------------------------------------------------

    def ack(self, bed: str, nurse: str) -> dict:
        with self.lock:
            b = self.beds.get(bed)
            pend = b.get("pending") if b else None
            if not pend:
                return {"ok": False, "error": "這床沒有待確認的告警"}
            now = time.time()
            after = int(round(now - pend["since"]))
            for e in self.events:
                if e.get("eid") in pend["eids"]:
                    e["ack_by"], e["ack_after_s"] = nurse, after
            b["ack"] = {"by": nurse, "at": now, "after_s": after}
            b.pop("pending")
            self._log({"type": "station_ack", "bed": bed, "nurse": nurse,
                       "after_s": after, "ts": _iso(now)})
            return {"ok": True, "after_s": after}

    def _pause_active(self, b: dict, now: float) -> bool:
        p = b.get("pause")
        if p and p["until"] <= now:
            b.pop("pause")
            return False
        return bool(p)

    def pause(self, bed: str, minutes: float, nurse: str) -> dict:
        with self.lock:
            b = self.beds.get(bed)
            if b is None:
                return {"ok": False, "error": "沒有這床"}
            now = time.time()
            minutes = max(1.0, min(60.0, float(minutes)))
            entry = {"bed": bed, "by": nurse, "start": now, "until": now + minutes * 60,
                     "end": None, "session": b.get("session")}
            b["pause"] = {"until": entry["until"], "by": nurse, "minutes": minutes,
                          "entry": entry}
            self.care_log.append(entry)
            s = self.sessions.get(b.get("session"))
            if s is not None:
                s["pauses"] += 1
            self._log({"type": "station_pause", "bed": bed, "nurse": nurse,
                       "minutes": minutes, "ts": _iso(now)})
            return {"ok": True}

    def resume(self, bed: str) -> dict:
        with self.lock:
            b = self.beds.get(bed)
            p = b.pop("pause", None) if b is not None else None
            if p:
                now = time.time()
                p["entry"]["end"] = now
                self._log({"type": "station_resume", "bed": bed, "ts": _iso(now)})
            return {"ok": True}

    # ------------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------------

    def _display(self, b: dict, now: float) -> tuple[str, bool, bool]:
        """(畫面上的狀態, 在線, 照護暫停中)。優先順序：結束 > 離線 > 告警 > 照護 > 其他。"""
        online = now - b.get("_seen", 0) < OFFLINE_AFTER_S
        state = b.get("state", "unknown")
        paused = self._pause_active(b, now)
        if state == "ended":
            return "ended", online, paused
        if not online:
            return "offline", online, paused
        if b.get("pending") or state == "alert":
            return "alert", online, paused
        if paused:
            return "care", online, paused
        if state in ("unreliable", "watching", "baseline"):
            return state, online, paused
        return "normal", online, paused

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            beds = []
            for bed, b in sorted(self.beds.items()):
                display, online, paused = self._display(b, now)
                pend = b.get("pending")
                ack = b.get("ack")
                if ack and now - ack["at"] > ACK_SHOW_S:
                    ack = None
                beds.append({
                    "bed": bed,
                    "session": b.get("session"),
                    "online": online,
                    "age_s": round(now - b.get("_seen", 0), 1),
                    "state": b.get("state", "unknown"),
                    "display": display,
                    "state_since_s": round(now - b.get("_state_since", now)),
                    "visibility": b.get("visibility"),
                    "value": b.get("value"),
                    "uptime": b.get("uptime"),
                    "start": b.get("_start"),
                    "events": b.get("events", 0),
                    "spark": list(b.get("spark", [])),
                    "pending": None if not pend else {
                        "since_s": round(now - pend["since"]),
                        "count": len(pend["eids"]),
                        "channel": pend.get("channel"),
                        "peak": pend.get("peak"),
                        "escalated": now - pend["since"] > ESCALATE_AFTER_S,
                    },
                    "ack": None if not ack else {
                        "by": ack["by"], "after_s": ack["after_s"],
                        "ago_s": round(now - ack["at"]),
                    },
                    "pause": None if not paused else {
                        "remaining_s": round(b["pause"]["until"] - now),
                        "minutes": b["pause"]["minutes"],
                        "by": b["pause"]["by"],
                    },
                    # ★ 結束時還在告警中 —— 這床不是平安結束，要看得出來
                    "open_at_end": b.get("open_at_end", False),
                })
            return {
                "beds": beds,
                "now": now,
                "enter": self.enter,
                "escalate_after_s": ESCALATE_AFTER_S,
                "records_pending": sum(1 for sid, s in self.sessions.items()
                                       if s["end"] is not None and sid not in self.signed),
                "demo": self.demo,
            }

    def event_log(self) -> list[dict]:
        """本班事件：告警、觀察、無法判定、照護暫停，統一成同一種列（新 → 舊）。

        「本班」= 護理站啟動之後。觀察與無法判定是從心跳推出來的時段，
        不是床邊送來的事件 —— 它們只記錄、不通知。
        """
        with self.lock:
            now = time.time()
            rows: list[dict] = []
            pending = {e for b in self.beds.values() if b.get("pending")
                       for e in b["pending"]["eids"]}

            for e in self.events:
                fast = e.get("channel") == "fast"
                peak = e.get("peak")
                if fast:
                    text = "突然移動" + (f"，幅度 {peak:.2f}" if _num(peak) else "")
                else:
                    text = "手臂偏離基準" + (f" {peak:.2f}" if _num(peak) else "")
                    if e.get("outcome") == "open_at_session_end":
                        text += "，治療結束時仍持續"
                if e.get("ack_by"):
                    st = ("acked", f"{e['ack_by']} 已確認 · {e['ack_after_s']} 秒")
                elif e.get("paused"):
                    st = ("info", "照護中，未通知")
                elif e.get("eid") in pending:
                    st = ("pending", "未確認")
                else:
                    st = ("info", "—")
                dur = e.get("duration")
                rows.append({
                    "kind": "alert", "bed": e.get("bed"), "session": e.get("session"),
                    "t": e["_seen"], "text": text, "eid": e.get("eid"),
                    "channel": "快通道" if fast else "慢通道",
                    "dur": "" if dur is None else ("< 1 秒" if dur < 1 else f"{round(dur)} 秒"),
                    "status": st[0], "status_text": st[1],
                })

            for sid, s in self.sessions.items():
                bed = s["bed"]
                b = self.beds.get(bed, {})
                live = s["end"] is None and b.get("session") == sid
                slow_seen = [e["_seen"] for e in s["events"]
                             if e.get("channel") == "slow" and "_seen" in e]
                for i, (a, z, peak) in enumerate(s["watch_eps"]):
                    if a < self.started:
                        continue
                    if any(a - 5 <= t <= z + 15 for t in slow_seen):
                        continue                      # 後來升級成告警，告警那列已經有了
                    ongoing = (live and i == len(s["watch_eps"]) - 1
                               and b.get("state") == "watching")
                    secs = max(HEARTBEAT_S, z - a)
                    rows.append({
                        "kind": "watch", "bed": bed, "session": sid, "t": a,
                        "text": "偏離 " + (f"{peak:.2f}" if peak is not None else "—")
                                + ("，觀察中" if ongoing else f"，約 {round(secs)} 秒後自行回復"),
                        "channel": "慢通道", "dur": f"{round(secs)} 秒",
                        "status": "live" if ongoing else "info",
                        "status_text": "進行中" if ongoing else "僅記錄",
                    })
                for i, (a, z) in enumerate(s["unrel_eps"]):
                    if a < self.started:
                        continue
                    ongoing = (live and i == len(s["unrel_eps"]) - 1
                               and b.get("state") == "unreliable")
                    secs = max(HEARTBEAT_S, (now if ongoing else z) - a)
                    rows.append({
                        "kind": "unrel", "bed": bed, "session": sid, "t": a,
                        "text": "看不到穿刺側手臂" + ("，建議目視確認" if ongoing else "，已恢復"),
                        "channel": "—",
                        "dur": f"{round(secs / 60)} 分鐘" if secs >= 90 else f"{round(secs)} 秒",
                        "status": "live" if ongoing else "info",
                        "status_text": "進行中" if ongoing else "已恢復",
                    })

            for c in self.care_log:
                end = c["end"] or (c["until"] if c["until"] <= now else None)
                left = max(0, int(c["until"] - now))
                rows.append({
                    "kind": "care", "bed": c["bed"], "session": c["session"], "t": c["start"],
                    "text": f"{c['by']} 在床邊，暫停通知",
                    "channel": "—",
                    "dur": f"{max(1, round((end - c['start']) / 60))} 分鐘" if end else "",
                    "status": "live" if end is None else "info",
                    "status_text": f"進行中 · 剩 {left // 60}:{left % 60:02d}"
                                   if end is None else "已恢復",
                })

            rows.sort(key=lambda r: r["t"], reverse=True)
            return rows

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
                "video_streams": 0,
                "note": "對照：單路 1080p 串流約 4 Mbps；30 床約 120 Mbps",
                "demo": self.demo,
            }

    def report(self) -> dict:
        """報表與裝置頁：本班統計 + 各床連線 + 網路上實際傳了什麼。"""
        with self.lock:
            now = time.time()
            rows = []
            bed_hours = 0.0
            n_alert = unrel_hb = pause_hb = status_hb = 0
            for bed, b in sorted(self.beds.items()):
                display, online, _ = self._display(b, now)
                s = self.sessions.get(b.get("session"), {})
                evs = s.get("events", [])
                slow = sum(1 for e in evs if e.get("channel") == "slow")
                fast = sum(1 for e in evs if e.get("channel") == "fast")
                hb = s.get("status_n", 0)
                un, pa = s.get("unreliable_n", 0), s.get("paused_n", 0)
                bed_hours += hb * HEARTBEAT_S / 3600
                n_alert += slow + fast
                unrel_hb, pause_hb, status_hb = unrel_hb + un, pause_hb + pa, status_hb + hb
                rows.append({
                    "bed": bed, "online": online, "display": display,
                    "age_s": round(now - b.get("_seen", 0)),
                    "uptime": b.get("uptime"), "visibility": b.get("visibility"),
                    "alerts": slow, "fast": fast, "watch": len(s.get("watch_eps", [])),
                    "unrel_min": round(un * HEARTBEAT_S / 60, 1),
                    "coverage": None if not hb else round(1 - (un + pa) / hb, 3),
                })
            acked = [e["ack_after_s"] for e in self.events if e.get("ack_after_s") is not None]
            quiet = sum(1 for r in self.event_log()
                        if r["kind"] == "watch" and r["status"] == "info")
            llm = {"enabled": self.llm is not None}
            if self.llm is not None:
                done = list(self._llm_cache.values())
                llm.update(url=self.llm.url, model=self.llm.model_name,
                           ok=sum(1 for c in done if c.get("status") == "ok"),
                           fallback=sum(1 for c in done if c.get("status") == "fallback"))
            return {
                "since_min": round((now - self.started) / 60),
                "beds": rows,
                "kpi": {
                    "alerts": n_alert,
                    # 累積不到 1 床·時，比例會被一兩次告警放大成誤導的數字，先不顯示
                    "burden": round(n_alert / bed_hours, 2) if bed_hours >= 1 else None,
                    "bed_hours": round(bed_hours, 2),
                    "resp_avg": round(sum(acked) / len(acked)) if acked else None,
                    "resp_n": len(acked),
                    "coverage": None if not status_hb
                    else round(1 - (unrel_hb + pause_hb) / status_hb, 3),
                    "quiet": quiet,
                },
                "network": self.stats(),
                "sample_status": self.last_status,
                "sample_event": self.last_event,
                "llm": llm,
                "demo": self.demo,
            }

    # ------------------------------------------------------------------
    # 紀錄整理
    # ------------------------------------------------------------------

    def records(self) -> list[dict]:
        with self.lock:
            out = [list_item(sid, s, self.signed.get(sid))
                   for sid, s in self.sessions.items() if s["end"] is not None]
            out.sort(key=lambda r: (r["status"] == "signed", -r["end_epoch"]))
            return out

    def draft(self, sid: str, start_llm: bool = True) -> dict | None:
        """紀錄草稿。有接 LLM 時，第一次打開會在背景請 LLM 改寫（start_llm=False 只讀快取）。"""
        with self.lock:
            s = self.sessions.get(sid)
            if s is None:
                return None
            start = session_window(s)[0]
            history = sorted(
                (h for h in self.sessions.values()
                 if h["bed"] == s["bed"] and h["end"] is not None
                 and session_window(h)[1] <= start),
                key=lambda h: session_window(h)[0])
            d = build_draft(sid, s, history, self.signed.get(sid))
            if d["signed"]:
                d["meta"]["method"] = d["signed"].get("generated_by", d["meta"]["method"])
            if self.llm is None or d["signed"]:
                d["llm"] = {"status": "off" if self.llm is None else "skip"}
                return d
            c = self._llm_cache.get(sid)
            if c is None and not start_llm:
                d["llm"] = {"status": "off"}
                return d
            if c is None:
                c = self._llm_cache[sid] = {"status": "running", "model": self.llm.model_name}
                threading.Thread(target=self._llm_job, args=(sid, d), daemon=True).start()
            d["llm"] = dict(c)
            if c["status"] == "ok":
                d["template_headline"] = d["headline"]
                d["headline"] = c["text"]
                d["meta"]["method"] = f"LLM 改寫（{c['model']}，數字已核對）"
            return d

    def _llm_job(self, sid: str, d: dict) -> None:
        result = llm_rewrite(d, self.llm)            # 不持有 Store 的鎖：模型可能要跑十幾秒
        with self.lock:
            self._llm_cache[sid] = result
            self._log({"type": "llm_draft", "session": sid, "status": result["status"],
                       "model": result.get("model"), "ms": result.get("ms"),
                       "reason": result.get("reason"), "ts": _iso(time.time())})

    def regen(self, sid: str) -> dict:
        with self.lock:
            c = self._llm_cache.get(sid)
            if c and c.get("status") == "running":
                return {"ok": False, "error": "AI 還在起草中"}
            self._llm_cache.pop(sid, None)
            return {"ok": True}

    def sign(self, sid: str, text: str, actions: str, complaint: str,
             nurse: str) -> dict:
        if sid in self.signed:
            return {"ok": False, "error": "這個床次已經簽核過了"}
        draft = self.draft(sid, start_llm=False)
        if draft is None:
            return {"ok": False, "error": "找不到這個床次"}
        unmatched = fact_check(text, draft["facts"])
        # 護理師編輯過的文字也比一次關鍵主張 —— 打錯的次數不會只靠存在性檢查擋。
        # 兩者都只是記錄，不阻擋簽核：簽名的人才是最後的權威。
        wrong_claims = check_claims(text, draft.get("claims"))
        with self.lock:
            rec = {
                "type": "nursing_record", "session": sid, "bed": draft["bed"],
                "signed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "nurse": nurse, "summary": text,
                "actions": actions, "complaint": complaint,
                "events": [it["ref"] for it in draft["items"] if it["ref"].startswith("#")],
                "fact_check": {"ok": not unmatched and not wrong_claims,
                               "unmatched": unmatched,
                               "wrong_claims": wrong_claims},
                "generated_by": draft["meta"]["method"],
            }
            self.signed[sid] = rec
            with self.records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return {"ok": True, "fact_check": rec["fact_check"]}


class Handler(BaseHTTPRequestHandler):
    store: Store = None                      # 由 main 注入

    def log_message(self, *args):            # 關掉預設的逐筆 log
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _body(self) -> tuple[dict | None, int]:
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("payload 必須是 JSON 物件")
            return payload, len(raw)
        except Exception as e:
            self._json({"error": f"bad payload: {e}"}, 400)
            return None, 0

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body, size = self._body()
        if body is None:
            return
        s = self.store
        nurse = str(body.get("nurse") or "護理師")[:20]
        if path == "/ingest":
            s.ingest(body, size)
            self._json({"ok": True, "bytes": size})
        elif path == "/api/ack":
            self._json(s.ack(str(body.get("bed")), nurse))
        elif path == "/api/pause":
            self._json(s.pause(str(body.get("bed")), body.get("minutes", 10), nurse))
        elif path == "/api/resume":
            self._json(s.resume(str(body.get("bed"))))
        elif path == "/api/records/regen":
            self._json(s.regen(str(body.get("session"))))
        elif path == "/api/records/sign":
            self._json(s.sign(str(body.get("session")), str(body.get("text", "")),
                              str(body.get("actions", ""))[:2000],
                              str(body.get("complaint", ""))[:2000], nurse))
        else:
            self._json({"error": "not found"}, 404)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        s = self.store
        if path in ("/", "/records"):
            if PAGE.exists():
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"dashboard.html not found", "text/plain")
        elif path == "/api/state":
            self._json(s.snapshot())
        elif path == "/api/events":
            self._json(s.event_log())
        elif path == "/api/report":
            self._json(s.report())
        elif path == "/api/stats":
            self._json(s.stats())
        elif path == "/api/records":
            self._json(s.records())
        elif path.startswith("/api/records/"):
            draft = s.draft(unquote(path[len("/api/records/"):]))
            self._json(draft if draft else {"error": "not found"}, 200 if draft else 404)
        else:
            self._json({"error": "not found"}, 404)


def lan_ips() -> list[str]:
    """這台電腦在區域網路上的 IP（只問作業系統，不連外網）。"""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith(("127.", "169.254.")))


def main() -> None:
    ap = argparse.ArgumentParser(description="DialySight 護理站。完整說明：docs/USAGE.md")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--log", default="results/nurse_station.jsonl")
    ap.add_argument("--records", default="results/nursing_records.jsonl",
                    help="簽核後的護理紀錄")
    ap.add_argument("--enter", type=float, default=0.15,
                    help="趨勢圖上畫的門檻線（跟床邊 state_machine.enter_threshold 一致）")
    ap.add_argument("--demo", action="store_true",
                    help="模擬 8 床與幾份已結束的床次，沒有床邊主機也能操作")
    ap.add_argument("--llm", metavar="URL",
                    help="選用：hailo-ollama 的網址（例 http://localhost:8000），"
                         "紀錄草稿交給 Hailo-10H 上的 LLM 改寫")
    ap.add_argument("--llm-model", default="auto",
                    help="hailo-ollama 上的模型名稱；auto = 挑已下載的模型（Qwen 優先）")
    args = ap.parse_args()

    log, records = Path(args.log), Path(args.records)
    if args.demo:                                   # 示範資料不要寫進正式紀錄
        if args.log == ap.get_default("log"):
            log = log.with_name("nurse_station_demo.jsonl")
        if args.records == ap.get_default("records"):
            records = records.with_name("nursing_records_demo.jsonl")

    llm = LLMClient(args.llm, args.llm_model) if args.llm else None
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        sys.exit(f"開不了 port {args.port}（{e}）。可能已經有一個護理站在跑，"
                 f"關掉它或改用 --port 8421。")

    store = Store(log, records, enter=args.enter, llm=llm)
    Handler.store = store
    if args.demo:
        from src.station.demo import start_demo
        start_demo(store)

    ips = lan_ips()
    line = "-" * 64
    print(line)
    print("DialySight 護理站已啟動" + ("（示範模式：8 床為模擬資料）" if args.demo else ""))
    print(line)
    print(f"1. 用瀏覽器打開      http://localhost:{args.port}")
    for ip in ips:
        print(f"   同網路的其他電腦  http://{ip}:{args.port}")
    print("2. 床邊主機要把資料送到這裡：")
    print(f"   同一台電腦  不用改，config 預設就是 http://localhost:{args.port}/ingest")
    if ips:
        more = "  ← 有多個 IP 時，選跟床邊同一個網段的" if len(ips) > 1 else ""
        print(f"   別台電腦    床邊 config 的 sink.http 改成 "
              f"http://{ips[0]}:{args.port}/ingest{more}")
    if llm is None:
        print("3. AI 草稿     未啟用，紀錄草稿用範本。"
              "要接 Hailo LLM：加 --llm http://localhost:8000")
    else:
        try:
            names = llm.models()
            print(f"3. AI 草稿     已連上 hailo-ollama {llm.url}，模型："
                  + ("、".join(names) if names else "（還沒下載任何模型）"))
        except Exception as e:                       # noqa: BLE001 — 啟動檢查，失敗不擋
            print(f"3. AI 草稿     連不上 {llm.url}（{e}）。"
                  "草稿先用範本，連上後自動改用 AI")
    print(f"   事件紀錄 {log} · 簽核紀錄 {records}")
    print(line)
    print("Ctrl+C 結束")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n" + json.dumps(store.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
