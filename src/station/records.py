"""
治療結束後的紀錄草稿 —— 預設是範本填空；可選擇接上 Hailo-10H 上的 LLM 改寫成通順的句子。

★ 三道保護（紀錄整理頁右側寫的就是這三條）：

  1. 事實由程式決定
     時間、數字、次數全部直接取自床邊送來的事件文字與心跳。
     範本先把它們排成句子；有接 LLM 時，LLM 只負責把這段話改寫得更通順。

  2. 逐一核對
     fact_check() 把草稿裡出現的每個時間與數字，跟事件紀錄推得出來的數值逐一比對。
     LLM 的輸出只要有一個數字對不上（或用了沒辦法核對的中文數字、給了臨床建議），
     整份退回範本 —— 不會讓模型自己編的數字進到紀錄裡。
     護理師編輯草稿後也會重跑，打錯的數字會被標出來。

  3. 簽核才寫入
     這裡只產生草稿。寫進 nursing_records.jsonl 的是護理師簽核的版本，
     而且簽核時會記下當時的核對結果與產生方式（範本或哪一個模型）。

LLM 是選用的：護理站用 --llm 指向 hailo-ollama（Hailo 官方、Ollama 相容的 API 伺服器）
才會啟用。沒接、連不上、逾時，都自動用範本，紀錄整理照常運作。

只用標準函式庫。session 的資料結構由 scripts/nurse_station.py 的 Store 維護。
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

HEARTBEAT_S = 5.0          # 床邊預設心跳間隔，用來把「心跳數」換算成時間

FEATURE_LABEL = {
    "arm_displacement": "手臂偏離基準",
    "torso_angle": "軀幹角度變化",
    "arm_depth_delta": "深度方向位移",
    "bed_occupancy": "離床",
    "roi_redness": "穿刺處滲血疑慮",
}

_NUM = re.compile(r"\d{1,2}:\d{2}|\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    """草稿裡出現的所有時間（HH:MM）與數字，依出現順序。"""
    return _NUM.findall(text or "")


def fact_check(text: str, facts: set[str] | list[str]) -> list[str]:
    """回傳草稿中「事件紀錄裡找不到」的時間或數字。空串列 = 全部對得上。"""
    allowed = set(facts)
    return [tok for tok in numbers_in(text) if tok not in allowed]


# ★ fact_check 只檢查「這個數字在事實集合裡存在嗎」，不檢查「它被用在對的地方嗎」。
#
#   所以合法的數字可以被用在錯的位置而不被發現：
#     事實集合因為「約 3 分鐘無法判定」而含有 "3"
#     → 模型寫「告警 3 次」（實際 5 次）→ 每個 token 都在集合裡 → 通過
#
#   下面針對最有後果的幾個主張再比一次「數值對不對」。
#   這是**額外**的一層，不是完整的語意剖析 —— 它只認得列出來的句型，
#   認不出的句型仍然只有 fact_check 擋著，最後由護理師簽核把關。
CLAIM_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("alerts",     re.compile(r"告警\s*(?:共\s*)?(\d+)\s*次")),
    ("alerts",     re.compile(r"(\d+)\s*次\s*告警")),
    ("slow",       re.compile(r"漸進(?:性)?\s*(\d+)")),
    ("fast",       re.compile(r"突發\s*(\d+)")),
    ("dismissed",  re.compile(r"觀察\s*(\d+)\s*次")),
    ("unrel_min",  re.compile(r"(\d+)\s*分鐘?\s*(?:無法判定|看不到)")),
    ("unrel_min",  re.compile(r"無法判定[^\d]{0,6}(\d+)\s*分")),
]

CLAIM_LABEL = {"alerts": "告警次數", "slow": "漸進次數", "fast": "突發次數",
               "dismissed": "觀察次數", "unrel_min": "無法判定分鐘數"}


def check_claims(text: str, claims: Optional[dict]) -> list[str]:
    """回傳與事件紀錄矛盾的主張描述。空串列 = 沒抓到矛盾。"""
    if not claims:
        return []
    bad = []
    for key, pat in CLAIM_PATTERNS:
        if key not in claims:
            continue
        for m in pat.finditer(text or ""):
            said = int(m.group(1))
            if said != int(claims[key]):
                bad.append(f"{CLAIM_LABEL.get(key, key)}寫 {said}，"
                           f"實際 {claims[key]}")
    return sorted(set(bad))


def _hm(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M")


def _md(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%m/%d")


def _event_epoch(e: dict) -> float:
    ts = e.get("ts")
    if ts:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            pass
    return float(e.get("_seen", time.time()))


def _duration_text(seconds: float) -> str:
    s = int(round(seconds))
    h, m = divmod(s // 60, 60)
    if h:
        return f"{h} 小時 {m} 分"
    return f"{m} 分鐘" if m else f"{s} 秒"


def _sec(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    return "不足 1 秒" if seconds < 1 else f"{int(round(seconds))} 秒"


def session_window(s: dict) -> tuple[float, float]:
    """(開始, 結束) epoch。床次結束摘要裡的 duration 比「第一次收到心跳」準。"""
    end = s.get("end") or {}
    last = s["last"]
    dur = end.get("duration")
    if isinstance(dur, (int, float)) and dur > 0:
        return last - float(dur), last
    return s["first"], last


def alerts_of(s: dict) -> list[dict]:
    """這個床次確認過的事件（慢通道告警與快通道突發）。"""
    return [e for e in s.get("events", []) if e.get("channel") in ("slow", "fast")]


def list_item(sid: str, s: dict, signed: Optional[dict]) -> dict:
    start, end = session_window(s)
    alerts = alerts_of(s)
    end_payload = s.get("end") or {}
    unrel_min = round(s.get("unreliable_n", 0) * HEARTBEAT_S / 60)
    watch = end_payload.get("dismissed", len(s.get("watch_eps", [])))
    chips = []
    chips.append(f"告警 {len(alerts)}" if alerts else "未告警")
    if watch:
        chips.append(f"觀察 {watch}")
    if unrel_min:
        chips.append(f"遮蔽 {unrel_min} 分")
    if s.get("pauses"):
        chips.append(f"照護暫停 {s['pauses']}")
    return {
        "session": sid, "bed": s["bed"],
        "date": _md(start), "start_hm": _hm(start), "end_hm": _hm(end),
        "end_epoch": end, "alerts": len(alerts), "chips": chips,
        "status": "signed" if signed else "pending",
    }


def build_draft(sid: str, s: dict, history: list[dict],
                signed: Optional[dict] = None) -> dict:
    """單一床次的紀錄草稿。

    Args:
        s:       Store 裡的 session 字典
        history: 同一床位、更早結束的床次（舊 → 新），用來比較告警次數
    """
    t0 = time.perf_counter()
    start, end = session_window(s)
    end_payload = s.get("end") or {}

    def ev_epoch(e: dict) -> float:
        # ★ 用「床次開始 + t_offset」= 事件開始的時刻，不用 ts（那是確認／送出的時刻）。
        #   回放錄影時床邊跑得比實際快，ts 會擠在一起；t_offset 永遠是治療時間軸上的位置。
        off = e.get("t_offset")
        if end_payload.get("duration") and isinstance(off, (int, float)):
            return start + float(off)
        return _event_epoch(e)

    events = sorted(alerts_of(s), key=ev_epoch)
    n_slow = sum(1 for e in events if e.get("channel") == "slow")
    n_fast = len(events) - n_slow
    dismissed = end_payload.get("dismissed", len(s.get("watch_eps", [])))
    unrel_min = round(s.get("unreliable_n", 0) * HEARTBEAT_S / 60)

    # -- 一句話摘要 -----------------------------------------------------
    parts = []
    if events:
        kinds = "、".join(x for x in (f"漸進 {n_slow}" if n_slow else "",
                                      f"突發 {n_fast}" if n_fast else "") if x)
        head = f"告警 {len(events)} 次（{kinds}）"
        acked = [e for e in events if e.get("ack_after_s") is not None]
        if len(events) == 1 and acked:
            head += f"，{acked[0].get('ack_by') or '護理師'} 於 {acked[0]['ack_after_s']} 秒內確認"
        elif acked and len(acked) == len(events):
            head += f"，皆已確認（最慢 {max(e['ack_after_s'] for e in acked)} 秒）"
        elif acked:
            head += f"，其中 {len(acked)} 次已確認"
        else:
            head += "，護理站未留下確認紀錄"
        parts.append(head)
    else:
        parts.append("本次治療未告警")
    if dismissed:
        parts.append(f"觀察 {dismissed} 次皆未升級為告警" if dismissed > 1
                     else "觀察 1 次，未升級為告警")
    trend = sorted(
        [(w[0], w[2]) for w in s.get("watch_eps", []) if w[2] is not None]
        + [(ev_epoch(e), float(e["peak"])) for e in events
           if e.get("channel") == "slow" and isinstance(e.get("peak"), (int, float))])
    trend_vals = [round(v, 2) for _, v in trend]
    if len(trend_vals) >= 3 and all(b > a for a, b in zip(trend_vals, trend_vals[1:])):
        parts.append("偏離量逐次增加（" + " → ".join(f"{v:.2f}" for v in trend_vals) + "）")
    if unrel_min:
        parts.append(f"約 {unrel_min} 分鐘無法判定（看不到穿刺側手臂）")
    if end_payload.get("open_at_end"):
        parts.append("治療結束時仍在告警中")
    headline = "；".join(parts) + "。"

    # -- 重點事件 -------------------------------------------------------
    items = []
    for e in events:
        ep = ev_epoch(e)
        label = FEATURE_LABEL.get(e.get("feature"), e.get("feature") or "事件")
        peak = e.get("peak")
        if e.get("channel") == "fast":
            text = f"突然移動，幅度 {peak:.2f}" if isinstance(peak, (int, float)) else "突然移動"
            kind, tag = "fast", "突發"
        else:
            text = f"{label} {peak:.2f}" if isinstance(peak, (int, float)) else label
            if e.get("duration") is not None:
                text += f"，持續 {_sec(e['duration'])}"
            kind, tag = "alert", "告警"
        if e.get("ack_after_s") is not None:
            text += f"；{e.get('ack_by') or '護理師'} 於 {e['ack_after_s']} 秒內確認"
        elif e.get("paused"):
            text += "；照護中，未通知"
        if e.get("outcome") == "open_at_session_end":
            text += "；治療結束時仍持續"
        items.append({"epoch": ep, "time": _hm(ep), "kind": kind, "tag": tag,
                      "text": text, "ref": f"#E-{e.get('eid', 0):04d}"})
    for a, b, peak in s.get("watch_eps", []):
        if any(abs(ev_epoch(e) - a) < 60 for e in events):
            continue                       # 這次觀察後來升級成告警，不重複列
        dur = max(HEARTBEAT_S, b - a)
        text = (f"偏離 {peak:.2f}，約 {int(round(dur))} 秒後自行回復"
                if peak is not None else f"約 {int(round(dur))} 秒後自行回復")
        items.append({"epoch": a, "time": _hm(a), "kind": "watch", "tag": "觀察",
                      "text": text, "ref": "心跳"})
    for a, b in s.get("unrel_eps", []):
        if b - a < 30:
            continue
        items.append({"epoch": a, "time": _hm(a), "kind": "unreliable", "tag": "無法判定",
                      "text": f"看不到穿刺側手臂約 {max(1, round((b - a) / 60))} 分鐘",
                      "ref": "心跳"})
    items.sort(key=lambda x: x["epoch"], reverse=True)

    # -- 與前幾次治療比較 -----------------------------------------------
    prev = history[-3:]
    bars = [{"label": _md(session_window(h)[0]), "alerts": len(alerts_of(h))} for h in prev]
    bars.append({"label": "本次", "alerts": len(events), "current": True})
    if prev:
        history_text = (f"本次告警 {len(events)} 次；前 {len(prev)} 次共 "
                        f"{sum(b['alerts'] for b in bars[:-1])} 次。")
    else:
        history_text = "沒有同床位的先前紀錄可比較。"

    # -- 完整模式多看的統計 ---------------------------------------------
    stats = [
        ("治療時長", _duration_text(end - start)),
        ("確認事件", f"{len(events)} 筆（漸進 {n_slow}、突發 {n_fast}）"),
        ("觀察後未升級", f"{dismissed} 次"),
        ("無法判定", f"約 {unrel_min} 分鐘"),
    ]
    if end_payload.get("glitch_rejected") is not None:
        stats.append(("快通道擋下偵測跳點", f"{end_payload['glitch_rejected']} 次"))
    stats.append(("收到心跳", f"{s.get('status_n', 0)} 筆"))

    facts = set(numbers_in(headline) + numbers_in(history_text))
    for it in items:
        facts.update(numbers_in(it["time"] + " " + it["text"]))
    # 床號、治療時段、統計值也是事件紀錄推得出來的事實（LLM 改寫時可能會提到）
    facts.update(numbers_in(f"{s['bed']} {_hm(start)} {_hm(end)} {_duration_text(end - start)}"))
    for _, value in stats:
        facts.update(numbers_in(value))
    return {
        "session": sid, "bed": s["bed"],
        "date": datetime.fromtimestamp(start).strftime("%Y/%m/%d"),
        "start_hm": _hm(start), "end_hm": _hm(end),
        "duration_text": _duration_text(end - start),
        "status": "signed" if signed else "pending",
        "headline": headline,
        "items": [{k: v for k, v in it.items() if k != "epoch"} for it in items],
        "history": bars, "history_text": history_text,
        "stats": [{"label": a, "value": b} for a, b in stats],
        "facts": sorted(facts),
        # 事實集合管「這個數字存在嗎」，claims 管「這個數字用對了嗎」。見 check_claims。
        "claims": {"alerts": len(events), "slow": n_slow, "fast": n_fast,
                   "dismissed": dismissed, "unrel_min": unrel_min},
        "signed": signed,
        "meta": {"method": "範本填空", "inputs": len(events)
                 + len(s.get("watch_eps", [])) + len(s.get("unrel_eps", [])),
                 "gen_ms": round((time.perf_counter() - t0) * 1000, 2)},
    }


# --------------------------------------------------------------------------
# 選用：用 Hailo-10H 上的 LLM 把摘要改寫得更通順
# --------------------------------------------------------------------------

LLM_SYSTEM = (
    "你是透析室的護理紀錄助理。請把使用者提供的事件紀錄，改寫成一段通順的繁體中文治療摘要，"
    "給護理師簽核前參考。規則：\n"
    "1. 只能使用紀錄裡出現的時間與數字，照抄，不要換算、不要四捨五入；數字一律用阿拉伯數字。\n"
    "2. 不要推測原因，不要給任何臨床建議。\n"
    "3. 80 字以內，只輸出摘要本身，不要標題、不要條列。"
)
LLM_MAX_CHARS = 160
_THINK = re.compile(r"<think>.*?</think>", re.S)          # 推理型模型的思考段落
_CN_NUMBER = re.compile(r"[零一二兩三四五六七八九十百]+\s*(?:次|分|秒|小時|筆|床)")
_ADVICE = ("建議", "應該", "應立即", "需要加強", "宜")
_QUOTES = "「」『』\"' "


class LLMClient:
    """hailo-ollama（Ollama 相容 API）的最小用戶端。

    hailo-ollama 一次只處理一個請求，所以這裡用一把鎖排隊。
    不需要另外安裝套件 —— 只是 HTTP POST。
    """

    def __init__(self, url: str, model: str = "auto", timeout: float = 90.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.lock = threading.Lock()
        self._resolved: Optional[str] = None

    def _request(self, path: str, body: Optional[dict] = None,
                 timeout: Optional[float] = None) -> bytes:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self.url + path, data=data, method="GET" if body is None else "POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return r.read()

    @property
    def model_name(self) -> str:
        return self._resolved or self.model

    def models(self, timeout: float = 3.0) -> list[str]:
        """已經下載到 hailo-ollama 的模型名稱（GET /api/tags）。"""
        obj = json.loads(self._request("/api/tags", timeout=timeout))
        return [m.get("name") or m.get("model") for m in obj.get("models", [])
                if m.get("name") or m.get("model")]

    def resolve_model(self) -> str:
        """--llm-model auto：挑一個已下載的模型，Qwen 優先。"""
        if self.model != "auto":
            return self.model
        if self._resolved is None:
            names = self.models()
            if not names:
                raise RuntimeError("hailo-ollama 上還沒有下載任何模型")
            ranked = sorted(names, key=lambda n: (0 if "qwen2.5" in n.lower()
                                                  else 1 if "qwen" in n.lower() else 2, n))
            self._resolved = ranked[0]
        return self._resolved

    def chat(self, system: str, user: str) -> str:
        raw = self._request("/api/chat", {
            "model": self.resolve_model(),
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
        })
        return _chat_text(raw)


def _chat_text(raw: bytes) -> str:
    """接受單一 JSON，或伺服器仍以串流回傳時的多行 JSON / SSE。"""
    text = raw.decode("utf-8", "replace").strip()
    try:
        chunks = [json.loads(text)]
    except json.JSONDecodeError:
        chunks = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line.startswith("{"):
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    out = []
    for c in chunks:
        msg = c.get("message")
        if isinstance(msg, dict):
            out.append(msg.get("content") or "")
        elif "response" in c:
            out.append(c.get("response") or "")
        elif c.get("choices"):                                  # OpenAI 格式
            ch = c["choices"][0]
            out.append((ch.get("message") or ch.get("delta") or {}).get("content") or "")
    return "".join(out)


def llm_prompt(draft: dict) -> str:
    """送給 LLM 的只有這段文字 —— 床邊送來的事件整理，沒有影像、沒有座標。"""
    lines = [f"床位：{draft['bed']}",
             f"治療時段：{draft['start_hm']}–{draft['end_hm']}，共 {draft['duration_text']}",
             f"程式整理的摘要：{draft['headline']}",
             "事件："]
    lines += [f"- {it['time']} {it['tag']}：{it['text']}" for it in reversed(draft["items"])]
    if not draft["items"]:
        lines.append("- 無")
    lines.append(f"與前幾次治療比較：{draft['history_text']}")
    return "\n".join(lines)


def check_llm_text(text: str, facts, claims: Optional[dict] = None) -> Optional[str]:
    """LLM 輸出能不能用。可以用回傳 None，不行回傳原因（會顯示在紀錄整理頁）。

    claims 省略時只做 fact_check 的存在性檢查（舊行為）；
    帶進來時會再比一次關鍵主張的數值，見 check_claims。
    """
    if not text:
        return "模型沒有回應內容"
    if len(text) > LLM_MAX_CHARS:
        return f"輸出太長（{len(text)} 字）"
    bad = fact_check(text, facts)
    if bad:
        return "數字對不上事件紀錄：" + "、".join(bad)
    wrong = check_claims(text, claims)
    if wrong:
        return "數字用錯位置：" + "；".join(wrong)
    if _CN_NUMBER.search(text):
        return "用了中文數字，無法逐一核對"
    hit = next((w for w in _ADVICE if w in text), None)
    if hit:
        return f"含臨床建議用語「{hit}」"
    return None


def llm_rewrite(draft: dict, client: LLMClient) -> dict:
    """請 LLM 改寫摘要並核對。回傳 {status: ok|fallback, text, reason, model, ms}。"""
    t0 = time.perf_counter()
    model = client.model

    def ms() -> int:
        return round((time.perf_counter() - t0) * 1000)

    try:
        with client.lock:
            model = client.resolve_model()
            text = client.chat(LLM_SYSTEM, llm_prompt(draft))
    except urllib.error.HTTPError as e:
        return {"status": "fallback", "model": model, "ms": ms(),
                "reason": f"hailo-ollama 回應錯誤 {e.code}（模型名稱對嗎？用 --llm-model 指定）"}
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"status": "fallback", "model": model, "ms": ms(),
                "reason": f"連不上 hailo-ollama（{e}）"}
    except (RuntimeError, ValueError, KeyError) as e:
        return {"status": "fallback", "model": model, "ms": ms(), "reason": str(e)}
    text = " ".join(_THINK.sub("", text).split()).strip(_QUOTES)
    reason = check_llm_text(text, draft["facts"], draft.get("claims"))
    if reason:
        return {"status": "fallback", "model": model, "ms": ms(), "reason": reason,
                "raw": text}
    return {"status": "ok", "model": model, "ms": ms(), "text": text}
