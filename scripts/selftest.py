#!/usr/bin/env python3
"""
自我測試 —— 不需要影片、不需要硬體，驗證整套系統有沒有裝對。

用法：
    python scripts/selftest.py

跑完會告訴你哪些子系統正常、哪些缺套件。
全部通過代表程式碼沒問題，接下來只差素材。

★ 這支腳本也是給評審看的：它證明核心邏輯可以在任何機器上重現，
  不依賴我們的影片或硬體。
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL, SKIP = "✓", "✗", "–"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool | None, detail: str = "") -> None:
    mark = SKIP if ok is None else (PASS if ok else FAIL)
    results.append((mark, name, detail))
    print(f"  {mark} {name:<34}{detail}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * 62}")


# --------------------------------------------------------------------------

def test_imports() -> bool:
    section("1. 模組載入")
    ok = True
    for mod, label in [
        ("src.inference.base", "推論契約 base.py"),
        ("src.inference.subject", "受試者選擇 subject.py"),
        ("src.temporal.filter", "層 0 filter.py"),
        ("src.temporal.features", "層 1 features.py"),
        ("src.temporal.smooth", "層 2 smooth.py"),
        ("src.temporal.window", "層 3 window.py"),
        ("src.temporal.state_machine", "層 4 state_machine.py（慢通道）"),
        ("src.temporal.fast_channel", "層 4 fast_channel.py（快通道）"),
        ("src.fusion", "層 5 fusion.py"),
        ("src.pipeline", "管線 pipeline.py"),
        ("src.report", "紀錄 report.py"),
        ("src.sink", "輸出層 sink.py"),
    ]:
        try:
            __import__(mod)
            record(label, True)
        except Exception as e:
            record(label, False, str(e)[:40])
            ok = False

    for mod, label in [("cv2", "OpenCV（視覺化需要）"),
                       ("matplotlib", "matplotlib（診斷圖需要）"),
                       ("ultralytics", "ultralytics（實際推論需要）"),
                       ("yaml", "PyYAML（讀 config 需要）")]:
        try:
            __import__(mod)
            record(label, True)
        except ImportError:
            record(label, None, "未安裝")
    return ok


def test_window() -> bool:
    section("2. 滑動窗 O(1) 統計")
    from src.temporal.window import SlidingWindow

    w = SlidingWindow(duration_s=10.0)
    for i in range(200):
        w.push(i * 0.1, 0.01 * i * 0.1)
    s = w.stats()

    slope_ok = abs(s.slope - 0.01) < 1e-4
    evict_ok = s.n <= 101
    record("趨勢斜率計算", slope_ok, f"slope={s.slope:.5f}（應為 0.01000）")
    record("窗長淘汰", evict_ok, f"n={s.n}（10 秒 @10FPS 應 ≤ 101）")
    return slope_ok and evict_ok


def test_state_machine() -> bool:
    section("3. 狀態機 — 偽陽性對照 ★ 核心論點")
    from src.temporal.state_machine import (
        HysteresisStateMachine, SingleThresholdStateMachine, StateMachineConfig,
    )

    cfg = StateMachineConfig(enter_threshold=0.15, exit_threshold=0.10,
                             t_enter=1.0, t_confirm=5.0, t_exit=3.0, cooldown=30.0)
    ours = HysteresisStateMachine("arm_displacement", cfg)
    base = SingleThresholdStateMachine("arm_displacement", 0.15)

    random.seed(0)
    confirmed = 0
    for i in range(1200):                       # 120 秒 @10FPS
        t = i * 0.1
        if 40.0 <= t <= 70.0:                   # 真事件：30 秒漸進漂移
            v = 0.10 + 0.012 * (t - 40.0)
        else:                                   # 其餘：2% 機率雜訊尖峰
            v = 0.05 + (0.25 if random.random() < 0.02 else 0.0)
        if ours.update(t, v) is not None:
            confirmed += 1
        base.update(t, v)

    n_base = len(base.confirmed_events())
    caught = confirmed >= 1
    reduced = confirmed < n_base / 3

    record("對照組（單閾值）告警", True, f"{n_base} 次")
    record("實驗組（時序版）告警", True, f"{confirmed} 次")
    record("真事件有抓到", caught)
    record("偽陽性大幅下降", reduced,
           f"降幅 {100 * (1 - confirmed / n_base):.1f}%" if n_base else "")
    return caught and reduced


def test_fast_channel() -> bool:
    section("4. 快通道 — 突發偵測 vs 偵測跳點 ★")
    from src.temporal.fast_channel import FastChannel, FastChannelConfig

    cfg = FastChannelConfig(window_s=1.0, amplitude_threshold=0.12,
                            min_samples=5, t_persist=0.8, cooldown=15.0)

    def drive(shape) -> tuple[int, int]:
        """回傳 (確認的事件數, 被當成跳點擋下的次數)。"""
        fc = FastChannel(cfg)
        n = 0
        for i in range(600):                 # 60 秒 @10FPS
            t = i * 0.1
            if fc.update(t, shape(t)) is not None:
                n += 1
        return n, fc.report()["glitch_rejected"]

    # (a) 真實突發：20 秒時手臂跳到新位置，然後停在那裡
    real, real_rej = drive(lambda t: 0.20 if t >= 20.0 else 0.02)

    # (b) 偵測跳點：單格暴衝，下一格就回來
    def glitch(t):
        return 0.30 if abs(t - 20.0) < 0.05 else 0.02
    fake, fake_rej = drive(glitch)

    # (c) 緩慢漂移：60 秒走完 0.20，快通道**不該**叫（那是慢通道的事）
    slow, _ = drive(lambda t: 0.02 + 0.003 * t)

    # (d) ★ 回歸測試：參考值必須取自振幅窗「之前」，不能取窗內前半段。
    #
    #     實際踩過的 bug：觸發時機偏晚時，振幅窗已經橫跨跳變點，
    #     前半段中位數落在跳變前後之間（實測 0.2269，真值 0.13），
    #     signed 位移因此算小 → 真實的持續位移被算進 glitch_rejected。
    #
    #     直接檢查機制，不靠時序湊巧：讓振幅窗內**完全沒有**跳變前的值。
    #     參考值若是從窗內算的，必然 ≥ 0.30；只有從窗前的參考區取，
    #     才會得到真正的靜止水準 0.13。任何「從窗內取參考值」的寫法都會紅。
    fc = FastChannel(FastChannelConfig(window_s=1.0, amplitude_threshold=0.12,
                                       min_samples=5, min_ref_samples=4,
                                       ref_window_s=2.0, t_persist=0.8))
    REST = 0.13

    def feed(fc, t, v):
        """餵一格，並在「剛觸發驗證」的那一刻回傳當時的參考值。"""
        before = fc._pending
        fc.update(t, v)
        return (fc._pending.reference
                if before is None and fc._pending is not None else None)

    captured = None
    for i in range(40):                       # 0.0–3.9 秒靜止，填滿參考區
        feed(fc, i * 0.1, REST)

    # ★ 關鍵：觸發要被延後到振幅窗內已經沒有跳變前的值。
    #   真實世界造成延後的原因是前一次事件的冷卻期 —— 這裡直接設定
    #   冷卻結束時間來模擬那個情境（這就是實際出 bug 的條件）。
    fc._cooldown_until = 5.0

    # 4.0 秒起手臂已經在外側並持續往外移（0.30 → 0.45）
    for n, i in enumerate(range(40, 51)):
        r = feed(fc, i * 0.1, 0.30 + 0.015 * n)
        if r is not None and captured is None:
            captured = r
    for i in range(51, 75):                   # 維持在外側
        feed(fc, i * 0.1, 0.45)

    ref_ok = captured is not None and abs(captured - REST) < 1e-6
    ref_got = f"{captured:.4f}" if captured is not None else "未觸發"
    post_rej = fc.report()["glitch_rejected"]

    record("真實突發有抓到", real >= 1, f"{real} 次")
    record("偵測跳點被擋下 ★", fake == 0 and fake_rej >= 1,
           f"告警 {fake} 次，擋下 {fake_rej} 次")
    record("緩慢漂移不誤觸", slow == 0,
           f"{slow} 次（這段該由慢通道處理）")
    record("參考值取自振幅窗之前 ★", ref_ok,
           f"ref={ref_got}（應為 {REST}；從窗內取必 ≥ 0.30）")
    record("持續位移不算跳點", post_rej == 0, f"誤判 {post_rej} 次")

    return (real >= 1 and fake == 0 and fake_rej >= 1
            and slow == 0 and ref_ok and post_rej == 0)


def test_fusion() -> bool:
    section("5. 融合層 — 遮蔽時說「不知道」而非「正常」★")
    from src.fusion import ConsensusFusion, FusionConfig, Reliability, default_signals

    fuse = ConsensusFusion(FusionConfig(enabled=True, signals=default_signals(False)))

    # ★ 閘門是「每個特徵自己依賴的關鍵點」的可見率，不是全身 17 點。
    #   躺著蓋被子的病人膝蓋腳踝永遠看不到，用全身可見率會永遠判成不可信。
    full = {"arm_displacement_visibility": 1.0, "torso_angle_visibility": 1.0}

    clear_normal = fuse.update({"arm_displacement": 0.02, "torso_angle": 2.0, **full})
    clear_alarm = fuse.update({"arm_displacement": 0.30, "torso_angle": 35.0, **full})
    # 全部遮蔽：兩個特徵的關鍵點都不見了
    occluded = fuse.update({"arm_displacement": float("nan"),
                            "torso_angle": float("nan"),
                            "arm_displacement_visibility": 0.0,
                            "torso_angle_visibility": 0.0})
    # 被子蓋住手臂：手腕不見，軀幹四點還在 → 軀幹照常投票
    partial = fuse.update({"arm_displacement": float("nan"),
                           "torso_angle": 30.0,
                           "arm_displacement_visibility": 0.5,
                           "torso_angle_visibility": 1.0})

    ok_normal = clear_normal.reliability is Reliability.OK and clear_normal.risk < 0.2
    ok_alarm = clear_alarm.reliability is Reliability.OK and clear_alarm.risk > 0.8
    # ★ 最重要的一條：遮蔽時不可以回報「正常」
    ok_unknown = (occluded.reliability is Reliability.UNRELIABLE
                  and not occluded.usable)
    ok_partial = partial.reliability is not Reliability.OK and partial.usable

    record("清楚畫面 · 正常", ok_normal,
           f"risk={clear_normal.risk:.2f} conf={clear_normal.confidence:.2f}")
    record("清楚畫面 · 異常", ok_alarm,
           f"risk={clear_alarm.risk:.2f} conf={clear_alarm.confidence:.2f}")
    record("全遮蔽 → UNRELIABLE ★", ok_unknown,
           f"{occluded.reliability.value}（不是 normal）")
    record("被子蓋手臂 → 軀幹照常投票", ok_partial,
           f"{partial.reliability.value} conf={partial.confidence:.2f}")

    # required 開關：主訊號不可用時直接判為 UNRELIABLE（預設關閉）
    sigs = default_signals(False)
    sigs[0].required = True
    strict = ConsensusFusion(FusionConfig(enabled=True, signals=sigs))
    s_covered = strict.update({"arm_displacement": float("nan"),
                               "torso_angle": 30.0,
                               "arm_displacement_visibility": 0.0,
                               "torso_angle_visibility": 1.0})
    s_clear = strict.update({"arm_displacement": 0.30, "torso_angle": 30.0, **full})
    ok_req = (s_covered.reliability is Reliability.UNRELIABLE
              and s_clear.reliability is Reliability.OK)
    record("required 開關可改成嚴格模式", ok_req,
           "開啟後蓋住手臂即判定不可信（預設關閉，待 10/9 實測後決定）")

    return ok_normal and ok_alarm and ok_unknown and ok_partial and ok_req


def test_subject() -> bool:
    section("6. 受試者選擇 — 多人畫面中認出病人")
    from src.inference.base import NUM_KEYPOINTS, KP, Keypoint
    from src.inference.subject import BedROI, SubjectConfig, SubjectSelector

    def person(sx, sy, hx, hy):
        k = [Keypoint(0.0, 0.0, 0.0) for _ in range(NUM_KEYPOINTS)]
        k[int(KP.LEFT_SHOULDER)] = Keypoint(sx - .03, sy, .9)
        k[int(KP.RIGHT_SHOULDER)] = Keypoint(sx + .03, sy, .9)
        k[int(KP.LEFT_HIP)] = Keypoint(hx - .03, hy, .9)
        k[int(KP.RIGHT_HIP)] = Keypoint(hx + .03, hy, .9)
        return k

    sel = SubjectSelector(SubjectConfig(
        roi=BedROI(0.15, 0.35, 0.85, 0.75), max_torso_angle_deg=45.0))

    patient = person(.30, .52, .60, .55)        # 橫躺
    nurse = person(.90, .25, .90, .62)          # 站立，床區外
    leaning = person(.55, .20, .58, .50)        # 彎腰，髖部進床區

    cases = [
        ("只有病人", [patient], True),
        ("病人 + 站立護理師", [patient, nurse], True),
        ("病人 + 彎腰護理師", [patient, leaning], True),
        ("只有護理師（病人離床）", [nurse], False),
        ("沒有人", [], False),
    ]
    ok = True
    for label, people, should_find in cases:
        kps, _ = sel.select(people)
        got = kps is not None
        ok &= (got == should_find)
        record(label, got == should_find, "選中病人" if got else "回傳 None")
    return ok


def test_color() -> bool:
    section("7. 滲血偵測 — 單調性鑑別器")
    try:
        import cv2
        import numpy as np
    except ImportError:
        record("需要 OpenCV", None, "跳過")
        return True

    from src.inference.base import NUM_KEYPOINTS, KP, FrameObservation, Keypoint
    from src.temporal.color import BleedingMonitor, ColorConfig

    H, W = 240, 320

    def frame(blood=0.0, cover=False):
        img = np.full((H, W, 3), (40, 45, 50), np.uint8)
        cv2.rectangle(img, (50, 100), (260, 140), (120, 150, 190), -1)
        if blood > 0:
            ov = img.copy(); cv2.circle(ov, (215, 120), 13, (60, 60, 200), -1)
            img = cv2.addWeighted(ov, blood, img, 1 - blood, 0)
        if cover:
            ov = img.copy(); cv2.rectangle(ov, (160, 95), (270, 145), (70, 70, 190), -1)
            img = cv2.addWeighted(ov, .85, img, .15, 0)
        return img

    def obs(t, **kw):
        k = [Keypoint(0, 0, 0.0) for _ in range(NUM_KEYPOINTS)]
        k[int(KP.RIGHT_SHOULDER)] = Keypoint(70 / W, 120 / H, .9)
        k[int(KP.RIGHT_ELBOW)] = Keypoint(155 / W, 120 / H, .9)
        k[int(KP.RIGHT_WRIST)] = Keypoint(240 / W, 120 / H, .9)
        return FrameObservation(int(t * 10), t, k, raw_frame=frame(**kw))

    def run(fn):
        m = BleedingMonitor(ColorConfig(wrist_side="right"), baseline_frames=100)
        out = []
        for i in range(900):
            t = i * 0.1
            f = m.update(obs(t, **fn(t)))
            if f["roi_redness"] == f["roi_redness"]:
                out.append(f)
        return out[-1] if out else None

    bleed = run(lambda t: {"blood": min(.6, max(0, (t - 30) * .012))})
    blanket = run(lambda t: {"cover": t > 40})

    ok = True
    if bleed and blanket:
        mono_sep = bleed["redness_monotonic"] > 0.8 and blanket["redness_monotonic"] < 0.7
        record("真實滲血", True,
               f"紅度 {bleed['roi_redness']:.1f}  單調性 {bleed['redness_monotonic']:.2f}")
        record("被子蓋前臂（持續）", True,
               f"紅度 {blanket['roi_redness']:.1f}  單調性 {blanket['redness_monotonic']:.2f}")
        record("單調性能區分兩者 ★", mono_sep,
               "被子紅度更高但單調性低" if mono_sep else "")
        ok = mono_sep
    else:
        record("顏色量測", False, "沒有有效輸出")
        ok = False
    return ok


def test_sink() -> bool:
    section("8. 事件輸出 — 影像不出床邊的程式碼保證")
    from src.sink import JSONLSink, PayloadTooLarge, event_payload, payload_size
    from src.temporal.state_machine import Event

    ev = Event(feature="arm_displacement", start_time=77.4, end_time=98.4,
               peak_value=0.3172, confirmed_at=82.4, channel="slow")
    p = event_payload(ev, "Bed07", "S1")
    size = payload_size(p)

    record("單筆 payload 大小", size < 300, f"{size} bytes")

    # 寫入
    with tempfile.TemporaryDirectory() as d:
        with JSONLSink(Path(d) / "ev.jsonl") as sink:
            sink.emit(p)
        lines = (Path(d) / "ev.jsonl").read_text(encoding="utf-8").strip().split("\n")
        record("JSONL 寫入", len(lines) == 1, f"{len(lines)} 行")

    # 保護機制：夾帶座標陣列要被擋下
    #
    # ⚠️ close() 一定要放在 finally。emit() 擋下壞 payload 時會直接拋例外，
    #    寫在 emit 後面的 close() 永遠跑不到，檔案句柄就留著沒關。
    #    Linux 可以刪開著的檔案所以看不出問題，Windows 會在暫存資料夾
    #    清理時噴 WinError 32，把真正的 PayloadTooLarge 蓋掉。
    blocked = False
    bad = dict(p)
    bad["keypoints"] = [[0.1, 0.2, 0.9]] * 17
    with tempfile.TemporaryDirectory() as d:
        try:
            # EventSink 本身就支援 with，用它就不會漏關
            with JSONLSink(Path(d) / "x.jsonl") as s:
                s.emit(bad)
        except PayloadTooLarge:
            blocked = True
    record("夾帶座標被擋下 ★", blocked, "程式強制，不靠自律")

    # 30 床推估
    per_bed_hour = 3600 / 5 + 3                 # 心跳每 5 秒 + 約 3 事件
    total = size * per_bed_hour * 30
    kbps = total * 8 / 3600 / 1000
    record("30 床推估流量", True,
           f"{kbps:.1f} kbps（對照 30 路 1080p ≈ 120000 kbps）")
    return size < 300 and blocked


def test_pipeline_end_to_end() -> bool:
    section("9. 端到端管線 ★ 事件真的有落到檔案上")

    from src.inference.base import NUM_KEYPOINTS, KP, Keypoint
    from src.pipeline import Pipeline

    # 造一段假的關鍵點序列：前 40 秒安靜，之後手腕漸進外移 30 秒。
    # 用 replay backend 跑，不需要影片、不需要模型。
    fps = 10.0
    frames = []
    for i in range(900):                         # 90 秒
        t = i / fps
        drift = 0.0 if t < 40.0 else min(0.22, 0.009 * (t - 40.0))
        k = [[0.0, 0.0, 0.0] for _ in range(NUM_KEYPOINTS)]
        k[int(KP.LEFT_SHOULDER)] = [0.30, 0.47, 0.9]
        k[int(KP.RIGHT_SHOULDER)] = [0.30, 0.53, 0.9]
        k[int(KP.LEFT_HIP)] = [0.60, 0.48, 0.9]
        k[int(KP.RIGHT_HIP)] = [0.60, 0.54, 0.9]
        k[int(KP.RIGHT_ELBOW)] = [0.42, 0.56, 0.9]
        k[int(KP.RIGHT_WRIST)] = [0.42 + drift, 0.58, 0.9]
        frames.append({"frame_idx": i, "timestamp": t, "keypoints": k})

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        kp_json = d / "kp.json"
        kp_json.write_text(json.dumps({"fps": fps, "frames": frames}),
                           encoding="utf-8")

        cfg = {
            "inference": {"backend": "replay", "keypoints_json": str(kp_json)},
            "filter": {"min_confidence": 0.5, "min_visibility_ratio": 0.0},
            "features": {"wrist_side": "right",
                         "enabled": ["arm_displacement", "torso_angle"]},
            "baseline": {"duration_s": 20.0, "min_valid_frames": 50},
            "smooth": {"default": {"method": "median", "window": 7}},
            "windows": {"short": 5.0, "medium": 60.0, "long": 900.0},
            "state_machine": {"enter_threshold": 0.08, "exit_threshold": 0.05,
                              "t_enter": 1.0, "t_confirm": 5.0,
                              "t_exit": 3.0, "cooldown": 30.0},
            "fast_channel": {"enabled": True},
            "fusion": {"enabled": True, "drive_state_machine": False},
            "color": {"enabled": False},
            "sink": {"bed_id": "TestBed",
                     "jsonl": str(d / "events_{bed}.jsonl"),
                     "status_interval_s": 5.0},
        }

        # try/finally：跑到一半出錯時 sink 也要關掉。
        # 不然 Windows 清理暫存資料夾會噴 WinError 32，把真正的錯誤蓋掉。
        pipe = Pipeline(cfg, fps_hint=fps)
        n_frames = 0
        saw_fusion = False
        try:
            for r in pipe.run(None):
                n_frames += 1
                if r.fusion is not None:
                    saw_fusion = True
            summary = pipe.summary()
        finally:
            pipe.close()

        out = d / "events_TestBed.jsonl"
        lines = (out.read_text(encoding="utf-8").strip().split("\n")
                 if out.exists() else [])
        records = [json.loads(l) for l in lines if l]

    kinds = {r.get("type", "event") for r in records}
    n_event = sum(1 for r in records if r.get("type") is None)
    n_status = sum(1 for r in records if r.get("type") == "status")

    # 每一筆都必須是扁平的純量 —— 這就是「影像不出床邊」的實證
    flat = all(
        not isinstance(v, (dict, list, tuple))
        for r in records for v in r.values()
    )
    biggest = max((len(json.dumps(r, ensure_ascii=False)) for r in records),
                  default=0)

    ok_run = n_frames == 900
    ok_baseline = summary["baseline"]["ready"]
    ok_alert = summary["alerts"] >= 1
    ok_file = n_event >= 1 and n_status >= 1 and "session_end" in kinds

    record("管線跑完全部影格", ok_run, f"{n_frames} 格")
    record("基準線建立（秒為單位）", ok_baseline,
           f"duration_s={summary['baseline']['duration_s']}")
    record("漸進事件有抓到", ok_alert,
           f"慢 {summary['alerts_slow']} / 快 {summary['alerts_fast']}")
    record("事件落到 JSONL ★", ok_file,
           f"{n_event} 事件 + {n_status} 心跳 + 收尾摘要")
    record("融合層有輸出", saw_fusion,
           f"不可信比例 {summary['fusion']['unreliable_rate'] * 100:.1f}%")
    record("每筆都是扁平純量 ★", flat, f"最大 {biggest} bytes")

    return ok_run and ok_baseline and ok_alert and ok_file and saw_fusion and flat


def test_render() -> bool:
    section("10. 視覺化")
    try:
        import cv2
        import numpy as np
    except ImportError:
        record("需要 OpenCV", None, "跳過")
        return True

    from src.inference.base import NUM_KEYPOINTS, KP, FrameObservation, Keypoint
    from src.temporal.state_machine import State
    from src.visualize import DemoRenderer, LayoutConfig

    H, W = 240, 320
    k = [Keypoint(0, 0, 0.0) for _ in range(NUM_KEYPOINTS)]
    k[int(KP.RIGHT_SHOULDER)] = Keypoint(.25, .5, .9)
    k[int(KP.LEFT_SHOULDER)] = Keypoint(.25, .44, .9)
    k[int(KP.RIGHT_ELBOW)] = Keypoint(.5, .5, .9)
    k[int(KP.RIGHT_WRIST)] = Keypoint(.75, .5, .9)
    k[int(KP.RIGHT_HIP)] = Keypoint(.28, .63, .9)
    k[int(KP.LEFT_HIP)] = Keypoint(.28, .58, .9)
    frame = np.full((H, W, 3), (40, 45, 50), np.uint8)

    r = DemoRenderer(layout=LayoutConfig(), thresholds={"enter": .15, "exit": .10})
    canvas = None
    for i in range(60):
        t = i * 0.1
        canvas = r.render(FrameObservation(i, t, k, raw_frame=frame),
                          {"arm_displacement": .05 + t * .01}, State.NORMAL, None, t)

    ok = canvas is not None and canvas.shape == (720, 1280, 3)
    out = Path("results/selftest_render.png")
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
    record("合成畫面", ok, f"{canvas.shape if canvas is not None else '—'} → {out}")
    return ok


def test_report() -> bool:
    section("11. 事件紀錄範本")
    from src.report import session_report
    from src.temporal.state_machine import Event

    evs = [
        Event("arm_displacement", 77.4, 98.4, .317, 82.4, "slow",
              {"outcome": "resolved"}),
        Event("arm_displacement", 5000.0, 5000.6, .402, 5000.0, "fast",
              {"outcome": "confirmed"}),
        Event("arm_displacement", 9200.0, 14400.0, .289, 9205.0, "slow",
              {"outcome": "open_at_session_end"}),
    ]
    text = session_report(evs, 14400.0, "Bed 07")

    ok_count = "事件總數：3" in text
    ok_label = "上肢姿態異常" in text
    ok_chan = "漸進" in text and "突發" in text
    ok_sub = "不足 1 秒" in text               # 快通道事件不能寫成「持續 0 秒」
    ok_open = "治療結束時仍持續" in text         # ★ 未解除的事件要標出來
    ok_len = "4.0 小時" in text

    record("事件計數與分類", ok_count and ok_chan, "漸進 2、突發 1")
    record("秒級事件不寫成 0 秒", ok_sub, "顯示「不足 1 秒」")
    record("未解除事件有標記 ★", ok_open, "不等於 resolved")
    record("治療時長單位正確", ok_len, "4.0 小時")

    ok = ok_count and ok_label and ok_chan and ok_sub and ok_open and ok_len
    if ok:
        print("\n" + "\n".join("    " + l for l in text.split("\n")[:10]) + "\n    ...")
    return ok


# --------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("  DialySight 自我測試 — 不需要影片，不需要硬體")
    print("=" * 62)

    t0 = time.perf_counter()
    checks = [test_imports, test_window, test_state_machine,
              test_fast_channel, test_fusion, test_subject,
              test_color, test_sink, test_pipeline_end_to_end,
              test_render, test_report]
    all_ok = True
    for fn in checks:
        try:
            all_ok &= bool(fn())
        except Exception as e:
            record(f"{fn.__name__} 發生例外", False, str(e)[:48])
            all_ok = False

    n_pass = sum(1 for m, _, _ in results if m == PASS)
    n_fail = sum(1 for m, _, _ in results if m == FAIL)
    n_skip = sum(1 for m, _, _ in results if m == SKIP)

    print("\n" + "=" * 62)
    print(f"  通過 {n_pass}　失敗 {n_fail}　跳過 {n_skip}"
          f"　（{time.perf_counter() - t0:.1f}s）")
    print("=" * 62)

    if all_ok and n_fail == 0:
        print("""
  程式碼全部正常。接下來只差素材：

    1. 拍 5 分鐘粗拍（手機即可，相機要固定）
    2. python scripts/dump_keypoints.py --video <影片> --out <json>
    3. python scripts/plot_feature.py --json <json> --out results/diag.png
    4. python scripts/run_demo.py --video <影片> --show
""")
    else:
        print("\n  有項目失敗，先修掉再往下走。\n")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
