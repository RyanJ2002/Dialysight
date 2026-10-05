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
        ("src.temporal.state_machine", "層 4 state_machine.py"),
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


def test_subject() -> bool:
    section("4. 受試者選擇 — 多人畫面中認出病人")
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
    section("5. 滲血偵測 — 單調性鑑別器")
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
    section("6. 事件輸出 — 影像不出床邊的程式碼保證")
    from src.sink import JSONLSink, PayloadTooLarge, event_payload, payload_size
    from src.temporal.state_machine import Event

    ev = Event(feature="arm_displacement", start_time=77.4, end_time=98.4,
               peak_value=0.3172, confirmed_at=82.4, channel="slow")
    p = event_payload(ev, "Bed07", "S1")
    size = payload_size(p)

    record("單筆 payload 大小", size < 300, f"{size} bytes")

    # 寫入
    with tempfile.TemporaryDirectory() as d:
        sink = JSONLSink(Path(d) / "ev.jsonl")
        sink.emit(p); sink.close()
        lines = (Path(d) / "ev.jsonl").read_text(encoding="utf-8").strip().split("\n")
        record("JSONL 寫入", len(lines) == 1, f"{len(lines)} 行")

    # 保護機制：夾帶座標陣列要被擋下
    blocked = False
    try:
        bad = dict(p); bad["keypoints"] = [[0.1, 0.2, 0.9]] * 17
        with tempfile.TemporaryDirectory() as d:
            s = JSONLSink(Path(d) / "x.jsonl"); s.emit(bad); s.close()
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


def test_render() -> bool:
    section("7. 視覺化")
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
    section("8. 事件紀錄範本")
    from src.report import session_report
    from src.temporal.state_machine import Event

    evs = [
        Event("arm_displacement", 77.4, 98.4, .317, 82.4, "slow"),
        Event("arm_displacement", 9200.0, 9218.0, .289, 9205.0, "slow"),
    ]
    text = session_report(evs, 14400.0, "Bed 07")
    ok = "2" in text and "上肢姿態異常" in text
    record("產生床次紀錄", ok, f"{len(text)} 字元")
    if ok:
        print("\n" + "\n".join("    " + l for l in text.split("\n")[:8]) + "\n    ...")
    return ok


# --------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("  DialySight 自我測試 — 不需要影片，不需要硬體")
    print("=" * 62)

    t0 = time.perf_counter()
    checks = [test_imports, test_window, test_state_machine, test_subject,
              test_color, test_sink, test_render, test_report]
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