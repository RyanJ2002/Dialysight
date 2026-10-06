#!/usr/bin/env python3
"""
試跑新功能 —— 快通道、融合層、事件輸出。

★ selftest.py 回答「有沒有壞」，這支腳本回答「它到底在做什麼」。

兩種模式：

    # 1. 不需要攝影機、不需要影片。四個情境各跑一次，印出每個功能做了什麼
    python scripts/try_new.py --synthetic

    # 2. 用筆電鏡頭即時跑，新功能一觸發就在終端印一行
    python scripts/try_new.py --video 0 --config config/webcam.yaml

    # 3. 用已有的影片跑
    python scripts/try_new.py --video data/clips/clip_01.mp4

模式 2 要看什麼：

    基準建立完（5 秒）後──
      把右手快速往旁邊移，停住      → FAST 事件（約 0.8 秒內）
      把右手快速移出去又馬上收回     → 被當成偵測跳點擋下，不告警
      把右手慢慢往外移並維持         → SLOW 事件（約 6 秒後）
      用手或衣物遮住右手腕與肩膀     → UNRELIABLE「無法判定」（不是「正常」）
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion import Reliability
from src.inference.base import NUM_KEYPOINTS, KP
from src.pipeline import Pipeline

FPS = 10.0


# ==========================================================================
# 模式 1：合成情境
# ==========================================================================

def _frame(t: float, wrist_dx: float, hide_arm: bool = False,
           hide_all: bool = False) -> dict:
    """造一格關鍵點。wrist_dx 是右腕相對基準位置往外移多少（正規化單位）。"""
    k = [[0.0, 0.0, 0.0] for _ in range(NUM_KEYPOINTS)]
    if hide_all:
        return {"frame_idx": int(t * FPS), "timestamp": t, "keypoints": k}

    k[int(KP.LEFT_SHOULDER)] = [0.30, 0.47, 0.9]
    k[int(KP.RIGHT_SHOULDER)] = [0.30, 0.53, 0.9]
    k[int(KP.LEFT_HIP)] = [0.60, 0.48, 0.9]
    k[int(KP.RIGHT_HIP)] = [0.60, 0.54, 0.9]

    if not hide_arm:
        # 被子蓋住手臂 → 肘與腕的信心掉下去，filter 會標記為缺失
        k[int(KP.RIGHT_ELBOW)] = [0.42, 0.56, 0.9]
        k[int(KP.RIGHT_WRIST)] = [0.42 + wrist_dx, 0.58, 0.9]

    return {"frame_idx": int(t * FPS), "timestamp": t, "keypoints": k}


SCENARIOS = {
    "gradual": (
        "漸進漂移",
        "手臂在 30 秒內慢慢往外移 0.22 並維持",
        "慢通道應該在約 6 秒後確認。快通道不該叫 —— 這不是突發。",
        lambda t: _frame(t, 0.0 if t < 30 else min(0.22, 0.008 * (t - 30))),
    ),
    "sudden": (
        "突發大動作",
        "手臂在第 30 秒瞬間移出 0.20，然後停在那裡",
        "快通道應該在 0.8 秒內確認。這是慢通道等 5 秒會錯過的事件。",
        lambda t: _frame(t, 0.20 if t >= 30 else 0.0),
    ),
    "glitch_1f": (
        "單格跳點",
        "手臂座標只有一格暴衝 0.30，下一格就彈回（最常見的模型認錯）",
        "這個不會走到快通道 —— 層 2 的中位數濾波就把它吃掉了。",
        lambda t: _frame(t, 0.30 if abs(t - 30.0) < 0.05 else 0.0),
    ),
    "glitch_5f": (
        "多格跳點 ★",
        "手臂座標連續 5 格（0.5 秒）暴衝 0.30，然後彈回原位",
        "★ 這個穿得過中位數濾波，振幅跟真實突發一樣大 ——\n"
        "            　　　　位移沒站住 → 快通道的持續性檢查應該擋下它。",
        lambda t: _frame(t, 0.30 if 30.0 <= t < 30.5 else 0.0),
    ),
    "arm_covered": (
        "被子只蓋住手臂",
        "第 30 秒後右肘與右腕的信心掉到門檻以下，軀幹四點還看得見",
        "主訊號消失但軀幹照常投票 → DEGRADED（降級但仍可用）。",
        lambda t: _frame(t, 0.0, hide_arm=t >= 30),
    ),
    "all_covered": (
        "整個人都看不到 ★",
        "第 30 秒後所有關鍵點的信心都掉到門檻以下",
        "★ 融合層應該輸出 UNRELIABLE（無法判定），不是 NORMAL（正常）。",
        lambda t: _frame(t, 0.0, hide_all=t >= 30),
    ),
}


def run_scenario(key: str, workdir: Path) -> dict:
    label, setup, expect, shape = SCENARIOS[key]

    frames = [shape(i / FPS) for i in range(700)]      # 70 秒
    kp = workdir / f"{key}.json"
    kp.write_text(json.dumps({"fps": FPS, "frames": frames}), encoding="utf-8")

    cfg = {
        "inference": {"backend": "replay", "keypoints_json": str(kp)},
        "filter": {"min_confidence": 0.5, "min_visibility_ratio": 0.0},
        "features": {"wrist_side": "right",
                     "enabled": ["arm_displacement", "torso_angle"]},
        "baseline": {"duration_s": 20.0, "min_valid_frames": 50},
        "smooth": {"default": {"method": "median", "window": 7}},
        "windows": {"short": 5.0, "medium": 60.0, "long": 900.0},
        "state_machine": {"enter_threshold": 0.08, "exit_threshold": 0.05,
                          "t_enter": 1.0, "t_confirm": 5.0,
                          "t_exit": 3.0, "cooldown": 30.0},
        "fast_channel": {"enabled": True, "amplitude_threshold": 0.10,
                         "t_persist": 0.8, "cooldown": 15.0},
        "fusion": {"enabled": True, "drive_state_machine": False},
        "color": {"enabled": False},
        "sink": {"bed_id": key, "jsonl": str(workdir / "events_{bed}.jsonl"),
                 "status_interval_s": 10.0},
    }

    pipe = Pipeline(cfg, fps_hint=FPS)
    first_slow = first_fast = None
    rel_counts: dict[str, int] = {}

    # try/finally：中途出錯時 sink 也要關掉，否則 Windows 清理暫存資料夾
    # 會噴 WinError 32，把真正的錯誤訊息蓋掉。
    try:
        for r in pipe.run(None):
            if r.new_event is not None and first_slow is None:
                first_slow = r.obs.timestamp - r.new_event.start_time
            if r.fast_event is not None and first_fast is None:
                first_fast = r.obs.timestamp - r.fast_event.start_time
            if r.fusion is not None:
                v = r.fusion.reliability.value
                rel_counts[v] = rel_counts.get(v, 0) + 1
        summary = pipe.summary()
    finally:
        pipe.close()

    out = workdir / f"events_{key}.jsonl"
    records = [json.loads(l) for l in
               out.read_text(encoding="utf-8").strip().split("\n") if l] \
        if out.exists() else []

    return {
        "label": label, "setup": setup, "expect": expect,
        "slow": summary["alerts_slow"], "fast": summary["alerts_fast"],
        "slow_latency": first_slow, "fast_latency": first_fast,
        "glitch_rejected": summary.get("fast_channel", {}).get("glitch_rejected", 0),
        "reliability": rel_counts,
        "unreliable_rate": summary.get("fusion", {}).get("unreliable_rate", 0.0),
        "records": records,
    }


def synthetic_mode() -> None:
    print("=" * 70)
    print("  新功能試跑 —— 六個情境，不需要攝影機也不需要影片")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        results = {k: run_scenario(k, d) for k in SCENARIOS}

        for key, r in results.items():
            print(f"\n■ {r['label']}")
            print(f"  餵進去　：{r['setup']}")
            print(f"  預期　　：{r['expect']}")
            print(f"  實際結果：", end="")

            bits = []
            if r["slow"]:
                lat = f"（延遲 {r['slow_latency']:.1f} 秒）" if r["slow_latency"] else ""
                bits.append(f"慢通道 {r['slow']} 次{lat}")
            if r["fast"]:
                lat = f"（延遲 {r['fast_latency']:.1f} 秒）" if r["fast_latency"] else ""
                bits.append(f"快通道 {r['fast']} 次{lat}")
            if r["glitch_rejected"]:
                bits.append(f"擋下跳點 {r['glitch_rejected']} 次")
            if r["unreliable_rate"] > 0.01:
                bits.append(f"無法判定 {r['unreliable_rate'] * 100:.0f}% 的時間")
            print("、".join(bits) if bits else "沒有告警")

            if key in ("arm_covered", "all_covered"):
                rel = r["reliability"]
                total = sum(rel.values()) or 1
                dist = "　".join(f"{k} {v / total * 100:.0f}%"
                                 for k, v in rel.items())
                print(f"  可信度分佈：{dist}")

        # 對照一：真實突發 vs 多格跳點
        print("\n" + "─" * 70)
        print("  ★ 對照一：這兩個情境是快通道存在的理由")
        print("─" * 70)
        s, g = results["sudden"], results["glitch_5f"]
        print(f"  真實突發（跳上去，停住）  → 快通道 {s['fast']} 次告警")
        print(f"  多格跳點（跳上去，彈回）  → 快通道 {g['fast']} 次告警，"
              f"擋下 {g['glitch_rejected']} 次")
        print("\n  兩者振幅一樣大，單看振幅無法區分。")
        print("  差別在之後那 0.8 秒：真動作停在新位置，跳點彈回原位。")

        print("\n  ⚠️ 單格跳點不需要快通道處理 —— 中位數濾波（層 2）就吃掉了。")
        print(f"     上面「單格跳點」情境的結果是「"
              f"{'沒有告警' if not results['glitch_1f']['fast'] else '有告警'}"
              f"」，而且 glitch_rejected = "
              f"{results['glitch_1f']['glitch_rejected']}，")
        print("     代表它根本沒走到層 4。兩層各擋不同長度的雜訊，不要混談。")

        # 對照二：遮蔽的兩種程度
        print("\n" + "─" * 70)
        print("  ★ 對照二：遮蔽的程度決定系統說什麼")
        print("─" * 70)
        a, b = results["arm_covered"], results["all_covered"]
        print(f"  只蓋住手臂    → 無法判定 {a['unreliable_rate'] * 100:4.0f}% 的時間"
              f"　（軀幹還看得見，降級但仍在判斷）")
        print(f"  整個人看不到  → 無法判定 {b['unreliable_rate'] * 100:4.0f}% 的時間"
              f"　（★ 顯示「無法判定」而不是綠燈）")
        print("\n  ⚠️ 這裡有一個要跟臨床端決定的設計問題，見下方。")

        # 實際送出去的 payload
        print("\n" + "─" * 70)
        print("  ★ 網路上實際跑的東西（漸進漂移情境的事件檔）")
        print("─" * 70)
        for rec in results["gradual"]["records"]:
            if rec.get("type") == "status":
                continue
            line = json.dumps(rec, ensure_ascii=False)
            print(f"  {len(line.encode()):>4} bytes │ {line}")
        print("\n  全是純量，沒有座標、沒有影像。想夾帶的話 "
              "sink.validate_payload() 會直接拋例外。")

        # 待決事項
        print("\n" + "─" * 70)
        print("  ⚠️ 一個要跟臨床端決定的事（程式已留開關，預設關閉）")
        print("─" * 70)
        print("""
  「只蓋住手臂」目前判為 DEGRADED，不是 UNRELIABLE。
  理由是軀幹四點還看得見，系統不是全盲。

  但可以反過來論證：針就在手臂上。看不到手臂時，
  宣稱自己還能評估針脫落風險，是在高估我們知道的東西。

  兩邊都站得住，取決於臨床上怎麼看：

    維持 DEGRADED   真實透析時手臂很可能大部分時間被蓋著。
                    若改成 UNRELIABLE，系統會整場都在說「不知道」
                    —— 那樣的告警等於沒有告警。

    改成 UNRELIABLE 符合「寧可讓人知道我們看不到」的原則，
                    但要先量出手臂實際被遮的時間比例才知道可不可行。

  ★ 10/9 實驗時要量的數字：靜臥時 arm_displacement_visibility 的分布。
    那個數字決定這題怎麼選，現在猜沒有意義。

  要改成 UNRELIABLE 的話，在 config 的 fusion.signals 裡
  給 arm_displacement 加一行 `required: true` 即可，程式已支援。
""")

    print("=" * 70)
    print("  想實機操作：python scripts/try_new.py --video 0 "
          "--config config/webcam.yaml")
    print("=" * 70)


# ==========================================================================
# 模式 2：即時
# ==========================================================================

def live_mode(video: str, config: str, show: bool) -> None:
    import yaml

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    source = int(video) if str(video).isdigit() else video

    from scripts.run_demo import probe_fps          # 共用 FPS 偵測
    fps = probe_fps(video)

    pipe = Pipeline(cfg, fps_hint=fps)

    renderer = None
    if show:
        from src.visualize import DemoRenderer, LayoutConfig
        layout = LayoutConfig(**{k: v for k, v in cfg.get("visualize", {}).items()
                                 if k in LayoutConfig.__dataclass_fields__})
        renderer = DemoRenderer(
            layout=layout,
            thresholds={"enter": cfg["state_machine"]["enter_threshold"],
                        "exit": cfg["state_machine"]["exit_threshold"]},
            primary_feature=cfg["features"]["enabled"][0],
        )

    print("=" * 70)
    print("  即時試跑。新功能一觸發就在這裡印一行。")
    print()
    print("  基準建立完之後試這四個動作：")
    print("    1. 右手快速往旁邊移，停住        → FAST 事件")
    print("    2. 右手快速移出去，馬上收回      → 被當跳點擋下")
    print("    3. 右手慢慢往外移並維持          → SLOW 事件")
    print("    4. 遮住右手腕與右肩              → UNRELIABLE")
    print()
    print("  按 q 離開（有開 --show 時）或 Ctrl-C")
    print("=" * 70)

    prev_rel = None
    prev_glitch = 0
    prev_baseline = False
    import cv2

    try:
        for r in pipe.run(source):
            ts = r.obs.timestamp

            if r.baseline_ready and not prev_baseline:
                done, need = r.baseline_progress
                print(f"[{ts:7.1f}s] ✓ 基準線建立完成（{need:.0f} 秒）"
                      f"—— 現在開始判定")
                prev_baseline = True

            if r.fast_event is not None:
                e = r.fast_event
                print(f"[{ts:7.1f}s] ⚡ FAST 事件　延遲 {e.meta['latency_s']}s"
                      f"　振幅 {e.meta['amplitude']}　峰值 {e.peak_value:.4f}")

            if r.new_event is not None:
                e = r.new_event
                print(f"[{ts:7.1f}s] ▲ SLOW 事件　起於 {e.start_time:.1f}s"
                      f"　峰值 {e.peak_value:.4f}")

            if pipe.fast is not None:
                g = pipe.fast.report()["glitch_rejected"]
                if g > prev_glitch:
                    print(f"[{ts:7.1f}s] ✗ 偵測跳點擋下（位移沒站住）"
                          f"　累計 {g} 次")
                    prev_glitch = g

            if r.fusion is not None and r.fusion.reliability is not prev_rel:
                rel = r.fusion.reliability
                mark = {"ok": "·", "degraded": "~", "unreliable": "?"}[rel.value]
                note = f"　{r.fusion.reason}" if r.fusion.reason else ""
                print(f"[{ts:7.1f}s] {mark} 可信度 → {rel.value.upper()}"
                      f"　conf={r.fusion.confidence:.2f}{note}")
                if rel is Reliability.UNRELIABLE:
                    print("            ★ 這格是「不知道」，不是「正常」。"
                          "護理站顯示紫色「無法判定」。")
                prev_rel = rel

            if renderer is not None:
                canvas = renderer.render(
                    r.obs, r.display_features, r.state, r.any_event, ts,
                    color_cfg=pipe.color_cfg if pipe.color_enabled else None,
                    baseline_progress=r.baseline_progress,
                )
                cv2.imshow("DialySight — try_new", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n(中斷)")
    finally:
        if renderer is not None:
            cv2.destroyAllWindows()

    summary = pipe.summary()
    pipe.close()

    print("\n" + "=" * 70)
    print(f"  慢通道 {summary['alerts_slow']} 次　"
          f"快通道 {summary['alerts_fast']} 次　"
          f"WATCHING 取消 {summary['dismissed_in_watching']} 次")
    if "fast_channel" in summary:
        fc = summary["fast_channel"]
        print(f"  快通道觸發 {fc['triggers']} 次，其中 "
              f"{fc['glitch_rejected']} 次被判定為偵測跳點")
    if "fusion" in summary:
        print(f"  無法判定的時間佔 "
              f"{summary['fusion']['unreliable_rate'] * 100:.1f}%")
    if "sink" in summary and "path" in summary["sink"]:
        print(f"  事件檔 → {summary['sink']['path']}")
    print("=" * 70)


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="試跑快通道、融合層、事件輸出")
    ap.add_argument("--synthetic", action="store_true",
                    help="不需攝影機，跑四個合成情境")
    ap.add_argument("--video", help="影片路徑，或 0 代表預設攝影機")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--show", action="store_true",
                    help="同時開啟畫面視窗")
    args = ap.parse_args()

    if args.synthetic or not args.video:
        synthetic_mode()
    else:
        live_mode(args.video, args.config, args.show)


if __name__ == "__main__":
    main()
