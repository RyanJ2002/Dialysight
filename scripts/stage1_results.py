#!/usr/bin/env python3
"""
Stage I 實驗數據一鍵產生器。

用法（在 repo 根目錄）：
    python scripts/stage1_results.py --video <跟 comparison.mp4 同一支來源 mp4>

跑完會得到 results/stage1/report.md 與 report.json，把這兩個檔案傳回來就好。

----------------------------------------------------------------------
為什麼要有這支
----------------------------------------------------------------------
對照組 config/baseline.yaml 刻意不做平滑（模擬「單幀 + 單閾值」的天真作法），
而沒有平滑的序列，上升緣計數對輸入的任何微擾都極度敏感。

同一支影片、同樣的影格數、同一份程式，只要換一台機器跑姿態推論
（PyTorch / ultralytics / OpenCV 解碼器版本不同，浮點結果就會在小數第三位
有差異），對照組的告警次數就會變動。實測兩台機器跑同一支 Session A：
30 筆 vs 29 筆，其中只有 19 筆時間點對得上。

實驗組不受影響（中位數濾波 window 7 把差異吃掉了），兩邊都是 3 次。

結論：**告警數字必須跟 comparison.mp4 出自同一台機器。**
這支就是用來在產生影片的那台機器上，一次把所有數字補齊。

----------------------------------------------------------------------
它會做什麼
----------------------------------------------------------------------
1. 記錄環境指紋（作業系統、CPU、記憶體、Python / torch / ultralytics / OpenCV 版本）
2. 跑對照組：config/baseline.yaml + 單閾值狀態機
3. 跑實驗組：config/default.yaml + 完整時序層
4. 每筆事件都輸出「起點 / 確認 / 結束」三個時間，不再混用
5. 用 4 秒間隔把對照組告警分群，重現叢集表
6. 量測實際處理吞吐量（FPS）
7. 全部寫進 report.md 與 report.json

兩份 config 都不要改。腳本不會寫入任何 config，也不會動到既有的 results。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")       # Windows 主控台的中文
except Exception:
    pass

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline                  # noqa: E402

OUT_DIR = ROOT / "results" / "stage1"
CLUSTER_GAP_S = 4.0


# ----------------------------------------------------------------------
# 環境指紋
# ----------------------------------------------------------------------

def _ver(mod: str) -> str:
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "?"))
    except Exception:
        return "not installed"


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:                                            # Linux / macOS 後備
        import os
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        return None


def environment() -> dict:
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
        "torch": _ver("torch"),
        "ultralytics": _ver("ultralytics"),
        "opencv": _ver("cv2"),
        "numpy": _ver("numpy"),
    }


def file_digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def probe_video(video: str) -> dict:
    try:
        import cv2
    except Exception:
        return {"fps": 30.0, "frames": None, "width": None, "height": None,
                "note": "OpenCV 不可用，fps 以 30.0 估計"}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"開不了影片：{video}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    if not (1.0 <= info["fps"] <= 240.0):
        info["note"] = f"影片回報的 fps 不合理（{info['fps']}），改用 30.0"
        info["fps"] = 30.0
    return info


# ----------------------------------------------------------------------
# 跑一組
# ----------------------------------------------------------------------

def collect_events(pipe) -> list[dict]:
    """直接讀 Event 物件 —— summary() 沒有 confirmed_at，但 A3 需要它。"""
    evs = list(pipe.machine.confirmed_events())
    if getattr(pipe, "fast", None) is not None:
        evs += list(pipe.fast.confirmed_events())
    out = []
    for e in sorted(evs, key=lambda x: x.start_time):
        out.append({
            "channel": e.channel,
            "feature": e.feature,
            "start_s": round(e.start_time, 2),
            "confirmed_s": (None if e.confirmed_at is None
                            else round(e.confirmed_at, 2)),
            "end_s": None if e.end_time is None else round(e.end_time, 2),
            "duration_s": None if e.duration is None else round(e.duration, 2),
            "peak": round(e.peak_value, 4),
        })
    return out


def run_arm(label: str, cfg_path: Path, video: str, baseline_machine: bool,
            fps: float) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    print(f"\n[{label}] 開始跑 {cfg_path.name} … 這一步會跑姿態推論，請耐心等",
          flush=True)
    pipe = Pipeline(cfg, use_baseline_machine=baseline_machine,
                    enable_sink=False, fps_hint=fps)
    t0 = time.perf_counter()
    frames, last_ts = 0, 0.0
    try:
        for r in pipe.run(video):
            frames += 1
            last_ts = r.obs.timestamp
            if frames % 900 == 0:
                print(f"    {frames} 格…", flush=True)
        summary = pipe.summary()
        events = collect_events(pipe)
    finally:
        pipe.close()
    elapsed = time.perf_counter() - t0

    out = {
        "label": label,
        "config": cfg_path.name,
        "config_sha256_12": file_digest(cfg_path),
        "state_machine": "single_threshold" if baseline_machine else "hysteresis+fast",
        "frames": frames,
        "clip_duration_s": round(last_ts, 2),
        "wall_clock_s": round(elapsed, 1),
        "throughput_fps": round(frames / elapsed, 1) if elapsed > 0 else None,
        "alerts": summary["alerts"],
        "alerts_slow": summary["alerts_slow"],
        "alerts_fast": summary["alerts_fast"],
        "dismissed_in_watching": summary["dismissed_in_watching"],
        "events": events,
        "filter": summary.get("filter", {}),
        "baseline": summary.get("baseline", {}),
        "fast_channel": summary.get("fast_channel"),
        "fusion": summary.get("fusion"),
    }
    print(f"[{label}] 完成：{out['alerts']} 次告警"
          f"（慢 {out['alerts_slow']} / 快 {out['alerts_fast']}）"
          f"，{frames} 格，吞吐 {out['throughput_fps']} FPS", flush=True)
    return out


def clusters(events: list[dict], gap: float = CLUSTER_GAP_S) -> list[dict]:
    out: list[dict] = []
    for e in sorted(events, key=lambda x: x["start_s"]):
        if out and e["start_s"] - out[-1]["end_s"] <= gap:
            out[-1]["end_s"] = e["start_s"]
            out[-1]["count"] += 1
            out[-1]["peak"] = max(out[-1]["peak"], e["peak"])
        else:
            out.append({"start_s": e["start_s"], "end_s": e["start_s"],
                        "count": 1, "peak": e["peak"]})
    for c in out:
        c["span_s"] = round(c["end_s"] - c["start_s"], 2)
    return out


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage I 實驗數據一鍵產生器（請在產生 comparison.mp4 的那台機器上跑）")
    ap.add_argument("--video", required=True,
                    help="跟 comparison.mp4 同一支來源 mp4")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--cluster-gap", type=float, default=CLUSTER_GAP_S)
    args = ap.parse_args()

    video = str(Path(args.video).expanduser())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = ROOT / "config" / "baseline.yaml"
    dflt_cfg = ROOT / "config" / "default.yaml"
    for p in (base_cfg, dflt_cfg):
        if not p.is_file():
            raise SystemExit(f"找不到 {p}")

    env = environment()
    vid = probe_video(video)

    print("=" * 64)
    print("Stage I 實驗數據")
    print("=" * 64)
    print(f"影片   {video}")
    print(f"       {vid['width']}x{vid['height']}, {vid['fps']:g} fps, "
          f"容器標示 {vid['frames']} 格")
    print(f"機器   {env['os']}")
    print(f"       Python {env['python']} / torch {env['torch']} / "
          f"ultralytics {env['ultralytics']} / OpenCV {env['opencv']}")
    print(f"設定   baseline.yaml {file_digest(base_cfg)} · "
          f"default.yaml {file_digest(dflt_cfg)}")

    control = run_arm("對照組 單閾值", base_cfg, video, True, vid["fps"])
    ours = run_arm("實驗組 DialySight", dflt_cfg, video, False, vid["fps"])

    cl = clusters(control["events"], args.cluster_gap)
    a, o = control["alerts"], ours["alerts"]
    reduction = round((a - o) / a * 100, 1) if a else None

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video": video,
        "video_info": vid,
        "environment": env,
        "cluster_gap_s": args.cluster_gap,
        "control": control,
        "dialysight": ours,
        "clusters": cl,
        "reduction_pct": reduction,
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- markdown ----------------
    L: list[str] = []
    A = L.append
    A("# Stage I 實驗數據")
    A("")
    A(f"> 產生時間：{payload['generated_at']}")
    A(f"> 影片：`{video}`（{vid['width']}×{vid['height']}, {vid['fps']:g} fps）")
    A(f"> 設定檔未修改：`baseline.yaml` {control['config_sha256_12']} · "
      f"`default.yaml` {ours['config_sha256_12']}")
    A("")
    A("## 環境指紋")
    A("")
    A("| 項目 | 值 |")
    A("|---|---|")
    A(f"| 作業系統 | {env['os']} |")
    A(f"| 處理器 | {env['processor']} |")
    A(f"| 記憶體 | {env['ram_gb'] if env['ram_gb'] else '（未取得，請手動填）'} GB |")
    A(f"| Python | {env['python']} |")
    A(f"| torch | {env['torch']} |")
    A(f"| ultralytics | {env['ultralytics']} |")
    A(f"| OpenCV | {env['opencv']} |")
    A(f"| numpy | {env['numpy']} |")
    A("")
    A("對照組不做平滑，告警次數會隨環境浮動，所以這張表要跟數字一起留存。")
    A("")
    A("## 主要數據")
    A("")
    A("| | 對照組（單閾值） | 實驗組（DialySight） |")
    A("|---|---|---|")
    A(f"| 總告警 | **{a}** | **{o}** |")
    A(f"| 慢 / 快 | — | {ours['alerts_slow']} / {ours['alerts_fast']} |")
    A(f"| 降幅 | — | **{reduction}%** |")
    A(f"| WATCHING 中取消 | — | {ours['dismissed_in_watching']} |")
    A(f"| 影格數 | {control['frames']} | {ours['frames']} |")
    A(f"| 片長 | {control['clip_duration_s']} s | {ours['clip_duration_s']} s |")
    A(f"| 吞吐量 | {control['throughput_fps']} FPS | {ours['throughput_fps']} FPS |")
    A("")
    A(f"**對照組的 {a} 這個數字，應該等於 `comparison.mp4` 結尾螢幕上的 `ALERTS` 數。**")
    A("若不相等，代表影片不是在這台機器／這個環境產生的。")
    A("")

    fc = ours.get("fast_channel") or {}
    fu = ours.get("fusion") or {}
    fl = ours.get("filter") or {}
    A("## 實驗組細項")
    A("")
    A("| 項目 | 值 |")
    A("|---|---|")
    if fc:
        A(f"| 快通道 觸發 / 確認 / 擋下 | {fc.get('triggers')} / "
          f"{fc.get('confirmed')} / {fc.get('glitch_rejected')} |")
        A(f"| 快通道 擋下率 | {round((fc.get('reject_rate') or 0)*100, 1)}% |")
        A(f"| 快通道 平均延遲 | {fc.get('mean_latency_s')} s |")
    if fl:
        A(f"| 遮蔽影格比例 | {round((fl.get('occlusion_rate') or 0)*100, 2)}% |")
        A(f"| 處理影格數 | {fl.get('frames')} |")
    if fu:
        A(f"| 不可信影格比例 | {round((fu.get('unreliable_rate') or 0)*100, 2)}% |")
        A(f"| 降級影格比例 | {round((fu.get('degraded_rate') or 0)*100, 2)}% |")
        A(f"| 融合層評估影格 | {fu.get('frames')} |")
    bl = (ours.get("baseline") or {}).get("values") or {}
    if "arm_displacement" in bl:
        A(f"| 學到的個人基準線 | arm_displacement {bl['arm_displacement']} |")
    A("")

    A(f"## 對照組告警叢集（相鄰間隔 < {args.cluster_gap:g} 秒算同一串）")
    A("")
    A("| # | 區間 (s) | 長度 (s) | 次數 | 峰值 |")
    A("|---|---|---|---|---|")
    for i, c in enumerate(cl, 1):
        A(f"| {i} | {c['start_s']:.2f}–{c['end_s']:.2f} | {c['span_s']:.2f} "
          f"| {c['count']} | {c['peak']:.4f} |")
    A("")

    A("## 實驗組事件（起點 / 確認 / 結束 分開列）")
    A("")
    A("| 通道 | 起點 (s) | 確認 (s) | 結束 (s) | 長度 (s) | 峰值 |")
    A("|---|---|---|---|---|---|")
    for e in ours["events"]:
        f = lambda v: "—" if v is None else f"{v:.2f}"
        A(f"| {e['channel']} | {e['start_s']:.2f} | {f(e['confirmed_s'])} "
          f"| {f(e['end_s'])} | {f(e['duration_s'])} | {e['peak']:.4f} |")
    A("")
    A("「確認」才是護理師看到告警的時間，「起點」是事件開始的時間。")
    A("儀表板事件列上的 `T+` 標的是**起點**。")
    A("")

    A("## 對照組告警逐筆時間")
    A("")
    A("拿來跟 `comparison.mp4` 左側計數器逐筆核對。")
    A("")
    A("| # | 起點 (s) | 峰值 | | # | 起點 (s) | 峰值 |")
    A("|---|---|---|---|---|---|---|")
    ev = control["events"]
    half = (len(ev) + 1) // 2
    for i in range(half):
        l = ev[i]
        left = f"| {i+1} | {l['start_s']:.2f} | {l['peak']:.4f} |"
        j = i + half
        if j < len(ev):
            r = ev[j]
            A(f"{left} | {j+1} | {r['start_s']:.2f} | {r['peak']:.4f} |")
        else:
            A(f"{left} | | | |")
    A("")

    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print()
    print("=" * 64)
    print(f"對照組 {a} 次 → 實驗組 {o} 次，降幅 {reduction}%")
    print(f"對照組叢集：{' / '.join(str(c['count']) for c in cl)}")
    print("=" * 64)
    print(f"\n報告已寫出：")
    print(f"  {out_dir / 'report.md'}")
    print(f"  {out_dir / 'report.json'}")
    print("\n把這兩個檔案傳回群組就好。")


if __name__ == "__main__":
    main()
