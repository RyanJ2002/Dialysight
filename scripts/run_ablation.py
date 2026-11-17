#!/usr/bin/env python3
"""
實驗一（偽陽性對照）+ 消融：把「28 → 3」的降幅拆成前處理與時序邏輯兩段。

為什麼需要這支
--------------
config/baseline.yaml 的對照組跟實驗組有兩個不同的地方，但只有一個是刻意的：

    1. smooth.method = none     ← 檔案裡有註解「對照組不做平滑」，刻意的
    2. 沒有 subject: 區塊        ← 沒有任何註解，看起來是漏的

第 2 點的後果比看起來大。SubjectSelector 是在 **backend 層** 選人
（src/inference/ultralytics_backend.py:42, 79），不是在 pipeline，
所以少了 subject: 代表對照組是「全畫面找人」，實驗組限制在床位 ROI。
兩邊連「選到誰、哪幾格算有效」都不一樣。

這代表目前簡報上的降幅裡，有一部分只是中位數濾波與 ROI 的功勞，
不是時序邏輯的功勞。Stage II 評審看得到 config，這題會被問。

這支跑四組，把降幅拆開：

    A    單閾值 · 無平滑 · 無 ROI    baseline.yaml 原樣      ← 目前簡報的對照組
    A'   單閾值 · 無平滑 · 有 ROI    baseline.yaml 的時序設定 ← 隔離 ROI
    B    單閾值 · 有平滑 · 有 ROI    default.yaml  + 單閾值   ← 隔離平滑
    C    DialySight 完整            default.yaml            ← 實驗組

    A → C   簡報的標題數字
    A → A'  ROI 的貢獻
    A'→ B   平滑的貢獻
    B → C   時序邏輯的貢獻（遲滯 + 確認時間 + 冷卻 + 快通道）

只跑兩趟姿態推論
----------------
A 用對照組的 subject 設定選人，A'/B/C 用實驗組的。選人結果會被 dump 進
關鍵點 JSON，之後四組都用 replay backend 讀同一份檔，所以：

  * 姿態推論只跑兩趟，不是四趟
  * A'/B/C 吃到的是「逐格完全相同」的輸入 —— 這很重要，因為對照組沒有
    平滑，上升緣計數對輸入的任何微擾都極度敏感（9,015 格 vs 9,019 格
    就足以翻掉兩次告警）。共用同一份 dump 可以把這個變因消掉。

用法
----
    python scripts/run_ablation.py --video data/clips/<demo 影片用的那支>.mp4

    # 關鍵點已經 dump 過，只想重跑時序層（秒級）
    python scripts/run_ablation.py --video <同一支> --reuse

輸出
----
    results/ablation/kp_noroi.json   關鍵點（對照組 subject 設定）
    results/ablation/kp_roi.json     關鍵點（實驗組 subject 設定）
    results/ablation/ablation.csv    給學弟直接用
    results/ablation/ablation.md     人看的表 + 每組事件清單 + 叢集分析
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline            # noqa: E402

OUT_DIR = ROOT / "results" / "ablation"

# 叢集判定：相鄰兩次告警間隔小於這個秒數就算同一串。
# 4.0 會重現比對文件裡的叢集 1–6 分法；用 --cluster-gap 可以改。
CLUSTER_GAP_S = 4.0


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def probe_fps(video: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if not (1.0 <= fps <= 240.0):
        print(f"[ablation] 影片未回報合理 FPS（{fps}），改用 30.0")
        return 30.0
    return float(fps)


def dump_keypoints(video: str, config: Path, out: Path, reuse: bool) -> int:
    """呼叫既有的 scripts/dump_keypoints.py，確保檔案格式跟 ReplayBackend 一致。"""
    if reuse and out.is_file():
        n = len(json.loads(out.read_text(encoding="utf-8"))["frames"])
        print(f"[ablation] 沿用既有 {out.name}（{n} 格）")
        return n

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ablation] 跑姿態推論 → {out.name}（config: {config.name}）…")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dump_keypoints.py"),
         "--video", video, "--out", str(out), "--config", str(config)],
        check=True,
    )
    return len(json.loads(out.read_text(encoding="utf-8"))["frames"])


def as_replay(cfg: dict, keypoints_json: Path) -> dict:
    """把 config 改成讀關鍵點 JSON，不重跑推論。"""
    cfg = copy.deepcopy(cfg)
    cfg["inference"]["backend"] = "replay"
    cfg["inference"]["keypoints_json"] = str(keypoints_json)
    return cfg


def run_arm(cfg: dict, baseline_machine: bool, fps: float) -> dict:
    pipe = Pipeline(cfg, use_baseline_machine=baseline_machine,
                    enable_sink=False, fps_hint=fps)
    try:
        last_ts, frames = 0.0, 0
        for r in pipe.run(None):
            last_ts = r.obs.timestamp
            frames += 1
        summary = pipe.summary()
    finally:
        pipe.close()
    summary["frames"] = frames
    summary["last_ts"] = round(last_ts, 2)
    return summary


def clusters(events: list[dict], gap: float = CLUSTER_GAP_S) -> list[dict]:
    """把連續的告警分串 —— 用來重現比對文件裡的叢集 1–6。"""
    out: list[dict] = []
    for ev in sorted(events, key=lambda e: e["start"]):
        if out and ev["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = ev["start"]
            out[-1]["count"] += 1
            out[-1]["peak"] = max(out[-1]["peak"], ev["peak"])
        else:
            out.append({"start": ev["start"], "end": ev["start"],
                        "count": 1, "peak": ev["peak"]})
    for c in out:
        c["span"] = round(c["end"] - c["start"], 1)
    return out


def pct(from_n: int, to_n: int) -> str:
    if from_n <= 0:
        return "—"
    return f"{(from_n - to_n) / from_n * 100:.1f}%"


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="demo 影片用的那支來源 mp4")
    ap.add_argument("--reuse", action="store_true",
                    help="關鍵點 JSON 已存在就不重跑推論")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--cluster-gap", type=float, default=CLUSTER_GAP_S,
                    help="叢集判定的間隔秒數（預設 4.0，重現比對文件的分法）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(
        (ROOT / "config" / "baseline.yaml").read_text(encoding="utf-8"))
    dflt_cfg = yaml.safe_load(
        (ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))

    fps = probe_fps(args.video)

    # -- 兩趟姿態推論 --------------------------------------------------
    kp_noroi = out_dir / "kp_noroi.json"
    kp_roi = out_dir / "kp_roi.json"
    n_noroi = dump_keypoints(args.video, ROOT / "config" / "baseline.yaml",
                             kp_noroi, args.reuse)
    n_roi = dump_keypoints(args.video, ROOT / "config" / "default.yaml",
                           kp_roi, args.reuse)
    if n_noroi != n_roi:
        print(f"[ablation] ⚠ 兩份 dump 影格數不同（{n_noroi} vs {n_roi}），"
              f"A 組與其他組的輸入長度不一致，比較時要註明")

    # -- 四組 ----------------------------------------------------------
    arms = [
        ("A",  "單閾值 · 無平滑 · 無 ROI", "baseline.yaml",
         as_replay(base_cfg, kp_noroi), True,  "none",     "無"),
        ("A'", "單閾值 · 無平滑 · 有 ROI", "baseline.yaml",
         as_replay(base_cfg, kp_roi),   True,  "none",     "有"),
        ("B",  "單閾值 · 有平滑 · 有 ROI", "default.yaml",
         as_replay(dflt_cfg, kp_roi),   True,  "median/7", "有"),
        ("C",  "DialySight 完整",         "default.yaml",
         as_replay(dflt_cfg, kp_roi),   False, "median/7", "有"),
    ]

    rows = []
    for tag, label, cfg_name, cfg, baseline_machine, smooth, roi in arms:
        print(f"[ablation] 跑 {tag}：{label}…", flush=True)
        s = run_arm(cfg, baseline_machine, fps)
        rows.append({
            "arm": tag, "label": label, "config": cfg_name,
            "keypoints": Path(cfg["inference"]["keypoints_json"]).name,
            "smoothing": smooth, "roi": roi,
            "state_machine": "single_threshold" if baseline_machine
                             else "hysteresis+fast",
            "frames": s["frames"], "duration_s": s["last_ts"],
            "alerts": s["alerts"], "alerts_slow": s["alerts_slow"],
            "alerts_fast": s["alerts_fast"],
            "dismissed_in_watching": s["dismissed_in_watching"],
            "_events": s["events"],
        })
        print(f"           → {s['alerts']} 次告警"
              f"（慢 {s['alerts_slow']} / 快 {s['alerts_fast']}）"
              f"，{s['frames']} 格 / {s['last_ts']}s")

    a_alerts = rows[0]["alerts"]
    for r in rows:
        r["reduction_vs_A_pct"] = pct(a_alerts, r["alerts"])

    # -- CSV -----------------------------------------------------------
    csv_path = out_dir / "ablation.csv"
    fields = ["arm", "label", "config", "keypoints", "smoothing", "roi",
              "state_machine", "frames", "duration_s", "alerts",
              "alerts_slow", "alerts_fast", "dismissed_in_watching",
              "reduction_vs_A_pct"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # -- Markdown ------------------------------------------------------
    by = {r["arm"]: r for r in rows}
    md = [
        "# 實驗一 + 消融結果",
        "",
        f"- 來源影片：`{args.video}`",
        f"- FPS：{fps:g}",
        f"- 關鍵點：`{kp_noroi.name}`（{n_noroi} 格）、"
        f"`{kp_roi.name}`（{n_roi} 格）",
        "- A'/B/C 吃同一份關鍵點，逐格相同；A 因為 subject 設定不同，"
        "選人結果不同，所以另外 dump 一份",
        "",
        "## 四組結果",
        "",
        "| 組別 | 設定 | 平滑 | 床位 ROI | 狀態機 | 告警數 | 慢 | 快 | vs A 降幅 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['arm']} | {r['label']} | {r['smoothing']} | {r['roi']} | "
            f"{r['state_machine']} | **{r['alerts']}** | {r['alerts_slow']} | "
            f"{r['alerts_fast']} | {r['reduction_vs_A_pct']} |")

    md += ["", "## 降幅拆解", "",
           "| 區段 | 差的是什麼 | 告警數變化 | 貢獻 |", "|---|---|---|---|"]
    for lo, hi, what in (("A", "A'", "床位 ROI"),
                         ("A'", "B", "中位數濾波 window 7"),
                         ("B", "C", "遲滯 + 確認時間 + 冷卻 + 快通道")):
        md.append(f"| {lo} → {hi} | {what} | "
                  f"{by[lo]['alerts']} → {by[hi]['alerts']} | "
                  f"{pct(by[lo]['alerts'], by[hi]['alerts'])} |")
    md.append(f"| **A → C** | **全部** | "
              f"**{by['A']['alerts']} → {by['C']['alerts']}** | "
              f"**{pct(by['A']['alerts'], by['C']['alerts'])}** |")

    md += ["", "## A 組告警叢集（對應比對文件的叢集 1–6）", "",
           f"相鄰間隔 < {args.cluster_gap:g} 秒算同一串。", "",
           "| # | 區間 (s) | 長度 (s) | 次數 | 峰值 |", "|---|---|---|---|---|"]
    for i, c in enumerate(clusters(by["A"]["_events"], args.cluster_gap), 1):
        md.append(f"| {i} | {c['start']:.1f}–{c['end']:.1f} | {c['span']:.1f} "
                  f"| {c['count']} | {c['peak']:.4f} |")

    md += ["", "## C 組（DialySight）事件清單", "",
           "| 通道 | 起點 (s) | 結束 (s) | 長度 (s) | 峰值 |",
           "|---|---|---|---|---|"]
    for e in by["C"]["_events"]:
        end = "—" if e["end"] is None else f"{e['end']:.1f}"
        dur = "—" if e["duration"] is None else f"{e['duration']:.1f}"
        md.append(f"| {e['channel']} | {e['start']:.1f} | {end} | {dur} "
                  f"| {e['peak']:.4f} |")

    md_path = out_dir / "ablation.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print("\n".join(md[6:]))
    print()
    print(f"✓ {csv_path}")
    print(f"✓ {md_path}")
    print(f"✓ 關鍵點保留在 {kp_noroi} / {kp_roi}（重現用，不要進 git）")


if __name__ == "__main__":
    main()
