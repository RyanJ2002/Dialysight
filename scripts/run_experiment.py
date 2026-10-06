#!/usr/bin/env python3
"""
偽陽性對照實驗 —— 整份提案最重要的一張圖。

同一段影片跑兩個版本，比較告警次數：
    對照組：單一閾值，無遲滯、無確認、無冷卻（模擬官方 demo）
    實驗組：遲滯狀態機 + 確認時間 + 冷卻期

用法：
    python scripts/run_experiment.py \
        --videos data/clips/clip_01.mp4 data/clips/clip_02.mp4 \
        --out results/false_positive_comparison

輸出：
    <out>.json   原始數據
    <out>.md     可直接貼進簡報的表格
    <out>.png    長條圖

★ 如果數據不漂亮，誠實呈現並分析原因。
  Stage II 明文說評審會抽驗程式碼邏輯，造假風險遠高於數據平淡。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Pipeline


def probe_fps(video: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps) if 1.0 <= fps <= 240.0 else 30.0


def run_one(video: str, cfg: dict, baseline_machine: bool) -> dict:
    # ★ enable_sink=False：對照實驗不該在 results/ 下寫出事件檔，
    #   不然同一個檔案會被兩組 append，數字會互相污染。
    pipe = Pipeline(cfg, use_baseline_machine=baseline_machine,
                    enable_sink=False, fps_hint=probe_fps(video))
    try:
        for _ in pipe.run(video):
            pass
        return pipe.summary()
    finally:
        pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--baseline-config", default="config/baseline.yaml")
    ap.add_argument("--out", default="results/false_positive_comparison")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bcfg = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))

    rows = []
    for video in args.videos:
        name = Path(video).stem
        print(f"[{name}] 對照組...", flush=True)
        base = run_one(video, bcfg, baseline_machine=True)
        print(f"[{name}] 實驗組...", flush=True)
        ours = run_one(video, cfg, baseline_machine=False)

        # ★ 降幅用「總告警數」計算，包含快通道。
        #   只算慢通道會讓數字好看，但護理師實際收到的是兩條通道的總和。
        #   用對自己不利的算法，數字才站得住。
        rows.append({
            "clip": name,
            "baseline_alerts": base["alerts"],
            "ours_alerts": ours["alerts"],                 # 慢 + 快
            "ours_alerts_slow": ours.get("alerts_slow", ours["alerts"]),
            "ours_alerts_fast": ours.get("alerts_fast", 0),
            "dismissed_in_watching": ours["dismissed_in_watching"],
            "glitch_rejected": ours.get("fast_channel", {}).get("glitch_rejected", 0),
            "unreliable_rate": ours.get("fusion", {}).get("unreliable_rate"),
            "reduction_pct": (
                round(100 * (1 - ours["alerts"] / base["alerts"]), 1)
                if base["alerts"] else None
            ),
            "ours_events": ours["events"],
        })
        print(f"  baseline={base['alerts']}  ours={ours['alerts']}"
              f"（慢 {ours.get('alerts_slow', 0)} / 快 {ours.get('alerts_fast', 0)}）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 偽陽性對照實驗",
        "",
        "Reduction 以**總告警數**計算（慢通道 + 快通道），",
        "不是只算慢通道 —— 護理師收到的是兩條通道的總和。",
        "",
        "| Clip | Baseline alerts | Ours (total) | slow | fast | Reduction |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        red = f"{r['reduction_pct']}%" if r["reduction_pct"] is not None else "n/a"
        lines.append(
            f"| {r['clip']} | {r['baseline_alerts']} | {r['ours_alerts']} | "
            f"{r['ours_alerts_slow']} | {r['ours_alerts_fast']} | {red} |")
    tb, to = sum(r["baseline_alerts"] for r in rows), sum(r["ours_alerts"] for r in rows)
    ts_, tf = (sum(r["ours_alerts_slow"] for r in rows),
               sum(r["ours_alerts_fast"] for r in rows))
    total_red = f"{round(100 * (1 - to / tb), 1)}%" if tb else "n/a"
    lines += [
        "|  |  |  |  |  |  |",
        f"| **Total** | **{tb}** | **{to}** | **{ts_}** | **{tf}** | **{total_red}** |",
        "",
        "## 其他可量測的數字",
        "",
        "| Clip | WATCHING 中取消 | 偵測跳點擋下 | 不可信影格比例 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        ur = ("n/a" if r["unreliable_rate"] is None
              else f"{r['unreliable_rate'] * 100:.1f}%")
        lines.append(f"| {r['clip']} | {r['dismissed_in_watching']} | "
                     f"{r['glitch_rejected']} | {ur} |")
    lines += [
        "",
        "- **WATCHING 中取消**：超標但沒撐過確認時間的次數。",
        "  單閾值版本會把這些全部叫出來。",
        "- **偵測跳點擋下**：快通道偵測到振幅、但位移沒站住而丟棄的次數。",
        "  這是單幀推論出錯的實際發生次數。",
        "- **不可信影格比例**：遮蔽導致無法判定的時間佔比。",
        "  ★ 這個數字要主動講，不要藏 —— 它是系統的已知限制，",
        "  被評審自己發現的代價遠高於自己說出來。",
    ]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - 0.2, [r["baseline_alerts"] for r in rows], 0.4,
               label="Single threshold (baseline)", color="#d94f4f")
        ax.bar(x + 0.2, [r["ours_alerts"] for r in rows], 0.4,
               label="Temporal state machine (ours)", color="#4f9fd9")
        ax.set_xticks(x, [r["clip"] for r in rows], rotation=20, ha="right")
        ax.set_ylabel("Number of alerts")
        ax.set_title("False positive reduction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=150)
        print(f"✓ 圖表 → {out.with_suffix('.png')}")
    except ImportError:
        print("（未安裝 matplotlib，略過繪圖）")

    print(f"✓ 數據 → {out.with_suffix('.json')}")
    print(f"✓ 表格 → {out.with_suffix('.md')}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
