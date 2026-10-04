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


def run_one(video: str, cfg: dict, baseline_machine: bool) -> dict:
    pipe = Pipeline(cfg, use_baseline_machine=baseline_machine)
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

        rows.append({
            "clip": name,
            "baseline_alerts": base["alerts"],
            "ours_alerts": ours["alerts"],
            "dismissed_in_watching": ours["dismissed_in_watching"],
            "reduction_pct": (
                round(100 * (1 - ours["alerts"] / base["alerts"]), 1)
                if base["alerts"] else None
            ),
            "ours_events": ours["events"],
        })
        print(f"  baseline={base['alerts']}  ours={ours['alerts']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 偽陽性對照實驗",
        "",
        "| Clip | Baseline alerts | Ours | Reduction |",
        "|---|---|---|---|",
    ]
    for r in rows:
        red = f"{r['reduction_pct']}%" if r["reduction_pct"] is not None else "n/a"
        lines.append(
            f"| {r['clip']} | {r['baseline_alerts']} | {r['ours_alerts']} | {red} |")
    tb, to = sum(r["baseline_alerts"] for r in rows), sum(r["ours_alerts"] for r in rows)
    total_red = f"{round(100 * (1 - to / tb), 1)}%" if tb else "n/a"
    lines += ["|  |  |  |  |", f"| **Total** | **{tb}** | **{to}** | **{total_red}** |"]
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
