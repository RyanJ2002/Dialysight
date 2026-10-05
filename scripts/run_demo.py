#!/usr/bin/env python3
"""
跑完整管線並輸出 demo 影片。

用法：
    # 即時預覽
    python scripts/run_demo.py --video data/clips/clip_03.mp4 --show

    # 輸出影片（三分鐘 demo 的素材）
    python scripts/run_demo.py --video data/clips/clip_03.mp4 \
                               --out results/clip_03_demo.mp4

    # 用對照組設定跑一次
    python scripts/run_demo.py --video data/clips/clip_02.mp4 \
                               --config config/baseline.yaml \
                               --out results/clip_02_baseline.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Pipeline
from src.report import session_report
from src.visualize import DemoRenderer, LayoutConfig, make_writer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", help="輸出 mp4 路徑")
    ap.add_argument("--show", action="store_true", help="即時預覽視窗")
    ap.add_argument("--report", help="輸出事件紀錄 markdown")
    ap.add_argument("--baseline-machine", action="store_true",
                    help="強制使用單閾值狀態機（對照組）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    layout = LayoutConfig(**{k: v for k, v in cfg.get("visualize", {}).items()
                             if k in LayoutConfig.__dataclass_fields__})
    renderer = DemoRenderer(
        layout=layout,
        thresholds={
            "enter": cfg["state_machine"]["enter_threshold"],
            "exit": cfg["state_machine"]["exit_threshold"],
        },
        primary_feature=cfg["features"]["enabled"][0],
    )

    pipe = Pipeline(cfg, use_baseline_machine=args.baseline_machine)

    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = make_writer(args.out, fps, layout) if args.out else None
    last_ts = 0.0

    try:
        for result in pipe.run(args.video):
            last_ts = result.obs.timestamp
            canvas = renderer.render(
                result.obs, result.deviations, result.state,
                result.new_event, result.obs.timestamp,
                baseline_progress=result.baseline_progress,
            )
            if writer:
                writer.write(canvas)
            if args.show:
                cv2.imshow("DialySight", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        pipe.close()

    summary = pipe.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.report:
        text = session_report(
            pipe.machine.confirmed_events(), last_ts,
            bed_label=cfg.get("visualize", {}).get("bed_label", "Bed 07"),
        )
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"✓ 事件紀錄 → {args.report}")


if __name__ == "__main__":
    main()