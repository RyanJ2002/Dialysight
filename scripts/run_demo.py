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

    # 不要寫事件檔（單純看畫面時用）
    python scripts/run_demo.py --video <影片> --show --no-sink

★ 預設會把事件寫到 config 的 sink.jsonl 指定的路徑。
  那個檔案就是「影像不出床邊」的證據 —— 跑完打開來看，
  裡面只有文字與數字，沒有任何可還原畫面的東西。
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


def probe_fps(video: str) -> float:
    """讀影片的實際 FPS。

    ★ 這個值會傳給 Pipeline，用來換算舊版 config 的 baseline.n_frames，
      以及顏色模組的基準影格數。拿錯會讓基準線建立在錯的時間長度上。
    """
    import cv2
    src = int(video) if str(video).isdigit() else video
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if not (1.0 <= fps <= 240.0):          # 攝影機常常回報 0 或荒謬值
        print(f"[run_demo] 影片未回報合理 FPS（{fps}），改用 30.0")
        fps = 30.0
    return float(fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", help="輸出 mp4 路徑")
    ap.add_argument("--show", action="store_true", help="即時預覽視窗")
    ap.add_argument("--report", help="輸出事件紀錄 markdown")
    ap.add_argument("--baseline-machine", action="store_true",
                    help="強制使用單閾值狀態機（對照組）")
    ap.add_argument("--no-sink", action="store_true",
                    help="不要寫事件檔／不送護理站")
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

    fps = probe_fps(args.video)
    pipe = Pipeline(
        cfg,
        use_baseline_machine=args.baseline_machine,
        enable_sink=not args.no_sink,
        fps_hint=fps,
    )

    import cv2
    writer = make_writer(args.out, fps, layout) if args.out else None
    color_cfg = pipe.color_cfg if pipe.color_enabled else None
    last_ts = 0.0

    try:
        for result in pipe.run(args.video):
            last_ts = result.obs.timestamp
            canvas = renderer.render(
                result.obs,
                result.display_features,     # 偏離量 + 滲血訊號
                result.state,
                result.any_event,            # ★ 兩條通道任一觸發都畫
                result.obs.timestamp,
                color_cfg=color_cfg,
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

    summary = pipe.summary()        # ★ 先取 summary，close() 之後 sink 統計就沒了
    pipe.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    sink_cfg = cfg.get("sink", {})
    if not args.no_sink and sink_cfg.get("jsonl"):
        path = str(sink_cfg["jsonl"]).replace("{bed}", pipe.bed_id)
        print(f"\n✓ 事件輸出 → {path}（打開來看：裡面沒有任何影像）")

    if args.report:
        events = pipe.machine.confirmed_events()
        if pipe.fast is not None:
            events = sorted(events + pipe.fast.confirmed_events(),
                            key=lambda e: e.start_time)
        text = session_report(
            events, last_ts,
            bed_label=cfg.get("visualize", {}).get("bed_label", "Bed 07"),
        )
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"✓ 事件紀錄 → {args.report}")


if __name__ == "__main__":
    main()
