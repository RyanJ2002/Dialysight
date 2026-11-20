#!/usr/bin/env python3
"""
效能量測 —— 分階段拆解延遲、CPU 佔用與記憶體。

★ 這支腳本的產出可以直接貼進簡報。評審看到逐階段的延遲拆解，
  會知道你們量過而不是猜的。

量什麼：
    1. 各階段延遲（解碼 / 推論 / 時序層）的 p50、p95、p99
    2. CPU 佔用率與記憶體
    3. 時序層單獨的成本（證明它可以忽略）
    4. 對照 10 FPS 的 100 ms 預算，看超支多少

用法：
    # 量 Ultralytics CPU 版（Stage I）
    python scripts/benchmark.py --video data/clips/clip_01.mp4 \
        --label "Laptop CPU + Ultralytics" --out results/bench_cpu.json

    # 10/26 之後量 Hailo 版，同一支影片
    python scripts/benchmark.py --video data/clips/clip_01.mp4 \
        --config config/hailo.yaml \
        --label "UGen300 M.2" --out results/bench_hailo.json

    # 產出對照表
    python scripts/benchmark.py --compare results/bench_cpu.json results/bench_hailo.json

★ 兩次量測必須用同一支影片、同一台主機，否則對照無效。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGET_FPS = 10.0
BUDGET_MS = 1000.0 / TARGET_FPS      # 單床每格預算 = 100 ms


# --------------------------------------------------------------------------

def host_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return info


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[k]


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(pct(values, 50), 3),
        "p95": round(pct(values, 95), 3),
        "p99": round(pct(values, 99), 3),
        "max": round(max(values), 3),
    }


# --------------------------------------------------------------------------

def run_benchmark(video: str, cfg: dict, label: str, warmup: int = 10) -> dict:
    """跑完一支影片，回傳分階段的延遲統計。"""
    import cv2
    from src.pipeline import Pipeline

    try:
        import psutil
        proc = psutil.Process()
        proc.cpu_percent(interval=None)          # 第一次呼叫是基準，丟棄
        has_psutil = True
    except ImportError:
        has_psutil = False
        print("⚠️  未安裝 psutil，無法量 CPU 與記憶體：pip install psutil")

    pipe = Pipeline(cfg)

    t_decode: list[float] = []
    t_infer: list[float] = []
    t_temporal: list[float] = []
    t_total: list[float] = []
    cpu_samples: list[float] = []
    rss_samples: list[float] = []

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    wall_start = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()                       # 解碼結束

            obs = pipe.backend.infer(frame, idx, idx / fps)
            t2 = time.perf_counter()                       # 推論結束

            pipe.step(obs)
            t3 = time.perf_counter()                       # 時序層結束

            if idx >= warmup:                              # 前幾格是暖機，丟棄
                t_decode.append((t1 - t0) * 1000)
                t_infer.append((t2 - t1) * 1000)
                t_temporal.append((t3 - t2) * 1000)
                t_total.append((t3 - t0) * 1000)

                if has_psutil and idx % 10 == 0:
                    cpu_samples.append(proc.cpu_percent(interval=None))
                    rss_samples.append(proc.memory_info().rss / 1e6)

            idx += 1
            if idx % 100 == 0:
                print(f"  {idx} frames...", flush=True)
    finally:
        cap.release()
        pipe.close()

    wall = time.perf_counter() - wall_start
    n = len(t_total)

    result = {
        "label": label,
        "video": video,
        "backend": cfg["inference"]["backend"],
        "model": cfg["inference"].get("model", "n/a"),
        "device": cfg["inference"].get("device", "n/a"),
        "frames_measured": n,
        "wall_clock_s": round(wall, 2),
        "achievable_fps": round(n / wall, 1) if wall > 0 else 0,
        "stages_ms": {
            "decode": summarize(t_decode),
            "inference": summarize(t_infer),
            "temporal_layer": summarize(t_temporal),
            "total": summarize(t_total),
        },
        "budget": {
            "target_fps": TARGET_FPS,
            "budget_ms": BUDGET_MS,
            "p95_total_ms": round(pct(t_total, 95), 2),
            "within_budget": pct(t_total, 95) <= BUDGET_MS,
            "utilisation_pct": round(100 * pct(t_total, 95) / BUDGET_MS, 1),
        },
        "host": host_info(),
    }

    if cpu_samples:
        cores = result["host"].get("logical_cores", 1)
        result["resources"] = {
            # psutil 的 cpu_percent 以單核 100% 計，除以核心數換算成整機佔用
            "cpu_pct_of_one_core": round(statistics.mean(cpu_samples), 1),
            "cpu_pct_of_machine": round(statistics.mean(cpu_samples) / cores, 1),
            "peak_cpu_pct_of_one_core": round(max(cpu_samples), 1),
            "rss_mb_mean": round(statistics.mean(rss_samples), 1),
            "rss_mb_peak": round(max(rss_samples), 1),
        }

    return result


# --------------------------------------------------------------------------

def print_report(r: dict) -> None:
    s = r["stages_ms"]
    print()
    print("=" * 62)
    print(f"  {r['label']}")
    print(f"  backend={r['backend']}  model={r['model']}  device={r['device']}")
    print("=" * 62)
    print(f"{'階段':<18}{'p50':>9}{'p95':>9}{'p99':>9}   (ms)")
    print("-" * 62)
    for key, name in [("decode", "影像解碼"), ("inference", "神經網路推論"),
                      ("temporal_layer", "時序層"), ("total", "合計")]:
        d = s[key]
        if d:
            print(f"{name:<16}{d['p50']:>9.2f}{d['p95']:>9.2f}{d['p99']:>9.2f}")
    print("-" * 62)

    b = r["budget"]
    status = "✓ 在預算內" if b["within_budget"] else "✗ 超出預算"
    print(f"10 FPS 預算 {b['budget_ms']:.0f} ms → 用掉 {b['utilisation_pct']}%  {status}")
    print(f"實際可達 FPS: {r['achievable_fps']}")

    if "resources" in r:
        res = r["resources"]
        print(f"CPU 佔用: 單核 {res['cpu_pct_of_one_core']}% "
              f"/ 整機 {res['cpu_pct_of_machine']}%")
        print(f"記憶體: 平均 {res['rss_mb_mean']} MB，峰值 {res['rss_mb_peak']} MB")

    # 時序層佔比 —— 這是我們要證明的重點
    tl, tot = s["temporal_layer"], s["total"]
    if tl and tot and tot["p50"] > 0:
        share = 100 * tl["p50"] / tot["p50"]
        print(f"\n★ 時序層只佔總延遲的 {share:.2f}%（p50 {tl['p50']:.3f} ms）")
    print()


def compare(paths: list[str]) -> None:
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]

    lines = [
        "# 效能對照",
        "",
        "| 項目 | " + " | ".join(r["label"] for r in results) + " |",
        "|---|" + "---|" * len(results),
    ]

    def row(name: str, fn) -> None:
        lines.append(f"| {name} | " + " | ".join(str(fn(r)) for r in results) + " |")

    row("推論延遲 p50 (ms)", lambda r: r["stages_ms"]["inference"]["p50"])
    row("推論延遲 p95 (ms)", lambda r: r["stages_ms"]["inference"]["p95"])
    row("時序層 p50 (ms)", lambda r: r["stages_ms"]["temporal_layer"]["p50"])
    row("端到端 p95 (ms)", lambda r: r["stages_ms"]["total"]["p95"])
    row("可達 FPS", lambda r: r["achievable_fps"])
    row("100 ms 預算用掉", lambda r: f"{r['budget']['utilisation_pct']}%")
    row("CPU（整機 %）", lambda r: r.get("resources", {}).get("cpu_pct_of_machine", "n/a"))
    row("記憶體峰值 (MB)", lambda r: r.get("resources", {}).get("rss_mb_peak", "n/a"))

    out = "\n".join(lines)
    print(out)
    Path("results/benchmark_comparison.md").write_text(out, encoding="utf-8")
    print("\n✓ results/benchmark_comparison.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--label", default="unnamed")
    ap.add_argument("--out")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compare", nargs="+", help="比較多個已產出的 json")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    if not args.video:
        ap.error("需要 --video，或用 --compare 比較既有結果")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    r = run_benchmark(args.video, cfg, args.label, args.warmup)
    print_report(r)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
