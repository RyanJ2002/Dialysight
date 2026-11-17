#!/usr/bin/env python3
"""
CPU 與 Hailo 的等價驗證 —— Stage II 最重要的一張投影片。

要回答的問題：換到 UGen300 之後，警報還是同一批嗎？

用法（三步，前兩步跑在有卡的機器上）：

    # 1. CPU 基準
    python scripts/dump_keypoints.py --video data/clips/clip_03.mp4 ^
        --out results/equivalence/clip_03_cpu.json --backend ultralytics

    # 2. Hailo —— 同一份 config，只用 --backend 覆寫那一行。
    #    不要另外開 config/hailo.yaml：多一份 config 就會悄悄差第二個地方，
    #    Stage I 的 28/29/30 就是這樣來的。
    python scripts/dump_keypoints.py --video data/clips/clip_03.mp4 ^
        --out results/equivalence/clip_03_hailo.json ^
        --backend hailo --hef models/yolov8s_pose.hef

    # 3. 比對（不需要硬體，任何機器都能跑）
    python scripts/verify_equivalence.py \
        --cpu   results/equivalence/clip_03_cpu.json \
        --hailo results/equivalence/clip_03_hailo.json

----------------------------------------------------------------------
為什麼要分三層比
----------------------------------------------------------------------
只比關鍵點會得到「平均誤差 2.3 px」這種聽起來很好但不回答問題的數字。
只比事件會在數字對上時說不出為什麼對、對不上時說不出哪裡壞了。

  第一層 關鍵點：INT8 量化造成的像素級差異。預期有，不應該大。
  第二層 特徵  ：誤差經過特徵計算後放大或抵銷了多少。
  第三層 事件  ：★ 真正要主張的那一層。護理師看到的是事件，不是像素。

關鍵點有誤差但事件完全一致 —— 這才是能站上台講的結論，
因為它說明了時序層對量化誤差的容忍度，而不是假裝沒有誤差。

----------------------------------------------------------------------
誠實的前提
----------------------------------------------------------------------
事件數對不上不是失敗，是結果。評審會問「你怎麼知道換硬體沒壞掉」，
拿得出這份報告（不管數字好壞）比拿不出來強得多。
對不上就寫對不上，連同第一層與第二層的數字一起放，讓人看見原因。
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.base import KP            # noqa: E402
from src.pipeline import Pipeline            # noqa: E402

# features.py 真正讀到的點。全 17 點的平均會被臉部細節稀釋，
# 但事件只由這幾個點決定，所以要單獨列出來。
FEATURE_KPS = [KP.RIGHT_WRIST, KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW,
               KP.LEFT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


def load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "frames" not in d:
        raise SystemExit(f"{path} 不是 dump_keypoints.py 的輸出格式")
    return d


def video_dims(path: str | None) -> tuple[int, int] | None:
    if not path or not Path(path).is_file():
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h) if w and h else None
    except Exception:
        return None


# ----------------------------------------------------------------------
# 第一層：關鍵點
# ----------------------------------------------------------------------

def compare_keypoints(a: dict, b: dict, dims) -> dict:
    fa = {f["frame_idx"]: f for f in a["frames"]}
    fb = {f["frame_idx"]: f for f in b["frames"]}
    shared = sorted(fa.keys() & fb.keys())

    # 兩邊都認為有效的點才量位移；有效性翻轉另外算，不要混在一起平均
    dists_all: list[float] = []
    dists_feat: list[float] = []
    dconf: list[float] = []
    flips = 0
    both_valid = 0
    total = 0

    for i in shared:
        ka, kb = fa[i]["keypoints"], fb[i]["keypoints"]
        for j in range(len(ka)):
            total += 1
            va, vb = ka[j][2] > 0.0, kb[j][2] > 0.0
            if va != vb:
                flips += 1
                continue
            if not va:
                continue
            both_valid += 1
            d = ((ka[j][0] - kb[j][0]) ** 2 + (ka[j][1] - kb[j][1]) ** 2) ** 0.5
            dists_all.append(d)
            if j in [int(k) for k in FEATURE_KPS]:
                dists_feat.append(d)
            dconf.append(abs(ka[j][2] - kb[j][2]))

    scale = max(dims) if dims else 1.0        # 正規化 → 長邊像素
    out = {
        "frames_cpu": len(fa), "frames_hailo": len(fb),
        "frames_shared": len(shared),
        "frame_count_match": len(fa) == len(fb),
        "keypoints_compared": total,
        "both_valid": both_valid,
        "validity_flips": flips,
        "validity_flip_pct": round(100.0 * flips / total, 3) if total else None,
        "px_scale": scale if dims else None,
        "dims": list(dims) if dims else None,
    }
    for label, xs in (("all17", dists_all), ("feature_kps", dists_feat)):
        out[label] = {
            "n": len(xs),
            "median_norm": round(statistics.median(xs), 6) if xs else None,
            "p95_norm": round(pct(xs, 95), 6) if xs else None,
            "max_norm": round(max(xs), 6) if xs else None,
            "median_px": round(statistics.median(xs) * scale, 2) if xs and dims else None,
            "p95_px": round(pct(xs, 95) * scale, 2) if xs and dims else None,
            "max_px": round(max(xs) * scale, 2) if xs and dims else None,
        }
    out["conf_abs_diff"] = {
        "median": round(statistics.median(dconf), 4) if dconf else None,
        "p95": round(pct(dconf, 95), 4) if dconf else None,
    }
    return out


# ----------------------------------------------------------------------
# 第二、三層：把兩份 dump 餵進同一條時序層
# ----------------------------------------------------------------------

def run_pipeline(cfg: dict, json_path: Path, fps: float, baseline: bool):
    """用 replay backend 跑完整時序層。enable_sink=False，不在 results/ 留垃圾。"""
    c = copy.deepcopy(cfg)
    c["inference"] = {"backend": "replay", "keypoints_json": str(json_path)}
    pipe = Pipeline(c, use_baseline_machine=baseline,
                    enable_sink=False, fps_hint=fps)
    primary = getattr(pipe, "primary", None) or "arm_displacement"
    series: list[tuple[float, float, float]] = []
    try:
        for r in pipe.run(None):
            series.append((r.obs.timestamp,
                           float(r.features.get(primary, float("nan"))),
                           float(r.deviations.get(primary, float("nan")))))
        summary = pipe.summary()
    finally:
        pipe.close()
    return summary, series, primary


def compare_series(sa, sb) -> dict:
    """第二層：逐格比平滑後特徵值與偏離量。"""
    n = min(len(sa), len(sb))
    out = {"frames": n}
    for k, idx in (("feature", 1), ("deviation", 2)):
        d = [abs(sa[i][idx] - sb[i][idx]) for i in range(n)
             if np.isfinite(sa[i][idx]) and np.isfinite(sb[i][idx])]
        out[k] = {
            "n": len(d),
            "median": round(statistics.median(d), 6) if d else None,
            "p95": round(pct(d, 95), 6) if d else None,
            "max": round(max(d), 6) if d else None,
        }
    return out


def compare_events(ea: list[dict], eb: list[dict], tol: float) -> dict:
    """第三層：以 channel + feature + 最近 start 配對。"""
    unmatched_b = list(range(len(eb)))
    pairs, only_a = [], []

    for i, x in enumerate(ea):
        best, best_dt = None, None
        for j in unmatched_b:
            y = eb[j]
            if y["channel"] != x["channel"] or y["feature"] != x["feature"]:
                continue
            dt = abs(y["start"] - x["start"])
            if dt <= tol and (best_dt is None or dt < best_dt):
                best, best_dt = j, dt
        if best is None:
            only_a.append(i)
        else:
            unmatched_b.remove(best)
            pairs.append({
                "channel": x["channel"], "feature": x["feature"],
                "cpu_start": x["start"], "hailo_start": eb[best]["start"],
                "dt_start": round(best_dt, 3),
                "dt_end": (round(abs((x["end"] or 0) - (eb[best]["end"] or 0)), 3)
                           if x["end"] is not None and eb[best]["end"] is not None
                           else None),
                "dpeak": round(abs(x["peak"] - eb[best]["peak"]), 5),
            })

    dts = [p["dt_start"] for p in pairs]
    return {
        "cpu_count": len(ea), "hailo_count": len(eb),
        "count_match": len(ea) == len(eb),
        "matched": len(pairs),
        "identical": len(ea) == len(eb) == len(pairs),
        "tolerance_s": tol,
        "max_dt_start": round(max(dts), 3) if dts else None,
        "median_dt_start": round(statistics.median(dts), 3) if dts else None,
        "pairs": pairs,
        "cpu_only": [ea[i] for i in only_a],
        "hailo_only": [eb[j] for j in unmatched_b],
    }


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="CPU 與 Hailo 的等價驗證")
    ap.add_argument("--cpu", required=True, help="ultralytics 後端的 dump")
    ap.add_argument("--hailo", required=True, help="hailo 後端的 dump")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--baseline-config", default="config/baseline.yaml",
                    help="對照組 config，用來確認兩邊的 30 次也一致")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="事件配對的起始時間容許差（秒）")
    ap.add_argument("--out-dir", default="results/equivalence")
    args = ap.parse_args()

    cpu_p, hailo_p = Path(args.cpu), Path(args.hailo)
    a, b = load(cpu_p), load(hailo_p)
    fps = float(a.get("fps") or 30.0)
    dims = video_dims(a.get("video_path"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("CPU / Hailo 等價驗證")
    print("=" * 64)
    print(f"CPU    {cpu_p}   {len(a['frames'])} frames")
    print(f"Hailo  {hailo_p}   {len(b['frames'])} frames")
    print(f"影片   {a.get('video_path')}  {fps:.2f} fps"
          f"{f'  {dims[0]}x{dims[1]}' if dims else '  (解析度未知)'}")

    if len(a["frames"]) != len(b["frames"]):
        print("\n⚠️  影格數不同。兩邊必須餵同一支影片的同一段，"
              "否則下面的數字沒有意義。")

    print("\n[第一層] 關鍵點 …", flush=True)
    kp = compare_keypoints(a, b, dims)
    unit = "px" if dims else "正規化"
    key = "median_px" if dims else "median_norm"
    print(f"  有效性翻轉 {kp['validity_flips']} / {kp['keypoints_compared']}"
          f"  ({kp['validity_flip_pct']}%)")
    print(f"  全 17 點位移中位數 {kp['all17'][key]} {unit}"
          f"   p95 {kp['all17']['p95_px' if dims else 'p95_norm']} {unit}")
    print(f"  特徵用點位移中位數 {kp['feature_kps'][key]} {unit}"
          f"   p95 {kp['feature_kps']['p95_px' if dims else 'p95_norm']} {unit}")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    arms: dict[str, dict] = {}

    for name, cfg_path, use_baseline in (
            ("dialysight", args.config, False),
            ("baseline", args.baseline_config, True)):
        p = Path(cfg_path)
        if not p.is_file():
            print(f"\n[跳過] 找不到 {p}")
            continue
        c = yaml.safe_load(p.read_text(encoding="utf-8"))
        print(f"\n[第二、三層] {name}（{p.name}）…", flush=True)
        try:
            sa, ser_a, primary = run_pipeline(c, cpu_p, fps, use_baseline)
            sb, ser_b, _ = run_pipeline(c, hailo_p, fps, use_baseline)
        except Exception as e:
            print(f"  ✗ 跑不起來：{type(e).__name__}: {e}")
            arms[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        ser = compare_series(ser_a, ser_b)
        ev = compare_events(sa["events"], sb["events"], args.tol)
        arms[name] = {"config": str(p), "primary_feature": primary,
                      "series": ser, "events": ev,
                      "cpu_summary": {k: sa[k] for k in
                                      ("alerts", "alerts_slow", "alerts_fast")},
                      "hailo_summary": {k: sb[k] for k in
                                        ("alerts", "alerts_slow", "alerts_fast")}}

        print(f"  特徵值逐格差異  中位數 {ser['feature']['median']}"
              f"  p95 {ser['feature']['p95']}  max {ser['feature']['max']}")
        print(f"  警報數  CPU {ev['cpu_count']}  Hailo {ev['hailo_count']}"
              f"  → {'完全一致' if ev['identical'] else '不一致'}")
        if ev["matched"]:
            print(f"  配對成功 {ev['matched']} 組，"
                  f"起始時間最大差 {ev['max_dt_start']} s")
        for side in ("cpu_only", "hailo_only"):
            for e in ev[side]:
                print(f"  ⚠️  只有 {side.split('_')[0]} 有："
                      f"{e['channel']}/{e['feature']} @ {e['start']}s")

    payload = {"cpu": str(cpu_p), "hailo": str(hailo_p),
               "video": a.get("video_path"), "fps": fps,
               "dims": list(dims) if dims else None,
               "keypoints": kp, "arms": arms}
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Markdown ----------------------------------------------------
    L = ["# CPU / Hailo 等價驗證", "",
         f"- CPU dump：`{cpu_p}`（{len(a['frames'])} frames）",
         f"- Hailo dump：`{hailo_p}`（{len(b['frames'])} frames）",
         f"- 影片：`{a.get('video_path')}`，{fps:.2f} fps"
         + (f"，{dims[0]}x{dims[1]}" if dims else ""), ""]

    d = arms.get("dialysight", {}).get("events")
    if d:
        L += ["## 結論", "",
              ("✅ **事件完全一致**："
               f"兩邊都是 {d['cpu_count']} 次警報，"
               f"起始時間最大差 {d['max_dt_start']} 秒。"
               if d["identical"] else
               f"⚠️ **事件不一致**：CPU {d['cpu_count']} 次、"
               f"Hailo {d['hailo_count']} 次，配對成功 {d['matched']} 組。"),
              "", "關鍵點層存在 INT8 量化造成的像素級差異（見下），"
              "事件層" + ("不受影響" if d["identical"] else "受到影響") + "。", ""]

    L += ["## 第一層　關鍵點", "", "| 項目 | 值 |", "|---|---|",
          f"| 比較的關鍵點數 | {kp['keypoints_compared']:,} |",
          f"| 兩邊都有效 | {kp['both_valid']:,} |",
          f"| 有效性翻轉 | {kp['validity_flips']:,}（{kp['validity_flip_pct']}%）|",
          ""]
    L += ["| 位移 | 中位數 | p95 | 最大 |", "|---|---|---|---|"]
    for label, zh in (("all17", "全 17 點"), ("feature_kps", "特徵用 6 點")):
        s = kp[label]
        if dims:
            L.append(f"| {zh} | {s['median_px']} px | {s['p95_px']} px "
                     f"| {s['max_px']} px |")
        else:
            L.append(f"| {zh} | {s['median_norm']} | {s['p95_norm']} "
                     f"| {s['max_norm']} |")
    L += ["", "> 特徵用 6 點 = 右腕、右肩、右肘、左肩、左髖、右髖。"
          "全 17 點的平均會被臉部細節稀釋，但事件只由這 6 點決定。", ""]

    for name, zh in (("dialysight", "DialySight"), ("baseline", "對照組")):
        arm = arms.get(name)
        if not arm:
            continue
        L += [f"## {zh}（`{Path(arm.get('config','?')).name}`）", ""]
        if "error" in arm:
            L += [f"跑不起來：`{arm['error']}`", ""]
            continue
        ev, ser = arm["events"], arm["series"]
        L += ["### 第二層　特徵", "", "| 項目 | 中位數 | p95 | 最大 |",
              "|---|---|---|---|",
              f"| 平滑後特徵值 | {ser['feature']['median']} "
              f"| {ser['feature']['p95']} | {ser['feature']['max']} |",
              f"| 相對基準偏離量 | {ser['deviation']['median']} "
              f"| {ser['deviation']['p95']} | {ser['deviation']['max']} |", "",
              "### 第三層　事件", "",
              f"- CPU **{ev['cpu_count']}** 次（慢 "
              f"{arm['cpu_summary']['alerts_slow']} / 快 "
              f"{arm['cpu_summary']['alerts_fast']}）",
              f"- Hailo **{ev['hailo_count']}** 次（慢 "
              f"{arm['hailo_summary']['alerts_slow']} / 快 "
              f"{arm['hailo_summary']['alerts_fast']}）",
              f"- 配對成功 {ev['matched']} 組"
              + (f"，起始時間最大差 **{ev['max_dt_start']} 秒**"
                 if ev["matched"] else ""), ""]
        if ev["pairs"]:
            L += ["| 通道 | 特徵 | CPU 起始 | Hailo 起始 | Δ起始 | Δ峰值 |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {p['channel']} | {p['feature']} | {p['cpu_start']} "
                  f"| {p['hailo_start']} | {p['dt_start']} | {p['dpeak']} |"
                  for p in ev["pairs"]]
            L.append("")
        for side, zh2 in (("cpu_only", "只有 CPU"), ("hailo_only", "只有 Hailo")):
            if ev[side]:
                L += [f"**{zh2} 偵測到：**", ""]
                L += [f"- {e['channel']}/{e['feature']} @ {e['start']}s "
                      f"（峰值 {e['peak']}）" for e in ev[side]]
                L.append("")

    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "=" * 64)
    print(f"✓ {out_dir / 'report.md'}")
    print(f"✓ {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
