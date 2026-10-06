#!/usr/bin/env python3
"""
特徵診斷圖 —— A 用這支腳本判斷「相機角度可不可用」。

★ 為什麼不能只看影片：
  看影片只能看到「人有沒有動」，看不到「模型抓到的關鍵點準不準」。
  人動了但模型沒抓到手腕，曲線一樣是平的 —— 那是相機角度的問題，
  用肉眼看影片永遠發現不了。

用法：
    # 先把影片轉成關鍵點 JSON（只要跑一次）
    python scripts/dump_keypoints.py --video data/clips/clip_01.mp4 \
                                     --out data/clips/clip_01.json

    # 再畫診斷圖
    python scripts/plot_feature.py --json data/clips/clip_01.json \
                                   --out results/diag_clip_01.png

    # 一次比較多支（建議：靜臥 + 漸進事件放一起看）
    python scripts/plot_feature.py --json data/clips/clip_01.json \
                                          data/clips/clip_03.json \
                                   --out results/diag_all.png

產出四張子圖：
    ① 關鍵點可見率 —— 相機角度好不好的第一指標
    ② arm_displacement 原始 vs 平滑 —— 主訊號
    ③ torso_tilt_deg —— 躺姿偵測（★ 不是 torso_angle，見第 ③ 格註解）
    ④ 各關鍵點的信心值分布 —— 調 min_confidence 的依據

★ 放行標準（三個都要過才回報「角度可用」）：
    1. 穿刺側手腕與肩膀的可見率 > 80%
    2. 靜臥段的 arm_displacement 標準差 < 0.02
    3. 手臂移開段有肉眼可見的上升
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.base import NUM_KEYPOINTS, KP, FrameObservation, Keypoint
from src.inference.subject import torso_tilt_deg
from src.temporal.features import FeatureConfig, FeatureExtractor
from src.temporal.filter import FilterConfig, filter_keypoints
from src.temporal.smooth import MedianFilter

# 放行門檻
VISIBILITY_PASS = 0.80
STILL_STD_PASS = 0.02


def load(path: str) -> tuple[list[FrameObservation], float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fps = payload.get("fps", 30.0)
    obs = [
        FrameObservation(
            frame_idx=fr["frame_idx"],
            timestamp=fr["timestamp"],
            keypoints=[Keypoint(*k) for k in fr["keypoints"]],
        )
        for fr in payload["frames"]
    ]
    return obs, fps


def analyse(obs_list: list[FrameObservation], side: str = "right") -> dict:
    fcfg = FilterConfig()
    ex = FeatureExtractor(FeatureConfig(
        wrist_side=side, enabled=["arm_displacement", "torso_angle"]
    ))
    med = MedianFilter(window=7)

    ts, raw_disp, sm_disp, angle, tilt, vis = [], [], [], [], [], []
    conf_by_kp = [[] for _ in range(NUM_KEYPOINTS)]

    for o in obs_list:
        for i, kp in enumerate(o.keypoints):
            conf_by_kp[i].append(kp.confidence)

        f = filter_keypoints(o, fcfg)
        feats = ex(f)

        ts.append(o.timestamp)
        raw_disp.append(feats["arm_displacement"])
        sm_disp.append(med.update(feats["arm_displacement"]))
        angle.append(feats["torso_angle"])
        # ★ 躺姿要看這個，不是 torso_angle。
        #   torso_angle 是「肩線 vs 髖線」的夾角，站著、躺著、坐起
        #   全都接近 0°，它分辨不出姿勢 —— 它量的是身體有沒有扭轉。
        #   真正判躺姿的是肩中心→髖中心向量與水平線的夾角，
        #   也就是 subject.py 的 torso_tilt_deg，45° 門檻是給它用的。
        t = torso_tilt_deg(f.keypoints)
        tilt.append(float("nan") if t is None else t)
        vis.append(f.visibility_ratio())

    wrist_i = int(KP.RIGHT_WRIST if side == "right" else KP.LEFT_WRIST)
    shoulder_i = int(KP.RIGHT_SHOULDER if side == "right" else KP.LEFT_SHOULDER)

    def seen(i: int) -> float:
        c = np.array(conf_by_kp[i])
        return float(np.mean(c >= fcfg.min_confidence))

    arr = np.array(raw_disp, dtype=float)
    valid = arr[~np.isnan(arr)]

    return {
        "ts": np.array(ts),
        "raw": arr,
        "smooth": np.array(sm_disp, dtype=float),
        "angle": np.array(angle, dtype=float),
        "tilt": np.array(tilt, dtype=float),
        "visibility": np.array(vis),
        "conf_by_kp": conf_by_kp,
        "wrist_seen": seen(wrist_i),
        "shoulder_seen": seen(shoulder_i),
        "nan_rate": float(np.mean(np.isnan(arr))),
        "overall_std": float(np.std(valid)) if valid.size else float("nan"),
        "duration": ts[-1] if ts else 0.0,
    }


def still_segment_std(a: dict, window_s: float = 20.0) -> float:
    """取最平穩的 20 秒窗的標準差，當作「靜臥段」的代表值。

    這樣就不需要人工標出哪一段是靜臥。
    """
    ts, sm = a["ts"], a["smooth"]
    mask = ~np.isnan(sm)
    if mask.sum() < 50:
        return float("nan")

    t, v = ts[mask], sm[mask]
    best = float("inf")
    i = 0
    for j in range(len(t)):
        while t[j] - t[i] > window_s:
            i += 1
        if t[j] - t[i] >= window_s * 0.8 and j - i >= 30:
            best = min(best, float(np.std(v[i:j + 1])))
    return best if best < float("inf") else float(np.std(v))


def plot(results: dict[str, dict], out: str, side: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 13))
    colors = plt.cm.tab10.colors

    # ① 可見率
    ax = axes[0]
    for k, (name, a) in enumerate(results.items()):
        ax.plot(a["ts"], a["visibility"], color=colors[k % 10], lw=1.2, label=name)
    ax.axhline(VISIBILITY_PASS, color="r", ls="--", lw=1,
               label=f"pass line {VISIBILITY_PASS}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("visibility ratio")
    ax.set_title("(1) Keypoint visibility — camera angle health")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # ② arm_displacement
    ax = axes[1]
    for k, (name, a) in enumerate(results.items()):
        ax.plot(a["ts"], a["raw"], color=colors[k % 10], lw=.7, alpha=.35)
        ax.plot(a["ts"], a["smooth"], color=colors[k % 10], lw=1.8, label=f"{name} (smoothed)")
    ax.set_ylabel("arm_displacement")
    ax.set_title("(2) arm_displacement — faint = raw, bold = median filtered")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # ③ 躺姿判定 —— ★ 畫 torso_tilt_deg，不是 torso_angle
    #
    #   原本這一格畫的是 torso_angle 並標「low = lying」，那是錯的：
    #   torso_angle 是肩線與髖線的夾角，站立、平躺、坐起全都接近 0°。
    #   照那個標籤看圖，會誤以為躺姿判定正常。
    #
    #   真正決定 SubjectSelector 選不選這個人的是 torso_tilt_deg
    #   （肩中心→髖中心 與水平線的夾角）：躺著接近 0°，站著接近 90°。
    ax = axes[2]
    for k, (name, a) in enumerate(results.items()):
        ax.plot(a["ts"], a["tilt"], color=colors[k % 10], lw=1.6, label=name)
        ax.plot(a["ts"], a["angle"], color=colors[k % 10], lw=0.8, alpha=.35,
                ls=":", label=f"{name} · torso_angle (twist, NOT posture)")
    ax.axhline(45, color="r", ls="--", lw=1,
               label="lying threshold 45° — above this = not lying")
    ax.set_ylim(-5, 95)
    ax.set_ylabel("torso tilt (deg)")
    ax.set_xlabel("time (s)")
    ax.set_title("(3) torso_tilt_deg — 0° = lying flat, 90° = standing/sitting up"
                 "   (this is what SubjectSelector uses)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # ④ 各關鍵點信心值
    ax = axes[3]
    first = next(iter(results.values()))
    names = [KP(i).name for i in range(NUM_KEYPOINTS)]
    means = [float(np.mean(first["conf_by_kp"][i])) for i in range(NUM_KEYPOINTS)]
    bars = ax.bar(range(NUM_KEYPOINTS), means, color="#4f9fd9")
    for i in (int(KP.RIGHT_WRIST if side == "right" else KP.LEFT_WRIST),
              int(KP.RIGHT_SHOULDER if side == "right" else KP.LEFT_SHOULDER)):
        bars[i].set_color("#d94f4f")                 # 我們真正在用的兩個點
    ax.axhline(0.5, color="r", ls="--", lw=1, label="min_confidence 0.5")
    ax.set_xticks(range(NUM_KEYPOINTS), names, rotation=70, fontsize=7)
    ax.set_ylabel("mean confidence")
    ax.set_title(f"(4) Per-keypoint confidence — {list(results)[0]} "
                 "(red = the two we actually use)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"✓ {out}")


def verdict(name: str, a: dict) -> bool:
    still_std = still_segment_std(a)
    checks = [
        ("穿刺側手腕可見率", a["wrist_seen"], VISIBILITY_PASS, ">"),
        ("同側肩膀可見率", a["shoulder_seen"], VISIBILITY_PASS, ">"),
        ("最平穩 20 秒的標準差", still_std, STILL_STD_PASS, "<"),
    ]

    print(f"\n{'=' * 60}\n  {name}   長度 {a['duration']:.0f}s\n{'=' * 60}")
    ok = True
    for label, value, thr, op in checks:
        passed = (value > thr) if op == ">" else (value < thr)
        ok &= passed
        mark = "✓" if passed else "✗"
        print(f"  {mark} {label:<22}{value:7.3f}   (需 {op} {thr})")

    print(f"    NaN 比例              {a['nan_rate']:7.3f}")
    print(f"    全段標準差            {a['overall_std']:7.3f}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", default="results/feature_diagnostic.png")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    args = ap.parse_args()

    results, all_ok = {}, True
    for path in args.json:
        obs, _ = load(path)
        name = Path(path).stem
        results[name] = analyse(obs, args.side)
        all_ok &= verdict(name, results[name])

    plot(results, args.out, args.side)

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓ 三項都通過 → 在群組回報「角度可用」，放行 C 正式拍攝")
    else:
        print("  ✗ 有項目未通過 → 看下面的對照表找原因，回報 RyanJ")
    print("=" * 60)
    print("""
  可見率低      → 相機角度不對，手臂被遮住或太遠。調整角度重拍粗拍
  靜臥標準差高  → 相機沒固定，或模型在這個角度抓不穩關鍵點
  看不到上升    → 手臂移動幅度太小，或移動方向正對鏡頭（2D 看不到縱深）
  NaN 比例高    → 關鍵點常缺失，考慮調低 config 的 min_confidence
""")


if __name__ == "__main__":
    main()
