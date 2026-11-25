#!/usr/bin/env python3
"""
HailoBackend 單格煙霧測試 —— 第一次讓我們的程式碼碰到卡。

用法（在 repo 根目錄，有卡的那台）：

    # 最常用：任何有人的照片都行，自拍也可以
    python scripts/smoke_hailo.py --image test.jpg

    # 從影片抓某一格
    python scripts/smoke_hailo.py --video data/clips/clip_03.mp4 --frame 300

    # 同一格同時跑 CPU 比對（★ 這個最有價值，見下）
    python scripts/smoke_hailo.py --image test.jpg --compare

輸出：results/device/smoke_hailo.png（骨架疊圖）與主控台的 17 點表格

----------------------------------------------------------------------
為什麼要有這支 —— 它把三件事分開
----------------------------------------------------------------------
從 probe_hef.py 直接跳到 dump_keypoints.py 的問題是：一次跑 9000 格，
中間任何一環壞掉都只會得到一個 traceback 或一堆垃圾數字，而且要等很久。
壞掉的可能是三件完全不同的事，修法也完全不同：

  1. 卡跑不動我們的程式
     症狀：例外。多半是 run() 的參數型別、緩衝區綁定、或 VDevice 開不起來。
     這一層跟解碼對不對無關 —— hailortcli run2 能跑就代表 HEF 沒問題。

  2. 卡跑得動，但解碼錯
     症狀：不報錯，17 個點全是 0，或座標落在畫面外。

  3. 解碼對，但通道順序跟我猜的不同
     症狀 ★ 最難抓：骨架位置大致對、但框歪掉或左右肢體互換。
     數字看起來「有在動」，所以很容易以為成功了。

--compare 是專門為第 3 種設計的。同一格、同一個模型族，CPU 與 Hailo
的關鍵點應該差在幾個像素內（INT8 量化）。如果差到幾十像素甚至上百，
那不是量化誤差，是 DFL 的 l,t,r,b 順序或關鍵點的 x,y,v 排列跟 Model Zoo
的慣例不一樣 —— 我是照慣例寫的，沒辦法從 HEF 檔案本身確認。

通過這一支之後才值得跑 dump_keypoints.py 的全片。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.base import KP, make_backend     # noqa: E402

KP_ZH = {
    0: "鼻", 1: "左眼", 2: "右眼", 3: "左耳", 4: "右耳",
    5: "左肩", 6: "右肩", 7: "左肘", 8: "右肘", 9: "左腕", 10: "右腕",
    11: "左髖", 12: "右髖", 13: "左膝", 14: "右膝", 15: "左踝", 16: "右踝",
}

# features.py 真正讀到的點。這幾個錯了事件就錯，其他點錯了不影響判斷。
FEATURE_KPS = {int(k) for k in (KP.RIGHT_WRIST, KP.RIGHT_SHOULDER,
                                KP.RIGHT_ELBOW, KP.LEFT_SHOULDER,
                                KP.LEFT_HIP, KP.RIGHT_HIP)}

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4)]


def load_frame(args) -> np.ndarray:
    import cv2
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"讀不到圖片：{args.image}")
        return frame
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"開不了影片：{args.video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"抓不到第 {args.frame} 格")
        return frame
    raise SystemExit(
        "要給 --image 或 --video。\n"
        "沒有素材的話，隨便一張有人的照片就夠測了（自拍也行）：\n"
        "  python scripts/smoke_hailo.py --image test.jpg")


def draw(frame: np.ndarray, kps, path: Path) -> None:
    import cv2
    h, w = frame.shape[:2]
    out = frame.copy()
    pt = [(int(k.x * w), int(k.y * h)) if k.is_valid else None for k in kps]
    for a, b in SKELETON:
        if pt[a] and pt[b]:
            cv2.line(out, pt[a], pt[b], (0, 255, 0), 2)
    for i, p in enumerate(pt):
        if not p:
            continue
        feat = i in FEATURE_KPS
        cv2.circle(out, p, 6 if feat else 4,
                   (0, 0, 255) if feat else (255, 200, 0), -1)
        if feat:
            cv2.putText(out, str(i), (p[0] + 7, p[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def main() -> None:
    ap = argparse.ArgumentParser(description="HailoBackend 單格煙霧測試")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--image")
    g.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0, help="搭配 --video")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--hef", default=None)
    ap.add_argument("--repeat", type=int, default=30, help="量延遲用的重複次數")
    ap.add_argument("--compare", action="store_true",
                    help="同一格也跑 ultralytics CPU 比對（抓通道順序問題）")
    ap.add_argument("--out", default="results/device/smoke_hailo.png")
    args = ap.parse_args()

    frame = load_frame(args)
    h, w = frame.shape[:2]
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["inference"]["backend"] = "hailo"
    if args.hef:
        cfg["inference"]["hef"] = args.hef

    print("=" * 64)
    print("HailoBackend 單格煙霧測試")
    print("=" * 64)
    print(f"輸入      {args.image or f'{args.video} 第 {args.frame} 格'}  {w}x{h}")
    print(f"HEF       {cfg['inference'].get('hef')}")
    print(f"conf/iou  {cfg['inference'].get('conf')} / "
          f"{cfg['inference'].get('iou', 0.7)}")

    # ---- 第 1 關：卡跑得動我們的程式嗎 -------------------------------
    print("\n[1/4] 開啟裝置與載入 HEF …", flush=True)
    t0 = time.perf_counter()
    try:
        backend = make_backend(cfg)
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")
        print("\n這一關失敗跟解碼無關。先確認 hailortcli run2 set-net 跑得動："
              "\n  hailortcli run2 set-net models\\yolov8s_pose.hef"
              "\n能跑就是 Python 那層的問題（緩衝區綁定或 run() 參數）。")
        raise SystemExit(1)
    print(f"  ✓ {(time.perf_counter()-t0)*1000:.0f} ms")

    try:
        print("\n[2/4] 單格推論 …", flush=True)
        obs = backend.infer(frame, 0, 0.0)
        print(f"  ✓ meta = {obs.meta}")

        valid = [i for i, k in enumerate(obs.keypoints) if k.is_valid]
        print(f"\n[3/4] 解碼結果：{len(valid)} / 17 點有效"
              f"（可見比例 {obs.visibility_ratio():.1%}）")

        if not valid:
            print("\n  ⚠️  一個點都沒有。三種可能，由易到難：")
            print("     a) 畫面裡沒有人 —— 換一張有人的照片再試")
            print(f"     b) conf 門檻太高（現在 {cfg['inference'].get('conf')}）"
                  " —— 試 --config 改成 0.05 看有沒有東西冒出來")
            print("     c) 解碼錯了 —— 這時候 meta 的 people 也會是 0")
            print(f"\n     現在 meta['people'] = {obs.meta.get('people')}")
            print("     people > 0 但點全是 0：關鍵點分支解錯")
            print("     people = 0：分類分支解錯，或真的沒人")
        else:
            print(f"\n  {'idx':>4} {'部位':<5} {'x':>8} {'y':>8} "
                  f"{'px':>12} {'conf':>6}")
            print("  " + "-" * 50)
            for i, k in enumerate(obs.keypoints):
                mark = "★" if i in FEATURE_KPS else " "
                if k.is_valid:
                    print(f"  {mark}{i:>3} {KP_ZH[i]:<5} {k.x:>8.4f} {k.y:>8.4f} "
                          f"{f'({int(k.x*w)},{int(k.y*h)})':>12} "
                          f"{k.confidence:>6.3f}")
                else:
                    print(f"  {mark}{i:>3} {KP_ZH[i]:<5} {'—':>8} {'—':>8} "
                          f"{'缺失':>12} {0.0:>6.3f}")
            print("\n  ★ = features.py 真正讀到的點。這幾個錯了事件就錯。")

            oob = [i for i in valid
                   if not (0.0 <= obs.keypoints[i].x <= 1.0
                           and 0.0 <= obs.keypoints[i].y <= 1.0)]
            if oob:
                print(f"\n  ⚠️  {len(oob)} 個點落在畫面外："
                      f"{[KP_ZH[i] for i in oob]}")
                print("     正規化座標應該都在 0~1。超出範圍代表 letterbox 的"
                      " padding 還原錯了。")

        draw(frame, obs.keypoints, Path(args.out))
        print(f"\n  ✓ 疊圖 {args.out} —— 打開看骨架貼不貼身，"
              "這比數字直觀")

        # ---- 第 4 關：延遲 -------------------------------------------
        print(f"\n[4/4] 延遲（{args.repeat} 次）…", flush=True)
        ms = []
        for i in range(args.repeat):
            t = time.perf_counter()
            backend.infer(frame, i + 1, 0.0)
            ms.append((time.perf_counter() - t) * 1000)
        ms_s = sorted(ms)
        med = ms_s[len(ms_s) // 2]
        print(f"  中位數 {med:.2f} ms  →  {1000/med:.1f} FPS"
              f"   最小 {ms_s[0]:.2f}  最大 {ms_s[-1]:.2f}")
        print(f"  （hailortcli benchmark 冷機報 161.93 FPS。"
              f"我們這邊含 letterbox 與解碼，比它低是正常的）")
    finally:
        backend.close()

    payload = {"input": args.image or f"{args.video}#{args.frame}",
               "dims": [w, h], "meta": obs.meta,
               "latency_ms": {"median": round(med, 2), "min": round(ms_s[0], 2),
                              "max": round(ms_s[-1], 2),
                              "fps": round(1000 / med, 1)},
               "keypoints": [[round(k.x, 5), round(k.y, 5), round(k.confidence, 4)]
                             for k in obs.keypoints]}

    # ---- 可選：同一格的 CPU 比對 --------------------------------------
    if args.compare:
        print("\n" + "=" * 64)
        print("同一格 CPU 比對 ★ 這段抓通道順序問題")
        print("=" * 64, flush=True)
        c = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        c["inference"]["backend"] = "ultralytics"
        try:
            cpu = make_backend(c)
        except Exception as e:
            print(f"  跳過：ultralytics 載不起來（{type(e).__name__}）")
        else:
            try:
                cobs = cpu.infer(frame, 0, 0.0)
            finally:
                cpu.close()
            scale = max(w, h)
            rows, ds = [], []
            for i in range(17):
                a, b = cobs.keypoints[i], obs.keypoints[i]
                if not (a.is_valid and b.is_valid):
                    rows.append((i, None, a.is_valid, b.is_valid))
                    continue
                d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 * scale
                rows.append((i, d, True, True))
                ds.append(d)
            print(f"\n  {'idx':>4} {'部位':<5} {'差異 px':>9}")
            print("  " + "-" * 24)
            for i, d, va, vb in rows:
                mark = "★" if i in FEATURE_KPS else " "
                s = f"{d:>9.1f}" if d is not None else (
                    f"{'僅 CPU':>9}" if va else
                    f"{'僅 Hailo':>9}" if vb else f"{'兩邊缺':>9}")
                print(f"  {mark}{i:>3} {KP_ZH[i]:<5} {s}")
            if ds:
                feat_ds = [d for i, d, va, vb in rows
                           if d is not None and i in FEATURE_KPS]
                med_d = sorted(ds)[len(ds) // 2]
                print(f"\n  全 17 點中位數 {med_d:.1f} px   最大 {max(ds):.1f} px")
                if feat_ds:
                    print(f"  特徵用 6 點中位數 "
                          f"{sorted(feat_ds)[len(feat_ds)//2]:.1f} px"
                          f"   最大 {max(feat_ds):.1f} px")
                print()
                if med_d < 10:
                    print("  ✅ 幾個像素內 —— 這是 INT8 量化該有的樣子。"
                          "可以往下跑全片 dump 了。")
                elif med_d < 30:
                    print("  ⚠️  偏大。先看疊圖：骨架貼身就只是量化較差，"
                          "不貼身要回頭查 letterbox。")
                else:
                    print("  ✗ 太大，不是量化誤差。優先懷疑 _decode() 的通道順序："
                          "\n     DFL 的 l,t,r,b 排列，或關鍵點的 x,y,v 排列。"
                          "\n     把疊圖和這張表一起貼回來。")
                payload["compare_px"] = {
                    "median": round(med_d, 2), "max": round(max(ds), 2),
                    "feature_median": (round(sorted(feat_ds)[len(feat_ds)//2], 2)
                                       if feat_ds else None)}

    out_json = Path(args.out).with_suffix(".json")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n✓ {out_json}")


if __name__ == "__main__":
    main()
