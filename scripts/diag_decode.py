#!/usr/bin/env python3
"""
解碼診斷 —— 一次跑完所有可能的組合，讓數字自己說哪個對。

用法：
    python scripts/diag_decode.py --video data\\clips\\demo.mp4 --frame 300
    python scripts/diag_decode.py --image test.jpg

----------------------------------------------------------------------
為什麼需要這支
----------------------------------------------------------------------
smoke_hailo.py 回報 people=2696。總 anchor 數只有 8400，所以三分之一的
anchor 都通過了門檻 —— 分類分數解錯了。

錯法有三個互相獨立的可能，而且從 HEF 檔案本身看不出來：

  1. cls 分支的 sigmoid 已經在晶片上做過（device_pre_post_layers）
     → 我再 sigmoid 一次，值域從 0~1 變成 0.5~0.73，全部過門檻
  2. DFL 的 softmax 已經在晶片上做過
     → 我再 softmax 一次，距離會被壓平，框的大小全部差不多
  3. 51 通道的排列是 component-major（x*17, y*17, v*17）
     而不是 kpt-major（kpt*3 + c）
     → 關鍵點整組錯位

三個都是二選一，所以總共 8 種組合。這支把 8 種全部算出來並排比較。
對的那一組會很明顯：偵測數 1~3、分數高、框合理、關鍵點全部落在畫面內。

猜一次改一次要來回好幾趟。一次量完比較快。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.base import make_backend          # noqa: E402
from src.inference.hailo_backend import (            # noqa: E402
    REG_MAX, _nms, _sigmoid, _softmax)

NK = 17


def load_frame(args) -> np.ndarray:
    import cv2
    if args.image:
        f = cv2.imread(args.image)
        if f is None:
            raise SystemExit(f"讀不到圖片：{args.image}")
        return f
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"開不了影片：{args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"抓不到第 {args.frame} 格")
    return f


def group_heads(outputs: dict, imgsz: int):
    """與 hailo_backend._group_heads 同邏輯，但不依賴實例。"""
    heads: dict[int, dict] = {}
    slot_of = {4 * REG_MAX: "box", 1: "cls", NK * 3: "kpt"}
    for name, t in outputs.items():
        while t.ndim > 3 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            continue
        slot = slot_of.get(t.shape[2])
        if slot:
            heads.setdefault(t.shape[0], {})[slot] = (name, t)
    out = []
    for size in sorted(heads, reverse=True):
        h = heads[size]
        if {"box", "cls", "kpt"} <= h.keys():
            out.append((size, imgsz // size, h))
    return out


# ----------------------------------------------------------------------
# 第一部分：原始張量的數值範圍 ★ 這段直接回答 sigmoid/softmax 做過沒有
# ----------------------------------------------------------------------

def report_ranges(grouped) -> dict:
    print("=" * 70)
    print("原始張量數值範圍  ★ 這段直接回答 sigmoid / softmax 做過沒有")
    print("=" * 70)
    info: dict = {}
    for size, stride, h in grouped:
        print(f"\nstride {stride}  ({size}x{size})")
        for slot in ("box", "cls", "kpt"):
            name, t = h[slot]
            d = {"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
                 "min": float(t.min()), "max": float(t.max()),
                 "mean": float(t.mean())}
            print(f"  {slot:4s} {name:28s} {t.dtype}  "
                  f"min {d['min']:9.4f}  max {d['max']:9.4f}  "
                  f"mean {d['mean']:8.4f}")
            info[f"s{stride}_{slot}"] = d

        # cls：值域在 0~1 就是晶片上已經 sigmoid 過了
        _, cls = h["cls"]
        in01 = cls.min() >= -1e-6 and cls.max() <= 1.0 + 1e-6
        print(f"       cls 值域 {'在 0~1 內 → sigmoid 已在晶片上做過' if in01 else '超出 0~1 → 是 logit，需要 sigmoid'}")
        info[f"s{stride}_cls_in01"] = bool(in01)
        n_pass_raw = int((cls[:, :, 0] >= 0.25).sum())
        n_pass_sig = int((_sigmoid(cls[:, :, 0]) >= 0.25).sum())
        print(f"       過 0.25 門檻的 anchor：直接用 {n_pass_raw} / {size*size}"
              f"　先 sigmoid {n_pass_sig} / {size*size}")
        info[f"s{stride}_cls_pass"] = {"raw": n_pass_raw, "sigmoid": n_pass_sig,
                                       "total": size * size}

        # box：每 16 個 bin 加起來接近 1 就是 softmax 已經做過
        _, box = h["box"]
        g = box.reshape(size, size, 4, REG_MAX)
        sums = g.sum(axis=-1)
        near1 = bool(abs(sums.mean() - 1.0) < 0.05 and sums.std() < 0.05)
        print(f"       DFL 每 16 bin 總和 mean {sums.mean():.4f} "
              f"std {sums.std():.4f}  "
              f"{'≈1 → softmax 已做過' if near1 else '≠1 → 是 logit，需要 softmax'}")
        info[f"s{stride}_dfl_softmaxed"] = near1

        # kpt：看兩種排列下「第三個分量」的值域，可見度應該最像機率
        _, kpt = h["kpt"]
        km = kpt.reshape(size, size, NK, 3)[:, :, :, 2]          # kpt-major
        cm = kpt.reshape(size, size, 3, NK)[:, :, 2, :]          # comp-major
        print(f"       51ch 第三分量值域  kpt-major [{km.min():.3f}, {km.max():.3f}]"
              f"   comp-major [{cm.min():.3f}, {cm.max():.3f}]")
        info[f"s{stride}_kpt_third"] = {
            "kpt_major": [float(km.min()), float(km.max())],
            "comp_major": [float(cm.min()), float(cm.max())]}
    return info


# ----------------------------------------------------------------------
# 第二部分：8 種組合並排
# ----------------------------------------------------------------------

def decode(grouped, cls_sig: bool, dfl_soft: bool, kpt_major: bool,
           conf: float, iou: float):
    bins = np.arange(REG_MAX, dtype=np.float32)
    B, S, KX, KC = [], [], [], []
    for size, stride, h in grouped:
        cls = h["cls"][1][:, :, 0]
        scores = _sigmoid(cls) if cls_sig else cls
        rows, cols = np.nonzero(scores >= conf)
        if rows.size == 0:
            continue
        if rows.size > 20000:                 # 明顯壞掉的組合不要算到天亮
            rows, cols = rows[:20000], cols[:20000]

        reg = h["box"][1][rows, cols].reshape(-1, 4, REG_MAX)
        dist = ((_softmax(reg, -1) if dfl_soft else reg) * bins).sum(-1)
        ax = cols.astype(np.float32) + 0.5
        ay = rows.astype(np.float32) + 0.5
        B.append(np.stack([(ax - dist[:, 0]) * stride,
                           (ay - dist[:, 1]) * stride,
                           (ax + dist[:, 2]) * stride,
                           (ay + dist[:, 3]) * stride], axis=1))

        raw = h["kpt"][1][rows, cols]
        kp = (raw.reshape(-1, NK, 3) if kpt_major
              else raw.reshape(-1, 3, NK).transpose(0, 2, 1))
        KX.append(np.stack([(kp[:, :, 0] * 2.0 + cols[:, None]) * stride,
                            (kp[:, :, 1] * 2.0 + rows[:, None]) * stride],
                           axis=2))
        KC.append(_sigmoid(kp[:, :, 2]))
        S.append(scores[rows, cols])

    if not B:
        return []
    B, S = np.concatenate(B), np.concatenate(S)
    KX, KC = np.concatenate(KX), np.concatenate(KC)
    return [(float(max(0.0, B[i][2]-B[i][0]) * max(0.0, B[i][3]-B[i][1])),
             float(S[i]), B[i], KX[i], KC[i]) for i in _nms(B, S, iou)]


def main() -> None:
    ap = argparse.ArgumentParser(description="解碼診斷：8 種組合並排")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--hef", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out-dir", default="results/device")
    args = ap.parse_args()

    frame = load_frame(args)
    h_img, w_img = frame.shape[:2]
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["inference"]["backend"] = "hailo"
    if args.hef:
        cfg["inference"]["hef"] = args.hef

    backend = make_backend(cfg)
    try:
        s, pad_x, pad_y = backend._letterbox(frame)
        outputs = {k: np.array(v, copy=True) for k, v in
                   backend._forward().items()}
        imgsz = backend.imgsz
        iou = backend.iou
    finally:
        backend.close()

    print(f"輸入  {args.image or f'{args.video} 第 {args.frame} 格'}"
          f"  {w_img}x{h_img}")
    print(f"letterbox  scale {s}  pad ({pad_x}, {pad_y})"
          f"  影像區域 x {pad_x:.0f}~{imgsz-pad_x:.0f}"
          f"  y {pad_y:.0f}~{imgsz-pad_y:.0f}\n")

    grouped = group_heads(outputs, imgsz)
    ranges = report_ranges(grouped)

    print("\n" + "=" * 70)
    print("8 種組合並排  ★ 對的那一組：偵測數 1~3、分數高、關鍵點全在畫面內")
    print("=" * 70)
    print(f"\n{'cls':>6} {'DFL':>8} {'51ch':>11} {'偵測數':>7} {'最高分':>7} "
          f"{'最大框 wxh':>13} {'點在框內':>8} {'點在畫面內':>10}")
    print("-" * 78)

    rows_out = []
    for cls_sig, dfl_soft, kpt_major in itertools.product(
            (True, False), (True, False), (True, False)):
        try:
            ppl = decode(grouped, cls_sig, dfl_soft, kpt_major,
                         args.conf, iou)
        except Exception as e:
            print(f"{'sig' if cls_sig else 'raw':>6} "
                  f"{'softmax' if dfl_soft else 'raw':>8} "
                  f"{'kpt-major' if kpt_major else 'comp-major':>11}"
                  f"   例外 {type(e).__name__}")
            continue

        rec = {"cls_sigmoid": cls_sig, "dfl_softmax": dfl_soft,
               "kpt_major": kpt_major, "n_det": len(ppl)}
        if ppl:
            area, score, box, kxy, kcf = max(ppl, key=lambda p: p[0])
            top = max(p[1] for p in ppl)
            bw, bh = box[2] - box[0], box[3] - box[1]
            # 關鍵點落在自己那個框內的比例
            inbox = int(((kxy[:, 0] >= box[0]) & (kxy[:, 0] <= box[2]) &
                         (kxy[:, 1] >= box[1]) & (kxy[:, 1] <= box[3])).sum())
            # 關鍵點落在 letterbox 的影像區域內的比例（不是灰邊）
            inimg = int(((kxy[:, 0] >= pad_x) & (kxy[:, 0] <= imgsz - pad_x) &
                         (kxy[:, 1] >= pad_y) & (kxy[:, 1] <= imgsz - pad_y)
                         ).sum())
            rec.update({"top_score": round(top, 4),
                        "box": [round(float(v), 1) for v in box],
                        "box_wh": [round(float(bw), 1), round(float(bh), 1)],
                        "kpt_in_box": inbox, "kpt_in_image": inimg})
            flag = " ←" if 1 <= len(ppl) <= 3 and inimg >= 13 else ""
            print(f"{'sig' if cls_sig else 'raw':>6} "
                  f"{'softmax' if dfl_soft else 'raw':>8} "
                  f"{'kpt-major' if kpt_major else 'comp-major':>11} "
                  f"{len(ppl):>7} {top:>7.3f} "
                  f"{f'{bw:.0f}x{bh:.0f}':>13} "
                  f"{f'{inbox}/17':>8} {f'{inimg}/17':>10}{flag}")
        else:
            print(f"{'sig' if cls_sig else 'raw':>6} "
                  f"{'softmax' if dfl_soft else 'raw':>8} "
                  f"{'kpt-major' if kpt_major else 'comp-major':>11} "
                  f"{0:>7} {'—':>7} {'—':>13} {'—':>8} {'—':>10}")
        rows_out.append(rec)

    print("\n← 標記 = 偵測數 1~3 且至少 13 個關鍵點落在影像區域內（不是灰邊）")
    print("\n判讀：")
    print("  偵測數上千 → cls 那一欄選錯了")
    print("  框大小全部差不多 → DFL 那一欄選錯了")
    print("  框合理但關鍵點散開 → 51ch 那一欄選錯了")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag_decode.json").write_text(json.dumps(
        {"input": args.image or f"{args.video}#{args.frame}",
         "dims": [w_img, h_img], "letterbox": {"scale": s, "pad": [pad_x, pad_y]},
         "ranges": ranges, "variants": rows_out},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {out / 'diag_decode.json'}")


if __name__ == "__main__":
    main()
