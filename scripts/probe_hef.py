#!/usr/bin/env python3
"""
探測 HEF 的輸入／輸出格式 —— 寫 HailoBackend 之前必跑。

用法：
    python scripts/probe_hef.py --hef models/yolov8s_pose.hef

----------------------------------------------------------------------
為什麼需要這支
----------------------------------------------------------------------
yolov8-pose 的 HEF 有兩種編譯方式，後處理完全不同：

  A. 內建 NMS  → 輸出已經是解碼好的偵測結果（每人一組 bbox + 17 點）
  B. 原始張量  → 輸出是各 stride 的特徵圖，要自己做 NMS 與關鍵點解碼

光看檔名分不出來。猜錯的話後處理會整個寫錯，而且症狀是「有數字但全是垃圾」，
比直接報錯更難除錯。

這支把 HEF 的輸入／輸出串流名稱、形狀、資料格式全部印出來，
同時確認這套 HailoRT 提供哪一組 Python API（新版 InferModel 或舊版 InferVStreams）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def describe(obj, keys) -> dict:
    out = {}
    for k in keys:
        try:
            v = getattr(obj, k)
            v = v() if callable(v) else v
            out[k] = str(v)
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="探測 HEF 格式")
    ap.add_argument("--hef", required=True)
    ap.add_argument("--out", default="results/device/hef_probe.json")
    args = ap.parse_args()

    hef_path = Path(args.hef)
    if not hef_path.is_file():
        raise SystemExit(f"找不到 HEF：{hef_path}")

    import hailo_platform as hp

    print("=" * 64)
    print("HEF 探測")
    print("=" * 64)
    print(f"檔案        {hef_path}  ({hef_path.stat().st_size:,} bytes)")
    print(f"hailo_platform {getattr(hp, '__version__', '?')}")

    # ---- 可用的 API ----------------------------------------------------
    api = {name: hasattr(hp, name) for name in (
        "VDevice", "HEF", "FormatType", "HailoSchedulingAlgorithm",
        "ConfigureParams", "InferVStreams", "HailoStreamInterface",
        "InputVStreamParams", "OutputVStreamParams",
    )}
    print("\n可用 API：")
    for k, v in api.items():
        print(f"  {'✓' if v else '✗'} {k}")

    hef = hp.HEF(str(hef_path))
    payload: dict = {
        "hef": str(hef_path),
        "hailo_platform": str(getattr(hp, "__version__", "?")),
        "api_available": api,
    }

    # ---- 網路群組 ------------------------------------------------------
    try:
        names = hef.get_network_group_names()
        payload["network_groups"] = list(names)
        print(f"\n網路群組：{list(names)}")
    except Exception as e:
        print(f"\n網路群組：讀取失敗 {e}")

    FIELDS = ("name", "shape", "format", "nms_shape", "quant_info",
              "network_name", "direction")

    # ---- 輸入 ----------------------------------------------------------
    print("\n" + "-" * 64)
    print("輸入串流")
    print("-" * 64)
    ins = []
    try:
        for vi in hef.get_input_vstream_infos():
            d = describe(vi, FIELDS)
            ins.append(d)
            print(f"  {d.get('name')}")
            for k, v in d.items():
                if k != "name":
                    print(f"      {k:14s} {v}")
    except Exception as e:
        print(f"  讀取失敗：{e}")
    payload["inputs"] = ins

    # ---- 輸出 ★ 最重要 --------------------------------------------------
    print("\n" + "-" * 64)
    print("輸出串流  ★ 這段決定後處理怎麼寫")
    print("-" * 64)
    outs = []
    try:
        for vi in hef.get_output_vstream_infos():
            d = describe(vi, FIELDS)
            outs.append(d)
            print(f"  {d.get('name')}")
            for k, v in d.items():
                if k != "name":
                    print(f"      {k:14s} {v}")
    except Exception as e:
        print(f"  讀取失敗：{e}")
    payload["outputs"] = outs

    # ---- 判讀 ----------------------------------------------------------
    print("\n" + "=" * 64)
    n = len(outs)
    has_nms = any("nms" in json.dumps(o).lower() for o in outs)
    if has_nms or n == 1:
        print("判讀：看起來是 **內建 NMS**（輸出少、或形狀含 nms 資訊）")
        print("      後處理較單純：直接讀解碼好的偵測結果")
    else:
        print(f"判讀：看起來是 **原始張量**（{n} 個輸出）")
        print("      後處理要自己做 NMS 與關鍵點解碼")
        print("      → 參考 hailo-apps 的 yolov8_pose 範例，不要自己推導")
    print("=" * 64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n✓ {out_path}")
    print("\n把這個檔案的內容貼回來，就能把 HailoBackend 的後處理寫完。")


if __name__ == "__main__":
    main()
