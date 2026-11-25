#!/usr/bin/env python3
"""
UGen300 裝置環境確認 —— 把硬體狀態記錄成可引用的報告。

用法（在 repo 根目錄）：
    python scripts/check_device.py                      # 只檢查裝置與環境
    python scripts/check_device.py --hef path/to.hef    # 另外單獨跑一次 HEF

輸出：results/device/report.md 與 report.json

----------------------------------------------------------------------
為什麼要有這支
----------------------------------------------------------------------
三個目的：

1. **把「HEF 能不能在卡上跑」和「我的後處理對不對」分開。**
   hailortcli 跑 HEF 用的是合成輸入，完全不碰我們的程式。先過這關，
   之後出問題就一定在 Python 那層。

2. **記錄 PCIe 實際協商到的連線寬度。** 規格是 Gen3 x4，但插槽共用、
   BIOS 設定或轉接都可能讓它掉到 x2 甚至 x1。不先確認，第三週會量出
   一組低於規格的數字卻找不到原因。

3. **產生簡報用的硬體整合證據。** 韌體版本、HailoRT 版本、裝置序號、
   連線寬度、單跑 HEF 的 FPS 與延遲 —— 這些是「我們真的把它跑起來了」
   的憑證，不是截圖。

HailoRT 的 Vision API（HEF，姿態走這條）與 GenAI API（LLM 走這條）
是不同路徑。LLM 跑得動只證明驅動與韌體沒問題，不代表 HEF 能跑。
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path("results/device")
TIMEOUT = 120


def run(cmd: list[str], timeout: int = TIMEOUT) -> dict:
    """跑一道指令，不讓失敗中斷整份檢查。"""
    if shutil.which(cmd[0]) is None:
        return {"cmd": " ".join(cmd), "status": "not_found",
                "note": f"找不到 {cmd[0]}"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return {"cmd": " ".join(cmd),
                "status": "ok" if p.returncode == 0 else "failed",
                "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "status": "timeout"}
    except Exception as e:
        return {"cmd": " ".join(cmd), "status": "error", "note": str(e)}


def host_info() -> dict:
    def ver(mod):
        try:
            return str(getattr(__import__(mod), "__version__", "?"))
        except Exception:
            return "not installed"

    ram = None
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        try:
            import os
            ram = round(os.sysconf("SC_PAGE_SIZE")
                        * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
        except Exception:
            pass
    return {
        "os": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "ram_gb": ram, "python": platform.python_version(),
        "numpy": ver("numpy"), "opencv": ver("cv2"),
        "hailo_platform": ver("hailo_platform"),
    }


def pcie_link() -> dict:
    """抓 PCIe 實際協商到的速度與寬度（Linux）。"""
    if platform.system() != "Linux":
        return {"status": "skipped",
                "note": "非 Linux。Windows 請在裝置管理員查看，"
                        "或用 HWiNFO 之類工具確認連線寬度"}
    r = run(["lspci", "-vvnn"])
    if r.get("status") != "ok":
        return {"status": r.get("status"), "note": r.get("note", "")}

    blocks = re.split(r"\n(?=\S)", r["stdout"])
    for b in blocks:
        if "hailo" not in b.lower():
            continue
        cap = re.search(r"LnkCap:.*?Speed\s+([\d.]+GT/s).*?Width\s+(x\d+)",
                        b, re.S)
        sta = re.search(r"LnkSta:.*?Speed\s+([\d.]+GT/s).*?Width\s+(x\d+)",
                        b, re.S)
        out = {"status": "found",
               "device_line": b.splitlines()[0].strip(),
               "cap_speed": cap.group(1) if cap else None,
               "cap_width": cap.group(2) if cap else None,
               "sta_speed": sta.group(1) if sta else None,
               "sta_width": sta.group(2) if sta else None}
        if out["sta_width"] and out["cap_width"]:
            out["degraded"] = out["sta_width"] != out["cap_width"]
        return out
    return {"status": "not_found",
            "note": "lspci 看不到 Hailo 裝置。卡沒插好、BIOS 關掉該槽，"
                    "或那個 M.2 槽是 SATA-only"}


def main() -> None:
    ap = argparse.ArgumentParser(description="UGen300 裝置環境確認")
    ap.add_argument("--hef", help="另外單獨跑這個 HEF（不經過我們的程式）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UGen300 裝置環境確認")
    print("=" * 60, flush=True)

    checks: dict[str, dict] = {}
    for name, cmd in [
        ("hailortcli_version", ["hailortcli", "--version"]),
        ("fw_identify", ["hailortcli", "fw-control", "identify"]),
        ("scan", ["hailortcli", "scan"]),
        ("measure_power", ["hailortcli", "measure-power"]),
    ]:
        print(f"\n[{name}] …", flush=True)
        checks[name] = run(cmd, timeout=30 if name == "measure_power" else TIMEOUT)
        s = checks[name]
        if s.get("status") == "ok":
            first = (s["stdout"].splitlines() or [""])[0][:100]
            print(f"  ✓ {first}")
        else:
            print(f"  ✗ {s.get('status')} {s.get('note','')}")

    print("\n[pcie_link] …", flush=True)
    link = pcie_link()
    checks["pcie_link"] = link
    if link.get("status") == "found":
        print(f"  能力 {link['cap_speed']} {link['cap_width']}"
              f"  /  實際 {link['sta_speed']} {link['sta_width']}")
        if link.get("degraded"):
            print("  ⚠️ 實際連線寬度低於能力值 —— 吞吐會低於規格，量測前要先處理")
    else:
        print(f"  {link.get('status')}: {link.get('note','')}")

    if args.hef:
        hef = Path(args.hef)
        if not hef.is_file():
            print(f"\n✗ 找不到 HEF：{hef}")
        else:
            print(f"\n[hef_run] 單獨跑 {hef.name}（合成輸入，不經過我們的程式）…",
                  flush=True)
            checks["hef_run"] = run(["hailortcli", "run", str(hef)], timeout=300)
            checks["hef_latency"] = run(
                ["hailortcli", "run", str(hef), "--measure-latency"], timeout=300)
            for k in ("hef_run", "hef_latency"):
                r = checks[k]
                print(f"  [{k}] {r.get('status')}")
                if r.get("status") == "ok":
                    for l in r["stdout"].splitlines():
                        if re.search(r"fps|latency|FPS|Latency", l):
                            print(f"    {l.strip()}")

    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "host": host_info(), "hef": args.hef, "checks": checks}
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    h = payload["host"]
    L: list[str] = ["# UGen300 裝置環境報告", "",
                    f"> 產生時間：{payload['generated_at']}", ""]
    L += ["## 主機", "", "| 項目 | 值 |", "|---|---|",
          f"| 作業系統 | {h['os']} |", f"| 處理器 | {h['processor']} |",
          f"| 記憶體 | {h['ram_gb']} GB |", f"| Python | {h['python']} |",
          f"| hailo_platform | {h['hailo_platform']} |",
          f"| numpy / OpenCV | {h['numpy']} / {h['opencv']} |", ""]

    L += ["## PCIe 連線", ""]
    if link.get("status") == "found":
        L += ["| | 速度 | 寬度 |", "|---|---|---|",
              f"| 能力 (LnkCap) | {link['cap_speed']} | {link['cap_width']} |",
              f"| **實際 (LnkSta)** | **{link['sta_speed']}** | **{link['sta_width']}** |",
              ""]
        L.append("⚠️ **實際連線寬度低於能力值**，吞吐會低於規格。"
                 "換個 M.2 槽或檢查 BIOS 的通道配置。" if link.get("degraded")
                 else "✅ 以完整寬度協商，符合 PCIe 3.0 x4 規格。")
    else:
        L.append(f"`{link.get('status')}` — {link.get('note','')}")
    L.append("")

    L += ["## 裝置檢查", ""]
    for name, r in checks.items():
        if name == "pcie_link":
            continue
        L += [f"### `{r.get('cmd', name)}`", "",
              f"狀態：**{r.get('status')}**", ""]
        if r.get("stdout"):
            L += ["```", r["stdout"][:4000], "```", ""]
        elif r.get("note"):
            L += [r["note"], ""]

    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"✓ {out_dir / 'report.md'}")
    print(f"✓ {out_dir / 'report.json'}")
    ok = sum(1 for k, v in checks.items() if v.get("status") in ("ok", "found"))
    print(f"\n{ok} / {len(checks)} 項檢查通過")


if __name__ == "__main__":
    main()
