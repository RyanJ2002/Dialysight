#!/usr/bin/env python3
"""
床位 ROI 標定工具。

★ 拍完正式素材後跑一次，把輸出貼進 config/default.yaml 的 subject.roi。

⚠️ 攝影機一動就要重標。這是「相機固定不可妥協」的另一個理由。

用法：
    python scripts/calibrate_roi.py --video data/clips/clip_01_baseline.mp4

操作：
    滑鼠拖曳框出床面範圍 → 按 Enter 確認 → 按 q 離開
    按 r 重新框選
    按 n / p 換下一格 / 上一格（找一格病人躺好的畫面）

框選原則：
    框住「病人躺著時髖部可能出現的範圍」，不是整張床。
    寧可框小一點 —— 護理師站在床邊時髖部不該落進來。
"""

from __future__ import annotations

import argparse

import cv2

WINDOW = "Calibrate Bed ROI  |  drag=select  enter=confirm  r=reset  n/p=frame  q=quit"


class ROISelector:
    def __init__(self) -> None:
        self.start: tuple[int, int] | None = None
        self.end: tuple[int, int] | None = None
        self.dragging = False

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.dragging = False

    @property
    def box(self) -> tuple[int, int, int, int] | None:
        if self.start is None or self.end is None:
            return None
        x0, y0 = self.start
        x1, y1 = self.end
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return None
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def reset(self) -> None:
        self.start = self.end = None
        self.dragging = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0, help="從第幾格開始")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, min(args.frame, total - 1))

    sel = ROISelector()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, sel.on_mouse)

    def read_at(i: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read_at(idx)
    if frame is None:
        raise SystemExit("讀不到影格")

    h, w = frame.shape[:2]
    print(f"影片尺寸 {w}x{h}，共 {total} 格")

    while True:
        canvas = frame.copy()
        box = sel.box
        if box:
            x0, y0, x1, y1 = box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 120), 2)
            txt = (f"x_min:{x0/w:.3f}  y_min:{y0/h:.3f}  "
                   f"x_max:{x1/w:.3f}  y_max:{y1/h:.3f}")
            cv2.rectangle(canvas, (8, 8), (8 + 9 * len(txt), 40), (0, 0, 0), -1)
            cv2.putText(canvas, txt, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"frame {idx}/{total - 1}", (14, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            sel.reset()
        if key in (ord("n"), ord("p")):
            idx = min(total - 1, idx + 30) if key == ord("n") else max(0, idx - 30)
            nf = read_at(idx)
            if nf is not None:
                frame = nf
        if key in (13, 10) and box:          # Enter
            x0, y0, x1, y1 = box
            print("\n把以下內容貼進 config/default.yaml 的 subject 區塊：\n")
            print("subject:")
            print("  roi:")
            print(f"    x_min: {x0 / w:.3f}")
            print(f"    y_min: {y0 / h:.3f}")
            print(f"    x_max: {x1 / w:.3f}")
            print(f"    y_max: {y1 / h:.3f}")
            print()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
