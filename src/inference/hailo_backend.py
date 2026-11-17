"""
Hailo-10H 推論後端（UGen300 M.2 8GB）。

實測環境（2026-11-17，MSI MS-7E02，Windows 11）：
    HailoRT 5.3.2 / Firmware 5.3.2 / HAILO10H / pci 0000:01:00.0
    yolov8s_pose.hef  13,799,424 bytes  輸入 640x640x3
    冷機 161.93 FPS @ 55.45 °C
    封閉機殼連續 300 秒 149.22 FPS @ 85.35 °C（THROTTLING_LIGHT）
    散熱：Hailo 裝 M2_1（CPU 直連）+ 主機板散熱片。
          裸卡插 M2_2 會到 115 °C、連續吞吐剩 55.87 FPS。

★ 這是整個硬體遷移唯一要新增的檔案。時序層一行都不用改。

----------------------------------------------------------------------
HEF 輸出格式（scripts/probe_hef.py 實測）
----------------------------------------------------------------------
原始張量，標準 YOLOv8-pose 三尺度頭，9 個輸出：

    stride 8   80x80 : conv43(64) conv44(1) conv45(51)
    stride 16  40x40 : conv57(64) conv58(1) conv59(51)
    stride 32  20x20 : conv70(64) conv71(1) conv72(51)

    64 ch = 4 邊 x 16 個 DFL bin（channel = side*16 + bin，順序 l,t,r,b）
     1 ch = 單類別（person）分數
    51 ch = 17 點 x 3（channel = kpt*3 + component）

本檔不依 conv43 這類名稱對應，改用「空間尺寸 + 通道數」分組，
換 HEF 版本或重新編譯後層名變了也不會壞。

----------------------------------------------------------------------
★ cls 分支的 sigmoid 已經折進 HEF 了（scripts/diag_decode.py 實測）
----------------------------------------------------------------------
Hailo 的 device_pre_post_layers 可以把 sigmoid 編譯進 HEF，在晶片上算完。
這個 HEF 就是這樣，所以 cls 輸出已經是 0~1 的機率，不是 logit：

    cls  conv44/58/71   min 0.0000   max 0.9137      ← 已是機率
    DFL  conv43/57/70   每 16 bin 總和 15.99 ± 0.09  ← 仍是 logit，要 softmax
    kpt  conv45/59/72   min -6.65  max 8.44          ← 仍是 logit，可見度要 sigmoid

折了沒有從 HEF 檔案本身看不出來，而猜錯的症狀很難認：再 sigmoid 一次會把
0~1 映到 0.5~0.73，於是**每個 anchor 都過門檻**。實測一格解出 2696 個人
（總 anchor 數 8400），然後 max(area) 從裡面挑了一個隨機的 anchor，
關鍵點偏移量解得對卻掛在錯的格子上 —— 骨架大致是人形，位置整團跑掉。

所以不寫死「不要 sigmoid」，改成開場量一次（_detect_format）。
授權問題若逼我們換成 centerpose_regnetx_800mf，新 HEF 未必也折了 sigmoid，
寫死會再踩一次同一個坑。要強制可用 config：
    inference:
      cls_activation: auto     # auto | sigmoid | none
      dfl_activation: auto     # auto | softmax | none

----------------------------------------------------------------------
三個會讓結果悄悄錯掉的地方
----------------------------------------------------------------------
1. **letterbox，不是 resize** —— ultralytics 用等比縮放加灰邊(114)。
   直接 resize 會拉伸長寬比，關鍵點與 CPU 端對不上，而且不會報錯。
2. **座標要還原回原始影格再正規化** —— 模型吐 640x640 座標，
   要扣 padding、除縮放比回原圖像素，最後除以原圖寬高。
3. **多人取面積最大的 bbox** —— 必須與 UltralyticsBackend 同規則，
   否則護理師入鏡時兩邊會選到不同的人，等價驗證就失去意義。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import NUM_KEYPOINTS, FrameObservation, InferenceBackend, Keypoint

PAD_VALUE = 114          # 與 ultralytics letterbox 相同
REG_MAX = 16             # DFL bin 數
IOU_THRESHOLD = 0.7      # 與 ultralytics predict 預設一致


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """單類別貪婪 NMS。boxes 為 xyxy。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= thr]
    return keep


class HailoBackend(InferenceBackend):
    """把 UGen300 的 HEF 輸出轉成 FrameObservation。"""

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, cfg: dict):
        try:
            import hailo_platform as hp
        except ImportError as e:
            raise ImportError(
                "找不到 hailo_platform。它不在 PyPI 上，要從 HailoRT 安裝目錄裝：\n"
                '  pip install "C:\\Program Files\\HailoRT\\python\\'
                'hailort-5.3.2-cp311-cp311-win_amd64.whl"\n'
                "（wheel 檔名是 hailort-*，但 import 名稱是 hailo_platform）"
            ) from e

        ic = cfg["inference"]
        hef_path = Path(ic.get("hef", "models/yolov8s_pose.hef"))
        if not hef_path.is_file():
            raise FileNotFoundError(
                f"找不到 HEF：{hef_path}\n"
                "從 Hailo Model Zoo 下載 HAILO10H 版本（不是 HAILO8）：\n"
                "  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
                "ModelZoo/Compiled/v5.5.0/hailo10h/yolov8s_pose.hef"
            )

        self.hp = hp
        self.imgsz = int(ic.get("imgsz", 640))
        self.conf = float(ic.get("conf", 0.25))
        self.iou = float(ic.get("iou", IOU_THRESHOLD))
        self._hef_path = hef_path
        self._closed = False

        # 輸出形態。auto 時在第一格量出來（見 _detect_format），之後沿用。
        self._cls_mode = str(ic.get("cls_activation", "auto")).lower()
        self._dfl_mode = str(ic.get("dfl_activation", "auto")).lower()
        self._cls_needs_sigmoid: bool | None = (
            None if self._cls_mode == "auto" else self._cls_mode == "sigmoid")
        self._dfl_needs_softmax: bool | None = (
            None if self._dfl_mode == "auto" else self._dfl_mode == "softmax")

        params = hp.VDevice.create_params()
        if hasattr(hp, "HailoSchedulingAlgorithm"):
            params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = hp.VDevice(params)

        # 輸入緩衝區配置一次就重複用：避免每格重新配置，
        # 也避免暫時陣列被回收後 set_buffer 指到無效記憶體。
        self._in_buffer = np.empty((self.imgsz, self.imgsz, 3), dtype=np.uint8)

        if hasattr(self._vdevice, "create_infer_model"):
            self._setup_infer_model(hp, hef_path)
        else:
            self._setup_vstreams(hp, hef_path)

    def _setup_infer_model(self, hp, hef_path: Path) -> None:
        """HailoRT 4.17+ 的 InferModel API（本機 5.3.2 走這條）。"""
        self._mode = "infer_model"
        self._infer_model = self._vdevice.create_infer_model(str(hef_path))
        self._infer_model.set_batch_size(1)

        # 輸出要 float32，讓 HailoRT 幫忙反量化；輸入維持 uint8
        for o in self._infer_model.outputs:
            try:
                o.set_format_type(hp.FormatType.FLOAT32)
            except Exception:
                pass
        try:
            self._infer_model.input().set_format_type(hp.FormatType.UINT8)
        except Exception:
            pass

        self._configured = self._infer_model.configure()
        self._bindings = self._configured.create_bindings()
        self._out_buffers = {
            o.name: np.empty(o.shape, dtype=np.float32)
            for o in self._infer_model.outputs
        }

        # 緩衝區綁定一次，之後每格只改內容
        self._bindings.input().set_buffer(self._in_buffer)
        for name, buf in self._out_buffers.items():
            self._bindings.output(name).set_buffer(buf)

        self._run_variant: str | None = None

    def _setup_vstreams(self, hp, hef_path: Path) -> None:
        """舊版 InferVStreams API。保留這條路是為了換主機時不會當場開不起來。"""
        self._mode = "vstreams"
        hef = hp.HEF(str(hef_path))
        cfg_params = hp.ConfigureParams.create_from_hef(
            hef, interface=hp.HailoStreamInterface.PCIe)
        self._network_group = self._vdevice.configure(hef, cfg_params)[0]

        in_params = hp.InputVStreamParams.make(
            self._network_group, format_type=hp.FormatType.UINT8)
        out_params = hp.OutputVStreamParams.make(
            self._network_group, format_type=hp.FormatType.FLOAT32)

        self._input_name = hef.get_input_vstream_infos()[0].name
        # 啟動狀態與 pipeline 必須跨呼叫存活，所以手動進入 context
        self._activation = self._network_group.activate(
            self._network_group.create_params())
        self._activation.__enter__()
        self._pipeline = hp.InferVStreams(
            self._network_group, in_params, out_params)
        self._pipeline.__enter__()
        self._out_buffers = {}

    # ------------------------------------------------------------------
    # 前處理
    # ------------------------------------------------------------------

    def _letterbox(self, frame: np.ndarray):
        """等比縮放 + 置中補邊，寫進持久緩衝區。回傳 (scale, pad_x, pad_y)。"""
        import cv2

        h, w = frame.shape[:2]
        s = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        left = int(round((self.imgsz - nw) / 2.0 - 0.1))
        top = int(round((self.imgsz - nh) / 2.0 - 0.1))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self._in_buffer[:] = PAD_VALUE
        # BGR→RGB：HEF 是用 ultralytics 的前處理編譯的
        self._in_buffer[top:top + nh, left:left + nw] = resized[:, :, ::-1]
        return s, float(left), float(top)

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def _forward(self) -> dict[str, np.ndarray]:
        """送一格進卡裡，回傳 {輸出名: 張量}。"""
        if self._mode == "vstreams":
            res = self._pipeline.infer({self._input_name: self._in_buffer[None]})
            return {k: (np.squeeze(v, axis=0) if v.ndim == 4 else v)
                    for k, v in res.items()}

        # HailoRT 版本之間 run() 的 timeout 參數型別不一致（ms int / timedelta），
        # 第一次呼叫試出來之後就記住，不要每格都 try/except。
        if self._run_variant is None:
            import datetime as _dt
            for variant, arg in (("ms", 10_000),
                                 ("timedelta", _dt.timedelta(seconds=10)),
                                 ("none", None)):
                try:
                    if arg is None:
                        self._configured.run([self._bindings])
                    else:
                        self._configured.run([self._bindings], arg)
                    self._run_variant = variant
                    break
                except TypeError:
                    continue
            if self._run_variant is None:
                raise RuntimeError(
                    "ConfiguredInferModel.run() 的參數型別試不出來。"
                    "印 help(self._configured.run) 看這版的簽名。")
        elif self._run_variant == "ms":
            self._configured.run([self._bindings], 10_000)
        elif self._run_variant == "timedelta":
            import datetime as _dt
            self._configured.run([self._bindings], _dt.timedelta(seconds=10))
        else:
            self._configured.run([self._bindings])

        return self._out_buffers

    # ------------------------------------------------------------------
    # 後處理
    # ------------------------------------------------------------------

    def _group_heads(self, outputs: dict[str, np.ndarray]):
        """依「空間尺寸 + 通道數」把 9 個輸出分成三個尺度的 (box, cls, kpt, stride)。"""
        heads: dict[int, dict[str, np.ndarray]] = {}
        slot_of = {4 * REG_MAX: "box", 1: "cls", NUM_KEYPOINTS * 3: "kpt"}
        for t in outputs.values():
            # 只剝掉前置的 batch 維。不能用 np.squeeze()：
            # cls 頭是 (H,W,1)，整體壓縮會把通道維一起吃掉變 (H,W)，
            # 然後分類分支就整個消失，而且不會報錯。
            while t.ndim > 3 and t.shape[0] == 1:
                t = t[0]
            if t.ndim != 3:
                continue
            slot = slot_of.get(t.shape[2])
            if slot is not None:
                heads.setdefault(t.shape[0], {})[slot] = t

        out = []
        for size in sorted(heads, reverse=True):          # 80, 40, 20
            h = heads[size]
            if {"box", "cls", "kpt"} <= h.keys():
                out.append((h["box"], h["cls"], h["kpt"], self.imgsz // size))
        if not out:
            raise RuntimeError(
                "無法辨識 HEF 輸出格式："
                f"{[(k, tuple(v.shape)) for k, v in outputs.items()]}\n"
                "跑 python scripts/probe_hef.py --hef <你的 hef> 重新確認。")
        return out

    def _detect_format(self, heads) -> None:
        """量一次輸出的實際形態。只在第一格做，結果沿用。

        判斷依據是數值範圍，不是 HEF 的 metadata —— metadata 看不出
        device_pre_post_layers 折了什麼進去。

          cls 全部落在 [0,1] 且沒有負值  → 已經是機率，不要再 sigmoid
            （logit 的背景值會是很負的數，不可能整張圖都非負）
          DFL 每 16 個 bin 總和 ≈ 1      → 已經 softmax 過

        猜錯不會報錯，只會安靜地給出垃圾，所以這裡把判斷結果印出來。
        """
        if self._cls_needs_sigmoid is None:
            lo = min(float(c.min()) for _, c, _, _ in heads)
            hi = max(float(c.max()) for _, c, _, _ in heads)
            self._cls_needs_sigmoid = not (lo >= -1e-6 and hi <= 1.0 + 1e-6)
            print(f"[hailo] cls 值域 [{lo:.4f}, {hi:.4f}] → "
                  + ("是 logit，套用 sigmoid" if self._cls_needs_sigmoid
                     else "已是機率（sigmoid 折在 HEF 裡），不再套用"),
                  flush=True)

        if self._dfl_needs_softmax is None:
            sums = [float(b.reshape(b.shape[0], b.shape[1], 4, REG_MAX)
                          .sum(axis=-1).mean()) for b, _, _, _ in heads]
            m = sum(sums) / len(sums)
            self._dfl_needs_softmax = abs(m - 1.0) > 0.05
            print(f"[hailo] DFL 每 16 bin 總和 {m:.4f} → "
                  + ("是 logit，套用 softmax" if self._dfl_needs_softmax
                     else "已 softmax 過，不再套用"), flush=True)

    def _decode(self, outputs: dict[str, np.ndarray], conf_thr: float):
        """解成 [(bbox_area, kp_xy(17,2), kp_conf(17,))]，座標在 640x640 模型座標系。"""
        boxes_all, scores_all, kxy_all, kcf_all = [], [], [], []
        bins = np.arange(REG_MAX, dtype=np.float32)

        heads = self._group_heads(outputs)
        if self._cls_needs_sigmoid is None or self._dfl_needs_softmax is None:
            self._detect_format(heads)

        for box_t, cls_t, kpt_t, stride in heads:
            scores = (_sigmoid(cls_t[:, :, 0]) if self._cls_needs_sigmoid
                      else cls_t[:, :, 0])
            rows, cols = np.nonzero(scores >= conf_thr)
            if rows.size == 0:
                continue

            # --- bbox：DFL 期望值 → 四邊距離（grid 單位，順序 l,t,r,b）---
            reg = box_t[rows, cols].reshape(-1, 4, REG_MAX)
            dist = ((_softmax(reg, -1) if self._dfl_needs_softmax else reg)
                    * bins).sum(-1)
            ax = cols.astype(np.float32) + 0.5          # anchor 在格中心
            ay = rows.astype(np.float32) + 0.5
            boxes = np.stack([
                (ax - dist[:, 0]) * stride,
                (ay - dist[:, 1]) * stride,
                (ax + dist[:, 2]) * stride,
                (ay + dist[:, 3]) * stride,
            ], axis=1)

            # --- 關鍵點：(raw*2 + (anchor-0.5)) * stride，anchor-0.5 即格索引 ---
            kp = kpt_t[rows, cols].reshape(-1, NUM_KEYPOINTS, 3)
            kxy = np.stack([
                (kp[:, :, 0] * 2.0 + cols[:, None]) * stride,
                (kp[:, :, 1] * 2.0 + rows[:, None]) * stride,
            ], axis=2)

            boxes_all.append(boxes)
            scores_all.append(scores[rows, cols])
            kxy_all.append(kxy)
            kcf_all.append(_sigmoid(kp[:, :, 2]))

        if not boxes_all:
            return []

        boxes = np.concatenate(boxes_all)
        scores = np.concatenate(scores_all)
        kxy = np.concatenate(kxy_all)
        kcf = np.concatenate(kcf_all)

        people = []
        for i in _nms(boxes, scores, self.iou):
            x1, y1, x2, y2 = boxes[i]
            area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
            people.append((area, kxy[i], kcf[i]))
        return people

    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray, frame_idx: int,
              timestamp: float) -> FrameObservation:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        s, pad_x, pad_y = self._letterbox(frame)
        outputs = self._forward()
        people = self._decode(outputs, self.conf)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        keypoints = [Keypoint(0.0, 0.0, 0.0) for _ in range(NUM_KEYPOINTS)]
        if people:
            _, kp_xy, kp_conf = max(people, key=lambda p: p[0])   # 面積最大
            for i in range(NUM_KEYPOINTS):
                c = float(kp_conf[i])
                if c <= 0.0:
                    continue                      # 缺失點留 confidence=0.0
                keypoints[i] = Keypoint(
                    x=(float(kp_xy[i][0]) - pad_x) / s / w,       # → 原圖 → 0~1
                    y=(float(kp_xy[i][1]) - pad_y) / s / h,
                    confidence=c,
                )

        return FrameObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=keypoints,
            raw_frame=frame,
            meta={"backend": "hailo", "infer_ms": round(elapsed_ms, 2),
                  "hef": self._hef_path.name, "people": len(people),
                  "cls_sigmoid": self._cls_needs_sigmoid,
                  "dfl_softmax": self._dfl_needs_softmax},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for attr in ("_pipeline", "_activation"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.__exit__(None, None, None)
                except Exception:
                    pass
                setattr(self, attr, None)
        for attr in ("_configured", "_vdevice"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            for m in ("shutdown", "release", "close"):
                f = getattr(obj, m, None)
                if f is None:
                    continue
                try:
                    f()
                    break
                except Exception:
                    continue
            setattr(self, attr, None)
