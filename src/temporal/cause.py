"""
不可信的「原因」分類 —— 把 `UNRELIABLE` 變成 `UNRELIABLE（離床）`。

現在系統說得出「我看不到」，說不出「為什麼看不到」。
護理師收到「7 床 不可信」沒有動作可做；收到「7 床 手臂被蓋住」就有。

----------------------------------------------------------------------
為什麼獨立一個模組，不寫進 fusion.py
----------------------------------------------------------------------
三個理由，第二個最重要：

1. `ConsensusFusion` 刻意無狀態、而且刻意只認識「特徵名 → 數值」的字典
   （它自己的註解寫了「保持無狀態讓它好測、好解釋」）。
   餵它亮度和人數會破壞那個契約。

2. ★ **原因分類不可以改變 UNRELIABLE 的判定。**
   放在融合層外面是結構上的保證，不是口頭承諾 ——
   這個模組拿不到 reliability 的決策權，想改也改不了。
   既有的等價驗證與 Stage I 數字因此完全不受影響。

3. 它是純函式，可以用合成輸入測完，不需要影片也不需要硬體。

`FusionResult` 已經有 `reason` 欄位，但那講的是**機制**
（「可信度 0.22 低於 0.35（缺 2 個訊號）」）；這裡講的是**原因**。
兩個並存，不互相取代。

----------------------------------------------------------------------
刻意只吃類別化的輸入
----------------------------------------------------------------------
人數用 0 / 1 / ≥2 三個桶，不用精確值。

因為 `UltralyticsBackend` 與 `HailoBackend` 的 NMS 不是同一套，
邊緣畫面上偵測到的人數可能差一個。分類結果如果對精確人數敏感，
兩個後端就會給出不同的原因 —— 那會變成另一個等價驗證問題。
"""

from __future__ import annotations

from typing import Optional

# 原因代碼。扁平字串，可以直接進 payload（validate_payload 擋巢狀結構）。
NONE = ""                                  # 沒有問題，或沒有被問
UNKNOWN = "unknown"                        # metadata 不足，無法判斷
NO_PERSON = "no_person"                    # 畫面上沒有人
PERSON_NOT_PATIENT = "person_not_patient"  # 有人但沒人符合病人判別
LOW_LIGHT = "low_light"                    # 畫面太暗
OTHER_PERSON = "other_person_present"      # 有選到病人，但畫面上還有別人
OCCLUDED = "occluded_unclassified"         # 以上皆非 —— 被子、肢體出框之類

LABEL_ZH = {
    NONE: "",
    UNKNOWN: "原因未知",
    NO_PERSON: "離床",
    PERSON_NOT_PATIENT: "床上無人",
    LOW_LIGHT: "光線不足",
    OTHER_PERSON: "他人遮擋",
    OCCLUDED: "遮蔽·未分類",
}

# 「暗」的門檻（0–255 的平均亮度）。
# ⚠️ 這個數字是**暫定值，尚未標定**。要用 clip4（關燈版）實測之後才算有依據。
# 標定之前 low_light 這一類的數字不要引用進簡報。
DEFAULT_DARK_LUMA = 40.0

# 需要哪些 meta 欄位才判斷得出來。少了就回 UNKNOWN，不要猜。
REQUIRED_KEYS = ("n_detected", "n_passed")


def frame_luma(frame, stride: int = 8) -> Optional[float]:
    """影格的平均亮度（0–255）。

    每 stride 個像素取一個，成本約為全圖的 1/stride² ——
    主機端已經是瓶頸（約 82% 的時間在 Python），逐格全圖平均付不起。
    """
    if frame is None:
        return None
    try:
        sub = frame[::stride, ::stride]
        return float(sub.mean())
    except Exception:
        return None


def classify(meta: Optional[dict], luma: Optional[float] = None,
             dark_luma: float = DEFAULT_DARK_LUMA) -> str:
    """判斷這一格「為什麼看不清楚」。

    只在 reliability 不是 ok 的時候呼叫才有意義 ——
    這個函式不知道也不該知道 reliability 是什麼。

    順序是刻意的，第一個命中就回傳：

      1. no_person            —— 最確定的一種：根本沒有人可看
      2. person_not_patient   —— 有人，但沒人符合病人判別（不在床位或非躺姿）
      3. low_light            —— 整張畫面的性質，而且它會「解釋」關鍵點為何不穩，
                                 所以排在遮蔽之前
      4. other_person_present —— 有選到病人，但畫面上還有別人（彎腰的護理師會遮）
      5. occluded             —— 兜底。★ CLIP 要接手的就是這一類
    """
    if not meta or any(meta.get(k) is None for k in REQUIRED_KEYS):
        # replay 舊的 keypoint dump、或 subject 判別沒開 → 資訊不足。
        # 回 UNKNOWN 而不是 NO_PERSON：沒有資料不等於沒有人。
        return UNKNOWN

    n_detected = int(meta["n_detected"])
    n_passed = int(meta["n_passed"])
    found = bool(meta.get("subject_found", n_passed > 0))

    if n_detected <= 0:
        return NO_PERSON
    if not found or n_passed <= 0:
        return PERSON_NOT_PATIENT
    if luma is not None and luma < dark_luma:
        return LOW_LIGHT
    if n_detected >= 2:
        return OTHER_PERSON
    return OCCLUDED


def label(cause: str) -> str:
    """給 UI 用的中文標籤。未知代碼原樣回傳，不要吞掉。"""
    return LABEL_ZH.get(cause, cause)


def histogram(causes) -> dict[str, int]:
    """原因直方圖。

    ★ 這是整個模組存在的主要目的：量出 occluded_unclassified 佔多少。

      九成落在前四類  → CLIP 不值得做，而且這個數字就是「已評估」的證據
      一半落在未分類  → CLIP 的價值有數據支撐，不是臆測

    用量的決定要不要做下一步，而不是先做了再說。
    """
    out: dict[str, int] = {}
    for c in causes:
        if c:
            out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
