"""
融合層（第 5 層）—— 把多個訊號併成一個風險分數，而且誠實地說出可信度。

★ 這一層解決的問題：單一特徵都有各自的盲點。

      arm_displacement    被子蓋住手臂 → 關鍵點不見 → 瞎了
      torso_angle         病人側躺     → 肩髖線重疊 → 數值不穩
      roi_redness         前臂被遮     → ROI 無效   → 瞎了

  任何一個單獨拿來告警都會在某些情境下失效。
  但它們不會「同時」失效 —— 被子蓋住手臂時軀幹角度還在，
  側躺時手臂位移還在。所以共識投票比單一訊號穩。

★ 但共識投票有一個很容易做錯的地方，而這正是我們的設計重點：

  天真的做法是「可信的訊號投票，不可信的忽略」。
  問題在於，當所有訊號都不可信時，這個做法會輸出
  「沒有人投異常票」，也就是 —— **正常**。

  那是錯的。遮蔽嚴重時系統的狀態不是「正常」，是「不知道」。
  把「不知道」顯示成「正常」，是監測系統最危險的失效模式：
  它讓護理師以為有人在看，其實沒有。

  所以我們輸出兩個數字，而不是一個：

      risk          0–1，加權共識分數
      confidence    0–1，這個分數本身有多可信

  confidence 低於門檻時，狀態是 UNRELIABLE，護理站顯示
  「遮蔽中 · 無法判定」而不是綠燈。寧可讓人知道我們看不到，
  也不要假裝看得到。

★ 這也是我們回答評審「被子遮住怎麼辦」的完整答案：
  不是「我們能看穿被子」（做不到，不要吹），
  而是「我們知道自己什麼時候看不到，並且會說出來」。

資料流：

    deviations ─┐
    color ──────┼→ 每個訊號各自評分 ─┐
    可見度 ─────┘                    ├→ 加權平均 → risk
                 每個訊號各自的可信度 ┘            → confidence
                                                   → reliable?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

NAN = float("nan")


class Reliability(Enum):
    OK = "ok"                   # 訊號足夠，分數可信
    DEGRADED = "degraded"       # 部分訊號遺失，分數仍可用但要標示
    UNRELIABLE = "unreliable"   # ★ 不知道。不等於正常。


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

@dataclass
class SignalSpec:
    """一個參與投票的訊號。

    Args:
        feature:    特徵名稱，會從 deviations / color 合併後的字典裡取
        weight:     基礎權重。相對值，不需要加總為 1
        t_lo / t_hi: 評分斜坡。<= t_lo 得 0 分，>= t_hi 得 1 分，中間線性
        gate:       哪個特徵代表這個訊號的可信度（0–1）。None 表示永遠可信
        gate_floor: gate 低於此值時，這個訊號的權重降為 0
        multiply_by: 分數要乘上的另一個特徵（例如紅度要乘單調性）
        required:   ★ 這個訊號不可用時，整格直接判為 UNRELIABLE

                    預設 False。開啟的意義是「沒有這個訊號就不要假裝在判斷」。

                    arm_displacement 是最可能想開的那個 —— 針就在手臂上，
                    看不到手臂時宣稱還能評估針脫落風險，是高估我們知道的東西。

                    ⚠️ 但這是臨床取捨，不是工程選擇：真實透析時手臂很可能
                    大部分時間被蓋著。全開的話系統會整場說「不知道」，
                    那樣的告警等於沒有告警。

                    決定之前要先量出靜臥時 arm_displacement_visibility
                    的實際分布（10/9 實驗）。現在猜沒有意義，所以預設關閉。
    """
    feature: str
    weight: float = 1.0
    t_lo: float = 0.0
    t_hi: float = 1.0
    gate: Optional[str] = None
    gate_floor: float = 0.3
    multiply_by: Optional[str] = None
    required: bool = False


@dataclass
class FusionConfig:
    enabled: bool = False

    # 共識門檻：加權分數超過這個值才算「多數認為異常」
    consensus_threshold: float = 0.5

    # 可信度門檻
    min_confidence: float = 0.35     # 低於此 → UNRELIABLE
    degraded_confidence: float = 0.7 # 低於此 → DEGRADED

    # 融合後的分數要不要拿去驅動慢通道狀態機
    # 關閉時 fusion 只是觀測用，不改變告警行為 —— 做對照實驗時用這個
    drive_state_machine: bool = False

    signals: list[SignalSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "FusionConfig":
        raw = dict(raw)
        sigs = raw.pop("signals", []) or []
        return cls(signals=[SignalSpec(**s) for s in sigs], **raw)


def default_signals(color_enabled: bool = False) -> list[SignalSpec]:
    """預設的訊號組合。

    權重的理由（這些數字要能解釋，不是隨便填的）：

      arm_displacement  1.0  主訊號。針就在手臂上，它最直接
      torso_angle       0.6  輔助。翻身坐起會帶動手臂，但也可能只是換姿勢
      roi_redness       0.8  有血就是硬證據，但 ROI 容易失效，所以不給 1.0

    ⚠️ 這三個權重和所有 t_lo/t_hi 都還沒有臨床驗證。
       10/9 實驗後要用實際資料重新標定，現在的值只是能跑的起點。

    ★ 閘門用的是「這個特徵自己依賴的關鍵點」的可見率
      （features.py 的 feature_visibility），不是全身 17 點的可見率。

      理由見 features.py 的 feature_keypoints() 註解：躺著蓋被子的病人
      永遠看不到膝蓋腳踝，用全身可見率當閘門會讓系統永遠自稱不可信。

      這樣降權也精準：被子蓋住手臂只降 arm_displacement，
      軀幹角度照常投票。
    """
    sigs = [
        SignalSpec(
            feature="arm_displacement", weight=1.0,
            t_lo=0.10, t_hi=0.25,
            gate="arm_displacement_visibility", gate_floor=0.5,
        ),
        SignalSpec(
            feature="torso_angle", weight=0.6,
            t_lo=8.0, t_hi=30.0,                  # 度
            gate="torso_angle_visibility", gate_floor=0.5,
        ),
    ]
    if color_enabled:
        sigs.append(SignalSpec(
            feature="roi_redness", weight=0.8,
            t_lo=1.5, t_hi=8.0,                   # a* 單位
            gate="roi_coverage", gate_floor=0.4,  # 顏色用 ROI 有效像素比例更直接
            multiply_by="redness_monotonic",      # ★ 紅得多 **且** 是累積上去的
        ))
    return sigs


# --------------------------------------------------------------------------
# 結果
# --------------------------------------------------------------------------

@dataclass
class FusionResult:
    risk: float = NAN                 # 0–1 加權共識分數
    confidence: float = 0.0           # 0–1 這個分數有多可信
    reliability: Reliability = Reliability.UNRELIABLE
    votes: dict[str, float] = field(default_factory=dict)      # 各訊號分數
    weights: dict[str, float] = field(default_factory=dict)    # 各訊號有效權重
    missing: list[str] = field(default_factory=list)           # 這格沒有資料的訊號
    reason: str = ""

    @property
    def is_consensus(self) -> bool:
        return self.risk == self.risk and self.risk >= 0.5

    @property
    def usable(self) -> bool:
        """這一格的分數可以餵給狀態機嗎？"""
        return self.reliability is not Reliability.UNRELIABLE

    def to_payload(self) -> dict:
        """給 sink 用的扁平摘要。只有數字，沒有結構。"""
        return {
            "risk": round(self.risk, 3) if self.risk == self.risk else None,
            "conf": round(self.confidence, 3),
            "reliability": self.reliability.value,
        }


# --------------------------------------------------------------------------
# 融合器
# --------------------------------------------------------------------------

def _ramp(value: float, lo: float, hi: float) -> float:
    """線性斜坡評分，夾在 0–1。"""
    if value != value:
        return NAN
    if hi <= lo:
        return 1.0 if value > lo else 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


class ConsensusFusion:
    """多訊號加權共識。

    用法：
        fuse = ConsensusFusion(cfg)
        result = fuse(signals)      # signals = deviations | color features

    這個類別**沒有狀態**。時序的部分已經在 window / state_machine 做過了，
    融合層只負責把同一格的多個數字併成一個。
    保持無狀態讓它好測、好解釋，也讓對照實驗可以單獨關掉它。
    """

    def __init__(self, cfg: Optional[FusionConfig] = None):
        self.cfg = cfg or FusionConfig(enabled=True, signals=default_signals())
        if self.cfg.enabled and not self.cfg.signals:
            self.cfg.signals = default_signals()

        # 統計：有多少格是在不可信狀態下度過的。這個比例要寫進結案報告。
        self._frames = 0
        self._unreliable_frames = 0
        self._degraded_frames = 0

    # ------------------------------------------------------------------

    def __call__(self, signals: dict[str, float]) -> FusionResult:
        return self.update(signals)

    def update(self, signals: dict[str, float]) -> FusionResult:
        """signals：特徵名 → 值。可以混合 deviations 與 color 的輸出。"""
        if not self.cfg.enabled:
            return FusionResult(reason="fusion disabled")

        self._frames += 1

        votes: dict[str, float] = {}
        weights: dict[str, float] = {}
        missing: list[str] = []
        missing_required: list[str] = []   # 標了 required 卻不可用的訊號

        total_base = sum(s.weight for s in self.cfg.signals) or 1.0
        covered_base = 0.0
        quality_num = 0.0          # Σ(基礎權重 × 該訊號的閘門值)，只算有貢獻的

        def _drop(spec: SignalSpec) -> None:
            missing.append(spec.feature)
            if spec.required:
                missing_required.append(spec.feature)

        for spec in self.cfg.signals:
            value = signals.get(spec.feature, NAN)
            if value != value:
                _drop(spec)
                continue

            score = _ramp(abs(value), spec.t_lo, spec.t_hi)
            if score != score:
                _drop(spec)
                continue

            # 乘法修飾：例如紅度要乘上單調性
            if spec.multiply_by is not None:
                m = signals.get(spec.multiply_by, NAN)
                if m != m:
                    _drop(spec)
                    continue
                score *= max(0.0, min(1.0, m))

            # 可信度閘門：遮蔽時降權，不是直接忽略
            gate_raw = 1.0
            attenuation = 1.0
            if spec.gate is not None:
                g = signals.get(spec.gate, NAN)
                gate_raw = 1.0 if g != g else max(0.0, min(1.0, g))
                if gate_raw < spec.gate_floor:
                    # 閘門關到底：這個訊號這一格不算數
                    _drop(spec)
                    continue
                # 線性降權：gate 從 floor 升到 1，權重從 0 升到滿
                span = 1.0 - spec.gate_floor
                attenuation = ((gate_raw - spec.gate_floor) / span
                               if span > 1e-6 else 1.0)

            w = spec.weight * attenuation
            if w <= 0.0:
                _drop(spec)
                continue

            votes[spec.feature] = round(score, 4)
            weights[spec.feature] = round(w, 4)
            covered_base += spec.weight
            quality_num += spec.weight * gate_raw

        coverage = covered_base / total_base

        if not weights:
            self._unreliable_frames += 1
            return FusionResult(
                risk=NAN, confidence=0.0,
                reliability=Reliability.UNRELIABLE,
                missing=missing,
                reason="所有訊號都不可用（遮蔽或關鍵點缺失）",
            )

        # risk 的分母只算有投票的訊號 —— 缺失的訊號不算「投了 0 票」。
        # 這點很重要：把缺失當成「正常」就是這一層要避免的錯誤，
        # 缺失應該反映在 confidence 上，不是壓低 risk。

        risk = sum(votes[k] * weights[k] for k in weights) / sum(weights.values())

        # ── 可信度 ─────────────────────────────────────────────────
        # 兩個因子：
        #   coverage  有多少比例的證據還在（權重計）
        #   quality   還在的那些證據本身有多清楚（閘門值，權重平均）
        #
        # ★ 用幾何平均而不是相乘。
        #
        #   相乘會把同一個原因罰兩次 —— 遮蔽同時讓某個訊號消失
        #   （拉低 coverage）又讓剩下的訊號變模糊（拉低 quality），
        #   兩者本來就高度相關。直接相乘會讓「少一個訊號但畫面還清楚」
        #   被判成跟「全黑」差不多，那是過度悲觀，會讓系統在還能判斷的
        #   時候就宣告放棄。
        #
        #   幾何平均保留「任一項趨近 0 則整體趨近 0」的性質，
        #   但不會把兩個中等值壓成一個極低值。
        quality = quality_num / covered_base if covered_base > 0 else 0.0
        confidence = (coverage * quality) ** 0.5

        if missing_required:
            # ★ 標了 required 的訊號不見了 → 不管其他訊號多清楚，都判為不可信。
            #   意義是「沒有這個訊號就不要假裝在判斷」。預設沒有任何訊號標
            #   required，所以這一條平常不會觸發 —— 見 SignalSpec.required。
            rel = Reliability.UNRELIABLE
            self._unreliable_frames += 1
            reason = f"必要訊號不可用：{', '.join(missing_required)}"
        elif confidence < self.cfg.min_confidence:
            rel = Reliability.UNRELIABLE
            self._unreliable_frames += 1
            reason = (f"可信度 {confidence:.2f} 低於 {self.cfg.min_confidence}"
                      f"（缺 {len(missing)} 個訊號）")
        elif confidence < self.cfg.degraded_confidence:
            rel = Reliability.DEGRADED
            self._degraded_frames += 1
            reason = f"部分訊號遺失：{', '.join(missing)}" if missing else "可見度偏低"
        else:
            rel = Reliability.OK
            reason = ""

        return FusionResult(
            risk=risk, confidence=confidence, reliability=rel,
            votes=votes, weights=weights, missing=missing, reason=reason,
        )

    # ------------------------------------------------------------------

    def report(self) -> dict:
        """★ unreliable_rate 要寫進結案報告，而且要主動講。

        「這段治療有 12% 的時間系統看不到手臂」是一個誠實的限制陳述。
        藏起來的話，評審或臨床端自己發現時，整個系統的可信度會一起垮。
        """
        n = max(self._frames, 1)
        return {
            "frames": self._frames,
            "unreliable_rate": round(self._unreliable_frames / n, 4),
            "degraded_rate": round(self._degraded_frames / n, 4),
            "signals": [s.feature for s in self.cfg.signals],
            "drives_state_machine": self.cfg.drive_state_machine,
        }

    def reset(self) -> None:
        self._frames = 0
        self._unreliable_frames = 0
        self._degraded_frames = 0
