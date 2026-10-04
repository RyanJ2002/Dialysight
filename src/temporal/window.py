"""
滑動窗統計（第 3 層）。

核心要求：O(1) 的增量更新。

為什麼重要？
  單床 10 FPS、連續運行 4 小時 = 144,000 影格。
  15 分鐘窗有 9,000 筆資料。如果每影格都重算整個窗的統計量，
  那是 9,000 次運算 × 10 次/秒 = 每秒 9 萬次，完全沒必要。

  改用「加入新值時累加、移除舊值時相減」的做法，每影格只要幾十次運算。

提供的統計量：
  - mean / std          基本統計
  - slope               趨勢斜率（線性迴歸），區分「持續漂移」與「來回震盪」
  - deviation_from(x)   相對基準的偏離
  - count_above(th)     過去 N 秒超過閾值幾次

★ slope 是這個模組最重要的輸出。
  均值只告訴你「現在偏了多少」，斜率告訴你「正在往哪裡走」。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindowStats:
    """單一時間窗在某一瞬間的統計快照。"""
    n: int
    mean: float
    std: float
    slope: float          # 單位：特徵值 / 秒。正值 = 上升趨勢
    minimum: float
    maximum: float
    latest: float

    def is_empty(self) -> bool:
        return self.n == 0


class SlidingWindow:
    """固定時間長度的滑動窗，支援 O(1) 增量統計。

    用法：
        w = SlidingWindow(duration_s=60.0)
        w.push(timestamp=1.0, value=0.12)
        w.push(timestamp=1.1, value=0.13)
        stats = w.stats()
        print(stats.slope)   # 正值代表特徵正在上升

    實作說明：
        維護以下累加量，加入時 +=、移除時 -=：
            n, Σy, Σy², Σt, Σt², Σty
        由這六個量可以 O(1) 算出均值、變異數、以及線性迴歸斜率。
    """

    def __init__(self, duration_s: float, max_samples: Optional[int] = None):
        """
        Args:
            duration_s: 窗長，單位秒。例如 5.0 / 60.0 / 900.0
            max_samples: 樣本數上限，防止 FPS 異常時記憶體爆掉。
                         預設為 duration_s * 60（假設最高 60 FPS）
        """
        if duration_s <= 0:
            raise ValueError("duration_s 必須為正數")

        self.duration_s = duration_s
        self.max_samples = max_samples or int(duration_s * 60) + 10

        self._buf: deque[tuple[float, float]] = deque()  # (timestamp, value)

        # 增量累加量
        self._n = 0
        self._sum_y = 0.0
        self._sum_yy = 0.0
        self._sum_t = 0.0
        self._sum_tt = 0.0
        self._sum_ty = 0.0

    # ------------------------------------------------------------------
    # 寫入
    # ------------------------------------------------------------------

    def push(self, timestamp: float, value: float) -> None:
        """加入一筆新的觀測，並自動淘汰超出窗長的舊資料。

        NaN 會被忽略（特徵抽取在關鍵點缺失時會回傳 NaN）。
        """
        if value is None or math.isnan(value):
            # 仍然要淘汰舊資料，否則窗會停滯
            self._evict_expired(timestamp)
            return

        self._buf.append((timestamp, value))
        self._add(timestamp, value)

        self._evict_expired(timestamp)

        # 保險：樣本數上限
        while len(self._buf) > self.max_samples:
            t_old, y_old = self._buf.popleft()
            self._remove(t_old, y_old)

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.duration_s
        while self._buf and self._buf[0][0] < cutoff:
            t_old, y_old = self._buf.popleft()
            self._remove(t_old, y_old)

    def _add(self, t: float, y: float) -> None:
        self._n += 1
        self._sum_y += y
        self._sum_yy += y * y
        self._sum_t += t
        self._sum_tt += t * t
        self._sum_ty += t * y

    def _remove(self, t: float, y: float) -> None:
        self._n -= 1
        self._sum_y -= y
        self._sum_yy -= y * y
        self._sum_t -= t
        self._sum_tt -= t * t
        self._sum_ty -= t * y

    # ------------------------------------------------------------------
    # 讀取
    # ------------------------------------------------------------------

    def stats(self) -> WindowStats:
        """回傳當前窗內的統計快照。O(1)，但 min/max 是 O(n)。

        若 min/max 成為瓶頸，可改用單調佇列（monotonic deque），
        但以我們的資料量（<10,000 筆）來說不需要。
        """
        if self._n == 0:
            return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        n = self._n
        mean = self._sum_y / n

        # 變異數：E[y²] - E[y]²，夾在 0 以上避免浮點誤差產生負數
        var = max(0.0, self._sum_yy / n - mean * mean)
        std = math.sqrt(var)

        slope = self._slope()

        values = [y for _, y in self._buf]
        return WindowStats(
            n=n,
            mean=mean,
            std=std,
            slope=slope,
            minimum=min(values),
            maximum=max(values),
            latest=values[-1],
        )

    def _slope(self) -> float:
        """最小平方法的線性迴歸斜率。

            slope = (n·Σty − Σt·Σy) / (n·Σt² − (Σt)²)

        分母為 0 代表所有時間戳相同（樣本數為 1，或 FPS 異常），
        此時回傳 0.0 表示「無趨勢」。
        """
        n = self._n
        denom = n * self._sum_tt - self._sum_t * self._sum_t
        if n < 2 or abs(denom) < 1e-12:
            return 0.0
        return (n * self._sum_ty - self._sum_t * self._sum_y) / denom

    def deviation_from(self, baseline: float) -> float:
        """當前均值相對於基準的偏離量。

        這是 arm_displacement 的主要用法 —— 比較的對象是
        「治療開始時建立的基準」，不是絕對座標。
        """
        if self._n == 0:
            return 0.0
        return self._sum_y / self._n - baseline

    def count_above(self, threshold: float) -> int:
        """窗內超過閾值的樣本數。O(n)。

        用於「過去 15 分鐘有幾次超標」這類事件計數。
        """
        return sum(1 for _, y in self._buf if y > threshold)

    def __len__(self) -> int:
        return self._n

    def clear(self) -> None:
        self._buf.clear()
        self._n = 0
        self._sum_y = self._sum_yy = 0.0
        self._sum_t = self._sum_tt = self._sum_ty = 0.0


class MultiWindow:
    """同時維護多個不同長度的窗。

    用法：
        mw = MultiWindow({"short": 5.0, "medium": 60.0, "long": 900.0})
        mw.push(ts, value)
        mw["medium"].slope      # 過去 60 秒的趨勢斜率

    為什麼要多個窗？
        短窗抓瞬時事件，長窗抓趨勢。
        「現在手臂偏了」和「這一小時越來越躁動」是兩個不同的問題。
    """

    def __init__(self, durations: dict[str, float]):
        self.windows = {name: SlidingWindow(d) for name, d in durations.items()}
        self._cache: dict[str, WindowStats] = {}

    def push(self, timestamp: float, value: float) -> None:
        for w in self.windows.values():
            w.push(timestamp, value)
        self._cache.clear()

    def __getitem__(self, name: str) -> WindowStats:
        if name not in self._cache:
            self._cache[name] = self.windows[name].stats()
        return self._cache[name]

    def all_stats(self) -> dict[str, WindowStats]:
        return {name: self[name] for name in self.windows}

    def clear(self) -> None:
        for w in self.windows.values():
            w.clear()
        self._cache.clear()


class FeatureWindows:
    """每個特徵各自擁有一組 MultiWindow。

    這是時序層對外的主要物件。

    用法：
        fw = FeatureWindows(
            feature_names=["arm_displacement", "torso_angle"],
            durations={"short": 5.0, "medium": 60.0, "long": 900.0},
        )
        fw.push(timestamp, {"arm_displacement": 0.12, "torso_angle": 8.3})
        s = fw.stats("arm_displacement", "medium")
        print(s.mean, s.slope)
    """

    def __init__(self, feature_names: list[str], durations: dict[str, float]):
        self.durations = durations
        self.features = {name: MultiWindow(durations) for name in feature_names}
        self.baselines: dict[str, float] = {}

    def push(self, timestamp: float, features: dict[str, float]) -> None:
        for name, value in features.items():
            if name in self.features:
                self.features[name].push(timestamp, value)

    def stats(self, feature: str, window: str) -> WindowStats:
        return self.features[feature][window]

    # -- 基準線 ---------------------------------------------------------

    def set_baseline(self, feature: str, value: float) -> None:
        """設定某特徵的基準值。

        基準線建立方式有兩種（待團隊決定，見提案討論稿待決事項 #4）：
          (a) 自動：治療開始前 N 影格的均值
          (b) 手動：護理師在 UI 上按一次「建立基準」

        TODO: 兩種都實作，由 config 決定用哪一種。
        """
        self.baselines[feature] = value

    def deviation(self, feature: str, window: str = "medium") -> float:
        """相對基準的偏離。基準未建立時回傳 0.0。"""
        if feature not in self.baselines:
            return 0.0
        return self.features[feature].windows[window].deviation_from(
            self.baselines[feature]
        )

    def is_baseline_ready(self) -> bool:
        return len(self.baselines) == len(self.features)
