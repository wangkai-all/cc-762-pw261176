"""峰型识别：在重建函数上（而非原始稀疏点上）做周期性峰值检测。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PeakInfo:
    time: float
    height: float
    prominence: float


@dataclass
class PeakSummary:
    pattern: str = "unknown"     # single | double | staggered | flat | unknown
    peaks: list = field(default_factory=list)
    primary_time: float | None = None
    primary_height: float | None = None
    secondary_time: float | None = None
    secondary_height: float | None = None
    peak_gap: float | None = None       # 双峰/错峰的时间间隔（小时）
    height_ratio: float | None = None   # 次峰高 / 主峰高
    max_height: float | None = None
    min_height: float | None = None
    daily_mean: float | None = None
    amplitude: float | None = None      # 峰谷差
    cv: float | None = None             # 日内变异系数
    extreme: bool = False


def _periodic_peaks(grid: np.ndarray, y: np.ndarray, period: float
                    ) -> list[tuple[int, float, float]]:
    """周期边界下的局部极大值，返回 (展开索引, 时间, 高度)；展开索引含虚拟副本。"""
    n = len(grid)
    step = grid[1] - grid[0]
    ext = np.concatenate([y[-3:], y, y[:3]])
    found = []
    for i in range(3, n + 3):
        if ext[i] >= ext[i - 1] and ext[i] > ext[i + 1]:
            gi = i - 3
            found.append((gi, grid[gi], float(y[gi])))
    return found


def _prominence(y: np.ndarray, idx: int) -> float:
    """简化的周期 prominence：峰到两侧最近更高点（或全局最低谷）的落差。"""
    n = len(y)
    h = y[idx]
    lo = h
    for step_dir in (1, -1):
        j = (idx + step_dir) % n
        cur_min = y[j]
        while j != idx:
            if y[j] > h:
                break
            cur_min = min(cur_min, y[j])
            j = (j + step_dir) % n
        lo = min(lo, cur_min)
    return float(h - lo)


def peak_summary(grid: np.ndarray, y: np.ndarray,
                 period: float = 24.0,
                 prominence_frac: float = 0.12,
                 double_sep: tuple = (1.5, 9.0),
                 extreme_z: float | None = None,
                 ref_mean: float | None = None,
                 ref_sd: float | None = None) -> PeakSummary:
    """识别日内峰型。

    pattern 判定：
      flat      : 最大 prominence 低于信号振幅的 prominence_frac
      double    : 两个显著峰且次峰高 ≥ 0.55 主峰
      staggered : 显著峰 ≥2 但次峰较弱（错峰/肩峰），或两峰间隔异常
      single    : 仅一个显著峰
    extreme 由跨日参考分布判定（训练集均值/标准差传入时）。
    """
    y = np.asarray(y)
    raw = _periodic_peaks(grid, y, period)
    amp = float(np.max(y) - np.min(y))
    prom_thr = max(prominence_frac * amp, 1e-9)
    peaks = [PeakInfo(t, h, _prominence(y, gi)) for gi, t, h in raw]
    peaks = [p for p in peaks if p.prominence >= prom_thr]
    peaks.sort(key=lambda p: p.height, reverse=True)

    s = PeakSummary(
        peaks=peaks,
        max_height=float(np.max(y)), min_height=float(np.min(y)),
        daily_mean=float(np.mean(y)), amplitude=amp,
        cv=float(np.std(y) / max(np.mean(y), 1e-9)),
    )
    if not peaks:
        s.pattern = "flat"
        return s

    s.primary_time, s.primary_height = peaks[0].time, peaks[0].height
    if len(peaks) >= 2:
        p1, p2 = peaks[0], peaks[1]
        s.secondary_time, s.secondary_height = p2.time, p2.height
        gap = abs(p1.time - p2.time)
        s.peak_gap = float(min(gap, period - gap))
        s.height_ratio = float(p2.height / max(p1.height, 1e-9))
        lo, hi = double_sep
        if s.height_ratio >= 0.55 and lo <= s.peak_gap <= hi:
            s.pattern = "double"
        else:
            s.pattern = "staggered"
    else:
        s.pattern = "single"

    if extreme_z is not None and ref_mean is not None and ref_sd is not None:
        z = (s.max_height - ref_mean) / max(ref_sd, 1e-9)
        s.extreme = bool(z >= extreme_z)
    return s
