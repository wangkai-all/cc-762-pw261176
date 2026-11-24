"""惩罚样条拟合与缺测段重建不确定性。

模型：y_i = f(t_i) + eps_i,  f = B c,  eps ~ N(0, sigma^2)
估计：c_hat = (B'B + λΩ)^-1 B'y        （λ = 平滑度 λ）
后验：Cov(c | y) = sigma^2 (B'B + λΩ)^-1
      网格重建方差 diag(B_g Cov B_g') —— 缺测区方差自动升高。

关键点：缺测不填零。缺测时刻没有观测项，其后验完全由平滑约束与邻近数据决定，
并以方差的形式把"重建有多不确定"传递给距离与聚类（见 distance.py / cluster.py）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._compat import trapz
from .basis import PeriodicBSplineBasis


@dataclass
class FitConfig:
    # 平滑度 λ；auto_lambda=True 时作为手动覆盖关闭自动选择
    lam: float = 1.0
    auto_lambda: bool = True
    lam_grid: np.ndarray = field(
        default_factory=lambda: np.logspace(-4, 6, 25))
    n_grid: int = 241
    # 少于该观测数的日子标记为低信息量（后续聚类弃权）；
    # 日内曲线 24 小时，低于 12 个点意味着平均每 2 小时不足 1 个观测
    min_obs: int = 12
    # 判断缺测的覆盖半径（小时）：网格点距任一观测超过该值即视为缺测
    coverage_radius: float = 0.75
    band_z: float = 1.96


@dataclass
class FitResult:
    day_id: str
    coeffs: np.ndarray
    grid: np.ndarray
    mean: np.ndarray            # 重建曲线（全天网格）
    sd: np.ndarray              # 逐点后验标准差
    lower: np.ndarray
    upper: np.ndarray
    t_obs: np.ndarray
    y_obs: np.ndarray
    observed_mask: np.ndarray   # 网格点是否被观测覆盖
    gap_intervals: list         # [(t0, t1), ...] 连续缺测区间
    lam: float
    df: float
    sigma2: float
    rss: float
    n_obs: int
    uncertainty_total: float    # sqrt(∫ Var f dt)，全天不确定性的 L2 量
    uncertainty_gap: float      # 缺测区内的平均后验标准差
    low_info: bool


def _gcv_lambda(btb: np.ndarray, bty: np.ndarray, omega: np.ndarray,
                y: np.ndarray, b: np.ndarray,
                lam_grid: np.ndarray) -> tuple[float, float]:
    """广义交叉验证选择 λ：GCV(λ)= RSS/n / (1 - df/n)^2。无需已知 σ²。

    在 K×K 矩阵上做一次对称特征分解，各 λ 复用；
    df = tr(B A^-1 B')，拟合值 = B A^-1 B'y。
    """
    n = len(y)
    best = None
    vals, vecs = np.linalg.eigh(btb)
    z = vecs.T @ bty                       # 特征基下的 B'y
    omega_e = vecs.T @ omega @ vecs
    for lam in lam_grid:
        d = vals + lam * np.diag(omega_e)
        c_hat_e = z / d                    # 特征基下的系数
        df = float(np.sum(vals / d))
        fitted = b @ (vecs @ c_hat_e)
        resid_energy = float(np.sum((y - fitted) ** 2))
        denom = (1.0 - df / n) ** 2
        if denom <= 1e-12:
            continue
        gcv = (resid_energy / n) / denom
        if best is None or gcv < best[1]:
            best = (float(lam), gcv)
    return best if best is not None else (float(lam_grid[len(lam_grid) // 2]),
                                          np.inf)


def _successive_diff_sigma2(t: np.ndarray, y: np.ndarray) -> float:
    """相邻观测一阶差分的噪声方差估计（gap 过大的对剔除）。"""
    order = np.argsort(t)
    ts, ys = t[order], y[order]
    if len(ts) < 4:
        return max(np.var(y), 1e-8)
    dt = np.diff(ts)
    dy = np.diff(ys)
    keep = dt <= max(2.0 * np.median(dt), 1e-9)
    if keep.sum() < 3:
        return max(np.var(y), 1e-8)
    return max(float(np.sum(dy[keep] ** 2) / (2.0 * keep.sum())), 1e-8)


def find_gaps(grid: np.ndarray, t_obs: np.ndarray,
              radius: float) -> tuple[np.ndarray, list]:
    """按覆盖半径在全天网格上标记缺测，并提取连续缺测区间。"""
    if len(t_obs) == 0:
        mask = np.zeros_like(grid, dtype=bool)
    else:
        idx = np.searchsorted(t_obs, grid)
        d_left = grid - t_obs[np.clip(idx - 1, 0, len(t_obs) - 1)]
        d_right = t_obs[np.clip(idx, 0, len(t_obs) - 1)] - grid
        nearest = np.minimum(np.abs(d_left), np.abs(d_right))
        mask = nearest <= radius
    gaps = []
    in_gap = False
    t0 = None
    step = grid[1] - grid[0] if len(grid) > 1 else 0.0
    for g, t in zip(mask, grid):
        if not g and not in_gap:
            in_gap, t0 = True, t
        elif g and in_gap:
            gaps.append((t0, t))
            in_gap = False
    if in_gap:
        gaps.append((t0, grid[-1] + step))
    return mask, gaps


def fit_day(day_id: str, t: np.ndarray, y: np.ndarray,
            basis: PeriodicBSplineBasis,
            config: FitConfig | None = None) -> FitResult:
    """对单日观测做惩罚样条拟合，返回曲线与后验不确定性。"""
    cfg = config or FitConfig()
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    # 所有数据入口统一在拟合层做合法性过滤：有限值且时间落在周期域内
    ok = (np.isfinite(t) & np.isfinite(y)
          & (t >= 0.0) & (t < basis.period))
    t, y = t[ok], y[ok]
    n = len(y)
    if n < 4:
        raise ValueError(f"{day_id}: 有效观测不足 4 个，无法拟合")

    b = basis.design(t)
    omega = basis.penalty()
    btb, bty = b.T @ b, b.T @ y

    if cfg.auto_lambda:
        lam, _ = _gcv_lambda(btb, bty, omega, y, b, cfg.lam_grid)
    else:
        lam = float(cfg.lam)

    a = btb + lam * omega
    a_inv = np.linalg.inv(a)
    coeffs = a_inv @ bty
    fitted = b @ coeffs
    resid = y - fitted
    rss = float(resid @ resid)
    df = float(np.trace(b @ a_inv @ b.T))

    s2_diff = _successive_diff_sigma2(t, y)
    s2_resid = rss / max(n - df, 1.0)
    # 残差估计在强平滑下会偏大，取两者较小者并保留地板
    sigma2 = max(min(s2_diff, s2_resid), 1e-8)

    grid = basis.grid(cfg.n_grid)
    bg = basis.design(grid)
    mean = bg @ coeffs
    cov_grid = sigma2 * (bg @ a_inv @ bg.T)
    sd = np.sqrt(np.clip(np.diag(cov_grid), 0.0, None))

    order = np.argsort(t)
    mask, gaps = find_gaps(grid, t[order], cfg.coverage_radius)
    if mask.all():
        u_gap = 0.0
    else:
        u_gap = float(np.mean(sd[~mask]))
    step = grid[1] - grid[0]
    u_total = float(np.sqrt(trapz(sd ** 2, grid) / basis.period))
    # 归一化到平均信号尺度，避免量纲影响后续阈值
    scale = max(float(np.mean(np.abs(mean))), 1e-9)

    return FitResult(
        day_id=str(day_id), coeffs=coeffs, grid=grid, mean=mean, sd=sd,
        lower=mean - cfg.band_z * sd, upper=mean + cfg.band_z * sd,
        t_obs=t[order], y_obs=y[order], observed_mask=mask,
        gap_intervals=gaps, lam=lam, df=df, sigma2=sigma2, rss=rss, n_obs=n,
        uncertainty_total=u_total / scale,
        uncertainty_gap=u_gap / scale,
        low_info=(n < cfg.min_obs),
    )
