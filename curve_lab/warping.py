"""斜率有界的单调时间扭曲（有限时间扭曲，禁止不合理拉伸）。

对齐 f（基准）与 g（待对齐）：寻找周期单调映射 w，使
    f(u) ≈ g(w(u)),  w(0)=0, w(T)=T,
    s_min <= w'(u) <= s_max。
s_max 即"允许的最大拉伸倍数"（如 1.5 表示局部时间最多拉伸/压缩 50%），
由界面可调。相位偏移因此被显式建模，与垂直峰高变化分开解释。

求解分两层：
1. 动态规划（correlation-optimized warping 风格）：u 轴分 L 段，
   节点位置在 P 个候选位置上，仅允许满足斜率界的转移，强制 w(T)=T，
   全局最优、确定性、毫秒级；
2. 以 DP 解为初值做少量 L-BFGS-B 抛光（斜率框约束 + 节点归一化闭合）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from ._compat import trapz


@dataclass
class WarpConfig:
    n_segments: int = 10        # DP 的 u 轴段数
    n_positions: int = 37       # DP 候选节点位置数（含端点）
    s_max: float = 1.5          # 最大局部斜率；最小斜率取 1/s_max
    n_eval: int = 61            # 积分评估网格
    reg: float = 1e-3           # 恒等扭曲偏好正则
    polish_iter: int = 0        # >0 时追加 L-BFGS-B 抛光（DP 已足够时关闭）


def _periodic_interp(grid: np.ndarray, values: np.ndarray, period: float):
    ext_t = np.concatenate([grid - period, grid, grid + period])
    ext_v = np.concatenate([values, values, values])
    base = interp1d(ext_t, ext_v, kind="linear", fill_value="extrapolate")
    return lambda t: base(np.mod(t, period))


def _signed_mean(delta: np.ndarray, period: float) -> float:
    """圆周意义下的平均时间差（小时），符号表示提前/滞后。"""
    ang = 2 * np.pi * delta / period
    return float(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
                 * period / (2 * np.pi))


def _dp_warp(f_interp, g_interp, period: float,
             cfg: WarpConfig) -> np.ndarray:
    """斜率受限动态规划，返回节点位置 v_0..v_L（v_0=0, v_L=T）。"""
    L, P = cfg.n_segments, cfg.n_positions
    s_hi, s_lo = cfg.s_max, 1.0 / cfg.s_max
    du = period / L
    u_nodes = np.linspace(0.0, period, L + 1)
    v_grid = np.linspace(0.0, period, P)
    r = 5
    rs = np.linspace(0.0, 1.0, r, endpoint=False) + 0.5 / r

    f_seg = f_interp(u_nodes[:-1, None] + rs[None, :] * du)   # (L, r)

    p_idx, q_idx = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    v_start = v_grid[p_idx]
    v_end = v_grid[q_idx]
    slope = (v_end - v_start) / du
    # 合法转移：单调推进且斜率在界内
    valid = (slope >= s_lo - 1e-9) & (slope <= s_hi + 1e-9) & (q_idx > p_idx)
    w_pts = (v_start[:, :, None]
             + slope[:, :, None] * (rs[None, None, :] * du))  # (P,P,r)
    g_pts = g_interp(w_pts)                                    # (P,P,r)

    INF = 1e18
    cost = np.full((L + 1, P), INF)
    back = np.full((L + 1, P), -1, dtype=int)
    cost[0, 0] = 0.0
    eye = np.arange(P)
    for k in range(1, L + 1):
        # 第 k 段失配：(P,P)，仅在 valid 处有限
        mismatch = np.mean((g_pts - f_seg[k - 1][None, None, :]) ** 2,
                           axis=-1) * du
        feasible = np.where(valid, mismatch, INF)
        cand = cost[k - 1][:, None] + feasible                 # (P,P)
        best_p = np.argmin(cand, axis=0)
        cost[k] = cand[best_p, eye]
        back[k] = best_p
    if not np.isfinite(cost[L, P - 1]):
        return u_nodes.copy()
    path = np.zeros(L + 1, dtype=int)
    path[L] = P - 1
    for k in range(L, 0, -1):
        path[k - 1] = back[k, path[k]]
    return v_grid[path]


def bounded_warp(f_grid: np.ndarray, f: np.ndarray,
                 g_grid: np.ndarray, g: np.ndarray,
                 period: float = 24.0,
                 config: WarpConfig | None = None) -> dict:
    """把 g 对齐到 f，返回扭曲函数、对齐残差与相位统计。"""
    cfg = config or WarpConfig()
    m = cfg.n_segments
    s_hi = float(cfg.s_max)
    s_lo = 1.0 / s_hi
    node_u = np.linspace(0.0, period, m + 1)
    h = period / m

    g_interp = _periodic_interp(g_grid, g, period)
    f_interp = _periodic_interp(f_grid, f, period)
    u_eval = np.linspace(0.0, period, cfg.n_eval)

    # ---- 第 1 层：DP 全局解 ----
    v_dp = _dp_warp(f_interp, g_interp, period, cfg)

    # ---- 第 2 层：以段斜率为变量做受限抛光 ----
    def nodes_from_q(q: np.ndarray) -> np.ndarray:
        v = np.concatenate([[0.0], np.cumsum(q) * h])
        return v * (period / v[-1])      # 严格周期闭合

    def objective(q: np.ndarray) -> float:
        v = nodes_from_q(q)
        wu = np.interp(u_eval, node_u, v)
        resid = f_interp(u_eval) - g_interp(wu)
        closure_pen = 100.0 * (np.sum(q) - m) ** 2
        reg = cfg.reg * np.sum((q - 1.0) ** 2) * h
        return float(trapz(resid ** 2, u_eval) + reg + closure_pen * h)

    q0 = np.diff(v_dp) / h
    if cfg.polish_iter > 0:
        res = minimize(objective, np.clip(q0, s_lo, s_hi),
                       method="L-BFGS-B", bounds=[(s_lo, s_hi)] * m,
                       options={"maxiter": cfg.polish_iter, "ftol": 1e-9})
        v = nodes_from_q(res.x)
        converged = bool(res.success)
    else:
        v = v_dp
        converged = True
    q_norm = np.diff(v) / h

    w_eval = np.interp(u_eval, node_u, v)
    aligned_g = g_interp(w_eval)
    aligned_f = f_interp(u_eval)
    resid = aligned_f - aligned_g
    shape_l2 = float(np.sqrt(trapz(resid ** 2, u_eval) / period))

    delta = w_eval - u_eval
    return {
        "u": u_eval,
        "w": w_eval,
        "node_u": node_u,
        "node_v": v,
        "aligned_f": aligned_f,
        "aligned_g": aligned_g,
        "shape_l2": shape_l2,
        "phase_l1": float(np.mean(np.abs(delta))),
        "phase_signed": _signed_mean(delta, period),
        "max_slope": float(np.max(q_norm)),
        "min_slope": float(np.min(q_norm)),
        "converged": converged,
    }
