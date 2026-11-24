"""周期三次 B 样条基与粗糙度惩罚。

日内时间被视作周期域 [0, 24) 上的完整函数，因此选用周期样条：
- 边界天然连续（0 点与 24 点的函数值及导数一致），不产生边界伪影；
- 基函数局部支撑，缺测段的重建不确定性保持局部，不会错误地"摊薄"到全天；
- 粗糙度惩罚 ∫f''² 以基矩阵二次型表达，拟合即岭回归，GCV 可选平滑度。
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


class PeriodicBSplineBasis:
    """周期 [0, period) 上的三次 B 样条基。

    n_basis 个基函数（周期节点等间距）。系数向量长度 == n_basis，
    BSpline 内部节点数组取 t = (-3*h, ..., 24+3*h)，
    周期延拓通过 wrap 后的系数实现。
    """

    def __init__(self, n_basis: int = 20, period: float = 24.0):
        if n_basis < 6:
            raise ValueError("n_basis 过小，无法表示周期三次样条")
        self.n_basis = int(n_basis)
        self.period = float(period)
        self.h = period / n_basis
        self.k = 3
        # 内部节点 0..T，两端各补 k 个周期节点
        grid = np.arange(-self.k, n_basis + self.k + 1) * self.h
        self.t = grid
        # 每个周期基函数 = 主 B 样条 + 跨边界回绕的副本
        self._splines = [
            BSpline(self.t, self._basis_coeffs(j), self.k,
                    extrapolate=False)
            for j in range(n_basis)
        ]

    def _basis_coeffs(self, j: int) -> np.ndarray:
        """周期回绕的基系数：位置 p 对应基编号 (p-k) mod n_basis。"""
        c = np.zeros(self.n_basis + 2 * self.k)
        for p in range(c.size):
            if (p - self.k) % self.n_basis == j:
                c[p] = 1.0
        return c

    def _wrapped(self, t: np.ndarray) -> np.ndarray:
        return np.mod(np.asarray(t, dtype=float), self.period)

    def design(self, t: np.ndarray) -> np.ndarray:
        """设计矩阵 B，B[i, j] = 基函数 j 在 t_i 处的值。"""
        tw = self._wrapped(t)
        out = np.empty((len(tw), self.n_basis))
        for j, sp in enumerate(self._splines):
            out[:, j] = sp(tw)
        return out

    def derivative_design(self, t: np.ndarray, order: int = 2) -> np.ndarray:
        """阶数为 order 的导函数设计矩阵。"""
        tw = self._wrapped(t)
        out = np.empty((len(tw), self.n_basis))
        for j, sp in enumerate(self._splines):
            d = sp.derivative(order)
            out[:, j] = d(tw)
        return out

    def eval_coeffs(self, coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
        """按系数重建函数值（周期延拓）。"""
        return self.design(t) @ np.asarray(coeffs)

    def penalty(self, n_quad: int = 240) -> np.ndarray:
        """粗糙度惩罚矩阵 Omega，c' Omega c ≈ ∫ f''(t)^2 dt。"""
        tq = np.linspace(0.0, self.period, n_quad, endpoint=False)
        d2 = self.derivative_design(tq, order=2)
        w = self.period / n_quad
        return (d2.T @ d2) * w

    def gram(self, n_quad: int = 240) -> np.ndarray:
        """L2 内积矩阵 G，c' G c = ∫ f(t)^2 dt。"""
        tq = np.linspace(0.0, self.period, n_quad, endpoint=False)
        bq = self.design(tq)
        w = self.period / n_quad
        return (bq.T @ bq) * w

    def grid(self, n: int = 241) -> np.ndarray:
        return np.linspace(0.0, self.period, n)
