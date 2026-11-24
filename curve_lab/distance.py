"""函数型距离：把缺测重建不确定性传入距离，并把相位/峰高/形状分开。

普通向量聚类的问题：缺测填零后，缺口被当作真实的零值参与欧氏距离，
缺一段与真的出现一段低谷不可区分。这里：
1. 两条曲线都以拟合后的完整函数参与距离（在全天网格上积分）；
2. 后验方差进入距离本身——
   E‖f-g‖² = ∫(μ_f-μ_g)² + Var(f)+Var(g)，
   重建越不确定的区间，距离上"拿不准"的罚越大；
3. 先做有界时间扭曲对齐，再把总差异拆成
   相位（扭曲量）、峰高（垂直尺度差）、形状（对齐后残差）三项，
   使"错峰"与"峰变高"不会被混为一谈。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._compat import trapz
from .fitting import FitResult
from .warping import bounded_warp, WarpConfig


@dataclass
class DistanceConfig:
    warp: WarpConfig | None = None
    # 综合距离的分项权重
    w_shape: float = 1.0
    w_height: float = 0.8
    w_phase: float = 0.15          # 相位以小时计，权重控制其与无量纲项平衡
    uncertainty_weight: float = 1.0
    gap_bonus: float = 0.5         # 缺测重叠区的额外不确定罚权重


def _integrate_rms(grid: np.ndarray, values: np.ndarray,
                   period: float) -> float:
    return float(np.sqrt(trapz(values, grid) / period))


def functional_distance(a: FitResult, b: FitResult,
                        period: float = 24.0,
                        config: DistanceConfig | None = None) -> dict:
    """计算两个拟合日之间的解释性函数型距离。"""
    cfg = config or DistanceConfig()
    warp_cfg = cfg.warp or WarpConfig()
    if not np.allclose(a.grid, b.grid):
        raise ValueError("两个 FitResult 的网格不一致（需同一 basis/n_grid）")
    grid = a.grid

    # ---- 未对齐的原始函数型距离（含不确定性传播）----
    diff2 = (a.mean - b.mean) ** 2
    var_sum = a.sd ** 2 + b.sd ** 2
    base_l2 = _integrate_rms(grid, diff2, period)
    uncertain_l2 = _integrate_rms(grid, diff2 + var_sum, period)

    both_gap = (~a.observed_mask) & (~b.observed_mask)
    if both_gap.any():
        gap_var = float(np.mean(var_sum[both_gap]))
    else:
        gap_var = 0.0
    inflation = float(np.sqrt(max(uncertain_l2 ** 2
                                  + cfg.gap_bonus * gap_var, 0.0)))

    # ---- 有界时间扭曲：相位与形状分离 ----
    warp = bounded_warp(grid, a.mean, grid, b.mean, period, warp_cfg)
    scale = max(float(np.mean(np.abs(a.mean)) + np.mean(np.abs(b.mean))) / 2,
                1e-9)
    shape = warp["shape_l2"] / scale

    # ---- 峰高差异（垂直变化，独立于相位）----
    ha = float(np.max(a.mean))
    hb = float(np.max(b.mean))
    height = abs(ha - hb) / scale

    phase_h = warp["phase_l1"]
    composite = (cfg.w_shape * shape
                 + cfg.w_height * height
                 + cfg.w_phase * (phase_h / period)
                 + cfg.uncertainty_weight * (inflation - base_l2) / scale)

    return {
        "raw_l2": base_l2 / scale,
        "uncertain_l2": inflation / scale,
        "shape": shape,
        "height": height,
        "phase_hours": phase_h,
        "phase_signed_hours": warp["phase_signed"],
        "max_stretch": warp["max_slope"],
        "composite": float(max(composite, 0.0)),
        "warp": warp,
        "gap_overlap_frac": float(both_gap.mean()),
    }
