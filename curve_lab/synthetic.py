"""虚构站点日内需求数据生成。

内置峰型原型：
  single    单峰（峰位可漂移，模拟相位偏移）
  double    双峰（早晚两个峰，峰高可独立变化）
  staggered 错峰/肩峰（主峰带明显次峰，间隔异常）
  flat      平缓无峰
  extreme   极端峰（峰高显著超出常规分布）

每个训练日随机叠加：峰位偏移、峰高缩放、噪声、随机缺测段（长缺口）、
稀疏采样。缺测只体现在观测被删除，底层"真实函数"仍保留用于验证重建。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _gauss(t, center, width, height):
    d = np.abs(t - center)
    d = np.minimum(d, 24.0 - d)               # 周期距离
    return height * np.exp(-0.5 * (d / width) ** 2)


def archetype_curve(kind: str, t: np.ndarray,
                    phase: float = 0.0,
                    height_scale: float = 1.0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """生成某原型的真实曲线（24h 周期，含随机效应）。"""
    rng = rng or np.random.default_rng()
    base = 10.0
    if kind == "single":
        c = 8.5 + phase
        y = base + _gauss(t, c, 2.0, 14.0 * height_scale)
    elif kind == "double":
        c1, c2 = 8.0 + phase, 18.0 + 0.6 * phase
        h1 = 13.0 * height_scale
        h2 = 11.0 * height_scale * rng.uniform(0.8, 1.15)
        y = base + _gauss(t, c1, 1.8, h1) + _gauss(t, c2, 2.1, h2)
    elif kind == "staggered":
        c = 12.5 + phase
        y = (base + _gauss(t, c, 1.7, 15.0 * height_scale)
             + _gauss(t, c + 2.6, 1.1, 7.5 * height_scale))
    elif kind == "flat":
        y = np.full_like(t, base + 3.0 * height_scale)
    elif kind == "extreme":
        c = 13.0 + phase
        y = base + _gauss(t, c, 1.1, 30.0 * height_scale)
    else:
        raise ValueError(f"未知原型 {kind}")
    return y


@dataclass
class DaySpec:
    day_id: str
    kind: str
    phase: float = 0.0
    height_scale: float = 1.0
    noise_sd: float = 0.8
    gaps: list = field(default_factory=list)       # [(起, 止)] 小时
    sparse: bool = False
    exclude: bool = False


def _sample_times(rng: np.random.Generator,
                  gaps: list, sparse: bool) -> np.ndarray:
    if sparse:
        # 不固定采样时刻、点数偏少的日子
        n = rng.integers(7, 11)
        candidates = np.sort(rng.uniform(0, 24, size=60))
    else:
        # 半小时一次的常规采样，时刻带轻微抖动
        candidates = np.arange(0.0, 24.0, 0.5) + rng.uniform(-0.08, 0.08,
                                                             size=48)
    keep = np.ones(len(candidates), dtype=bool)
    for g0, g1 in gaps:
        keep &= ~((candidates >= g0) & (candidates <= g1))
    t = candidates[keep]
    if sparse and len(t) > n:
        t = np.sort(rng.choice(t, size=int(n), replace=False))
    return np.sort(np.mod(t, 24.0))


def generate_day(spec: DaySpec,
                 rng: np.random.Generator | None = None,
                 n_truth: int = 241):
    """返回 (观测 t, 观测 y, 全天真实网格 t, 真实 y)。"""
    rng = rng or np.random.default_rng()
    t_truth = np.linspace(0.0, 24.0, n_truth)
    y_truth = archetype_curve(spec.kind, t_truth, spec.phase,
                              spec.height_scale, rng)
    t_obs = _sample_times(rng, spec.gaps, spec.sparse)
    y_clean = archetype_curve(spec.kind, t_obs, spec.phase,
                              spec.height_scale, rng)
    y_obs = y_clean + rng.normal(0.0, spec.noise_sd, size=t_obs.shape)
    return t_obs, y_obs, t_truth, y_truth


def default_training_specs(seed: int = 42) -> list[DaySpec]:
    """训练日：三类常规原型为主，含长缺口与采样稀疏日，外加 1 个极端峰。"""
    rng = np.random.default_rng(seed)
    specs = []
    plan = (["single"] * 16 + ["double"] * 16 + ["staggered"] * 12
            + ["flat"] * 3 + ["extreme"] * 1)
    rng.shuffle(plan)
    for i, kind in enumerate(plan):
        gaps = []
        # 约 1/3 的日子带缺口，其中若干为长缺口
        if rng.random() < 0.35:
            g0 = rng.uniform(1.0, 20.0)
            length = rng.uniform(2.0, 7.5) if rng.random() < 0.4 \
                else rng.uniform(0.8, 2.5)
            gaps = [(g0, min(g0 + length, 23.9))]
        specs.append(DaySpec(
            day_id=f"D{i+1:03d}", kind=kind,
            phase=rng.normal(0.0, 1.1),
            height_scale=rng.uniform(0.8, 1.2),
            noise_sd=rng.uniform(0.5, 1.2),
            gaps=gaps,
            sparse=(rng.random() < 0.12),
        ))
    return specs


def builtin_examples() -> dict[str, DaySpec]:
    """网页内置样例：双峰、错峰、长缺口、极端峰、稀疏未定日。"""
    return {
        "double_peak": DaySpec("EX_double", "double", phase=0.4,
                               height_scale=1.05,
                               gaps=[(11.5, 14.0)]),
        "staggered": DaySpec("EX_staggered", "staggered", phase=-0.7,
                             gaps=[(2.0, 4.0)]),
        "long_gap": DaySpec("EX_longgap", "single", phase=0.3,
                            gaps=[(6.0, 13.5)], noise_sd=0.7),
        "extreme": DaySpec("EX_extreme", "extreme", phase=0.0,
                           height_scale=1.05, gaps=[(20.0, 22.0)]),
        "sparse": DaySpec("EX_sparse", "double", sparse=True,
                          gaps=[(10.0, 16.0)]),
    }


def generate_dataset(specs: list[DaySpec], seed: int = 42) -> dict:
    """批量生成，返回 {day_id: {obs, truth, spec}}。"""
    rng = np.random.default_rng(seed)
    out = {}
    for spec in specs:
        t, y, tt, yt = generate_day(spec, rng)
        out[spec.day_id] = {
            "t_obs": t, "y_obs": y, "t_truth": tt, "y_truth": yt,
            "spec": spec,
        }
    return out
