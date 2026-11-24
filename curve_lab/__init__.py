"""curve_lab — 缺测日内曲线的函数型建模、不确定性距离与稳健聚类。

模块划分
--------
basis     : 周期三次 B 样条基 + 粗糙度惩罚矩阵
fitting   : 惩罚最小二乘 / GCV 平滑 / 重建后验方差（缺测段不确定性）
features  : 峰型识别（单峰、双峰、错峰、峰高等）
warping   : 斜率有界的单调时间扭曲（限制不合理拉伸）
distance  : 含重建不确定性的函数型距离，相位/峰高/形状分项解释
cluster   : k-medoids + 异常日 + "未定" 弃权规则
synthetic : 虚构站点数据生成（内置双峰、错峰、长缺口、极端峰样例）
dataio    : CSV / JSON 接入
pipeline  : 训练基线冻结、新日评分、类中心受控更新、人工排除
app       : 标准库 HTTP 后端
"""

from .basis import PeriodicBSplineBasis
from .fitting import fit_day, FitConfig, FitResult
from .features import peak_summary
from .warping import bounded_warp, WarpConfig
from .distance import functional_distance, DistanceConfig
from .cluster import kmedoids, assign_day, ClusterConfig
from .pipeline import Pipeline, PipelineParams
from .synthetic import (generate_dataset, default_training_specs,
                        builtin_examples, generate_day)
from . import dataio

__all__ = [
    "PeriodicBSplineBasis",
    "fit_day",
    "FitConfig",
    "FitResult",
    "peak_summary",
    "bounded_warp",
    "WarpConfig",
    "functional_distance",
    "DistanceConfig",
    "kmedoids",
    "assign_day",
    "ClusterConfig",
    "Pipeline",
    "PipelineParams",
    "generate_dataset",
    "default_training_specs",
    "builtin_examples",
    "generate_day",
    "dataio",
]
