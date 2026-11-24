"""聚类与日状态判定：k-medoids + 异常日 + "未定"弃权。

设计要点
--------
- 距离矩阵来自 function型距离（已含重建不确定性），不做填零向量聚类；
- medoid 是真实存在的训练日（曲线中心可解释、可展示）；
- 阈值全部在训练日分布上以稳健统计量标定，基线冻结后不再移动；
- 仅凭少数点无法可靠分类的日子给 "undetermined"，而不是硬塞进某类；
- 异常与"未定"分开：证据充分但远离所有类 => anomaly；证据不足 => undetermined。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterConfig:
    k: int = 3
    max_iter: int = 100
    n_init: int = 4
    # 异常阈值的稳健 z 系数（相对训练日到 medoid 距离分布）
    anomaly_robust_z: float = 3.0
    # 弃权：重建不确定性超过训练分布的分位数
    uncertainty_quantile: float = 0.90
    # 弃权：最长缺测超过该小时数，或观测数低于拟合 min_obs
    long_gap_hours: float = 6.0
    # 不确定时，最近两类距离差小于该相对量也弃权
    margin_frac: float = 0.15


def _robust_z_threshold(values: np.ndarray, z: float) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + z * 1.4826 * max(mad, 1e-12)


def kmedoids(dist: np.ndarray, k: int,
             max_iter: int = 100, n_init: int = 4,
             seed: int = 7) -> dict:
    """确定性多起点 PAM 风格 k-medoids。"""
    n = dist.shape[0]
    rng = np.random.default_rng(seed)
    k = min(k, n)
    best = None
    starts = [np.arange(k)]
    for _ in range(n_init - 1):
        starts.append(rng.choice(n, size=k, replace=False))
    for init_ids in starts:
        med = np.array(init_ids, dtype=int)
        labels = None
        for _ in range(max_iter):
            d_to_med = dist[:, med]
            new_labels = np.argmin(d_to_med, axis=1)
            new_med = med.copy()
            changed = False
            for c in range(k):
                members = np.where(new_labels == c)[0]
                if len(members) == 0:
                    continue
                costs = dist[np.ix_(members, members)].sum(axis=1)
                pick = members[int(np.argmin(costs))]
                if pick != new_med[c]:
                    new_med[c], changed = pick, True
            if labels is not None and np.array_equal(new_labels, labels) \
                    and not changed:
                break
            labels, med = new_labels, new_med
        cost = float(dist[np.arange(n), med[labels]].sum())
        if best is None or cost < best["cost"]:
            best = {"medoids": med.copy(), "labels": labels.copy(),
                    "cost": cost}
    return best


def calibrate_thresholds(dist_to_medoids: np.ndarray,
                         uncertainties: np.ndarray,
                         cfg: ClusterConfig) -> dict:
    """在训练分布上标定异常/弃权阈值（基线冻结的一部分）。"""
    nearest = np.min(dist_to_medoids, axis=1)
    return {
        "anomaly_distance": _robust_z_threshold(nearest,
                                                cfg.anomaly_robust_z),
        "uncertainty_limit": float(
            np.quantile(uncertainties, cfg.uncertainty_quantile)),
        "nearest_median": float(np.median(nearest)),
    }


def assign_day(dist_to_medoids: np.ndarray,
               uncertainty_total: float,
               longest_gap_hours: float,
               low_info: bool,
               thresholds: dict,
               cfg: ClusterConfig) -> dict:
    """单日落类：返回状态（cluster 编号 / anomaly / undetermined）与理由。"""
    order = np.argsort(dist_to_medoids)
    d1, d2 = dist_to_medoids[order[0]], dist_to_medoids[order[1]]
    reasons = []

    weak_evidence = (
        low_info
        or uncertainty_total > thresholds["uncertainty_limit"]
        or longest_gap_hours >= cfg.long_gap_hours
    )
    if weak_evidence:
        reasons.append("evidence_weak")
    margin = (d2 - d1) / max(d1, 1e-12)
    if margin < cfg.margin_frac:
        reasons.append("classes_indistinguishable")

    anomalous = d1 > thresholds["anomaly_distance"]
    if anomalous:
        reasons.append("far_from_all_baseline_classes")

    if anomalous and not weak_evidence:
        status = "anomaly"
    elif weak_evidence or (anomalous and margin < cfg.margin_frac):
        status = "undetermined"
    else:
        status = "cluster"
    return {
        "status": status,
        "cluster": int(order[0]) if status != "anomaly" else None,
        "nearest_cluster": int(order[0]),
        "nearest_distance": float(d1),
        "second_distance": float(d2),
        "margin": float(margin),
        "reasons": reasons,
    }
