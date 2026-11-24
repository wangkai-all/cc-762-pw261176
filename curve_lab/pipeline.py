"""训练/增量流水线：基线冻结、新日评分、类中心受控更新、人工排除。

约束（对应需求）
----------------
1. 训练阶段建立平滑约束基函数表示与类结构；medoids 与判定阈值构成"基线"，
   序列化后冻结。新日一律对照冻结基线评分。
2. 只有"明确重建"的日子（低不确定性、非异常、非弃权）才能更新历史类中心
   （功能型均值曲线），且 medoid 与阈值永不因新数据移动；每次更新留痕。
3. 人工排除日保留全部原始数据与拟合结果，仅不参与中心更新，可在界面对比。
4. 新日不触发重新拟合/重聚类，保证历史类稳定、结果可复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .basis import PeriodicBSplineBasis
from .fitting import fit_day, FitConfig, FitResult, find_gaps
from .features import peak_summary
from .distance import functional_distance, DistanceConfig
from .warping import WarpConfig
from .cluster import (kmedoids, calibrate_thresholds, assign_day,
                      ClusterConfig)


@dataclass
class PipelineParams:
    n_basis: int = 20
    fit: FitConfig = field(default_factory=FitConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    period: float = 24.0


def _longest_gap(gaps: list, period: float) -> float:
    if not gaps:
        return 0.0
    return float(max(g1 - g0 for g0, g1 in gaps))


class Pipeline:
    def __init__(self, params: PipelineParams | None = None):
        self.params = params or PipelineParams()
        self.basis = PeriodicBSplineBasis(self.params.n_basis,
                                          self.params.period)
        self.fits: dict[str, FitResult] = {}
        self.features: dict[str, dict] = {}
        self.pairwise: dict[tuple[str, str], dict] = {}
        self._warp_cache: dict[tuple, dict] = {}
        self.days: dict[str, dict] = {}
        self.baseline: dict | None = None
        self.centroid_history: list[dict] = []
        self.excluded: set[str] = set()

    # ------------------------------------------------------------------ fit
    def _fit_one(self, day_id: str, t: np.ndarray, y: np.ndarray,
                 raw: dict | None = None) -> FitResult:
        fr = fit_day(day_id, t, y, self.basis, self.params.fit)
        self.fits[day_id] = fr
        ref = self.baseline
        ps = peak_summary(
            fr.grid, fr.mean, self.params.period,
            extreme_z=3.0,
            ref_mean=ref["peak_height_mean"] if ref else None,
            ref_sd=ref["peak_height_sd"] if ref else None,
        )
        self.features[day_id] = asdict(ps)
        self.days.setdefault(day_id, {})
        self.days[day_id].update({
            "day_id": day_id,
            "role": "new",
            "t_obs": fr.t_obs, "y_obs": fr.y_obs,
            "longest_gap": _longest_gap(fr.gap_intervals, self.params.period),
            "excluded": day_id in self.excluded,
            "raw": raw or {},
        })
        return fr

    def _dist(self, a: str, b: str) -> dict:
        key = tuple(sorted((a, b)))
        if key not in self.pairwise:
            wc = self.params.distance.warp or WarpConfig()
            warp_key = key + (wc.s_max, wc.n_segments, wc.n_positions)
            if warp_key not in self._warp_cache:
                self._warp_cache[warp_key] = functional_distance(
                    self.fits[key[0]], self.fits[key[1]],
                    self.params.period, self.params.distance)
            self.pairwise[key] = self._warp_cache[warp_key]
        return self.pairwise[key]

    # ---------------------------------------------------------------- train
    def train(self, observations: dict[str, dict]) -> dict:
        """observations: {day_id: {"times", "values", 可选 "raw"}}。"""
        ids = sorted(observations)
        for did in ids:
            rec = observations[did]
            self._fit_one(did, rec["times"], rec["values"],
                          rec.get("raw"))

        n = len(ids)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._dist(ids[i], ids[j])["composite"]
                dist_mat[i, j] = dist_mat[j, i] = d

        cfg = self.params.cluster
        km = kmedoids(dist_mat, cfg.k, cfg.max_iter, cfg.n_init)
        medoid_ids = [ids[i] for i in km["medoids"]]
        d_to_med = dist_mat[:, km["medoids"]]
        uncertainties = np.array([self.fits[i].uncertainty_total
                                  for i in ids])
        thresholds = calibrate_thresholds(d_to_med, uncertainties, cfg)

        heights = np.array([self.features[i]["max_height"] for i in ids])
        self.baseline = {
            "medoid_ids": medoid_ids,
            "thresholds": thresholds,
            "training_ids": ids,
            "labels": {ids[i]: int(km["labels"][i]) for i in range(n)},
            "peak_height_mean": float(np.mean(heights)),
            "peak_height_sd": float(np.std(heights)),
            "params_snapshot": self._params_snapshot(),
        }
        # 初始功能型类中心（成员均值曲线）
        self.baseline["centroids"] = self._compute_centroids(
            self.baseline["labels"])
        self.centroid_history.append({
            "event": "baseline_created",
            "centroids": {c: m.copy() for c, m in
                          self.baseline["centroids"].items()},
        })

        for i, did in enumerate(ids):
            verdict = assign_day(
                d_to_med[i], self.fits[did].uncertainty_total,
                self.days[did]["longest_gap"], self.fits[did].low_info,
                thresholds, cfg)
            self.days[did].update({"role": "training", **verdict})
        return {"n_days": n, "medoids": medoid_ids,
                "thresholds": thresholds}

    def _compute_centroids(self, labels: dict[str, int]) -> dict[str, np.ndarray]:
        groups: dict[int, list[str]] = {}
        for did, c in labels.items():
            if did in self.excluded:
                continue
            groups.setdefault(c, []).append(did)
        out = {}
        for c, members in groups.items():
            out[str(c)] = np.mean([self.fits[m].mean for m in members],
                                  axis=0)
        return out

    # --------------------------------------------------------------- scoring
    def add_day(self, day_id: str, times: np.ndarray, values: np.ndarray,
                raw: dict | None = None,
                excluded: bool = False) -> dict:
        """新日对照冻结基线评分；合格者才推动类中心更新。"""
        if self.baseline is None:
            raise RuntimeError("需先 train() 建立基线")
        if day_id in self.fits:
            raise ValueError(f"{day_id} 已存在")
        if excluded:
            self.excluded.add(day_id)
        self._fit_one(day_id, times, values, raw)

        d = [self._dist(day_id, m)["composite"]
             for m in self.baseline["medoid_ids"]]
        fr = self.fits[day_id]
        verdict = assign_day(
            np.array(d), fr.uncertainty_total,
            self.days[day_id]["longest_gap"], fr.low_info,
            self.baseline["thresholds"], self.params.cluster)
        verdict["distances_to_medoids"] = [float(x) for x in d]
        self.days[day_id].update(verdict)

        # 重新做峰型极端判定（基线参考分布已冻结）
        ps = peak_summary(
            fr.grid, fr.mean, self.params.period, extreme_z=3.0,
            ref_mean=self.baseline["peak_height_mean"],
            ref_sd=self.baseline["peak_height_sd"])
        self.features[day_id] = asdict(ps)

        updated = False
        if (not excluded and verdict["status"] == "cluster"
                and verdict["cluster"] is not None):
            updated = self._update_centroid(day_id, verdict["cluster"])
        verdict["centroid_updated"] = updated
        return verdict

    def _update_centroid(self, day_id: str, cluster: int) -> bool:
        """受控更新：仅功能型均值曲线移动，medoid/阈值保持冻结。"""
        key = str(cluster)
        old = self.baseline["centroids"].get(key)
        if old is None:
            return False
        members = [d for d, c in self.baseline["labels"].items()
                   if c == cluster and d not in self.excluded]
        members = members + [
            d for d, info in self.days.items()
            if info.get("role") == "new" and info.get("cluster") == cluster
            and d not in self.excluded and d != day_id
        ]
        members.append(day_id)
        new_mean = np.mean([self.fits[m].mean for m in members], axis=0)
        shift = float(np.sqrt(np.mean((new_mean - old) ** 2)))
        self.baseline["centroids"][key] = new_mean
        self.centroid_history.append({
            "event": "centroid_updated",
            "day_id": day_id,
            "cluster": cluster,
            "shift_rms": shift,
            "n_members": len(members),
        })
        return True

    def set_excluded(self, day_id: str, excluded: bool) -> None:
        """人工排除/恢复：原始数据与拟合始终保留。"""
        if day_id not in self.days:
            raise KeyError(day_id)
        if excluded:
            self.excluded.add(day_id)
        else:
            self.excluded.discard(day_id)
        self.days[day_id]["excluded"] = excluded

    # ------------------------------------------------------------- snapshot
    def _params_snapshot(self) -> dict:
        dc = self.params.distance
        return {
            "n_basis": self.params.n_basis,
            "lam": self.params.fit.lam,
            "auto_lambda": self.params.fit.auto_lambda,
            "coverage_radius": self.params.fit.coverage_radius,
            "s_max": (dc.warp or WarpConfig()).s_max,
            "k": self.params.cluster.k,
        }

    def day_payload(self, day_id: str) -> dict:
        fr = self.fits[day_id]
        info = self.days[day_id]
        return {
            "day_id": day_id,
            "role": info["role"],
            "status": info.get("status"),
            "cluster": info.get("cluster"),
            "nearest_cluster": info.get("nearest_cluster"),
            "reasons": info.get("reasons", []),
            "nearest_distance": info.get("nearest_distance"),
            "margin": info.get("margin"),
            "excluded": info.get("excluded", False),
            "n_obs": fr.n_obs,
            "lam": fr.lam,
            "df": fr.df,
            "sigma2": fr.sigma2,
            "uncertainty_total": fr.uncertainty_total,
            "uncertainty_gap": fr.uncertainty_gap,
            "longest_gap": info["longest_gap"],
            "low_info": fr.low_info,
            "gap_intervals": fr.gap_intervals,
            "t_obs": fr.t_obs.tolist(),
            "y_obs": fr.y_obs.tolist(),
            "grid": fr.grid.tolist(),
            "mean": fr.mean.tolist(),
            "lower": fr.lower.tolist(),
            "upper": fr.upper.tolist(),
            "sd": fr.sd.tolist(),
            "observed_mask": fr.observed_mask.tolist(),
            "features": self.features.get(day_id),
            "raw": info.get("raw") or {},
        }

    def full_state(self) -> dict:
        history = []
        for h in self.centroid_history:
            item = {k: v for k, v in h.items() if k != "centroids"}
            if "centroids" in h:
                item["centroids"] = {c: np.asarray(m).tolist()
                                     for c, m in h["centroids"].items()}
            history.append(item)
        return {
            "baseline": {
                "medoid_ids": self.baseline["medoid_ids"],
                "thresholds": self.baseline["thresholds"],
                "training_ids": self.baseline["training_ids"],
                "labels": self.baseline["labels"],
                "peak_height_mean": self.baseline["peak_height_mean"],
                "peak_height_sd": self.baseline["peak_height_sd"],
                "params_snapshot": self.baseline["params_snapshot"],
                "centroids": {c: m.tolist() for c, m in
                              self.baseline["centroids"].items()},
            },
            "grid": self.basis.grid(self.params.fit.n_grid).tolist(),
            "days": [self.day_payload(d) for d in sorted(self.days)],
            "centroid_history": history,
            "excluded": sorted(self.excluded),
        }

    def save(self, path: str) -> None:
        state = self.full_state()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False,
                      default=lambda o: o.tolist()
                      if isinstance(o, np.ndarray) else str(o))
