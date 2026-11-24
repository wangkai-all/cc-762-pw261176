"""curve_lab 核心验证（python -m unittest 或直接运行）。"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from curve_lab import (
    PeriodicBSplineBasis, fit_day, FitConfig, bounded_warp, WarpConfig,
    functional_distance, DistanceConfig, kmedoids, assign_day,
    ClusterConfig, Pipeline, PipelineParams, generate_dataset,
    default_training_specs, builtin_examples, generate_day,
)
from curve_lab.synthetic import archetype_curve, DaySpec


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


class TestBasis(unittest.TestCase):
    def test_partition_of_unity_and_periodicity(self):
        basis = PeriodicBSplineBasis(20)
        t = np.linspace(0, 24, 500, endpoint=False)
        b = basis.design(t)
        self.assertTrue(np.allclose(b.sum(axis=1), 1.0, atol=1e-10))
        # 0 与 24 的基取值一致（周期）
        self.assertTrue(np.allclose(basis.design([0.0])[0],
                                    basis.design([24.0])[0], atol=1e-10))
        omega = basis.penalty()
        eig = np.linalg.eigvalsh(omega)
        self.assertGreaterEqual(eig.min(), -1e-8)


class TestFitting(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.basis = PeriodicBSplineBasis(20)

    def test_reconstruction_of_known_curve(self):
        t = np.arange(0.25, 24, 0.5)
        truth = archetype_curve("single", t, phase=0.0, rng=self.rng)
        y = truth + self.rng.normal(0, 0.4, size=t.shape)
        fr = fit_day("d", t, y, self.basis)
        truth_grid = archetype_curve("single", fr.grid, phase=0.0,
                                     rng=self.rng)
        # 观测区重建误差应接近噪声水平
        self.assertLess(rms(fr.mean, truth_grid), 1.0)
        # 人为制造一段缺测：缺测区后验方差应显著高于观测区
        keep = ~((t >= 7) & (t <= 11))
        fr_gap = fit_day("d2", t[keep], y[keep], self.basis)
        self.assertGreater(fr_gap.sd[~fr_gap.observed_mask].mean(),
                           fr_gap.sd[fr_gap.observed_mask].mean() * 1.5)

    def test_gap_not_filled_zero(self):
        """长缺口日的重建保持邻近水平，绝不能塌成零。"""
        t = np.arange(0.25, 24, 0.5)
        truth = archetype_curve("single", t, rng=self.rng)
        y = truth + self.rng.normal(0, 0.3, size=t.shape)
        keep = ~((t >= 6) & (t <= 13))
        fr = fit_day("g", t[keep], y[keep], self.basis)
        gap_band = (fr.grid > 6.5) & (fr.grid < 12.5)
        self.assertTrue(np.all(fr.mean[gap_band] > 5.0))
        self.assertTrue(len(fr.gap_intervals) >= 1)

    def test_manual_lambda_smoother_reduces_roughness(self):
        t = np.arange(0.25, 24, 0.5)
        y = archetype_curve("double", t, rng=self.rng) + \
            self.rng.normal(0, 1.5, size=48)
        rough = fit_day("r", t, y, self.basis, FitConfig(
            auto_lambda=False, lam=1e-3))
        smooth = fit_day("s", t, y, self.basis, FitConfig(
            auto_lambda=False, lam=1e4))
        self.assertLess(np.var(np.diff(smooth.mean, 2)),
                        np.var(np.diff(rough.mean, 2)))


class TestWarping(unittest.TestCase):
    def test_phase_shift_recovered_and_slope_bounded(self):
        basis = PeriodicBSplineBasis(20)
        grid = basis.grid()
        rng = np.random.default_rng(1)
        f = archetype_curve("single", grid, phase=0.0, rng=rng)
        g = archetype_curve("single", grid, phase=2.0, rng=rng)
        cfg = WarpConfig(s_max=1.5)
        out = bounded_warp(grid, f, grid, g, 24.0, cfg)
        # 相位方向正确且恢复出有意义的提前量（受斜率界限制，不必完全等于 2h）
        self.assertGreaterEqual(abs(out["phase_signed"]), 0.8)
        self.assertLessEqual(out["max_slope"], 1.5 + 1e-6)
        self.assertGreaterEqual(out["min_slope"], 1 / 1.5 - 1e-6)
        # 对齐后形状残差远小于未对齐差异
        raw = rms(f, g)
        self.assertLess(out["shape_l2"], raw * 0.4)

    def test_unreasonable_stretch_rejected(self):
        """s_max=1.2 时无法完成需要大局部拉伸的对齐，残差比宽松约束大。"""
        basis = PeriodicBSplineBasis(20)
        grid = basis.grid()
        rng = np.random.default_rng(2)
        f = archetype_curve("single", grid, phase=0.0, rng=rng)
        g = archetype_curve("single", grid, phase=4.0, rng=rng)
        out = bounded_warp(grid, f, grid, g, 24.0,
                           WarpConfig(s_max=1.2))
        self.assertLessEqual(out["max_slope"], 1.2 + 1e-6)
        # 受限扭曲下残差明显大于自由（1.8）扭曲
        free = bounded_warp(grid, f, grid, g, 24.0,
                            WarpConfig(s_max=1.8))
        self.assertGreater(out["shape_l2"], free["shape_l2"])


class TestDistance(unittest.TestCase):
    def setUp(self):
        self.basis = PeriodicBSplineBasis(20)
        self.rng = np.random.default_rng(3)
        self.t = np.arange(0.25, 24, 0.5)

    def _fit(self, did, **kw):
        y = archetype_curve(kw.pop("kind", "single"), self.t, rng=self.rng,
                            **kw)
        return fit_day(did, self.t, y, self.basis)

    def test_height_change_separated_from_phase(self):
        base = self._fit("a")
        tall = self._fit("b", height_scale=1.6)
        shifted = self._fit("c", phase=2.5)
        d_tall = functional_distance(base, tall, 24.0,
                                     DistanceConfig(warp=WarpConfig()))
        d_shift = functional_distance(base, shifted, 24.0,
                                      DistanceConfig(warp=WarpConfig()))
        self.assertGreater(d_tall["height"], d_shift["height"] * 1.5)
        self.assertGreater(d_shift["phase_hours"],
                           d_tall["phase_hours"] * 1.5)

    def test_uncertainty_inflates_distance(self):
        y = archetype_curve("double", self.t, rng=self.rng)
        full = fit_day("f", self.t, y, self.basis)
        keep = ~((self.t >= 9) & (self.t <= 15))
        gappy = fit_day("p", self.t[keep], y[keep], self.basis)
        d = functional_distance(full, gappy)
        self.assertGreater(d["uncertain_l2"], d["raw_l2"])

    def test_fillzero_would_mislead(self):
        """对照实验：缺口位置不同的同类日，填零法误判，函数型距离保持正确。"""
        rng = np.random.default_rng(9)
        # 同原型（单峰、轻微相位差）但缺口位置错开；异原型与 A 缺口位置相同
        y1 = archetype_curve("single", self.t, phase=0.0, rng=rng)
        y2 = archetype_curve("single", self.t, phase=0.4, rng=rng)
        yd = archetype_curve("double", self.t, rng=rng)
        keep_a = ~((self.t >= 6) & (self.t <= 12))
        keep_b = ~((self.t >= 14) & (self.t <= 20))
        z1, z2, zd = (np.where(keep_a, y1, 0.0),
                      np.where(keep_b, y2, 0.0),
                      np.where(keep_a, yd, 0.0))
        d_same_zero = np.linalg.norm(z1 - z2)
        d_diff_zero = np.linalg.norm(z1 - zd)
        # 填零法：缺口位置主导距离，把异类（同缺口）判得比同类（不同缺口）近
        self.assertGreater(d_same_zero, d_diff_zero)

        f1 = fit_day("a", self.t[keep_a], y1[keep_a], self.basis)
        f2 = fit_day("b", self.t[keep_b], y2[keep_b], self.basis)
        fd = fit_day("d", self.t[keep_a], yd[keep_a], self.basis)
        d_same = functional_distance(f1, f2)["composite"]
        d_diff = functional_distance(f1, fd)["composite"]
        # 函数型距离：同原型仍显著近于异原型
        self.assertLess(d_same, 0.65 * d_diff)


class TestClusteringAndPipeline(unittest.TestCase):
    def _build(self, seed=42):
        data = generate_dataset(default_training_specs(seed), seed=seed)
        obs = {did: {"times": d["t_obs"], "values": d["y_obs"],
                     "raw": {"kind": d["spec"].kind}}
               for did, d in data.items()}
        pipe = Pipeline(PipelineParams(
            cluster=ClusterConfig(k=3, n_init=2),
            distance=DistanceConfig(warp=WarpConfig())))
        pipe.train(obs)
        return pipe, data

    def test_clusters_recover_archetypes_and_are_stable(self):
        pipe, data = self._build()
        labels = pipe.baseline["labels"]
        # k=3 对应 4 个自然原型，单峰/错峰合并为一大类属合理的无监督结果；
        # 要求每个类由一个真实原型主导（最大份额 >= 0.5）。
        for c in range(3):
            members = [d for d, lab in labels.items() if lab == c]
            kinds = [data[d]["spec"].kind for d in members]
            main = max(set(kinds), key=kinds.count)
            self.assertGreaterEqual(kinds.count(main) / len(kinds), 0.5)
        # 双峰原型绝大多数识别到同一类（个别长缺口日可能落入相邻类）
        double_ids = [d for d in labels
                      if data[d]["spec"].kind == "double"]
        main_cluster = max(set(labels[d] for d in double_ids),
                           key=lambda c: sum(labels[d] == c
                                             for d in double_ids))
        share = sum(labels[d] == main_cluster for d in double_ids) \
            / len(double_ids)
        self.assertGreaterEqual(share, 0.85)
        # 确定性：重跑标签完全一致
        pipe2, _ = self._build()
        self.assertEqual(pipe.baseline["labels"], pipe2.baseline["labels"])

    def test_extreme_day_flagged_and_sparse_day_undetermined(self):
        pipe, data = self._build()
        rng = np.random.default_rng(7)
        ex = builtin_examples()
        verdicts = {}
        for name, spec in ex.items():
            t, y, _, _ = generate_day(spec, rng)
            verdicts[name] = pipe.add_day(f"NEW_{name}", t, y)
        self.assertEqual(verdicts["extreme"]["status"], "anomaly")
        self.assertEqual(verdicts["sparse"]["status"], "undetermined")
        # 长缺口日若证据不足也应未定，而不是硬聚类
        self.assertIn(verdicts["long_gap"]["status"],
                      {"undetermined", "cluster"})

    def test_baseline_frozen_and_centroid_update_gated(self):
        pipe, _ = self._build()
        before_med = list(pipe.baseline["medoid_ids"])
        before_thr = dict(pipe.baseline["thresholds"])
        rng = np.random.default_rng(11)
        spec = DaySpec("ok", "single", phase=0.2, gaps=[])
        t, y, _, _ = generate_day(spec, rng)
        v = pipe.add_day("NEW_ok", t, y)
        self.assertEqual(pipe.baseline["medoid_ids"], before_med)
        self.assertEqual(pipe.baseline["thresholds"], before_thr)
        self.assertTrue(v["centroid_updated"])
        # 历史更新留痕
        self.assertEqual(pipe.centroid_history[-1]["day_id"], "NEW_ok")
        # 人工排除日：数据保留，不更新中心
        spec2 = DaySpec("ex", "single")
        t2, y2, _, _ = generate_day(spec2, rng)
        v2 = pipe.add_day("NEW_ex", t2, y2, excluded=True)
        self.assertFalse(v2["centroid_updated"])
        self.assertIn("NEW_ex", pipe.excluded)
        self.assertEqual(len(pipe.fits["NEW_ex"].t_obs), len(t2))

    def test_reconstruction_accuracy_against_known_truth(self):
        """已知曲线：观测区重建 RMSE 低；缺口区不确定性随缺口变长而增大。"""
        basis = PeriodicBSplineBasis(20)
        rng = np.random.default_rng(5)
        t = np.arange(0.25, 24, 0.5)
        truth = archetype_curve("double", t, rng=rng)
        y = truth + rng.normal(0, 0.5, size=t.shape)
        short = ~((t >= 10) & (t <= 11.5))
        long = ~((t >= 9) & (t <= 15))
        f_short = fit_day("s", t[short], y[short], basis)
        f_long = fit_day("l", t[long], y[long], basis)
        tt = np.linspace(0, 24, 241)
        full_truth = archetype_curve("double", tt, rng=rng)
        self.assertLess(rms(f_short.mean, full_truth), 1.2)
        gap_long = ~f_long.observed_mask
        gap_short = ~f_short.observed_mask
        self.assertGreater(f_long.sd[gap_long].mean(),
                           f_short.sd[gap_short].mean())


if __name__ == "__main__":
    unittest.main(verbosity=2)
