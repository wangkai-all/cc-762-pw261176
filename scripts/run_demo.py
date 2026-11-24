"""命令行演示：数据接入 → 拟合 → 聚类基线 → 新日评分 → 报告导出。

用法
----
.venv/bin/python -m scripts.run_demo                 # 全流程 + 内置样例
.venv/bin/python -m scripts.run_demo --emit-csv data/sample.csv
.venv/bin/python -m scripts.run_demo --input data/sample.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve_lab import (
    Pipeline, PipelineParams, ClusterConfig, DistanceConfig, WarpConfig,
    generate_dataset, default_training_specs, builtin_examples, generate_day,
    dataio,
)


def build_pipeline(k: int = 3, s_max: float = 1.5) -> Pipeline:
    return Pipeline(PipelineParams(
        cluster=ClusterConfig(k=k),
        distance=DistanceConfig(warp=WarpConfig(s_max=s_max)),
    ))


def emit_csv(path: str) -> None:
    specs = default_training_specs()
    data = generate_dataset(specs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("day_id,time,value\n")
        for did, d in data.items():
            for t, v in zip(d["t_obs"], d["y_obs"]):
                fh.write(f"{did},{t:.3f},{v:.3f}\n")
    print(f"已写出样例 CSV（缺测时刻直接缺行）：{path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CurveLab 命令行演示")
    ap.add_argument("--input", help="CSV/JSON 输入文件；缺省使用内置虚构站点数据")
    ap.add_argument("--emit-csv", help="生成样例 CSV 后退出")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--s-max", type=float, default=1.5)
    ap.add_argument("--report", default="data/report.json")
    args = ap.parse_args()

    if args.emit_csv:
        emit_csv(args.emit_csv)
        return

    if args.input:
        records = dataio.load_table(args.input)
        obs = {did: {"times": r["times"], "values": r["values"]}
               for did, r in records.items()}
        print(f"接入 {len(obs)} 天：{args.input}")
    else:
        data = generate_dataset(default_training_specs())
        obs = {did: {"times": d["t_obs"], "values": d["y_obs"],
                     "raw": {"kind": d["spec"].kind}}
               for did, d in data.items()}

    pipe = build_pipeline(args.k, args.s_max)
    info = pipe.train(obs)
    print(f"训练完成：{info['n_days']} 天，medoid = {info['medoids']}")
    print(f"异常阈值 {info['thresholds']['anomaly_distance']:.3f}，"
          f"不确定性弃权限 {info['thresholds']['uncertainty_limit']:.3f}")

    # 内置样例作为"新日期"对照冻结基线评分
    rng = np.random.default_rng(2026)
    print("\n新日评分（基线冻结）：")
    for name, spec in builtin_examples().items():
        t, y, _, _ = generate_day(spec, rng)
        verdict = pipe.add_day(f"NEW_{name}", t, y,
                               raw={"kind": spec.kind, "example": name})
        print(f"  NEW_{name:12s} -> {verdict['status']:13s} "
              f"类={verdict['cluster']} 理由={verdict['reasons']}")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    pipe.save(args.report)
    print(f"\n完整状态（含每条曲线/带/判定/留痕）已导出：{args.report}")


if __name__ == "__main__":
    main()
