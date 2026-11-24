import sys, time
sys.path.insert(0, "/workspace")
import numpy as np
from curve_lab import (Pipeline, PipelineParams, ClusterConfig,
                       DistanceConfig, WarpConfig,
                       generate_dataset, default_training_specs)

t0 = time.time()
data = generate_dataset(default_training_specs(42), seed=42)
obs = {did: {"times": d["t_obs"], "values": d["y_obs"],
             "raw": {"kind": d["spec"].kind}}
       for did, d in data.items()}
pipe = Pipeline(PipelineParams(
    cluster=ClusterConfig(k=3, n_init=2),
    distance=DistanceConfig(warp=WarpConfig())))
pipe.train(obs)
print("train seconds:", round(time.time() - t0, 1))
labels = pipe.baseline["labels"]
for c in range(3):
    members = [d for d, lab in labels.items() if lab == c]
    kinds = [data[d]["spec"].kind for d in members]
    print(f"cluster {c}: n={len(members)}",
          {k: kinds.count(k) for k in set(kinds)})
print("medoids:", pipe.baseline["medoid_ids"],
      [data[m]["spec"].kind for m in pipe.baseline["medoid_ids"]])

# 单次 warp 计时
from curve_lab import bounded_warp
a = pipe.fits[pipe.baseline["medoid_ids"][0]]
b = pipe.fits[pipe.baseline["training_ids"][5]]
t0 = time.time()
bounded_warp(a.grid, a.mean, b.grid, b.mean, 24.0, WarpConfig())
print("one warp seconds:", round(time.time() - t0, 3))
