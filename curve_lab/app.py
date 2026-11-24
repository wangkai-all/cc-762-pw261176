"""标准库 HTTP 后端：无需任何第三方 Web 框架。

端点
----
GET  /                         前端页面
GET  /api/state                全部拟合/聚类状态
POST /api/rebuild              按界面参数重建基线（平滑 λ、s_max、k 等）
POST /api/day                  加入新日（冻结基线评分）
POST /api/exclude              人工排除/恢复（数据保留）
GET  /api/pair?a=...&b=...     两日的分项距离与扭曲细节
POST /api/upload               CSV/JSON 数据接入
GET  /api/examples             内置样例定义
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .pipeline import Pipeline, PipelineParams
from .fitting import FitConfig
from .distance import DistanceConfig
from .warping import WarpConfig
from .cluster import ClusterConfig
from .synthetic import (default_training_specs, builtin_examples,
                        generate_dataset, generate_day)
from . import dataio


WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "web")


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pipe = None
        self.dataset = None
        self.last_params = {}
        self.reset()

    def reset(self, *, n_basis=20, lam=1.0, auto_lambda=True,
              s_max=1.5, k=3, coverage_radius=0.75):
        params = PipelineParams(
            n_basis=n_basis,
            fit=FitConfig(lam=lam, auto_lambda=auto_lambda,
                          coverage_radius=coverage_radius),
            distance=DistanceConfig(warp=WarpConfig(s_max=s_max)),
            cluster=ClusterConfig(k=k),
        )
        self.pipe = Pipeline(params)
        specs = default_training_specs()
        self.dataset = generate_dataset(specs)
        obs = {did: {"times": d["t_obs"], "values": d["y_obs"],
                     "raw": {"kind": d["spec"].kind,
                             "gaps": d["spec"].gaps}}
               for did, d in self.dataset.items()}
        self.pipe.train(obs)
        self.last_params = {"n_basis": n_basis, "lam": lam,
                            "auto_lambda": auto_lambda, "s_max": s_max,
                            "k": k, "coverage_radius": coverage_radius}

    def add_example(self, name: str) -> dict:
        spec = builtin_examples()[name]
        t, y, _, _ = generate_day(spec, np.random.default_rng())
        did = f"NEW_{name}"
        if did in self.pipe.fits:
            return {"error": f"{did} 已加入"}
        verdict = self.pipe.add_day(did, t, y,
                                    raw={"kind": spec.kind,
                                         "example": name})
        return {"day_id": did, "verdict": verdict}


STATE = AppState()


def _json(handler, payload, code=200):
    body = json.dumps(payload, ensure_ascii=False,
                      default=lambda o: o.tolist()
                      if isinstance(o, np.ndarray) else str(o)).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as fh:
                content = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)
            return
        if path in ("/app.js", "/styles.css"):
            fpath = os.path.join(WEB_DIR, path.lstrip("/"))
            if not os.path.isfile(fpath):
                self.send_error(404)
                return
            ctype = ("application/javascript; charset=utf-8"
                     if fpath.endswith(".js") else "text/css; charset=utf-8")
            with open(fpath, "rb") as fh:
                content = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/api/state":
            with STATE.lock:
                return _json(self, STATE.pipe.full_state())
        if path == "/api/examples":
            return _json(self, {"examples": list(builtin_examples().keys())})
        if path == "/api/pair":
            from urllib.parse import parse_qs
            qs = {k: v[0] for k, v in parse_qs(
                self.path.split("?", 1)[-1]).items()}
            with STATE.lock:
                d = STATE.pipe._dist(qs["a"], qs["b"])
                out = {k: v for k, v in d.items() if k != "warp"}
                out["warp"] = {kk: (vv.tolist() if isinstance(vv, np.ndarray)
                                    else vv)
                               for kk, vv in d["warp"].items()}
                return _json(self, out)
        self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            payload = self._body()
        except Exception as exc:
            return _json(self, {"error": f"请求体解析失败: {exc}"}, 400)
        try:
            with STATE.lock:
                if path == "/api/rebuild":
                    STATE.reset(**{k: payload[k] for k in
                                   ("n_basis", "lam", "auto_lambda", "s_max",
                                    "k", "coverage_radius") if k in payload})
                    return _json(self, {"ok": True,
                                        "params": STATE.last_params})
                if path == "/api/add_example":
                    return _json(self, STATE.add_example(payload["name"]))
                if path == "/api/day":
                    did = str(payload["day_id"])
                    t = np.asarray(payload["times"], dtype=float)
                    y = np.asarray(payload["values"], dtype=float)
                    verdict = STATE.pipe.add_day(
                        did, t, y,
                        excluded=bool(payload.get("excluded", False)))
                    return _json(self, {"day_id": did, "verdict": verdict})
                if path == "/api/exclude":
                    STATE.pipe.set_excluded(payload["day_id"],
                                            bool(payload["excluded"]))
                    return _json(self, {"ok": True})
                if path == "/api/upload":
                    return self._handle_upload(payload)
        except (ValueError, KeyError, RuntimeError) as exc:
            return _json(self, {"error": str(exc)}, 422)
        self.send_error(404)

    def _handle_upload(self, payload):
        """接收前端打包的观测（或服务端路径，便于脚本/测试调用）。"""
        if "path" in payload:
            records = dataio.load_table(payload["path"])
        else:
            records = {d["day_id"]: {"times": np.asarray(d["times"]),
                                    "values": np.asarray(d["values"])}
                       for d in payload.get("days", [])}
        added = []
        for did, rec in records.items():
            if did in STATE.pipe.fits:
                continue
            try:
                v = STATE.pipe.add_day(did, rec["times"], rec["values"])
                added.append({"day_id": did, "status": v["status"]})
            except ValueError as exc:
                added.append({"day_id": did, "error": str(exc)})
        return _json(self, {"added": added})


def serve(host: str = "127.0.0.1", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"curve_lab 服务已启动: http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()
