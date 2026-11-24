"""数据接入：CSV / JSON。

CSV 约定（首行表头，时间单位小时，取值 [0,24)）：
    day_id,time,value
    2026-11-01,0.5,12.3
    ...
缺测无需也不应填零——该时刻直接缺行即可，拟合端按真实缺测处理。
JSON 约定：{"days": [{"day_id": ..., "times": [...], "values": [...]}]}
"""

from __future__ import annotations

import csv
import json

import numpy as np


def load_csv(path: str) -> dict[str, dict]:
    """读取长表 CSV，按 day_id 聚合成 {day_id: {times, values}}。"""
    days: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"day_id", "time", "value"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV 需包含列 day_id,time,value")
        for row in reader:
            did = str(row["day_id"]).strip()
            try:
                t = float(row["time"])
                v = float(row["value"])
            except (TypeError, ValueError):
                continue
            if not (0.0 <= t < 24.0) or not np.isfinite(v):
                continue
            days.setdefault(did, {"times": [], "values": []})
            days[did]["times"].append(t)
            days[did]["values"].append(v)
    out = {}
    for did, d in days.items():
        order = np.argsort(d["times"])
        out[did] = {
            "times": np.asarray(d["times"])[order],
            "values": np.asarray(d["values"])[order],
        }
    return out


def dump_days_json(records: dict[str, dict], path: str) -> None:
    payload = {"days": []}
    for did, rec in records.items():
        payload["days"].append({
            "day_id": did,
            "times": np.asarray(rec["times"]).tolist(),
            "values": np.asarray(rec["values"]).tolist(),
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def load_days_json(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    out = {}
    for rec in payload.get("days", []):
        t = np.asarray(rec.get("times", []), dtype=float)
        v = np.asarray(rec.get("values", []), dtype=float)
        ok = np.isfinite(t) & np.isfinite(v) & (t >= 0) & (t < 24)
        order = np.argsort(t[ok])
        out[str(rec["day_id"])] = {"times": t[ok][order],
                                   "values": v[ok][order]}
    return out


def load_table(path: str) -> dict[str, dict]:
    """按扩展名自动选择读取器。"""
    if path.lower().endswith(".json"):
        return load_days_json(path)
    return load_csv(path)
