/* CurveLab 前端：状态管理 + 纯 SVG 渲染（无第三方依赖）。 */
"use strict";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)",
                "var(--status-critical)", "var(--muted)"];
const STATUS_TEXT = {
  cluster: "已聚类", anomaly: "异常日", undetermined: "未定",
};
const PATTERN_TEXT = {
  single: "单峰", double: "双峰", staggered: "错峰/肩峰",
  flat: "平缓", unknown: "未知",
};

const state = {
  data: null,
  view: "overlay",
  selected: null,        // 单日视图选中
  pairA: null, pairB: null,
  showBand: true, showGaps: true, showCentroids: false, showObs: true,
  filterStatus: "", filterCluster: "",
  params: { lam: 1.0, auto_lambda: true, s_max: 1.5, k: 3,
            n_basis: 20, coverage_radius: 0.75 },
};

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const opt = body ? { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
};
const toast = (msg) => {
  let el = document.querySelector(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2600);
};

// --------------------------------------------------------------- 数据加载
async function refresh() {
  state.data = await api("/api/state");
  if (!state.selected) state.selected = state.data.days[0]?.day_id;
  renderAll();
}

// --------------------------------------------------------------- 工具
const dayById = (id) => state.data.days.find((d) => d.day_id === id);
const clusterColor = (c) => COLORS[((c ?? 0) % 3)];

function pathFrom(grid, vals, x, y) {
  let d = "";
  for (let i = 0; i < grid.length; i++) {
    d += (i ? "L" : "M") + x(grid[i]).toFixed(1) + "," + y(vals[i]).toFixed(1);
  }
  return d;
}

function bandPath(grid, lower, upper, x, y) {
  let d = "";
  for (let i = 0; i < grid.length; i++) {
    d += (i ? "L" : "M") + x(grid[i]).toFixed(1) + "," + y(lower[i]).toFixed(1);
  }
  for (let i = grid.length - 1; i >= 0; i--) {
    d += "L" + x(grid[i]).toFixed(1) + "," + y(upper[i]).toFixed(1);
  }
  return d + "Z";
}

// --------------------------------------------------------------- SVG 图表
class Chart {
  constructor(svg) {
    this.svg = svg;
    this.NS = "http://www.w3.org/2000/svg";
    this.margin = { top: 18, right: 18, bottom: 36, left: 52 };
  }

  el(name, attrs = {}, parent) {
    const node = document.createElementNS(this.NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    (parent || this.svg).appendChild(node);
    return node;
  }

  clear() {
    this.svg.innerHTML = "";
  }

  render(series, opts = {}) {
    this.clear();
    const W = this.svg.clientWidth || 900;
    const H = this.svg.clientHeight || 460;
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const m = this.margin;
    const iw = W - m.left - m.right, ih = H - m.top - m.bottom;

    const allY = series.flatMap((s) => [
      ...(s.lower || s.values), ...(s.upper || s.values)]);
    const ymin0 = Math.min(...allY), ymax0 = Math.max(...allY);
    const pad = (ymax0 - ymin0) * 0.06 + 1e-6;
    const ymin = ymin0 - pad, ymax = ymax0 + pad;
    const x = (t) => m.left + (t / 24) * iw;
    const y = (v) => m.top + (1 - (v - ymin) / (ymax - ymin)) * ih;
    this.scales = { x, y, m, iw, ih };

    // 网格与轴
    for (let h = 0; h <= 24; h += 4) {
      this.el("line", { x1: x(h), x2: x(h), y1: m.top, y2: m.top + ih,
        stroke: "var(--gridline)", "stroke-width": 1 });
      const t = this.el("text", { x: x(h), y: m.top + ih + 18,
        "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
      t.textContent = String(h).padStart(2, "0") + ":00";
    }
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const v = ymin + (ymax - ymin) * i / ticks;
      this.el("line", { x1: m.left, x2: m.left + iw, y1: y(v), y2: y(v),
        stroke: "var(--gridline)", "stroke-width": 1 });
      const t = this.el("text", { x: m.left - 8, y: y(v) + 4,
        "text-anchor": "end", fill: "var(--muted)", "font-size": 11 });
      t.textContent = v.toFixed(1);
    }
    this.el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih,
      y2: m.top + ih, stroke: "var(--axis)", "stroke-width": 1.2 });
    this.el("line", { x1: m.left, x2: m.left, y1: m.top, y2: m.top + ih,
      stroke: "var(--axis)", "stroke-width": 1.2 });

    // 缺测段阴影
    if (opts.gaps) {
      for (const g of opts.gaps) {
        const gx = x(g[0]), gw = Math.max(x(g[1]) - x(g[0]), 2);
        this.el("rect", { x: gx, y: m.top, width: gw, height: ih,
          fill: "var(--gap-shade)" });
      }
    }

    // 曲线
    for (const s of series) {
      if (s.band && state.showBand && s.lower) {
        this.el("path", { d: bandPath(s.grid, s.lower, s.upper, x, y),
          fill: s.bandColor || "var(--band)", stroke: "none" });
      }
      this.el("path", { d: pathFrom(s.grid, s.values, x, y), fill: "none",
        stroke: s.color, "stroke-width": s.width || 2,
        "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round" });
      if (s.points && state.showObs) {
        for (let i = 0; i < s.t.length; i++) {
          this.el("circle", { cx: x(s.t[i]), cy: y(s.v[i]), r: 3,
            fill: s.color, stroke: "var(--surface-1)", "stroke-width": 1 });
        }
      }
    }
    return { x, y, m, iw, ih };
  }

  addCrosshair(onMove) {
    const { x, m, ih } = this.scales;
    const line = this.el("line", { y1: m.top, y2: m.top + ih,
      stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "4 3",
      visibility: "hidden" });
    const dot = this.el("circle", { r: 4, fill: "var(--series-1)",
      stroke: "var(--surface-1)", "stroke-width": 1.5, visibility: "hidden" });
    const rect = this.el("rect", { x: m.left, y: m.top, width: this.scales.iw,
      height: this.scales.ih, fill: "transparent" });
    const move = (evt) => {
      const rectBox = this.svg.getBoundingClientRect();
      const px = evt.clientX - rectBox.left;
      const t = Math.min(24, Math.max(0, (px - m.left) / this.scales.iw * 24));
      line.setAttribute("x1", x(t)); line.setAttribute("x2", x(t));
      line.setAttribute("visibility", "visible");
      onMove(t, { line, dot, x, px });
    };
    rect.addEventListener("pointerdown", (e) => {
      rect.setPointerCapture(e.pointerId); move(e);
    });
    rect.addEventListener("pointermove", (e) => {
      if (e.buttons) move(e);
    });
    rect.addEventListener("pointerup", () => {
      line.setAttribute("visibility", "hidden");
      dot.setAttribute("visibility", "hidden");
    });
    return { line, dot };
  }
}

// --------------------------------------------------------------- 视图渲染
function buildSeries() {
  const days = state.data.days;
  if (state.view === "overlay") {
    const series = [];
    for (const d of days) {
      if (state.filterStatus && d.status !== state.filterStatus) continue;
      if (state.filterCluster !== "" &&
          String(d.cluster ?? -1) !== state.filterCluster) continue;
      const c = d.status === "anomaly" ? COLORS[3]
        : d.status === "undetermined" ? "var(--muted)"
        : clusterColor(d.cluster);
      series.push({ grid: d.grid, values: d.mean, color: c, width: 1.4,
        dash: d.excluded ? "5 4" : "none" });
    }
    if (state.showCentroids) {
      for (const [c, vals] of Object.entries(state.data.baseline.centroids)) {
        series.push({ grid: state.data.grid, values: vals,
          color: clusterColor(+c), width: 3.2, dash: "8 4" });
      }
    }
    return series;
  }
  if (state.view === "single") {
    const d = dayById(state.selected);
    if (!d) return [];
    const c = d.status === "anomaly" ? COLORS[3]
      : d.status === "undetermined" ? "var(--muted)" : clusterColor(d.cluster);
    return [{
      grid: d.grid, values: d.mean, lower: d.lower, upper: d.upper,
      color: c, band: true, points: true, t: d.t_obs, v: d.y_obs,
    }];
  }
  // pair
  const a = dayById(state.pairA), b = dayById(state.pairB);
  const out = [];
  if (a) out.push({ grid: a.grid, values: a.mean, color: COLORS[0],
    lower: a.lower, upper: a.upper, band: true, bandColor: "var(--band)",
    width: 2.2 });
  if (b) out.push({ grid: b.grid, values: b.mean, color: COLORS[1],
    lower: b.lower, upper: b.upper, band: true, bandColor: "var(--band-2)",
    width: 2.2 });
  return out;
}

function renderChart() {
  const chart = new Chart($("chart"));
  const series = buildSeries();
  if (!series.length) return;
  const opts = {};
  if (state.view === "single") {
    const d = dayById(state.selected);
    if (state.showGaps) opts.gaps = d.gap_intervals;
  }
  chart.render(series, opts);

  if (state.view === "single") {
    const d = dayById(state.selected);
    const tip = $("tooltip");
    chart.addCrosshair((t, h) => {
      const i = d.grid.findIndex((g) => g >= t);
      const idx = Math.max(0, Math.min(i, d.grid.length - 1));
      const { x, y } = h;
      h.dot.setAttribute("cx", x(d.grid[idx]));
      h.dot.setAttribute("cy", y(d.mean[idx]));
      h.dot.setAttribute("visibility", "visible");
      const inGap = !d.observed_mask[idx];
      tip.hidden = false;
      tip.innerHTML =
        `<div class="tt-title">${d.grid[idx].toFixed(1)}h ` +
        `${inGap ? "· <span style='color:var(--status-warning)'>缺测重建</span>" : ""}</div>` +
        `<div class="tt-row"><span>重建值</span><b>${d.mean[idx].toFixed(2)}</b></div>` +
        `<div class="tt-row"><span>95% 区间</span><b>${d.lower[idx].toFixed(2)} ~ ${d.upper[idx].toFixed(2)}</b></div>` +
        `<div class="tt-row"><span>后验 SD</span><b>${d.sd[idx].toFixed(3)}</b></div>`;
      const wrap = $("chart-wrap").getBoundingClientRect();
      tip.style.left = Math.min(x(d.grid[idx]) + 14, wrap.width - 190) + "px";
      tip.style.top = "12px";
    });
  } else {
    $("tooltip").hidden = true;
  }
}

// --------------------------------------------------------------- 侧栏与详情
function renderDayList() {
  const ul = $("day-list");
  ul.innerHTML = "";
  let shown = 0;
  for (const d of state.data.days) {
    if (state.filterStatus && d.status !== state.filterStatus) continue;
    if (state.filterCluster !== "" &&
        String(d.cluster ?? -1) !== state.filterCluster) continue;
    shown++;
    const li = document.createElement("li");
    if (d.day_id === state.selected) li.classList.add("selected");
    const dot = document.createElement("span");
    dot.className = "day-dot";
    dot.style.background = d.status === "anomaly" ? "var(--status-critical)"
      : d.status === "undetermined" ? "var(--muted)"
      : clusterColor(d.cluster);
    const name = document.createElement("span");
    name.className = "day-name";
    name.textContent = d.day_id;
    const kind = document.createElement("span");
    kind.className = "day-kind";
    kind.textContent = d.raw?.kind ? PATTERN_TEXT[d.raw.kind] || d.raw.kind : "";
    const pill = document.createElement("span");
    pill.className = `pill ${d.excluded ? "excluded" : d.status}`;
    pill.textContent = d.excluded ? "已排除" : STATUS_TEXT[d.status] || d.status;
    li.append(dot, name, kind, pill);
    li.addEventListener("click", () => {
      state.selected = d.day_id;
      if (state.view === "pair") {
        if (!state.pairA) state.pairA = d.day_id;
        else if (!state.pairB) state.pairB = d.day_id;
        else { state.pairA = state.pairB; state.pairB = d.day_id; }
      }
      renderAll();
    });
    ul.appendChild(li);
  }
  $("day-count").textContent = `(${shown}/${state.data.days.length})`;
}

function renderClusterFilter() {
  const sel = $("filter-cluster");
  const cur = sel.value;
  sel.innerHTML = `<option value="">全部类</option>` +
    [...Array(state.params.k).keys()].map(
      (c) => `<option value="${c}">类 ${c + 1}</option>`).join("");
  sel.value = cur;
}

function metricBar(label, value, max, color) {
  const pct = Math.min(100, value / max * 100);
  return `<div class="metric-row"><span>${label}</span>
    <span class="metric-track"><span class="metric-fill"
      style="width:${pct.toFixed(1)}%;background:${color}"></span></span>
    <span>${value.toFixed(3)}</span></div>`;
}

function renderDayDetail() {
  const d = dayById(state.selected);
  const body = $("day-detail-body");
  $("day-detail").querySelector("h2").textContent =
    `单日解释 · ${d ? d.day_id : ""}`;
  if (!d) { body.className = "detail-body muted"; body.textContent = "无"; return; }
  body.className = "detail-body";
  const f = d.features || {};
  const reasonText = {
    evidence_weak: "观测证据不足（缺测长/点数少/重建不确定性高）",
    classes_indistinguishable: "最近两类距离差过小，无法区分",
    far_from_all_baseline_classes: "远离全部基线类",
  };
  body.innerHTML = `
    <dl class="kv">
      <dt>判定状态</dt><dd><span class="pill ${d.status}">${STATUS_TEXT[d.status] || d.status}</span></dd>
      <dt>峰型</dt><dd>${PATTERN_TEXT[f.pattern] || f.pattern}</dd>
      <dt>主峰时刻</dt><dd>${f.primary_time == null ? "—" : f.primary_time.toFixed(2) + " h"}</dd>
      <dt>主峰高度</dt><dd>${f.primary_height == null ? "—" : f.primary_height.toFixed(2)}</dd>
      <dt>次峰/主峰</dt><dd>${f.height_ratio == null ? "—" : f.height_ratio.toFixed(2)}</dd>
      <dt>峰谷振幅</dt><dd>${(f.amplitude ?? 0).toFixed(2)}</dd>
      <dt>观测数</dt><dd>${d.n_obs}</dd>
      <dt>最长缺测</dt><dd>${d.longest_gap.toFixed(1)} h</dd>
      <dt>GCV λ / 自由度</dt><dd>${d.lam.toFixed(3)} / ${d.df.toFixed(1)}</dd>
      <dt>全天不确定性</dt><dd>${d.uncertainty_total.toFixed(3)}</dd>
      <dt>缺口内不确定性</dt><dd>${d.uncertainty_gap.toFixed(3)}</dd>
    </dl>
    <div class="metric-bars">
      ${metricBar("最近类距离", d.nearest_distance ?? 0,
        state.data.baseline.thresholds.anomaly_distance * 1.1, "var(--series-1)")}
      ${metricBar("异常阈值", state.data.baseline.thresholds.anomaly_distance,
        state.data.baseline.thresholds.anomaly_distance * 1.1, "var(--status-critical)")}
    </div>
    <p class="hint">${(d.reasons || []).map((r) => "· " + (reasonText[r] || r)).join("<br>") || "· 证据充分，判定稳定"}</p>
    <button class="btn small" id="exclude-btn" style="margin-top:8px">
      ${d.excluded ? "恢复参与中心更新" : "人工排除（保留数据）"}</button>`;
  $("exclude-btn").addEventListener("click", async () => {
    await api("/api/exclude", { day_id: d.day_id, excluded: !d.excluded });
    toast(d.excluded ? "已恢复，原始数据始终保留" : "已排除，原始数据保留且不参与中心更新");
    await refresh();
  });
}

async function renderPairDetail() {
  const sec = $("pair-detail");
  sec.hidden = state.view !== "pair";
  if (state.view !== "pair" || !state.pairA || !state.pairB) return;
  const d = await api(`/api/pair?a=${encodeURIComponent(state.pairA)}&b=${encodeURIComponent(state.pairB)}`);
  const maxV = Math.max(d.shape, d.height, d.phase_hours / 24, d.uncertain_l2, 0.3);
  $("pair-detail-body").innerHTML = `
    <p class="hint" style="margin-top:0">${state.pairA} ↔ ${state.pairB}</p>
    <div class="metric-bars">
      ${metricBar("形状(对齐后)", d.shape, maxV, "var(--series-1)")}
      ${metricBar("峰高差", d.height, maxV, "var(--series-2)")}
      ${metricBar("相位/24h", d.phase_hours / 24, maxV, "var(--series-3)")}
      ${metricBar("含不确定性", d.uncertain_l2, maxV, "var(--muted)")}
    </div>
    <dl class="kv" style="margin-top:10px">
      <dt>综合距离</dt><dd><b>${d.composite.toFixed(3)}</b></dd>
      <dt>平均相位</dt><dd>${d.phase_hours.toFixed(2)} h
        （${d.phase_signed_hours >= 0 ? "滞后" : "提前"} ${Math.abs(d.phase_signed_hours).toFixed(2)}h）</dd>
      <dt>实际最大拉伸</dt><dd>${d.max_stretch.toFixed(2)}×（限值 ${state.params.s_max}×）</dd>
      <dt>缺测重叠</dt><dd>${(d.gap_overlap_frac * 100).toFixed(0)}%</dd>
    </dl>`;
}

function renderHistory() {
  const ul = $("history-list");
  ul.innerHTML = "";
  for (const h of state.data.centroid_history.slice().reverse().slice(0, 12)) {
    const li = document.createElement("li");
    if (h.event === "baseline_created") {
      li.innerHTML = `基线建立：medoid 与阈值冻结<br><span class="muted">类中心初始化为训练成员均值</span>`;
    } else {
      li.innerHTML = `<b>${h.day_id}</b> 推动类 ${h.cluster + 1} 中心更新` +
        `<br><span class="muted">中心位移 RMS ${h.shift_rms.toFixed(4)} · 成员 ${h.n_members}</span>`;
    }
    ul.appendChild(li);
  }
}

function renderExamples() {
  const grid = $("example-buttons");
  grid.innerHTML = "";
  const labels = { double_peak: "双峰样例", staggered: "错峰样例",
    long_gap: "长缺口样例", extreme: "极端峰样例", sparse: "稀疏未定日" };
  for (const name of ["double_peak", "staggered", "long_gap",
                       "extreme", "sparse"]) {
    const b = document.createElement("button");
    b.className = "btn";
    b.type = "button";
    b.textContent = labels[name];
    b.addEventListener("click", async () => {
      const r = await api("/api/add_example", { name });
      if (r.error) { toast(r.error); return; }
      state.selected = r.day_id;
      toast(`${r.day_id}：${STATUS_TEXT[r.verdict.status] || r.verdict.status}`);
      await refresh();
    });
    grid.appendChild(b);
  }
}

function renderLegend() {
  const el = $("legend");
  const items = [];
  if (state.view === "overlay") {
    items.push(
      ['<i style="background:var(--series-1)"></i>类 1 成员', 0],
      ['<i style="background:var(--series-2)"></i>类 2 成员', 0],
      ['<i style="background:var(--series-3)"></i>类 3 成员', 0],
      ['<i style="background:var(--status-critical)"></i>异常日', 0],
      ['<i style="background:var(--muted)"></i>未定日', 0],
      ['<i class="gap"></i>缺测段', 0],
    );
    if (state.showCentroids) {
      items.push(['<i class="dash" style="color:var(--series-1)"></i>类中心（冻结 medoid 之外的功能均值）', 0]);
    }
  } else if (state.view === "single") {
    items.push(
      ['<i style="background:var(--series-1)"></i>重建曲线', 0],
      ['<i style="background:var(--band);height:10px;width:14px;border-radius:2px"></i>95% 重建带', 0],
      ['<i style="background:var(--series-1);border-radius:50%;width:8px;height:8px"></i>原始观测点', 0],
      ['<i class="gap"></i>缺测段（未填零）', 0],
    );
  } else {
    items.push(
      ['<i style="background:var(--series-1)"></i>曲线 A', 0],
      ['<i style="background:var(--series-2)"></i>曲线 B（有界扭曲对齐后比较）', 0],
    );
  }
  el.innerHTML = items.map(([h]) => `<span>${h}</span>`).join("");
}

function renderAll() {
  renderClusterFilter();
  renderDayList();
  renderChart();
  renderLegend();
  renderDayDetail();
  renderPairDetail();
  renderHistory();
}

// --------------------------------------------------------------- 文件接入
function parseUpload(text, name) {
  if (name.toLowerCase().endsWith(".json")) {
    const obj = JSON.parse(text);
    return obj.days.map((d) => ({ day_id: String(d.day_id),
      times: d.times.map(Number), values: d.values.map(Number) }));
  }
  const rows = text.split(/\r?\n/).filter(Boolean);
  const header = rows.shift().split(",").map((s) => s.trim().toLowerCase());
  const idx = (k) => header.indexOf(k);
  const acc = new Map();
  for (const row of rows) {
    const c = row.split(",");
    const did = c[idx("day_id")]?.trim();
    const t = parseFloat(c[idx("time")]);
    const v = parseFloat(c[idx("value")]);
    if (!did || !Number.isFinite(t) || !Number.isFinite(v)) continue;
    if (!acc.has(did)) acc.set(did, { day_id: did, times: [], values: [] });
    acc.get(did).times.push(t);
    acc.get(did).values.push(v);
  }
  return [...acc.values()];
}

function bindUpload() {
  $("upload-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    const status = $("upload-status");
    if (!file) return;
    try {
      const text = await file.text();
      const days = parseUpload(text, file.name);
      if (!days.length) throw new Error("未解析出有效记录");
      const r = await api("/api/upload", { days });
      const ok = r.added.filter((x) => !x.error).length;
      const skipped = r.added.length - ok;
      status.textContent = `已接入 ${ok} 天${skipped ? `，跳过 ${skipped} 天（重复/数据不足）` : ""}`;
      toast(`数据接入完成：${ok} 天按冻结基线评分`);
      await refresh();
    } catch (err) {
      status.textContent = "接入失败：" + err.message;
    } finally {
      e.target.value = "";
    }
  });
}

// --------------------------------------------------------------- 控件绑定
function bindControls() {
  document.querySelectorAll(".view-toggle .seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-toggle .seg")
        .forEach((b) => b.classList.toggle("active", b === btn));
      state.view = btn.dataset.view;
      if (state.view === "pair" && !state.pairA) state.pairA = state.selected;
      renderAll();
    });
  });
  for (const [id, key, fmt] of [
    ["smax", "s_max", (v) => v.toFixed(2) + "×"],
    ["k", "k", (v) => String(v)],
    ["nbasis", "n_basis", (v) => String(v)],
    ["cov", "coverage_radius", (v) => v.toFixed(2)],
  ]) {
    $(id).addEventListener("input", (e) => {
      state.params[key] = parseFloat(e.target.value);
      $(id + "-val").textContent = fmt(state.params[key]);
    });
  }
  $("auto-lam").addEventListener("change", (e) => {
    state.params.auto_lambda = e.target.checked;
    $("lam").disabled = e.target.checked;
    $("lam-val").textContent = e.target.checked ? "自动"
      : "1e" + state.params.lam;
  });
  $("lam").addEventListener("input", (e) => {
    state.params.lam = Math.pow(10, parseFloat(e.target.value));
    $("lam-val").textContent = "1e" + e.target.value;
  });
  $("rebuild").addEventListener("click", async () => {
    const btn = $("rebuild");
    btn.disabled = true; btn.textContent = "重建中…";
    try {
      await api("/api/rebuild", state.params);
      state.selected = null; state.pairA = state.pairB = null;
      toast("基线已按新参数重建并重新冻结");
      await refresh();
    } finally {
      btn.disabled = false; btn.textContent = "按参数重建基线";
    }
  });
  $("filter-status").addEventListener("change", (e) => {
    state.filterStatus = e.target.value; renderDayList();
  });
  $("filter-cluster").addEventListener("change", (e) => {
    state.filterCluster = e.target.value; renderDayList();
  });
  for (const id of ["show-band", "show-gaps", "show-centroids", "show-obs"]) {
    $(id).addEventListener("change", (e) => {
      state[Object.fromEntries([
        ["show-band", "showBand"], ["show-gaps", "showGaps"],
        ["show-centroids", "showCentroids"], ["show-obs", "showObs"]
      ])[id]] = e.target.checked;
      renderChart();
    });
  }
  $("theme-toggle").addEventListener("click", () => {
    const dark = document.body.dataset.theme === "dark";
    document.body.dataset.theme = dark ? "light" : "dark";
    $("theme-toggle").textContent = dark ? "深色" : "浅色";
  });
  window.addEventListener("resize", () => renderChart());
}

(async function init() {
  bindControls();
  bindUpload();
  renderExamples();
  await refresh();
})();
