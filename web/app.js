// Model debugging dashboard for the Vélib' forecasting pipeline.
//
// Reads <dir>/models/diagnostics.json (holdout errors, series, importance) and,
// if present, <dir>/forecast.json (next-week forecast). <dir> is `?dir=...`,
// otherwise `data`, falling back to the synthetic demo in `data/demo`.
// The public front end lives in a separate repository; this page is for
// finding out where the model is wrong.
"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const METHODS = [
  { key: "model", label: "Model", color: "--series-model", dash: "" },
  { key: "seasonal_naive", label: "Same hour last week", color: "--series-naive", dash: "6 4" },
  { key: "weekly_profile", label: "Weekly average", color: "--series-profile", dash: "2 3" },
];
const ACTUAL = { key: "actual", label: "Actual", color: "--ink", dash: "" };
const DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const TZ = "Europe/Paris";

// Diverging: red (bad / empty) <-> grey <-> blue (good / full).
const DIVERGING = ["--div-neg-2", "--div-neg-1", "--div-mid", "--div-pos-1", "--div-pos-2"];
// Sequential: one hue, light -> dark.
const SEQUENTIAL = ["--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"];

const MODES = {
  error: {
    label: "Model error",
    title: "Model MAE on the holdout (bikes)",
    value: (s) => (s.mae ? s.mae.model : null),
    format: (v) => `${v.toFixed(2)} bikes`,
    bins: () => quantileBins(state.diag.stations.map((s) => (s.mae ? s.mae.model : null))),
    colors: SEQUENTIAL,
  },
  skill: {
    label: "Skill vs baseline",
    title: "Error reduction vs the best baseline",
    value: (s) => s.skill,
    format: (v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`,
    bins: () => [-0.1, -0.02, 0.02, 0.1],
    binLabels: ["Worse by > 10%", "Worse by 2–10%", "Same (± 2%)", "Better by 2–10%", "Better by > 10%"],
    colors: DIVERGING,
  },
  coverage: {
    label: "Data coverage",
    title: "Share of hours with an observation",
    value: (s) => s.coverage,
    format: (v) => `${(v * 100).toFixed(0)}%`,
    bins: () => [0.5, 0.8, 0.9, 0.97],
    colors: SEQUENTIAL,
  },
  forecast: {
    label: "Forecast",
    title: "Predicted fill level",
    value: (s) => forecastFill(s.code),
    format: (v) => `${(v * 100).toFixed(0)}% full`,
    bins: () => [0.1, 0.3, 0.7, 0.9],
    binLabels: ["Almost empty (< 10%)", "Few bikes", "Balanced (30–70%)", "Few docks", "Almost full (≥ 90%)"],
    colors: DIVERGING,
  },
};

const WORST_COLUMNS = [
  { key: "name", label: "Station", sort: (s) => s.name, numeric: false },
  { key: "mae", label: "Model MAE", sort: (s) => (s.mae ? s.mae.model : null), numeric: true },
  { key: "skill", label: "vs baseline", sort: (s) => s.skill, numeric: true },
  { key: "coverage", label: "Coverage", sort: (s) => s.coverage, numeric: true },
  { key: "capacity", label: "Capacity", sort: (s) => s.capacity, numeric: true },
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  dir: null,
  diag: null,
  forecast: null,
  forecastByCode: new Map(),
  mode: "error",
  day: 0,
  hour: 8,
  selected: null,
  markers: new Map(),
  sort: { key: "mae", descending: true },
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function quantileBins(values) {
  const sorted = values.filter((v) => v !== null && v !== undefined).sort((a, b) => a - b);
  if (!sorted.length) return [1, 2, 3, 4];
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  return [q(0.2), q(0.4), q(0.6), q(0.8)];
}

function binIndex(value, bins) {
  let i = 0;
  while (i < bins.length && value >= bins[i]) i++;
  return i;
}

const localFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: TZ,
  weekday: "short",
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
});
const localParts = new Intl.DateTimeFormat("en-GB", { timeZone: TZ, hour: "2-digit", hourCycle: "h23" });

function hourTimes(startIso, count) {
  const start = new Date(startIso).getTime();
  return Array.from({ length: count }, (_, i) => new Date(start + i * 3600_000));
}

function formatHour(hour) {
  return `${String(hour).padStart(2, "0")}:00`;
}

function fmt(value, digits = 2) {
  return value === null || value === undefined ? "–" : value.toFixed(digits);
}

// ---------------------------------------------------------------------------
// Line chart (SVG) with crosshair + tooltip
// ---------------------------------------------------------------------------

/**
 * @param {HTMLElement} container
 * @param {{labels: string[], series: {label, color, dash, values, width?}[],
 *          ticks: {index:number,label:string}[], refLines?: {value, label}[],
 *          height?: number, markers?: boolean}} spec
 */
function lineChart(container, spec) {
  container.replaceChildren();
  const width = container.clientWidth || 320;
  const height = spec.height || 150;
  const pad = { top: 8, right: 8, bottom: 20, left: 32 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const n = spec.labels.length;
  const all = spec.series.flatMap((s) => s.values).filter((v) => v !== null);
  const refMax = Math.max(0, ...(spec.refLines || []).map((r) => r.value));
  const yMax = niceMax(Math.max(1e-9, ...all, refMax));
  const x = (i) => pad.left + (n <= 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const y = (v) => pad.top + innerH - (v / yMax) * innerH;

  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, class: "line-chart" });

  for (const t of [0, 0.5, 1]) {
    const value = yMax * t;
    root.append(svg("line", { x1: pad.left, x2: width - pad.right, y1: y(value), y2: y(value), class: t === 0 ? "baseline" : "grid" }));
    const label = svg("text", { x: pad.left - 6, y: y(value) + 3, class: "tick", "text-anchor": "end" });
    label.textContent = value.toFixed(yMax >= 10 ? 0 : 1);
    root.append(label);
  }
  for (const tick of spec.ticks) {
    const label = svg("text", { x: x(tick.index), y: height - 5, class: "tick", "text-anchor": "middle" });
    label.textContent = tick.label;
    root.append(label);
  }
  for (const ref of spec.refLines || []) {
    root.append(svg("line", { x1: pad.left, x2: width - pad.right, y1: y(ref.value), y2: y(ref.value), class: "ref" }));
    const label = svg("text", { x: width - pad.right, y: y(ref.value) - 3, class: "tick", "text-anchor": "end" });
    label.textContent = ref.label;
    root.append(label);
  }

  // Draw baselines first so the model and actual lines sit on top.
  for (const series of [...spec.series].reverse()) {
    let d = "";
    let pen = false;
    series.values.forEach((v, i) => {
      if (v === null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    root.append(
      svg("path", {
        d,
        fill: "none",
        stroke: `var(${series.color})`,
        "stroke-width": series.width || 2,
        "stroke-dasharray": series.dash || "",
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      }),
    );
    if (spec.markers) {
      series.values.forEach((v, i) => {
        if (v !== null) root.append(svg("circle", { cx: x(i), cy: y(v), r: 3, fill: `var(${series.color})`, class: "dot" }));
      });
    }
  }

  // Hover layer.
  const cross = svg("line", { y1: pad.top, y2: pad.top + innerH, class: "crosshair", visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { x: pad.left, y: 0, width: innerW, height, fill: "transparent" });
  root.append(hit);

  const tooltip = el("div", { class: "tooltip", hidden: true });
  const wrap = el("div", { class: "chart-wrap" }, root, tooltip);
  container.append(wrap);

  const show = (event) => {
    const rect = root.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * width;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - pad.left) / innerW) * (n - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    tooltip.replaceChildren(
      el("div", { class: "tooltip-title", text: spec.labels[i] }),
      ...spec.series.map((s) =>
        el(
          "div",
          { class: "tooltip-row" },
          el("span", { class: "swatch-line", style: `border-color:var(${s.color});border-top-style:${s.dash ? "dashed" : "solid"}` }),
          el("span", { text: s.label }),
          el("b", { text: fmt(s.values[i], 1) }),
        ),
      ),
    );
    tooltip.hidden = false;
    const left = (x(i) / width) * rect.width;
    tooltip.style.left = `${Math.min(Math.max(left + 12, 0), rect.width - tooltip.offsetWidth)}px`;
    if (left + 12 + tooltip.offsetWidth > rect.width) tooltip.style.left = `${Math.max(left - tooltip.offsetWidth - 12, 0)}px`;
  };
  const hide = () => {
    tooltip.hidden = true;
    cross.setAttribute("visibility", "hidden");
  };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerleave", hide);
}

function niceMax(value) {
  const exponent = Math.pow(10, Math.floor(Math.log10(value)));
  for (const step of [1, 2, 2.5, 5, 10]) if (value <= step * exponent) return step * exponent;
  return 10 * exponent;
}

function renderLegends() {
  const items = {
    methods: METHODS,
    holdout: [ACTUAL, ...METHODS],
  };
  for (const node of document.querySelectorAll("[data-legend]")) {
    node.replaceChildren(
      ...items[node.dataset.legend].map((s) =>
        el(
          "span",
          { class: "legend-item" },
          el("span", { class: "swatch-line", style: `border-color:var(${s.color});border-top-style:${s.dash ? "dashed" : "solid"}` }),
          s.label,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function renderOverview() {
  const { metrics, test_start, test_end } = state.diag;
  $("holdout-window").textContent = `${test_start.slice(0, 10)} → ${test_end.slice(0, 10)}`;
  $("scores").replaceChildren(
    ...METHODS.map((m) => {
      const score = metrics.scores[m.key];
      return el(
        "tr",
        { class: m.key === "model" ? "is-model" : "" },
        el("th", { scope: "row" }, el("span", { class: "swatch-line", style: `border-color:var(${m.color});border-top-style:${m.dash ? "dashed" : "solid"}` }), m.label),
        el("td", { text: fmt(score.mae) }),
        el("td", { text: fmt(score.rmse) }),
      );
    }),
  );
  const skill = metrics.skill_vs_best_baseline;
  $("skill").textContent =
    skill > 0
      ? `Model is ${(skill * 100).toFixed(1)}% better than the best baseline on ${metrics.rows.toLocaleString()} station-hours.`
      : `Model does not beat the best baseline (${(skill * 100).toFixed(1)}%) on ${metrics.rows.toLocaleString()} station-hours.`;
  $("overview").hidden = false;
}

function renderErrorCharts() {
  const methods = (source) => METHODS.map((m) => ({ ...m, values: source[m.key] }));
  lineChart($("by-hour"), {
    labels: Array.from({ length: 24 }, (_, h) => formatHour(h)),
    series: methods(state.diag.by_hour),
    ticks: [0, 6, 12, 18, 23].map((h) => ({ index: h, label: formatHour(h) })),
  });
  $("by-hour-section").hidden = false;
  lineChart($("by-day"), {
    labels: DAY_SHORT,
    series: methods(state.diag.by_day_of_week),
    ticks: DAY_SHORT.map((label, index) => ({ index, label })),
    markers: true,
  });
  $("by-day-section").hidden = false;
}

function renderImportance() {
  const items = state.diag.importance.slice(0, 14);
  const max = Math.max(1e-9, ...items.map((i) => i.mae_increase_pp));
  $("importance").replaceChildren(
    ...items.map((item) =>
      el(
        "li",
        { title: `${item.feature}: +${item.mae_increase_pp.toFixed(2)} pp (± ${item.std_pp.toFixed(2)})` },
        el("span", { class: "feature", text: item.feature }),
        el(
          "span",
          { class: "bar-track" },
          el("span", { class: "bar", style: `width:${Math.max(0, (item.mae_increase_pp / max) * 100)}%` }),
        ),
        el("span", { class: "value", text: item.mae_increase_pp.toFixed(2) }),
      ),
    ),
  );
  $("importance-section").hidden = false;
}

function renderDataFacts() {
  const d = state.diag.data;
  const facts = [
    ["Grid", `${d.grid_start.slice(0, 10)} → ${d.grid_end.slice(0, 10)}`],
    ["Hours", d.grid_hours.toLocaleString()],
    ["Stations", d.stations.toLocaleString()],
    ["Observed", `${(d.observed_share * 100).toFixed(1)}% of station-hours`],
  ];
  $("data-facts").replaceChildren(...facts.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
  $("data-section").hidden = false;
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

const map = L.map("map", { preferCanvas: true }).setView([48.8566, 2.3522], 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

// Date (local) of the forecast cell for a weekday/hour: the window is 168 hours
// from forecast_start, so when it starts mid-day a weekday spans two dates.
function cellDate(day, hour) {
  if (!state.forecast) return "";
  const match = hourTimes(state.forecast.forecast_start, 168).find(
    (t) => localDayIndex(t) === day && Number(localParts.format(t)) === hour,
  );
  return match ? match.toLocaleDateString("en-CA", { timeZone: TZ }) : "";
}

function forecastFill(code) {
  const station = state.forecastByCode.get(code);
  if (!station || station.capacity <= 0) return null;
  return Math.min(station.bikes[state.day][state.hour] / station.capacity, 1);
}

function currentScale() {
  const mode = MODES[state.mode];
  return { mode, bins: mode.bins(), colors: mode.colors.map(cssVar) };
}

function styleFor(station, scale) {
  const value = scale.mode.value(station);
  const selected = state.selected === station.code;
  if (value === null || value === undefined) {
    return { radius: 6, color: cssVar("--muted"), weight: 1.5, fillOpacity: 0, dashArray: "2 2" };
  }
  return {
    radius: selected ? 10 : 7,
    color: selected ? cssVar("--ink") : cssVar("--marker-ring"),
    weight: selected ? 3 : 1,
    fillColor: scale.colors[binIndex(value, scale.bins)],
    fillOpacity: 0.9,
    dashArray: null,
  };
}

function renderMarkers() {
  const scale = currentScale();
  for (const { station, marker } of state.markers.values()) {
    marker.setStyle(styleFor(station, scale));
    const value = scale.mode.value(station);
    marker.setTooltipContent(`${station.name} · ${value === null || value === undefined ? "no data" : scale.mode.format(value)}`);
  }
  renderMapLegend(scale);
}

function renderMapLegend(scale) {
  const { mode, bins, colors } = scale;
  const labels =
    mode.binLabels ||
    colors.map((_, i) => {
      if (i === 0) return `< ${mode.format(bins[0])}`;
      if (i === colors.length - 1) return `≥ ${mode.format(bins[bins.length - 1])}`;
      return `${mode.format(bins[i - 1])} – ${mode.format(bins[i])}`;
    });
  let title = mode.title;
  if (state.mode === "forecast" && state.forecast) {
    title += ` · ${DAY_SHORT[state.day]} ${cellDate(state.day, state.hour)} ${formatHour(state.hour)}`;
  }
  $("map-legend").replaceChildren(
    el("strong", { text: title }),
    ...colors.map((color, i) => el("div", { class: "legend-row" }, el("span", { class: "swatch", style: `background:${color}` }), labels[i])),
    el("div", { class: "legend-row" }, el("span", { class: "swatch hollow" }), "No data"),
  );
}

function buildMarkers() {
  const bounds = [];
  for (const station of state.diag.stations) {
    const marker = L.circleMarker([station.lat, station.lon], { radius: 7 })
      .bindTooltip("", { direction: "top", offset: [0, -6] })
      .on("click", () => selectStation(station.code, { pan: false }))
      .addTo(map);
    state.markers.set(station.code, { station, marker });
    bounds.push([station.lat, station.lon]);
  }
  if (bounds.length) map.fitBounds(bounds, { padding: [24, 24] });
  $("station-list").replaceChildren(...state.diag.stations.map((s) => el("option", { value: `${s.name} (${s.code})` })));
}

function renderModes() {
  const container = $("mode");
  container.replaceChildren(
    ...Object.entries(MODES).map(([key, mode]) =>
      el("button", {
        type: "button",
        role: "radio",
        "aria-checked": String(key === state.mode),
        disabled: key === "forecast" && !state.forecast,
        title: key === "forecast" && !state.forecast ? "forecast.json not found; run `python -m velib forecast`" : null,
        text: mode.label,
        onclick: () => {
          state.mode = key;
          renderModes();
          $("forecast-controls").hidden = key !== "forecast";
          renderMarkers();
        },
      }),
    ),
  );
}

function renderDays() {
  $("days").replaceChildren(
    ...DAY_SHORT.map((label, day) =>
      el("button", {
        type: "button",
        role: "radio",
        "aria-checked": String(day === state.day),
        title: state.forecast ? state.forecast.day_dates[day] : null,
        text: label,
        onclick: () => {
          state.day = day;
          renderDays();
          renderMarkers();
        },
      }),
    ),
  );
}

// ---------------------------------------------------------------------------
// Worst-stations table
// ---------------------------------------------------------------------------

function renderWorst() {
  const { key, descending } = state.sort;
  const column = WORST_COLUMNS.find((c) => c.key === key);
  const rows = state.diag.stations
    .filter((s) => column.sort(s) !== null && column.sort(s) !== undefined)
    .sort((a, b) => {
      const va = column.sort(a);
      const vb = column.sort(b);
      const order = column.numeric ? va - vb : String(va).localeCompare(String(vb));
      return descending ? -order : order;
    })
    .slice(0, 25);

  $("worst-head").replaceChildren(
    ...WORST_COLUMNS.map((c) =>
      el(
        "th",
        { scope: "col", class: c.numeric ? "num" : "", "aria-sort": c.key === key ? (descending ? "descending" : "ascending") : "none" },
        el("button", {
          type: "button",
          text: `${c.label}${c.key === key ? (descending ? " ↓" : " ↑") : ""}`,
          onclick: () => {
            state.sort = { key: c.key, descending: c.key === key ? !descending : c.numeric && c.key !== "skill" };
            renderWorst();
          },
        }),
      ),
    ),
  );
  $("worst-body").replaceChildren(
    ...rows.map((s) =>
      el(
        "tr",
        { class: s.code === state.selected ? "is-selected" : "", tabindex: "0", onclick: () => selectStation(s.code), onkeydown: (e) => e.key === "Enter" && selectStation(s.code) },
        el("td", {}, el("span", { text: s.name }), el("span", { class: "muted code", text: ` ${s.code}` })),
        el("td", { class: "num", text: s.mae ? fmt(s.mae.model) : "–" }),
        el("td", { class: "num", text: s.skill === null ? "–" : MODES.skill.format(s.skill) }),
        el("td", { class: "num", text: MODES.coverage.format(s.coverage) }),
        el("td", { class: "num", text: String(s.capacity) }),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// Station drawer
// ---------------------------------------------------------------------------

function selectStation(code, { pan = true } = {}) {
  const entry = state.markers.get(code);
  if (!entry) return;
  state.selected = code;
  history.replaceState(null, "", `${window.location.pathname}${window.location.search}#station=${encodeURIComponent(code)}`);
  renderDrawer(entry.station);
  map.invalidateSize({ pan: false }); // the drawer just took space from the map
  if (pan) map.setView(entry.marker.getLatLng(), Math.max(map.getZoom(), 15));
  renderMarkers();
  renderWorst();
}

function closeDrawer() {
  state.selected = null;
  $("drawer").hidden = true;
  history.replaceState(null, "", window.location.pathname + window.location.search);
  renderMarkers();
  renderWorst();
  map.invalidateSize({ pan: false });
}

function renderDrawer(station) {
  $("drawer").hidden = false;
  $("drawer-title").textContent = station.name;
  $("drawer-subtitle").textContent = `Station ${station.code} · capacity ${station.capacity}`;

  const mae = station.mae || {};
  const facts = [
    ["Model MAE", fmt(mae.model)],
    ["Last week MAE", fmt(mae.seasonal_naive)],
    ["Weekly avg MAE", fmt(mae.weekly_profile)],
    ["vs baseline", station.skill === null ? "–" : MODES.skill.format(station.skill)],
    ["Mean bikes", fmt(station.mean_bikes, 1)],
    ["Coverage", MODES.coverage.format(station.coverage)],
  ];
  $("drawer-facts").replaceChildren(...facts.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));

  const series = state.diag.series[station.code];
  const container = $("holdout-chart");
  if (series) {
    const times = hourTimes(state.diag.series_start, state.diag.series_hours);
    lineChart(container, {
      labels: times.map((t) => localFormat.format(t)),
      series: [ACTUAL, ...METHODS].map((m) => ({ ...m, values: series[m.key], width: m.key === "actual" ? 1.5 : 2 })),
      ticks: dayTicks(times),
      refLines: [{ value: station.capacity, label: "capacity" }],
      height: 190,
    });
  } else {
    container.replaceChildren(el("p", { class: "muted", text: "No holdout rows for this station." }));
  }

  const forecast = state.forecastByCode.get(station.code);
  $("drawer-forecast").hidden = !forecast;
  if (forecast) {
    const times = hourTimes(state.forecast.forecast_start, 168);
    const values = times.map((t) => forecast.bikes[localDayIndex(t)][Number(localParts.format(t))]);
    lineChart($("forecast-chart"), {
      labels: times.map((t) => localFormat.format(t)),
      series: [{ ...METHODS[0], values }],
      ticks: dayTicks(times),
      refLines: [{ value: forecast.capacity, label: "capacity" }],
      height: 150,
    });
  }
}

const weekdayFormat = new Intl.DateTimeFormat("en-GB", { timeZone: TZ, weekday: "short" });
function localDayIndex(date) {
  return DAY_SHORT.indexOf(weekdayFormat.format(date));
}

function dayTicks(times) {
  const ticks = [];
  times.forEach((t, index) => {
    if (Number(localParts.format(t)) === 12) ticks.push({ index, label: weekdayFormat.format(t) });
  });
  return ticks;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

function onSearch(event) {
  if (event.type === "keydown" && event.key !== "Enter") return;
  const q = $("search").value.trim().toLowerCase();
  if (!q) return;
  const stations = state.diag.stations;
  const match =
    stations.find((s) => s.code.toLowerCase() === q) ||
    stations.find((s) => `${s.name} (${s.code})`.toLowerCase() === q) ||
    stations.find((s) => s.name.toLowerCase().includes(q));
  if (match) selectStation(match.code);
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

async function load() {
  const requested = new URLSearchParams(window.location.search).get("dir");
  const dirs = requested ? [requested.replace(/\/$/, "")] : ["data", "data/demo"];
  for (const dir of dirs) {
    try {
      const diag = await fetchJson(`${dir}/models/diagnostics.json`);
      let forecast = null;
      try {
        forecast = await fetchJson(`${dir}/forecast.json`);
      } catch (error) {
        console.warn(error);
      }
      return { dir, diag, forecast };
    } catch (error) {
      console.warn(error);
    }
  }
  throw new Error(`No diagnostics found in ${dirs.map((d) => `${d}/models/diagnostics.json`).join(" or ")}.`);
}

function showBanner(text, kind) {
  const banner = $("banner");
  banner.textContent = text;
  banner.className = `banner banner-${kind}`;
  banner.hidden = false;
}

async function main() {
  renderLegends();
  let loaded;
  try {
    loaded = await load();
  } catch (error) {
    $("run-info").textContent = "Nothing to show yet.";
    showBanner(`${error.message} Run \`python -m velib demo\` (synthetic) or \`python -m velib run\` (real data).`, "error");
    return;
  }
  Object.assign(state, loaded);
  if (state.forecast) {
    for (const s of state.forecast.stations) state.forecastByCode.set(s.code, s);
  }

  const trainedUntil = state.forecast?.model?.trained_until;
  $("run-info").textContent = `${state.dir}/ · ${state.diag.stations.length} stations${trainedUntil ? ` · trained until ${trainedUntil.slice(0, 16)}` : ""}`;
  if (state.forecast?.synthetic) {
    showBanner("Synthetic demo data: useful for checking the tooling, not for judging the model.", "warn");
  }

  $("hour").addEventListener("input", (event) => {
    state.hour = Number(event.target.value);
    $("hour-label").textContent = formatHour(state.hour);
    renderMarkers();
  });
  $("search").addEventListener("change", onSearch);
  $("search").addEventListener("keydown", onSearch);
  $("drawer-close").addEventListener("click", closeDrawer);

  const requestedMode = new URLSearchParams(window.location.search).get("mode");
  if (MODES[requestedMode] && (requestedMode !== "forecast" || state.forecast)) state.mode = requestedMode;
  $("forecast-controls").hidden = state.mode !== "forecast";

  renderOverview();
  renderErrorCharts();
  renderImportance();
  renderDataFacts();
  renderModes();
  renderDays();
  buildMarkers();
  renderMarkers();
  renderWorst();

  const linked = new URLSearchParams(window.location.hash.slice(1)).get("station");
  if (linked) selectStation(linked);

  // Charts size to their container; redraw them when the layout changes.
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      renderErrorCharts();
      if (state.selected) renderDrawer(state.markers.get(state.selected).station);
    }, 150);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderMarkers);
}

main();
