"use strict";

// ------------------------------------------------------------------ constants
const STEPS = 160;          // one value per step across the screen
const MIN_POINTS = 12;      // below this a forecast is mostly fan
const MAX_CONTEXT = 160;    // what the models read: the last 160 values
const MAX_H = 160;          // fallback until a ready model reports its native cap
const TWEEN_MS = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 220;
const MAX_DPR = 2;
const PAPER = "#f7f4ed";
const INK = "#16181c";
const RED = "#c92a2a";

const HINT = {
  draw: "Draw a line from left to right. The models forecast the rest.",
};

// ------------------------------------------------------------------ dom & state
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const hintEl = document.getElementById("hint");
const statusEl = document.getElementById("status");
const liveEl = document.getElementById("live");
const rowsEl = document.getElementById("rows");
const countEl = document.getElementById("count");
const subEl = document.getElementById("sub");
const railFoot = document.getElementById("railFoot");
const legendPanel = document.getElementById("legendPanel");

let models = [];
let W = 0, H = 0, dx = 1, railW = 300, raf = 0, touched = false;
let plot = { x0: 0, x1: 0, y0: 0, y1: 0 };
let example = null, ws = null, wsReady = false, reconnectTimer = 0, pollTimer = 0, exampleTimer = 0;
let strokeSeq = 0;

const S = {
  view: "overlay",       // overlay | lanes
  stroke: null,
  fc: {},                // key -> { median, from, t0, at, ms, seq }
  lastSeq: {},           // per model, the newest answer applied
  errors: {},            // per model, newest forecast/load error text
  off: new Set(),         // models the user switched off (or has not loaded)
  want: new Set(),        // models the user asked the server to load
  initialized: false,
  focus: null,
  seq: 0,
  generation: 0,          // invalidates replies from an older stroke/toggle/preset
  sent: new Map(),       // seq -> { id, t0, generation }
  lastSend: 0,
  pointer: null,
  modelSig: "",
};

// ------------------------------------------------------------------ small helpers
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const xOf = (k) => plot.x0 + k * dx;
const yOf = (v) => plot.y1 - v * (plot.y1 - plot.y0);
const ease = (t) => 1 - Math.pow(1 - t, 3);
const modelOf = (key) => models.find((m) => m.key === key);

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function setHint(text) {
  hintEl.textContent = text;
}

function setStatus(html) {
  statusEl.innerHTML = html;
}

function announce(text) {
  liveEl.textContent = text;
}

function invalidate() {
  if (!raf) raf = requestAnimationFrame(frame);
}

function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ layout
function layout() {
  const mobile = innerWidth <= 760;
  railW = mobile ? 0 : 300;
  const dpr = Math.min(devicePixelRatio || 1, MAX_DPR);
  W = innerWidth; H = innerHeight;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const top = mobile ? 128 : 118;
  const bottom = mobile ? 210 : 66;
  plot = { x0: 14, x1: W - railW - 14, y0: top, y1: H - bottom };
  dx = (plot.x1 - plot.x0) / STEPS;
}

// ------------------------------------------------------------------ strokes
function newStroke() {
  return { id: ++strokeSeq, raw: [], ys: [], k0: 0, last: null, done: false };
}

// Only rightward motion extends a stroke; each crossed step gets a linearly interpolated value.
function addPoint(s, x, y) {
  const k = (x - plot.x0) / dx;
  const v = clamp((plot.y1 - y) / (plot.y1 - plot.y0), 0, 1);
  if (!s.last) {
    s.k0 = clamp(Math.round(k), 0, STEPS - 1);
    s.ys = [v];
    s.last = { k, v };
    s.raw.push([x, y]);
    return;
  }
  if (k <= s.last.k) return;
  const first = s.k0 + s.ys.length;
  for (let j = first; j <= k; j++) {
    s.ys.push(clamp(s.last.v + (v - s.last.v) * ((j - s.last.k) / (k - s.last.k)), 0, 1));
  }
  s.last = { k, v };
  s.raw.push([x, y]);
}

const tipK = (s) => s.k0 + s.ys.length - 1;

function strokeFromValues(ys, k0) {
  const s = newStroke();
  s.k0 = k0;
  s.ys = ys.slice();
  s.done = true;
  s.raw = ys.map((v, i) => [xOf(k0 + i), yOf(v)]);
  return s;
}

const PRESETS = {
  sine: (k, r) => 0.56 + 0.15 * Math.sin(((k - 8) / 30) * Math.PI * 2) + 0.018 * Math.sin((k - 8) / 6.5) + 0.004 * (r() - 0.5),
  trend: (k, r) => 0.3 + 0.0042 * (k - 8) + 0.012 * (r() - 0.5),
  step: (k, r) => 0.36 + (k > 52 ? 0.26 : 0) + 0.01 * (r() - 0.5),
  wiggle: (k, r) => 0.5 + 0.1 * Math.sin((k - 8) / 9) + 0.05 * Math.sin((k - 8) / 3.3) + 0.006 * (r() - 0.5),
};

function presetStroke(kind, k0 = 8, k1 = 84) {
  const r = rng(kind.length * 97 + 13);
  const fn = PRESETS[kind] || PRESETS.sine;
  const ys = [];
  for (let k = k0; k <= k1; k++) ys.push(clamp(fn(k, r), 0.03, 0.97));
  return strokeFromValues(ys, k0);
}

function exampleStroke() {
  const r = rng(7);
  const ys = [];
  for (let k = 10; k <= 78; k++) {
    ys.push(clamp(0.57 + 0.14 * Math.sin(((k - 10) / 27) * Math.PI * 2) + 0.004 * (r() - 0.5), 0.05, 0.95));
  }
  return strokeFromValues(ys, 10);
}

// ------------------------------------------------------------------ model list
async function loadModels() {
  try {
    const res = await fetch("/api/models", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    models = data.models;

    if (!S.initialized) {
      S.initialized = true;
      for (const m of models) if (!m.default_on) S.off.add(m.key);
    }
    updateChrome();
    for (const m of models) {
      if (m.status === "ready" && S.want.has(m.key)) {
        S.want.delete(m.key);
        S.off.delete(m.key);
      }
      if (m.status === "error" && S.want.has(m.key)) S.want.delete(m.key);
    }

    const sig = data.models.map((m) => `${m.key}:${m.status}:${Math.round((m.load_ms || 0) / 100) / 10}`).join("|");
    if (sig !== S.modelSig) {
      S.modelSig = sig;
      buildRows();
      updateCount();
      scheduleExample();
      if (S.stroke && touched) requestForecast(true);
    }
    if (data.models.some((m) => m.status === "loading") || S.want.size) {
      clearTimeout(pollTimer);
      pollTimer = setTimeout(loadModels, 900);
    }
  } catch (e) {
    setStatus("Can't reach the server. Is it running?");
    clearTimeout(pollTimer);
    pollTimer = setTimeout(loadModels, 2500);
  }
}

function statusText(m) {
  if (m.status === "ready") return "ready";
  if (m.status === "loading") return "loading…";
  if (m.status === "idle") return "load";
  if (m.status === "missing") return "not set up";
  return "retry";
}

function buildRows() {
  rowsEl.innerHTML = "";
  for (const m of models) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mrow";
    btn.dataset.key = m.key;
    const on = m.status === "ready" && !S.off.has(m.key);
    btn.classList.toggle("off", !on);
    btn.classList.toggle("idle", m.status === "idle");
    btn.classList.toggle("unready", m.status === "error" || m.status === "missing");
    btn.classList.toggle("ready", m.status === "ready");
    btn.classList.toggle("loading", m.status === "loading");
    btn.setAttribute("aria-pressed", String(on));
    btn.innerHTML =
      `<span class="swatch" style="--c:${m.color}"></span>` +
      `<span class="mi"><span class="mname">${m.name}</span><span class="mmeta">${m.org} · ${m.variant}</span></span>` +
      `<span class="mms">${statusText(m)}</span>`;
    btn.title = `${m.note}\n${m.url}\n\n${m.status === "idle" ? "Click to load this model." : m.status === "error" ? m.error : statusText(m)}`;
    btn.addEventListener("click", () => toggleModel(m.key));
    btn.addEventListener("mouseenter", () => { S.focus = m.key; invalidate(); });
    btn.addEventListener("mouseleave", () => { if (S.focus === m.key) { S.focus = null; invalidate(); } });
    rowsEl.appendChild(btn);
  }
  updateMsLabels();
}

function updateChrome() {
  const total = models.length;
  if (!total) return;
  const defaults = models.filter((m) => m.default_on).length;
  subEl.textContent = `${total} small time-series foundation model${total === 1 ? "" : "s"}` +
    (defaults < total ? `. Start with ${defaults}.` : ".");
  railFoot.textContent = defaults < total
    ? `${defaults} model${defaults === 1 ? " is" : "s are"} on to start. Click a greyed model to load and add it. Hover one to emphasise it.`
    : "Zero-shot median on CPU. Hover a model to emphasise it; click to switch it off.";
}

function updateCount() {
  if (!models.length) { countEl.textContent = "warming up"; return; }
  const on = models.filter((m) => m.status === "ready" && !S.off.has(m.key)).length;
  countEl.textContent = `${on} of ${models.length} on`;
}

function activeFc() {
  if (S.stroke) return S.fc;
  return example ? example.fc : S.fc;
}

function updateMsLabels() {
  const fc = activeFc();
  for (const m of models) {
    const row = rowsEl.querySelector(`[data-key="${m.key}"] .mms`);
    if (!row) continue;
    const f = fc[m.key];
    if (m.status === "idle") row.textContent = "load";
    else if (m.status !== "ready") row.textContent = statusText(m);
    else if (S.off.has(m.key)) row.textContent = "off";
    else if (S.errors[m.key]) row.textContent = "error";
    else if (f && f.ms != null) row.textContent = `${Math.round(f.ms)} ms`;
    else row.textContent = "—";
  }
}

function loadModel(key) {
  const m = modelOf(key);
  if (!m || S.want.has(key) || m.status === "loading") return;
  S.want.add(key);
  delete S.errors[key];
  setStatus(`Loading ${m.name} (${m.variant})…`);
  fetch(`/api/models/${key}/load`, { method: "POST" })
    .then((r) => { if (!r.ok) throw new Error(r.status); })
    .catch(() => {
      S.want.delete(key);
      setStatus(`Could not ask the server to load ${m.name}.`);
    });
  clearTimeout(pollTimer);
  pollTimer = setTimeout(loadModels, 500);
  const row = rowsEl.querySelector(`[data-key="${key}"]`);
  if (row) {
    row.classList.remove("idle");
    row.classList.add("loading");
    const ms = row.querySelector(".mms");
    if (ms) ms.textContent = "loading…";
  }
  announce(`Loading ${m.name}.`);
}

function toggleModel(key) {
  const m = modelOf(key);
  if (!m) return;
  if (m.status === "idle") { loadModel(key); return; }
  if (m.status === "loading") { setStatus(`${m.name} is still loading.`); return; }
  if (m.status === "error") { loadModel(key); return; }
  if (m.status !== "ready") {
    setStatus(m.error ? `${m.name}: ${m.error.split("\n")[0]}` : `${m.name} is ${statusText(m)}.`);
    return;
  }
  S.generation += 1;
  delete S.errors[key];
  if (S.off.has(key)) S.off.delete(key); else S.off.add(key);
  if (S.off.has(key)) delete S.fc[key];
  buildRows();
  updateCount();
  if (S.stroke) requestForecast(true);
  else if (!touched) scheduleExample();
  invalidate();
}

function visibleModels() {
  return models.filter((m) => m.status === "ready" && !S.off.has(m.key));
}

function nativeHorizonCap() {
  const caps = visibleModels()
    .map((m) => m.max_horizon)
    .filter((h) => Number.isInteger(h) && h > 0);
  return caps.length ? Math.min(...caps) : MAX_H;
}

function alphaFor(key) {
  if (!S.focus) return 1;
  return S.focus === key ? 1 : 0.14;
}

// ------------------------------------------------------------------ websocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  try {
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch (e) {
    setStatus("Can't open a live connection.");
    return;
  }
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    wsReady = true;
    setStatus("Connected. Forecasts stream in as each model finishes.");
    updateBestStatus();
    if (S.stroke) requestForecast(true);
  };
  ws.onclose = () => {
    wsReady = false;
    if (!reconnectTimer) reconnectTimer = setTimeout(() => { reconnectTimer = 0; connect(); }, 1200);
    clearTimeout(pollTimer);
    pollTimer = setTimeout(loadModels, 900);
  };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
  ws.onmessage = (ev) => {
    try {
      const text = typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data);
      handleMessage(JSON.parse(text));
    } catch (e) {}
  };
}

function validSent(m) {
  const sent = S.sent.get(m.seq);
  if (!sent) return null;
  if (!S.stroke || S.stroke.id !== sent.id) return null;
  if (sent.generation !== S.generation) return null;
  if (m.key && S.off.has(m.key)) return null;
  return sent;
}

function handleMessage(m) {
  if (m.type === "error") {
    const sent = m.seq == null ? null : validSent(m);
    if (m.seq != null && !sent) return;
    const model = modelOf(m.key);
    const text = String(m.error).split("\n")[0];
    if (m.key) S.errors[m.key] = text;
    setStatus(`${model ? model.name : m.key || "Server"}: ${text}`);
    updateMsLabels();
    return;
  }
  if (m.type !== "forecast") return;
  const sent = validSent(m);
  if (!sent) return;
  if (m.seq < (S.lastSeq[m.key] || 0)) return;
  if (!Array.isArray(m.median) || !m.median.every((v) => Number.isFinite(v))) return;
  S.lastSeq[m.key] = m.seq;
  delete S.errors[m.key];
  applyForecast(m.key, m, sent.t0);
}

function applyForecast(key, msg, t0) {
  const median = msg.median.slice();
  const prev = S.fc[key];
  let from = median;
  if (prev && prev.median && prev.median.length === median.length) {
    const shift = t0 - prev.t0;
    from = median.map((v, i) => {
      const j = i + shift;
      return j >= 0 && j < prev.median.length ? prev.median[j] : v;
    });
  }
  S.fc[key] = { median, from, t0, at: performance.now(), ms: msg.ms, seq: msg.seq };
  updateMsLabels();
  updateBestStatus();
  invalidate();
}

function updateBestStatus() {
  const fs = Object.entries(activeFc())
    .filter(([key, f]) => !S.off.has(key) && f.ms != null)
    .map(([, f]) => f);
  if (!fs.length) return;
  const slow = Math.max(...fs.map((f) => f.ms));
  const fast = Math.min(...fs.map((f) => f.ms));
  setStatus(`<b>${fs.length}</b> forecast${fs.length === 1 ? "" : "s"} · ${Math.round(fast)}–${Math.round(slow)} ms · all local`);
}

function requestForecast(final = false) {
  const s = S.stroke;
  if (!s || s.ys.length < MIN_POINTS || !wsReady) return 0;
  const keys = visibleModels().map((m) => m.key);
  if (!keys.length) return 0;
  const now = performance.now();
  if (!final && now - S.lastSend < 85) return 0;
  const t0 = tipK(s);
  const h = Math.min(nativeHorizonCap(), STEPS - 1 - t0);
  if (h < 4) return 0;
  S.lastSend = now;
  const seq = ++S.seq;
  S.sent.set(seq, { id: s.id, t0, generation: S.generation });
  for (const k of S.sent.keys()) { if (k < seq - 40) S.sent.delete(k); else break; }
  ws.send(JSON.stringify({
    type: "forecast",
    seq,
    y: s.ys.slice(-MAX_CONTEXT),
    h,
    models: keys,
  }));
  return seq;
}

// ------------------------------------------------------------------ example
function scheduleExample() {
  clearTimeout(exampleTimer);
  exampleTimer = setTimeout(showExample, 250);
}

async function showExample() {
  if (touched || !plot.x1) return;
  const stroke = exampleStroke();
  const target = { stroke, fc: {}, t0: tipK(stroke) };
  example = target;
  invalidate();
  const ready = visibleModels().map((m) => m.key);
  if (!ready.length) return;
  const h = Math.min(nativeHorizonCap(), STEPS - 1 - target.t0);
  if (h < 4) return;
  try {
    const res = await fetch("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ y: stroke.ys, h, models: ready }),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (example !== target || touched) return;
    for (const [key, msg] of Object.entries(data.results || {})) {
      if (!msg || msg.error || !Array.isArray(msg.median)) continue;
      target.fc[key] = {
        median: msg.median.slice(),
        from: msg.median.slice(),
        t0: target.t0,
        at: performance.now(),
        ms: msg.ms,
      };
    }
    updateMsLabels();
    updateBestStatus();
    invalidate();
  } catch (e) { /* the example is decoration; ignore */ }
}

// ------------------------------------------------------------------ rendering
function frame() {
  raf = 0;
  const now = performance.now();
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, plot.y0 - 18, plot.x1 + 14, plot.y1 - plot.y0 + 36);
  ctx.clip();

  if (S.view === "lanes") {
    drawLanes(now);
  } else {
    drawGrid();
    if (example && !S.stroke && !touched) drawOverlay(example.stroke, example.fc, now, 0.68);
    if (S.stroke) drawOverlay(S.stroke, S.fc, now, 1);
  }
  ctx.restore();

  const tweening =
    Object.values(S.fc).some((f) => now - f.at < TWEEN_MS) ||
    (example && Object.values(example.fc).some((f) => now - f.at < TWEEN_MS));
  if (tweening) invalidate();
}

function drawGrid() {
  const w = plot.x1 - plot.x0;
  ctx.save();
  ctx.strokeStyle = "rgba(22,24,28,.055)";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 5]);
  for (const v of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(plot.x0, Math.round(yOf(v)) + 0.5);
    ctx.lineTo(plot.x0 + w, Math.round(yOf(v)) + 0.5);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function blend(f, i, now) {
  const t = TWEEN_MS ? Math.min(1, (now - f.at) / TWEEN_MS) : 1;
  return f.from[i] + (f.median[i] - f.from[i]) * ease(t);
}

function anchorV(stroke, f) {
  const i = clamp(f.t0 - stroke.k0, 0, stroke.ys.length - 1);
  return stroke.ys[i];
}

function drawMedian(stroke, f, model, alpha, now, halo) {
  const n = f.median.length;
  ctx.beginPath();
  ctx.moveTo(xOf(f.t0), yOf(anchorV(stroke, f)));
  for (let i = 0; i < n; i++) ctx.lineTo(xOf(f.t0 + 1 + i), yOf(blend(f, i, now)));
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = halo ? 5 : 2.3;
  ctx.strokeStyle = halo ? "rgba(247,244,237,.8)" : hexA(model.color, alpha);
  ctx.stroke();
  if (!halo) {
    const dotX = xOf(f.t0), dotY = yOf(anchorV(stroke, f));
    ctx.beginPath();
    ctx.arc(dotX, dotY, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = hexA(model.color, alpha);
    ctx.fill();
  }
}

function drawStroke(s, color, width) {
  const r = s.raw;
  if (!r.length) return;
  ctx.beginPath();
  ctx.moveTo(r[0][0], r[0][1]);
  if (r.length === 1) ctx.lineTo(r[0][0] + 0.1, r[0][1]);
  for (let i = 1; i < r.length - 1; i++) {
    const mx = (r[i][0] + r[i + 1][0]) / 2, my = (r[i][1] + r[i + 1][1]) / 2;
    ctx.quadraticCurveTo(r[i][0], r[i][1], mx, my);
  }
  if (r.length > 1) ctx.lineTo(r[r.length - 1][0], r[r.length - 1][1]);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

function drawOverlay(stroke, fc, now, baseAlpha) {
  const list = models.filter((m) => m.status === "ready" && !S.off.has(m.key) && fc[m.key]);
  for (const m of list) drawMedian(stroke, fc[m.key], m, 0, now, true);
  for (const m of list) {
    drawMedian(stroke, fc[m.key], m, alphaFor(m.key) * baseAlpha, now, false);
  }
  drawStroke(stroke, `rgba(22,24,28,${0.92 * baseAlpha})`, 3);
  if (!stroke.done && stroke.raw.length) {
    const tip = stroke.raw[stroke.raw.length - 1];
    ctx.beginPath();
    ctx.arc(tip[0], tip[1], 4.2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(22,24,28,${0.92 * baseAlpha})`;
    ctx.fill();
  }
}

function drawLanes(now) {
  const list = visibleModels();
  if (!list.length) return;
  const laneH = (plot.y1 - plot.y0) / list.length;
  const stroke = S.stroke || (example && !touched ? example.stroke : null);
  const fc = S.stroke ? S.fc : (example && !touched ? example.fc : {});

  list.forEach((m, i) => {
    const top = plot.y0 + i * laneH;
    const lane = { y0: top + 9, y1: top + laneH - 11 };
    const laneY = (v) => lane.y1 - v * (lane.y1 - lane.y0);
    const f = fc[m.key];

    if (i > 0) {
      ctx.beginPath();
      ctx.moveTo(plot.x0, Math.round(top) + 0.5);
      ctx.lineTo(plot.x1, Math.round(top) + 0.5);
      ctx.strokeStyle = "rgba(22,24,28,.07)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // context, faint, so each lane reads as its own chart
    if (stroke) {
      ctx.beginPath();
      stroke.ys.forEach((v, j) => {
        const x = xOf(stroke.k0 + j), y = laneY(v);
        if (j) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.strokeStyle = "rgba(22,24,28,.24)";
      ctx.lineWidth = 1.9;
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    if (f) {
      ctx.beginPath();
      ctx.moveTo(xOf(f.t0), laneY(anchorV(stroke, f)));
      for (let j = 0; j < f.median.length; j++) ctx.lineTo(xOf(f.t0 + 1 + j), laneY(blend(f, j, now)));
      ctx.strokeStyle = hexA(m.color, S.focus && S.focus !== m.key ? 0.2 : 1);
      ctx.lineWidth = 2.2;
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    if (stroke) {
      const endV = stroke.ys[stroke.ys.length - 1];
      ctx.beginPath();
      ctx.arc(xOf(tipK(stroke)), laneY(endV), 2.6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(22,24,28,.55)";
      ctx.fill();
    }

    // label
    ctx.beginPath();
    ctx.arc(plot.x0 + 7, top + 14, 3.6, 0, Math.PI * 2);
    ctx.fillStyle = m.color;
    ctx.fill();
    label(m.name, plot.x0 + 15, top + 17.5, "600 11px Archivo, system-ui, sans-serif", "rgba(22,24,28,.9)");
    const meta = f ? `${Math.round(f.ms)} ms` : "waiting";
    label(meta, plot.x1 - 2, top + 17.5, "500 10px 'IBM Plex Sans', system-ui, sans-serif", "rgba(22,24,28,.42)", "right");
  });

  if (stroke && stroke.raw.length && !stroke.done) {
    const tipV = stroke.ys[stroke.ys.length - 1];
    ctx.beginPath();
    ctx.arc(xOf(tipK(stroke)), yOf(tipV), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = INK;
    ctx.fill();
  }
}

function label(text, x, y, font, color, align = "left") {
  ctx.font = font;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = "rgba(247,244,237,.85)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// ------------------------------------------------------------------ input
function insidePlot(x, y) {
  return x >= plot.x0 - 8 && x <= plot.x1 + 8 && y >= plot.y0 - 8 && y <= plot.y1 + 8;
}

function pointsOf(e) {
  const list = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  return list.length ? list : [e];
}

canvas.addEventListener("pointerdown", (e) => {
  if (S.pointer !== null || !e.isPrimary || (e.pointerType === "mouse" && e.button !== 0)) return;
  if (!insidePlot(e.clientX, e.clientY)) return;

  S.pointer = e.pointerId;
  try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
  touched = true;
  example = null;
  S.generation += 1;
  S.stroke = newStroke();
  S.fc = {};
  S.errors = {};
  S.sent.clear();
  S.lastSeq = {};
  S.focus = null;
  setHint(HINT.draw);
  addPoint(S.stroke, e.clientX, e.clientY);
  invalidate();
});

canvas.addEventListener("pointermove", (e) => {
  if (e.pointerId !== S.pointer || !S.stroke) return;
  for (const p of pointsOf(e)) addPoint(S.stroke, p.clientX, p.clientY);
  requestForecast(false);
  invalidate();
});

function endStroke(e) {
  if (e.pointerId !== S.pointer) return;
  S.pointer = null;
  if (!S.stroke) return;
  S.stroke.done = true;
  if (S.stroke.ys.length < MIN_POINTS) {
    setHint("Too short to forecast. Draw a longer line.");
    invalidate();
    return;
  }
  requestForecast(true);
  invalidate();
}

canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);
canvas.addEventListener("lostpointercapture", endStroke);

// ------------------------------------------------------------------ view controls
function setView(view) {
  S.view = view;
  for (const b of document.querySelectorAll(".seg button")) {
    b.setAttribute("aria-pressed", String(b.dataset.view === view));
  }
  invalidate();
}
for (const b of document.querySelectorAll(".seg button")) {
  b.addEventListener("click", () => setView(b.dataset.view));
}

for (const b of document.querySelectorAll("[data-preset]")) {
  b.addEventListener("click", () => {
    touched = true;
    example = null;
    S.generation += 1;
    S.stroke = presetStroke(b.dataset.preset);
    S.fc = {};
    S.errors = {};
    S.sent.clear();
    S.lastSeq = {};
    setHint("A synthetic line, to compare the models on something repeatable.");
    requestForecast(true);
    invalidate();
  });
}

// ------------------------------------------------------------------ keyboard
addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "g") { setView(S.view === "overlay" ? "lanes" : "overlay"); return; }
  if (k === "escape") {
    S.stroke = null;
    S.fc = {};
    S.errors = {};
    S.sent.clear();
    S.lastSeq = {};
    S.generation += 1;
    setHint(HINT.draw);
    scheduleExample();
    invalidate();
    return;
  }
  if (k === "r") {
    const keys = Object.keys(PRESETS);
    document.querySelector(`[data-preset="${keys[Math.floor(Math.random() * keys.length)]}"]`)?.click();
    return;
  }
  if (k >= "1" && k <= "5") {
    const m = models[Number(k) - 1];
    if (m) toggleModel(m.key);
  }
});

// ------------------------------------------------------------------ resize & boot
function relayoutStrokes() {
  if (S.stroke) S.stroke.raw = S.stroke.ys.map((v, i) => [xOf(S.stroke.k0 + i), yOf(v)]);
  if (example) example.stroke.raw = example.stroke.ys.map((v, i) => [xOf(example.stroke.k0 + i), yOf(v)]);
}

function onResize() {
  layout();
  relayoutStrokes();
  if (!touched && !S.stroke) scheduleExample();
  invalidate();
}

let resizeTimer = 0;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(onResize, 90);
});
if (window.ResizeObserver) {
  new ResizeObserver(() => {
    if (innerWidth !== W || innerHeight !== H) onResize();
  }).observe(canvas);
}

layout();
const hashPreset = location.hash.replace("#", "");
if (PRESETS[hashPreset]) setTimeout(() => document.querySelector(`[data-preset="${hashPreset}"]`)?.click(), 60);
connect();
loadModels();
showExample();
document.fonts.ready.then(invalidate);
setTimeout(() => { if (!models.length) loadModels(); }, 1500);
