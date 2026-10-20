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

const TEMPLATE = `
<canvas data-role="board" aria-label="Drawing area. Draw a line from left to right; each model's forecast appears to its right."></canvas>
<header class="top">
  <div class="intro">
    <h1>Line-up</h1>
    <p class="sub" data-role="sub">Small time-series foundation models.</p>
    <p data-role="hint">Draw a line from left to right. The models forecast the rest.</p>
  </div>
  <div class="controls">
    <div class="seg" role="group" aria-label="Layout">
      <button type="button" data-view="overlay" aria-pressed="true">Overlay</button>
      <button type="button" data-view="lanes" aria-pressed="false">Lanes</button>
    </div>
    <button type="button" data-role="bands" class="chip" aria-pressed="false" title="Show the central 80% native prediction interval">Bands 80%</button>
    <button type="button" data-role="copy" class="chip" disabled title="Copy the graph and model legend as a PNG image">Copy image</button>
  </div>
</header>
<aside class="rail" data-role="rail">
  <section data-role="legendPanel">
    <div class="cap"><span>The line-up</span><span data-role="count">warming up</span></div>
    <div data-role="rows" class="rows"></div>
    <div data-role="counts" class="counts" aria-live="polite">Original — · Predicted —</div>
    <p class="band-note hidden" data-role="bandNote">Shaded bands show each model's <b>80% prediction interval</b> from native 0.1–0.9 quantiles.</p>
    <p class="rail-foot" data-role="railFoot">Zero-shot median on CPU. Click a model to load or switch it off.</p>
  </section>
</aside>
<footer class="bottom">
  <div class="presets">
    <span class="plabel">Try</span>
    <button type="button" data-preset="sine">sine</button>
    <button type="button" data-preset="trend">trend</button>
    <button type="button" data-preset="step">step</button>
    <button type="button" data-preset="wiggle">wiggle</button>
  </div>
  <div data-role="status">warming up…</div>
</footer>
<div data-role="live" class="sr" aria-live="polite"></div>`;

export function mountLineup(root, host) {
  const controller = new AbortController();
  const { signal } = controller;
  let destroyed = false;

  // ------------------------------------------------------------------ dom & state
  root.classList.add("lineup-root");
  if (host.page) root.classList.add("lineup-page");
  root.tabIndex = 0;
  root.innerHTML = TEMPLATE;
  const q = (role) => root.querySelector(`[data-role="${role}"]`);
  const canvas = q("board");
  const ctx = canvas.getContext("2d");
  const hintEl = q("hint");
  const statusEl = q("status");
  const liveEl = q("live");
  const rowsEl = q("rows");
  const countEl = q("count");
  const subEl = q("sub");
  const railFoot = q("railFoot");
  const countsEl = q("counts");
  const bandsBtn = q("bands");
  const bandNote = q("bandNote");
  const copyBtn = q("copy");
  const legendPanel = q("legendPanel");
let models = [];
let W = 0, H = 0, dx = 1, railW = 300, raf = 0, touched = false;
let plot = { x0: 0, x1: 0, y0: 0, y1: 0 };
let example = null, exampleTimer = 0;
let strokeSeq = 0, copyBusy = false, copyFallback = false, copyResetTimer = 0;
const COPY_TITLE = "Copy the graph and model legend as a PNG image";

const S = {
  view: "overlay",       // overlay | lanes
  bands: false,          // request and show each model's central 80% interval
  stroke: null,
  fc: {},                // key -> { median, interval, from, t0, at, ms, seq }
  lastSeq: {},           // per model, the newest answer applied
  errors: {},            // per model, newest forecast/load error text
  off: new Set(),         // models the user switched off (or has not loaded)
  want: new Set(),        // models the user asked the server to load
  initialized: false,
  focus: null,
  seq: 0,
  generation: 0,          // invalidates replies from an older stroke/toggle/preset
  sent: new Map(),       // seq -> { id, t0, generation }
  cancels: new Map(),    // seq -> host cancel/unsubscribe
  lastSend: 0,
  pointer: null,
  noForecastSpace: false,
  modelSig: "",
};

// ------------------------------------------------------------------ small helpers
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const xOf = (k) => plot.x0 + k * dx;
const yOf = (v) => plot.y1 - v * (plot.y1 - plot.y0);
const ease = (t) => 1 - Math.pow(1 - t, 3);
const modelOf = (key) => models.find((m) => m.key === key);
const modelDisplayName = (m) => `${m.name}${m.variant ? ` ${m.variant}` : ""}`;

function parameterLabel(count) {
  if (!Number.isFinite(count) || count <= 0) return "";
  if (count >= 1e6) {
    const millions = count / 1e6;
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M parameters`;
  }
  if (count >= 1e3) {
    const thousands = count / 1e3;
    return `${Number.isInteger(thousands) ? thousands : thousands.toFixed(1)}K parameters`;
  }
  return `${count} parameters`;
}

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
  if (!destroyed && !raf) raf = requestAnimationFrame(frame);
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
  const mobile = root.clientWidth <= 760;
  root.classList.toggle("narrow", mobile);
  railW = mobile ? 0 : 300;
  const dpr = Math.min(devicePixelRatio || 1, MAX_DPR);
  W = root.clientWidth; H = root.clientHeight;
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
  const k = clamp((x - plot.x0) / dx, 0, STEPS - 1);
  const px = plot.x0 + k * dx;
  const v = clamp((plot.y1 - y) / (plot.y1 - plot.y0), 0, 1);
  if (!s.last) {
    s.k0 = clamp(Math.round(k), 0, STEPS - 1);
    s.ys = [v];
    s.last = { k, v };
    s.raw.push([px, y]);
    return;
  }
  if (k <= s.last.k) return;
  const first = s.k0 + s.ys.length;
  for (let j = first; j <= k; j++) {
    s.ys.push(clamp(s.last.v + (v - s.last.v) * ((j - s.last.k) / (k - s.last.k)), 0, 1));
  }
  s.last = { k, v };
  s.raw.push([px, y]);
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
function setModels(next) {
  models = Array.isArray(next) ? next : [];
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

  const sig = models.map((m) => `${m.key}:${m.status}:${Math.round((m.load_ms || 0) / 100) / 10}`).join("|");
  if (sig !== S.modelSig) {
    S.modelSig = sig;
    buildRows();
    updateCount();
    scheduleExample();
    if (S.stroke && touched) requestForecast(true);
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
    btn.addEventListener("click", () => toggleModel(m.key), { signal });
    btn.addEventListener("mouseenter", () => { S.focus = m.key; invalidate(); }, { signal });
    btn.addEventListener("mouseleave", () => { if (S.focus === m.key) { S.focus = null; invalidate(); } }, { signal });
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

function currentGraph() {
  const stroke = S.stroke || (example && !touched ? example.stroke : null);
  if (!stroke) return null;
  const fc = S.stroke ? S.fc : (example && !touched ? example.fc : {});
  const forecasts = visibleModels()
    .map((model) => ({ model, forecast: fc[model.key] }))
    .filter(({ forecast }) => forecast && Array.isArray(forecast.median));
  return { stroke, fc, forecasts };
}

function updateCopyAvailability() {
  const available = !!currentGraph() && !copyBusy;
  copyBtn.disabled = !available;
  if (copyBtn.textContent === "Copy image") {
    copyBtn.title = available ? COPY_TITLE : "Draw a line to copy an image.";
  }
}

function updateCounts() {
  const graph = currentGraph();
  if (!graph) {
    countsEl.textContent = "Original — · Predicted —";
    updateCopyAvailability();
    return;
  }
  const original = graph.stroke.ys.length;
  const selected = visibleModels();
  let predicted = "Predicted —";
  if (S.noForecastSpace) {
    predicted = "Predicted: 0 points · leave room on the right";
  } else if (selected.length) {
    const lengths = selected.map((m) => {
      const f = graph.fc[m.key];
      return f && Array.isArray(f.median) ? f.median.length : null;
    });
    const complete = lengths.every((length) => length != null);
    const shared = complete && lengths.every((length) => length === lengths[0]);
    if (shared) {
      predicted = `Predicted ${lengths[0]} point${lengths[0] === 1 ? "" : "s"}${selected.length > 1 ? " each" : ""}`;
    } else {
      const details = selected.map((m, i) => {
        const value = lengths[i] != null
          ? `${lengths[i]} pts`
          : (S.errors[m.key] ? "no result" : "waiting");
        return `${m.name} ${value}`;
      });
      predicted = `Predicted: ${details.join(" · ")}`;
    }
  }
  countsEl.textContent = `Original ${original} point${original === 1 ? "" : "s"} · ${predicted}`;
  updateCopyAvailability();
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
  updateCounts();
}

function loadModel(key) {
  const m = modelOf(key);
  if (!m || S.want.has(key) || m.status === "loading") return;
  S.want.add(key);
  delete S.errors[key];
  setStatus(`Loading ${m.name} (${m.variant})…`);
  Promise.resolve(host.loadModel ? host.loadModel(key) : undefined).catch(() => {
    S.want.delete(key);
    setStatus(`Could not ask the server to load ${m.name}.`);
  });
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

function updateBandChrome() {
  bandsBtn.setAttribute("aria-pressed", String(S.bands));
  bandNote.classList.toggle("hidden", !S.bands);
}

function removeIntervals() {
  for (const fc of [S.fc, example && example.fc]) {
    if (!fc) continue;
    for (const forecast of Object.values(fc)) delete forecast.interval;
  }
}

function setBands(enabled) {
  if (S.bands === enabled) return;
  S.bands = enabled;
  S.generation += 1;  // an old median-only or banded reply must not land after this switch
  if (!enabled) removeIntervals();
  updateBandChrome();
  if (S.stroke) requestForecast(true);
  else if (!touched) scheduleExample();
  updateMsLabels();
  invalidate();
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

// ------------------------------------------------------------------ transport
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
    if (m.key && m.seq != null && m.seq < (S.lastSeq[m.key] || 0)) return;
    const model = modelOf(m.key);
    const text = String(m.error).split("\n")[0];
    if (m.key) {
      S.lastSeq[m.key] = m.seq == null ? S.lastSeq[m.key] : m.seq;
      S.errors[m.key] = text;
      delete S.fc[m.key];
    }
    setStatus(`${model ? model.name : m.key || "Server"}: ${text}`);
    updateMsLabels();
    invalidate();
    return;
  }
  if (m.type !== "forecast") return;
  const sent = validSent(m);
  if (!sent) return;
  if (m.seq < (S.lastSeq[m.key] || 0)) return;
  if (!Array.isArray(m.median) || !m.median.every((v) => Number.isFinite(v))) return;
  const interval = intervalFrom(m, m.median.length);
  if (S.bands && !interval) {
    S.lastSeq[m.key] = m.seq;
    S.errors[m.key] = "Invalid 80% interval response.";
    delete S.fc[m.key];
    setStatus(`${modelOf(m.key)?.name || m.key}: invalid 80% interval response.`);
    updateMsLabels();
    invalidate();
    return;
  }
  S.lastSeq[m.key] = m.seq;
  delete S.errors[m.key];
  applyForecast(m.key, m, sent.t0, S.bands ? interval : null);
}

function intervalFrom(msg, length) {
  const interval = msg.interval;
  const levels = [0.1, 0.5, 0.9];
  if (!interval || !Array.isArray(interval.levels) || interval.levels.length !== levels.length) return null;
  if (!interval.levels.every((v, i) => Number.isFinite(v) && Math.abs(v - levels[i]) < 1e-9)) return null;
  if (!Array.isArray(interval.lower) || interval.lower.length !== length) return null;
  if (!Array.isArray(interval.upper) || interval.upper.length !== length) return null;
  if (![...interval.lower, ...interval.upper].every((v) => Number.isFinite(v))) return null;
  return {
    levels: interval.levels.slice(),
    lower: interval.lower.slice(),
    upper: interval.upper.slice(),
  };
}

function applyForecast(key, msg, t0, interval) {
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
  S.fc[key] = {
    median, interval, from, t0, at: performance.now(), ms: msg.ms, seq: msg.seq,
  };
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

function cancelRequests() {
  for (const cancel of S.cancels.values()) {
    try { cancel(); } catch (e) {}
  }
  S.cancels.clear();
  S.sent.clear();
}

function clearNoForecastSpace() {
  S.noForecastSpace = true;
  S.fc = {};
  cancelRequests();
  S.lastSeq = {};
  setHint("Leave room on the right for a forecast.");
  updateMsLabels();
  invalidate();
}

function requestForecast(final = false) {
  const s = S.stroke;
  if (!s || s.ys.length < MIN_POINTS) return 0;
  const t0 = tipK(s);
  const h = Math.min(nativeHorizonCap(), STEPS - 1 - t0);
  if (h < 4) {
    if (!S.noForecastSpace) clearNoForecastSpace();
    return 0;
  }
  S.noForecastSpace = false;
  if (!final && !host.liveUpdates) return 0;
  const keys = visibleModels().map((m) => m.key);
  if (!keys.length) return 0;
  const now = performance.now();
  if (!final && now - S.lastSend < 85) return 0;
  S.lastSend = now;
  const seq = ++S.seq;
  const requestId = `${Date.now().toString(36)}-${seq}-${Math.random().toString(36).slice(2, 8)}`;
  S.sent.set(seq, { id: s.id, t0, generation: S.generation });
  for (const k of S.sent.keys()) {
    if (k >= seq - 40) break;
    const oldCancel = S.cancels.get(k);
    if (oldCancel) {
      try { oldCancel(); } catch (e) {}
      S.cancels.delete(k);
    }
    S.sent.delete(k);
  }
  const cancel = host.request({
    type: "forecast",
    seq,
    request_id: requestId,
    y: s.ys.slice(-MAX_CONTEXT),
    h,
    models: keys,
    include_interval: S.bands,
  }, {
    onResult: (m) => handleMessage({ type: "forecast", seq, ...m }),
    onError: (m) => handleMessage({ type: "error", seq, ...m }),
  });
  if (!cancel) {
    S.sent.delete(seq);
    return 0;
  }
  if (typeof cancel === "function") S.cancels.set(seq, cancel);
  return seq;
}

// ------------------------------------------------------------------ example
function scheduleExample() {
  clearTimeout(exampleTimer);
  example = null;
  S.noForecastSpace = false;
  if (!touched && !S.stroke) {
    S.errors = {};
    invalidate();
  }
  updateCounts();
  if (host.preview !== false) exampleTimer = setTimeout(showExample, 250);
}

function setPreviewError(keys, text) {
  for (const key of keys) {
    S.errors[key] = text;
    delete S.fc[key];
  }
  updateMsLabels();
  invalidate();
}

async function showExample() {
  if (touched || !plot.x1 || host.preview === false || !host.preview) return;
  const stroke = exampleStroke();
  const target = { stroke, fc: {}, errors: {}, t0: tipK(stroke) };
  example = target;
  updateCounts();
  invalidate();
  const ready = visibleModels().map((m) => m.key);
  if (!ready.length) return;
  const h = Math.min(nativeHorizonCap(), STEPS - 1 - target.t0);
  if (h < 4) return;
  try {
    const data = await host.preview({ y: stroke.ys, h, models: ready, include_interval: S.bands });
    if (example !== target || touched) return;
    for (const [key, msg] of Object.entries(data.results || {})) {
      if (!msg) {
        target.errors[key] = "No result for this model.";
        delete target.fc[key];
        continue;
      }
      if (msg.error) {
        target.errors[key] = String(msg.error).split("\n")[0];
        delete target.fc[key];
        continue;
      }
      if (!Array.isArray(msg.median)) {
        target.errors[key] = "Invalid median response.";
        delete target.fc[key];
        continue;
      }
      const interval = intervalFrom(msg, msg.median.length);
      if (S.bands && !interval) {
        target.errors[key] = "Invalid 80% interval response.";
        delete target.fc[key];
        continue;
      }
      target.fc[key] = {
        median: msg.median.slice(),
        interval: S.bands ? interval : null,
        from: msg.median.slice(),
        t0: target.t0,
        at: performance.now(),
        ms: msg.ms,
      };
      delete target.errors[key];
    }
    S.errors = { ...target.errors };
    updateMsLabels();
    updateBestStatus();
    invalidate();
  } catch (e) {
    if (example === target && !touched) {
      setPreviewError(ready, `Example request failed: ${String(e.message || e)}`);
    }
  }
}

// ------------------------------------------------------------------ rendering
function frame() {
  raf = 0;
  if (destroyed) return;
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

  if (hasActiveTween(now)) invalidate();
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

function drawBand(stroke, f, color, alpha, toY) {
  const interval = f && f.interval;
  if (!interval) return;
  const n = interval.lower.length;
  if (!n || interval.upper.length !== n) return;
  ctx.beginPath();
  ctx.moveTo(xOf(f.t0), toY(anchorV(stroke, f)));
  for (let i = 0; i < n; i++) ctx.lineTo(xOf(f.t0 + 1 + i), toY(interval.lower[i]));
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(xOf(f.t0 + 1 + i), toY(interval.upper[i]));
  ctx.closePath();
  ctx.fillStyle = hexA(color, 0.13 * alpha);
  ctx.fill();
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
  for (const m of list) {
    drawBand(stroke, fc[m.key], m.color, alphaFor(m.key) * baseAlpha, yOf);
  }
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
      const alpha = S.focus && S.focus !== m.key ? 0.2 : 1;
      drawBand(stroke, f, m.color, alpha, laneY);
      ctx.beginPath();
      ctx.moveTo(xOf(f.t0), laneY(anchorV(stroke, f)));
      for (let j = 0; j < f.median.length; j++) ctx.lineTo(xOf(f.t0 + 1 + j), laneY(blend(f, j, now)));
      ctx.strokeStyle = hexA(m.color, alpha);
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

// ------------------------------------------------------------------ copy image
function hasActiveTween(now = performance.now()) {
  return Object.values(activeFc()).some((f) => f && f.at != null && now - f.at < TWEEN_MS);
}

async function waitForSettledGraph() {
  await document.fonts.ready;
  while (hasActiveTween()) await new Promise((resolve) => setTimeout(resolve, 30));
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function restoreCopyButton() {
  clearTimeout(copyResetTimer);
  copyResetTimer = 0;
  copyFallback = false;
  copyBtn.textContent = "Copy image";
  copyBtn.title = currentGraph() ? COPY_TITLE : "Draw a line to copy an image.";
  updateCopyAvailability();
}

function showCopyFeedback(text, title, ms) {
  copyBtn.textContent = text;
  copyBtn.title = title;
  clearTimeout(copyResetTimer);
  copyResetTimer = setTimeout(restoreCopyButton, ms);
  updateCopyAvailability();
}

function offerDownload(message) {
  copyFallback = true;
  showCopyFeedback("Download", `${message}. Click to download the PNG instead.`, 6000);
  announce(`${message}. Use Download for a PNG instead.`);
}

function downloadBlob(blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "line-up.png";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function makeExportBlob() {
  await waitForSettledGraph();
  const graph = currentGraph();
  if (!graph) throw new Error("Nothing is plotted");
  const scale = canvas.width / W;
  const cropX = Math.max(0, plot.x0 - 10);
  const cropY = Math.max(0, plot.y0 - 8);
  const cropW = Math.min(W - cropX, plot.x1 - plot.x0 + 20);
  const cropH = Math.min(H - cropY, plot.y1 - plot.y0 + 16);
  const original = graph.stroke.ys.length;
  const legend = [{ color: INK, text: `Original · ${original} points` }];
  if (graph.forecasts.some(({ forecast }) => forecast.interval)) {
    legend.push({
      color: "rgba(22,24,28,.34)",
      text: "Central 80% prediction interval · native 0.1–0.9 quantiles",
    });
  }
  for (const { model, forecast } of graph.forecasts) {
    const bits = [modelDisplayName(model)];
    const params = parameterLabel(model.parameter_count);
    if (params) bits.push(params);
    bits.push(`${forecast.median.length} forecast points`);
    legend.push({ color: model.color, text: bits.join(" · ") });
  }

  const rowH = 18;
  const topPad = 13;
  const bottomPad = 11;
  const footerH = topPad + bottomPad + legend.length * rowH;
  const outW = Math.max(1, Math.round(cropW));
  const outH = Math.max(1, Math.round(cropH + footerH));
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(outW * scale));
  out.height = Math.max(1, Math.round(outH * scale));
  const g = out.getContext("2d");
  if (!g) throw new Error("Canvas is unavailable");

  g.setTransform(scale, 0, 0, scale, 0, 0);
  g.fillStyle = PAPER;
  g.fillRect(0, 0, outW, outH);
  g.drawImage(
    canvas,
    Math.round(cropX * scale), Math.round(cropY * scale),
    Math.round(cropW * scale), Math.round(cropH * scale),
    0, 0, outW, cropH
  );
  g.strokeStyle = "rgba(22,24,28,.16)";
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(0, cropH + 0.5);
  g.lineTo(outW, cropH + 0.5);
  g.stroke();

  g.font = "500 10.5px 'IBM Plex Sans', system-ui, sans-serif";
  g.textAlign = "left";
  g.textBaseline = "middle";
  for (let i = 0; i < legend.length; i++) {
    const row = legend[i];
    const y = cropH + topPad + rowH * i + rowH / 2;
    g.strokeStyle = row.color;
    g.lineWidth = 3;
    g.lineCap = "round";
    g.beginPath();
    g.moveTo(13, y);
    g.lineTo(31, y);
    g.stroke();
    g.fillStyle = INK;
    g.fillText(row.text, 39, y);
  }

  return await new Promise((resolve, reject) => {
    out.toBlob((blob) => blob ? resolve(blob) : reject(new Error("PNG encoding failed")), "image/png");
  });
}

async function copyGraphImage() {
  if (copyBusy) return;
  if (!currentGraph()) {
    showCopyFeedback("No graph", "Draw a line first.", 1800);
    return;
  }
  copyBusy = true;
  updateCopyAvailability();
  let blob;
  try {
    blob = await makeExportBlob();
  } catch (e) {
    copyBusy = false;
    showCopyFeedback("Copy failed", `Could not render the image: ${String(e.message || e)}`, 3000);
    announce("Could not render the image.");
    updateCopyAvailability();
    return;
  }

  try {
    if (copyFallback) {
      downloadBlob(blob);
      copyFallback = false;
      showCopyFeedback("Downloaded", "PNG downloaded.", 1800);
      announce("PNG downloaded.");
      return;
    }
    if (!navigator.clipboard || typeof navigator.clipboard.write !== "function" || typeof ClipboardItem === "undefined") {
      offerDownload("Clipboard unavailable");
      return;
    }
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    copyFallback = false;
    showCopyFeedback("Copied", "PNG copied to the clipboard.", 1800);
    announce("Graph image copied to the clipboard.");
  } catch (e) {
    const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
    offerDownload(denied ? "Clipboard permission denied" : "Could not copy the image");
  } finally {
    copyBusy = false;
    updateCopyAvailability();
  }
}

// ------------------------------------------------------------------ input
function localPoint(e) {
  const rect = root.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function insidePlot(x, y) {
  return x >= plot.x0 - 8 && x <= plot.x1 + 8 && y >= plot.y0 - 8 && y <= plot.y1 + 8;
}

function pointsOf(e) {
  const list = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  return list.length ? list : [e];
}

canvas.addEventListener("pointerdown", (e) => {
  if (S.pointer !== null || !e.isPrimary || (e.pointerType === "mouse" && e.button !== 0)) return;
  const point = localPoint(e);
  if (!insidePlot(point.x, point.y)) return;

  S.pointer = e.pointerId;
  try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
  touched = true;
  example = null;
  S.generation += 1;
  S.stroke = newStroke();
  S.fc = {};
  S.errors = {};
  cancelRequests();
  S.lastSeq = {};
  S.focus = null;
  S.noForecastSpace = false;
  setHint(HINT.draw);
  root.focus({ preventScroll: true });
  addPoint(S.stroke, point.x, point.y);
  updateCounts();
  invalidate();
}, { signal });

canvas.addEventListener("pointermove", (e) => {
  if (e.pointerId !== S.pointer || !S.stroke) return;
  for (const p of pointsOf(e)) {
    const point = localPoint(p);
    addPoint(S.stroke, point.x, point.y);
  }
  requestForecast(false);
  updateCounts();
  invalidate();
}, { signal });

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
  updateCounts();
  invalidate();
}

canvas.addEventListener("pointerup", endStroke, { signal });
canvas.addEventListener("pointercancel", endStroke, { signal });
canvas.addEventListener("lostpointercapture", endStroke, { signal });

// ------------------------------------------------------------------ view controls
function setView(view) {
  S.view = view;
  for (const b of root.querySelectorAll(".seg button")) {
    b.setAttribute("aria-pressed", String(b.dataset.view === view));
  }
  invalidate();
}
for (const b of root.querySelectorAll(".seg button")) {
  b.addEventListener("click", () => setView(b.dataset.view), { signal });
}
copyBtn.addEventListener("click", copyGraphImage, { signal });
bandsBtn.addEventListener("click", () => setBands(!S.bands), { signal });

for (const b of root.querySelectorAll("[data-preset]")) {
  b.addEventListener("click", () => {
    touched = true;
    example = null;
    S.generation += 1;
    S.stroke = presetStroke(b.dataset.preset);
    S.fc = {};
    S.errors = {};
    cancelRequests();
    S.lastSeq = {};
    S.noForecastSpace = false;
    setHint("A synthetic line, to compare the models on something repeatable.");
    requestForecast(true);
    updateCounts();
    invalidate();
  }, { signal });
}

// ------------------------------------------------------------------ keyboard
root.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "g") { setView(S.view === "overlay" ? "lanes" : "overlay"); return; }
  if (k === "escape") {
    S.stroke = null;
    S.fc = {};
    S.errors = {};
    cancelRequests();
    S.lastSeq = {};
    S.noForecastSpace = false;
    S.generation += 1;
    setHint(HINT.draw);
    scheduleExample();
    invalidate();
    return;
  }
  if (k === "r") {
    const keys = Object.keys(PRESETS);
    root.querySelector(`[data-preset="${keys[Math.floor(Math.random() * keys.length)]}"]`)?.click();
    return;
  }
  if (k >= "1" && k <= "5") {
    const m = models[Number(k) - 1];
    if (m) toggleModel(m.key);
  }
}, { signal });

// ------------------------------------------------------------------ resize & boot
function relayoutStrokes() {
  if (S.stroke) S.stroke.raw = S.stroke.ys.map((v, i) => [xOf(S.stroke.k0 + i), yOf(v)]);
  if (example) example.stroke.raw = example.stroke.ys.map((v, i) => [xOf(example.stroke.k0 + i), yOf(v)]);
}

function onResize() {
  layout();
  relayoutStrokes();
  if (!touched && !S.stroke) scheduleExample();
  updateCounts();
  invalidate();
}

let resizeObserver = null;
if (typeof ResizeObserver !== "undefined") {
  resizeObserver = new ResizeObserver(() => {
    if (root.clientWidth !== W || root.clientHeight !== H) onResize();
  });
  resizeObserver.observe(root);
}

layout();
updateBandChrome();
setModels(host.models || []);
if (host.useHash) {
  const hashPreset = location.hash.replace("#", "");
  if (PRESETS[hashPreset]) {
    setTimeout(() => {
      if (!destroyed) root.querySelector(`[data-preset="${hashPreset}"]`)?.click();
    }, 60);
  }
}
scheduleExample();
document.fonts.ready.then(invalidate);

return {
  destroy() {
    if (destroyed) return;
    destroyed = true;
    controller.abort();
    if (resizeObserver) resizeObserver.disconnect();
    clearTimeout(exampleTimer);
    clearTimeout(copyResetTimer);
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    for (const cancel of S.cancels.values()) {
      try { cancel(); } catch (e) {}
    }
    S.cancels.clear();
    S.sent.clear();
    if (host.destroy) host.destroy();
  },
  setModels,
  setStatus,
  setHint,
  requestForecast,
};
}
