import { mountLineup } from "./lineup.js";

const root = document.getElementById("lineup-root");
const waiters = new Map();
let app = null;
let ws = null;
let wsReady = false;
let reconnectTimer = 0;
let pollTimer = 0;
let closed = false;

function pruneWaiters() {
  const newest = Math.max(0, ...waiters.keys());
  for (const seq of waiters.keys()) {
    if (seq < newest - 40) waiters.delete(seq);
  }
}

function request(payload, { onResult, onError }) {
  if (!wsReady || !ws) return null;
  const seq = payload.seq;
  const waiter = { onResult, onError, remaining: payload.models.length };
  waiters.set(seq, waiter);
  pruneWaiters();
  ws.send(JSON.stringify(payload));
  return () => waiters.delete(seq);
}

async function preview(payload) {
  const response = await fetch("/api/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`Example request failed (HTTP ${response.status})`);
  return response.json();
}

async function refreshModels() {
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (app) app.setModels(data.models);
    if (data.models.some((model) => model.status === "loading")) {
      clearTimeout(pollTimer);
      pollTimer = setTimeout(refreshModels, 900);
    }
  } catch (error) {
    if (app) app.setStatus("Can't reach the server. Is it running?");
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refreshModels, 2500);
  }
}

async function loadModel(key) {
  const response = await fetch(`/api/models/${key}/load`, { method: "POST" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refreshModels, 400);
}

function connect() {
  if (closed) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  try {
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch (error) {
    if (app) app.setStatus("Can't open a live connection.");
    return;
  }
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    wsReady = true;
    if (app) {
      app.setStatus("Connected. Forecasts stream in as each model finishes.");
      app.requestForecast(true);
    }
  };
  ws.onclose = () => {
    wsReady = false;
    waiters.clear();
    if (closed) return;
    if (app) app.setStatus("Live connection closed. Reconnecting…");
    if (!reconnectTimer) reconnectTimer = setTimeout(() => { reconnectTimer = 0; connect(); }, 1200);
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refreshModels, 900);
  };
  ws.onerror = () => { try { ws.close(); } catch (error) {} };
  ws.onmessage = (event) => {
    try {
      const text = typeof event.data === "string" ? event.data : new TextDecoder().decode(event.data);
      const message = JSON.parse(text);
      if (message.type !== "forecast" && message.type !== "error") return;
      const waiter = waiters.get(message.seq);
      if (!waiter) return;
      if (message.type === "forecast") waiter.onResult(message);
      else waiter.onError(message);
      waiter.remaining -= 1;
      if (waiter.remaining <= 0) waiters.delete(message.seq);
    } catch (error) {}
  };
}

const host = {
  page: true,
  liveUpdates: true,
  preview,
  useHash: true,
  models: [],
  loadModel,
  request,
  destroy() {
    closed = true;
    clearTimeout(reconnectTimer);
    clearTimeout(pollTimer);
    reconnectTimer = 0;
    pollTimer = 0;
    waiters.clear();
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      try { ws.close(); } catch (error) {}
      ws = null;
    }
    wsReady = false;
  },
};

app = mountLineup(root, host);
connect();
refreshModels();
