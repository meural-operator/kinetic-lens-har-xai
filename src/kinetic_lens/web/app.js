"use strict";
const $ = id => document.getElementById(id);
const descriptions = {
  spatial: "Where is class-associated evidence? Coarse Grad-CAM maps need perturbation checks before a localization claim.",
  temporal: "What changes when sequence order or individual segments change? The replacement reference affects the answer.",
  concept: "Is action-family information available in pooled features, and how does deleting its direction affect the head? Bundled concepts are label-derived proxies.",
  contrastive: "Why this class rather than a specific rival? Exact contributions explain the affine head, not the entire backbone.",
  mechanistic: "What happens when candidate internal channels are removed or retained? Bundled training rankings cover Drink, Two hand wave and Forward Kick."
};
let model, stream, ws, timer, lastResult, lastExplanation, lastSnapshotAge, explaining = false,
  starting = false,
  sessionGeneration = 0;
const capture = document.createElement("canvas"),
  ctx = capture.getContext("2d");

function text(id, value) {
  $(id).textContent = value;
}

function error(message) {
  text("error", message);
}

function node(tag, value) {
  const n = document.createElement(tag);
  if (value !== undefined) n.textContent = value;
  return n;
}

function options(select, labels) {
  labels.forEach((label, i) => {
    const o = node("option", label);
    o.value = i;
    select.append(o);
  });
}

function resetResults() {
  lastResult = lastExplanation = undefined;
  text("winner", "Collect a video window");
  $("scores").replaceChildren();
  $("explanation").replaceChildren();
  $("raw-result").hidden = true;
  $("download").disabled = true;
  text("latency", "—");
  text("age", "—");
  text("sample-rate", "—");
}
async function load() {
  try {
    const response = await fetch("/api/model");
    if (!response.ok) throw new Error("Model information unavailable");
    model = await response.json();
    text("model-name", model.name);
    text("input-policy", `${model.fps} fps · ${model.window_seconds}s live window · max ${model.max_frames} samples`);
    text("runtime", model.provenance.devices?.join(", ") || model.provenance.providers?.join(", ") || model.provenance.runtime);
    text("provenance", JSON.stringify(model.provenance, null, 2));
    text("connection", "Model ready");
    $("connection").className = "pill good";
    options($("target"), model.labels);
    options($("rival"), model.labels);
    model.labels.forEach(label => $("classes").append(node("span", label)));
    model.methods.forEach(m => {
      const o = node("option", m.id + (m.available ? "" : " — unsupported"));
      o.value = m.id;
      o.disabled = !m.available;
      $("method").append(o);
    });
    const first = model.methods.find(m => m.available);
    if (first) $("method").value = first.id;
    $("method").dispatchEvent(new Event("change"));
    $("demo-warning").hidden = model.provenance.trained !== false;
    $("start").disabled = false;
  } catch (e) {
    error(e.message);
    text("connection", "Load failed");
  }
}
$("method").onchange = () => text("method-description", descriptions[$("method").value] || "");
async function start() {
  if (starting || stream) return;
  starting = true;
  const generation = ++sessionGeneration;
  $("start").disabled = true;
  $("stop").disabled = false;
  error("");
  resetResults();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires localhost or HTTPS and a supported browser.");
    const requested = navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: $("camera").value ? {
          exact: $("camera").value
        } : undefined,
        width: {
          ideal: 640
        },
        height: {
          ideal: 480
        }
      },
      audio: false
    });
    requested.then(media => {
      if (generation !== sessionGeneration) media.getTracks().forEach(track => track.stop());
    }, () => {});
    const media = await Promise.race([requested, new Promise((_, reject) => setTimeout(() => reject(new Error("Camera permission or startup timed out. Allow camera access for this page and retry.")), 20000))]);
    if (generation !== sessionGeneration) {
      media.getTracks().forEach(track => track.stop());
      return;
    }
    stream = media;
    $("video").srcObject = stream;
    await $("video").play();
    if (generation !== sessionGeneration) return;
    $("placeholder").hidden = true;
    $("camera").disabled = true;
    $("stop").disabled = false;
    const devices = await navigator.mediaDevices.enumerateDevices();
    if (generation !== sessionGeneration) return;
    const selected = media.getVideoTracks()[0].getSettings().deviceId;
    $("camera").replaceChildren();
    devices.filter(d => d.kind === "videoinput").forEach(d => {
      const o = node("option", d.label || "Camera");
      o.value = d.deviceId;
      $("camera").append(o);
    });
    $("camera").value = selected;
    const socket = new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/live`);
    ws = socket;
    socket.onopen = () => {
      if (generation !== sessionGeneration) { socket.close(); return; }
      text("connection", "Camera connected");
      text("busy", "Collecting");
      timer = setInterval(sendFrame, 1000 / model.fps);
    };
    socket.onmessage = e => {
      if (generation !== sessionGeneration) return;
      try {
        receive(JSON.parse(e.data));
      } catch (err) {
        error(err.message);
      }
    };
    socket.onerror = () => { if (generation === sessionGeneration) error("Camera connection failed. Check that the local server is running."); };
    socket.onclose = () => { if (generation === sessionGeneration) stop("Camera disconnected"); };
    media.getVideoTracks()[0].onended = () => { if (generation === sessionGeneration) stop("Camera access ended"); };
  } catch (e) {
    if (generation !== sessionGeneration) return;
    stop();
    error(e.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access for this local page, then retry." : e.message);
  } finally {
    if (generation === sessionGeneration) {
      starting = false;
      if (!stream) $("start").disabled = false;
    }
  }
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || $("video").readyState < 2) return;
  if (ws.bufferedAmount > 600000) {
    text("busy", "Dropping a delayed frame");
    return;
  }
  const video = $("video");
  const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
  capture.width = Math.max(8, Math.round(video.videoWidth * scale));
  capture.height = Math.max(8, Math.round(video.videoHeight * scale));
  ctx.drawImage(video, 0, 0, capture.width, capture.height);
  ws.send(JSON.stringify({
    type: "frame",
    timestamp_ms: performance.now(),
    jpeg: capture.toDataURL("image/jpeg", .78).split(",")[1]
  }));
}

function stop(message = "Camera stopped") {
  sessionGeneration++;
  starting = false;
  clearInterval(timer);
  timer = undefined;
  const socket = ws;
  ws = undefined;
  if (socket && socket.readyState < 2) socket.close();
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = undefined;
  $("video").srcObject = null;
  $("placeholder").hidden = false;
  $("camera").disabled = false;
  $("stop").disabled = true;
  $("start").disabled = !model;
  $("explain").disabled = true;
  explaining = false;
  text("connection", message);
  $("connection").className = "pill";
  text("busy", "Idle");
  text("buffer", "Temporal context cleared");
  text("freshness", "Stopped");
  $("freshness").className = "pill stale";
}

function receive(data) {
  if (data.type === "error") {
    error(data.message);
    explaining = false;
    text("lab-status", "Explanation or inference failed; see the message above.");
    return;
  }
  if (data.type === "buffer") {
    text("buffer", `${data.count}/${data.required} sampled frames · skipped windows ${data.dropped_windows}`);
    text("busy", explaining ? "Explaining" : data.busy ? "Processing" : "Collecting");
  }
  if (data.type === "prediction") {
    lastResult = data;
    error("");
    text("winner", data.label);
    text("latency", `${data.processing_ms.toFixed(0)} ms`);
    text("sample-rate", `${data.effective_fps.toFixed(1)} fps`);
    $("scores").replaceChildren();
    data.top3.forEach(p => {
      const row = node("div");
      row.className = "score-row";
      const title = node("div");
      title.append(node("span", p.label), node("span", `${(100*p.score).toFixed(1)}%`));
      const bar = node("div");
      bar.className = "bar";
      const fill = node("span");
      fill.style.width = `${100*p.score}%`;
      bar.append(fill);
      row.append(title, bar);
      $("scores").append(row);
    });
    text("busy", "Collecting");
  }
  if (data.type === "explanation") {
    lastExplanation = data;
    explaining = false;
    lastSnapshotAge = (performance.now() - data.window_end_ms) / 1000;
    renderExplanation(data.result);
    text("lab-status", `Explained a completed window ending ${lastSnapshotAge.toFixed(1)}s ago · ${data.processing_ms.toFixed(0)} ms processing. These results do not update with later camera frames.`);
    $("download").disabled = false;
  }
}

function table(headers, rows) {
  const t = node("table");
  t.className = "explanation-table";
  const head = node("tr");
  headers.forEach(x => head.append(node("th", x)));
  t.append(head);
  rows.forEach(row => {
    const tr = node("tr");
    row.forEach(x => tr.append(node("td", typeof x === "number" ? x.toFixed(4) : x)));
    t.append(tr);
  });
  return t;
}
async function renderExplanation(r) {
  const out = $("explanation");
  out.replaceChildren();
  out.append(node("p", `Target: ${model.labels[r.target]}${r.data.rival!==undefined&&r.data.rival!==null?` versus ${model.labels[r.data.rival]}`:""}`));
  if (r.method === "temporal") out.append(table(["Intervention / reference", "Target logit drop", "New prediction"], r.data.measurements.map(x => [x.intervention + (x.bin !== undefined ? ` bin ${x.bin+1} / ${x.reference}` : ""), x.target_logit_drop, model.labels[x.predicted_class]])));
  if (r.method === "contrastive") {
    out.append(node("p", `Class margin ${r.data.margin.toFixed(3)} · reconstruction error ${r.data.reconstruction_error.toExponential(2)} · actual winner ${model.labels[r.data.actual_winner]}`));
    out.append(table(["Feature coordinate", "Signed contribution"], r.data.largest_positive_indices.slice(0, 10).map(i => [String(i), r.data.contributions[i]])));
  }
  if (r.method === "concept") out.append(table(["Proxy concept", "Projection", "Directional derivative", "Logit drop"], r.data.concepts.map(x => [x.name, x.feature_projection, x.target_directional_derivative, x.target_logit_drop])));
  if (r.method === "mechanistic") out.append(table(["Ranking / operation", "Target margin drop", "New prediction"], r.data.interventions.map(x => [`${x.ranking} / ${x.operation}`, x.target_margin_drop, model.labels[x.prediction]])));
  if (r.method === "spatial") {
    const grid = node("div");
    grid.className = "map-grid";
    out.append(grid);
    const n = r.data.maps.length;
    const chosen = [...new Set([0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1])];
    for (const i of chosen) {
      const canvas = node("canvas");
      canvas.width = canvas.height = 192;
      grid.append(canvas);
      const image = new Image();
      image.src = "data:image/jpeg;base64," + r.thumbnails[i];
      await image.decode();
      const c = canvas.getContext("2d");
      c.fillStyle = "#14243b";
      c.fillRect(0, 0, 192, 192);
      const s = Math.min(192 / image.width, 192 / image.height),
        w = image.width * s,
        h = image.height * s,
        x = (192 - w) / 2,
        y = (192 - h) / 2;
      c.drawImage(image, x, y, w, h);
      const map = r.data.maps[i];
      for (let row = 0; row < map.length; row++)
        for (let col = 0; col < map[row].length; col++) {
          const value = map[row][col] / (r.data.display_scale || 1);
          c.fillStyle = value >= 0 ? `rgba(245,72,53,${Math.min(.65,Math.abs(value)*.65)})` : `rgba(48,124,241,${Math.min(.65,Math.abs(value)*.65)})`;
          c.fillRect(col * 192 / map[row].length, row * 192 / map.length, 192 / map[row].length, 192 / map.length);
        }
      c.fillStyle = "#fff";
      c.font = "11px sans-serif";
      c.fillText(`sample ${i+1}/${n}`, 8, 184);
    }
  }
  const note = node("div");
  note.className = "explanation-note";
  r.limitations.forEach(x => note.append(node("p", x)));
  out.append(note);
  const copy = {
    ...r
  };
  delete copy.thumbnails;
  text("raw", JSON.stringify(copy, null, 2));
  $("raw-result").hidden = false;
}
$("explain").onclick = () => {
  if (!ws || ws.readyState !== 1 || !lastResult) return;
  const opts = {};
  if ($("target").value !== "") opts.target = Number($("target").value);
  if ($("rival").value !== "" && ["contrastive", "spatial"].includes($("method").value)) opts.rival = Number($("rival").value);
  explaining = true;
  $("explain").disabled = true;
  text("lab-status", "Computing explanation for the last completed window…");
  ws.send(JSON.stringify({
    type: "explain",
    method: $("method").value,
    options: opts
  }));
};
$("download").onclick = () => {
  if (!lastExplanation) return;
  const result = JSON.parse(JSON.stringify(lastExplanation));
  delete result.result.thumbnails;
  result.model = model.provenance;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json"
  }));
  const link = node("a");
  link.href = url;
  link.download = `kinetic-lens-${result.result.method}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("start").onclick = start;
$("stop").onclick = () => stop();
window.addEventListener("pagehide", () => stop());
setInterval(() => {
  const age = lastResult ? (performance.now() - lastResult.window_end_ms) / 1000 : Infinity;
  const fresh = !!stream && age < Math.max(3, model?.update_seconds * 3);
  text("age", Number.isFinite(age) ? `${Math.max(0,age).toFixed(1)} s` : "—");
  text("freshness", fresh ? "Fresh" : lastResult ? "Stale" : "Waiting");
  $("freshness").className = "pill " + (fresh ? "good" : "stale");
  $("explain").disabled = !fresh || explaining || !ws || !$("method").selectedOptions[0] || $("method").selectedOptions[0].disabled;
}, 400);
load();
