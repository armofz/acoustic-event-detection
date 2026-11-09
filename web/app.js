// Page of the MVP
const EVENTS = {
  alarm: { name: "Alarm", icon: "⏰" },
  door_knock_bell: { name: "Door (knock or bell)", icon: "🚪" },
  glass_breaking: { name: "Glass breaking", icon: "💥" },
  phone: { name: "Phone", icon: "📞" },
  siren: { name: "Siren", icon: "🚨" },
};
const $ = (id) => document.getElementById(id);
const bandView = new URLSearchParams(location.search).get("view") === "band";
let socket, detectorStatus = {}, playNextSimulation = false;

// Connection with the detector
function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.onmessage = (message) => {
    const data = JSON.parse(message.data);
    if (data.type === "status") showStatus(data);
    if (data.type === "frame") showFrame(data);
    if (data.type === "alert") showAlert(data);
    if (data.type === "simulation") playSimulation(data);
    if (data.type === "error") alert(data.message);
  };
  socket.onclose = () => setTimeout(connect, 1000);
}
const send = (command) => socket.send(JSON.stringify(command));

// Status and settings
function showStatus(data) {
  detectorStatus = data;
  $("listen").textContent = data.listening ? "Stop listening" : "Start listening";
  $("listen").classList.toggle("on", data.listening);
  $("model").innerHTML = `<option value="">Choose a model…</option>` +
    data.models.map((m) => `<option ${m === data.model ? "selected" : ""}>${m}</option>`).join("");
  $("threshold").value = data.threshold;
  $("threshold-value").textContent = `${Math.round(data.threshold * 100)} %`;
  $("events").innerHTML = Object.keys(data.patterns).map((event) => `
    <li>
      <label class="check"><input type="checkbox" value="${event}" ${data.events.includes(event) ? "checked" : ""}>
        ${EVENTS[event].icon} ${EVENTS[event].name}</label>
      ${patternBar(data.patterns[event])}
      <button data-test="${event}">Test</button>
      <button data-simulate="${event}" ${!data.model || data.simulating ? "disabled" : ""}>Simulate</button>
    </li>`).join("");
}

function patternBar(pattern) {
  // Vibration in black, pauses in white.
  return `<div class="pattern">${pattern.map((ms, i) =>
    `<span class="${i % 2 === 0 ? "on" : ""}" style="width:${(100 * ms) / 3000}%"></span>`).join("")}</div>`;
}

function sendSettings() {
  const events = [...document.querySelectorAll("#events input:checked")].map((box) => box.value);
  send({ type: "settings", events, threshold: Number($("threshold").value) });
}

$("listen").onclick = () => send({ type: "listen", on: !detectorStatus.listening });
$("model").onchange = () => { if ($("model").value) send({ type: "model", name: $("model").value }); };
$("threshold").oninput = () => { $("threshold-value").textContent = `${Math.round($("threshold").value * 100)} %`; };
$("threshold").onchange = sendSettings;
$("events").onchange = sendSettings;
$("events").onclick = (click) => {
  const { test, simulate } = click.target.dataset;
  if (test) buzz(detectorStatus.patterns[test], { event: test });
  if (simulate) { playNextSimulation = true; send({ type: "simulate", event: simulate }); }
};

// What it hears
function showFrame(data) {
  drawSpectrogram(data.spectrogram);
  const level = Math.max(0, Math.min(1, (data.level_db + 80) / 80));    // -80 dB .. 0 dB
  $("level").style.width = `${100 * level}%`;
  $("level-db").textContent = `${Math.round(data.level_db)} dB`;
  if (data.probabilities) showProbabilities(data.probabilities);
}

function drawSpectrogram(columns) {
  // Scroll to the left and draw the new 0.5 s on the right: low frequencies at the bottom
  const canvas = $("spectrogram"), context = canvas.getContext("2d");
  const width = columns.length - 1, bandHeight = canvas.height / columns[0].length;
  context.drawImage(canvas, -width, 0);
  columns.slice(0, width).forEach((column, x) => column.forEach((db, band) => {
    context.fillStyle = color((db + 80) / 100);    // -80 dB .. +20 dB
    context.fillRect(canvas.width - width + x, canvas.height - (band + 1) * bandHeight, 1, bandHeight);
  }));
}

function color(value) {
  // Dark blue (quiet) -> purple -> orange -> yellow (loud)
  const v = Math.max(0, Math.min(1, value));
  const stops = [[0, 0, 30], [90, 20, 120], [230, 90, 40], [255, 240, 120]];
  const i = Math.min(2, Math.floor(v * 3)), t = v * 3 - i;
  const [a, b] = [stops[i], stops[i + 1]];
  return `rgb(${a.map((c, k) => Math.round(c + t * (b[k] - c))).join(",")})`;
}

function showProbabilities(probabilities) {
  const top = Object.keys(probabilities).reduce((a, b) => (probabilities[a] > probabilities[b] ? a : b));
  $("probabilities").innerHTML = Object.entries(probabilities).map(([event, p]) => `
    <div class="probability ${event === top ? "top" : ""}">
      <span>${EVENTS[event].icon} ${EVENTS[event].name}</span>
      <div class="bar"><div style="width:${100 * p}%"></div><span class="threshold-line" style="left:${100 * detectorStatus.threshold}%"></span></div>
      <span>${Math.round(100 * p)} %</span>
    </div>`).join("");
}

// Alerts and wristband
function showAlert(alert) {
  buzz(alert.pattern, alert);
  if (bandView) return;
  const item = document.createElement("li");
  item.textContent = `${alert.time}  ${EVENTS[alert.event].icon} ${EVENTS[alert.event].name} (${Math.round(100 * alert.confidence)} %)`;
  $("history").prepend(item);
}

function buzz(pattern, alert) {
  // Shows the event on the wristband and shakes it with the same pattern the real one would vibrate
  for (const screen of [$("band-screen"), $("band-screen-big")]) {
    const confidence = alert.confidence ? `<span class="confidence">${Math.round(100 * alert.confidence)} %</span>` : "";
    screen.innerHTML = `<span class="icon">${EVENTS[alert.event].icon}</span><span class="name">${EVENTS[alert.event].name}</span>${confidence}`;
    const band = screen.parentElement;
    band.classList.add("alerting");
    let start = 0;
    pattern.forEach((ms, i) => {
      if (i % 2 === 0) {
        setTimeout(() => band.classList.add("buzz"), start);
        setTimeout(() => band.classList.remove("buzz"), start + ms);
      }
      start += ms;
    });
    setTimeout(() => band.classList.remove("alerting"), Math.max(start, 3000));
  }
  if ((bandView || $("vibrate").checked) && navigator.vibrate) navigator.vibrate(pattern);    // Android only
}

function playSimulation(data) {
  // Only the page that asked for the simulation plays the sound, after the 1 s of silence of the detector
  if (!playNextSimulation) return;
  playNextSimulation = false;
  setTimeout(() => new Audio(`/clip/${encodeURIComponent(data.clip_id)}`).play(), 1000);
}

// Wristband view (phone)
if (bandView) {
  $("app").hidden = true;
  $("band-view").hidden = false;
  // Browsers only let a page vibrate after the user has touched it
  $("enable-vibration").onclick = () => {
    $("enable-vibration").textContent = navigator.vibrate ? "Vibration on" : "This browser cannot vibrate (iPhone)";
    if (navigator.vibrate) navigator.vibrate(100);
  };
}
connect();