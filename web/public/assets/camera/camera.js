// VAR Box web camera: any phone records for the rig from a link, with every frame stamped on a
// clock shared with the other phones, and the take kept on the phone until it has uploaded.

import { SharedClock } from '/assets/camera/clock.js';
import { Uploader } from '/assets/camera/upload.js';

const TOKEN_KEY = 'varbox.camera.token';
const PREFS_KEY = 'varbox.camera.prefs';
const $ = (sel) => document.querySelector(sel);

const state = {
  token: null,
  phone: null,
  stream: null,
  clock: new SharedClock('/api/clock'),
  worker: null,
  recording: false,
  stopping: false,
  wakeLock: null,
  procTrack: null,
  procAudio: null,
  rvfcActive: false,
  started: 0,
  take: null,
  uploader: null,
};

function load(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || 'null') ?? fallback; } catch (err) { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (err) { /* private mode */ }
}

function prefs() {
  return {
    letter: $('#letter').value,
    take: $('#take').value.trim(),
    res: $('#res').value,
    fps: Number($('#fps').value),
  };
}

const RES = { '1080': [1920, 1080], '720': [1280, 720] };
const BITRATE = { '1080-60': 16e6, '1080-30': 10e6, '720-60': 8e6, '720-30': 5e6 };

/** Which route this browser has for per-frame capture times, best first. */
function captureRoute() {
  if ('MediaStreamTrackProcessor' in window && 'VideoEncoder' in window) return 'videoframe';
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype && 'VideoEncoder' in window && 'VideoFrame' in window) return 'rvfc';
  return null;
}

function renderCapabilities() {
  const rows = [
    ['Frame stamps from the camera (MediaStreamTrackProcessor)', 'MediaStreamTrackProcessor' in window],
    ['Frame callback (requestVideoFrameCallback)', 'requestVideoFrameCallback' in HTMLVideoElement.prototype],
    ['Video encoder (WebCodecs)', 'VideoEncoder' in window],
    ['Audio encoder (WebCodecs)', 'AudioEncoder' in window],
    ['Phone storage (Origin Private File System)', Boolean(navigator.storage?.getDirectory)],
    ['Keep screen on (Wake Lock)', 'wakeLock' in navigator],
  ];
  $('#caps').innerHTML = rows.map(([label, ok]) => `<li class="${ok ? 'ok' : 'no'}">${ok ? '✓' : '✗'} ${label}</li>`).join('');
  const route = captureRoute();
  $('#route').textContent = route === 'videoframe'
    ? 'This browser stamps every frame from the camera.'
    : route === 'rvfc'
      ? 'This browser stamps frames as they reach the page. The sync test shows how close that is.'
      : 'This browser cannot stamp frames yet, so it cannot record for the rig. Use Chrome on Android.';
  $('#record').disabled = !route;
}

// ------------------------------------------------------------------ sign-in

async function signIn() {
  const fromLink = location.hash.length > 1 ? location.hash.slice(1) : '';
  if (/^[a-f0-9]{48}$/.test(fromLink)) {
    save(TOKEN_KEY, fromLink);
    history.replaceState(null, '', location.pathname);
  }
  state.token = load(TOKEN_KEY, null);
  if (!state.token) {
    $('#who').textContent = 'Not signed in';
    showNote('Open this page by scanning the camera QR on the dashboard (Team → Add a phone).', 'warn');
    return false;
  }
  const res = await fetch('/api/phone/ping', { headers: { Authorization: 'Bearer ' + state.token }, cache: 'no-store' }).catch(() => null);
  if (res && res.status === 401) {
    showNote('This phone was revoked. Add it again on the Team page and scan the new QR.', 'warn');
    return false;
  }
  if (res && res.ok) {
    state.phone = (await res.json()).phone;
    $('#who').textContent = state.phone.name;
    const letter = /([A-F])$/i.exec(state.phone.name || '');
    if (letter && !load(PREFS_KEY, null)) $('#letter').value = letter[1].toUpperCase();
  } else {
    $('#who').textContent = 'Offline';
  }
  return true;
}

function showNote(text, kind = '') {
  const n = $('#note');
  n.textContent = text;
  n.className = 'note ' + kind;
  n.hidden = !text;
}

// ------------------------------------------------------------------ clock

function renderClock() {
  const el = $('#sync');
  if (!state.clock.ready) { el.textContent = 'Measuring the clock…'; el.className = 'sync'; return; }
  const u = state.clock.uncertainty;
  const halfFrame = 500 / prefs().fps;
  const good = u <= halfFrame;
  el.textContent = good
    ? `Synced ±${u.toFixed(1)} ms`
    : `Sync ±${u.toFixed(0)} ms is too loose for ${prefs().fps} fps. Move to Wi-Fi.`;
  el.className = 'sync ' + (good ? 'good' : 'bad');
}

async function startClock() {
  renderClock();
  try {
    await state.clock.measure(40);
  } catch (err) {
    $('#sync').textContent = 'Cannot reach VAR Box to measure the clock.';
    $('#sync').className = 'sync bad';
  }
  state.clock.keepMeasuring(10000, 16);
  state.clock.onChange((m) => {
    renderClock();
    state.worker?.postMessage({ type: 'clock', measurement: m });
  });
  renderClock();
}

// ------------------------------------------------------------------ camera

async function openCamera() {
  const { res, fps } = prefs();
  const [width, height] = RES[res];
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: fps } },
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  }).catch(() => navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: fps } },
  }));
  const video = $('#preview');
  video.srcObject = state.stream;
  await video.play().catch(() => {});
  renderSettings();
}

/** Lock exposure, focus and white balance where the camera allows it; return what held. */
async function lockCamera(track) {
  const caps = typeof track.getCapabilities === 'function' ? track.getCapabilities() : {};
  const now = track.getSettings();
  const wanted = [];
  if (caps.exposureMode?.includes('manual') && now.exposureTime) wanted.push(['exposure', { exposureMode: 'manual', exposureTime: now.exposureTime }]);
  if (caps.focusMode?.includes('manual') && now.focusDistance !== undefined) wanted.push(['focus', { focusMode: 'manual', focusDistance: now.focusDistance }]);
  else if (caps.focusMode?.includes('single-shot')) wanted.push(['focus', { focusMode: 'single-shot' }]);
  if (caps.whiteBalanceMode?.includes('manual') && now.colorTemperature) wanted.push(['white_balance', { whiteBalanceMode: 'manual', colorTemperature: now.colorTemperature }]);
  const locked = [];
  const refused = [];
  for (const [name, constraint] of wanted) {
    try { await track.applyConstraints({ advanced: [constraint] }); locked.push(name); } catch (err) { refused.push(name); }
  }
  for (const name of ['exposure', 'focus', 'white_balance']) {
    if (!locked.includes(name) && !refused.includes(name)) refused.push(name);
  }
  return { locked, refused };
}

function renderSettings() {
  const track = state.stream?.getVideoTracks()[0];
  if (!track) return;
  const s = track.getSettings();
  $('#actual').textContent = `${s.width}×${s.height} at ${Math.round(s.frameRate || 0)} fps`;
}

// ------------------------------------------------------------------ recording

function takeName(p) {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '');
  return `${p.take}_${p.letter}_${stamp}`.replace(/[^A-Za-z0-9_-]+/g, '_');
}

async function startTake() {
  const route = captureRoute();
  if (!route || state.recording) return;
  const p = prefs();
  if (!p.take) { showNote('Give the take a name. Every camera of one take uses the same name.', 'warn'); return; }
  save(PREFS_KEY, p);
  showNote('');
  if (!state.stream) await openCamera();
  const videoTrack = state.stream.getVideoTracks()[0];
  const audioTrack = state.stream.getAudioTracks()[0] || null;
  const locks = await lockCamera(videoTrack);
  const settings = videoTrack.getSettings();
  navigator.storage.persist?.().catch(() => {});
  try { state.wakeLock = await navigator.wakeLock?.request('screen'); } catch (err) { state.wakeLock = null; }

  const name = takeName(p);
  const [width, height] = RES[p.res];
  state.take = { name, ...p, startAt: state.clock.now() };
  const worker = new Worker(document.body.dataset.worker, { type: 'module' });
  state.worker = worker;
  worker.onmessage = (e) => onWorker(e.data);
  const meta = {
    recording_id: p.take,
    round: '',
    camera: p.letter,
    device: navigator.userAgent,
    width,
    height,
    start_at_ms: state.take.startAt,
    settings: { width, height, fps: p.fps },
    applied: { ...settings, locked: locks.locked, refused: locks.refused },
    capture_route: route,
  };
  const msg = {
    type: 'start', name, meta, fps: p.fps, bitrate: BITRATE[`${p.res}-${p.fps}`],
    mainTimeOrigin: performance.timeOrigin, clockHistory: state.clock.history, audio: Boolean(audioTrack),
  };
  if (route === 'videoframe') {
    state.procTrack = videoTrack.clone();
    const vp = new MediaStreamTrackProcessor({ track: state.procTrack });
    const transfer = [vp.readable];
    msg.videoReadable = vp.readable;
    if (audioTrack && 'AudioEncoder' in window) {
      state.procAudio = audioTrack.clone();
      const ap = new MediaStreamTrackProcessor({ track: state.procAudio });
      msg.audioReadable = ap.readable;
      transfer.push(ap.readable);
    } else {
      msg.audio = false;
    }
    worker.postMessage(msg, transfer);
  } else {
    msg.audio = false;
    worker.postMessage(msg);
  }
  state.recording = true;
  state.started = performance.now();
  document.body.classList.add('recording');
  $('#record').textContent = 'Stop';
  tick();
}

/** Safari route: stamp each frame as it reaches the page and hand it to the worker. */
function startRvfc() {
  const video = $('#preview');
  state.rvfcActive = true;
  const step = (now, md) => {
    if (!state.rvfcActive) return;
    const capture = typeof md.captureTime === 'number' ? performance.timeOrigin + md.captureTime : null;
    const local = capture ?? performance.timeOrigin + (md.presentationTime ?? now);
    try {
      const frame = new VideoFrame(video, { timestamp: Math.round(local * 1000) });
      state.worker.postMessage({ type: 'frame', frame, captureLocal: local, source: capture ? 'web-rvfc-capture' : 'web-arrival' }, [frame]);
    } catch (err) { /* a frame not ready yet is skipped */ }
    video.requestVideoFrameCallback(step);
  };
  video.requestVideoFrameCallback(step);
}

function onWorker(m) {
  if (m.type === 'started') {
    if (captureRoute() === 'rvfc') startRvfc();
  } else if (m.type === 'progress') {
    $('#stats').textContent = `${m.frames} frames · ${(m.size / 1e6).toFixed(1)} MB${m.dropped ? ` · ${m.dropped} dropped` : ''}`;
  } else if (m.type === 'done') {
    finishTake(m);
  } else if (m.type === 'error') {
    showNote('Recording problem: ' + m.message, 'warn');
  }
}

function stopTake(interrupted = false) {
  if (!state.recording || state.stopping) return;
  state.stopping = true;
  state.rvfcActive = false;
  $('#record').textContent = 'Saving…';
  $('#record').disabled = true;
  state.worker.postMessage({ type: 'stop', interrupted });
}

function finishTake(m) {
  const t = state.take;
  state.procTrack?.stop();
  state.procAudio?.stop();
  state.procTrack = null;
  state.procAudio = null;
  state.wakeLock?.release().catch(() => {});
  state.wakeLock = null;
  state.worker.terminate();
  state.worker = null;
  state.recording = false;
  state.stopping = false;
  document.body.classList.remove('recording');
  $('#record').textContent = 'Record';
  $('#record').disabled = false;
  $('#stats').textContent = `Saved ${m.frames} frames (${(m.size / 1e6).toFixed(1)} MB), stamps: ${m.sidecar.frame_times_source}`;
  state.uploader.add({
    name: m.name, rec_uid: t.take, letter: t.letter, fps: t.fps, title: t.take,
    expected_cameras: 2, device: navigator.userAgent.slice(0, 120), size: m.size,
  });
  renderUploads();
  refreshStorage();
}

function tick() {
  if (!state.recording) return;
  const s = Math.floor((performance.now() - state.started) / 1000);
  $('#timer').textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  setTimeout(tick, 250);
}

// ------------------------------------------------------------------ storage and uploads

async function refreshStorage() {
  try {
    const { quota, usage } = await navigator.storage.estimate();
    $('#storage').textContent = `${((quota - usage) / 1e9).toFixed(1)} GB free for takes`;
  } catch (err) { $('#storage').textContent = ''; }
}

function renderUploads(status = state.uploader?.status || {}) {
  const q = state.uploader?.queue || [];
  const recent = q.slice(-6).reverse();
  $('#uploads').innerHTML = recent.length ? recent.map((e) => {
    const s = status[e.name] || {};
    const label = e.done ? (e.error ? 'Failed: ' + e.error : 'Uploaded') : s.state === 'uploading' ? `Uploading ${s.pct || 0}%` : s.error || 'Waiting to upload';
    return `<li><span class="mono">${e.name}</span><span>${label}</span></li>`;
  }).join('') : '<li class="muted">No takes yet.</li>';
}

// ------------------------------------------------------------------ boot

async function boot() {
  const p = load(PREFS_KEY, null);
  if (p) {
    $('#letter').value = p.letter || 'A';
    $('#take').value = p.take || '';
    $('#res').value = p.res || '1080';
    $('#fps').value = String(p.fps || 30);
  }
  if (!$('#take').value) $('#take').value = 'sync-' + new Date().toISOString().slice(0, 10).replace(/-/g, '');
  renderCapabilities();
  const ok = await signIn();
  if (!ok) { $('#record').disabled = true; return; }
  state.uploader = new Uploader(state.token, renderUploads);
  renderUploads();
  state.uploader.run();
  refreshStorage();
  startClock();
  openCamera().catch((err) => showNote('The camera did not open: ' + err.message, 'warn'));

  $('#record').onclick = () => (state.recording ? stopTake(false) : startTake().catch((err) => showNote(err.message, 'warn')));
  for (const id of ['#res', '#fps']) {
    $(id).onchange = () => { if (!state.recording) openCamera().catch(() => {}); renderClock(); };
  }
  // A phone that leaves the page stops recording; close the file cleanly and mark it.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.recording) stopTake(true);
  });
}

boot();
