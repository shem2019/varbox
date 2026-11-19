// Sync flash: draws the shared clock as a machine-readable code on every display frame.
//
// Two phones film this screen. Each video frame then shows which display tick it caught, and the
// phone's sidecar says when it captured that frame on the shared clock. Comparing the two phones
// over hundreds of frames measures how closely their clocks agree, independent of sound.
//
// Layout, in units of a 16 × 10 grid (tools/measure_web_sync.py reads the same numbers):
//   ArUco 4×4 markers 0–3 at (0.5, 0.5), (12.5, 0.5), (12.5, 6.5), (0.5, 6.5), 3 units square
//   24 Gray-coded bits, 8 columns × 3 rows of 1-unit cells from (4, 3.5), most significant first
//   black reference: the marker border at (0.75, 2); white reference: background at (2, 5)

import { SharedClock } from '/assets/camera/clock.js';

export const TICK_MS = 1000 / 60;
export const BITS = 24;

const MARKERS = [
  ['000000', '010110', '001010', '000110', '000100', '000000'],
  ['000000', '000000', '011110', '010010', '010100', '000000'],
  ['000000', '000110', '000110', '000100', '011010', '000000'],
  ['000000', '010010', '010010', '001000', '001100', '000000'],
];
const MARKER_AT = [[0.5, 0.5], [12.5, 0.5], [12.5, 6.5], [0.5, 6.5]];

export const gray = (n) => (n ^ (n >>> 1)) >>> 0;

const canvas = document.getElementById('flash');
const ctx = canvas.getContext('2d', { alpha: false });
const label = document.getElementById('label');
const clock = new SharedClock('/api/clock');
let geom = null;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  const unit = Math.floor(Math.min(canvas.width / 16, canvas.height / 10));
  geom = { unit, x0: Math.floor((canvas.width - 16 * unit) / 2), y0: Math.floor((canvas.height - 10 * unit) / 2) };
}

function rect(x, y, w, h) {
  const { unit, x0, y0 } = geom;
  ctx.fillRect(x0 + Math.round(x * unit), y0 + Math.round(y * unit), Math.round(w * unit), Math.round(h * unit));
}

function draw() {
  const now = clock.now();
  const tick = Math.floor(now / TICK_MS) % 2 ** BITS;
  const code = gray(tick);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#000';
  MARKERS.forEach((rows, m) => {
    const [mx, my] = MARKER_AT[m];
    rows.forEach((row, r) => [...row].forEach((bit, c) => { if (bit === '0') rect(mx + c * 0.5, my + r * 0.5, 0.5, 0.5); }));
  });
  for (let i = 0; i < BITS; i++) {
    if ((code >>> (BITS - 1 - i)) & 1) rect(4 + (i % 8), 3.5 + Math.floor(i / 8), 1, 1);
  }
  const { unit, x0, y0 } = geom;
  ctx.font = `600 ${Math.round(unit * 0.9)}px "IBM Plex Mono", monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(((now / 1000) % 1000).toFixed(3), x0 + 8 * unit, y0 + 8.4 * unit);
  requestAnimationFrame(draw);
}

function renderClock() {
  label.textContent = clock.ready
    ? `Clock ±${clock.uncertainty.toFixed(1)} ms. Film this screen with every phone for 30 seconds.`
    : 'Measuring the clock…';
}

async function boot() {
  resize();
  addEventListener('resize', resize);
  renderClock();
  await clock.measure(40).catch(() => { label.textContent = 'Cannot reach VAR Box to measure the clock.'; });
  clock.keepMeasuring(10000, 16);
  clock.onChange(renderClock);
  renderClock();
  try { await navigator.wakeLock?.request('screen'); } catch (err) { /* stays on while the page is used */ }
  requestAnimationFrame(draw);
}

boot();
