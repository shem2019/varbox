// VAR Box dashboard: a small router and the pages behind it.

import { LayersPlayer } from './layers.js';
import { Viewer4D } from './viewer4d.js';
import { eventTime, fmtTime, parseJson } from './meshdata.js';

const main = document.getElementById('main');
const CHUNK = 8 * 1024 * 1024;
let cleanup = [];
let pendingSeek = null;

// ------------------------------------------------------------------ helpers
async function api(path, { method = 'GET', body, raw } = {}) {
  const headers = { 'X-VarBox': '1' };
  if (body !== undefined && !raw) headers['Content-Type'] = 'application/json';
  const res = await fetch('/api/' + path, { method, headers, body: raw ? body : body !== undefined ? JSON.stringify(body) : undefined });
  if (res.status === 401) { location.href = '/login'; throw new Error('Sign in to continue.'); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 409) throw new Error(data.error || `Request failed (${res.status})`);
  return { status: res.status, ...data };
}

function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = message;
  document.getElementById('toasts').append(el);
  setTimeout(() => el.remove(), 4200);
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtBytes = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1e3)) + ' KB');
const fmtDate = (iso) => new Date(iso).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const ago = (iso) => {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return fmtDate(iso);
};
const landed = (t) => (t ? (t.landed_head || 0) + (t.landed_torso || 0) : 0);

function confirmDialog(title, text, action = 'Delete') {
  return new Promise((resolve) => {
    const d = document.createElement('dialog');
    d.innerHTML = `<h3>${esc(title)}</h3><p>${esc(text)}</p><div class="actions"><button class="btn" value="no">Keep</button><button class="btn primary danger" value="yes">${esc(action)}</button></div>`;
    document.body.append(d);
    d.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { d.close(); resolve(b.value === 'yes'); d.remove(); } });
    d.addEventListener('cancel', () => { resolve(false); d.remove(); });
    d.showModal();
  });
}

// ------------------------------------------------------------------ router
const routes = [
  [/^\/app\/?$/, () => analysesPage()],
  [/^\/app\/analysis\/(\d+)(?:\/(layers|4d|compare|punches|files))?$/, (m) => analysisPage(+m[1], m[2] || 'layers')],
  [/^\/app\/import$/, () => importPage()],
  [/^\/app\/jobs$/, () => jobsPage()],
  [/^\/app\/gpu$/, () => gpuPage()],
];

function navigate(path, replace = false) {
  if (replace) history.replaceState({}, '', path); else history.pushState({}, '', path);
  render();
}

function render() {
  cleanup.forEach((fn) => fn());
  cleanup = [];
  const path = location.pathname;
  document.querySelectorAll('.nav a').forEach((a) => {
    const key = a.dataset.nav;
    const on = (key === 'analyses' && (path === '/app' || path.startsWith('/app/analysis'))) || path.startsWith('/app/' + key);
    a.classList.toggle('active', on);
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  for (const [re, fn] of routes) {
    const m = path.match(re);
    if (m) { fn(m); main.focus({ preventScroll: true }); window.scrollTo(0, 0); return; }
  }
  main.innerHTML = '<div class="card empty"><h3>Page moved</h3><p>Pick a section from the menu.</p></div>';
}

document.addEventListener('click', (e) => {
  const a = e.target.closest('a[data-link]');
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey || a.target) return;
  e.preventDefault();
  navigate(a.getAttribute('href'));
});
window.addEventListener('popstate', render);

document.getElementById('sign-out').addEventListener('click', async () => {
  await api('logout', { method: 'POST' }).catch(() => {});
  location.href = '/login';
});
document.getElementById('theme-toggle').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('varbox-theme', root.dataset.theme); } catch (err) { /* private mode */ }
});

// Sidebar status: running jobs and GPU presence, refreshed in the background.
async function refreshStatus() {
  try {
    const [{ jobs }, { workers }] = await Promise.all([api('jobs'), api('workers')]);
    const active = jobs.filter((j) => j.status === 'running' || j.status === 'queued').length;
    const badge = document.getElementById('jobs-badge');
    badge.hidden = active === 0; badge.textContent = active;
    document.getElementById('gpu-dot').classList.toggle('online', workers.some((w) => w.online));
  } catch (err) { /* offline moments are fine */ }
}
refreshStatus();
setInterval(refreshStatus, 10000);

// ------------------------------------------------------------------ analyses
async function analysesPage() {
  main.innerHTML = `
    <div class="page-head"><div><h1>Analyses</h1><div class="sub">Every finished session, ready to replay and review.</div></div>
      <a class="btn primary" href="/app/import" data-link>Import videos</a></div>
    <div class="grid">${'<div class="card skeleton" style="height:300px"></div>'.repeat(3)}</div>`;
  const { analyses } = await api('analyses');
  const grid = main.querySelector('.grid');
  if (!analyses.length) {
    grid.outerHTML = `<div class="card empty">
      <h3>Your first analysis starts with two videos</h3>
      <p>Import the two phone recordings of a round. A connected GPU turns them into layered replays, 3D bodies and a punch-by-punch review, and the finished analysis appears here.</p>
      <a class="btn primary" href="/app/import" data-link>Import videos</a></div>`;
    return;
  }
  grid.innerHTML = analyses.map((a) => {
    const m = a.meta || {};
    const t = m.tally_after || m.tally_before || {};
    const r = landed(t.red), b = landed(t.blue), total = Math.max(1, r + b);
    return `<a class="card a-card" href="/app/analysis/${a.id}" data-link>
      <div class="poster"><img src="${a.poster}" alt="" loading="lazy"><span class="dur">${fmtTime(m.duration_s || 0)}</span></div>
      <div class="body">
        <h3>${esc(a.title)}</h3>
        <div class="chips">
          <span class="chip">${m.views === 2 ? 'Two cameras' : 'One camera'}</span>
          <span class="chip">${m.punches ?? 0} punches</span>
          <span class="chip">${fmtDate(a.created_at)}</span>
        </div>
        <div class="versus"><span class="red-ink">Red ${r}</span>
          <div class="bar"><i style="width:${(100 * r / total).toFixed(1)}%;background:var(--red)"></i><i style="width:${(100 * b / total).toFixed(1)}%;background:var(--blue)"></i></div>
          <span class="blue-ink" style="text-align:right">${b} Blue</span></div>
      </div></a>`;
  }).join('');
}

async function analysisPage(id, tab) {
  main.innerHTML = `<div class="card skeleton" style="height:420px"></div>`;
  const { analysis, files, base } = await api('analyses/' + id);
  const m = analysis.meta || {};
  const tabs = [['layers', 'Layers'], ['4d', '4D replay'], ['compare', 'Before and after'], ['punches', 'Punches'], ['files', 'Downloads']];
  main.innerHTML = `
    <div class="crumbs"><a href="/app" data-link>Analyses</a><span>/</span><span>${esc(analysis.title)}</span></div>
    <div class="page-head">
      <div><h1 class="title" title="Rename">${esc(analysis.title)}</h1>
        <div class="chips" style="margin-top:10px">
          <span class="chip">${m.views === 2 ? 'Two cameras, fused' : 'One camera'}</span>
          <span class="chip">${fmtTime(m.duration_s || 0)} · ${m.fps ? Math.round(m.fps) + ' fps' : ''}</span>
          <span class="chip">${m.punches ?? 0} punches</span>
          <span class="chip">${esc(m.source_video || '')}</span>
        </div></div>
      <div style="display:flex;gap:8px"><button class="btn sm" data-act="rename">Rename</button><button class="btn sm danger" data-act="delete">Delete</button></div>
    </div>
    <nav class="tabs">${tabs.map(([k, l]) => `<a href="/app/analysis/${id}/${k}" data-link class="${k === tab ? 'active' : ''}">${l}</a>`).join('')}</nav>
    <div class="tab-body"></div>`;
  main.querySelector('[data-act=rename]').onclick = async () => {
    const title = prompt('New title', analysis.title);
    if (!title) return;
    await api('analyses/' + id, { method: 'PATCH', body: { title } });
    toast('Renamed'); render();
  };
  main.querySelector('[data-act=delete]').onclick = async () => {
    if (!(await confirmDialog('Delete this analysis?', 'Its replays, meshes and downloads leave the server for good.'))) return;
    await api('analyses/' + id, { method: 'DELETE' });
    toast('Analysis deleted'); navigate('/app');
  };
  const body = main.querySelector('.tab-body');
  const fetchJson = (name) => fetch(base + name).then((r) => (r.ok ? r.text() : null)).then((t) => (t ? parseJson(t) : null)).catch(() => null);
  const events = (await fetchJson('events.json')) || [];
  const fps = m.fps || 50;

  if (tab === 'layers') {
    const player = new LayersPlayer(body, { base, analysis: m, events, seekTo: pendingSeek });
    pendingSeek = null;
    cleanup.push(() => player.destroy());
  } else if (tab === '4d') {
    if (!m.has_mesh) { body.innerHTML = '<div class="card empty"><h3>3D bodies arrive with the next analysis</h3></div>'; return; }
    const viewer = new Viewer4D(body, { base: base + 'mesh/', events });
    cleanup.push(() => viewer.destroy());
  } else if (tab === 'compare') {
    const before = (await fetchJson('events_before.json')) || [];
    const row = (label, t) => ['red', 'blue'].map((r) => {
      const c = (t || {})[r] || {};
      return `<tr><td>${label}</td><td><span class="swatch ${r}"></span> ${r}</td><td>${c.thrown || 0}</td><td>${c.landed_head || 0}</td><td>${c.landed_torso || 0}</td><td>${c.blocked || 0}</td><td>${c.missed || 0}</td></tr>`;
    }).join('');
    body.innerHTML = `
      <div style="display:grid;gap:16px">
        ${m.videos?.before_after ? `<div class="card media-box" style="overflow:hidden"><video src="${base}before_after.mp4" controls playsinline preload="auto" style="width:100%;aspect-ratio:32/9"></video><div class="media-wait" role="status"><div class="spinner"></div><div>Loading before and after<small>two replays side by side</small></div></div></div>` : ''}
        <div class="card" style="padding:16px"><div class="table-wrap"><table class="data">
          <thead><tr><th>Method</th><th>Boxer</th><th>Thrown</th><th>Head</th><th>Body</th><th>Blocked</th><th>Missed</th></tr></thead>
          <tbody>${row('Before: geometry', m.tally_before)}${m.tally_after ? row('After: geometry and strike model', m.tally_after) : ''}</tbody></table></div>
          <p class="muted" style="margin:12px 0 0;font-size:13px">Before uses the 3D bodies alone. After adds the strike model trained on labelled bouts, which reads each punch from the footage around the glove. ${before.length} punches in both.</p></div>
      </div>`;
    const vid = body.querySelector('video'), wait = body.querySelector('.media-wait');
    if (vid && wait) {
      const done = () => wait.remove();
      if (vid.readyState >= 2) done();
      vid.addEventListener('loadeddata', done, { once: true });
      vid.addEventListener('playing', done, { once: true });
      vid.addEventListener('error', () => { wait.innerHTML = '<div>The video stopped loading<small>Reload the page to try again.</small></div>'; }, { once: true });
    }
  } else if (tab === 'punches') {
    renderPunches(body, events, fps, id, m.start_s || 0);
  } else {
    body.innerHTML = `<div class="card" style="padding:6px 16px"><div class="table-wrap"><table class="data">
      <thead><tr><th>File</th><th>Size</th><th></th></tr></thead>
      <tbody>${files.map((f) => `<tr><td class="mono">${esc(f.path)}</td><td class="num">${fmtBytes(f.size)}</td>
        <td style="text-align:right"><a class="btn sm" href="${base}${esc(f.path)}?download=1">Download</a></td></tr>`).join('')}</tbody></table></div></div>`;
  }
}

function renderPunches(body, events, fps, id, startS) {
  const state = { who: 'all', outcome: 'all' };
  const draw = () => {
    const rows = events.filter((e) => (state.who === 'all' || e.attacker === state.who) && (state.outcome === 'all' || e.outcome === state.outcome));
    body.innerHTML = `
      <div class="filters">${['all', 'red', 'blue'].map((w) => `<button data-who="${w}" class="${state.who === w ? 'on' : ''}">${w === 'all' ? 'Both boxers' : w}</button>`).join('')}
        <span style="width:12px"></span>${['all', 'landed', 'blocked', 'missed'].map((o) => `<button data-out="${o}" class="${state.outcome === o ? 'on' : ''}">${o === 'all' ? 'Every outcome' : o}</button>`).join('')}</div>
      <div class="card" style="padding:6px 16px"><div class="table-wrap"><table class="data">
        <thead><tr><th>Time</th><th>Boxer</th><th>Hand</th><th>Outcome</th><th>Target</th><th>Confidence</th><th>Geometry said</th></tr></thead>
        <tbody>${rows.map((e) => `<tr class="click" data-t="${eventTime(e, fps) - startS}">
          <td class="num">${fmtTime(eventTime(e, fps) - startS)}</td>
          <td><span class="swatch ${e.attacker}"></span> ${e.attacker}</td><td>${e.hand}</td>
          <td><span class="pill ${e.outcome}">${e.outcome}</span></td><td>${e.target ? (e.target === 'torso' ? 'body' : e.target) : ''}</td>
          <td class="num">${Math.round((e.confidence || 0) * 100)}%</td><td class="muted">${esc((e.mesh_outcome || '').replace('_', ' '))}</td></tr>`).join('')}</tbody></table>
        ${rows.length ? '' : '<p class="muted" style="padding:16px 0;margin:0">Every punch in this view is filtered out. Pick another filter.</p>'}</div></div>
      <p class="muted" style="font-size:13px">Pick a punch to open it in the layered replay. Times count from the start of this analysis.</p>`;
    body.querySelectorAll('[data-who]').forEach((b) => { b.onclick = () => { state.who = b.dataset.who; draw(); }; });
    body.querySelectorAll('[data-out]').forEach((b) => { b.onclick = () => { state.outcome = b.dataset.out; draw(); }; });
    body.querySelectorAll('tr.click').forEach((tr) => {
      tr.onclick = () => {
        pendingSeek = Math.max(0, +tr.dataset.t - 0.4);
        navigate(`/app/analysis/${id}/layers`);
      };
    });
  };
  draw();
}

// ------------------------------------------------------------------ import
function importPage() {
  const picked = { a: null, b: null };
  main.innerHTML = `
    <div class="page-head"><div><h1>Import videos</h1><div class="sub">Add the two phone recordings of one round. Camera B is optional; two cameras give the most accurate 3D.</div></div></div>
    <div class="import-grid">
      ${['a', 'b'].map((c) => `
      <label class="drop" data-cam="${c}">
        <h3><span class="swatch ${c === 'a' ? 'red' : 'blue'}" style="border-radius:50%"></span>Camera ${c.toUpperCase()}${c === 'b' ? ' <span class="chip">optional</span>' : ''}</h3>
        <p>Drop the video here, or choose it from this device. Large files upload in pieces and pick up again after a dropped connection.</p>
        <input type="file" accept="video/*" hidden>
        <span class="btn sm">Choose video</span>
        <div class="file" hidden></div><div class="progress" hidden><i></i></div>
      </label>`).join('')}
    </div>
    <div class="form-row">
      <div class="field"><label for="title">Session title</label><input class="input" id="title" placeholder="Sparring, round 1"></div>
      <div class="field"><label for="fps">Detail</label><select class="input" id="fps"><option value="30">Standard, 30 fps (faster)</option><option value="60">Full, 60 fps</option></select></div>
    </div>
    <div class="actions"><button class="btn primary" id="start" disabled>Start analysis</button></div>`;
  const start = main.querySelector('#start');
  const refresh = () => { start.disabled = !(picked.a?.ready && (!picked.b || picked.b.ready)); };
  main.querySelectorAll('.drop').forEach((drop) => {
    const cam = drop.dataset.cam;
    const input = drop.querySelector('input');
    const choose = (file) => { if (file) upload(cam, file, drop); };
    input.onchange = () => choose(input.files[0]);
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); choose(e.dataTransfer.files[0]); });
  });

  async function upload(cam, file, drop) {
    const fileEl = drop.querySelector('.file'), bar = drop.querySelector('.progress'), fill = bar.querySelector('i');
    fileEl.hidden = false; bar.hidden = false; bar.classList.remove('done');
    fileEl.textContent = `${file.name} · ${fmtBytes(file.size)}`;
    picked[cam] = { ready: false };
    refresh();
    try {
      const { upload: up } = await api('uploads', { method: 'POST', body: { name: file.name, size: file.size } });
      let offset = 0;
      while (offset < file.size) {
        const res = await api(`uploads/${up.id}?offset=${offset}`, { method: 'PUT', body: file.slice(offset, offset + CHUNK), raw: true });
        offset = res.received;
        fill.style.width = `${(100 * offset / file.size).toFixed(1)}%`;
        fileEl.textContent = `${file.name} · ${fmtBytes(offset)} of ${fmtBytes(file.size)}`;
      }
      picked[cam] = { ready: true, id: up.id };
      bar.classList.add('done');
      fileEl.textContent = `${file.name} · ${fmtBytes(file.size)} · uploaded`;
    } catch (err) {
      toast(err.message, 'error');
      fileEl.textContent = `${file.name} · upload paused, choose it again to continue`;
    }
    refresh();
  }

  start.onclick = async () => {
    start.disabled = true;
    try {
      await api('jobs', { method: 'POST', body: {
        title: main.querySelector('#title').value, camera_a: picked.a.id, camera_b: picked.b?.id || null, fps: +main.querySelector('#fps').value,
      } });
      toast('Analysis queued. The GPU picks it up next.');
      navigate('/app/jobs');
    } catch (err) { toast(err.message, 'error'); start.disabled = false; }
  };
}

// ------------------------------------------------------------------ jobs
const STAGES = [['download', 'Fetch'], ['sync', 'Sync'], ['seed', 'Find boxers'], ['masks', 'Track'], ['meshes', '3D bodies'], ['world', 'Ring floor'],
  ['fusion', 'Fuse'], ['contact', 'Punches'], ['render', 'Render'], ['rescoring', 'Strike model'], ['packaging', 'Package'], ['uploading', 'Deliver']];

async function jobsPage() {
  main.innerHTML = `<div class="page-head"><div><h1>Jobs</h1><div class="sub">Analyses in progress, reported live by the GPU.</div></div>
    <a class="btn" href="/app/import" data-link>Import videos</a></div><div class="jobs"><div class="card skeleton" style="height:180px"></div></div>`;
  const box = main.querySelector('.jobs');
  const draw = async () => {
    const { jobs } = await api('jobs');
    if (!jobs.length) {
      box.innerHTML = `<div class="card empty"><h3>The queue is clear</h3><p>Import two phone recordings to start an analysis. Progress shows here, stage by stage.</p><a class="btn primary" href="/app/import" data-link>Import videos</a></div>`;
      return;
    }
    box.innerHTML = jobs.map((j) => {
      const at = STAGES.findIndex(([k]) => k === j.stage);
      const steps = STAGES.map(([k, l], i) => {
        const cls = j.status === 'done' || (at >= 0 && i < at) ? 'done' : i === at && j.status === 'running' ? 'now' : '';
        return `<div class="s ${cls}"><i></i><span>${l}</span></div>`;
      }).join('');
      const pct = j.pct != null && j.status === 'running' ? ` · ${Math.round(j.pct * 100)}%` : '';
      return `<article class="card job">
        <div class="job-head"><div><h3>${esc(j.title)}</h3><div class="muted" style="font-size:13px;margin-top:4px">${j.camera_b ? 'Two cameras' : 'One camera'} · ${j.fps} fps · added ${ago(j.created_at)}${j.worker ? ' · on ' + esc(j.worker) : ''}</div></div>
          <div style="display:flex;gap:8px;align-items:center"><span class="state ${j.status}">${j.status === 'queued' ? 'waiting for a GPU' : j.status}${pct}</span>
          ${j.analysis_id ? `<a class="btn sm primary" href="/app/analysis/${j.analysis_id}" data-link>Open analysis</a>` : ''}
          ${['queued', 'failed', 'done'].includes(j.status) ? `<button class="btn sm ghost" data-del="${j.id}">Remove</button>` : ''}</div></div>
        <div class="stepper" style="--n:${STAGES.length}">${steps}</div>
        ${j.status === 'failed' ? `<div class="msg" style="color:var(--red)">${esc(j.error || 'Processing stopped')}</div>` : j.message ? `<div class="msg">${esc(j.message)}</div>` : ''}
        ${j.previews.length ? `<div class="previews">${j.previews.map((p) => `<img src="${p}?t=${Date.parse(j.updated_at)}" alt="Seed preview">`).join('')}</div>` : ''}
      </article>`;
    }).join('');
    box.querySelectorAll('[data-del]').forEach((b) => { b.onclick = async () => { await api('jobs/' + b.dataset.del, { method: 'DELETE' }); draw(); }; });
  };
  await draw();
  const t = setInterval(() => draw().catch(() => {}), 3000);
  cleanup.push(() => clearInterval(t));
}

// ------------------------------------------------------------------ GPU
async function gpuPage() {
  main.innerHTML = `<div class="page-head"><div><h1>GPU</h1><div class="sub">Rented GPUs connect out to this dashboard, take queued analyses and report back.</div></div></div><div class="gpu-body"></div>`;
  const body = main.querySelector('.gpu-body');
  const draw = async () => {
    const { workers, token } = await api('workers');
    const cmd = `git clone https://github.com/shem2019/varbox.git ~/work/varbox; cd ~/work/varbox && git pull -q \\
  && read -rsp "Hugging Face token: " HF_TOKEN && echo && export HF_TOKEN \\
  && bash scripts/gpu/setup.sh && source ~/work/env.sh \\
  && export VARBOX_WORKER_TOKEN=${token || 'WORKER_TOKEN'} \\
  && (nohup python -m boxing_analytics.mesh4d.worker serve > ~/work/worker.log 2>&1 &)`;
    body.innerHTML = `
      <div class="workers" style="margin-bottom:22px">${workers.length ? workers.map((w) => {
        const i = w.info || {};
        const pct = (a, b) => (b ? Math.min(100, Math.round(100 * a / b)) : 0);
        const gb = (mb) => `${((mb || 0) / 1024).toFixed(1)} GB`;
        const meter = (label, value, detail) => `<div class="meter-row"><div class="meter-label"><span>${label}</span><span class="num">${detail}</span></div><div class="meter"><i style="width:${value}%"></i></div></div>`;
        return `<div class="card worker"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><strong>${esc(w.name)}</strong>
          <span class="chip"><span class="dot" style="color:${w.online ? 'var(--landed)' : 'var(--ink-3)'}"></span>${w.online ? 'online' : 'last seen ' + ago(w.last_seen)}</span></div>
          <div class="muted" style="font-size:13px">${esc(i.gpu || 'GPU')} · ${i.cores || '?'} CPU cores${i.slots ? ` · ${i.slots} job${i.slots > 1 ? 's' : ''} at once` : ''}</div>
          ${meter('GPU busy', i.util ?? 0, `${i.util ?? 0}%`)}
          ${meter('GPU memory', pct(i.mem_used_mb, i.mem_total_mb), `${gb(i.mem_used_mb)} of ${gb(i.mem_total_mb)}`)}
          ${meter('CPU busy', Math.round(i.cpu_util ?? 0), `${Math.round(i.cpu_util ?? 0)}%`)}
          ${meter('RAM', pct(i.ram_used_mb, i.ram_total_mb), `${gb(i.ram_used_mb)} of ${gb(i.ram_total_mb)}`)}
        </div>`;
      }).join('') : '<div class="card empty" style="grid-column:1/-1"><h3>Connect a GPU to start analysing</h3><p>Rent an NVIDIA GPU with 48 GB or more, then run the command below on it. It shows up here within a minute.</p></div>'}</div>
      <div class="card" style="padding:18px;display:grid;gap:10px"><h3 style="font-size:24px">Connect a GPU</h3>
        <p class="muted" style="margin:0">Paste this whole command on a fresh machine. It asks for your Hugging Face token (typing stays hidden), installs everything in about 25 minutes, and connects the GPU to this dashboard. Keep the worker token private.</p>
        <pre class="cmd">${esc(cmd)}</pre><div><button class="btn sm" id="copy">Copy command</button></div></div>`;
    body.querySelector('#copy').onclick = async () => {
      try { await navigator.clipboard.writeText(cmd); toast('Command copied'); } catch (err) { toast('Select the command and copy it by hand.', 'error'); }
    };
  };
  await draw();
  const t = setInterval(() => draw().catch(() => {}), 10000);
  cleanup.push(() => clearInterval(t));
}

render();
