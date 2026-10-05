// Layers player: real footage composited per person from the stacked layers video
// (footage on top, label map underneath), with 3D bodies drawn through the real camera.

import { MeshData, ROLE_HEX, ROLE_CSS, OUTCOME_HEX, eventTime, fmtTime, parseJson } from './meshdata.js';

const MODES = [
  { key: 'real', label: 'Real' },
  { key: 'mask', label: 'Silhouette' },
  { key: 'mesh', label: '3D' },
  { key: 'hide', label: 'Hidden' },
];
const BACKGROUNDS = [
  { key: 'footage', label: 'Footage' },
  { key: 'dim', label: 'Dimmed' },
  { key: 'plate', label: 'Empty ring' },
];
const MODE_INDEX = { real: 0, mask: 1, mesh: 2, hide: 3 };
const BG_INDEX = { footage: 0, dim: 1, plate: 2 };

const VERT = `
attribute vec2 p; varying vec2 uv;
void main() { uv = vec2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`;
const FRAG = `
precision mediump float;
varying vec2 uv;
uniform sampler2D vid; uniform sampler2D plate;
uniform vec3 modes; uniform float bg;
uniform vec3 colR; uniform vec3 colB; uniform vec3 colF;
vec3 background(vec3 c, vec3 pl, float mode) {
  vec3 base = mode > 1.5 ? pl : c;
  if (mode > 0.5 && mode < 1.5) { float g = dot(c, vec3(.299, .587, .114)); return mix(c, vec3(g), .75) * .38; }
  return base;
}
vec3 person(vec3 c, vec3 pl, float m, vec3 col, float bgm) {
  if (m < 0.5) return c;
  if (m < 1.5) return col;
  return background(pl, pl, bgm < 1.5 ? (bgm > 0.5 ? 1.0 : 0.0) : 2.0);
}
void main() {
  vec3 c = texture2D(vid, vec2(uv.x, uv.y * 0.5)).rgb;
  float l = texture2D(vid, vec2(uv.x, 0.5 + uv.y * 0.5)).r;
  vec3 pl = texture2D(plate, uv).rgb;
  vec3 outc;
  if (l < 0.19) outc = background(c, pl, bg);
  else if (l < 0.47) outc = person(c, pl, modes.x, colR, bg);
  else if (l < 0.78) outc = person(c, pl, modes.y, colB, bg);
  else outc = person(c, pl, modes.z, colF, bg);
  gl_FragColor = vec4(outc, 1.0);
}`;

function hexToVec(hex) { return [(hex >> 16 & 255) / 255, (hex >> 8 & 255) / 255, (hex & 255) / 255]; }

export class LayersPlayer {
  constructor(root, { base, analysis, events, seekTo = null }) {
    this.root = root;
    this.base = base;
    this.analysis = analysis;
    this.events = [...events].sort((a, b) => (a.contact_frame ?? a.peak_frame) - (b.contact_frame ?? b.peak_frame));
    this.modes = { red: 'real', blue: 'real', referee: 'real' };
    this.bg = 'footage';
    this.flashes = true;
    this.speed = 1;
    this.dirty = true;
    this.seekTo = seekTo;
    this.raf = 0;
    this.build();
    this.init().catch((err) => { this.loading.textContent = 'This analysis could not load: ' + err.message; });
  }

  build() {
    this.root.innerHTML = `
      <div class="player">
        <div>
          <div class="stage" tabindex="0" aria-label="Layered replay">
            <canvas class="gl"></canvas><canvas class="three"></canvas>
            <div class="hud"></div>
            <div class="loading">Loading layers…</div>
          </div>
          <div class="transport">
            <button class="btn icon-btn" data-act="back" title="Previous frame (←)" aria-label="Previous frame"><svg viewBox="0 0 16 16"><path d="M10 3 5 8l5 5z"/></svg></button>
            <button class="btn icon-btn primary" data-act="play" title="Play (space)" aria-label="Play"><svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg></button>
            <button class="btn icon-btn" data-act="fwd" title="Next frame (→)" aria-label="Next frame"><svg viewBox="0 0 16 16"><path d="m6 3 5 5-5 5z"/></svg></button>
            <div class="scrub"><div class="marks"></div><input type="range" min="0" max="1" step="0.001" value="0" aria-label="Timeline"></div>
            <span class="time">0:00.0</span>
            <select class="input" data-act="speed" style="width:auto;height:36px" aria-label="Playback speed">
              <option value="0.1">0.1×</option><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option>
            </select>
          </div>
        </div>
        <div class="panel-side">
          <div class="card"><h4>Layers</h4><div class="layers"></div>
            <div class="layer-row"><div class="who">Background</div><div class="seg bgseg"></div></div>
            <label style="display:flex;gap:8px;align-items:center;font-size:14px"><input type="checkbox" class="flash" checked> Flash each punch at contact</label>
          </div>
          <div class="card"><h4>Tally to this moment</h4><table class="tally"><thead><tr><th></th><th>Thrown</th><th>Head</th><th>Body</th><th>Blocked</th></tr></thead><tbody></tbody></table></div>
          <div class="card"><h4>Punches</h4><div class="ev-list"></div></div>
        </div>
      </div>`;
    const $ = (s) => this.root.querySelector(s);
    this.stage = $('.stage');
    this.glCanvas = $('canvas.gl');
    this.threeCanvas = $('canvas.three');
    this.hud = $('.hud');
    this.loading = $('.loading');
    this.range = $('input[type=range]');
    this.timeEl = $('.time');
    this.playBtn = $('[data-act=play]');
    this.evList = $('.ev-list');
    this.tallyBody = $('.tally tbody');

    this.playBtn.addEventListener('click', () => this.toggle());
    $('[data-act=back]').addEventListener('click', () => this.step(-1));
    $('[data-act=fwd]').addEventListener('click', () => this.step(1));
    $('[data-act=speed]').addEventListener('change', (e) => { this.speed = +e.target.value; if (this.video) this.video.playbackRate = this.speed; });
    this.range.addEventListener('input', () => { if (this.video) { this.video.currentTime = +this.range.value; this.dirty = true; } });
    $('.flash').addEventListener('change', (e) => { this.flashes = e.target.checked; this.dirty = true; });
    this.stage.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); this.toggle(); }
      if (e.code === 'ArrowLeft') { e.preventDefault(); this.step(-1); }
      if (e.code === 'ArrowRight') { e.preventDefault(); this.step(1); }
    });
    this.stage.addEventListener('click', () => this.stage.focus());
    this.renderControls();
    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(this.stage);
  }

  renderControls() {
    const roles = (this.camera?.roles || ['red', 'blue']).filter((r) => r in this.modes);
    const layers = this.root.querySelector('.layers');
    layers.innerHTML = roles.map((r) => `
      <div class="layer-row"><div class="who"><span class="swatch ${r}"></span>${r}</div>
        <div class="seg" data-role="${r}">${MODES.map((m) => `<button type="button" data-mode="${m.key}" class="${this.modes[r] === m.key ? 'on' : ''}"${m.key === 'mesh' && !this.meshes ? ' disabled title="3D bodies are loading"' : ''}>${m.label}</button>`).join('')}</div>
      </div>`).join('');
    layers.querySelectorAll('.seg').forEach((seg) => seg.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b || b.disabled) return;
      this.modes[seg.dataset.role] = b.dataset.mode; this.dirty = true; this.renderControls();
    }));
    const bgseg = this.root.querySelector('.bgseg');
    bgseg.innerHTML = BACKGROUNDS.map((b) => `<button type="button" data-bg="${b.key}" class="${this.bg === b.key ? 'on' : ''}">${b.label}</button>`).join('');
    bgseg.onclick = (e) => { const b = e.target.closest('button'); if (!b) return; this.bg = b.dataset.bg; this.dirty = true; this.renderControls(); };
  }

  async init() {
    const base = this.base;
    this.camera = parseJson(await (await fetch(base + 'camera.json')).text());
    this.fps = this.camera.fps;
    this.renderControls();
    this.renderEvents();
    const [w, h] = this.camera.web_size;
    this.stage.style.aspectRatio = `${w} / ${h}`;

    const video = document.createElement('video');
    video.src = base + 'layers.mp4';
    video.muted = true; video.playsInline = true; video.preload = 'auto';
    video.addEventListener('play', () => this.setPlaying(true));
    video.addEventListener('pause', () => this.setPlaying(false));
    video.addEventListener('seeked', () => { this.dirty = true; });
    video.addEventListener('loadeddata', () => { this.dirty = true; });
    this.video = video;
    const plate = new Image();
    plate.src = base + 'plate.jpg';
    await Promise.all([
      new Promise((ok, bad) => { video.addEventListener('loadedmetadata', ok, { once: true }); video.addEventListener('error', () => bad(new Error('video unavailable')), { once: true }); }),
      plate.decode(),
    ]);
    this.range.max = video.duration;
    this.range.step = 1 / this.fps;
    this.renderMarks();
    this.setupGL(plate);
    this.setupThree();
    this.loading.remove();
    if (this.seekTo != null) { video.currentTime = Math.max(0, this.seekTo); }
    this.loop();
    if (this.analysis.has_mesh !== false) {
      MeshData.load(base + 'mesh/').then((m) => { this.meshes = m; this.addMeshes(); this.renderControls(); this.dirty = true; })
        .catch(() => {});
    }
  }

  setupGL(plate) {
    const gl = this.glCanvas.getContext('webgl', { premultipliedAlpha: false });
    this.gl = gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    this.prog = prog;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const tex = (unit) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      return t;
    };
    this.vidTex = tex(0);
    this.plateTex = tex(1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, plate);
    gl.uniform1i(gl.getUniformLocation(prog, 'vid'), 0);
    gl.uniform1i(gl.getUniformLocation(prog, 'plate'), 1);
    gl.uniform3fv(gl.getUniformLocation(prog, 'colR'), hexToVec(ROLE_HEX.red));
    gl.uniform3fv(gl.getUniformLocation(prog, 'colB'), hexToVec(ROLE_HEX.blue));
    gl.uniform3fv(gl.getUniformLocation(prog, 'colF'), hexToVec(ROLE_HEX.referee));
    this.uModes = gl.getUniformLocation(prog, 'modes');
    this.uBg = gl.getUniformLocation(prog, 'bg');
  }

  setupThree() {
    const THREE = window.THREE;
    if (!THREE) return;
    const renderer = new THREE.WebGLRenderer({ canvas: this.threeCanvas, alpha: true, antialias: true });
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x303030, 1.0));
    const key = new THREE.DirectionalLight(0xffffff, 0.7);
    key.position.set(2, 6, 4);
    scene.add(key);
    const cam = new THREE.PerspectiveCamera();
    const [W, H] = this.camera.source_size;
    const K = this.camera.intrinsics;
    const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2], n = 0.05, f = 100;
    cam.projectionMatrix.set(
      2 * fx / W, 0, 1 - 2 * cx / W, 0,
      0, 2 * fy / H, 2 * cy / H - 1, 0,
      0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
      0, 0, -1, 0,
    );
    cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    const M = this.camera.world_from_cam;
    const worldFromCv = new THREE.Matrix4().set(...M[0], ...M[1], ...M[2], ...M[3]);
    const cvFromGl = new THREE.Matrix4().makeScale(1, -1, -1);
    cam.matrixAutoUpdate = false;
    cam.matrix.copy(worldFromCv.multiply(cvFromGl));
    cam.updateMatrixWorld(true);
    this.three = { THREE, renderer, scene, cam, meshes: {}, gloves: [] };
    const glove = () => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 }));
      m.visible = false; scene.add(m); return m;
    };
    this.three.gloves = [glove(), glove(), glove(), glove()];
    this.resize();
  }

  addMeshes() {
    if (!this.three) return;
    const { THREE, scene } = this.three;
    for (const role of this.meshes.names) {
      const mat = new THREE.MeshStandardMaterial({ color: ROLE_HEX[role] || 0x88cc88, roughness: .5, metalness: .05, transparent: true, opacity: 0.95 });
      const mesh = this.meshes.makeMesh(role, THREE, mat);
      mesh.visible = false;
      scene.add(mesh);
      this.three.meshes[role] = mesh;
    }
  }

  resize() {
    const r = this.stage.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(2, Math.round(r.width * dpr)), h = Math.max(2, Math.round(r.height * dpr));
    this.glCanvas.width = w; this.glCanvas.height = h;
    if (this.three) this.three.renderer.setSize(w, h, false);
    this.dirty = true;
  }

  sourceFrame() { return this.camera.start_frame + Math.round(this.video.currentTime * this.fps); }

  loop() {
    this.raf = requestAnimationFrame(() => this.loop());
    const v = this.video;
    if (!v || !this.gl || v.readyState < 2) return;
    if (!this.dirty && v.paused) return;
    this.dirty = false;
    const gl = this.gl;
    gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.vidTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, v);
    gl.uniform3f(this.uModes, MODE_INDEX[this.modes.red], MODE_INDEX[this.modes.blue], MODE_INDEX[this.modes.referee]);
    gl.uniform1f(this.uBg, BG_INDEX[this.bg]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const src = this.sourceFrame();
    this.drawThree(src);
    this.range.value = v.currentTime;
    this.timeEl.textContent = `${fmtTime(v.currentTime)} / ${fmtTime(v.duration)}`;
    this.hud.textContent = `frame ${src}`;
    this.updateSide(src);
  }

  drawThree(src) {
    if (!this.three) return;
    const { renderer, scene, cam, meshes, gloves } = this.three;
    const m = this.meshes;
    const i = m ? m.indexFor(src) : -1;
    for (const [role, mesh] of Object.entries(meshes)) {
      const show = this.modes[role] === 'mesh' && m.valid(role, i);
      mesh.visible = show;
      if (show) m.update(mesh, role, i);
    }
    gloves.forEach((g) => { g.visible = false; });
    if (this.flashes && m) {
      const win = Math.round(0.25 * this.fps);
      let k = 0;
      for (const e of this.events) {
        const c = e.contact_frame;
        if (c == null || src < c || src > c + win || k >= gloves.length) continue;
        const gl = m.gloves(e.attacker, i);
        if (!gl) continue;
        const p = gl[e.hand === 'left' ? 0 : 1];
        const g = gloves[k++];
        g.position.set(p[0], p[1], p[2]);
        g.material.color.setHex(OUTCOME_HEX[e.outcome] || 0xffffff);
        g.visible = true;
      }
    }
    renderer.render(scene, cam);
  }

  renderMarks() {
    const marks = this.root.querySelector('.marks');
    const d = this.video.duration || 1, start = this.camera.start_frame / this.fps;
    marks.innerHTML = this.events.filter((e) => e.outcome !== 'missed').map((e) => {
      const t = eventTime(e, this.fps) - start;
      return `<i style="left:${(100 * t / d).toFixed(2)}%;background:var(--${e.outcome})"></i>`;
    }).join('');
  }

  renderEvents() {
    const start = this.camera.start_frame / this.fps;
    this.evList.innerHTML = this.events.map((e, idx) => `
      <button class="ev" data-i="${idx}" type="button">
        <span class="t">${fmtTime(eventTime(e, this.fps) - start)}</span>
        <span class="dot" style="background:${ROLE_CSS[e.attacker] || 'var(--ink)'}"></span>
        <span>${e.attacker} ${e.hand}${e.target ? ' → ' + (e.target === 'torso' ? 'body' : e.target) : ''}</span>
        <span class="pill ${e.outcome}">${e.outcome}</span>
      </button>`).join('') || '<p class="muted" style="margin:0">Punches appear here once detected.</p>';
    this.evList.onclick = (ev) => {
      const b = ev.target.closest('.ev'); if (!b || !this.video) return;
      const e = this.events[+b.dataset.i];
      this.video.pause();
      this.video.currentTime = Math.max(0, eventTime(e, this.fps) - start - 0.4);
      this.stage.focus();
    };
  }

  updateSide(src) {
    const counts = {};
    for (const r of ['red', 'blue']) counts[r] = { thrown: 0, head: 0, torso: 0, blocked: 0 };
    let nowIdx = -1;
    this.events.forEach((e, idx) => {
      const f = e.contact_frame ?? e.peak_frame;
      if (Math.abs(f - src) <= this.fps * 0.3) nowIdx = idx;
      if (f > src || !counts[e.attacker]) return;
      const c = counts[e.attacker]; c.thrown++;
      if (e.outcome === 'landed' && e.target === 'head') c.head++;
      else if (e.outcome === 'landed' && e.target === 'torso') c.torso++;
      else if (e.outcome === 'blocked') c.blocked++;
    });
    const html = Object.entries(counts).map(([r, c]) => `<tr><td><span class="swatch ${r}"></span> ${r}</td><td>${c.thrown}</td><td>${c.head}</td><td>${c.torso}</td><td>${c.blocked}</td></tr>`).join('');
    if (html !== this._tally) { this.tallyBody.innerHTML = html; this._tally = html; }
    if (nowIdx !== this._now) {
      this.evList.querySelectorAll('.ev.now').forEach((x) => x.classList.remove('now'));
      const cur = this.evList.querySelector(`.ev[data-i="${nowIdx}"]`);
      if (cur) { cur.classList.add('now'); cur.scrollIntoView({ block: 'nearest' }); }
      this._now = nowIdx;
    }
  }

  setPlaying(on) {
    this.playBtn.innerHTML = on
      ? '<svg viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>'
      : '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>';
    this.playBtn.setAttribute('aria-label', on ? 'Pause' : 'Play');
  }

  toggle() {
    const v = this.video; if (!v) return;
    v.playbackRate = this.speed;
    if (v.paused) v.play(); else v.pause();
  }

  step(n) {
    const v = this.video; if (!v) return;
    v.pause();
    v.currentTime = Math.min(v.duration, Math.max(0, v.currentTime + n / this.fps));
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.resizeObs?.disconnect();
    if (this.video) { this.video.pause(); this.video.removeAttribute('src'); this.video.load(); }
    this.three?.renderer.dispose();
    this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
