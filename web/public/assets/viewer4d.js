// 4D viewer: every frame's 3D bodies on the ring floor; orbit, slow down, solo a boxer.

import { MeshData, ROLE_HEX, OUTCOME_HEX, eventTime, fmtTime } from './meshdata.js';

export class Viewer4D {
  constructor(root, { base, events }) {
    this.root = root;
    this.base = base;
    this.events = events;
    this.cur = 0;
    this.playing = false;
    this.speed = 1;
    this.solo = null;
    this.raf = 0;
    this.root.innerHTML = `
      <div class="player">
        <div>
          <div class="stage" tabindex="0" aria-label="3D replay" style="aspect-ratio:16/9"><div class="loading" role="status"><div class="spinner"></div><div>Loading 3D bodies<small>every frame of both boxers</small></div></div><div class="hud"></div></div>
          <div class="transport">
            <button class="btn icon-btn primary" data-act="play" aria-label="Play"><svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg></button>
            <div class="scrub"><div class="marks"></div><input type="range" min="0" max="0" value="0" aria-label="Timeline"></div>
            <span class="time">0:00.0</span>
            <select class="input" data-act="speed" style="width:auto;height:36px" aria-label="Playback speed">
              <option value="0.1">0.1×</option><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option>
            </select>
          </div>
        </div>
        <div class="panel-side">
          <div class="card"><h4>Boxers</h4><div class="who-list" style="display:grid;gap:8px"></div>
            <p class="muted" style="margin:0;font-size:13px">Drag to orbit, scroll or pinch to zoom. Solo keeps the camera on one boxer.</p></div>
          <div class="card"><h4>Glove gap, this frame</h4><table class="tally gaps"><thead><tr><th>Glove</th><th>Head</th><th>Body</th><th>Guard</th></tr></thead><tbody></tbody></table>
            <p class="muted" style="margin:0;font-size:13px">Centimetres from glove surface to the opponent. Zero or below means contact.</p></div>
        </div>
      </div>`;
    this.stage = root.querySelector('.stage');
    this.range = root.querySelector('input[type=range]');
    this.timeEl = root.querySelector('.time');
    this.playBtn = root.querySelector('[data-act=play]');
    this.playBtn.addEventListener('click', () => { this.playing = !this.playing; this.renderPlay(); });
    root.querySelector('[data-act=speed]').addEventListener('change', (e) => { this.speed = +e.target.value; });
    this.range.addEventListener('input', () => { this.cur = +this.range.value; this.setFrame(this.cur); });
    this.stage.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); this.playing = !this.playing; this.renderPlay(); }
      if (e.code === 'ArrowRight') { this.cur = Math.min(this.n - 1, this.cur + 1); this.setFrame(this.cur); }
      if (e.code === 'ArrowLeft') { this.cur = Math.max(0, this.cur - 1); this.setFrame(this.cur); }
    });
    this.init().catch((err) => { root.querySelector('.loading').innerHTML = `<div>The 3D bodies stopped loading<small>${err.message}. Reload the page to try again.</small></div>`; });
  }

  async init() {
    const THREE = window.THREE;
    this.data = await MeshData.load(this.base);
    this.n = this.data.frames.length;
    this.fps = this.data.manifest.fps;
    this.sourceFps = this.data.manifest.source_fps;
    this.range.max = this.n - 1;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.stage.prepend(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e12);
    const cam = new THREE.PerspectiveCamera(40, 16 / 9, 0.05, 100);
    cam.position.set(0, 2.2, 6.5);
    const controls = new THREE.OrbitControls(cam, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xdfe6f5, 0x202020, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 0.8); key.position.set(3, 6, 4); scene.add(key);
    scene.add(new THREE.GridHelper(8, 16, 0x3a4150, 0x22272f));
    this.three = { THREE, renderer, scene, cam, controls };
    this.meshes = {};
    this.gloves = {};
    for (const role of this.data.names) {
      const mesh = this.data.makeMesh(role, THREE);
      scene.add(mesh);
      this.meshes[role] = mesh;
      this.gloves[role] = [0, 1].map(() => {
        const g = new THREE.Mesh(new THREE.SphereGeometry(0.085, 20, 14), new THREE.MeshStandardMaterial({ color: ROLE_HEX[role], transparent: true, opacity: 0.35 }));
        scene.add(g); return g;
      });
    }
    this.root.querySelector('.loading').remove();
    this.renderWho();
    this.renderMarks();
    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(this.stage);
    this.resize();
    this.setFrame(0);
    let last = performance.now(), acc = 0;
    const tick = (now) => {
      this.raf = requestAnimationFrame(tick);
      const dt = (now - last) / 1000; last = now;
      if (this.playing) {
        acc += dt * this.fps * this.speed;
        if (acc >= 1) { const s = Math.floor(acc); acc -= s; this.cur = (this.cur + s) % this.n; this.setFrame(this.cur); }
      }
      controls.update();
      renderer.render(scene, cam);
    };
    this.raf = requestAnimationFrame(tick);
  }

  renderPlay() {
    this.playBtn.innerHTML = this.playing
      ? '<svg viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>'
      : '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>';
  }

  renderWho() {
    const box = this.root.querySelector('.who-list');
    box.innerHTML = this.data.names.map((r) => `
      <div style="display:flex;align-items:center;gap:10px">
        <span class="swatch ${r}"></span><span style="font-weight:600;text-transform:capitalize;flex:1">${r}</span>
        <button class="btn sm ${this.solo === r ? 'primary' : ''}" data-solo="${r}" type="button">${this.solo === r ? 'Show all' : 'Solo'}</button>
      </div>`).join('');
    box.onclick = (e) => {
      const b = e.target.closest('[data-solo]'); if (!b) return;
      this.solo = this.solo === b.dataset.solo ? null : b.dataset.solo;
      this.renderWho(); this.setFrame(this.cur);
    };
  }

  renderMarks() {
    const first = this.data.frames[0], last = this.data.frames[this.n - 1];
    this.root.querySelector('.marks').innerHTML = this.events.filter((e) => e.outcome !== 'missed').map((e) => {
      const f = e.contact_frame ?? e.peak_frame;
      return `<i style="left:${(100 * (f - first) / Math.max(1, last - first)).toFixed(2)}%;background:var(--${e.outcome})"></i>`;
    }).join('');
  }

  resize() {
    const r = this.stage.getBoundingClientRect();
    this.three.renderer.setSize(r.width, r.height, false);
    this.three.cam.aspect = r.width / r.height;
    this.three.cam.updateProjectionMatrix();
  }

  setFrame(i) {
    const src = this.data.frames[i];
    for (const [role, mesh] of Object.entries(this.meshes)) {
      const ok = this.data.valid(role, i) && (!this.solo || this.solo === role);
      mesh.visible = ok;
      this.gloves[role].forEach((g) => { g.visible = ok; g.material.opacity = 0.35; g.material.color.setHex(ROLE_HEX[role]); });
      if (!ok) continue;
      this.data.update(mesh, role, i);
      const gl = this.data.gloves(role, i);
      if (gl) { this.gloves[role][0].position.set(...gl[0]); this.gloves[role][1].position.set(...gl[1]); }
    }
    const win = Math.round(0.25 * this.sourceFps);
    for (const e of this.events) {
      const c = e.contact_frame;
      if (c == null || src < c || src > c + win || e.outcome === 'missed') continue;
      const g = this.gloves[e.attacker]?.[e.hand === 'left' ? 0 : 1];
      if (g && g.visible) { g.material.opacity = 0.95; g.material.color.setHex(OUTCOME_HEX[e.outcome]); }
    }
    if (this.solo && this.meshes[this.solo]?.visible) {
      const geo = this.meshes[this.solo].geometry; geo.computeBoundingSphere();
      const c = geo.boundingSphere.center;
      this.three.controls.target.lerp(new this.three.THREE.Vector3(c.x, 1.0, c.z), 0.25);
    }
    const gaps = [];
    for (const role of this.data.names) {
      const g = this.data.gaps(role, i); if (!g) continue;
      for (const hand of ['left', 'right']) {
        const v = g[hand]; if (!v) continue;
        const cell = (x) => (x >= 8.9 ? '·' : Math.round(x * 100));
        gaps.push(`<tr><td><span class="swatch ${role}"></span> ${role} ${hand}</td><td>${cell(v[0])}</td><td>${cell(v[1])}</td><td>${cell(v[2])}</td></tr>`);
      }
    }
    this.root.querySelector('.gaps tbody').innerHTML = gaps.join('');
    this.range.value = i;
    this.timeEl.textContent = fmtTime((src - this.data.frames[0]) / this.sourceFps);
    this.root.querySelector('.hud').textContent = `frame ${src}`;
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.resizeObs?.disconnect();
    this.three?.renderer.dispose();
  }
}
