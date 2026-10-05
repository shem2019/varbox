// Per-frame decimated body meshes exported by the GPU (int16-quantised positions, shared faces).

export const ROLE_HEX = { red: 0xe23b4e, blue: 0x3b6fe8, referee: 0xb7bcc7 };
export const ROLE_CSS = { red: 'var(--red)', blue: 'var(--blue)', referee: 'var(--ref)' };
export const OUTCOME_HEX = { landed: 0x2fbf71, blocked: 0xf0b43a, missed: 0x8b93a3 };

export class MeshData {
  static async load(base) {
    const manifest = parseJson(await (await fetch(base + 'manifest.json')).text());
    const roles = {};
    await Promise.all(manifest.roles.map(async (r) => {
      const [v, f] = await Promise.all([
        fetch(base + r.name + '_verts.bin').then((x) => x.arrayBuffer()),
        fetch(base + r.name + '_faces.bin').then((x) => x.arrayBuffer()),
      ]);
      roles[r.name] = { meta: r, q: new Int16Array(v), faces: new Uint32Array(f), pos: new Float32Array(r.vertex_count * 3) };
    }));
    return new MeshData(manifest, roles);
  }

  constructor(manifest, roles) {
    this.manifest = manifest;
    this.roles = roles;
    this.frames = manifest.frames;
  }

  get names() { return Object.keys(this.roles); }

  /** Index into the mesh sequence closest to a source-video frame number. */
  indexFor(sourceFrame) {
    const f = this.frames;
    let lo = 0, hi = f.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (f[mid] < sourceFrame) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && Math.abs(f[lo - 1] - sourceFrame) < Math.abs(f[lo] - sourceFrame)) lo -= 1;
    return lo;
  }

  valid(role, i) { return this.roles[role]?.meta.valid[i] === 1; }

  /** Decoded world positions for one role at one sequence index (reuses one buffer). */
  positions(role, i) {
    const r = this.roles[role];
    const n = r.meta.vertex_count * 3, base = i * n, lo = r.meta.bounds_lo, hi = r.meta.bounds_hi;
    for (let k = 0; k < n; k++) {
      const c = k % 3;
      r.pos[k] = (r.q[base + k] + 32767) / 65534 * (hi[c] - lo[c]) + lo[c];
    }
    return r.pos;
  }

  gloves(role, i) { return this.roles[role]?.meta.gloves?.[i] || null; }

  gaps(role, i) {
    const g = this.roles[role]?.meta.gaps;
    if (!g) return null;
    return { left: g.left?.[i] || null, right: g.right?.[i] || null };
  }

  /** A three.js mesh whose geometry is refreshed each frame. */
  makeMesh(role, THREE, material) {
    const r = this.roles[role];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(r.pos, 3));
    geo.setIndex(new THREE.BufferAttribute(r.faces, 1));
    const mesh = new THREE.Mesh(geo, material || new THREE.MeshStandardMaterial({ color: ROLE_HEX[role] || 0x88cc88, roughness: .55, metalness: .05 }));
    mesh.frustumCulled = false;
    return mesh;
  }

  update(mesh, role, i) {
    this.positions(role, i);
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
  }
}

/** JSON.parse that tolerates NaN from older packages. */
export function parseJson(text) { return JSON.parse(text.replace(/\bNaN\b|-?Infinity\b/g, 'null')); }

export function eventTime(e, fps) { return (e.contact_frame ?? e.peak_frame) / fps; }

export function fmtTime(s) {
  if (!isFinite(s)) return '0:00.0';
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, '0')}`;
}
