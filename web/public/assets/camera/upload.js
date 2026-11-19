// Uploads finished takes from the phone's storage to VAR Box, resuming after any drop.
//
// The queue lives in localStorage and the files in the Origin Private File System, so a reload,
// a dead network or a closed tab only delays an upload. A take's files are deleted from the phone
// once the server confirms it holds the whole video and its sidecar.

const CHUNK = 8 * 1024 * 1024;
const QUEUE_KEY = 'varbox.camera.uploads';

function readQueue() {
  try { return JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]'); } catch (err) { return []; }
}
function writeQueue(q) {
  try { localStorage.setItem(QUEUE_KEY, JSON.stringify(q)); } catch (err) { /* private mode: in-memory only */ }
}

async function takesDir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('takes', { create: true });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class Uploader {
  constructor(token, onChange) {
    this.token = token;
    this.onChange = onChange || (() => {});
    this.running = false;
    this.status = {}; // name → { pct, state, error }
    window.addEventListener('online', () => this.run());
  }

  get queue() {
    return readQueue();
  }

  /** Add a finished take. `entry` carries name, rec_uid, letter, size and the recording fields. */
  add(entry) {
    const q = readQueue().filter((e) => e.name !== entry.name);
    q.push({ ...entry, upload_id: null, done: false });
    writeQueue(q);
    this.run();
  }

  update(name, patch) {
    writeQueue(readQueue().map((e) => (e.name === name ? { ...e, ...patch } : e)));
  }

  setStatus(name, s) {
    this.status[name] = { ...(this.status[name] || {}), ...s };
    this.onChange(this.status);
  }

  async api(path, opts = {}) {
    const res = await fetch('/api/phone/' + path, {
      ...opts,
      headers: { Authorization: 'Bearer ' + this.token, ...(opts.headers || {}) },
      cache: 'no-store',
    });
    let body = null;
    try { body = await res.json(); } catch (err) { body = null; }
    return { status: res.status, body };
  }

  /** Work through the queue until it is empty; network errors back off and retry. */
  async run() {
    if (this.running) return;
    this.running = true;
    try {
      let wait = 2000;
      for (;;) {
        const next = readQueue().find((e) => !e.done);
        if (!next) return;
        try {
          await this.uploadOne(next);
          wait = 2000;
        } catch (err) {
          if (err.fatal) {
            this.update(next.name, { done: true, error: err.message });
            this.setStatus(next.name, { state: 'failed', error: err.message });
            continue;
          }
          this.setStatus(next.name, { state: 'waiting', error: navigator.onLine ? err.message : 'Offline. The take is safe on this phone.' });
          await sleep(wait);
          wait = Math.min(wait * 2, 30000);
        }
      }
    } finally {
      this.running = false;
    }
  }

  async uploadOne(entry) {
    const dir = await takesDir();
    let file;
    let sidecar;
    try {
      file = await (await dir.getFileHandle(entry.name + '.mp4')).getFile();
      sidecar = await (await (await dir.getFileHandle(entry.name + '.json')).getFile()).text();
    } catch (err) {
      throw Object.assign(new Error('The take is no longer on this phone.'), { fatal: true });
    }
    this.setStatus(entry.name, { state: 'uploading', pct: 0, error: null });

    let uploadId = entry.upload_id;
    if (!uploadId) {
      const r = await this.api('uploads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rec_uid: entry.rec_uid, letter: entry.letter, name: entry.name + '.mp4', size: file.size,
          fps: entry.fps, title: entry.title, expected_cameras: entry.expected_cameras, device: entry.device,
        }),
      });
      if (r.status === 401) throw Object.assign(new Error('This phone was revoked. Add it again on the Team page.'), { fatal: true });
      if (r.status !== 200) throw new Error(r.body?.error || 'The server did not open the upload.');
      uploadId = r.body.upload.id;
      this.update(entry.name, { upload_id: uploadId });
    }

    // The sidecar goes first: the last video byte completes the recording, and the pending task it
    // creates reads the sidecar.
    const sc = await this.api(`uploads/${uploadId}/sidecar`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: sidecar });
    if (sc.status !== 200) throw new Error(sc.body?.error || 'The sidecar did not upload.');

    // Where the server stands, so a resumed upload carries on instead of starting over.
    const st = await this.api('uploads/' + uploadId);
    if (st.status !== 200) throw new Error(st.body?.error || 'The server did not report the upload.');
    let offset = st.body.received;
    while (offset < file.size) {
      const part = file.slice(offset, Math.min(file.size, offset + CHUNK));
      const r = await this.api(`uploads/${uploadId}?offset=${offset}`, { method: 'PUT', body: part });
      if (r.status !== 200 && r.status !== 409) throw new Error(r.body?.error || 'Upload interrupted.');
      offset = r.body.received;
      this.setStatus(entry.name, { state: 'uploading', pct: Math.round((offset * 100) / file.size) });
    }
    const done = await this.api('uploads/' + uploadId);
    if (done.status !== 200 || done.body.received !== file.size || !done.body.sidecar) {
      throw new Error('The server does not yet hold the whole take.');
    }
    this.update(entry.name, { done: true });
    this.setStatus(entry.name, { state: 'uploaded', pct: 100 });
    await dir.removeEntry(entry.name + '.mp4').catch(() => {});
    await dir.removeEntry(entry.name + '.json').catch(() => {});
  }
}
