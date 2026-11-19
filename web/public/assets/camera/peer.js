// Rig clock: phones on the same Wi-Fi or hotspot sync their clocks to each other directly.
//
// The site is far away (about 250 ms round trip from Kenya), so its clock can only promise about
// ±120 ms. Phones on one network reach each other in a few milliseconds. Every camera page polls
// /api/phone/peer to find the others; the page that joined first is the clock reference. The
// reference fixes K, its offset to the server measured once, and the rig's shared time is the
// reference's own clock plus K. Every other page opens a WebRTC data channel to the reference
// (local network only, no outside servers) and measures its offset with timestamp pings.
// A page that cannot reach the reference falls back to the server clock.
//
// It offers the interface camera.js uses from SharedClock: ready, now(), toShared(), offset,
// uncertainty, history, onChange(), plus mode ('reference' | 'peer' | 'server'), referenceName
// and lock()/unlock(), which hold the mode and reference steady through a take.

import { SharedClock, localNow, summarise } from '/assets/camera/clock.js';

const POLL_MS = 1000;
const FALLBACK_MS = 8000;
const REMEASURE_MS = 10000;
const CONNECT_TIMEOUT_MS = 10000;
const ICE_WAIT_MS = 3000;
const PING_GAP_MS = 25;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function randomId() {
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  return [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
}

/** Resolve once ICE gathering finishes or after `ms`, so one SDP carries every local candidate. */
function iceGathered(pc, ms) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => { pc.removeEventListener('icegatheringstatechange', check); resolve(); };
    const check = () => { if (pc.iceGatheringState === 'complete') done(); };
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(done, ms);
  });
}

export class RigClock {
  constructor(token) {
    this.token = token;
    this.session = randomId();
    this.server = new SharedClock('/api/clock');
    this.mode = null;
    this.offset = null;
    this.uncertainty = null;
    this.history = [];
    this.listeners = new Set();
    this.locked = false;
    this.K = null;
    this.reference = null;
    this.referenceName = null;
    this.peers = [];
    this.outbox = [];
    this.link = null; // { target, pc, channel, startedAt }
    this.served = new Map(); // session → RTCPeerConnection, while this page is the reference
    this.replies = null;
    this.seq = 0;
    this.failures = 0;
    this.startedAt = localNow();
    this.lastPeerOk = 0;
  }

  get ready() {
    return this.offset !== null;
  }

  now() {
    return localNow() + (this.offset ?? 0);
  }

  toShared(localMs) {
    return localMs + (this.offset ?? 0);
  }

  onChange(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** Hold the mode and reference through a take; measurements carry on in the current mode. */
  lock() {
    this.locked = true;
  }

  unlock() {
    this.locked = false;
    this.reconsider();
  }

  async start() {
    this.startedAt = localNow();
    this.server.onChange((m) => { if (this.mode === 'server') this.record({ ...m, source: 'server' }); });
    this.pollLoop();
    await this.server.measure(40).catch(() => {});
    this.server.keepMeasuring(REMEASURE_MS, 16);
    this.reconsider();
    this.measureLoop();
  }

  // ---------------------------------------------------------------- state

  record(m, fresh = false) {
    this.offset = m.offset_ms;
    this.uncertainty = m.uncertainty_ms;
    // A new mode starts a new history, so a take never interpolates across two clocks.
    if (fresh) this.history = [];
    this.history.push(m);
    if (this.history.length > 500) this.history.shift();
    this.listeners.forEach((fn) => fn(m));
  }

  setMode(mode, m) {
    this.mode = mode;
    this.record(m, true);
  }

  becomeReference() {
    if (!this.server.ready) return;
    this.closeLink();
    this.K = this.server.offset;
    const at = localNow();
    this.setMode('reference', {
      offset_ms: this.K, uncertainty_ms: 0, rtt_min_ms: 0, samples: 0,
      at_local_ms: at, at_shared_ms: at + this.K, source: 'reference', reference: this.session,
    });
  }

  fallBackToServer() {
    if (!this.server.ready) return;
    const last = this.server.history[this.server.history.length - 1];
    this.setMode('server', { ...last, source: 'server' });
  }

  /** Decide the mode from the latest poll. Called after each poll and on unlock. */
  reconsider() {
    if (this.locked && this.mode !== null) return;
    if (this.reference === null) {
      // The site has not answered yet; without it, only the server clock is possible.
      if (this.mode === null && localNow() - this.startedAt > FALLBACK_MS) this.fallBackToServer();
      return;
    }
    if (this.reference === this.session) {
      if (this.mode !== 'reference') this.becomeReference();
      return;
    }
    if (this.mode === 'reference') {
      // Another page is older (this one lost its place); follow it.
      this.mode = null;
      this.startedAt = localNow();
    }
    const linkAlive = this.link && this.link.target === this.reference;
    const linkStale = this.link && !this.link.open && localNow() - this.link.startedAt > CONNECT_TIMEOUT_MS;
    if (!linkAlive || linkStale) this.connect(this.reference);
    if (this.mode === null && localNow() - this.startedAt > FALLBACK_MS) this.fallBackToServer();
  }

  // ---------------------------------------------------------------- polling

  async pollLoop() {
    for (;;) {
      const signals = this.outbox.splice(0);
      try {
        const res = await fetch('/api/phone/peer', {
          method: 'POST',
          headers: { Authorization: 'Bearer ' + this.token, 'Content-Type': 'application/json' },
          body: JSON.stringify({ session: this.session, signals }),
          cache: 'no-store',
        });
        if (!res.ok) throw new Error('poll ' + res.status);
        const out = await res.json();
        this.peers = out.peers;
        this.reference = out.reference;
        this.referenceName = (out.peers.find((p) => p.session === out.reference) || {}).name || null;
        for (const msg of out.inbox) this.onSignal(msg.from, msg.data).catch(() => {});
        this.reconsider();
        this.listeners.forEach((fn) => fn(null));
      } catch (err) {
        this.outbox.unshift(...signals);
      }
      await sleep(POLL_MS);
    }
  }

  async onSignal(from, data) {
    if (!data || typeof data !== 'object') return;
    if (data.type === 'offer' && this.mode === 'reference') await this.answer(from, data.sdp);
    if (data.type === 'answer' && this.link && this.link.target === from && !this.link.answered) {
      this.link.answered = true;
      await this.link.pc.setRemoteDescription({ type: 'answer', sdp: data.sdp });
    }
  }

  // ---------------------------------------------------------------- reference side

  async answer(from, sdp) {
    this.served.get(from)?.close();
    const pc = new RTCPeerConnection({ iceServers: [] });
    this.served.set(from, pc);
    pc.ondatachannel = (e) => {
      const ch = e.channel;
      ch.onmessage = (m) => {
        const t1 = localNow();
        let p;
        try { p = JSON.parse(m.data); } catch (err) { return; }
        ch.send(JSON.stringify({ i: p.i, t0: p.t0, t1, t2: localNow(), k: this.K }));
      };
    };
    pc.onconnectionstatechange = () => {
      if (['failed', 'closed'].includes(pc.connectionState)) this.served.delete(from);
    };
    await pc.setRemoteDescription({ type: 'offer', sdp });
    await pc.setLocalDescription(await pc.createAnswer());
    await iceGathered(pc, ICE_WAIT_MS);
    this.outbox.push({ to: from, data: { type: 'answer', sdp: pc.localDescription.sdp } });
  }

  // ---------------------------------------------------------------- follower side

  closeLink() {
    if (this.link) this.link.pc.close();
    this.link = null;
  }

  async connect(target) {
    this.closeLink();
    const pc = new RTCPeerConnection({ iceServers: [] });
    const channel = pc.createDataChannel('clock', { ordered: false, maxRetransmits: 0 });
    const link = { target, pc, channel, startedAt: localNow(), open: false, answered: false };
    this.link = link;
    channel.onmessage = (e) => {
      const c3 = localNow();
      try { this.replies?.push({ ...JSON.parse(e.data), c3 }); } catch (err) { /* ignore */ }
    };
    channel.onopen = () => { link.open = true; this.measurePeer(40); };
    channel.onclose = () => { link.open = false; };
    await pc.setLocalDescription(await pc.createOffer());
    await iceGathered(pc, ICE_WAIT_MS);
    if (this.link === link) this.outbox.push({ to: target, data: { type: 'offer', sdp: pc.localDescription.sdp } });
  }

  /** A burst of pings to the reference; keeps the fastest quarter, as SharedClock does. */
  async measurePeer(count) {
    const link = this.link;
    if (!link || !link.open || this.replies) return false;
    this.replies = [];
    for (let n = 0; n < count && link.channel.readyState === 'open'; n++) {
      link.channel.send(JSON.stringify({ i: ++this.seq, t0: localNow() }));
      await sleep(PING_GAP_MS);
    }
    await sleep(300);
    const replies = this.replies;
    this.replies = null;
    const samples = replies
      .filter((p) => typeof p.k === 'number')
      .map((p) => ({ offset: ((p.t1 - p.t0) + (p.t2 - p.c3)) / 2 + p.k, rtt: (p.c3 - p.t0) - (p.t2 - p.t1), at: p.c3 }));
    if (samples.length < Math.min(8, count)) {
      this.failures += 1;
      if (this.failures >= 2 && !this.locked && this.mode === 'peer') this.fallBackToServer();
      return false;
    }
    this.failures = 0;
    this.lastPeerOk = localNow();
    const m = summarise(samples);
    m.at_local_ms = localNow();
    m.at_shared_ms = m.at_local_ms + m.offset_ms;
    m.source = 'peer';
    m.reference = link.target;
    if (this.mode === 'peer') this.record(m);
    else if (!this.locked || this.mode === null) this.setMode('peer', m);
    return true;
  }

  async measureLoop() {
    for (;;) {
      await sleep(REMEASURE_MS);
      if (this.link && this.link.open) await this.measurePeer(16);
    }
  }
}
