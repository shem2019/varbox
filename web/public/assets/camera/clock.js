// Shared clock: this device's offset to the VAR Box server, measured NTP-style over HTTP.
//
// Every phone measures against the same server, so their shared-clock times agree with each other
// to within their uncertainties, whatever the server's own clock says. Times are milliseconds on
// `performance.timeOrigin + performance.now()`, which is monotonic for the life of the page.

export function localNow() {
  return performance.timeOrigin + performance.now();
}

/**
 * One exchange: c0 sent, t1 server received, t2 server replied, c3 reply arrived.
 * offset = ((t1 - c0) + (t2 - c3)) / 2, round trip = (c3 - c0) - (t2 - t1).
 */
async function sample(url) {
  const c0 = localNow();
  const res = await fetch(url, { cache: 'no-store' });
  const { t1, t2 } = await res.json();
  const c3 = localNow();
  return { offset: ((t1 - c0) + (t2 - c3)) / 2, rtt: (c3 - c0) - (t2 - t1), at: c3 };
}

/** Median of the offsets from the fastest quarter of the samples; uncertainty is half the fastest round trip. */
export function summarise(samples) {
  const sorted = [...samples].sort((a, b) => a.rtt - b.rtt);
  const keep = sorted.slice(0, Math.max(1, Math.ceil(sorted.length / 4)));
  const offsets = keep.map((s) => s.offset).sort((a, b) => a - b);
  const mid = offsets.length >> 1;
  const offset = offsets.length % 2 ? offsets[mid] : (offsets[mid - 1] + offsets[mid]) / 2;
  return {
    offset_ms: offset,
    uncertainty_ms: Math.max(0, sorted[0].rtt / 2),
    rtt_min_ms: sorted[0].rtt,
    samples: samples.length,
  };
}

export class SharedClock {
  constructor(url = '/api/clock') {
    this.url = url;
    this.offset = null;
    this.uncertainty = null;
    this.history = []; // every measurement, so the pipeline can correct drift over a take
    this.listeners = new Set();
    this.timer = null;
  }

  get ready() {
    return this.offset !== null;
  }

  /** Shared-clock time now. */
  now() {
    return localNow() + (this.offset ?? 0);
  }

  /** Convert a local time (same base as `localNow`) to the shared clock. */
  toShared(localMs) {
    return localMs + (this.offset ?? 0);
  }

  async measure(count = 40) {
    const samples = [];
    for (let i = 0; i < count; i++) {
      try {
        samples.push(await sample(this.url));
      } catch (err) {
        // A dropped request is skipped; the fastest survivors still give the answer.
      }
    }
    if (samples.length < Math.min(8, count)) {
      throw new Error('The clock server could not be reached.');
    }
    const m = summarise(samples);
    m.at_local_ms = localNow();
    m.at_shared_ms = m.at_local_ms + m.offset_ms;
    this.offset = m.offset_ms;
    this.uncertainty = m.uncertainty_ms;
    this.history.push(m);
    if (this.history.length > 500) this.history.shift();
    this.listeners.forEach((fn) => fn(m));
    return m;
  }

  /** Re-measure on an interval with fewer samples. A failed round leaves the last good offset. */
  keepMeasuring(everyMs = 10000, count = 16) {
    this.stop();
    this.timer = setInterval(() => { this.measure(count).catch(() => {}); }, everyMs);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  onChange(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}
