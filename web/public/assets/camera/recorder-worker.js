// Recording worker: encodes camera frames itself, stamps each with its capture time and writes a
// fragmented MP4 to the phone's own storage (the Origin Private File System) as it records.
//
// Nothing here waits on the network. The MP4 timestamps are the frames' capture times relative to
// the first frame, and the sidecar gives every frame's capture time on the shared clock, so the
// cameras line up by clock however loud the arena is.

import { Muxer, StreamTarget } from '/assets/vendor/mp4-muxer-5.2.2/mp4-muxer.js';

const KEYFRAME_SECONDS = 2;
const SIDECAR_EVERY_MS = 2000;
const MAX_CAPTURE_LAG_MS = 250;

let take = null;

const localNow = () => performance.timeOrigin + performance.now();

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'start') await start(msg);
    else if (msg.type === 'frame') onVideoFrame(msg.frame, msg.captureLocal, msg.source);
    else if (msg.type === 'clock') take?.clockHistory.push(msg.measurement);
    else if (msg.type === 'stop') await stop(Boolean(msg.interrupted));
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message || err) });
  }
};

async function start(msg) {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle('takes', { create: true });
  const videoHandle = await dir.getFileHandle(msg.name + '.mp4', { create: true });
  const sidecarHandle = await dir.getFileHandle(msg.name + '.json', { create: true });
  take = {
    name: msg.name,
    meta: msg.meta,
    fps: msg.fps,
    bitrate: msg.bitrate,
    mainTimeOrigin: msg.mainTimeOrigin,
    clockHistory: [...msg.clockHistory],
    file: await videoHandle.createSyncAccessHandle(),
    sidecarFile: await sidecarHandle.createSyncAccessHandle(),
    size: 0,
    muxer: null,
    encoder: null,
    audioEncoder: null,
    audioWanted: Boolean(msg.audio),
    audioInfo: null,
    pendingVideo: [],
    pendingAudio: [],
    raw: new Map(), // relative timestamp (µs) → frame record, until the encoder returns it
    frames: [], // frame records in the order they were written to the file
    firstTs: null,
    firstArrival: null,
    lastRel: -1,
    count: 0,
    dropped: 0,
    reordered: 0,
    firstAudioTs: null,
    firstAudioArrival: null,
    stopping: false,
    startedLocal: localNow(),
    lastSidecar: 0,
    codec: null,
  };
  take.file.truncate(0);
  if (msg.videoReadable) pump(msg.videoReadable.getReader(), (f) => onVideoFrame(f, null, null));
  if (msg.audioReadable) pump(msg.audioReadable.getReader(), onAudioData);
  self.postMessage({ type: 'started' });
}

async function pump(reader, handle) {
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    if (!take || take.stopping) {
      value.close();
      reader.cancel().catch(() => {});
      return;
    }
    handle(value);
  }
}

// ------------------------------------------------------------------ video

async function pickVideoCodec(width, height, fps, bitrate) {
  // High profile at levels that cover 1080p60 down to 720p30, then Main and Baseline.
  for (const codec of ['avc1.640033', 'avc1.64002A', 'avc1.640028', 'avc1.4D4028', 'avc1.42E028', 'avc1.42E01F']) {
    const config = {
      codec, width, height, bitrate, framerate: fps,
      avc: { format: 'avc' }, hardwareAcceleration: 'prefer-hardware', latencyMode: 'quality',
    };
    try {
      const { supported } = await VideoEncoder.isConfigSupported(config);
      if (supported) return config;
    } catch (err) { /* try the next */ }
  }
  throw new Error('This browser cannot encode H.264 video.');
}

function onVideoFrame(frame, captureLocal, source) {
  const t = take;
  if (!t || t.stopping) { frame.close(); return; }
  const arrival = localNow();
  let metaCapture = null;
  try {
    const md = typeof frame.metadata === 'function' ? frame.metadata() : null;
    if (md && typeof md.captureTime === 'number') metaCapture = md.captureTime;
  } catch (err) { /* metadata() is optional */ }
  const ts = frame.timestamp;
  if (t.firstTs === null) {
    t.firstTs = ts;
    t.firstArrival = arrival;
    initVideo(frame.displayWidth, frame.displayHeight);
  }
  let rel = ts - t.firstTs;
  if (rel <= t.lastRel) { rel = t.lastRel + 1; t.reordered += 1; }
  // An encoder more than a second behind drops frames rather than running the phone out of memory.
  if (t.encoder && t.encoder.encodeQueueSize > t.fps) { t.dropped += 1; frame.close(); return; }
  t.lastRel = rel;
  t.raw.set(rel, { ts, arrival, metaCapture, captureLocal, source });
  const keyFrame = t.count % Math.max(1, Math.round(t.fps * KEYFRAME_SECONDS)) === 0;
  const stamped = new VideoFrame(frame, { timestamp: rel });
  frame.close();
  t.count += 1;
  t.videoReady.then(() => {
    if (t.encoder.state === 'configured') t.encoder.encode(stamped, { keyFrame });
    stamped.close();
  });
}

function initVideo(width, height) {
  const t = take;
  t.width = width;
  t.height = height;
  t.videoReady = (async () => {
    const config = await pickVideoCodec(width, height, t.fps, t.bitrate);
    t.codec = config.codec;
    t.encoder = new VideoEncoder({
      output: (chunk, meta) => {
        const rec = t.raw.get(chunk.timestamp);
        t.raw.delete(chunk.timestamp);
        if (rec) t.frames.push(rec);
        addVideo(chunk, meta);
      },
      error: (err) => self.postMessage({ type: 'error', message: 'Video encoder: ' + err.message }),
    });
    t.encoder.configure(config);
  })();
  // Build the muxer once the audio format is known, or after a short wait without audio.
  setTimeout(() => ensureMuxer(true), 1500);
}

function addVideo(chunk, meta) {
  const t = take;
  if (!ensureMuxer(false)) { t.pendingVideo.push([chunk, meta]); return; }
  t.muxer.addVideoChunk(chunk, meta);
  maybeWriteSidecar();
}

// ------------------------------------------------------------------ audio

async function onAudioData(data) {
  const t = take;
  if (t.firstTs === null || t.stopping) { data.close(); return; } // audio starts with the first video frame
  const arrival = localNow();
  if (t.firstAudioTs === null) {
    t.firstAudioTs = data.timestamp;
    t.firstAudioArrival = arrival;
    t.audioInfo = { sampleRate: data.sampleRate, numberOfChannels: data.numberOfChannels };
    t.audioReady = initAudio(t.audioInfo);
  }
  // Audio is a cross-check only; it is placed by arrival against the first video frame.
  const rel = Math.round((t.firstAudioArrival - t.firstArrival) * 1000) + (data.timestamp - t.firstAudioTs);
  // AudioData cannot be re-stamped, so copy its samples (planar) into a new one.
  const merged = new Float32Array(data.numberOfFrames * data.numberOfChannels);
  for (let p = 0; p < data.numberOfChannels; p++) {
    data.copyTo(merged.subarray(p * data.numberOfFrames, (p + 1) * data.numberOfFrames), { planeIndex: p, format: 'f32-planar' });
  }
  const restamped = new AudioData({
    format: 'f32-planar', sampleRate: data.sampleRate, numberOfFrames: data.numberOfFrames,
    numberOfChannels: data.numberOfChannels, timestamp: Math.max(0, rel), data: merged,
  });
  data.close();
  await t.audioReady;
  if (t.audioEncoder && t.audioEncoder.state === 'configured') t.audioEncoder.encode(restamped);
  restamped.close();
}

async function initAudio(info) {
  const t = take;
  for (const [codec, mux] of [['mp4a.40.2', 'aac'], ['opus', 'opus']]) {
    const config = { codec, sampleRate: info.sampleRate, numberOfChannels: info.numberOfChannels, bitrate: 128000 };
    try {
      const { supported } = await AudioEncoder.isConfigSupported(config);
      if (!supported) continue;
      t.audioCodec = mux;
      t.audioEncoder = new AudioEncoder({
        output: (chunk, meta) => {
          if (!ensureMuxer(false)) { t.pendingAudio.push([chunk, meta]); return; }
          t.muxer.addAudioChunk(chunk, meta);
        },
        error: () => { t.audioEncoder = null; },
      });
      t.audioEncoder.configure(config);
      ensureMuxer(false);
      return;
    } catch (err) { /* try the next */ }
  }
  t.audioWanted = false; // no usable audio encoder; the take records video only
  ensureMuxer(false);
}

// ------------------------------------------------------------------ file

function ensureMuxer(timedOut) {
  const t = take;
  if (t.muxer) return true;
  if (!t.codec) return false;
  const audioKnown = !t.audioWanted || Boolean(t.audioCodec) || timedOut;
  if (!audioKnown) return false;
  const withAudio = Boolean(t.audioCodec);
  t.muxer = new Muxer({
    target: new StreamTarget({
      onData: (data, position) => {
        t.file.write(data, { at: position });
        t.size = Math.max(t.size, position + data.byteLength);
      },
      chunked: true,
    }),
    video: { codec: 'avc', width: t.width, height: t.height, frameRate: t.fps },
    audio: withAudio ? { codec: t.audioCodec, ...t.audioInfo } : undefined,
    fastStart: 'fragmented',
    minFragmentDuration: 1,
    firstTimestampBehavior: 'cross-track-offset',
  });
  if (!withAudio) t.audioEncoder = null;
  for (const [c, m] of t.pendingVideo.splice(0)) t.muxer.addVideoChunk(c, m);
  if (withAudio) for (const [c, m] of t.pendingAudio.splice(0)) t.muxer.addAudioChunk(c, m);
  return true;
}

async function stop(interrupted) {
  const t = take;
  if (!t || t.stopping) return;
  t.stopping = true;
  try {
    if (t.videoReady) await t.videoReady;
    if (t.encoder && t.encoder.state === 'configured') await t.encoder.flush();
    if (t.audioReady) await t.audioReady;
    if (t.audioEncoder && t.audioEncoder.state === 'configured') await t.audioEncoder.flush();
    ensureMuxer(true);
    t.muxer?.finalize();
  } finally {
    t.file.flush();
    t.file.close();
    const sidecar = buildSidecar(interrupted);
    writeSidecar(sidecar);
    t.sidecarFile.close();
    self.postMessage({ type: 'done', name: t.name, size: t.size, frames: t.frames.length, sidecar });
    take = null;
  }
}

function maybeWriteSidecar() {
  const t = take;
  const now = localNow();
  if (now - t.lastSidecar < SIDECAR_EVERY_MS) return;
  t.lastSidecar = now;
  t.file.flush();
  // Written as the take runs, marked interrupted, so a page killed mid-take still leaves usable times.
  writeSidecar(buildSidecar(true));
  self.postMessage({ type: 'progress', frames: t.frames.length, size: t.size, dropped: t.dropped });
}

function writeSidecar(sidecar) {
  const bytes = new TextEncoder().encode(JSON.stringify(sidecar));
  take.sidecarFile.truncate(0);
  take.sidecarFile.write(bytes, { at: 0 });
  take.sidecarFile.flush();
}

// ------------------------------------------------------------------ timing

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[s.length >> 1] : NaN;
};
const quantile = (xs, q) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : NaN;
};

/**
 * Which clock the frames' own timestamps are on. Each candidate maps a frame to a local capture
 * time; the right one puts every frame shortly before it arrived here (0–250 ms), steadily.
 */
function captureTimes(frames, mainTimeOrigin) {
  if (frames.length === 0) return { source: 'web-none', local: [], lag: null };
  if (frames[0].captureLocal !== null && frames[0].captureLocal !== undefined) {
    return { source: frames[0].source || 'web-rvfc-capture', local: frames.map((f) => f.captureLocal), lag: lagStats(frames, frames.map((f) => f.captureLocal)) };
  }
  const candidates = [];
  if (frames.every((f) => f.metaCapture !== null)) {
    candidates.push(['web-videoframe-metadata', frames.map((f) => mainTimeOrigin + f.metaCapture)]);
    candidates.push(['web-videoframe-metadata-worker', frames.map((f) => performance.timeOrigin + f.metaCapture)]);
  }
  candidates.push(['web-videoframe-capture', frames.map((f) => mainTimeOrigin + f.ts / 1000)]);
  candidates.push(['web-videoframe-capture-worker', frames.map((f) => performance.timeOrigin + f.ts / 1000)]);
  candidates.push(['web-videoframe-capture-epoch', frames.map((f) => f.ts / 1000)]);
  for (const [source, local] of candidates) {
    const lag = lagStats(frames, local);
    if (lag.min >= -1 && lag.p95 <= MAX_CAPTURE_LAG_MS) return { source, local, lag };
  }
  // The timestamps keep the frames' spacing but sit on an unknown clock: anchor them on the
  // fastest delivery, so the offset includes that phone's shortest camera-to-page delay.
  const rel = frames.map((f) => (f.ts - frames[0].ts) / 1000);
  const steady = rel.every((r, i) => i === 0 || r > rel[i - 1]);
  if (steady) {
    const anchor = Math.min(...frames.map((f, i) => f.arrival - rel[i]));
    const local = rel.map((r) => anchor + r);
    return { source: 'web-frame-intervals', local, lag: lagStats(frames, local) };
  }
  const local = frames.map((f) => f.arrival);
  return { source: 'web-arrival', local, lag: lagStats(frames, local) };
}

function lagStats(frames, local) {
  const lags = frames.map((f, i) => f.arrival - local[i]);
  return { min: Math.min(...lags), median: median(lags), p95: quantile(lags, 0.95) };
}

/** Offset to the shared clock at a local time, interpolated between measurements (drift). */
function offsetAt(history, localMs) {
  if (history.length === 0) return 0;
  if (history.length === 1 || localMs <= history[0].at_local_ms) return history[0].offset_ms;
  for (let i = 1; i < history.length; i++) {
    const a = history[i - 1];
    const b = history[i];
    if (localMs <= b.at_local_ms) {
      const f = (localMs - a.at_local_ms) / Math.max(1e-9, b.at_local_ms - a.at_local_ms);
      return a.offset_ms + f * (b.offset_ms - a.offset_ms);
    }
  }
  return history[history.length - 1].offset_ms;
}

function buildSidecar(interrupted) {
  const t = take;
  const { source, local, lag } = captureTimes(t.frames, t.mainTimeOrigin);
  const history = t.clockHistory;
  const frames = local.map((l) => Math.round((l + offsetAt(history, l)) * 1000) / 1000);
  const inTake = history.filter((h) => h.at_local_ms >= t.startedLocal - 60000);
  const last = history[history.length - 1] || { offset_ms: 0, uncertainty_ms: 0 };
  return {
    ...t.meta,
    fps: t.fps,
    width: t.width ?? t.meta.width,
    height: t.height ?? t.meta.height,
    video_first_frame_ms: frames.length ? frames[0] : null,
    frame_times_source: source,
    recorded_duration_ms: frames.length > 1 ? frames[frames.length - 1] - frames[0] : 0,
    clock_offset_ms: last.offset_ms,
    clock_uncertainty_ms: inTake.length ? Math.max(...inTake.map((h) => h.uncertainty_ms)) : last.uncertainty_ms,
    clock_history: history,
    capture_lag_ms: lag,
    encoder: { video: t.codec, audio: t.audioCodec || null, bitrate: t.bitrate },
    dropped_frames: t.dropped,
    reordered_timestamps: t.reordered,
    interrupted,
    frames,
  };
}
