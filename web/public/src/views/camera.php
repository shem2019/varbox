<?php
$cameraModules = module_import_map(['camera/camera.js', 'camera/clock.js', 'camera/upload.js']);
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0c0e12">
<title>Camera · VAR Box</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<link rel="stylesheet" href="<?= asset('varbox.css') ?>">
<link rel="icon" href="/assets/icon.svg" type="image/svg+xml">
<style>
  body { background: var(--stage); color: #eceef3; min-height: 100dvh; margin: 0; }
  .cam { max-width: 640px; margin: 0 auto; padding: 12px 16px 40px; display: grid; gap: 12px; }
  .top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .brand { display: inline-flex; align-items: center; gap: 8px; font: 800 20px/1 var(--display); letter-spacing: .06em; text-transform: uppercase; color: inherit; text-decoration: none; }
  .brand img { width: 24px; height: 24px; }
  #who { font: 500 14px/1.2 var(--mono); color: #b2b8c6; }
  .stage { position: relative; background: #000; border-radius: var(--radius); overflow: hidden; aspect-ratio: 16 / 9; }
  #preview { width: 100%; height: 100%; object-fit: contain; display: block; }
  #timer { position: absolute; top: 10px; left: 10px; font: 600 15px/1 var(--mono); background: rgba(0,0,0,.6); padding: 6px 8px; border-radius: 6px; display: none; }
  body.recording #timer { display: block; }
  body.recording .stage { outline: 3px solid #f0566a; }
  .sync { font: 600 18px/1.3 var(--body); padding: 10px 12px; border-radius: var(--radius); background: #1f232c; }
  .sync.good { background: #123524; color: #3fca84; }
  .sync.bad { background: #3a2a0c; color: #f0b43a; }
  .row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .row label { display: grid; gap: 4px; font-size: 12px; color: #b2b8c6; }
  .row .take { grid-column: span 2; }
  .row select, .row input { width: 100%; min-width: 0; background: #1f232c; color: #eceef3; border: 1px solid #2a2f3a; border-radius: 8px; padding: 10px 8px; font: 500 15px var(--body); }
  #record { width: 100%; font: 800 26px/1 var(--display); letter-spacing: .08em; text-transform: uppercase; padding: 20px; border-radius: 14px; border: 0; background: #f0566a; color: #fff; }
  #record:disabled { opacity: .5; }
  body.recording #record { background: #eceef3; color: #12151c; }
  .meta { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; font-size: 13px; color: #b2b8c6; }
  .note { padding: 10px 12px; border-radius: var(--radius); background: #1f232c; font-size: 14px; }
  .note.warn { background: #3a2a0c; color: #f0b43a; }
  h2 { font: 700 16px/1 var(--display); letter-spacing: .06em; text-transform: uppercase; margin: 8px 0 0; color: #b2b8c6; }
  ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; font-size: 13px; }
  #uploads li { display: flex; justify-content: space-between; gap: 8px; background: #171a21; padding: 8px 10px; border-radius: 8px; }
  #uploads .mono { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--mono); }
  #caps li.ok { color: #3fca84; } #caps li.no { color: #848b9b; }
  details { font-size: 13px; color: #b2b8c6; }
  @media (max-width: 420px) { .row { grid-template-columns: repeat(2, 1fr); } }
</style>
<?= $cameraModules ?>
</head>
<body data-worker="<?= e(asset('camera/recorder-worker.js')) ?>">
  <main class="cam">
    <div class="top">
      <a class="brand" href="/"><img src="/assets/icon.svg" alt="">VAR Box Camera</a>
      <span id="who">…</span>
    </div>
    <div id="note" class="note" hidden></div>
    <div id="sync" class="sync">Measuring the clock…</div>
    <div class="stage">
      <video id="preview" playsinline muted autoplay></video>
      <span id="timer">00:00</span>
    </div>
    <div class="meta"><span id="actual"></span><span id="storage"></span></div>
    <div class="row">
      <label>Camera
        <select id="letter"><option>A</option><option>B</option><option>C</option><option>D</option><option>E</option><option>F</option></select>
      </label>
      <label class="take">Take (same on every camera)
        <input id="take" autocapitalize="none" spellcheck="false" maxlength="60">
      </label>
      <label>Quality
        <select id="res"><option value="1080">1080p</option><option value="720">720p</option></select>
      </label>
      <label>Frame rate
        <select id="fps"><option value="30">30 fps</option><option value="60">60 fps</option></select>
      </label>
    </div>
    <button id="record" type="button">Record</button>
    <div class="meta"><span id="stats"></span></div>
    <h2>Takes on this phone</h2>
    <ul id="uploads"></ul>
    <details>
      <summary>What this browser supports</summary>
      <p id="route"></p>
      <ul id="caps"></ul>
    </details>
  </main>
  <script type="module" src="<?= e(asset('camera/camera.js')) ?>"></script>
</body>
</html>
