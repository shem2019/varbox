<?php
/** @var string $title @var string $heading @var string $intro @var string $form @var string $footer */
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title><?= e($title) ?></title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<link rel="stylesheet" href="/assets/varbox.css?v=<?= VARBOX_VERSION ?>">
<link rel="icon" href="/assets/icon.svg" type="image/svg+xml">
<style>
  body { min-height: 100vh; display: grid; grid-template-columns: 1fr 1fr; }
  .side { background: var(--stage); color: #eef0f5; position: relative; overflow: hidden; display: grid; align-content: end; padding: 40px; }
  .side img.bg { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: .85; }
  .side .quote { position: relative; display: grid; gap: 10px; max-width: 30em; }
  .side h2 { font-size: 40px; }
  .side p { margin: 0; color: #b9bfcc; }
  .main { display: grid; place-items: center; padding: 40px 20px; }
  .panel { width: 100%; max-width: 400px; display: grid; gap: 22px; }
  .brand { display: inline-flex; align-items: center; gap: 10px; font: 800 22px/1 var(--display); letter-spacing: .06em; text-transform: uppercase; text-decoration: none; }
  .brand img { width: 28px; height: 28px; }
  .panel h1 { font-size: 40px; }
  .panel p.intro { margin: 0; color: var(--ink-2); }
  form { display: grid; gap: 16px; }
  .alt { font-size: 14px; color: var(--ink-3); }
  @media (max-width: 860px) { body { grid-template-columns: 1fr; } .side { display: none; } }
</style>
</head>
<body>
  <aside class="side">
    <img class="bg" src="/assets/hero.jpg" alt="">
    <div class="quote">
      <h2>Every exchange, seen from every side</h2>
      <p>Two phone cameras, one plank crack, and each punch rebuilt in 3D for the judges' review.</p>
    </div>
  </aside>
  <main class="main">
    <div class="panel">
      <a class="brand" href="/"><img src="/assets/icon.svg" alt="">VAR Box</a>
      <div style="display:grid;gap:8px">
        <h1><?= e($heading) ?></h1>
        <p class="intro"><?= e($intro) ?></p>
      </div>
      <?= $form ?>
      <?= $footer ?>
    </div>
  </main>
  <script>
    const form = document.querySelector('form[data-endpoint]');
    form?.addEventListener('submit', async (event) => {
      event.preventDefault();
      const button = form.querySelector('button[type=submit]');
      const error = form.querySelector('.form-error');
      error.textContent = '';
      button.disabled = true;
      const label = button.textContent;
      button.textContent = button.dataset.busy || 'Working…';
      try {
        const res = await fetch(form.dataset.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-VarBox': '1' },
          body: JSON.stringify(Object.fromEntries(new FormData(form))),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) { location.href = '/app'; return; }
        error.textContent = data.error || 'Something went wrong. Try again in a moment.';
      } catch (e) {
        error.textContent = 'The connection dropped. Check the network and try again.';
      }
      button.disabled = false;
      button.textContent = label;
    });
  </script>
</body>
</html>
