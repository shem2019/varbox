<?php $user = current_user(); ?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>VAR Box</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<link rel="stylesheet" href="<?= asset('varbox.css') ?>">
<link rel="stylesheet" href="<?= asset('app.css') ?>">
<link rel="icon" href="/assets/icon.svg" type="image/svg+xml">
<script>
  try { const t = localStorage.getItem('varbox-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js" defer></script>
<?= module_import_map() ?>
<script type="module" src="<?= asset('app.js') ?>"></script>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <a class="brand" href="/app" data-link><img src="/assets/icon.svg" alt="">VAR Box</a>
    <nav class="nav" aria-label="Main">
      <a href="/app" data-link data-nav="analyses"><svg viewBox="0 0 20 20"><rect x="3" y="4" width="14" height="10" rx="2"/><path d="M7 17h6"/></svg><span>Analyses</span></a>
      <a href="/app/import" data-link data-nav="import"><svg viewBox="0 0 20 20"><path d="M10 13V3M6 7l4-4 4 4"/><path d="M3 13v3h14v-3"/></svg><span>Import</span></a>
      <a href="/app/jobs" data-link data-nav="jobs"><svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="M10 6v4l3 2"/></svg><span>Jobs</span><b class="badge" id="jobs-badge" hidden></b></a>
      <a href="/app/gpu" data-link data-nav="gpu"><svg viewBox="0 0 20 20"><rect x="4" y="4" width="12" height="12" rx="2"/><path d="M8 1v3M12 1v3M8 16v3M12 16v3M1 8h3M1 12h3M16 8h3M16 12h3"/></svg><span>GPU</span><i class="status-dot" id="gpu-dot"></i></a>
      <a href="/app/team" data-link data-nav="team"><svg viewBox="0 0 20 20"><circle cx="7.5" cy="7" r="3"/><path d="M2 17c0-3 2.5-5 5.5-5s5.5 2 5.5 5"/><circle cx="14" cy="6" r="2.4"/><path d="M14 11.5c2.4 0 4 1.6 4 4"/></svg><span>Team</span></a>
    </nav>
    <div class="sidebar-foot">
      <button class="btn ghost sm" id="theme-toggle" type="button" aria-label="Switch theme"><svg viewBox="0 0 20 20" width="16" height="16"><path d="M10 3a7 7 0 1 0 7 7 5 5 0 0 1-7-7z"/></svg><span>Theme</span></button>
      <div class="me"><span class="avatar"><?= e(strtoupper(substr((string) ($user['name'] ?? 'V'), 0, 1))) ?></span><span class="me-name"><?= e((string) ($user['name'] ?? '')) ?></span></div>
      <button class="btn ghost sm" id="sign-out" type="button">Sign out</button>
    </div>
  </aside>
  <main class="main" id="main" tabindex="-1"></main>
</div>
<div class="toast-host" id="toasts" aria-live="polite"></div>
</body>
</html>
