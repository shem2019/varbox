<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Privacy · VAR Box</title>
<meta name="description" content="What VAR Box stores and how it is used.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<link rel="stylesheet" href="<?= asset('varbox.css') ?>">
<link rel="icon" href="/assets/icon.svg" type="image/svg+xml">
<style>
  .wrap { max-width: 720px; margin: 0 auto; padding: 24px 20px 64px; }
  .nav { display: flex; align-items: center; justify-content: space-between; padding-block: 18px; }
  .brand { display: inline-flex; align-items: center; gap: 10px; font: 800 22px/1 var(--display); letter-spacing: .06em; text-transform: uppercase; text-decoration: none; }
  .brand img { width: 28px; height: 28px; }
  h1 { font-size: clamp(32px, 5vw, 44px); margin: 24px 0 8px; }
  h2 { font-size: 22px; margin: 28px 0 8px; }
  p, li { color: var(--ink-2); line-height: 1.6; }
  ul { padding-left: 20px; display: grid; gap: 6px; }
  .updated { color: var(--ink-3); font-size: 14px; }
  footer { margin-top: 48px; color: var(--ink-3); font-size: 13px; }
</style>
</head>
<body>
  <div class="wrap">
    <nav class="nav">
      <a class="brand" href="/"><img src="/assets/icon.svg" alt="">VAR Box</a>
      <a class="btn" href="/login">Sign in</a>
    </nav>

    <h1>Privacy</h1>
    <p class="updated">VAR Box is a private tool for reviewing boxing footage. Access is by invitation only.</p>

    <h2>What the site stores</h2>
    <ul>
      <li>The name and Google email of each person the owner adds to the team, and the Google account ID returned when they sign in.</li>
      <li>The videos uploaded for analysis and the analyses produced from them: replays, 3D bodies and punch data.</li>
      <li>A short, rotating record of sign-in attempts per network address, to slow down guessing.</li>
    </ul>

    <h2>How it is used</h2>
    <p>The stored data is used only to sign team members in and to run and show the boxing analysis. Nothing is shared with, sold to, or disclosed to anyone else.</p>

    <h2>Contact</h2>
    <p>Questions, or to have an account and its data removed, reach <a href="mailto:shemvilejado@gmail.com">shemvilejado@gmail.com</a>.</p>

    <footer>© <?= date('Y') ?> VAR Box · <a href="/">Home</a> · <a href="/login">Sign in</a></footer>
  </div>
</body>
</html>
