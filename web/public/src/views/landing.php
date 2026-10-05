<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>VAR Box · Boxing review in 3D</title>
<meta name="description" content="VAR Box turns two phone cameras into a 3D replay of both boxers, with every punch timed, sided and classified for the judges' review.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<link rel="stylesheet" href="/assets/varbox.css?v=<?= VARBOX_VERSION ?>">
<link rel="icon" href="/assets/icon.svg" type="image/svg+xml">
<style>
  .wrap { max-width: 1160px; margin: 0 auto; padding-inline: 20px; }
  .nav { display: flex; align-items: center; justify-content: space-between; padding-block: 18px; }
  .brand { display: inline-flex; align-items: center; gap: 10px; font: 800 22px/1 var(--display); letter-spacing: .06em; text-transform: uppercase; text-decoration: none; }
  .brand img { width: 28px; height: 28px; }
  .hero { position: relative; overflow: hidden; background: var(--stage); color: #eef0f5; border-radius: 18px; margin-top: 4px; }
  .hero-grid { display: grid; grid-template-columns: 1.05fr 1fr; align-items: stretch; min-height: 520px; }
  .hero-copy { padding: 56px 48px; display: grid; align-content: center; gap: 22px; position: relative; z-index: 1; }
  .hero h1 { font-size: clamp(44px, 6.4vw, 84px); font-weight: 800; }
  .hero h1 .r { color: #f0566a; } .hero h1 .b { color: #6c97f5; }
  .hero p { font-size: 18px; color: #b9bfcc; max-width: 34em; margin: 0; }
  .hero .eyebrow { color: #8d95a7; }
  .hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
  .hero .btn.primary { background: #eef0f5; color: #12151c; border-color: #eef0f5; }
  .hero .btn.ghost { color: #eef0f5; border-color: #3a4050; }
  .hero-visual { position: relative; min-height: 320px; background: radial-gradient(120% 90% at 60% 55%, #1b2130 0%, #0c0e12 70%); }
  .hero-visual img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center 40%; }
  .hero-visual .tag { position: absolute; left: 20px; bottom: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
  .hero-visual .tag span { background: rgba(12, 14, 18, .72); border: 1px solid #2a3040; color: #d7dbe4; font: 500 12px var(--mono); padding: 6px 9px; border-radius: 6px; backdrop-filter: blur(6px); }
  section { padding-block: 72px; }
  .section-head { display: grid; gap: 10px; margin-bottom: 32px; max-width: 44em; }
  .section-head h2 { font-size: clamp(32px, 4vw, 48px); }
  .section-head p { color: var(--ink-2); font-size: 17px; margin: 0; }
  .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; counter-reset: step; }
  .step { padding: 24px; display: grid; gap: 10px; align-content: start; }
  .step .n { font: 800 44px/1 var(--display); color: var(--ink-3); }
  .step h3 { font-size: 26px; }
  .step p { margin: 0; color: var(--ink-2); }
  .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .feature { padding: 22px; display: grid; gap: 8px; align-content: start; }
  .feature h3 { font-size: 22px; }
  .feature p { margin: 0; color: var(--ink-2); font-size: 14.5px; }
  .feature .icon { width: 36px; height: 36px; border-radius: 9px; display: grid; place-items: center; background: var(--surface-2); }
  .proof { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .stat { padding: 22px; display: grid; gap: 6px; }
  .stat .big { font: 800 48px/1 var(--display); }
  .stat p { margin: 0; color: var(--ink-2); font-size: 14px; }
  .pledge { display: grid; grid-template-columns: auto 1fr auto; gap: 20px; align-items: center; padding: 28px; }
  .pledge h3 { font-size: 28px; }
  .pledge p { margin: 4px 0 0; color: var(--ink-2); }
  footer { padding-block: 32px 48px; color: var(--ink-3); font-size: 13px; display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  @media (max-width: 900px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-copy { padding: 36px 24px 8px; }
    .hero-visual { min-height: 300px; }
    .steps, .features, .proof { grid-template-columns: 1fr; }
    .pledge { grid-template-columns: 1fr; }
    section { padding-block: 52px; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <nav class="nav">
      <a class="brand" href="/"><img src="/assets/icon.svg" alt="">VAR Box</a>
      <a class="btn" href="/login">Sign in</a>
    </nav>

    <header class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow">Boxing review in 3D</div>
          <h1>Every exchange,<br>seen from <span class="r">every</span> <span class="b">side</span>.</h1>
          <p>Judges and coaches need a clear view of each punch, and fast exchanges blur on video while camera angles hide the contact. VAR Box gives them that view by turning two phone cameras into a 3D replay of both boxers, with every punch timed, sided and classified for review.</p>
          <div class="hero-actions">
            <a class="btn primary" href="/login">Sign in</a>
            <a class="btn ghost" href="#how">See how it works</a>
          </div>
        </div>
        <div class="hero-visual">
          <img src="/assets/hero.jpg" alt="Two boxers rebuilt as 3D bodies in the middle of an exchange" loading="eager">
          <div class="tag"><span>red · right hand</span><span>landed · head</span><span>21.2 s into the clip</span></div>
        </div>
      </div>
    </header>

    <section id="how">
      <div class="section-head">
        <div class="eyebrow">How it works</div>
        <h2>Two phones in, a 3D bout out</h2>
        <p>A session needs two phones on stands and one crack of two planks to line the recordings up. The rest runs on a rented GPU and arrives in the dashboard.</p>
      </div>
      <div class="steps">
        <article class="card step">
          <div class="n">01</div>
          <h3>Record</h3>
          <p>Two phones at adjacent sides of the ring, one boxer in red and one in blue. A single plank crack at the start of each round syncs both cameras to the frame.</p>
        </article>
        <article class="card step">
          <div class="n">02</div>
          <h3>Reconstruct</h3>
          <p>Each boxer is tracked through every frame and rebuilt as a full 3D body, anchored to the ring floor and merged across both cameras.</p>
        </article>
        <article class="card step">
          <div class="n">03</div>
          <h3>Review</h3>
          <p>Every punch arrives timed, with the hand that threw it and its outcome: landed to the head or body, blocked, or missed. Each call links straight to its replay.</p>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div class="eyebrow">For the reviewer</div>
        <h2>One bout, many ways to look</h2>
      </div>
      <div class="features">
        <article class="card feature">
          <div class="icon"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="14" height="10" rx="2"/><path d="M6 17h8"/></svg></div>
          <h3>Layers</h3>
          <p>The real footage with a switch for each boxer: true pixels, a solid silhouette, the 3D body, or hidden from the scene.</p>
        </article>
        <article class="card feature">
          <div class="icon"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 2l7 4v8l-7 4-7-4V6z"/><path d="M10 10l7-4M10 10v8M10 10L3 6"/></svg></div>
          <h3>4D replay</h3>
          <p>Orbit any moment of the bout, slow it to a tenth of real speed, and isolate one boxer to study their form alone.</p>
        </article>
        <article class="card feature">
          <div class="icon"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 10h16"/><path d="M5 6v8M10 4v12M15 7v6"/></svg></div>
          <h3>Punch timeline</h3>
          <p>Landed, blocked and missed punches marked along the round, filterable by boxer and outcome, one click from the replay.</p>
        </article>
        <article class="card feature">
          <div class="icon"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="2" y="4" width="7" height="12" rx="1.5"/><rect x="11" y="4" width="7" height="12" rx="1.5"/></svg></div>
          <h3>Before and after</h3>
          <p>Geometry alone beside geometry plus the trained strike model, so every improvement is visible on the same footage.</p>
        </article>
        <article class="card feature">
          <div class="icon"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 3v10M6 9l4 4 4-4"/><path d="M3 16h14"/></svg></div>
          <h3>Exports</h3>
          <p>Overlay videos, isolated renders, per-frame data and the full event list, ready to download for the coaching file.</p>
        </article>
        <article class="card feature">
          <div class="icon"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="10" cy="10" r="7"/><path d="M10 6v4l3 2"/></svg></div>
          <h3>Live progress</h3>
          <p>Each stage of the analysis reports in as the GPU works: syncing, tracking, rebuilding, scoring and packaging.</p>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div class="eyebrow">First results</div>
        <h2>Measured on Olympic bout footage</h2>
        <p>Early tests ran on a public dataset of Olympic bouts with every punch labelled by hand.</p>
      </div>
      <div class="proof">
        <div class="card stat"><div class="big num">12 / 13</div><p>Labelled punches found in the first test clip, each one within a fraction of a second.</p></div>
        <div class="card stat"><div class="big num">12 / 12</div><p>Found punches credited to the correct hand.</p></div>
        <div class="card stat"><div class="big num">100%</div><p>Frames with both boxers tracked across a 30 second exchange.</p></div>
      </div>
    </section>

    <section>
      <div class="card pledge">
        <img src="/assets/icon.svg" alt="" width="44" height="44">
        <div>
          <h3>Built to support the officials</h3>
          <p>VAR Box is decision support for licensed judges and referees. Every call stays with them, now with a clearer view of each exchange.</p>
        </div>
        <a class="btn primary" href="/login">Sign in</a>
      </div>
    </section>

    <footer>
      <span>© <?= date('Y') ?> VAR Box</span>
      <span>Decision support for boxing officials</span>
    </footer>
  </div>
</body>
</html>
