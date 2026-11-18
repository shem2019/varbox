<?php
$hasUsers = (int) db()->query('SELECT COUNT(*) FROM users')->fetchColumn() > 0;
$clientId = e(google_client_id());
$title = 'Sign in · VAR Box';
$heading = 'Sign in';
$intro = 'Open your sessions, replays and punch reviews.';
$setup = $hasUsers ? '' : '<p class="alt">First visit? <a href="/setup">Set up the owner account</a>.</p>';
$form = <<<HTML
<div class="gsi" style="display:grid;gap:10px;justify-items:start">
  <script src="https://accounts.google.com/gsi/client" async></script>
  <div id="g_id_onload"
       data-client_id="{$clientId}"
       data-callback="onVarboxGoogle"
       data-ux_mode="popup"
       data-auto_prompt="false"></div>
  <div class="g_id_signin" data-type="standard" data-theme="outline" data-size="large"
       data-text="signin_with" data-shape="rectangular" data-logo_alignment="left" data-width="320"></div>
  <div class="form-error" role="alert" id="google-error"></div>
</div>
<p class="alt"><a href="#" id="show-password">Sign in with a password</a></p>
<form data-endpoint="/api/login" novalidate hidden id="password-form">
  <div class="field"><label for="email">Email or username</label><input class="input" id="email" name="email" type="text" autocomplete="username" autocapitalize="none" spellcheck="false"></div>
  <div class="field"><label for="password">Password</label><input class="input" id="password" name="password" type="password" autocomplete="current-password"></div>
  <div class="form-error" role="alert"></div>
  <button class="btn primary" type="submit" data-busy="Signing in…">Sign in</button>
</form>
<script>
  window.onVarboxGoogle = async (response) => {
    const error = document.getElementById('google-error');
    error.textContent = '';
    try {
      const res = await fetch('/api/login/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-VarBox': '1' },
        body: JSON.stringify({ credential: response.credential }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) { location.href = '/app'; return; }
      error.textContent = data.error || 'Google sign-in failed. Try again.';
    } catch (e) {
      error.textContent = 'The connection dropped. Check the network and try again.';
    }
  };
  document.getElementById('show-password')?.addEventListener('click', (event) => {
    event.preventDefault();
    document.getElementById('password-form').hidden = false;
    event.target.closest('p').hidden = true;
    document.getElementById('email').focus();
  });
</script>
HTML;
$footer = $setup . '<p class="alt"><a href="/privacy">Privacy</a></p>';
require __DIR__ . '/auth_layout.php';
