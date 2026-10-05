<?php
$hasUsers = (int) db()->query('SELECT COUNT(*) FROM users')->fetchColumn() > 0;
$title = 'Sign in · VAR Box';
$heading = 'Sign in';
$intro = 'Open your sessions, replays and punch reviews.';
$form = <<<HTML
<form data-endpoint="/api/login" novalidate>
  <div class="field"><label for="email">Email</label><input class="input" id="email" name="email" type="email" autocomplete="username" required autofocus></div>
  <div class="field"><label for="password">Password</label><input class="input" id="password" name="password" type="password" autocomplete="current-password" required></div>
  <div class="form-error" role="alert"></div>
  <button class="btn primary" type="submit" data-busy="Signing in…">Sign in</button>
</form>
HTML;
$footer = $hasUsers ? '' : '<p class="alt">First visit? <a href="/setup">Set up the owner account</a>.</p>';
require __DIR__ . '/auth_layout.php';
