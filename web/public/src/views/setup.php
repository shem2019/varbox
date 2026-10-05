<?php
if ((int) db()->query('SELECT COUNT(*) FROM users')->fetchColumn() > 0) {
    header('Location: /login');
    exit;
}
$title = 'Set up · VAR Box';
$heading = 'Owner account';
$intro = 'Create the account that runs this VAR Box. The setup code comes from the server and works once.';
$form = <<<HTML
<form data-endpoint="/api/setup" novalidate>
  <div class="field"><label for="code">Setup code</label><input class="input mono" id="code" name="code" autocomplete="one-time-code" required autofocus></div>
  <div class="field"><label for="name">Your name</label><input class="input" id="name" name="name" autocomplete="name" required></div>
  <div class="field"><label for="email">Email</label><input class="input" id="email" name="email" type="email" autocomplete="username" required></div>
  <div class="field"><label for="password">Password (10 characters or more)</label><input class="input" id="password" name="password" type="password" autocomplete="new-password" minlength="10" required></div>
  <div class="form-error" role="alert"></div>
  <button class="btn primary" type="submit" data-busy="Creating…">Create account</button>
</form>
HTML;
$footer = '<p class="alt">Already set up? <a href="/login">Sign in</a>.</p>';
require __DIR__ . '/auth_layout.php';
