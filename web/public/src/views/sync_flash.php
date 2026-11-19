<?php
$flashModules = module_import_map(['camera/sync-flash.js', 'camera/clock.js']);
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sync flash · VAR Box</title>
<link rel="icon" href="/assets/icon.svg" type="image/svg+xml">
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden; }
  #flash { display: block; width: 100vw; height: 100vh; }
  #label { position: fixed; left: 0; right: 0; bottom: 8px; text-align: center; font: 500 14px/1.3 system-ui, sans-serif; color: #4b5263; }
</style>
<?= $flashModules ?>
</head>
<body>
  <canvas id="flash"></canvas>
  <div id="label">Measuring the clock…</div>
  <script type="module" src="<?= e(asset('camera/sync-flash.js')) ?>"></script>
</body>
</html>
