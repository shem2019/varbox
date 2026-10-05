<?php
declare(strict_types=1);

// Front controller: pages, the dashboard API, the GPU worker API and authorised media.

require __DIR__ . '/src/bootstrap.php';
require __DIR__ . '/src/api.php';

$path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH) ?: '/';
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';

// PHP dev server: let real files through, except the private folders nginx blocks in production.
if (str_starts_with($path, '/storage/') || str_starts_with($path, '/src/')) {
    http_response_code(404);
    exit;
}
if (PHP_SAPI === 'cli-server' && $path !== '/' && is_file(__DIR__ . $path)) {
    return false;
}

if (str_starts_with($path, '/api/worker/')) {
    worker_api(substr($path, strlen('/api/worker/')), $method);
}
if (str_starts_with($path, '/api/')) {
    user_api(substr($path, strlen('/api/')), $method);
}
if (preg_match('#^/media/(\d+)/(.+)$#', $path, $m)) {
    require_user(false);
    send_storage_file('analyses/' . (int) $m[1] . '/' . safe_rel_path($m[2]), isset($_GET['download']));
}
if (preg_match('#^/preview/job/(\d+)/([A-Za-z0-9_]+)\.jpg$#', $path, $m)) {
    require_user(false);
    send_storage_file('previews/job_' . (int) $m[1] . '/' . $m[2] . '.jpg');
}

$view = match (true) {
    $path === '/' => 'landing',
    $path === '/login' => 'login',
    $path === '/setup' => 'setup',
    $path === '/app' || str_starts_with($path, '/app/') => 'app',
    default => '404',
};

if ($view === 'app') {
    require_user(false);
}
if ($view === 'login' && current_user() !== null) {
    header('Location: /app');
    exit;
}
if ($view === '404') {
    http_response_code(404);
}
header('Content-Type: text/html; charset=utf-8');
header('Cache-Control: no-cache');
header('X-Frame-Options: SAMEORIGIN');
header('Referrer-Policy: same-origin');
require __DIR__ . '/src/views/' . $view . '.php';
