<?php
declare(strict_types=1);

// Shared setup for every request: paths, database, session, small helpers.

const VARBOX_VERSION = '1.0.0';

define('APP_DIR', __DIR__);
define('SITE_DIR', dirname(__DIR__));
define('STORAGE_DIR', getenv('VARBOX_STORAGE') ?: SITE_DIR . '/storage');

function storage_path(string $rel = ''): string
{
    return rtrim(STORAGE_DIR, '/') . ($rel === '' ? '' : '/' . ltrim($rel, '/'));
}

function config(): array
{
    static $config = null;
    if ($config === null) {
        $file = storage_path('config.json');
        $config = is_file($file) ? (json_decode((string) file_get_contents($file), true) ?: []) : [];
    }
    return $config;
}

function db(): PDO
{
    static $pdo = null;
    if ($pdo instanceof PDO) {
        return $pdo;
    }
    if (!is_dir(STORAGE_DIR)) {
        mkdir(STORAGE_DIR, 0775, true);
    }
    $pdo = new PDO('sqlite:' . storage_path('varbox.sqlite'));
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $pdo->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);
    // Rollback journal rather than WAL: WAL leaves -wal/-shm files that break writes whenever a
    // different system user opens the database (for example during maintenance over SSH).
    $pdo->exec('PRAGMA journal_mode = DELETE');
    $pdo->exec('PRAGMA busy_timeout = 5000');
    $pdo->exec('PRAGMA foreign_keys = ON');
    migrate($pdo);
    return $pdo;
}

function migrate(PDO $pdo): void
{
    $pdo->exec(<<<SQL
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            pass_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            size INTEGER NOT NULL,
            received INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'receiving',
            created_by INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            camera_a INTEGER NOT NULL REFERENCES uploads(id),
            camera_b INTEGER REFERENCES uploads(id),
            fps INTEGER NOT NULL DEFAULT 30,
            start_s REAL NOT NULL DEFAULT 0,
            duration_s REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'queued',
            stage TEXT,
            pct REAL,
            message TEXT,
            worker TEXT,
            analysis_id INTEGER,
            error TEXT,
            previews TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            job_id INTEGER,
            status TEXT NOT NULL DEFAULT 'uploading',
            meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workers (
            name TEXT PRIMARY KEY,
            info TEXT NOT NULL DEFAULT '{}',
            last_seen TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS login_attempts (
            ip TEXT NOT NULL,
            at INTEGER NOT NULL
        );
        -- Phones sign in with their own upload-only token (stored hashed); revoking deletes the row.
        CREATE TABLE IF NOT EXISTS phones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_by INTEGER,
            created_at TEXT NOT NULL,
            last_used_at TEXT
        );
        -- One recording ties the cameras of a single take together.
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rec_uid TEXT NOT NULL UNIQUE,
            rig TEXT,
            round TEXT,
            title TEXT,
            fps INTEGER NOT NULL DEFAULT 30,
            expected_cameras INTEGER NOT NULL DEFAULT 2,
            status TEXT NOT NULL DEFAULT 'open',
            job_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recording_cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
            letter TEXT NOT NULL,
            upload_id INTEGER REFERENCES uploads(id),
            device TEXT,
            status TEXT NOT NULL DEFAULT 'receiving',
            created_at TEXT NOT NULL,
            UNIQUE(recording_id, letter)
        );
    SQL);
    // Columns added after the first release: the sidecar beside each phone video, the phone that
    // sent an upload, and the recording a job came from.
    ensure_column($pdo, 'uploads', 'sidecar', 'TEXT');
    ensure_column($pdo, 'uploads', 'phone_id', 'INTEGER');
    ensure_column($pdo, 'jobs', 'recording_id', 'INTEGER');
    // Google sign-in: the account's email and stable id. A unique index allows many NULLs (a
    // column added by ALTER cannot itself carry a UNIQUE constraint in SQLite).
    ensure_column($pdo, 'users', 'google_email', 'TEXT');
    ensure_column($pdo, 'users', 'google_sub', 'TEXT');
    $pdo->exec('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_email ON users(google_email)');
    $pdo->exec('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub)');
}

/** Add a column if an earlier schema lacked it, so existing databases upgrade in place. */
function ensure_column(PDO $pdo, string $table, string $column, string $type): void
{
    $cols = $pdo->query("PRAGMA table_info($table)")->fetchAll();
    foreach ($cols as $c) {
        if ($c['name'] === $column) {
            return;
        }
    }
    $pdo->exec("ALTER TABLE $table ADD COLUMN $column $type");
}

function now(): string
{
    return gmdate('Y-m-d\TH:i:s\Z');
}

function json_out(mixed $data, int $status = 200): never
{
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    echo json_encode($data, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    exit;
}

function fail(string $message, int $status = 400): never
{
    json_out(['error' => $message], $status);
}

function body_json(): array
{
    $raw = file_get_contents('php://input') ?: '';
    $data = json_decode($raw, true);
    return is_array($data) ? $data : [];
}

function start_session(): void
{
    if (session_status() === PHP_SESSION_ACTIVE) {
        return;
    }
    $secure = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off')
        || ($_SERVER['HTTP_X_FORWARDED_PROTO'] ?? '') === 'https'
        || ($_SERVER['HTTP_CF_VISITOR'] ?? '') !== '' && str_contains((string) $_SERVER['HTTP_CF_VISITOR'], 'https');
    // Sessions live with the rest of the app state; the server's default temp path sits
    // outside what this site may write.
    $dir = storage_path('sessions');
    if (!is_dir($dir)) {
        mkdir($dir, 0770, true);
    }
    session_save_path($dir);
    ini_set('session.gc_maxlifetime', (string) (60 * 60 * 24 * 14));
    session_name('varbox_session');
    session_set_cookie_params([
        'lifetime' => 60 * 60 * 24 * 14,
        'path' => '/',
        'secure' => $secure,
        'httponly' => true,
        'samesite' => 'Lax',
    ]);
    session_start();
}

function current_user(): ?array
{
    start_session();
    $id = $_SESSION['user_id'] ?? null;
    if (!$id) {
        return null;
    }
    $stmt = db()->prepare('SELECT id, email, name FROM users WHERE id = ?');
    $stmt->execute([$id]);
    $user = $stmt->fetch();
    return $user ?: null;
}

function require_user(bool $api = true): array
{
    $user = current_user();
    if ($user === null) {
        if ($api) {
            fail('Sign in to continue.', 401);
        }
        header('Location: /login');
        exit;
    }
    // State-changing calls must carry the app header, which other sites cannot send.
    $method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
    if ($api && $method !== 'GET' && ($_SERVER['HTTP_X_VARBOX'] ?? '') !== '1') {
        fail('Request blocked.', 403);
    }
    return $user;
}

function client_ip(): string
{
    return (string) ($_SERVER['HTTP_CF_CONNECTING_IP'] ?? $_SERVER['REMOTE_ADDR'] ?? '0.0.0.0');
}

function safe_rel_path(string $rel): string
{
    $rel = ltrim(str_replace('\\', '/', $rel), '/');
    if ($rel === '' || !preg_match('#^[A-Za-z0-9_.\-/]+$#', $rel) || str_contains($rel, '..')) {
        fail('Invalid file path.');
    }
    return $rel;
}

function mime_for(string $path): string
{
    return match (strtolower(pathinfo($path, PATHINFO_EXTENSION))) {
        'mp4' => 'video/mp4',
        'mov' => 'video/quicktime',
        'jpg', 'jpeg' => 'image/jpeg',
        'png' => 'image/png',
        'json' => 'application/json',
        'bin' => 'application/octet-stream',
        'html' => 'text/html; charset=utf-8',
        'gz', 'tgz' => 'application/gzip',
        default => 'application/octet-stream',
    };
}

/**
 * Serve a file from storage. Behind nginx the file goes out through an internal location
 * (fast, with seeking); on the PHP dev server it is streamed here with Range support.
 */
function send_storage_file(string $rel, bool $download = false): never
{
    $path = storage_path($rel);
    if (!is_file($path)) {
        fail('File unavailable.', 404);
    }
    $type = mime_for($path);
    header('Content-Type: ' . $type);
    header('Cache-Control: private, max-age=3600');
    if ($download) {
        header('Content-Disposition: attachment; filename="' . basename($path) . '"');
    }
    if (str_contains((string) ($_SERVER['SERVER_SOFTWARE'] ?? ''), 'nginx')) {
        header('X-Accel-Redirect: /storage/' . ltrim($rel, '/'));
        exit;
    }
    $size = filesize($path);
    $start = 0;
    $end = $size - 1;
    header('Accept-Ranges: bytes');
    if (preg_match('/bytes=(\d*)-(\d*)/', (string) ($_SERVER['HTTP_RANGE'] ?? ''), $m)) {
        $start = $m[1] === '' ? max(0, $size - (int) $m[2]) : (int) $m[1];
        $end = ($m[1] !== '' && $m[2] !== '') ? min((int) $m[2], $size - 1) : $end;
        http_response_code(206);
        header("Content-Range: bytes $start-$end/$size");
    }
    header('Content-Length: ' . ($end - $start + 1));
    $fh = fopen($path, 'rb');
    fseek($fh, $start);
    $left = $end - $start + 1;
    while ($left > 0 && !feof($fh)) {
        $chunk = fread($fh, (int) min(1 << 20, $left));
        echo $chunk;
        $left -= strlen($chunk);
        flush();
    }
    fclose($fh);
    exit;
}

/** Asset URL fingerprinted by content, so browsers fetch a file again the moment it changes. */
function asset(string $file): string
{
    static $cache = [];
    if (!isset($cache[$file])) {
        $path = SITE_DIR . '/assets/' . $file;
        $cache[$file] = '/assets/' . $file . '?v=' . (is_file($path) ? substr(md5_file($path), 0, 10) : VARBOX_VERSION);
    }
    return $cache[$file];
}

/** Import map pinning a page's modules (the dashboard's by default) to their fingerprinted URLs. */
function module_import_map(array $files = ['app.js', 'layers.js', 'viewer4d.js', 'meshdata.js']): string
{
    $map = [];
    foreach ($files as $file) {
        $map['/assets/' . $file] = asset($file);
    }
    return '<script type="importmap">' . json_encode(['imports' => $map], JSON_UNESCAPED_SLASHES) . '</script>';
}

function e(string $value): string
{
    return htmlspecialchars($value, ENT_QUOTES, 'UTF-8');
}

/** The site's own origin, for QR codes a phone scans. config()['site_url'] overrides it. */
function site_origin(): string
{
    $url = (string) (config()['site_url'] ?? '');
    if ($url !== '') {
        return rtrim($url, '/');
    }
    $https = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off')
        || ($_SERVER['HTTP_X_FORWARDED_PROTO'] ?? '') === 'https';
    $host = (string) ($_SERVER['HTTP_HOST'] ?? 'localhost');
    return ($https ? 'https' : 'http') . '://' . $host;
}

require_once __DIR__ . '/mail.php';
