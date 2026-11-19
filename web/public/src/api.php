<?php
declare(strict_types=1);

// JSON API for the dashboard (session login) and for GPU workers (bearer token).

const CHUNK_LIMIT = 16 * 1024 * 1024;
const WORKER_ONLINE_SECONDS = 60;

function owner_id(): int
{
    return (int) db()->query('SELECT MIN(id) FROM users')->fetchColumn();
}

function is_owner(array $user): bool
{
    return (int) $user['id'] === owner_id();
}

function analysis_row(array $row): array
{
    $meta = json_decode($row['meta'] ?? '{}', true) ?: [];
    return [
        'id' => (int) $row['id'],
        'title' => $row['title'],
        'status' => $row['status'],
        'job_id' => $row['job_id'] !== null ? (int) $row['job_id'] : null,
        'created_at' => $row['created_at'],
        'meta' => $meta,
        'poster' => '/media/' . (int) $row['id'] . '/poster.jpg',
    ];
}

function job_row(array $row): array
{
    $out = $row;
    foreach (['id', 'camera_a', 'camera_b', 'fps', 'analysis_id'] as $k) {
        $out[$k] = $row[$k] !== null ? (int) $row[$k] : null;
    }
    $out['pct'] = $row['pct'] !== null ? (float) $row['pct'] : null;
    $out['previews'] = array_map(
        fn (string $name) => '/preview/job/' . (int) $row['id'] . '/' . $name . '.jpg',
        json_decode($row['previews'] ?: '[]', true) ?: []
    );
    return $out;
}

function upload_row(array $row): array
{
    return [
        'id' => (int) $row['id'],
        'name' => $row['name'],
        'size' => (int) $row['size'],
        'received' => (int) $row['received'],
        'status' => $row['status'],
        'created_at' => $row['created_at'],
    ];
}

function find(string $table, int $id): array
{
    $stmt = db()->prepare("SELECT * FROM $table WHERE id = ?");
    $stmt->execute([$id]);
    $row = $stmt->fetch();
    if (!$row) {
        fail('Item unavailable.', 404);
    }
    return $row;
}

function delete_analysis(int $id): void
{
    $dir = storage_path("analyses/$id");
    if (is_dir($dir)) {
        $it = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($dir, FilesystemIterator::SKIP_DOTS),
            RecursiveIteratorIterator::CHILD_FIRST
        );
        foreach ($it as $f) {
            $f->isDir() ? rmdir($f->getPathname()) : unlink($f->getPathname());
        }
        rmdir($dir);
    }
    db()->prepare('DELETE FROM analyses WHERE id = ?')->execute([$id]);
}

function append_chunk(string $file, int $offset, int $total): int
{
    if ($offset < 0 || $total <= 0) {
        fail('Invalid offset.');
    }
    $data = file_get_contents('php://input') ?: '';
    if (strlen($data) > CHUNK_LIMIT) {
        fail('Chunk too large.', 413);
    }
    $dir = dirname($file);
    if (!is_dir($dir)) {
        mkdir($dir, 0775, true);
    }
    $current = is_file($file) ? filesize($file) : 0;
    if ($offset === 0 && $current > 0) {
        unlink($file);
        $current = 0;
    }
    if ($offset !== $current) {
        // Resume support: tell the sender where to continue.
        json_out(['received' => $current, 'resume' => true], 409);
    }
    $fh = fopen($file, 'ab');
    fwrite($fh, $data);
    fclose($fh);
    clearstatcache(true, $file);
    return (int) filesize($file);
}

function user_api(string $route, string $method): never
{
    if ($route === 'login' && $method === 'POST') {
        api_login();
    }
    if ($route === 'login/google' && $method === 'POST') {
        api_login_google();
    }
    if ($route === 'setup' && $method === 'POST') {
        api_setup();
    }
    $user = require_user();
    $db = db();

    if ($route === 'logout' && $method === 'POST') {
        $_SESSION = [];
        session_destroy();
        json_out(['ok' => true]);
    }
    if ($route === 'me') {
        json_out(['user' => $user + ['owner' => is_owner($user)], 'version' => VARBOX_VERSION]);
    }

    // Team: everyone sees it; the owner (first account) adds and removes people.
    if ($route === 'users' && $method === 'GET') {
        $rows = $db->query('SELECT id, email, name, google_email, google_sub, created_at FROM users ORDER BY id')->fetchAll();
        $owner = owner_id();
        json_out([
            'users' => array_map(fn ($r) => [
                'id' => (int) $r['id'],
                'login' => $r['email'],
                'name' => $r['name'],
                'google_email' => $r['google_email'],
                'google_linked' => $r['google_sub'] !== null && $r['google_sub'] !== '',
                'created_at' => $r['created_at'],
                'owner' => (int) $r['id'] === $owner,
            ], $rows),
            'can_manage' => is_owner($user),
        ]);
    }
    if ($route === 'users' && $method === 'POST') {
        if (!is_owner($user)) {
            fail('Only the owner adds people.', 403);
        }
        $in = body_json();
        $googleEmail = strtolower(trim((string) ($in['google_email'] ?? '')));
        $name = trim((string) ($in['name'] ?? '')) ?: $googleEmail;
        if (!filter_var($googleEmail, FILTER_VALIDATE_EMAIL)) {
            fail('Enter the Google email for this person.');
        }
        $exists = $db->prepare('SELECT COUNT(*) FROM users WHERE email = ? OR google_email = ?');
        $exists->execute([$googleEmail, $googleEmail]);
        if ((int) $exists->fetchColumn() > 0) {
            fail('That Google account is already on the team.', 409);
        }
        // The team member signs in with Google; a random hash fills the unused password column.
        $unused = password_hash(bin2hex(random_bytes(16)), PASSWORD_DEFAULT);
        $db->prepare('INSERT INTO users (email, name, google_email, pass_hash, created_at) VALUES (?, ?, ?, ?, ?)')
            ->execute([$googleEmail, $name, $googleEmail, $unused, now()]);
        json_out(['ok' => true]);
    }
    if (preg_match('#^users/(\d+)$#', $route, $m) && $method === 'PATCH') {
        if (!is_owner($user)) {
            fail('Only the owner sets Google emails.', 403);
        }
        $id = (int) $m[1];
        find('users', $id);
        $googleEmail = strtolower(trim((string) (body_json()['google_email'] ?? '')));
        if (!filter_var($googleEmail, FILTER_VALIDATE_EMAIL)) {
            fail('Enter a valid Google email.');
        }
        $exists = $db->prepare('SELECT COUNT(*) FROM users WHERE google_email = ? AND id != ?');
        $exists->execute([$googleEmail, $id]);
        if ((int) $exists->fetchColumn() > 0) {
            fail('Another member already uses that Google account.', 409);
        }
        // Changing the email clears the stored account id, so the next Google sign-in links afresh.
        $db->prepare('UPDATE users SET google_email = ?, google_sub = NULL WHERE id = ?')
            ->execute([$googleEmail, $id]);
        json_out(['ok' => true]);
    }
    if (preg_match('#^users/(\d+)$#', $route, $m) && $method === 'DELETE') {
        if (!is_owner($user)) {
            fail('Only the owner removes people.', 403);
        }
        $id = (int) $m[1];
        if ($id === (int) $user['id']) {
            fail('The owner account stays on the team.', 409);
        }
        $db->prepare('DELETE FROM users WHERE id = ?')->execute([$id]);
        json_out(['ok' => true]);
    }
    if ($route === 'password' && $method === 'POST') {
        $in = body_json();
        $row = find('users', (int) $user['id']);
        if (!password_verify((string) ($in['current'] ?? ''), $row['pass_hash'])) {
            fail('Check the current password, then try again.', 403);
        }
        $next = (string) ($in['next'] ?? '');
        if (strlen($next) < 8) {
            fail('Use a new password of at least 8 characters.');
        }
        $db->prepare('UPDATE users SET pass_hash = ? WHERE id = ?')
            ->execute([password_hash($next, PASSWORD_DEFAULT), $user['id']]);
        json_out(['ok' => true]);
    }

    // Analyses
    if ($route === 'analyses' && $method === 'GET') {
        $rows = $db->query("SELECT * FROM analyses WHERE status = 'ready' ORDER BY id DESC")->fetchAll();
        json_out(['analyses' => array_map('analysis_row', $rows)]);
    }
    if (preg_match('#^analyses/(\d+)$#', $route, $m)) {
        $id = (int) $m[1];
        $row = find('analyses', $id);
        if ($method === 'PATCH') {
            $title = trim((string) (body_json()['title'] ?? ''));
            if ($title === '') {
                fail('Give the analysis a title.');
            }
            $db->prepare('UPDATE analyses SET title = ? WHERE id = ?')->execute([$title, $id]);
            $row['title'] = $title;
        }
        if ($method === 'DELETE') {
            delete_analysis($id);
            json_out(['ok' => true]);
        }
        $files = [];
        $dir = storage_path("analyses/$id");
        if (is_dir($dir)) {
            $it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($dir, FilesystemIterator::SKIP_DOTS));
            foreach ($it as $f) {
                $rel = substr($f->getPathname(), strlen($dir) + 1);
                $files[] = ['path' => $rel, 'size' => $f->getSize()];
            }
            usort($files, fn ($a, $b) => strcmp($a['path'], $b['path']));
        }
        json_out(['analysis' => analysis_row($row), 'files' => $files, 'base' => "/media/$id/"]);
    }

    // Uploads (chunked, resumable)
    if ($route === 'uploads' && $method === 'GET') {
        $rows = $db->query('SELECT * FROM uploads ORDER BY id DESC LIMIT 100')->fetchAll();
        json_out(['uploads' => array_map('upload_row', $rows)]);
    }
    if ($route === 'uploads' && $method === 'POST') {
        $in = body_json();
        $name = preg_replace('/[^A-Za-z0-9_.\-]+/', '_', (string) ($in['name'] ?? 'video.mp4'));
        $size = (int) ($in['size'] ?? 0);
        if ($size <= 0) {
            fail('Choose a video file.');
        }
        $db->prepare('INSERT INTO uploads (name, size, created_by, created_at) VALUES (?, ?, ?, ?)')
            ->execute([$name, $size, $user['id'], now()]);
        json_out(['upload' => upload_row(find('uploads', (int) $db->lastInsertId()))]);
    }
    if (preg_match('#^uploads/(\d+)$#', $route, $m) && $method === 'PUT') {
        $row = find('uploads', (int) $m[1]);
        $received = append_chunk(storage_path('uploads/' . $row['id'] . '.bin'), (int) ($_GET['offset'] ?? -1), (int) $row['size']);
        $status = $received >= (int) $row['size'] ? 'ready' : 'receiving';
        $db->prepare('UPDATE uploads SET received = ?, status = ? WHERE id = ?')->execute([$received, $status, $row['id']]);
        json_out(['received' => $received, 'status' => $status]);
    }

    // Jobs
    if ($route === 'jobs' && $method === 'GET') {
        $rows = $db->query('SELECT * FROM jobs ORDER BY id DESC LIMIT 50')->fetchAll();
        json_out(['jobs' => array_map('job_row', $rows)]);
    }
    if ($route === 'jobs' && $method === 'POST') {
        $in = body_json();
        $a = find('uploads', (int) ($in['camera_a'] ?? 0));
        if ($a['status'] !== 'ready') {
            fail('Camera A is still uploading.');
        }
        $b = null;
        if (!empty($in['camera_b'])) {
            $b = find('uploads', (int) $in['camera_b']);
            if ($b['status'] !== 'ready') {
                fail('Camera B is still uploading.');
            }
        }
        $title = trim((string) ($in['title'] ?? '')) ?: 'Session ' . date('j M Y, H:i');
        $fps = (int) ($in['fps'] ?? 30);
        $fps = in_array($fps, [30, 60], true) ? $fps : 30;
        $db->prepare(
            'INSERT INTO jobs (title, camera_a, camera_b, fps, start_s, duration_s, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        )->execute([
            $title, $a['id'], $b['id'] ?? null, $fps,
            max(0.0, (float) ($in['start_s'] ?? 0)), max(0.0, (float) ($in['duration_s'] ?? 0)), now(), now(),
        ]);
        json_out(['job' => job_row(find('jobs', (int) $db->lastInsertId()))]);
    }
    if (preg_match('#^jobs/(\d+)/process$#', $route, $m) && $method === 'POST') {
        $row = find('jobs', (int) $m[1]);
        if ($row['status'] !== 'pending') {
            fail('This task is not waiting to be processed.', 409);
        }
        $db->prepare("UPDATE jobs SET status = 'queued', updated_at = ? WHERE id = ?")
            ->execute([now(), $row['id']]);
        json_out(['job' => job_row(find('jobs', (int) $row['id']))]);
    }
    if (preg_match('#^jobs/(\d+)$#', $route, $m)) {
        $row = find('jobs', (int) $m[1]);
        if ($method === 'DELETE' && in_array($row['status'], ['queued', 'failed', 'done', 'pending'], true)) {
            $db->prepare('DELETE FROM jobs WHERE id = ?')->execute([$row['id']]);
            json_out(['ok' => true]);
        }
        json_out(['job' => job_row($row)]);
    }

    // Phones: per-device upload tokens, added and revoked from the Team page.
    if ($route === 'phones' && $method === 'GET') {
        $rows = $db->query('SELECT id, name, created_at, last_used_at FROM phones ORDER BY id')->fetchAll();
        json_out(['phones' => array_map(fn ($r) => [
            'id' => (int) $r['id'],
            'name' => $r['name'],
            'created_at' => $r['created_at'],
            'last_used_at' => $r['last_used_at'],
        ], $rows)]);
    }
    if ($route === 'phones' && $method === 'POST') {
        $name = trim((string) (body_json()['name'] ?? '')) ?: 'Phone';
        $token = bin2hex(random_bytes(24));
        $db->prepare('INSERT INTO phones (name, token_hash, created_by, created_at) VALUES (?, ?, ?, ?)')
            ->execute([$name, hash('sha256', $token), $user['id'], now()]);
        $id = (int) $db->lastInsertId();
        // The token is shown once, inside a QR the phone scans; only its hash is stored. The camera
        // link opens the web camera with the token in the fragment, which never reaches the server.
        json_out([
            'phone' => ['id' => $id, 'name' => $name],
            'camera_url' => site_origin() . '/camera#' . $token,
            'pairing' => json_encode([
                'v' => 1,
                'kind' => 'varbox-phone',
                'url' => site_origin(),
                'token' => $token,
                'name' => $name,
            ], JSON_UNESCAPED_SLASHES),
        ]);
    }
    if (preg_match('#^phones/(\d+)$#', $route, $m) && $method === 'DELETE') {
        $db->prepare('DELETE FROM phones WHERE id = ?')->execute([(int) $m[1]]);
        json_out(['ok' => true]);
    }

    // Workers
    if ($route === 'workers') {
        $rows = $db->query('SELECT * FROM workers ORDER BY last_seen DESC')->fetchAll();
        $out = array_map(fn ($r) => [
            'name' => $r['name'],
            'info' => json_decode($r['info'], true) ?: [],
            'last_seen' => $r['last_seen'],
            'online' => (time() - strtotime($r['last_seen'])) < WORKER_ONLINE_SECONDS,
        ], $rows);
        json_out(['workers' => $out, 'token' => config()['worker_token'] ?? null]);
    }
    fail('Unknown request.', 404);
}

function api_login(): never
{
    start_session();
    $db = db();
    $ip = client_ip();
    $db->prepare('DELETE FROM login_attempts WHERE at < ?')->execute([time() - 900]);
    $count = $db->prepare('SELECT COUNT(*) FROM login_attempts WHERE ip = ?');
    $count->execute([$ip]);
    if ((int) $count->fetchColumn() >= 8) {
        fail('Too many attempts. Wait fifteen minutes, then try again.', 429);
    }
    $in = body_json();
    $stmt = $db->prepare('SELECT * FROM users WHERE email = ?');
    $stmt->execute([strtolower(trim((string) ($in['email'] ?? '')))]);
    $user = $stmt->fetch();
    // Only the owner keeps a password, as a fallback. Everyone else signs in with Google.
    if (
        !$user
        || (int) $user['id'] !== owner_id()
        || !password_verify((string) ($in['password'] ?? ''), $user['pass_hash'])
    ) {
        $db->prepare('INSERT INTO login_attempts (ip, at) VALUES (?, ?)')->execute([$ip, time()]);
        fail('Check the email and password, then try again.', 401);
    }
    session_regenerate_id(true);
    $_SESSION['user_id'] = (int) $user['id'];
    json_out(['ok' => true]);
}

/** The public OAuth web client ID; overridable in config. The secret is never needed. */
const GOOGLE_CLIENT_ID_DEFAULT =
    '744697579975-1rm1mp8r9s7bc15k7lqba8stlgshbdo1.apps.googleusercontent.com';

function google_client_id(): string
{
    return (string) (config()['google_client_id'] ?? GOOGLE_CLIENT_ID_DEFAULT);
}

/** Fetch and decode Google's claims for an ID token, or null when the request fails. */
function google_verify_token(string $idToken): ?array
{
    // tokeninfo checks the signature and expiry and returns the claims. The URL is overridable so
    // tests can point it at a stub.
    $base = (string) (config()['google_tokeninfo_url'] ?? 'https://oauth2.googleapis.com/tokeninfo');
    $url = $base . (str_contains($base, '?') ? '&' : '?') . 'id_token=' . urlencode($idToken);
    $raw = http_get($url);
    if ($raw === null) {
        return null;
    }
    $data = json_decode($raw, true);
    return is_array($data) ? $data : null;
}

/** Simple HTTPS GET, via curl when available, else the stream wrapper. Null on failure. */
function http_get(string $url): ?string
{
    if (function_exists('curl_init')) {
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 10,
            CURLOPT_FOLLOWLOCATION => true,
        ]);
        $out = curl_exec($ch);
        $ok = $out !== false && curl_getinfo($ch, CURLINFO_HTTP_CODE) < 400;
        // No curl_close(): the handle frees itself when it goes out of scope, and the function is
        // deprecated from PHP 8.5, where its warning would corrupt the response body.
        return $ok ? (string) $out : null;
    }
    $ctx = stream_context_create(['http' => ['timeout' => 10, 'ignore_errors' => true]]);
    $out = @file_get_contents($url, false, $ctx);
    return $out === false ? null : $out;
}

/** Sign in with a Google ID token. Only accounts already on the team get in. */
function api_login_google(): never
{
    start_session();
    $db = db();
    $ip = client_ip();
    $db->prepare('DELETE FROM login_attempts WHERE at < ?')->execute([time() - 900]);
    $count = $db->prepare('SELECT COUNT(*) FROM login_attempts WHERE ip = ?');
    $count->execute([$ip]);
    if ((int) $count->fetchColumn() >= 8) {
        fail('Too many attempts. Wait fifteen minutes, then try again.', 429);
    }
    $record_attempt = function () use ($db, $ip): void {
        $db->prepare('INSERT INTO login_attempts (ip, at) VALUES (?, ?)')->execute([$ip, time()]);
    };

    $token = (string) (body_json()['credential'] ?? '');
    $claims = $token !== '' ? google_verify_token($token) : null;
    $okIssuers = ['accounts.google.com', 'https://accounts.google.com'];
    $verified = is_array($claims)
        && hash_equals(google_client_id(), (string) ($claims['aud'] ?? ''))
        && in_array((string) ($claims['iss'] ?? ''), $okIssuers, true)
        && (int) ($claims['exp'] ?? 0) > time()
        && filter_var($claims['email_verified'] ?? false, FILTER_VALIDATE_BOOLEAN);
    if (!$verified) {
        $record_attempt();
        fail('Google sign-in could not be verified. Try again.', 401);
    }

    $sub = (string) ($claims['sub'] ?? '');
    $email = strtolower(trim((string) ($claims['email'] ?? '')));
    $user = null;
    if ($sub !== '') {
        $stmt = $db->prepare('SELECT * FROM users WHERE google_sub = ?');
        $stmt->execute([$sub]);
        $user = $stmt->fetch() ?: null;
    }
    if (!$user && $email !== '') {
        $stmt = $db->prepare('SELECT * FROM users WHERE google_email = ?');
        $stmt->execute([$email]);
        $user = $stmt->fetch() ?: null;
        // A row already linked to one Google account accepts only that account, even when another
        // account carries the same email. The owner unlinks it by setting the email again.
        if ($user && (string) ($user['google_sub'] ?? '') !== '' && !hash_equals((string) $user['google_sub'], $sub)) {
            $user = null;
        }
    }
    if (!$user) {
        $record_attempt();
        fail('This Google account is not on the VAR Box team. Ask Shem to add it.', 403);
    }
    // Remember the account id on first sign-in, so later sign-ins match on it even if the email moves.
    if (($user['google_sub'] ?? null) === null || $user['google_sub'] === '') {
        $db->prepare('UPDATE users SET google_sub = ?, google_email = COALESCE(google_email, ?) WHERE id = ?')
            ->execute([$sub, $email, $user['id']]);
    }
    session_regenerate_id(true);
    $_SESSION['user_id'] = (int) $user['id'];
    json_out(['ok' => true]);
}

/** First run: the owner account, gated by the one-time setup code written at deploy time. */
function api_setup(): never
{
    start_session();
    $db = db();
    if ((int) $db->query('SELECT COUNT(*) FROM users')->fetchColumn() > 0) {
        fail('Setup is already complete. Sign in instead.', 409);
    }
    $in = body_json();
    $code = (string) (config()['setup_code'] ?? '');
    if ($code === '' || !hash_equals($code, trim((string) ($in['code'] ?? '')))) {
        fail('Check the setup code, then try again.', 403);
    }
    $email = strtolower(trim((string) ($in['email'] ?? '')));
    $name = trim((string) ($in['name'] ?? ''));
    $password = (string) ($in['password'] ?? '');
    if (!filter_var($email, FILTER_VALIDATE_EMAIL) || $name === '' || strlen($password) < 10) {
        fail('Enter a name, a valid email and a password of at least 10 characters.');
    }
    $db->prepare('INSERT INTO users (email, name, pass_hash, created_at) VALUES (?, ?, ?, ?)')
        ->execute([$email, $name, password_hash($password, PASSWORD_DEFAULT), now()]);
    session_regenerate_id(true);
    $_SESSION['user_id'] = (int) $db->lastInsertId();
    json_out(['ok' => true]);
}

function worker_api(string $route, string $method): never
{
    $token = (string) (config()['worker_token'] ?? '');
    $given = preg_replace('/^Bearer\s+/i', '', (string) ($_SERVER['HTTP_AUTHORIZATION'] ?? ''));
    if ($token === '' || !hash_equals($token, (string) $given)) {
        fail('Worker token rejected.', 401);
    }
    $db = db();
    $in = in_array($method, ['POST', 'PATCH'], true)
        && str_contains((string) ($_SERVER['CONTENT_TYPE'] ?? ''), 'json') ? body_json() : [];

    if ($route === 'heartbeat') {
        $name = (string) ($in['name'] ?? 'gpu');
        $db->prepare('INSERT INTO workers (name, info, last_seen) VALUES (?, ?, ?)
                      ON CONFLICT(name) DO UPDATE SET info = excluded.info, last_seen = excluded.last_seen')
            ->execute([$name, json_encode($in), now()]);
        json_out(['ok' => true]);
    }
    if ($route === 'claim') {
        $db->beginTransaction();
        $job = $db->query("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")->fetch();
        if ($job) {
            $db->prepare("UPDATE jobs SET status = 'running', worker = ?, stage = 'starting', updated_at = ? WHERE id = ?")
                ->execute([(string) ($in['name'] ?? 'gpu'), now(), $job['id']]);
        }
        $db->commit();
        if (!$job) {
            json_out(['job' => null]);
        }
        $out = job_row($job);
        foreach (['a', 'b'] as $cam) {
            if (!$job["camera_$cam"]) {
                $out["camera_$cam"] = null;
                continue;
            }
            $upload = find('uploads', (int) $job["camera_$cam"]);
            $row = upload_row($upload);
            // The phone's sidecar (settings, clock offset, per-frame capture times) rides along so
            // the GPU can align the cameras by timestamp as well as by audio.
            $row['sidecar'] = $upload['sidecar'] ? json_decode($upload['sidecar'], true) : null;
            $out["camera_$cam"] = $row;
        }
        json_out(['job' => $out]);
    }
    // A recording's cameras by its take name, with their sidecars: what the sync test measures.
    if (preg_match('#^recordings/([A-Za-z0-9_.\-]+)$#', $route, $m) && $method === 'GET') {
        $stmt = $db->prepare('SELECT * FROM recordings WHERE rec_uid = ?');
        $stmt->execute([$m[1]]);
        $rec = $stmt->fetch();
        if (!$rec) {
            fail('No recording with that take name.', 404);
        }
        $cams = $db->prepare('SELECT * FROM recording_cameras WHERE recording_id = ? ORDER BY letter');
        $cams->execute([$rec['id']]);
        $out = [];
        foreach ($cams->fetchAll() as $c) {
            $u = find('uploads', (int) $c['upload_id']);
            $out[] = [
                'letter' => $c['letter'],
                'upload_id' => (int) $u['id'],
                'status' => $u['status'],
                'size' => (int) $u['size'],
                'sidecar' => $u['sidecar'] ? json_decode($u['sidecar'], true) : null,
            ];
        }
        json_out(['recording' => ['rec_uid' => $rec['rec_uid'], 'title' => $rec['title']], 'cameras' => $out]);
    }
    if (preg_match('#^uploads/(\d+)$#', $route, $m)) {
        send_storage_file('uploads/' . (int) $m[1] . '.bin', true);
    }
    if (preg_match('#^jobs/(\d+)/(progress|preview|done|fail)$#', $route, $m)) {
        $job = find('jobs', (int) $m[1]);
        if ($m[2] === 'progress') {
            $db->prepare('UPDATE jobs SET stage = ?, pct = ?, message = ?, updated_at = ? WHERE id = ?')
                ->execute([(string) ($in['stage'] ?? ''), isset($in['pct']) ? (float) $in['pct'] : null,
                    (string) ($in['message'] ?? ''), now(), $job['id']]);
        } elseif ($m[2] === 'preview') {
            $name = preg_replace('/[^A-Za-z0-9_]/', '', (string) ($_GET['name'] ?? 'preview')) ?: 'preview';
            $dir = storage_path('previews/job_' . $job['id']);
            if (!is_dir($dir)) {
                mkdir($dir, 0775, true);
            }
            file_put_contents("$dir/$name.jpg", file_get_contents('php://input') ?: '');
            $list = json_decode($job['previews'] ?: '[]', true) ?: [];
            if (!in_array($name, $list, true)) {
                $list[] = $name;
            }
            $db->prepare('UPDATE jobs SET previews = ? WHERE id = ?')->execute([json_encode($list), $job['id']]);
        } elseif ($m[2] === 'done') {
            $db->prepare("UPDATE jobs SET status = 'done', stage = 'done', pct = 1, updated_at = ? WHERE id = ?")
                ->execute([now(), $job['id']]);
        } else {
            $db->prepare("UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?")
                ->execute([(string) ($in['message'] ?? 'Processing stopped'), now(), $job['id']]);
        }
        json_out(['ok' => true]);
    }
    if ($route === 'analyses' && $method === 'POST') {
        $analysis = (array) ($in['analysis'] ?? []);
        $jobId = isset($in['job_id']) ? (int) $in['job_id'] : null;
        $db->prepare('INSERT INTO analyses (title, job_id, meta, created_at) VALUES (?, ?, ?, ?)')
            ->execute([(string) ($analysis['title'] ?? 'Analysis'), $jobId, json_encode($analysis), now()]);
        json_out(['id' => (int) $db->lastInsertId()]);
    }
    if (preg_match('#^analyses/(\d+)/files$#', $route, $m) && $method === 'PUT') {
        $row = find('analyses', (int) $m[1]);
        $rel = safe_rel_path((string) ($_GET['path'] ?? ''));
        $received = append_chunk(storage_path('analyses/' . $row['id'] . '/' . $rel), (int) ($_GET['offset'] ?? -1), (int) ($_GET['total'] ?? 0));
        json_out(['received' => $received]);
    }
    if (preg_match('#^analyses/(\d+)/finalize$#', $route, $m)) {
        $row = find('analyses', (int) $m[1]);
        $analysis = (array) ($in['analysis'] ?? json_decode($row['meta'], true));
        if (!empty($in['replace'])) {
            // A republished analysis takes the place of earlier versions with the same title.
            $old = $db->prepare("SELECT id FROM analyses WHERE title = ? AND id != ? AND status = 'ready'");
            $old->execute([(string) ($analysis['title'] ?? $row['title']), $row['id']]);
            foreach ($old->fetchAll() as $o) {
                delete_analysis((int) $o['id']);
            }
        }
        $db->prepare("UPDATE analyses SET status = 'ready', title = ?, meta = ? WHERE id = ?")
            ->execute([(string) ($analysis['title'] ?? $row['title']), json_encode($analysis), $row['id']]);
        if ($row['job_id']) {
            $db->prepare('UPDATE jobs SET analysis_id = ? WHERE id = ?')->execute([$row['id'], $row['job_id']]);
        }
        json_out(['ok' => true]);
    }
    fail('Unknown worker request.', 404);
}

/**
 * Server time for an NTP-style exchange: when PHP received the request and when it replies, both
 * in milliseconds with microsecond precision. A client takes many of these and keeps the fastest.
 */
function clock_reply(): never
{
    $received = (float) ($_SERVER['REQUEST_TIME_FLOAT'] ?? microtime(true)) * 1000.0;
    header('Content-Type: application/json');
    header('Cache-Control: no-store');
    echo json_encode(['t1' => $received, 't2' => microtime(true) * 1000.0]);
    exit;
}

/** The phone that owns the request, by its upload token; updates last seen. Rejects otherwise. */
function require_phone(): array
{
    $given = preg_replace('/^Bearer\s+/i', '', (string) ($_SERVER['HTTP_AUTHORIZATION'] ?? ''));
    if ($given === '') {
        fail('Phone token missing.', 401);
    }
    $stmt = db()->prepare('SELECT * FROM phones WHERE token_hash = ?');
    $stmt->execute([hash('sha256', (string) $given)]);
    $phone = $stmt->fetch();
    if (!$phone) {
        fail('Phone token rejected.', 401);
    }
    db()->prepare('UPDATE phones SET last_used_at = ? WHERE id = ?')->execute([now(), $phone['id']]);
    return $phone;
}

/** Seconds a sidecar covers, from its first and last frame capture times (shared clock). */
function sidecar_duration(?string $sidecar): float
{
    if ($sidecar === null || $sidecar === '') {
        return 0.0;
    }
    $data = json_decode($sidecar, true);
    $frames = is_array($data) ? ($data['frames'] ?? null) : null;
    if (!is_array($frames) || count($frames) < 2) {
        return 0.0;
    }
    $first = (float) reset($frames);
    $last = (float) end($frames);
    return max(0.0, ($last - $first) / 1000.0);
}

/**
 * Upload-only API for paired phones. Token-authenticated, it reuses the chunked upload code and
 * groups a take's cameras under one recording. The last camera to arrive makes the pending task.
 */
function phone_api(string $route, string $method): never
{
    $phone = require_phone();
    $db = db();
    $in = $method === 'POST' && str_contains((string) ($_SERVER['CONTENT_TYPE'] ?? ''), 'json')
        ? body_json() : [];

    if ($route === 'ping') {
        json_out(['ok' => true, 'phone' => ['id' => (int) $phone['id'], 'name' => $phone['name']]]);
    }

    // Register (or update) a recording, then open an upload slot for this phone's camera.
    if ($route === 'uploads' && $method === 'POST') {
        $recUid = trim((string) ($in['rec_uid'] ?? ''));
        $letter = strtoupper(substr(trim((string) ($in['letter'] ?? '')), 0, 1));
        if ($recUid === '' || !preg_match('/^[A-Z]$/', $letter)) {
            fail('A recording id and a camera letter (A, B, …) are required.');
        }
        $rec = find_or_make_recording($recUid, $in);
        $name = preg_replace('/[^A-Za-z0-9_.\-]+/', '_', (string) ($in['name'] ?? "r_{$letter}.mp4"));
        $size = (int) ($in['size'] ?? 0);
        if ($size <= 0) {
            fail('Give the video size.');
        }
        $db->prepare('INSERT INTO uploads (name, size, phone_id, created_at) VALUES (?, ?, ?, ?)')
            ->execute([$name, $size, $phone['id'], now()]);
        $uploadId = (int) $db->lastInsertId();
        $db->prepare(
            'INSERT INTO recording_cameras (recording_id, letter, upload_id, device, created_at)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(recording_id, letter)
             DO UPDATE SET upload_id = excluded.upload_id, device = excluded.device, status = \'receiving\''
        )->execute([$rec['id'], $letter, $uploadId, (string) ($in['device'] ?? ''), now()]);
        json_out(['recording_id' => $rec['rec_uid'], 'upload' => upload_row(find('uploads', $uploadId))]);
    }

    // How much of an upload has arrived, so a phone resumes where it stopped after a drop.
    if (preg_match('#^uploads/(\d+)$#', $route, $m) && $method === 'GET') {
        $row = find('uploads', (int) $m[1]);
        if ((int) $row['phone_id'] !== (int) $phone['id']) {
            fail('That upload belongs to another phone.', 403);
        }
        $file = storage_path('uploads/' . $row['id'] . '.bin');
        json_out([
            'received' => is_file($file) ? (int) filesize($file) : 0,
            'size' => (int) $row['size'],
            'status' => $row['status'],
            'sidecar' => $row['sidecar'] !== null && $row['sidecar'] !== '',
        ]);
    }

    if (preg_match('#^uploads/(\d+)$#', $route, $m) && $method === 'PUT') {
        $row = find('uploads', (int) $m[1]);
        if ((int) $row['phone_id'] !== (int) $phone['id']) {
            fail('That upload belongs to another phone.', 403);
        }
        $received = append_chunk(storage_path('uploads/' . $row['id'] . '.bin'), (int) ($_GET['offset'] ?? -1), (int) $row['size']);
        $status = $received >= (int) $row['size'] ? 'ready' : 'receiving';
        $db->prepare('UPDATE uploads SET received = ?, status = ? WHERE id = ?')->execute([$received, $status, $row['id']]);
        if ($status === 'ready') {
            $db->prepare("UPDATE recording_cameras SET status = 'ready' WHERE upload_id = ?")->execute([$row['id']]);
            complete_recording_for_upload((int) $row['id']);
        }
        json_out(['received' => $received, 'status' => $status]);
    }

    // The sidecar JSON beside a phone's video: device, settings applied, clock offset, frame times.
    if (preg_match('#^uploads/(\d+)/sidecar$#', $route, $m) && $method === 'PUT') {
        $row = find('uploads', (int) $m[1]);
        if ((int) $row['phone_id'] !== (int) $phone['id']) {
            fail('That upload belongs to another phone.', 403);
        }
        $raw = file_get_contents('php://input') ?: '';
        if (strlen($raw) > CHUNK_LIMIT || json_decode($raw) === null) {
            fail('Send the sidecar as one JSON body.');
        }
        $db->prepare('UPDATE uploads SET sidecar = ? WHERE id = ?')->execute([$raw, $row['id']]);
        json_out(['ok' => true]);
    }

    fail('Unknown phone request.', 404);
}

function find_or_make_recording(string $recUid, array $in): array
{
    $db = db();
    $stmt = $db->prepare('SELECT * FROM recordings WHERE rec_uid = ?');
    $stmt->execute([$recUid]);
    $rec = $stmt->fetch();
    if ($rec) {
        return $rec;
    }
    $fps = (int) ($in['fps'] ?? 30);
    $fps = in_array($fps, [30, 60], true) ? $fps : 30;
    $expected = max(1, min(6, (int) ($in['expected_cameras'] ?? 2)));
    $title = trim((string) ($in['title'] ?? '')) ?: 'Recording ' . date('j M Y, H:i');
    $db->prepare(
        'INSERT INTO recordings (rec_uid, rig, round, title, fps, expected_cameras, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
    )->execute([
        $recUid, (string) ($in['rig'] ?? ''), (string) ($in['round'] ?? ''),
        $title, $fps, $expected, now(), now(),
    ]);
    return find('recordings', (int) $db->lastInsertId());
}

/** After a camera finishes, make the pending task once every expected camera has arrived. */
function complete_recording_for_upload(int $uploadId): void
{
    $db = db();
    $stmt = $db->prepare('SELECT recording_id FROM recording_cameras WHERE upload_id = ?');
    $stmt->execute([$uploadId]);
    $recId = $stmt->fetchColumn();
    if ($recId === false) {
        return;
    }
    $rec = find('recordings', (int) $recId);
    if ($rec['status'] !== 'open') {
        return;
    }
    $cams = $db->prepare('SELECT * FROM recording_cameras WHERE recording_id = ? ORDER BY letter');
    $cams->execute([$rec['id']]);
    $cams = $cams->fetchAll();
    $ready = array_filter($cams, fn ($c) => $c['status'] === 'ready');
    if (count($ready) < (int) $rec['expected_cameras']) {
        return;
    }
    create_pending_job($rec, array_values($ready));
}

/** Build the pending job from a recording's ready cameras and email Shem to start the GPU. */
function create_pending_job(array $rec, array $cams): void
{
    $db = db();
    // The job schema carries two cameras; take the first two by letter (A, B).
    $a = $cams[0];
    $b = $cams[1] ?? null;
    $db->prepare(
        'INSERT INTO jobs (title, camera_a, camera_b, fps, status, recording_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, \'pending\', ?, ?, ?)'
    )->execute([
        $rec['title'], (int) $a['upload_id'], $b ? (int) $b['upload_id'] : null,
        (int) $rec['fps'], (int) $rec['id'], now(), now(),
    ]);
    $jobId = (int) $db->lastInsertId();
    $db->prepare("UPDATE recordings SET status = 'ready', job_id = ?, updated_at = ? WHERE id = ?")
        ->execute([$jobId, now(), $rec['id']]);

    [$subject, $body] = pending_job_email($rec, $cams, $jobId);
    send_mail($subject, $body);
}

/** Subject and body for the "ready to process" email. */
function pending_job_email(array $rec, array $cams, int $jobId): array
{
    $lines = [];
    foreach ($cams as $c) {
        $u = find('uploads', (int) $c['upload_id']);
        $dur = sidecar_duration($u['sidecar']);
        $parts = ["camera {$c['letter']}"];
        if ($c['device']) {
            $parts[] = (string) $c['device'];
        }
        $parts[] = number_format((int) $u['size'] / 1e6, 1) . ' MB';
        if ($dur > 0) {
            $parts[] = gmdate('i:s', (int) round($dur)) . ' long';
        }
        $lines[] = '  - ' . implode(', ', $parts);
    }
    $token = (string) (config()['worker_token'] ?? 'WORKER_TOKEN');
    $origin = site_origin();
    $body = implode("\n", [
        "A new recording is ready to process on VAR Box.",
        '',
        $rec['title'] . ' (' . (int) $rec['fps'] . ' fps)',
        implode("\n", $lines),
        '',
        'Open Pending tasks and press Process:',
        "  $origin/app/jobs",
        '',
        'The GPU worker claims it once it is queued. Start the worker with:',
        "  export VARBOX_WORKER_TOKEN=$token",
        "  python -m boxing_analytics.mesh4d.worker serve --server $origin",
        '',
    ]);
    return ['VAR Box: ' . $rec['title'] . ' ready to process', $body];
}
