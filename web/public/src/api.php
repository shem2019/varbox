<?php
declare(strict_types=1);

// JSON API for the dashboard (session login) and for GPU workers (bearer token).

const CHUNK_LIMIT = 16 * 1024 * 1024;
const WORKER_ONLINE_SECONDS = 60;

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
        json_out(['user' => $user, 'version' => VARBOX_VERSION]);
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
        $fps = in_array((int) ($in['fps'] ?? 30), [30, 60], true) ? (int) $in['fps'] : 30;
        $db->prepare(
            'INSERT INTO jobs (title, camera_a, camera_b, fps, start_s, duration_s, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        )->execute([
            $title, $a['id'], $b['id'] ?? null, $fps,
            max(0.0, (float) ($in['start_s'] ?? 0)), max(0.0, (float) ($in['duration_s'] ?? 0)), now(), now(),
        ]);
        json_out(['job' => job_row(find('jobs', (int) $db->lastInsertId()))]);
    }
    if (preg_match('#^jobs/(\d+)$#', $route, $m)) {
        $row = find('jobs', (int) $m[1]);
        if ($method === 'DELETE' && in_array($row['status'], ['queued', 'failed', 'done'], true)) {
            $db->prepare('DELETE FROM jobs WHERE id = ?')->execute([$row['id']]);
            json_out(['ok' => true]);
        }
        json_out(['job' => job_row($row)]);
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
    if (!$user || !password_verify((string) ($in['password'] ?? ''), $user['pass_hash'])) {
        $db->prepare('INSERT INTO login_attempts (ip, at) VALUES (?, ?)')->execute([$ip, time()]);
        fail('Check the email and password, then try again.', 401);
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
            $out["camera_$cam"] = $job["camera_$cam"] ? upload_row(find('uploads', (int) $job["camera_$cam"])) : null;
        }
        json_out(['job' => $out]);
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
