<?php
declare(strict_types=1);
require __DIR__ . '/control_common.php';
control_require_https();

$config = control_config();
if (!$config) control_json(503, ['ok' => false, 'configured' => false, 'error' => 'Operator controls are not configured']);
control_session();
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$action = $_GET['action'] ?? '';

if ($method === 'GET') {
    if (empty($_SESSION['operator'])) control_json(200, ['ok' => true, 'configured' => true, 'authenticated' => false]);
    $state = control_store($config, function (&$s) { return control_public_state($s); });
    control_json(200, ['ok' => true, 'configured' => true, 'authenticated' => true,
        'csrf' => control_csrf(), 'status' => $state]);
}
if ($method !== 'POST') control_json(405, ['ok' => false]);
if ($action === 'login') {
    $key = hash('sha256', (string)($_SERVER['REMOTE_ADDR'] ?? 'unknown'));
    $blocked = control_store($config, function (&$s) use ($key) {
        $entry = $s['login_attempts'][$key] ?? ['count' => 0, 'until' => 0];
        if (($entry['until'] ?? 0) < time() - 900) $entry = ['count' => 0, 'until' => 0];
        return ($entry['count'] ?? 0) >= 5 && ($entry['until'] ?? 0) > time();
    });
    if ($blocked) control_json(429, ['ok' => false, 'error' => 'Try again later']);
    $password = (string)($_POST['password'] ?? '');
    if (strlen($password) > 1024 || !password_verify($password, $config['password_hash'])) {
        control_store($config, function (&$s) use ($key) {
            $entry = $s['login_attempts'][$key] ?? ['count' => 0, 'until' => 0];
            $entry['count'] = (int)$entry['count'] + 1;
            $entry['until'] = time() + ($entry['count'] >= 5 ? 300 : 900);
            $s['login_attempts'][$key] = $entry;
        });
        control_json(401, ['ok' => false, 'error' => 'Invalid password']);
    }
    control_store($config, function (&$s) use ($key) { unset($s['login_attempts'][$key]); });
    session_regenerate_id(true);
    $_SESSION['operator'] = true;
    control_json(200, ['ok' => true, 'csrf' => control_csrf()]);
}
control_require_login();
control_check_csrf();
if ($action === 'logout') { $_SESSION = []; session_destroy(); control_json(200, ['ok' => true]); }
if (!in_array($action, ['start', 'stop'], true)) control_json(400, ['ok' => false]);

$uploadId = null;
$sourceUrl = null;
if ($action === 'start') {
    $type = $_POST['source_type'] ?? '';
    if ($type === 'url') {
        $sourceUrl = control_source_url((string)($_POST['source_url'] ?? ''), $config);
        if (!$sourceUrl) control_json(400, ['ok' => false, 'error' => 'Use an approved HTTPS source host']);
    } elseif ($type === 'upload') {
        $file = $_FILES['audio'] ?? null;
        $max = min(50 * 1024 * 1024, (int)($config['max_upload_bytes'] ?? 50 * 1024 * 1024));
        if (!$file || $file['error'] !== UPLOAD_ERR_OK || $file['size'] < 1 || $file['size'] > $max || !is_uploaded_file($file['tmp_name'])) {
            control_json(400, ['ok' => false, 'error' => 'Upload failed or exceeds the configured limit']);
        }
        $ext = strtolower(pathinfo((string)$file['name'], PATHINFO_EXTENSION));
        if (!in_array($ext, ['mp3', 'm4a', 'wav', 'webm', 'ogg', 'mp4'], true)) control_json(400, ['ok' => false, 'error' => 'Unsupported audio format']);
        $uploadId = bin2hex(random_bytes(16)) . '.' . $ext;
        if (!move_uploaded_file($file['tmp_name'], $config['storage_dir'] . '/' . $uploadId)) control_json(500, ['ok' => false, 'error' => 'Could not save upload']);
    } else control_json(400, ['ok' => false, 'error' => 'Choose an audio source']);
}
$result = control_store($config, function (&$s) use ($action, $uploadId, $sourceUrl, $config) {
    $worker = control_public_state($s);
    if ($action === 'start' && !$worker['online']) return 'offline';
    if ($action === 'start' && ($s['command'] || ($worker['online'] && !in_array($worker['phase'], ['idle', 'stopped', 'error'], true)))) return false;
    if ($action === 'stop' && ($s['command']['action'] ?? '') === 'stop') return true;
    if ($action === 'stop' && ($s['command']['action'] ?? '') === 'start' && !empty($s['active_upload_id'])) {
        $id = (string)$s['active_upload_id'];
        if (preg_match('/^[a-f0-9]{32}\.(mp3|m4a|wav|webm|ogg|mp4)$/', $id)) @unlink($config['storage_dir'] . '/' . $id);
        $s['active_upload_id'] = null;
    }
    $s['command'] = ['id' => bin2hex(random_bytes(16)), 'action' => $action, 'created_at' => time(),
        'source_type' => $uploadId ? 'upload' : ($sourceUrl ? 'url' : null),
        'upload_id' => $uploadId, 'source_url' => $sourceUrl];
    if ($action === 'start') {
        $s['active_upload_id'] = $uploadId;
        $s['upload_created_at'] = $uploadId ? time() : null;
    }
    return true;
});
if ($result === 'offline') {
    if ($uploadId) @unlink($config['storage_dir'] . '/' . $uploadId);
    control_json(409, ['ok' => false, 'error' => 'Local worker is offline']);
}
if (!$result) {
    if ($uploadId) @unlink($config['storage_dir'] . '/' . $uploadId);
    control_json(409, ['ok' => false, 'error' => 'A session or command is already active']);
}
control_json(200, ['ok' => true]);
