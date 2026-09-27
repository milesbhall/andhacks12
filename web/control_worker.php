<?php
declare(strict_types=1);
require __DIR__ . '/control_common.php';
control_require_https();
control_worker_auth();
$config = control_config();
if (!$config) control_json(503, ['ok' => false, 'error' => 'Operator controls are not configured']);
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$download = $_GET['download'] ?? null;
if ($method === 'GET' && $download !== null) {
    if (!preg_match('/^[a-f0-9]{32}\.(mp3|m4a|wav|webm|ogg|mp4)$/', (string)$download)) control_json(400, ['ok' => false]);
    $path = $config['storage_dir'] . '/' . $download;
    if (!is_file($path)) control_json(404, ['ok' => false]);
    header('Content-Type: application/octet-stream');
    header('Content-Length: ' . filesize($path));
    header('Cache-Control: no-store');
    header('X-Content-Type-Options: nosniff');
    readfile($path);
    exit;
}
if ($method === 'GET') {
    $out = control_store($config, function (&$s) {
        $jobs = [];
        foreach (($s['jobs'] ?? []) as $id => $job) {
            if (($job['status'] ?? '') !== 'queued') continue;
            $jobs[] = ['id' => $id, 'kind' => $job['kind'], 'text' => $job['text'], 'options' => $job['options'] ?? []];
            $s['jobs'][$id]['status'] = 'running';
        }
        return ['command' => $s['command'] ?? null, 'jobs' => $jobs];
    });
    control_json(200, ['ok' => true, 'command' => $out['command'], 'jobs' => $out['jobs']]);
}
if ($method !== 'POST') control_json(405, ['ok' => false]);
$body = file_get_contents('php://input', false, null, 0, 400001);
if (strlen($body) > 400000) control_json(413, ['ok' => false]);
$data = json_decode($body, true);
if (!is_array($data)) control_json(400, ['ok' => false]);
if (isset($data['job_id'])) {
    $jid = (string)$data['job_id'];
    if (!preg_match('/^[a-f0-9]{32}$/', $jid)) control_json(400, ['ok' => false]);
    $ok = in_array($data['job_status'] ?? '', ['done', 'error'], true);
    if (!$ok) control_json(400, ['ok' => false]);
    control_store($config, function (&$s) use ($jid, $data) {
        if (!isset($s['jobs'][$jid])) return;
        $s['jobs'][$jid]['status'] = $data['job_status'];
        $s['jobs'][$jid]['result'] = $data['job_result'] ?? null;
        $s['jobs'][$jid]['error'] = substr((string)($data['job_error'] ?? ''), 0, 300);
    });
    control_json(200, ['ok' => true]);
}
$phase = $data['phase'] ?? '';
if (!in_array($phase, ['idle', 'starting', 'running', 'stopping', 'stopped', 'error'], true)) control_json(400, ['ok' => false]);
$ack = (string)($data['ack'] ?? '');
$source = substr(preg_replace('/[^a-zA-Z0-9 ._:\/-]/', '', (string)($data['source'] ?? '')), 0, 120);
$message = substr(preg_replace('/[^a-zA-Z0-9 .,:;()_\/-]/', '', (string)($data['message'] ?? '')), 0, 160);
control_store($config, function (&$s) use ($phase, $ack, $source, $message, $config) {
    $s['worker'] = ['phase' => $phase, 'source' => $source, 'message' => $message, 'heartbeat' => time()];
    if ($ack !== '' && hash_equals((string)($s['command']['id'] ?? ''), $ack)) $s['command'] = null;
    if (in_array($phase, ['stopped', 'error'], true) && empty($s['command']) && !empty($s['active_upload_id'])) {
        $id = (string)$s['active_upload_id'];
        if (preg_match('/^[a-f0-9]{32}\.(mp3|m4a|wav|webm|ogg|mp4)$/', $id)) @unlink($config['storage_dir'] . '/' . $id);
        $s['active_upload_id'] = null;
    }
});
control_json(200, ['ok' => true]);
