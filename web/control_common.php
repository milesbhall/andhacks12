<?php
declare(strict_types=1);

// Keep the config in the private folder beside public_html, or set a server-side
// environment variable to an absolute path outside the web root.
const CONTROL_TOKEN_SHA256 = '8c18635fa05439208b04a01531183e978bc6307e837b582c916eb207b1b9a38c';

function control_json(int $code, array $body): never {
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    header('X-Content-Type-Options: nosniff');
    echo json_encode($body, JSON_UNESCAPED_SLASHES | JSON_INVALID_UTF8_SUBSTITUTE);
    exit;
}

function control_config(): ?array {
    $path = getenv('CONTROL_CONFIG_PATH') ?: dirname(__DIR__) . '/incredible_trades_private/control_config.php';
    $real = realpath($path);
    $web = realpath(__DIR__);
    if (!$real || !$web || str_starts_with($real, $web . DIRECTORY_SEPARATOR)) return null;
    $config = require $real;
    if (!is_array($config) || empty($config['password_hash']) || empty($config['storage_dir'])) return null;
    $dir = realpath($config['storage_dir']);
    if (!$dir || str_starts_with($dir, $web . DIRECTORY_SEPARATOR) || $dir === $web || !is_writable($dir)) return null;
    $config['storage_dir'] = $dir;
    return $config;
}

function control_worker_auth(): void {
    $token = $_SERVER['HTTP_X_UPLOAD_TOKEN'] ?? '';
    if (!is_string($token) || !hash_equals(CONTROL_TOKEN_SHA256, hash('sha256', $token))) {
        control_json(403, ['ok' => false]);
    }
}

function control_require_https(): void {
    if (!in_array(strtolower((string)($_SERVER['HTTPS'] ?? '')), ['on', '1'], true) &&
        (string)($_SERVER['SERVER_PORT'] ?? '') !== '443') {
        control_json(403, ['ok' => false, 'error' => 'HTTPS required']);
    }
}

function control_session(): void {
    if (session_status() === PHP_SESSION_ACTIVE) return;
    session_name('it_operator');
    session_set_cookie_params(['httponly' => true, 'secure' => true,
        'samesite' => 'Strict', 'path' => '/']);
    session_start();
}

function control_csrf(): string {
    if (empty($_SESSION['csrf'])) $_SESSION['csrf'] = bin2hex(random_bytes(32));
    return $_SESSION['csrf'];
}

function control_require_login(): void {
    if (empty($_SESSION['operator'])) control_json(401, ['ok' => false, 'error' => 'Sign in required']);
}

function control_check_csrf(): void {
    $given = $_SERVER['HTTP_X_CSRF_TOKEN'] ?? ($_POST['csrf'] ?? '');
    if (!is_string($given) || !hash_equals(control_csrf(), $given)) control_json(403, ['ok' => false, 'error' => 'Invalid session token']);
}

function control_store(array $config, callable $callback): mixed {
    $path = $config['storage_dir'] . '/control_state.json';
    $lock = fopen($config['storage_dir'] . '/control.lock', 'c');
    if (!$lock || !flock($lock, LOCK_EX)) control_json(500, ['ok' => false, 'error' => 'Storage unavailable']);
    try {
        $state = is_file($path) ? json_decode((string)file_get_contents($path), true) : null;
        if (!is_array($state)) $state = ['command' => null, 'worker' => ['phase' => 'offline']];
        $pending = $state['command'] ?? null;
        if (is_array($pending) && ($pending['action'] ?? '') === 'start' &&
            (int)($pending['created_at'] ?? 0) < time() - 120) {
            $id = (string)($pending['upload_id'] ?? '');
            if (preg_match('/^[a-f0-9]{32}\.(mp3|m4a|wav|webm|ogg|mp4)$/', $id)) @unlink($config['storage_dir'] . '/' . $id);
            $state['command'] = null;
            $state['active_upload_id'] = null;
        }
        if (is_array($pending) && ($pending['action'] ?? '') === 'stop' &&
            (int)($pending['created_at'] ?? 0) < time() - 10800) $state['command'] = null;
        if (!empty($state['active_upload_id']) && (int)($state['upload_created_at'] ?? 0) < time() - 10800) {
            $id = (string)$state['active_upload_id'];
            if (preg_match('/^[a-f0-9]{32}\.(mp3|m4a|wav|webm|ogg|mp4)$/', $id)) @unlink($config['storage_dir'] . '/' . $id);
            $state['active_upload_id'] = null;
        }
        foreach (glob($config['storage_dir'] . '/*') ?: [] as $candidate) {
            $id = basename($candidate);
            if (preg_match('/^[a-f0-9]{32}\.(mp3|m4a|wav|webm|ogg|mp4)$/', $id) &&
                is_file($candidate) && filemtime($candidate) < time() - 10800) @unlink($candidate);
        }
        foreach (($state['jobs'] ?? []) as $jid => $job) {
            if ((int)($job['created_at'] ?? 0) < time() - 900) unset($state['jobs'][$jid]);
        }
        $result = $callback($state);
        $tmp = $path . '.' . bin2hex(random_bytes(6)) . '.tmp';
        $encoded = json_encode($state, JSON_INVALID_UTF8_SUBSTITUTE | JSON_UNESCAPED_SLASHES);
        if ($encoded === false || file_put_contents($tmp, $encoded, LOCK_EX) === false || !rename($tmp, $path)) {
            @unlink($tmp);
            control_json(500, ['ok' => false, 'error' => 'Storage unavailable']);
        }
        return $result;
    } finally {
        flock($lock, LOCK_UN);
        fclose($lock);
    }
}

function control_public_state(array $state): array {
    $worker = $state['worker'] ?? [];
    $heartbeat = (int)($worker['heartbeat'] ?? 0);
    $online = $heartbeat > 0 && time() - $heartbeat <= 15;
    return [
        'online' => $online,
        'phase' => $online ? ($worker['phase'] ?? 'unknown') : 'offline',
        'source' => $online ? ($worker['source'] ?? '') : '',
        'message' => $online ? ($worker['message'] ?? '') : '',
        'updated_at' => $heartbeat ?: null,
        'pending' => !empty($state['command']),
    ];
}

function control_source_url(string $value, array $config): ?string {
    if (strlen($value) > 2048 || preg_match('/[\x00-\x20\x7f]/', $value)) return null;
    $url = parse_url($value);
    if (!$url || strtolower($url['scheme'] ?? '') !== 'https' || empty($url['host']) || isset($url['user']) || isset($url['pass']) || isset($url['port'])) return null;
    $host = strtolower(rtrim($url['host'], '.'));
    $allowed = array_map('strtolower', array_merge($config['source_hosts'] ?? [],
        ['www.federalreserve.gov', 'www.youtube.com', 'youtube.com', 'm.youtube.com', 'youtu.be']));
    if (!in_array($host, $allowed, true)) return null;
    return $value;
}

function control_session_options(array $p): array {
    $mode = in_array($p['mode'] ?? 'dry', ['dry', 'demo', 'live'], true) ? (string)$p['mode'] : 'dry';
    // Real money only with the typed confirmation from the page.
    if ($mode === 'live' && (string)($p['confirm_live'] ?? '') !== 'LIVE') $mode = 'dry';
    $speed = (float)($p['speed'] ?? 0);
    if (!in_array($speed, [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0], true)) $speed = 0.0;
    $speaker = (string)($p['speaker'] ?? 'kevin_warsh');
    if (!preg_match('/^[a-z_]{3,40}$/', $speaker)) $speaker = 'kevin_warsh';
    $qty = max(1, min(5, (int)($p['qty'] ?? 2)));
    $venues = array_values(array_intersect(['kalshi', 'polymarket'], (array)($p['venues'] ?? ['kalshi', 'polymarket'])));
    if (!$venues) $venues = ['kalshi', 'polymarket'];
    return ['mode' => $mode, 'speaker' => $speaker, 'qty' => $qty, 'venues' => $venues,
            'speed' => $speed, 'confirm_live' => $mode === 'live' ? 'LIVE' : '', 'recommenders' => true];
}
