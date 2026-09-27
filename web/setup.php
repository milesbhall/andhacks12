<?php
declare(strict_types=1);
require __DIR__ . '/control_common.php';
control_require_https();
header('Cache-Control: no-store');
header('Content-Security-Policy: default-src \'none\'; style-src \'unsafe-inline\'; form-action \'self\'; base-uri \'none\'; frame-ancestors \'none\'');
header('Referrer-Policy: no-referrer');
header('X-Content-Type-Options: nosniff');

$private = dirname(__DIR__) . '/incredible_trades_private';
$configPath = $private . '/control_config.php';
$keyPath = $private . '/setup.key';
$message = '';
$complete = false;

if (is_file($configPath)) {
    http_response_code(410);
    $message = 'Setup has already been completed.';
} elseif (!is_dir($private) || !is_writable($private) || !is_file($keyPath)) {
    http_response_code(503);
    $message = 'Operator setup is not ready.';
} elseif (($_SERVER['REQUEST_METHOD'] ?? 'GET') === 'POST') {
    $key = (string)($_POST['setup_key'] ?? '');
    $password = (string)($_POST['password'] ?? '');
    $confirm = (string)($_POST['confirm'] ?? '');
    $expected = trim((string)file_get_contents($keyPath));
    if (strlen($key) !== 64 || strlen($expected) !== 64 || !hash_equals($expected, $key)) {
        http_response_code(403);
        $message = 'Setup key is invalid.';
    } elseif (strlen($password) < 16 || strlen($password) > 256 || $password !== $confirm) {
        http_response_code(400);
        $message = 'Use a matching password of 16 to 256 characters.';
    } else {
        $lock = fopen($private . '/setup.lock', 'c');
        if (!$lock || !flock($lock, LOCK_EX)) {
            http_response_code(500);
            $message = 'Setup storage is unavailable.';
        } else {
            try {
                if (is_file($configPath) || !is_file($keyPath) ||
                    !hash_equals($expected, trim((string)file_get_contents($keyPath)))) {
                    http_response_code(409);
                    $message = 'Setup has already been used.';
                } else {
                    $hash = password_hash($password, PASSWORD_DEFAULT);
                    $contents = "<?php\nreturn [\n" .
                        "    'password_hash' => " . var_export($hash, true) . ",\n" .
                        "    'storage_dir' => __DIR__,\n" .
                        "    'source_hosts' => ['www.federalreserve.gov'],\n" .
                        "    'max_upload_bytes' => 50 * 1024 * 1024,\n" .
                        "];\n";
                    $temporary = $configPath . '.' . bin2hex(random_bytes(8)) . '.tmp';
                    if (file_put_contents($temporary, $contents, LOCK_EX) === false ||
                        !rename($temporary, $configPath)) {
                        @unlink($temporary);
                        http_response_code(500);
                        $message = 'Could not save the operator setup.';
                    } else {
                        @chmod($configPath, 0600);
                        @unlink($keyPath);
                        $complete = true;
                        $message = 'Operator setup complete. Sign in on the dashboard.';
                    }
                }
            } finally {
                flock($lock, LOCK_UN);
                fclose($lock);
            }
        }
    }
} elseif (($_SERVER['REQUEST_METHOD'] ?? 'GET') !== 'GET') {
    http_response_code(405);
    $message = 'Method not allowed.';
}

function h(string $value): string { return htmlspecialchars($value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); }
?><!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Incredible Trades operator setup</title>
<style>body{margin:0;background:#0b1019;color:#e5edf9;font:16px/1.5 system-ui,sans-serif}
main{max-width:520px;margin:8vh auto;padding:28px;background:#141d2b;border:1px solid #2b3950;border-radius:12px}
h1{font-size:24px;margin:0 0 10px}p{color:#aebed4}label{display:block;margin:18px 0 6px}
input{display:block;width:100%;box-sizing:border-box;padding:12px;background:#0b1019;color:#fff;border:1px solid #486083;border-radius:7px;font:inherit}
button{margin-top:22px;padding:12px 18px;background:#5b9dff;color:#09111e;border:0;border-radius:7px;font:inherit;font-weight:700;cursor:pointer}
a{color:#8fb9ff}</style></head><body><main><h1>Operator setup</h1>
<p><?=h($message ?: 'Enter the one-time setup key and choose a password. The password stays on this site; only its hash is saved outside the public website.')?></p>
<?php if ($complete): ?><p><a href="/">Open the dashboard</a></p>
<?php elseif (http_response_code() === 200 || (($_SERVER['REQUEST_METHOD'] ?? '') === 'POST' && in_array(http_response_code(), [400, 403], true))): ?>
<form method="post" autocomplete="off">
<label for="setup_key">One-time setup key</label><input id="setup_key" name="setup_key" required maxlength="64" autocomplete="off">
<label for="password">New operator password (16 characters minimum)</label><input id="password" name="password" type="password" required minlength="16" maxlength="256" autocomplete="new-password">
<label for="confirm">Confirm password</label><input id="confirm" name="confirm" type="password" required minlength="16" maxlength="256" autocomplete="new-password">
<button type="submit">Activate controls</button></form>
<?php endif; ?></main></body></html>
