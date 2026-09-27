<?php
// Receives live state from the laptop (publish.py) and stores it for index.html.
// Only requests carrying the right token are accepted; the token itself is never stored here.
header('Content-Type: application/json');
$expected = '8c18635fa05439208b04a01531183e978bc6307e837b582c916eb207b1b9a38c';   // sha256 of the upload token
$token = $_SERVER['HTTP_X_UPLOAD_TOKEN'] ?? '';
if ($_SERVER['REQUEST_METHOD'] !== 'POST' || !hash_equals($expected, hash('sha256', $token))) {
    http_response_code(403);
    echo '{"ok":false}';
    exit;
}
$body = file_get_contents('php://input');
if (strlen($body) > 4000000 || json_decode($body) === null) {
    http_response_code(400);
    echo '{"ok":false,"error":"bad json"}';
    exit;
}
$dir = __DIR__ . '/data';
if (!is_dir($dir)) { mkdir($dir, 0755, true); }
$name = (($_GET['name'] ?? '') === 'archive') ? 'archive.json' : 'state.json';
$tmp = $dir . '/' . $name . '.tmp';
file_put_contents($tmp, $body, LOCK_EX);
rename($tmp, $dir . '/' . $name);
echo '{"ok":true}';
