<?php
// Auth0 authorization-code sign-in. Store secrets and allowlists outside the web root.
// Optional redirect_uri must be the exact Auth0 callback URL registered for this site.
declare(strict_types=1);
require __DIR__ . '/control_common.php';
control_require_https();
header('Cache-Control: no-store');
header('Referrer-Policy: no-referrer');
header('X-Content-Type-Options: nosniff');
control_session();

function auth0_fail(string $message, int $status = 403): never {
    http_response_code($status);
    header('Content-Type: text/html; charset=utf-8');
    echo '<!doctype html><meta charset="utf-8"><body style="background:#0b0e13;color:#e8edf5;font:16px system-ui;padding:40px">'
       . htmlspecialchars($message, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')
       . '<p><a style="color:#8fb9ff" href="/">Back to MarketPulse</a></p>';
    exit;
}

function auth0_config(): array {
    $path = dirname(__DIR__) . '/incredible_trades_private/auth0_config.php';
    $config = is_file($path) ? require $path : null;
    if (!is_array($config)) auth0_fail('Auth0 is not configured.', 503);
    $domain = $config['domain'] ?? '';
    $clientId = $config['client_id'] ?? '';
    $clientSecret = $config['client_secret'] ?? '';
    $redirect = $config['redirect_uri'] ?? 'https://yellow-boar-756344.hostingersite.com/auth0.php';
    $parts = is_string($redirect) ? parse_url($redirect) : false;
    if (!is_string($domain) || !preg_match('/^(?:[a-z0-9-]+\.)+[a-z0-9-]+$/iD', $domain) ||
        !is_string($clientId) || $clientId === '' || !is_string($clientSecret) || $clientSecret === '' ||
        !$parts || strtolower((string)($parts['scheme'] ?? '')) !== 'https' ||
        empty($parts['host']) || ($parts['path'] ?? '') !== '/auth0.php' ||
        isset($parts['user']) || isset($parts['pass']) || isset($parts['port']) ||
        isset($parts['query']) || isset($parts['fragment']) ||
        !is_array($config['allowed'] ?? null)) {
        auth0_fail('Auth0 is not configured correctly.', 503);
    }
    $config['redirect_uri'] = $redirect;
    return $config;
}

function auth0_request(string $url, ?array $fields = null, ?string $token = null): ?array {
    $ch = curl_init($url);
    if ($ch === false) return null;
    $headers = $fields === null ? ['Authorization: Bearer ' . $token] :
        ['Content-Type: application/x-www-form-urlencoded'];
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 15,
        CURLOPT_CONNECTTIMEOUT => 5, CURLOPT_FOLLOWLOCATION => false,
        CURLOPT_HTTPHEADER => $headers,
    ]);
    if ($fields !== null) {
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query($fields));
    }
    $body = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    if ($code !== 200 || !is_string($body) || strlen($body) > 65536) return null;
    $decoded = json_decode($body, true);
    return is_array($decoded) ? $decoded : null;
}

$config = auth0_config();
$redirect = $config['redirect_uri'];
$site = 'https://' . parse_url($redirect, PHP_URL_HOST) . '/';
$provider = 'https://' . $config['domain'];

if (isset($_GET['logout'])) {
    // Keep an external page from silently signing out an operator via a link.
    if (!in_array($_SERVER['HTTP_SEC_FETCH_SITE'] ?? 'same-origin', ['same-origin', 'none'], true))
        auth0_fail('Sign-out must start from this site.');
    $_SESSION = [];
    if (ini_get('session.use_cookies')) {
        $cookie = session_get_cookie_params();
        setcookie(session_name(), '', ['expires' => time() - 3600, 'path' => $cookie['path'],
            'domain' => $cookie['domain'], 'secure' => $cookie['secure'],
            'httponly' => $cookie['httponly'], 'samesite' => 'Lax']);
    }
    session_destroy();
    header('Location: ' . $provider . '/v2/logout?' . http_build_query([
        'client_id' => $config['client_id'], 'returnTo' => $site]));
    exit;
}

if (!isset($_GET['code']) && !isset($_GET['error'])) {
    $state = bin2hex(random_bytes(32));
    $verifier = rtrim(strtr(base64_encode(random_bytes(32)), '+/', '-_'), '=');
    $_SESSION['auth0_pending'] = ['state' => $state, 'verifier' => $verifier, 'started' => time()];
    $challenge = rtrim(strtr(base64_encode(hash('sha256', $verifier, true)), '+/', '-_'), '=');
    header('Location: ' . $provider . '/authorize?' . http_build_query([
        'response_type' => 'code', 'client_id' => $config['client_id'],
        'redirect_uri' => $redirect, 'scope' => 'openid email profile',
        'state' => $state, 'code_challenge' => $challenge,
        'code_challenge_method' => 'S256', 'prompt' => 'login']));
    exit;
}

$pending = $_SESSION['auth0_pending'] ?? null;
unset($_SESSION['auth0_pending']); // A callback can be used only once.
$state = $_GET['state'] ?? null;
if (!is_array($pending) || !is_string($state) ||
    !hash_equals((string)($pending['state'] ?? ''), $state) ||
    (int)($pending['started'] ?? 0) < time() - 600 ||
    !is_string($pending['verifier'] ?? null))
    auth0_fail('Sign-in expired or was tampered with. Please try again.');
if (isset($_GET['error'])) auth0_fail('Auth0 did not complete the sign-in.');
$code = $_GET['code'] ?? null;
if (!is_string($code) || $code === '' || strlen($code) > 2048)
    auth0_fail('Auth0 did not return a valid sign-in code.');

$tokens = auth0_request($provider . '/oauth/token', [
    'grant_type' => 'authorization_code', 'client_id' => $config['client_id'],
    'client_secret' => $config['client_secret'], 'code' => $code,
    'code_verifier' => $pending['verifier'], 'redirect_uri' => $redirect]);
$accessToken = $tokens['access_token'] ?? null;
if (!is_string($accessToken) || $accessToken === '') auth0_fail('Auth0 did not accept the sign-in.');
$user = auth0_request($provider . '/userinfo', null, $accessToken);
$email = $user['email'] ?? null;
$subject = $user['sub'] ?? null;
if (!is_string($email) || !filter_var($email, FILTER_VALIDATE_EMAIL) ||
    ($user['email_verified'] ?? null) !== true || !is_string($subject) || $subject === '')
    auth0_fail('Your Auth0 account needs a verified email.');
$email = strtolower($email);
$allowed = array_map('strtolower', array_filter($config['allowed'], 'is_string'));
if (!in_array($email, $allowed, true)) auth0_fail('This account is not an operator for this desk.');
$subjects = $config['allowed_subjects'] ?? [];
if (!is_array($subjects) || ($subjects !== [] && !in_array($subject, $subjects, true)))
    auth0_fail('This account is not an operator for this desk.');
$liveAllowed = $config['live_allowed'] ?? [];
if (!is_array($liveAllowed)) $liveAllowed = [];

session_regenerate_id(true);
$_SESSION = ['operator' => true, 'auth' => 'auth0', 'email' => $email,
    'name' => is_string($user['name'] ?? null) ? $user['name'] : $email,
    'live_ok' => in_array($email, array_map('strtolower', array_filter($liveAllowed, 'is_string')), true),
    'authenticated_at' => time()];
header('Location: ' . $site);
exit;
