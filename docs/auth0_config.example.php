<?php
// Example only. Save a filled-in copy outside public_html as
// incredible_trades_private/auth0_config.php. Never commit real values.
return [
    'domain' => 'YOUR_TENANT.us.auth0.com',
    'client_id' => 'AUTH0_REGULAR_WEB_APP_CLIENT_ID',
    'client_secret' => 'AUTH0_REGULAR_WEB_APP_CLIENT_SECRET',
    'redirect_uri' => 'https://yellow-boar-756344.hostingersite.com/auth0.php',
    'allowed' => ['verified-operator@example.com'],
    // Optional: restrict by immutable Auth0 user ID as well as verified email.
    'allowed_subjects' => [],
    // Leave empty unless this operator may enable real-money sessions.
    'live_allowed' => [],
];
