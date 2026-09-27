<?php
// Copy to control_config.php OUTSIDE the public web directory and fill in the
// password hash with PHP password_hash(..., PASSWORD_DEFAULT). Enter the
// password interactively; never put it in a shell command or this file.
// Create the storage directory outside the public web directory and make it
// writable by PHP. Do not commit the populated file.
return [
    'password_hash' => '',
    // When this file is placed beside public_html in Hostinger's File Manager:
    'storage_dir' => __DIR__ . '/incredible_trades_private',
    'source_hosts' => ['www.federalreserve.gov'],
    'max_upload_bytes' => 50 * 1024 * 1024,
];
