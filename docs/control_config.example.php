<?php
// Copy to incredible_trades_private/control_config.php OUTSIDE the public web directory and fill in the
// password hash with PHP password_hash(..., PASSWORD_DEFAULT). Enter the
// password interactively; never put it in a shell command or this file.
// Keep the private directory writable by PHP. Do not commit the populated file.
return [
    'password_hash' => '',
    // When this file is placed in incredible_trades_private:
    'storage_dir' => __DIR__,
    'source_hosts' => ['www.federalreserve.gov'],
    'max_upload_bytes' => 50 * 1024 * 1024,
];
