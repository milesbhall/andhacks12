# Hostinger operator controls

The public dashboard remains `web/index.html`. `web/control.php` accepts only an authenticated operator's Start/Stop request. `control_worker.py` runs on the local machine, polls Hostinger outbound over HTTPS, and starts the existing transcription, dry-run desk, and publisher processes. PHP never launches a background process.

## Hostinger setup

1. Use PHP **8.1 or newer**. Upload `web/` as the site's public files. Keep `control_config.php` and its private storage directory **outside** the public web root. On this Hostinger account, File Manager already has `incredible_trades_private` beside `public_html`. Place `control_config.php` in that private directory and set `storage_dir` to `__DIR__`. The PHP process must be able to read and write that folder. If the config location differs from this default, set `CONTROL_CONFIG_PATH` to its absolute path in the server environment.
2. The one-time `web/setup.php` page lets the operator enter a password directly over HTTPS after a separate high-entropy setup key is uploaded to `incredible_trades_private/setup.key`. The page writes only a password hash to the private config and removes the setup key. Do not put the password or its hash in the public files or in Git. Alternatively, copy `control_config.example.php` to the private location and set `password_hash` to a hash generated with `password_hash(..., PASSWORD_DEFAULT)` on a trusted machine. The controls stay disabled until configured. No operator credential is included in this repository.
3. Set `storage_dir` to an absolute private directory. It holds queued uploads and `control_state.json`. Do not use a directory inside `public_html` or another public web directory.
4. Set `source_hosts` to exact trusted hostnames. The supplied example allows only `www.federalreserve.gov`. The local worker also has `ALLOWED_SOURCE_HOSTS` in `control_worker.py`; keep both lists in sync. Only HTTPS URLs without URL credentials or custom ports are accepted, and the worker rejects private IP resolutions. Review a new host and its redirects/media behavior before adding it. For arbitrary untrusted sources, upload a recording instead.
5. `web/.user.ini` requests `upload_max_filesize=50M`, `post_max_size=52M`, and `max_input_time=120`. Verify the effective values in Hostinger's PHP settings; hosting level limits can override `.user.ini`. The control endpoint independently caps each file at 50 MiB. Adjust the page's displayed limit and the server/Python caps together if a smaller hosting limit is required.
6. Confirm HTTPS is active. The operator session cookie is `Secure` on HTTPS, `HttpOnly`, and `SameSite=Strict`. Run `php -l` on each new PHP file in the deployment environment before enabling access if no local PHP binary is available.

Do not change or expose the existing upload token. The worker reads `hostinger_token.txt` or `HOSTINGER_TOKEN` locally and sends it only in the `X-Upload-Token` header. The browser never receives it. The operator password is a separate credential and must be created by the operator.

## Local worker

Install `requirements.txt`, verify that the existing local API keys and `hostinger_url.txt`/`hostinger_token.txt` are configured, then run `python control_worker.py`. Keep that process running on the local machine. It does not use Streamlit. Its requests are outbound; no inbound laptop port or Hostinger process manager is needed.

The worker is intentionally fixed to `--mode dry`, `kevin_warsh`, quantity 2, and the existing Kalshi/Polymarket market-data venues. Website fields cannot select live or demo execution. A local lock prevents two workers in the same checkout; the process manifest records PID and creation time for crash recovery. Do not run the separate Streamlit audio session or other writers of `live_state.json` at the same time.

Start requires a heartbeat from an online worker. A queued Start expires after 120 seconds, and its private upload is removed. Stop remains queued until the worker acknowledges it, with a three-hour expiry for abandoned commands. The local worker ends any session after two hours. It removes downloaded recording copies on Stop; PHP removes the queued upload after the worker reports stopped or error. Private uploads are also pruned after three hours. If the worker is offline, the page shows that status and disables Start.

Browser recording is not part of these controls. The Upload input accepts an existing recording; it is not a live microphone.

## Verification

Run `python -m unittest test_hosted_controls.py test_audio_sources.py` for offline source validation, fixed dry-mode launch arguments, publisher field filtering, and existing audio-source behavior. This test does not contact Hostinger, transcribe audio, publish a state, or place an order. Confirm PHP syntax and effective upload limits on the hosting PHP runtime before deployment.
