"""Outbound-polling Hostinger controller. Run locally beside the existing pipeline.

The hosted site can request only uploaded audio or an approved HTTPS URL.
Every desk launch here fixes --mode dry; browser requests never supply CLI flags.
"""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import psutil
import requests

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'control_worker_processes.json'
LOCK_PATH = ROOT / 'control_worker.lock'
ALLOWED_SOURCE_HOSTS = {'www.federalreserve.gov'}
POLL_SECONDS = 2
MAX_SESSION_SECONDS = 2 * 60 * 60


def validate_url(value: str, allowed_hosts=ALLOWED_SOURCE_HOSTS) -> str:
    if len(value) > 2048 or any(ord(c) < 33 or ord(c) == 127 for c in value):
        raise ValueError('Invalid source URL')
    url = urlparse(value)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.port:
        raise ValueError('Source must be an approved HTTPS URL')
    if url.hostname.lower().rstrip('.') not in allowed_hosts:
        raise ValueError('Source host is not approved locally')
    for result in socket.getaddrinfo(url.hostname, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise ValueError('Source host did not resolve to a public address')
    return value


def _read_secret(name: str, env: str) -> str:
    value = os.environ.get(env)
    if value:
        return value.strip()
    path = ROOT / name
    return path.read_text(encoding='utf-8').strip() if path.is_file() else ''


class SingleWorkerLock:
    def __enter__(self):
        self.handle = open(LOCK_PATH, 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.handle.seek(0)
                if self.handle.read(1) == b'':
                    self.handle.seek(0)
                    self.handle.write(b'0')
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError('A control worker is already running') from exc
        return self

    def __exit__(self, *_):
        self.handle.close()


def _owned_processes() -> list[psutil.Process]:
    try:
        entries = json.loads(MANIFEST.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    found = []
    for item in entries:
        try:
            p = psutil.Process(int(item['pid']))
            if abs(p.create_time() - float(item['created'])) < 2:
                found.append(p)
        except (psutil.Error, KeyError, ValueError, TypeError):
            pass
    return found


def _kill_owned():
    processes = _owned_processes()
    for parent in processes:
        try:
            children = parent.children(recursive=True)
            for p in children:
                p.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs(children + [parent], timeout=4)
            for p in alive:
                p.kill()
        except psutil.Error:
            pass
    MANIFEST.unlink(missing_ok=True)


class Worker:
    def __init__(self, url: str, token: str):
        self.base = url.rstrip('/')
        self.http = requests.Session()
        self.http.headers.update({'X-Upload-Token': token})
        self.processes = {}
        self.tempdir = None
        self.phase = 'idle'
        self.source = ''
        self.message = ''
        self.last_command = None
        self.audio_finished_at = None
        self.started_at = None
        self.pending_ack = ''

    def request(self, method: str, path: str, **kwargs):
        r = self.http.request(method, self.base + path, timeout=15, **kwargs)
        r.raise_for_status()
        return r

    def status(self, ack=''):
        acknowledged = ack or self.pending_ack
        self.request('POST', '/control_worker.php', json={'phase': self.phase,
            'source': self.source, 'message': self.message, 'ack': acknowledged})
        if acknowledged == self.pending_ack:
            self.pending_ack = ''

    def _remember_process(self, name, process):
        self.processes[name] = process
        entries = []
        for p in self.processes.values():
            try:
                entries.append({'pid': p.pid, 'created': psutil.Process(p.pid).create_time()})
            except psutil.Error:
                pass
        MANIFEST.write_text(json.dumps(entries), encoding='utf-8')

    def _launch(self, name, args):
        log = open(ROOT / f'control_{name}.log', 'w', encoding='utf-8')
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            p = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=flags,
                                 start_new_session=(os.name != 'nt'))
            self._remember_process(name, p)
        finally:
            log.close()
        return p

    def _download(self, upload_id):
        if not isinstance(upload_id, str) or len(upload_id) > 40 or '/' in upload_id or '\\' in upload_id:
            raise ValueError('Invalid upload id')
        self.tempdir = tempfile.TemporaryDirectory(prefix='incredible_controls_')
        path = Path(self.tempdir.name) / upload_id
        with self.request('GET', '/control_worker.php', params={'download': upload_id}, stream=True) as r:
            size = 0
            with open(path, 'wb') as f:
                for chunk in r.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > 50 * 1024 * 1024:
                        raise ValueError('Upload exceeds local size limit')
                    f.write(chunk)
        if size == 0:
            raise ValueError('Empty upload')
        return str(path)

    def start(self, command):
        if self.phase not in ('idle', 'stopped', 'error'):
            self.status(command['id'])
            return
        self.stop()
        source_type = command.get('source_type')
        if source_type == 'upload':
            source_value = self._download(command.get('upload_id'))
            self.source = 'uploaded recording'
            source_flag = '--realtime-file'
        elif source_type == 'url':
            source_value = validate_url(command.get('source_url', ''))
            self.source = urlparse(source_value).hostname
            source_flag = '--url'
        else:
            raise ValueError('Invalid source type')
        latest = self.request('GET', '/control_worker.php').json().get('command')
        if not latest or latest.get('id') != command['id']:
            self.stop()
            return
        (ROOT / 'live_transcript.json').write_text(json.dumps({'source': self.source, 'segments': []}), encoding='utf-8')
        for name in ('live_state.json', 'live_recommendations.json', 'live_polymarket_recommendations.json'):
            (ROOT / name).unlink(missing_ok=True)
        self.phase, self.message = 'starting', 'Preparing the dry-run desk'
        desk = self._launch('desk', ['live.py', '--speaker', 'kevin_warsh', '--fast', '--mode', 'dry',
                                     '--qty', '2', '--venues', 'kalshi', 'polymarket',
                                     '--source-label', self.source])
        self.source_flag, self.source_value = source_flag, source_value
        self.started_at = time.monotonic()
        self.status(command['id'])

    def _ready(self):
        try:
            state = json.loads((ROOT / 'live_state.json').read_text(encoding='utf-8'))
            return state.get('status') == 'listening'
        except (OSError, ValueError):
            return False

    def _advance(self):
        if self.phase in ('starting', 'running') and self.started_at and time.monotonic() - self.started_at >= MAX_SESSION_SECONDS:
            self.stop()
            self.message = 'Session time limit reached'
            return
        if self.phase == 'starting':
            if self.processes['desk'].poll() is not None:
                raise RuntimeError('The dry-run desk exited during startup')
            if self._ready():
                self._launch('audio', ['speechtxt.py', self.source_flag, self.source_value])
                self._launch('publisher', ['publish.py'])
                self.phase, self.message = 'running', 'Dry-run session active'
            elif time.monotonic() - self.started_at > 60:
                raise RuntimeError('The dry-run desk did not become ready')
        elif self.phase == 'running':
            if self.processes['desk'].poll() is not None:
                raise RuntimeError('The dry-run desk stopped unexpectedly')
            if self.processes['publisher'].poll() is not None:
                raise RuntimeError('Website updates stopped unexpectedly')
            if self.processes['audio'].poll() is not None:
                if self.source_flag == '--realtime-file' and self.processes['audio'].returncode == 0:
                    self.audio_finished_at = self.audio_finished_at or time.monotonic()
                    if time.monotonic() - self.audio_finished_at > 12:
                        self.stop()
                else:
                    raise RuntimeError('The audio source stopped unexpectedly')

    def stop(self):
        self.phase = 'stopping'
        _kill_owned()
        self.processes.clear()
        if self.tempdir:
            self.tempdir.cleanup()
            self.tempdir = None
        self.audio_finished_at = None
        self.phase, self.message, self.source = 'stopped', 'Session stopped', ''

    def run_once(self):
        data = self.request('GET', '/control_worker.php').json()
        command = data.get('command')
        if command and command.get('id') != self.last_command:
            self.last_command = command['id']
            self.pending_ack = command['id']
            try:
                if command.get('action') == 'stop':
                    self.stop()
                    self.status(command['id'])
                elif command.get('action') == 'start':
                    if int(command.get('created_at', 0)) < time.time() - 120:
                        self.status(command['id'])
                    else:
                        self.start(command)
                else:
                    self.status(command['id'])
            except Exception as exc:
                self.stop()
                self.phase, self.message = 'error', type(exc).__name__ + ' during session setup'
                self.status(command['id'])
        try:
            self._advance()
        except Exception as exc:
            self.stop()
            self.phase, self.message = 'error', type(exc).__name__ + ' during session'
        self.status()


def main():
    parser = argparse.ArgumentParser(description='Local outbound-polling dry-run website worker')
    parser.add_argument('--url', default=os.environ.get('HOSTINGER_URL') or _read_secret('hostinger_url.txt', 'HOSTINGER_URL'))
    args = parser.parse_args()
    token = _read_secret('hostinger_token.txt', 'HOSTINGER_TOKEN')
    if not args.url or not token or urlparse(args.url).scheme != 'https':
        raise SystemExit('Configure an HTTPS Hostinger URL and the existing upload token locally')
    with SingleWorkerLock():
        _kill_owned()  # Recover only PIDs whose creation times match this worker's manifest.
        worker = Worker(args.url, token)
        try:
            while True:
                try:
                    worker.run_once()
                except requests.RequestException:
                    pass  # Network failure cannot create a new session; retry outbound.
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            pass
        finally:
            worker.stop()
            try:
                worker.status()
            except requests.RequestException:
                pass


if __name__ == '__main__':
    main()
