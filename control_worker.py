"""Outbound-polling Hostinger controller. Run locally beside the existing pipeline.

    python control_worker.py

The hosted MarketPulse page (web/index.html + web/control.php) queues commands;
this worker polls for them over HTTPS and runs everything on this laptop:

  * Start/Stop a live session from the laptop microphone, an uploaded recording,
    an approved HTTPS stream, or a replay of a past press conference.
  * Session options: speaker, trading mode (dry run or Kalshi demo), contracts
    per order, venues, and the live Kalshi + Polymarket recommenders.
  * Jobs: Ask the desk (Backboard), Analyze a statement, search memories.
  * Publishes data/archive.json (replays + order/signal history) every minute.

Real-money (live) mode requires both typed confirmation on the website and
MARKETPULSE_ALLOW_LIVE=1 in the local worker environment.
"""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import psutil
import requests

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'control_worker_processes.json'
LOCK_PATH = ROOT / 'control_worker.lock'
ALLOWED_SOURCE_HOSTS = {'www.federalreserve.gov', 'www.youtube.com', 'youtube.com', 'm.youtube.com', 'youtu.be'}
WEB_MODES = ('dry', 'demo', 'live')
SPEEDS = (1, 2, 3, 4, 5, 10, 20)
POLL_SECONDS = 2
ARCHIVE_SECONDS = 60
MAX_SESSION_SECONDS = 2 * 60 * 60
REPLAY_SPEED = 10


def replay_catalog() -> list:
    try:
        return json.loads((ROOT / 'transcripts' / 'catalog.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []


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


def known_speakers() -> set:
    try:
        return set(json.loads((ROOT / 'stance_baselines.json').read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return {'kevin_warsh'}


def clean_options(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    mode = raw.get('mode') if raw.get('mode') in WEB_MODES else 'dry'
    if mode == 'live':
        if raw.get('confirm_live') != 'LIVE':
            raise ValueError('LIVE mode requires typed LIVE confirmation')
        if os.environ.get('MARKETPULSE_ALLOW_LIVE') != '1':
            raise ValueError('LIVE mode requires local MARKETPULSE_ALLOW_LIVE=1 opt-in')
    speaker = raw.get('speaker') if raw.get('speaker') in known_speakers() else 'kevin_warsh'
    try:
        qty = max(1, min(5, int(raw.get('qty', 2))))
    except (TypeError, ValueError):
        qty = 2
    venues = [v for v in ('kalshi', 'polymarket') if v in (raw.get('venues') or [])] or ['kalshi', 'polymarket']
    try:
        speed = float(raw.get('speed', 0))
    except (TypeError, ValueError):
        speed = 0
    return {'mode': mode, 'speaker': speaker, 'qty': qty, 'venues': venues,
            'speed': speed if speed in SPEEDS else 0, 'recommenders': True}


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


def _owned_processes() -> list:
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
    for parent in _owned_processes():
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


# ---------------------------------------------------------------- jobs --- #

def _trim_trade(t: dict) -> dict:
    import trading_common as tc
    return {k: t.get(k) for k in ('time', 'venue', 'market', 'title', 'side', 'qty', 'yes_limit',
                                   'max_cost', 'reason', 'trigger')} | {
        'status': (f"skipped: {t['error']}" if t.get('error') else tc.trade_status(t)),
        'skipped': bool(t.get('error') or t.get('blocked'))}


def _trim_record(r: dict) -> dict:
    return {
        'statement': (r.get('statement') or '')[:600], 'stance': r.get('stance'), 'z': r.get('z'),
        'baseline_mean': r.get('baseline_mean'), 'direction': r.get('direction'), 'summary': r.get('summary'),
        'solana': (r.get('solana') or {}).get('explorer'),
        'matches': [{'venue': m.get('venue'), 'market': m.get('market'), 'title': m.get('title'),
                     'direction': m.get('direction'), 'relevance': m.get('relevance'),
                     'bid': (m.get('quote') or {}).get('best_bid'), 'ask': (m.get('quote') or {}).get('best_ask'),
                     'reason': m.get('reason')} for m in (r.get('matches') or [])[:10]],
        'trades': [_trim_trade(t) for t in (r.get('trades') or [])[:10]],
    }


def run_job(job: dict):
    kind, text = job.get('kind'), (job.get('text') or '').strip()
    raw_options = job.get('options')
    if kind in ('order_snapshot', 'order_preview', 'order_submit'):
        import manual_orders
        if kind == 'order_snapshot':
            return manual_orders.snapshot()
        if kind == 'order_preview':
            return manual_orders.preview(raw_options or {}, job.get('owner', ''))
        return manual_orders.submit((raw_options or {}).get('preview_id'), job.get('owner', ''))
    if kind in ('ask', 'analyze', 'memories'):
        raw_options = {**(raw_options if isinstance(raw_options, dict) else {}), 'mode': 'dry'}
    opts = clean_options(raw_options)
    if kind == 'ask':
        import backboard_client as bb
        if not bb.enabled():
            raise RuntimeError('Backboard is not configured on the laptop')
        r = bb.ask(text)
        return {'answer': r.get('answer') or '(no answer)', 'model': r.get('model'),
                'memories': len(r.get('memories') or [])}
    if kind == 'memories':
        import backboard_client as bb
        if not bb.enabled():
            raise RuntimeError('Backboard is not configured on the laptop')
        mems = bb.search_memories(text, limit=20) if text else bb.list_memories()[:30]
        return {'memories': [{'content': (m.get('content') or '')[:600],
                              'tag': (m.get('metadata') or {}).get('direction') or (m.get('metadata') or {}).get('kind', ''),
                              'created_at': str(m.get('created_at', ''))[:16]} for m in mems[:30]]}
    if kind == 'analyze':
        import pipeline
        import stance_scorer
        result = stance_scorer.score_statement(opts['speaker'], text)
        record = pipeline.act_on(result, opts['speaker'], opts['venues'], 'dry', opts['qty'],
                                 bool((job.get('options') or {}).get('stance_only')), source='website')
        pipeline.save(f"website_{int(time.time())}", [record])
        return {'record': _trim_record(record), 'mode': 'dry'}
    raise ValueError('Unknown job')


def build_archive() -> dict:
    import trading_common as tc
    replays = []
    order = {f"replay_{i['id']}": n for n, i in enumerate(replay_catalog())}
    labels = {f"replay_{i['id']}": i['label'] for i in replay_catalog()}
    for path in sorted((ROOT / 'results').glob('replay_*.json'), key=lambda p: order.get(p.stem, -1))[:30]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        replays.append({'name': path.stem, 'label': labels.get(path.stem, path.stem), 'created': data.get('created'),
                        'records': [_trim_record(r) for r in data.get('records', [])]})
    trades = []
    try:
        with open(tc.TRADE_LOG_PATH, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('reason') == 'manual order':
                    continue  # authenticated account activity belongs only in Orders
                trades.append({'time': r.get('time'), 'venue': r.get('venue'), 'market': r.get('market'),
                               'side': r.get('side'), 'qty': r.get('qty'), 'yes_limit': r.get('yes_limit'),
                               'max_cost': r.get('max_cost'), 'status': tc.trade_status(r),
                               'reason': r.get('reason')})
    except OSError:
        pass
    positions = {}
    for t in trades:   # net contracts per market from our own orders (demo/real only)
        st = str(t.get('status') or '')
        if not st.startswith('SENT'):
            continue
        key = (t.get('venue'), t.get('market'), 'demo' if 'demo' in st else 'real')
        pos = positions.setdefault(key, {'venue': key[0], 'market': key[1], 'account': key[2],
                                         'yes': 0, 'no': 0, 'cost': 0.0, 'last': t.get('time')})
        pos[t.get('side') or 'yes'] = pos.get(t.get('side') or 'yes', 0) + int(t.get('qty') or 0)
        pos['cost'] += float(t.get('max_cost') or 0)
        pos['last'] = t.get('time')
    signals = []
    try:
        import tiger_store
        if tiger_store.enabled():
            rows = tiger_store.query('SELECT time, speaker, direction, z, summary, solana_sig, source '
                                     'FROM signals ORDER BY time DESC LIMIT 150')
            signals = [{k: (str(v) if k == 'time' else v) for k, v in dict(r).items()} for r in rows]
    except Exception:
        signals = []
    replay_dates = [{'id': i['id'], 'label': i['label'], 'speaker': i['speaker']} for i in replay_catalog()]
    macro, social = {}, {}
    try:
        import fred_client
        macro = fred_client.latest()
    except Exception:
        pass
    try:
        import social_sentiment
        social = social_sentiment.latest()
    except Exception:
        pass
    return {'updated_at': datetime.now(timezone.utc).isoformat(), 'replays': replays,
            'macro': macro, 'social': social,
            'positions': sorted(positions.values(), key=lambda p: str(p['last']), reverse=True)[:40],
            'trades': trades[-200:][::-1], 'signals': signals, 'replay_dates': replay_dates,
            'speakers': sorted(known_speakers())}


# -------------------------------------------------------------- worker --- #

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
        self.kind = None
        self.opts = clean_options({})
        self.last_archive = -1e9
        self.archive_lock = threading.Lock()

    def request(self, method: str, path: str, **kwargs):
        r = self.http.request(method, self.base + path, timeout=kwargs.pop('timeout', 15), **kwargs)
        r.raise_for_status()
        return r

    def status(self, ack=''):
        acknowledged = ack or self.pending_ack
        self.request('POST', '/control_worker.php', json={'phase': self.phase,
            'source': self.source, 'message': self.message, 'ack': acknowledged})
        if acknowledged == self.pending_ack:
            self.pending_ack = ''

    # -- archive + jobs ------------------------------------------------- #
    def publish_archive(self):
        if not self.archive_lock.acquire(blocking=False):
            return
        try:
            body = json.dumps(build_archive(), default=str)
            self.request('POST', '/update.php', params={'name': 'archive'}, data=body,
                         headers={'Content-Type': 'application/json'}, timeout=30)
            self.last_archive = time.monotonic()
        except Exception as exc:
            print('archive upload failed:', exc)
        finally:
            self.archive_lock.release()

    def _job_thread(self, job):
        try:
            payload = {'job_id': job['id'], 'job_status': 'done', 'job_result': run_job(job)}
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'[:300]
            if str(job.get('kind', '')).startswith('order_'):
                safe = ('Local MARKETPULSE_ALLOW_LIVE=1', 'Order outcome is unknown',
                        'Sell quantity exceeds', 'Preview expired', 'Invalid preview',
                        'Market is not open', 'Limit is more than', 'daily cap:',
                        'STOP_TRADING', 'No YES', 'Missing Kalshi keys',
                        'Missing Polymarket keys')
                message = str(exc)[:200] if str(exc).startswith(safe) else 'Order service could not verify the request; check the laptop and exchange account'
            payload = {'job_id': job['id'], 'job_status': 'error',
                       'job_error': message}
        body = json.dumps(payload, default=str)
        if len(body) > 390000:
            body = json.dumps({'job_id': job['id'], 'job_status': 'error', 'job_error': 'Result too large'})
        for _ in range(3):
            try:
                self.request('POST', '/control_worker.php', data=body,
                             headers={'Content-Type': 'application/json'}, timeout=30)
                break
            except requests.RequestException:
                time.sleep(2)
        if job.get('kind') == 'analyze':
            self.publish_archive()

    # -- sessions ------------------------------------------------------- #
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
        self.tempdir = tempfile.TemporaryDirectory(prefix='marketpulse_controls_')
        path = Path(self.tempdir.name) / upload_id
        with self.request('GET', '/control_worker.php', params={'download': upload_id}, stream=True, timeout=120) as r:
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
        self.opts = clean_options(command.get('options'))
        kind = command.get('source_type')
        self.source_flag = self.source_value = None
        if kind == 'upload':
            self.source_value = self._download(command.get('upload_id'))
            self.source, self.source_flag = 'uploaded recording', '--realtime-file'
        elif kind == 'url':
            self.source_value = validate_url(command.get('source_url', ''))
            self.source, self.source_flag = urlparse(self.source_value).hostname, '--url'
        elif kind == 'mic':
            self.source = 'laptop microphone'
        elif kind == 'replay':
            rid = str(command.get('replay_date') or '')
            item = next((i for i in replay_catalog() if i.get('id') == rid), None)
            if not item or not (ROOT / 'transcripts' / f'{rid}.json').is_file():
                raise ValueError('Unknown replay')
            self.opts['speaker'] = item['speaker'] if item['speaker'] in known_speakers() else self.opts['speaker']
            self.source_value = rid
            self.replay_speed = self.opts['speed'] or REPLAY_SPEED
            self.source = f"replay {item.get('date', rid)} x{self.replay_speed:g}"
        else:
            raise ValueError('Invalid source type')
        self.kind = kind
        latest = self.request('GET', '/control_worker.php').json().get('command')
        if not latest or latest.get('id') != command['id']:
            self.stop()
            return
        (ROOT / 'live_transcript.json').write_text(json.dumps({'source': self.source, 'segments': []}), encoding='utf-8')
        for name in ('live_state.json', 'live_partial.json', 'live_recommendations.json',
                     'live_polymarket_recommendations.json'):
            (ROOT / name).unlink(missing_ok=True)
        o = self.opts
        desk = ['live.py', '--speaker', o['speaker'], '--fast', '--mode', o['mode'],
                '--qty', str(o['qty']), '--venues', *o['venues'], '--source-label', self.source]
        if kind == 'replay':
            desk += ['--simulate', self.source_value, '--speed', str(self.replay_speed)]
        elif kind == 'mic':
            desk += ['--source', 'mic', '--surprises-only']
        self.phase, self.message = 'starting', f"Loading markets ({o['mode']} mode)"
        self._launch('desk', desk)
        self._launch('publisher', ['publish.py'])
        if o['recommenders']:
            for name, script in (('kalshi_recs', 'run_kalshi_ticker2.py'), ('poly_recs', 'run_polymarket.py')):
                self._launch(name, [script, '--watch', '--speaker', o['speaker']])
        self.started_at = time.monotonic()
        self.status(command['id'])

    def _ready(self):
        try:
            state = json.loads((ROOT / 'live_state.json').read_text(encoding='utf-8'))
            return state.get('status') in ('listening', 'finished')
        except (OSError, ValueError):
            return False

    def _advance(self):
        if self.phase in ('starting', 'running') and self.started_at and time.monotonic() - self.started_at >= MAX_SESSION_SECONDS:
            self.stop()
            self.message = 'Session time limit reached'
            return
        desk = self.processes.get('desk')
        if self.phase == 'starting':
            if desk.poll() is not None and not (self.kind == 'replay' and desk.returncode == 0):
                raise RuntimeError('The desk exited during startup (see control_desk.log)')
            if self._ready():
                if self.kind == 'mic':
                    self._launch('audio', ['mic.py'])
                elif self.kind in ('upload', 'url'):
                    audio = ['speechtxt.py', self.source_flag, self.source_value]
                    if self.opts['speed'] and self.opts['speed'] > 1:
                        audio += ['--speed', f"{self.opts['speed']:g}"]
                    self._launch('audio', audio)
                self.phase = 'running'
                self.message = f"{self.opts['mode'].upper()} session active · {self.opts['speaker'].replace('_', ' ')}"
            elif time.monotonic() - self.started_at > 90:
                raise RuntimeError('The desk did not become ready')
        elif self.phase == 'running':
            if desk.poll() is not None:
                if self.kind == 'replay' and desk.returncode == 0:
                    self.audio_finished_at = self.audio_finished_at or time.monotonic()
                    self.message = 'Replay finished'
                    if time.monotonic() - self.audio_finished_at > 6:
                        self.stop()
                        self.message = 'Replay finished'
                        self.publish_archive()
                    return
                raise RuntimeError('The desk stopped unexpectedly (see control_desk.log)')
            if self.processes['publisher'].poll() is not None:
                self._launch('publisher', ['publish.py'])      # website updates are not critical; restart
            audio = self.processes.get('audio')
            if audio is not None and audio.poll() is not None:
                if self.kind == 'upload' and audio.returncode == 0:
                    self.audio_finished_at = self.audio_finished_at or time.monotonic()
                    if time.monotonic() - self.audio_finished_at > 12:
                        self.stop()
                        self.message = 'Recording finished'
                else:
                    raise RuntimeError('The audio source stopped (see control_audio.log)')

    def stop(self):
        was_running = self.phase in ('starting', 'running')
        self.phase = 'stopping'
        _kill_owned()
        self.processes.clear()
        if was_running:
            try:   # the site should say the session ended, not keep showing "Listening"
                path = ROOT / 'live_state.json'
                st = json.loads(path.read_text(encoding='utf-8'))
                st['status'] = 'stopped'
                path.write_text(json.dumps(st, default=str), encoding='utf-8')
                subprocess.run([sys.executable, 'publish.py', '--once'], cwd=ROOT, timeout=20,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        if was_running:
            state_path = ROOT / 'live_state.json'
            try:
                state = json.loads(state_path.read_text(encoding='utf-8'))
                state['status'] = 'stopped' if state.get('status') != 'finished' else 'finished'
                state['updated_at'] = datetime.now(timezone.utc).isoformat()
                replacement = state_path.with_name(state_path.name + '.tmp')
                replacement.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
                replacement.replace(state_path)
                import publish
                self.request('POST', '/update.php', data=json.dumps(publish.build_payload()),
                             headers={'Content-Type': 'application/json'}, timeout=10)
            except Exception as exc:
                print('final live state upload failed:', exc)
        if self.tempdir:
            self.tempdir.cleanup()
            self.tempdir = None
        self.audio_finished_at = None
        self.phase, self.message, self.source = 'stopped', 'Session stopped', ''

    def run_once(self):
        data = self.request('GET', '/control_worker.php').json()
        for job in data.get('jobs') or []:
            threading.Thread(target=self._job_thread, args=(job,), daemon=True).start()
        command = data.get('command')
        if command and command.get('id') != self.last_command:
            self.last_command = command['id']
            self.pending_ack = command['id']
            try:
                if command.get('action') == 'stop':
                    self.stop()
                    self.status(command['id'])
                    threading.Thread(target=self.publish_archive, daemon=True).start()
                elif command.get('action') == 'start':
                    if int(command.get('created_at', 0)) < time.time() - 120:
                        self.status(command['id'])
                    else:
                        self.start(command)
                else:
                    self.status(command['id'])
            except Exception as exc:
                self.stop()
                self.phase, self.message = 'error', f'{type(exc).__name__}: {exc}'[:150]
                self.status(command['id'])
        try:
            self._advance()
        except Exception as exc:
            self.stop()
            self.phase, self.message = 'error', f'{exc}'[:150]
        self.status()


def main():
    parser = argparse.ArgumentParser(description='Local worker for the hosted MarketPulse page')
    parser.add_argument('--url', default=os.environ.get('HOSTINGER_URL') or _read_secret('hostinger_url.txt', 'HOSTINGER_URL'))
    args = parser.parse_args()
    token = _read_secret('hostinger_token.txt', 'HOSTINGER_TOKEN')
    if not args.url or not token or urlparse(args.url).scheme != 'https':
        raise SystemExit('Configure an HTTPS Hostinger URL and the existing upload token locally')
    with SingleWorkerLock():
        _kill_owned()
        worker = Worker(args.url, token)
        print(f'MarketPulse worker polling {args.url} (Ctrl+C to stop)')
        try:
            while True:
                if time.monotonic() - worker.last_archive > ARCHIVE_SECONDS:
                    worker.last_archive = time.monotonic()   # archive uploads even if controls are not configured yet
                    threading.Thread(target=worker.publish_archive, daemon=True).start()
                try:
                    worker.run_once()
                except requests.RequestException as exc:
                    print('network:', exc)
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
