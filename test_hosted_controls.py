"""Offline checks; no website calls, subprocesses, or audio processing."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import control_worker
import publish


class HostedControlTests(unittest.TestCase):
    def test_only_approved_public_https_source(self):
        public = [(None, None, None, None, ('8.8.8.8', 443))]
        with patch.object(control_worker.socket, 'getaddrinfo', return_value=public):
            self.assertEqual(control_worker.validate_url('https://www.federalreserve.gov/live-broadcast.htm'),
                             'https://www.federalreserve.gov/live-broadcast.htm')
            for value in ('http://www.federalreserve.gov/live-broadcast.htm',
                          'https://www.federalreserve.gov.evil.test/a',
                          'https://user:pass@www.federalreserve.gov/a',
                          'https://www.federalreserve.gov:8443/a'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    control_worker.validate_url(value)
        private = [(None, None, None, None, ('127.0.0.1', 443))]
        with patch.object(control_worker.socket, 'getaddrinfo', return_value=private):
            with self.assertRaises(ValueError):
                control_worker.validate_url('https://www.federalreserve.gov/a')

    def test_worker_defaults_to_dry_when_mode_is_missing(self):
        worker = control_worker.Worker('https://example.test', 'existing-token')
        commands = []
        fake_process = Mock(pid=123, poll=Mock(return_value=None))
        with tempfile.TemporaryDirectory() as folder, patch.object(control_worker, 'ROOT', Path(folder)), \
             patch.object(worker, 'stop'), patch.object(worker, '_download', return_value='C:/private/audio.mp3'), \
             patch.object(worker, '_launch', side_effect=lambda name, args: commands.append((name, args)) or fake_process), \
             patch.object(worker, 'request', return_value=Mock(json=Mock(return_value={'command': {'id': 'one'}}))), \
             patch.object(worker, 'status'), patch.dict(os.environ, {'MARKETPULSE_ALLOW_LIVE': ''}):
            worker.start({'id': 'one', 'action': 'start', 'source_type': 'upload', 'upload_id': 'audio.mp3',
                          'options': {'speaker': 'attacker', 'qty': 999}})
        desk = commands[0][1]
        self.assertEqual(desk[desk.index('--mode') + 1], 'dry')
        self.assertEqual(desk[desk.index('--speaker') + 1], 'kevin_warsh')
        self.assertNotIn('live', desk)
        self.assertNotIn('999', desk)

    def test_worker_accepts_live_only_with_local_opt_in_and_confirmation(self):
        requested = {'mode': 'live', 'confirm_live': 'LIVE'}
        with patch.dict(os.environ, {'MARKETPULSE_ALLOW_LIVE': '1'}):
            self.assertEqual(control_worker.clean_options(requested)['mode'], 'live')
            with self.assertRaisesRegex(ValueError, 'typed LIVE confirmation'):
                control_worker.clean_options({'mode': 'live'})
        for value in ('', 'true', '0'):
            with self.subTest(local_opt_in=value), patch.dict(os.environ, {'MARKETPULSE_ALLOW_LIVE': value}):
                with self.assertRaisesRegex(ValueError, 'local MARKETPULSE_ALLOW_LIVE=1'):
                    control_worker.clean_options(requested)

    def test_worker_rejects_live_before_launch_without_local_opt_in(self):
        worker = control_worker.Worker('https://example.test', 'existing-token')
        with patch.dict(os.environ, {'MARKETPULSE_ALLOW_LIVE': ''}), \
             patch.object(worker, 'stop'), patch.object(worker, '_launch') as launch:
            with self.assertRaisesRegex(ValueError, 'local MARKETPULSE_ALLOW_LIVE=1'):
                worker.start({'id': 'one', 'source_type': 'mic',
                              'options': {'mode': 'live', 'confirm_live': 'LIVE'}})
        launch.assert_not_called()

    def test_analyze_job_is_always_dry_even_when_live_is_requested(self):
        import pipeline
        import stance_scorer
        with patch.dict(os.environ, {'MARKETPULSE_ALLOW_LIVE': ''}), \
             patch.object(stance_scorer, 'score_statement', return_value={'score': 1}), \
             patch.object(pipeline, 'act_on', return_value={'statement': 'test'}) as act_on, \
             patch.object(pipeline, 'save'):
            result = control_worker.run_job({'kind': 'analyze', 'text': 'test',
                'options': {'mode': 'live', 'confirm_live': 'LIVE'}})
        self.assertEqual(result['mode'], 'dry')
        self.assertEqual(act_on.call_args.args[3], 'dry')

    def test_publisher_excludes_untrusted_fields_and_non_speaker_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            path.write_text(json.dumps({'speaker': 'kevin_warsh', 'secret': 'do-not-publish',
                'chunks': [{'role': 'reporter', 'text': 'private question', 'summary': 'private'},
                           {'role': 'speaker', 'text': 'Public remarks'}],
                'watchlist': {'HAWKISH': [{'market': 'market-1', 'quote': {'best_bid': .2,
                    'api_key': 'nested-do-not-publish'}}]},
                'trades': [{'status': 'DRY RUN', 'api_key': 'do-not-publish'}]}), encoding='utf-8')
            with patch.object(publish, 'STATE_PATH', str(path)), patch.object(publish, 'RECOMMENDERS', {}):
                payload = publish.build_payload()
            body = json.dumps(payload)
            self.assertNotIn('do-not-publish', body)
            self.assertNotIn('nested-do-not-publish', body)
            self.assertNotIn('private question', body)
            self.assertEqual(payload['live']['chunks'][1]['text'], 'Public remarks')

    def test_superseded_start_does_not_launch(self):
        worker = control_worker.Worker('https://example.test', 'existing-token')
        with patch.object(worker, 'stop') as stop, patch.object(worker, '_download', return_value='local.mp3'), \
             patch.object(worker, 'request', return_value=Mock(json=Mock(return_value={'command': {'id': 'stop-id'}}))), \
             patch.object(worker, '_launch') as launch:
            worker.start({'id': 'start-id', 'source_type': 'upload', 'upload_id': 'audio.mp3'})
        self.assertEqual(stop.call_count, 2)
        launch.assert_not_called()

    def test_session_time_limit_stops_owned_processes(self):
        worker = control_worker.Worker('https://example.test', 'existing-token')
        worker.phase = 'running'
        worker.started_at = time.monotonic() - control_worker.MAX_SESSION_SECONDS - 1
        with patch.object(worker, 'stop') as stop:
            worker._advance()
        stop.assert_called_once()
        self.assertEqual(worker.message, 'Session time limit reached')


if __name__ == '__main__':
    unittest.main()
