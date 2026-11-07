"""Exercise the four-turn capture at the real loopback HTTP boundary."""
import difflib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import coding_capture
import coding_panel
import stream
import verify_coding_capture
import campaign
import coding_history_proof
from unittest.mock import patch as mocked


class CodingCaptureTests(unittest.TestCase):
    def test_four_turns_retain_raw_tools_and_semantic_failure_without_running_patch(self):
        task = coding_panel.tasks()[0]
        patch = ''.join(difflib.unified_diff(task['source'].splitlines(True),
                          'def solve(value):\n    return value\n'.splitlines(True), 'a/solution.py', 'b/solution.py'))
        requests = []; bad = [False]; channel = [None]
        panel_by_id = {row['id']: row for row in coding_panel.tasks()}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length']))); requests.append(body)
                active_task = panel_by_id[self.headers['X-Session-Id']]
                turn = int(self.headers['X-Request-Id'].rsplit('-', 1)[-1]); events = []
                if turn < 2:
                    for i, (name, args) in enumerate(coding_panel.expected_calls(active_task, turn)):
                        if bad[0]: args = dict(args, unexpected=True)
                        encoded = json.dumps(args); split = len(encoded) // 2
                        events.extend([{'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': i, 'id': f'call-{i}', 'type': 'function', 'function': {'name': name, 'arguments': encoded[:split]}}]}}]},
                                       {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': i, 'function': {'arguments': encoded[split:]}}]}}]}])
                    finish = 'tool_calls'
                else:
                    content = json.dumps({'marker': active_task['exact_identifier'], 'path': 'solution.py', 'function': 'solve'}) if turn == 2 else patch
                    events.append({'choices': [{'index': 0, 'delta': {'content': content}}]}); finish = 'stop'
                if channel[0] is not None:
                    events.insert(0, {'choices': [{'index': 0, 'delta': channel[0]}]})
                events.extend([{'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
                               {'choices': [], 'usage': {'prompt_tokens': 1024 + turn, 'completion_tokens': 80 + turn}}])
                payload = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode() + b'data: [DONE]\n\n'
                self.send_response(200); self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
            def log_message(self, *args): pass
        class Server(ThreadingHTTPServer):
            # Eight concurrent conversations can open together. Keep fixture
            # admission above that population without changing request limits.
            request_queue_size=64
        server = Server(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        class Session:
            deadline_ns = time.monotonic_ns() + 600_000_000_000
            def ensure_healthy(self): pass
            def abort(self, reason): raise AssertionError(reason)
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); url = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
                result = coding_capture.conversation(Session(), url, '/model', task, root / 'valid', 5, stream.Cancellation(), 'fixture')
                self.assertTrue(result['conversation_valid']); self.assertTrue(result['tool_valid'])
                self.assertEqual(result['completion_tokens'], 326)
                self.assertEqual(len(result['turns']), 4)
                self.assertEqual(len(requests[2]['messages']), 10)
                self.assertEqual(requests[2]['messages'][-2]['role'], 'system')
                self.assertEqual(sum(x['role'] == 'tool' for x in requests[2]['messages']), 3)
                self.assertTrue(all('min_tokens' not in r and 'ignore_eos' not in r for r in requests))
                self.assertEqual(result['executable_score'], 'pending_separate_isolated_scoring')
                self.assertEqual((root / 'valid/candidate.diff').read_text(), patch)
                verified = verify_coding_capture.conversation(root / 'valid')
                self.assertTrue(verified['conversation_valid'])
                self.assertEqual(verified['completion_tokens'], 326)
                class Tokenizer:
                    def __init__(self,offset=0):self.calls=[];self.offset=offset
                    def apply_chat_template(self,messages,**kwargs):
                        self.calls.append((messages,kwargs));return str(1023+len(self.calls)+self.offset)
                    def encode(self,text,add_special_tokens):
                        self.assert_false=add_special_tokens is False;return list(range(int(text)))
                tokenizer=Tokenizer();history=coding_history_proof.build(tokenizer,root/'valid',whole_panel=False)
                self.assertEqual(history['status'],'matched');self.assertEqual(history['requests'],4)
                self.assertTrue(tokenizer.assert_false)
                self.assertEqual([r['tools'] for r in history['rows']],[1,2,0,0])
                self.assertEqual(tokenizer.calls[2][0][-2]['role'],'system')
                self.assertTrue(all(k['thinking'] is False and k['reasoning_effort']==50 and k['tokenize'] is False for _,k in tokenizer.calls))
                mismatch=coding_history_proof.build(Tokenizer(offset=1),root/'valid',whole_panel=False)
                self.assertEqual(mismatch['status'],'mismatch');self.assertEqual({r['difference'] for r in mismatch['rows']},{-1})
                saved = root / 'valid/conversation.json'; original = saved.read_text()
                tampered = json.loads(original); tampered['completion_tokens'] += 1; saved.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError, 'arithmetic'): verify_coding_capture.conversation(root / 'valid')
                saved.write_text(original)
                request_path = root / 'valid/turn-2-request.json'; original_request = request_path.read_text()
                changed = json.loads(original_request); changed['body']['messages'][-2]['role'] = 'user'; request_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'history'): verify_coding_capture.conversation(root / 'valid')
                request_path.write_text(original_request)
                bad[0] = True
                failed = coding_capture.conversation(Session(), url, '/model', task, root / 'invalid', 5, stream.Cancellation(), 'fixture')
                self.assertFalse(failed['conversation_valid']); self.assertFalse(failed['tool_valid'])
                self.assertEqual(len(failed['turns']), 1)
                self.assertTrue((root / 'invalid/turn-0-raw.json').exists())
                self.assertFalse((root / 'invalid/candidate.diff').exists())
                self.assertFalse(verify_coding_capture.conversation(root / 'invalid')['conversation_valid'])
                bad[0] = False
                for index, delta in enumerate(({'role': 'user'}, {'reasoning_content': 'unexpected'}, {'reasoning': 'unexpected'})):
                    channel[0] = delta
                    dest = root / ('bad-channel-' + str(index))
                    with self.assertRaisesRegex(ValueError, 'thinking-off'):
                        coding_capture.conversation(Session(), url, '/model', task, dest, 5, stream.Cancellation(), 'fixture')
                    retained = json.loads((dest / 'turn-0-raw.json').read_text())
                    with self.assertRaisesRegex(ValueError, 'thinking-off'):
                        verify_coding_capture.response(retained)
                channel[0] = None
                try:
                    result = coding_capture.run(Session(), url, '/model', root / 'panel', 8, 5, 'fixture')
                except Exception:
                    failed=[]
                    for path in sorted((root/'panel').rglob('turn-*-raw.json')):
                        raw=json.loads(path.read_text())
                        if not raw.get('complete'):
                            failed.append({k:raw.get(k) for k in ('request_id','error','error_phase','http_status','wire_byte_count','wire_bytes_observed')})
                    print('SYNTHETIC_CODING_TRANSPORT_FAILURE '+json.dumps(failed),file=sys.stderr)
                    raise
                self.assertEqual(result['captured_tasks'], 100)
                self.assertEqual(result['conversation_valid'], 100)
                output = io.StringIO()
                with mocked.object(sys, 'argv', ['campaign.py', 'verify-coding-capture', '--run-dir', str(root / 'panel')]), redirect_stdout(output):
                    campaign.main()
                self.assertEqual(json.loads(output.getvalue())['conversation_valid'], 100)
                summary_path = root / 'panel/capture-summary.json'; summary = json.loads(summary_path.read_text()); summary['tasks'] = 99; summary_path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError, 'summary'): verify_coding_capture.verify(root / 'panel')
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main()
