"""Check an explicit publication file list or the exact staged Git blobs."""
import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess


ROOT_FILES = {'.gitignore', '.gitattributes', 'README.md'}
RESULTS = {'fixed-comparison', 'context-1m-output2048', 'tuning-adaptation'}
SUFFIXES = {'.py', '.sh', '.md', '.json', '.csv', '.svg', '.png', '.txt', '.lock'}
PRIVATE_PARTS = {'raw', 'evidence', 'job-monitor', 'campaign-control', 'failures', 'attempts', '__pycache__'}
SECRET_PATTERNS = [
    re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    re.compile(rb'AK' + rb'IA[A-Z0-9]{16}'),
    re.compile(rb'gh[pousr]_' + rb'[A-Za-z0-9]{20,}'),
    re.compile(rb'github_' + rb'pat_[A-Za-z0-9_]{20,}'),
    re.compile(rb'xox[baprs]-' + rb'[A-Za-z0-9-]{10,}'),
    re.compile(rb'(?i)(?:[?&](?:x-amz-signature|access_token|sig)=)[A-Za-z0-9%+/=_-]{12,}'),
    re.compile(rb'(?i)(?:"(?:secret_key|access_key|api_key|access_token|refresh_token)"\s*:\s*")[^"\s]{12,}"'),
]
PRIVATE_PATTERNS = [
    re.compile(rb'/(?:Users|home)/' + rb'(?:stew|stewtong)(?:/|\b)'),
    re.compile(rb'(?:tenant|computeinstance|serviceaccount)-' + rb'[a-z0-9]{12,}'),
    re.compile(rb'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
]
CJK = re.compile('[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]')


def check(name, data):
    p = PurePosixPath(name)
    issues = []
    if p.is_absolute() or '..' in p.parts or set(p.parts) & PRIVATE_PARTS:
        issues.append('private or unsafe path')
    elif name not in ROOT_FILES:
        if len(p.parts) < 2 or p.parts[0] not in {'configs', 'docs', 'reproduce', 'tests', 'results'}:
            issues.append('path is outside publication surfaces')
        elif p.parts[0] == 'results' and (len(p.parts) < 3 or p.parts[1] not in RESULTS):
            issues.append('result scope is not approved for this package')
        elif p.suffix not in SUFFIXES and p.name not in {'SHA256SUMS', 'SOURCE_SHA256SUMS'}:
            issues.append('file type is not approved')
    if p.name.startswith('.env') or p.suffix in {'.key', '.pem', '.log', '.jsonl'}:
        issues.append('runtime or credential artifact')
    if p.suffix == '.png':
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            issues.append('malformed PNG')
        return issues
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return issues + ['non-UTF-8 publication text']
    if CJK.search(text):
        issues.append('non-English publication text')
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        issues.append('possible credential value (redacted)')
    for pattern in PRIVATE_PATTERNS:
        for match in pattern.finditer(data):
            value = match.group()
            if pattern is PRIVATE_PATTERNS[-1]:
                # Loopback, unspecified and documentation addresses are portable.
                if value in {b'0.0.0.0', b'127.0.0.1'} or value.startswith((b'192.0.2.', b'198.51.100.', b'203.0.113.')):
                    continue
                # Dependency version strings can have four numeric components.
                if any(int(x) > 255 for x in value.split(b'.')):
                    continue
            issues.append('possible private connection detail (redacted)')
            break
    return sorted(set(issues))


def staged_blobs():
    names = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z']).split(b'\0')
    for raw in names:
        if not raw:
            continue
        name = raw.decode('utf-8')
        metadata = subprocess.check_output(['git', 'ls-files', '--stage', '--', name]).decode()
        if not metadata.startswith('100'):
            raise ValueError('staged entry must be a regular file: ' + name)
        yield name, subprocess.check_output(['git', 'show', ':' + name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', required=True)
    parser.parse_args()
    checked = 0
    failures = []
    for name, data in staged_blobs():
        checked += 1
        for issue in check(name, data):
            failures.append({'path': name, 'issue': issue})
    print(json.dumps({'checked_files': checked, 'status': 'failed' if failures else 'passed', 'findings': failures}))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
