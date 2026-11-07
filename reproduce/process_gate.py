"""Keep an owned child inert until its identity is durably recorded."""
import os
from pathlib import Path
import sys
import time

def main(root, command):
    if not command: raise ValueError('empty child command')
    root = Path(root)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (root/'watchdog-failure.json').exists(): return 1
        if (root/'router-start.ok').exists():
            if (root/'watchdog-failure.json').exists(): return 1
            os.execvp(command[0], command)
        time.sleep(0.05)
    return 1

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
