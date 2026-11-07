"""Download one pinned public file after the parent records this process."""
import os
from pathlib import Path
import sys
import time

def main(root, target, revision, filename, gate_name):
    root = Path(root)
    deadline = time.monotonic()+15
    while not (root/gate_name).exists():
        if (root/'stage-watchdog-failure.json').exists() or time.monotonic() >= deadline: return 1
        time.sleep(0.05)
    if (root/'stage-watchdog-failure.json').exists(): return 1
    # The pinned hub rejects regular HTTP downloads above 50 GB; two shards
    # exceed that limit. Keep the Xet cache disabled and suppress its logs,
    # which may contain signed request URLs.
    os.environ['HF_HUB_DISABLE_XET'] = '0'
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    os.environ['HF_XET_LOG_DEST'] = '/dev/null'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_DEBUG'] = '0'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HOME'] = str(Path(target)/'.hf-client')
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download('deepseek-ai/DeepSeek-V4.1-Flash-1017', filename=filename,
                        revision=revision, local_dir=target, token=False)
    except Exception as exc:
        # Never log exception text or request URLs: signed URLs can contain secrets.
        chain = []
        seen = set()
        while exc is not None and id(exc) not in seen and len(chain) < 8:
            seen.add(id(exc))
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            chain.append(type(exc).__name__ + (f':HTTP_{status}' if isinstance(status, int) else ''))
            exc = exc.__cause__ or exc.__context__
        print('DOWNLOAD_FAILED', ' -> '.join(chain), file=sys.stderr)
        return 1
    print('DOWNLOAD_COMPLETED')
    return 0

if __name__ == '__main__': raise SystemExit(main(*sys.argv[1:]))
