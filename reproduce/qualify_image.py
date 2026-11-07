"""Inspect a pulled vLLM image and its installed graph-fix invariants."""
import hashlib
import json
from pathlib import Path
import re
import check_fix
import runtime

FLAGS = ('--model', '--host', '--port', '--language-model-only', '--tokenizer-mode', '--tensor-parallel-size',
         '--data-parallel-size', '--data-parallel-size-local', '--enable-expert-parallel',
         '--kv-cache-dtype', '--engram-config', '--attention-config', '--kernel-config', '--max-model-len',
         '--max-num-seqs', '--max-num-batched-tokens', '--enable-chunked-prefill', '--gpu-memory-utilization', '--enable-prefix-caching',
         '--reasoning-parser', '--tool-call-parser', '--enable-auto-tool-choice', '--compilation-config')
CODE = '''import importlib.util,json,runpy,torch,vllm
from pathlib import Path
p=Path(importlib.util.find_spec("vllm").submodule_search_locations[0]) / "models/deepseek_v41"
check=runpy.run_path("/qualification/check_fix.py")["check"]
print(json.dumps({"invariants":check(p),"vllm_version":getattr(vllm,"__version__",None),"torch_version":torch.__version__,"cuda_version":torch.version.cuda,"nccl_version":torch.cuda.nccl.version()}))
'''

def qualify(image, revision, run_id, out, session, invoke=runtime.command):
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}', image): raise ValueError('digest-pinned image required')
    if not re.fullmatch(r'[0-9a-f]{40}', revision): raise ValueError('full source revision required')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', run_id): raise ValueError('invalid run ID')
    inspect = json.loads(invoke(['sudo','-n','docker','image','inspect',image]))
    if len(inspect) != 1: raise ValueError('image inspect returned unexpected count')
    img = inspect[0]
    if image not in (img.get('RepoDigests') or []): raise ValueError('pulled image digest mismatch')
    image_config=img.get('Config',{})
    labels = image_config.get('Labels') or {}
    if labels.get('org.opencontainers.image.revision') != revision: raise ValueError('source revision mismatch')
    if image_config.get('Entrypoint') != ['vllm','serve']:
        raise ValueError('image entrypoint differs from launcher assumption')
    fix = Path(__file__).with_name('check_fix.py').resolve()
    if session.run_id != run_id or not session.preflight_done:
        raise runtime.RuntimeErrorClosed('qualified supervisor and node preflight required')
    argv = ['docker','run','-d','--network','none','--gpus','all','--restart','no',
            '--label','campaign=dsv41-1017','--label','run_id='+run_id,
            '--mount',f'type=bind,src={fix},dst=/qualification/check_fix.py,readonly',
            '--entrypoint','sleep',image,'infinity']
    cid = session.create_start(argv)
    row = runtime.inspect_owned(cid, run_id, invoke)
    if row.get('Image') != img.get('Id') or not row.get('State', {}).get('Running'):
        raise runtime.RuntimeErrorClosed('qualification container identity mismatch')
    session.ensure_healthy()
    prefix = ['sudo','-n','docker','exec',cid]
    installed = invoke(prefix + ['python3','-c',CODE], timeout=120)
    session.ensure_healthy()
    details = json.loads(installed.strip().splitlines()[-1])
    expected = {name for names in check_fix.REQUIRED.values() for name in names}
    invariants = details.get('invariants', {})
    if set(invariants) != expected or set(invariants.values()) != {'invariants-present'} or not details.get('cuda_version') or not details.get('vllm_version') or not details.get('torch_version') or not details.get('nccl_version'):
        raise ValueError('installed fix or dependency evidence missing')
    help_text = invoke(prefix + ['vllm','serve','--help=all'], timeout=120)
    session.ensure_healthy()
    missing = [flag for flag in FLAGS if flag not in help_text]
    if missing: raise ValueError('image help missing flags: ' + ', '.join(missing))
    drivers=[x.strip() for x in invoke(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],timeout=20).splitlines()]
    if len(drivers)!=8 or len(set(drivers))!=1 or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+',drivers[0]):
        raise ValueError('GPU driver version evidence incomplete')
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    runtime.atomic_bytes(root/'serve-help.txt',help_text.encode())
    record = {'image':image, 'image_id':img.get('Id'), 'source_revision':revision,
              'installed':details, 'driver_versions':drivers, 'entrypoint':image_config['Entrypoint'], 'help_sha256':hashlib.sha256(help_text.encode()).hexdigest(),
              'fix_checker_sha256':hashlib.sha256(fix.read_bytes()).hexdigest(),
              'scope':'source invariants and flag presence only; GPU serving unqualified'}
    runtime.atomic_json(root/'qualification.json', record)
    return record
