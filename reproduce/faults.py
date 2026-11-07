"""Detect actionable GPU failures in bounded, recent owned-container logs."""
from concurrent.futures import ThreadPoolExecutor
import os
import re
import selectors
import subprocess
import time
import runtime

PATTERNS=(re.compile(r'\b(?:torch\.)?(?:cuda\.)?OutOfMemoryError\b'),
          re.compile(r'\bCUDA out of memory\b',re.I),
          re.compile(r'\bCUDA error:\s*(?:out of memory|an illegal memory access was encountered)',re.I),
          re.compile(r'\bCUDA_ERROR_(?:OUT_OF_MEMORY|ILLEGAL_ADDRESS)\b'))


def read_logs(argv, timeout=3, limit=16*1024*1024):
    proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    selector=selectors.DefaultSelector(); chunks=[]; total=0; deadline=time.monotonic()+timeout
    try:
        selector.register(proc.stdout,selectors.EVENT_READ)
        while True:
            left=deadline-time.monotonic()
            if left<=0: raise runtime.RuntimeErrorClosed('container log check timed out')
            if not selector.select(left): continue
            chunk=os.read(proc.stdout.fileno(),65536)
            if not chunk: break
            total+=len(chunk)
            if total>limit: raise runtime.RuntimeErrorClosed('container log check exceeded size limit')
            chunks.append(chunk)
        if proc.wait(timeout=max(.1,deadline-time.monotonic()))!=0:
            raise runtime.RuntimeErrorClosed('container log check failed')
        return b''.join(chunks).decode('utf-8','replace')
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait(timeout=5)
        selector.close();proc.stdout.close()


def check(container_ids, invoke=read_logs):
    if len(set(container_ids))!=len(container_ids) or any(not runtime.re_full_id(cid) for cid in container_ids):
        raise runtime.RuntimeErrorClosed('invalid container log identities')
    def one(cid):
        text=invoke(['sudo','-n','docker','logs','--since','10s',cid],timeout=3)
        if any(pattern.search(line) for line in text.splitlines() for pattern in PATTERNS):
            raise runtime.RuntimeErrorClosed('GPU OOM or illegal access in owned-container log')
    if container_ids:
        with ThreadPoolExecutor(max_workers=len(container_ids)) as pool:
            for future in [pool.submit(one,cid) for cid in container_ids]: future.result()
