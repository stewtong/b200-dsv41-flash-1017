"""Resolve the adopted operational plan without rewriting scientific pins."""
import hashlib
import json
from pathlib import Path
import re

PRIOR = 'b04977194f7a6a9b9e0657f58daee73b3bf3cf96c238f30d577f522dc209f46e'
CURRENT = '6533a58988501c43d0504a7a63f297c21febb88b7be155f4762fcc9539cb010b'
AMENDMENT = '6e3713070ae7bdbfd88be1c5d212abcd7bed3b04780757029356f7d12152e515'
MATRIX = '6ef298ab721d91342a6640f048b4358ce291f7c6ff00068f5a26db88fe668413'
PROPOSAL = '3f52372bcef33661ac969347f62e77c0e8027563ed458e30caac7248ed64ec3c'
DECISION = 'decisions/execution-efficiency-20261104/'
FILES = {'benchmark-plan.md': CURRENT, 'benchmark-matrix.json': MATRIX,
         DECISION+'amendment.json': AMENDMENT,
         DECISION+'previous-benchmark-plan.md': PRIOR,
         DECISION+'source-proposal.md': PROPOSAL}


def sections(text):
    parts = re.split(r'(?m)^## ([0-9]+)\. ', text)
    return {int(parts[i]): parts[i+1] for i in range(1, len(parts), 2)}


def verify(root):
    root = Path(root).resolve(strict=True)
    values = {}
    for name, expected in FILES.items():
        path = root/name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('authority source must be a contained regular file')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('exact authority fingerprint differs: '+name)
        values[name] = data.decode('utf-8')
    amendment = json.loads(values[DECISION+'amendment.json'])
    if (amendment['schema'] != 'dsv41-operational-amendment/v1' or
            amendment['scope_id'] != 'context-concurrency-336-v1' or
            amendment['prior_plan_sha256'] != PRIOR or amendment['current_plan_sha256'] != CURRENT or
            amendment['unchanged_matrix_sha256'] != MATRIX or amendment['ownership_transferred'] is not False):
        raise ValueError('operational amendment linkage differs')
    old, new = sections(values[DECISION+'previous-benchmark-plan.md']), sections(values['benchmark-plan.md'])
    if any(old[n] != new[n] for n in (2, 4, 5, 7, 8)):
        raise ValueError('unchanged scientific or protection section differs')
    before = 'Matched inputs and balanced execution order remain required.'
    after = ('Matched inputs and balanced placement remain required. For unstarted work, Section 6 adopts a common '
             'within-configuration order and records the increased temporal-drift exposure from running configurations sequentially.')
    if old[3].count(before) != 1 or old[3].replace(before, after) != new[3]:
        raise ValueError('section three exceeds the exact operational replacement')
    return {'schema': 'dsv41-operational-compatibility/v1',
            'scientific_plan': str(root/(DECISION+'previous-benchmark-plan.md')),
            'scientific_plan_sha256': PRIOR, 'operational_plan_sha256': CURRENT,
            'amendment_sha256': AMENDMENT, 'matrix_sha256': MATRIX,
            'unchanged_sections': [2, 4, 5, 7, 8], 'section_three_exact_replacement': True,
            'historical_manifests_rewritten': False, 'scientific_acceptance': False}
