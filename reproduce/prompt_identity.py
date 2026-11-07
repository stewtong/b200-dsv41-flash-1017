"""Short ownership aliases, separate from immutable scientific task identities."""
from datetime import datetime
import re

import comparison_scope
import launcher


def _dimensions(input_tokens, cache, stamp):
    allowed = comparison_scope.INPUTS if cache == 'cold' else comparison_scope.WARM_INPUTS
    if cache not in ('cold', 'warm_prefix') or type(input_tokens) is not int or input_tokens not in allowed:
        raise ValueError('unknown input preparation panel')
    if not isinstance(stamp, str) or not re.fullmatch(r'[0-9]{8}t[0-9]{4}z', stamp):
        raise ValueError('exact preparation timestamp required')
    datetime.strptime(stamp, '%Y%m%dt%H%Mz')
    return 'c' if cache == 'cold' else 'w'


def child(input_tokens, concurrency, repetition, cache, stamp):
    code = _dimensions(input_tokens, cache, stamp)
    cs = comparison_scope.CONCURRENCIES if cache == 'cold' else comparison_scope.WARM_CONCURRENCIES
    if type(concurrency) is not int or concurrency not in cs or type(repetition) is not int or repetition not in range(3):
        raise ValueError('unknown input source population')
    alias = f'pi-{code}-i{input_tokens // 1000}-c{concurrency}-r{repetition}-{stamp}'
    launcher.validate_run_id(alias)
    return alias


def panel(input_tokens, cache, stamp):
    code = _dimensions(input_tokens, cache, stamp)
    alias = f'ib-{code}-i{input_tokens // 1000}-{stamp}'
    launcher.validate_run_id(alias)
    return alias
