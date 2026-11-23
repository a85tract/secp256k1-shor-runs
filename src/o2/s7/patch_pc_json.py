#!/usr/bin/env python3
"""add to a capped V-B JSON of a leader-family stream what the harness prices besides the window counts
(as h1/hexcap/patch_cap_json.py does for the leader stream): extra_toffoli and its source, first-lookup counts (copied
from the reference kind file REF of the same window kind: they do not depend on the stream), the stream cap (read from
the JSON's own params, STREAM_CAP=...), the raw's emitted CCX + CCZ and its sha256; checks abort_cap = emitted -
stream_toffoli + stream_cap. usage: O2_OPS_RAW=RAW python3 -I patch_pc_json.py JSON REF"""
import hashlib, json, os, sys
import numpy as np
DT = np.dtype([('kind', '<u4'), ('pad', '<u4'), ('c2', '<u8'), ('c1', '<u8'), ('t', '<u8'), ('ct', '<u8'),
               ('cc', '<u8'), ('r', '<u8')])
path, ref = sys.argv[1], sys.argv[2]
raw = os.environ['O2_OPS_RAW']
d, r = json.load(open(path)), json.load(open(ref))
assert d['kind'] == r['kind']
d['extra_toffoli'], d['extra_toffoli_src'] = r['extra_toffoli'], r['extra_toffoli_src']
cap = [int(p.split('=')[1]) for p in d['params'] if p.startswith('STREAM_CAP=')]
assert len(cap) == 1, 'STREAM_CAP must be a -p param of the V-B run'
k = np.array(np.memmap(raw, dtype=DT, mode='r')['kind'])
d['stream_toffoli'] = int(((k == 13) | (k == 14)).sum())
d['stream_cap'] = cap[0]
h = hashlib.sha256()
with open(raw, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 24), b''):
        h.update(blk)
d['stream_raw_sha256'] = h.hexdigest()
d['stream_src'] = os.environ.get('O2_SITEFILE', '')
rs = {e['n']: e for e in r['sizes']}
for e in d['sizes']:
    if e['n'] == 256:
        e['first_lookup'] = rs[256]['first_lookup']
        e['first_lookup_offset'] = rs[256]['first_lookup_offset']
        assert e['abort_cap'] == e['counts']['toffoli'] - d['stream_toffoli'] + d['stream_cap'], 'cap mismatch'
json.dump(d, open(path, 'w'), indent=1)
print(path, d['extra_toffoli'], d['stream_toffoli'], d['stream_cap'], d['stream_raw_sha256'][:16],
      [(e['abort_cap'], e['abort_cap'] + d['extra_toffoli'], e['counts']['peak_qubits']) for e in d['sizes'] if e['n'] == 256])
