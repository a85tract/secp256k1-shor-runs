#!/usr/bin/env python3
"""Add to a V-B JSON of h1/hexcap/hexlead_cap.py what the harness prices besides the window counts (as
W = 9, o1/hexdigit_check.py), the first-lookup counts, and for the abort cap the stream's cap, its emitted CCX + CCZ
(counted in the raw stream the V-B used) and the raw's sha256. The harness checks abort_cap = emitted - stream_toffoli
+ stream_cap and that a Lemma G path Monte Carlo counted aborts under the same stream cap and stream.
usage: O2_OPS_RAW=RAW python3 -I -B h1/hexcap/patch_cap_json.py JSON W"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hexlead_cap  # noqa: E402
import offdep       # noqa: E402  (o2/port, on the path through hexlead_cap)

path, W = sys.argv[1], int(sys.argv[2])
raw = os.environ['O2_OPS_RAW']
d = json.load(open(path))
d['extra_toffoli'] = {9: 532, 8: 486}[W]
d['extra_toffoli_src'] = 'o1 correction-phase digit register (o1/hexdigit_check.py identity - conv, o2/results5/digit_reg.txt)'
k = np.array(np.memmap(raw, dtype=offdep.DT, mode='r')['kind'])
d['stream_toffoli'] = int(((k == 13) | (k == 14)).sum())
d['stream_cap'] = hexlead_cap.STREAM_CAP
h = hashlib.sha256()
with open(raw, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 24), b''):
        h.update(blk)
d['stream_raw_sha256'] = h.hexdigest()
for e in d['sizes']:
    if e['n'] == 256:
        e['first_lookup'] = hexlead_cap.first_lookup(256)
        e['first_lookup_offset'] = hexlead_cap.first_lookup(256, offset=True)
        assert e['abort_cap'] == e['counts']['toffoli'] - d['stream_toffoli'] + d['stream_cap'], 'cap mismatch'
json.dump(d, open(path, 'w'), indent=1)
print(path, d['extra_toffoli'], d['stream_toffoli'], d['stream_cap'], d['stream_raw_sha256'][:16],
      [(e['abort_cap'], e['first_lookup'], e['first_lookup_offset']) for e in d['sizes'] if e['n'] == 256])
