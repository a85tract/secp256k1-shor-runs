#!/usr/bin/env python3
"""Add to a hexterm V-B JSON what the harness / o1/hexva.py read besides the window counts (as o2/hex/patch_json.py):
o2/results5/digit_reg.txt; the terminal's digit register is a hex window's) and the first-lookup counts.
usage: python3 -I -B s4/term/patch_json.py JSON W"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('O2_OPS_RAW', os.path.join(HERE, '..', 'ext', 'run', 'bind81-builder.raw'))
import hexterm
path, W = sys.argv[1], int(sys.argv[2])
d = json.load(open(path))
d['extra_toffoli'] = {9: 532, 8: 486}[W]
d['extra_toffoli_src'] = 'o1 correction-phase digit register (o1/hexdigit_check.py identity - conv, o2/results5/digit_reg.txt)'
d['stream_toffoli'] = 392944      # emitted CCX + CCZ of the replayed prefix [0, STOP) (V-B force-1 batch: toff_exec_stream 392,944)
d['stream_cap'] = hexterm.STREAM_CAP
d['terminal'] = dict(kind='hex_terminal_add', stream_stop_op=hexterm.STOP, pay_bits=hexterm.PAY, check_bits=hexterm.CHK,
                     select_swap_slots=1 << hexterm.T_SS, stream_cap=hexterm.STREAM_CAP)
for e in d['sizes']:
    if e['n'] == 256:
        e['declared_failure'] = hexterm.declared_failure(256)
        e['first_lookup'] = hexterm.first_lookup(256)
        e['first_lookup_offset'] = hexterm.first_lookup(256, offset=True)
json.dump(d, open(path, 'w'), indent=1)
print(path, d['extra_toffoli'], [(e['first_lookup'], e['first_lookup_offset']) for e in d['sizes'] if e['n'] == 256])
