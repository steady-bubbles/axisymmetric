"""Compare the ring branch computed at the standard resolution (rings_up.json,
M = 128, resolved to res_hi < 1e-6) with the high-precision rerun
(rings_up_hp.json, Fourier tail < 1e-13) at the same hole radii.

Usage: python scripts/compare_precision.py
Writes results/rings/precision.json and prints the table.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from bubbles.ring import RingMap, ring_z_max  # noqa: E402

RINGS = os.path.join(ROOT, 'results', 'rings')


def zmax(p):
    Y = np.concatenate([p['a'], [p['lam'], p['rho_h'], p['We']]])
    return ring_z_max(Y, RingMap(np.array(p['map_b'])), n=20001)


def main():
    std = json.load(open(os.path.join(RINGS, 'rings_up.json')))
    hp_file = os.path.join(RINGS, 'rings_up_hp.json')
    if not os.path.exists(hp_file):
        sys.exit('results/rings/rings_up_hp.json not found (the high-precision rerun, scripts/run_rings.py '
                 'results/rings/rings_down_part1.json:0 results/rings/rings_up_hp.json 1.15 40 320 --tail=1e-13)')
    hp = json.load(open(hp_file))
    rows = []
    print(f"{'r_h':>9} {'M':>4} {'M_hp':>4} {'We (hp)':>18} {'dWe':>9} {'dlam':>9} {'dGamma':>9} "
          f"{'dz_max':>9} {'tail_hp':>8}")
    for q in hp:
        p = min(std, key=lambda s: abs(np.log(s['r_h'] / q['r_h'])))
        if abs(p['r_h'] / q['r_h'] - 1) > 1e-9:
            continue
        d = dict(r_h=q['r_h'], M=len(p['a']), M_hp=len(q['a']), We_hp=q['We'], lam_hp=q['lam'],
                 circulation_hp=q['circulation'], z_max_hp=zmax(q),
                 dWe=p['We'] - q['We'], dlam=p['lam'] - q['lam'],
                 dcirculation=p['circulation'] - q['circulation'], dz_max=zmax(p) - zmax(q),
                 res_hi=p['res_hi'], res_hi_hp=q['res_hi'], tail_hp=q.get('tail'))
        rows.append(d)
        print(f"{d['r_h']:9.4f} {d['M']:4d} {d['M_hp']:4d} {d['We_hp']:18.14f} {d['dWe']:9.1e} "
              f"{d['dlam']:9.1e} {d['dcirculation']:9.1e} {d['dz_max']:9.1e} {d['tail_hp']:8.0e}")
    if rows:
        for k in ('dWe', 'dlam', 'dcirculation', 'dz_max'):
            print(f"max |{k}| = {max(abs(r[k]) for r in rows):.1e}")
    json.dump(rows, open(os.path.join(RINGS, 'precision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
