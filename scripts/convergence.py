"""Discretisation study: re-solve selected branch points (pole height fixed,
We unknown) with increasingly fine discretisations and compare We, lambda.

Usage: python scripts/convergence.py [results/branch.json] [out.json]
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.continuation import Branch, solve_constrained  # noqa: E402
from bubbles.geometry import ParamMap  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

src = sys.argv[1] if len(sys.argv) > 1 else 'results/branch.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'results/convergence.json'
pts = json.load(open(src))
zps = np.array([p['z_pole'] for p in pts])
targets = [0.9, 0.6, 0.4, 0.2, 1e-2, 1e-3, 1e-4]
if zps.min() < 2e-5:
    targets.append(zps.min() * 1.0001)
LEVELS = [dict(Mfac=1.0, c_h=2.0, bie={}),
          dict(Mfac=1.5, c_h=1.5, bie={}),
          dict(Mfac=2.0, c_h=1.0, bie=dict(eta=2.0, ng=14))]
rows = []
for zt in targets:
    k = int(np.argmin(np.abs(np.log(zps / zt))))
    p = pts[k]
    zt = p['z_pole']
    res = []
    for lev in LEVELS:
        t0 = time.time()
        br = Branch(pmap=ParamMap(p['map_b']), c_h=lev['c_h'], bie_opts=lev['bie'])
        M = int(np.ceil(lev['Mfac'] * p['M'] / 4) * 4)
        Y = Branch.resize(np.concatenate([p['a'], [p['lam'], p['We']]]), M)
        Y, st, JY, hist = solve_constrained(br, Y, 'z_pole', zt)
        res.append(dict(M=M, N=st.P.N, We=float(Y[-1]), lam=float(Y[-2]),
                        res_hi=st.residual_hi(Y[-1], Y[-2]), t=time.time() - t0))
    dWe = max(abs(r['We'] - res[-1]['We']) for r in res[:-1])
    dlam = max(abs(r['lam'] - res[-1]['lam']) for r in res[:-1])
    rows.append(dict(z_pole=zt, levels=res, dWe=dWe, dlam=dlam))
    print(f"z_pole={zt:.4e}: We={res[-1]['We']:.15f} lam={res[-1]['lam']:+.15f} "
          f"max|dWe|={dWe:.1e} max|dlam|={dlam:.1e} "
          f"(M,N)={[(r['M'], r['N']) for r in res]}", flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
