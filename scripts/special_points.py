"""High-precision computation of distinguished points on the branch:

  * the fold (maximal Weber number) We_max,
  * the dimple onset lambda = 0 (pole curvature H_pole = lambda vanishes),

each at several discretisations to estimate the error.  Requires the branch
from scripts/run_branch.py.

Usage: python scripts/special_points.py [results/branch.json] [out.json]
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.continuation import Branch, solve_constrained  # noqa: E402
from bubbles.geometry import ParamMap  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

src = sys.argv[1] if len(sys.argv) > 1 else 'results/branch.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'results/special_points.json'
pts = json.load(open(src))

RESOLUTIONS = [dict(Mfac=1.0, c_h=2.0), dict(Mfac=1.5, c_h=1.5), dict(Mfac=2.0, c_h=1.0)]


def solver(res):
    cache = {}
    warm = {}

    def solve(zp):
        # start from the branch point closest in log z_pole; the parameter map
        # and M of that point are kept for all solves of this resolution
        k = int(np.argmin([abs(np.log(p['z_pole'] / zp)) for p in pts]))
        p = pts[k]
        if warm.get('k') != k:
            warm.clear()
            warm['k'] = k
            br = Branch(pmap=ParamMap(p['map_b']), c_h=res['c_h'])
            M = int(np.ceil(res['Mfac'] * p['M'] / 4) * 4)
            warm['br'] = br
            warm['panels'] = br.panels_for(Branch.resize(
                np.concatenate([p['a'], [p['lam'], p['We']]]), M)[:M])
            warm['Y'] = Branch.resize(np.concatenate([p['a'], [p['lam'], p['We']]]), M)
        br = warm['br']
        M = len(warm['Y']) - 2
        Y, st, JY, hist = solve_constrained(br, warm['Y'].copy(), 'z_pole', zp,
                                            panels=warm['panels'])
        warm['Y'] = Y
        d = st.diagnostics()
        d.update(We=float(Y[-1]), lam=float(Y[-2]), M=M, N=st.P.N,
                 res_hi=st.residual_hi(Y[-1], Y[-2]), newton=hist[-1])
        cache[zp] = d
        return d
    return solve


result = {}
We = np.array([p['We'] for p in pts])
zps = np.array([p['z_pole'] for p in pts])
lam = np.array([p['lam'] for p in pts])
k = int(np.argmax(We))
bracket = (zps[min(k + 1, len(pts) - 1)], zps[max(k - 1, 0)])
k0 = int(np.nonzero(lam < 0)[0][0])
bracket_l = (zps[k0], zps[k0 - 1])
star = np.array([p.get('star_margin', 1.0) for p in pts])
k1 = int(np.nonzero(star < 0)[0][0])
bracket_s = (zps[k1], zps[k1 - 1])

for res in RESOLUTIONS:
    tag = f"M x{res['Mfac']}, c_h={res['c_h']}"
    solve = solver(res)
    opt = minimize_scalar(lambda z: -solve(z)['We'], bounds=bracket, method='bounded',
                          options=dict(xatol=1e-9))
    fold = solve(opt.x)
    z0 = brentq(lambda z: solve(z)['lam'], *bracket_l, xtol=1e-15, rtol=1e-15)
    dimple = solve(z0)
    z1 = brentq(lambda z: solve(z)['star_margin'], *bracket_s, xtol=1e-14, rtol=1e-14)
    star = solve(z1)
    result[tag] = dict(fold=fold, dimple_onset=dimple, star_loss=star)
    print(f"{tag}: We_max={fold['We']:.15f} at z_pole={fold['z_pole']:.10f} "
          f"chi={fold['aspect']:.10f} lam={fold['lam']:+.12f} (M={fold['M']}, N={fold['N']})")
    print(f"{' ' * len(tag)}  dimple onset: We={dimple['We']:.15f} z_pole={dimple['z_pole']:.12f} "
          f"chi={dimple['aspect']:.10f}")
    print(f"{' ' * len(tag)}  loss of star-shapedness: We={star['We']:.15f} "
          f"z_pole={star['z_pole']:.12f}")
    json.dump(result, open(out, 'w'), indent=1)
