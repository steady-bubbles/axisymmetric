"""Discretisation study for zero-flux rings at fixed hole radius.

Each selected ring is re-represented with M Fourier modes (curvature-density map
re-designed from the solution) and re-solved with Newton at the same r_h, on
panels resolving the modes at two panel densities (c_h) and with a tighter
panel tolerance.  Reports We, lam, circulation, r_o, z_max and the resolution
indicators, so the spread over discretisations measures the error. Rings already in out.json
are skipped; the others are appended.

Usage: python scripts/ring_convergence.py out.json run.json:i [run.json:j ...] [--M=128,192,256]
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.ring import (RingMap, RingPanels, RingState, ring_curve_points, ring_newton, ring_panels,  # noqa: E402
                          ring_z_max)
from bubbles.ring_init import curve_theta, design_map, fit_ring  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()


def remap(Y, rm, M_new):
    pts, _ = ring_curve_points(Y, rm, n=200000)
    s, sm, th = curve_theta(pts)
    rm2 = design_map(sm, th, s[-1], K=96, ell=0.3, alpha=1.0, n=20000)
    M = len(Y) - 3
    a, rho = fit_ring(pts, rm2, M_new, n=32768)
    return np.concatenate([a, [Y[M], rho, Y[M + 2]]]), rm2


def summary(Y, st, rm, hist, t0, **kw):
    M = len(Y) - 3
    return dict(M=M, N=st.P.N, We=float(Y[M + 2]), lam=float(Y[M]), circulation=float(st.circulation),
                r_h=float(st.g.r_h), r_o=float(st.g.r_o), z_max=float(ring_z_max(Y, rm)),
                res_hi=float(st.res_hi(Y[M + 2], Y[M])), tail=float(np.abs(Y[int(0.85 * M):M]).max()),
                F=float(hist[-1][0]) if hist else None, closed=bool(hist and hist[-1][2]), its=len(hist),
                t=time.time() - t0, **kw)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    Ms = [int(x) for x in opts.get('M', '128,192,256').split(',')]
    out, specs = args[0], args[1:]
    res = json.load(open(out)) if os.path.exists(out) else []
    for spec in specs:
        if any(r['source'] == spec for r in res):
            print(f"{spec}: already in {out}", flush=True)
            continue
        fn, idx = spec.rsplit(':', 1)
        p = json.load(open(fn))[int(idx)]
        Y0 = np.concatenate([p['a'], [p['lam'], p['rho_h'], p['We']]])
        rm0 = RingMap(np.array(p['map_b']))
        st0 = RingState(Y0[:len(p['a'])], p['rho_h'], rm0, RingPanels(np.array(p['breaks'])))
        r_h = st0.g.r_h
        rows = [dict(source=spec, M=len(p['a']), We=p['We'], lam=p['lam'], circulation=p['circulation'],
                     r_h=p['r_h'], r_o=p['r_o'], z_max=p['z_max'], res_hi=p['res_hi'], original=True)]
        # all M on the standard panels, plus the largest M on panels twice as dense
        for M, c_h, tol in [(M, 6.0, 1e-11) for M in Ms] + [(Ms[-1], 3.0, 1e-13)]:
            t0 = time.time()
            Yr, rm = remap(Y0, rm0, M)
            P = ring_panels(Yr[:M], rm, M, c_h=c_h, tol=tol)
            Y, st, hist, _ = ring_newton(Yr, rm, P, r_h, tol=1e-13, maxit=30, log=lambda m: None)
            rows.append(summary(Y, st, rm, hist, t0, c_h=c_h, ptol=tol))
            d = rows[-1]
            print(f"{spec} r_h={r_h:.5g} M={M} c_h={c_h} N={d['N']}: We={d['We']:.14f} "
                  f"lam={d['lam']:+.13f} circ={d['circulation']:+.12f} tail={d['tail']:.0e} "
                  f"res_hi={d['res_hi']:.0e} |F|={d['F']:.0e} {d['t']:.0f}s", flush=True)
        good = [d for d in rows[1:] if d['closed']]          # unclosed re-solves do not count
        rec = dict(source=spec, r_h=r_h, rows=rows, n_closed=len(good))
        for k, ko in (('We', 'We'), ('lam', 'lam'), ('circulation', 'circulation')):
            v = np.array([d[k] for d in good])
            if len(v):
                rec[k + '_spread_rel'] = float((v.max() - v.min()) / abs(v).max())
                rec[k + '_vs_original_rel'] = float(np.abs(v - p[ko]).max() / abs(p[ko]))
        res.append(rec)
        json.dump(res, open(out, 'w'), indent=1)
        print(f"   {len(good)} closed re-solves; relative spread We {rec.get('We_spread_rel', np.nan):.1e}, "
              f"lam {rec.get('lam_spread_rel', np.nan):.1e}, Gamma {rec.get('circulation_spread_rel', np.nan):.1e}; "
              f"vs the run's values We {rec.get('We_vs_original_rel', np.nan):.1e}", flush=True)


if __name__ == '__main__':
    main()
