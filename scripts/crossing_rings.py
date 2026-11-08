"""Rings solved exactly at the crossings of the even sectors m on the zero-flux branch: the
rings where non-axisymmetric branches with azimuthal wave number m leave (results/rings/
crossing_rings.json; the branch points on the page and in the README).

mu_m(r_h) is the lowest eigenvalue of the even sector m (bubbles.sectors.RingSector, K
cosines, translation removed for m = 1); each even sector crosses zero once along the
branch.  For each m the neighbouring records of --src (ordered by r_h) with mu_m of opposite
signs are found by bisection.  Secant steps in log r_h on mu_m (kept inside the bracket) then
each solve the ring at fixed r_h with ring_newton, from a Lagrange predictor in log r_h
through the nearest samples with the same parametrisation, until |mu_m| < --tol (else, after
--maxit solves, the best sample is kept with converged = false and a warning).  The slopes
along the branch are central differences from two more solves at log r_h* +- --dx:
dmu_dWe = (dmu/dr_h) / (dWe/dr_h), and dmu_dWe_err its change when the next nearest sample is
added (a cubic through four samples).  With --dQ > 0 the ring is also solved at fixed r_h* with
the flux Q = +-dQ R^2 (R = (r_h + r_o)/2), which gives the Q-derivatives and dmu_dWe_Gamma, the
slope along the axisymmetric rings of fixed circulation (dQ/dr_h = -Gamma_r/Gamma_Q there),
for a Lyapunov-Schmidt reduction at fixed circulation (for m = 1 take --dQ=2e-4: the flux acts on
the neck on the scale |Q| ~ 0.55 r_h^1.5).  Solves and sectors use the same panels,
ring_panels(c_h, 1e-10).

Usage: python scripts/crossing_rings.py [out.json] [--src=results/rings/rings_up.json]
       [--m=2,3,4,5,6] [--tol=1e-3] [--dQ=2e-3] [--dx=1e-3] [--K=64] [--c_h=20] [--maxit=8]

Finite-difference Jacobians use $BUBBLES_NPROC processes.  out defaults to
results/rings/crossing_rings.json: a list of records with the rings_up.json keys plus Q (= 0),
m, parity ('even'), converged, mu (lowest eigenvalue of the sector at the ring), mu_next,
dmu_drh, dWe_drh, dGam_drh, dmu_dWe, dmu_dWe_err; with --dQ also dQ, dmu_dQ, dWe_dQ, dGam_dQ,
dmu_dWe_Gamma, dmu_dWe_Gamma_err (from the r_h slopes only); secant [[r_h, We, mu], ...] (every
sample: the bracketing records, the secant steps, the two central samples), bracket
[r_lo, r_hi], src, idx_bracket, settings, time.  The output is rewritten after each m and a
restart skips the m already there.  An m that fails (no sign change in the records, or a
solve that does not converge) is logged and skipped, and the exit status is 1.
"""
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings('ignore')

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from bubbles.geometry import decay_rate  # noqa: E402
from bubbles.ring import (RingMap, RingState, residual_floor, ring_newton, ring_panels,  # noqa: E402
                          ring_theta, ring_z_max)
from bubbles.sectors import RingSector  # noqa: E402


def save_json(obj, path):
    """Write atomically, so readers never see a half-written file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f)
    os.replace(path + '.tmp', path)


def log(s):
    print(s, flush=True)


def record(Y, st, rm, F, Q=0.0):
    """The rings_up.json keys of a solved ring (plus its flux Q)."""
    M = len(Y) - 3
    g = st.g
    _, tht0 = ring_theta(Y[:M], np.array([0.0]))
    k_waist = -tht0[0] / (g.L * rm.Sp(np.array([0.0]))[0])
    rate, needed = decay_rate(Y[:M])
    return dict(r_h=float(g.r_h), We=float(Y[M + 2]), lam=float(Y[M]), r_o=float(g.r_o),
                z_max=ring_z_max(Y, rm), circulation=float(st.circulation), Q=Q,
                kappa1_waist=float(k_waist), H_waist_rh=float((k_waist - 1 / g.r_h) * g.r_h),
                u_max=float(np.sqrt(st.q.max())), kmax=float(np.abs(g.kappa1).max()),
                res_hi=st.res_hi(Y[M + 2], Y[M]), tail=float(np.abs(Y[int(0.85 * M):M]).max()),
                F=F, M=M, N=st.P.N, decay=float(rate), modes_needed=int(needed),
                a=Y[:M].tolist(), rho_h=float(Y[M + 1]), map_b=rm.b.tolist(), breaks=st.P.breaks.tolist())


class Rings:
    """Samples (record or solved ring) with their sector eigenvalues, cached."""

    def __init__(self, recs, K, c_h):
        self.recs, self.K, self.c_h = recs, K, c_h
        self.order = sorted(range(len(recs)), key=lambda i: recs[i]['r_h'])
        self.cache = {}

    def sample(self, Y, rm, st, F=None):
        return dict(x=float(np.log(st.g.r_h)), Y=Y, rm=rm, st=st, We=float(Y[-1]),
                    Gam=float(st.circulation), F=F, mu={})

    def of_record(self, k):
        """Sample of the k-th record in r_h order."""
        if k not in self.cache:
            p = self.recs[self.order[k]]
            a, rm = np.array(p['a']), RingMap(np.array(p['map_b']))
            st = RingState(a, p['rho_h'], rm, ring_panels(a, rm, len(a), c_h=self.c_h, tol=1e-10))
            self.cache[k] = self.sample(np.concatenate([a, [p['lam'], p['rho_h'], p['We']]]), rm, st)
        return self.cache[k]

    def mu(self, s, m):
        if m not in s['mu']:
            s['mu'][m] = RingSector(s['st'], m).eigs(s['We'], 'even', K=self.K, nev=2)[0]
        return s['mu'][m]

    def solve(self, Yp, rm, r_h, Q, J):
        """ring_newton at fixed r_h and flux Q; returns (sample, J)."""
        M = len(Yp) - 3
        P = ring_panels(Yp[:M], rm, M, c_h=self.c_h, tol=1e-10)
        Y, st, hist, J = ring_newton(Yp, rm, P, r_h, Q=Q, J=J, log=lambda s: log('     ' + s))
        if not (hist and hist[-1][2] and (hist[-1][0] < max(1e-9, residual_floor(st.g.r_o))
                                          or hist[-1][1] < 1e-10)):
            raise RuntimeError(f'ring_newton failed at r_h={r_h:.6g}, Q={Q:.3g}')
        return self.sample(Y, rm, st, F=hist[-1][0]), J


def find_bracket(rings, m, lo):
    """Positions (k, k + 1) in r_h order of neighbouring records with mu_m of opposite signs,
    by bisection from lo (mu_m changes sign once along the branch)."""
    hi = len(rings.order) - 1

    def sgn(k):
        s = rings.of_record(k)
        mu = rings.mu(s, m)[0]
        log(f'm={m}: record {rings.order[k]} r_h={np.exp(s["x"]):.6f} We={s["We"]:.9f} mu={mu:+.6e}')
        return np.sign(mu)
    s_lo = sgn(lo)
    if s_lo == sgn(hi):
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if sgn(mid) == s_lo else (lo, mid)
    return lo, hi


def next_x(pts, m):
    """Secant step on mu_m(log r_h) through the two samples with the smallest |mu|, or regula
    falsi on the tightest bracket if the secant leaves it."""
    s = sorted(pts, key=lambda p: p['x'])
    b0, b1 = next((p, q) for p, q in zip(s[:-1], s[1:]) if np.sign(p['mu'][m][0]) != np.sign(q['mu'][m][0]))
    p, q = sorted(pts, key=lambda p: abs(p['mu'][m][0]))[:2]
    x = p['x'] - p['mu'][m][0] * (q['x'] - p['x']) / (q['mu'][m][0] - p['mu'][m][0])
    lo, hi = b0['x'], b1['x']
    if not lo + 0.01 * (hi - lo) < x < hi - 0.01 * (hi - lo):
        x = lo - b0['mu'][m][0] * (hi - lo) / (b1['mu'][m][0] - b0['mu'][m][0])
    return x


def predictor(pts, x):
    """Lagrange interpolation in log r_h of Y through the (up to) three samples nearest x
    that share the parametrisation (map and M) of the nearest one."""
    near = sorted(pts, key=lambda p: abs(p['x'] - x))
    same = [p for p in near if len(p['Y']) == len(near[0]['Y']) and np.array_equal(p['rm'].b, near[0]['rm'].b)][:3]
    xs = np.array([p['x'] for p in same])
    w = [np.prod([(x - xs[j]) / (xs[i] - xs[j]) for j in range(len(xs)) if j != i]) for i in range(len(xs))]
    return sum(wi * p['Y'] for wi, p in zip(w, same)), same[0]['rm']


def slope(pts, x0, key, n=3):
    """d(key)/d log r_h at x0 from the polynomial through the n samples nearest x0."""
    near = sorted(pts, key=lambda p: abs(p['x'] - x0))[:n]
    return float(np.polyfit([p['x'] - x0 for p in near], [key(p) for p in near], len(near) - 1)[-2])


def crossing(rings, m, lo, opts):
    """Solve the ring at the crossing of sector m; returns (record or None, bracket position)."""
    t0 = time.time()
    br = find_bracket(rings, m, lo)
    if br is None:
        log(f'm={m}: no sign change of mu_{m} in the records')
        return None, lo
    pts = [rings.of_record(k) for k in br]
    J = None
    for it in range(opts['maxit']):
        x = next_x(pts, m)
        Yp, rm = predictor(pts, x)
        s, J = rings.solve(Yp, rm, np.exp(x), 0.0, J)
        pts.append(s)
        mu = rings.mu(s, m)
        log(f'm={m} it {it}: r_h={np.exp(x):.9f} We={s["We"]:.12f} mu={mu[0]:+.6e} (next {mu[1]:+.4f}) '
            f'|F|={s["F"]:.1e} [{time.time() - t0:.0f}s]')
        if abs(mu[0]) < opts['tol']:
            break
    best = min(pts[2:], key=lambda p: abs(p['mu'][m][0]))
    x0, r_h = best['x'], float(np.exp(best['x']))
    converged = bool(abs(best['mu'][m][0]) < opts['tol'])
    if not converged:
        log(f'm={m}: WARNING |mu| = {abs(best["mu"][m][0]):.1e} >= tol after {opts["maxit"]} solves')
    for sg in (1, -1):                                  # central samples for the slopes
        x = x0 + sg * opts['dx']
        Yp, rm = predictor(pts, x)
        s, J = rings.solve(Yp, rm, np.exp(x), 0.0, J)
        pts.append(s)
        log(f'm={m} r_h={np.exp(x):.9f}: We={s["We"]:.12f} mu={rings.mu(s, m)[0]:+.6e}')
    keys = (('mu', lambda p: p['mu'][m][0]), ('We', lambda p: p['We']), ('Gam', lambda p: p['Gam']))
    d, d4 = [{k: slope(pts, x0, f, n) for k, f in keys} for n in (3, 4)]
    out = record(best['Y'], best['st'], best['rm'], best['F'])
    out.update(m=m, parity='even', converged=converged, mu=float(best['mu'][m][0]),
               mu_next=float(best['mu'][m][1]), dmu_drh=d['mu'] / r_h, dWe_drh=d['We'] / r_h,
               dGam_drh=d['Gam'] / r_h, dmu_dWe=d['mu'] / d['We'],
               dmu_dWe_err=abs(d['mu'] / d['We'] - d4['mu'] / d4['We']))
    if opts['dQ'] > 0:
        dq = opts['dQ'] * (0.5 * (out['r_h'] + out['r_o'])) ** 2
        q = {}
        for sg in (1, -1):
            s, J = rings.solve(best['Y'], best['rm'], r_h, sg * dq, J)
            q[sg] = (rings.mu(s, m)[0], s['We'], s['Gam'])
            log(f'm={m} Q={sg * dq:+.3e}: We={s["We"]:.12f} Gam={s["Gam"]:+.10f} mu={q[sg][0]:+.6e}')
        mu_Q, We_Q, Gam_Q = [(q[1][i] - q[-1][i]) / (2 * dq) for i in range(3)]

        def along_gamma(d):
            dQdx = -d['Gam'] / Gam_Q                    # along fixed circulation
            return (d['mu'] + mu_Q * dQdx) / (d['We'] + We_Q * dQdx)
        out.update(dQ=dq, dmu_dQ=mu_Q, dWe_dQ=We_Q, dGam_dQ=Gam_Q, dmu_dWe_Gamma=along_gamma(d),
                   dmu_dWe_Gamma_err=abs(along_gamma(d) - along_gamma(d4)))
    out.update(secant=[[float(np.exp(p['x'])), p['We'], float(p['mu'][m][0])] for p in pts],
               bracket=[float(np.exp(p['x'])) for p in pts[:2]], src=opts['src'],
               idx_bracket=[rings.order[k] for k in br],
               settings=dict(K=rings.K, c_h=rings.c_h, tol=opts['tol'], dQ=opts['dQ'], dx=opts['dx']),
               time=round(time.time() - t0, 1))
    log(f'm={m}: r_h*={r_h:.9f} We*={out["We"]:.12f} mu={out["mu"]:+.2e} '
        f'dmu/dWe={out["dmu_dWe"]:+.6f}+-{out["dmu_dWe_err"]:.0e}'
        + (f' (fixed Gamma {out["dmu_dWe_Gamma"]:+.6f})' if 'dmu_dWe_Gamma' in out else '')
        + f' [{out["time"]:.0f}s]')
    return out, br[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    out = args[0] if args else os.path.join(ROOT, 'results', 'rings', 'crossing_rings.json')
    src = opts.get('src', os.path.join(ROOT, 'results', 'rings', 'rings_up.json'))
    ms = [int(x) for x in opts.get('m', '2,3,4,5,6').split(',')]
    cfg = dict(src=os.path.relpath(src, ROOT), tol=float(opts.get('tol', 1e-3)), dQ=float(opts.get('dQ', 2e-3)),
               dx=float(opts.get('dx', 1e-3)), maxit=int(opts.get('maxit', 8)))
    rings = Rings(json.load(open(src)), K=int(opts.get('K', 64)), c_h=float(opts.get('c_h', 20)))
    done = json.load(open(out)) if os.path.exists(out) else []
    lo, failed = 0, []           # by monotonicity in m, mu_{m+1} > mu_m: brackets move up with m
    for m in sorted(ms):
        if any(d['m'] == m for d in done):
            log(f'm={m}: already in {out}')
            continue
        try:
            rec, k = crossing(rings, m, lo, cfg)
        except Exception:  # noqa: BLE001
            log(f'm={m}: FAILED\n{traceback.format_exc()}')
            rec = None
        if rec is None:
            failed.append(m)
            continue
        lo = k
        done = sorted(done + [rec], key=lambda d: d['m'])
        save_json(done, out)
    log(f'{len(done)} rings -> {out}')
    if failed:
        sys.exit(f'FAILED: m = {failed}')


if __name__ == '__main__':
    main()
