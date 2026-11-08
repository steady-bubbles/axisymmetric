"""Pseudo-arclength continuation of ring bubbles in the hole flux Q at fixed hole radius r_h.

Unknowns X = (a_1..a_M, lam, rho_h, We, Q).  Equations: the ring rows of
bubbles.ring._ring_F at the current Q (M+1 Bernoulli rows, closure, r_h - target) and
the arclength row (W^2 tau).(X - X_pred) = 0, tau the unit tangent of the last point in
the norm |W x|.  The weights favour (We, Q, lam) (--w):
    |W dX|^2 = (w_We dWe/We)^2 + (w_Q dQ/Q_ref)^2 + (w_lam dlam/lam_ref)^2
               + w_shape^2 (|da|^2 + (drho_h/rho_ref)^2),
Q_ref = R^2/2, lam_ref = max(1, |lam|) and rho_ref at the start, We relative to the
current point (We grows without bound towards the Hicks end).  Full weight on the shape
coefficients made the steps tiny at r_h <= 0.42.

Jacobian: gamma = V^{-1}[Q - r^2/2] is affine in Q, so with gamma_1 = V^{-1}[1] the Q column
dF_B/dQ = proj(-We gamma gamma_1) is exact.  The a and rho_h columns are finite differences
(ring_jacobian).  Q is frozen into the worker context when a Pool is created, so every
Jacobian gets a fresh Pool.  The Jacobian is recomputed at every accepted point; it gives
the exact tangent (null vector of the ring rows), the sign of det[J; tau^T] (tau towards
increasing s_arc; the sign changes at branch points in the symmetric class, not at
folds) and sigma_min of the square Jacobian at fixed (r_h, Q).

Flags: FOLD_Q (dQ/ds changes sign), EXTREMUM_We (dWe/ds changes sign), LAM0 (lam changes
sign; thin-tube prediction We2D = 2), GAMMA0 (circulation changes sign: 3 more steps,
then stop), PINCH (interior minimum of z(t) below 0.1 z_max: the cross-section pinches,
stop), DET_SIGN (det sign changed since the previous point; the crossing is located by
regula falsi on the signed smallest singular value of the bordered Jacobian and stored
as an extra record flagged BRANCH), rebased (re-represented with more modes; the det sign
is compared before the remap, in the representation of the previous point).

Resolution: res_hi > 1e-6 or tail > 1e-6 -> remap (design_map with K = min(48, M/2) sine
coefficients, fit_ring, M x 1.35 up to --Mmax) and re-solve at fixed (r_h, Q), also at the
start; stop when res_hi > 1e-4 or tail > 1e-6 remain.  A map with K > M/2 spoils the fit
(res_hi 1e-4 at M = 32, K = 48 where K = 16 gives 2e-10).
Steps: halved when not converged after 8 quasi-Newton iterations, when the corrector lands
more than ds/2 from the predictor or when the tangent turns by more than 60 degrees (a jump
to another branch); x1.5 after 3 easy steps (<= 6 iterations, no Jacobian refresh; the
per-point Jacobian costs as much as ~30 iterations), |dQ| <= 0.1 R^2.  The first step is
|dQ| = --dQ0 (default for r_h <= 0.24: 0.05 r_h^(3/2), the neck scale), else ds = --ds0.

Output: a plain list of records sorted by the signed arclength s_arc (start at 0 with
dir = 0, the dir = -1 side negative), with the keys of rings_up.json plus Q, Qhat = 2Q/R^2,
R = (r_h + r_o)/2, u_waist (sheet strength at the inner rim), We2D = We Gamma^2/(4 pi^2 a)
(a = (r_o - r_h)/2), z_dip (interior minimum of z), det_sign, sigma_min, sigma_b (signed
smallest singular value of the bordered Jacobian), dQ_ds, dWe_ds, dlam_ds (tangent towards
increasing s_arc), dir, s_arc, ds, its, nJ, flags.  ring_sectors.py and export_rings_viz.py
read it unchanged.  Written atomically after every point; settings and
per-direction status go to <out>.meta.json.  Re-running the command continues every
direction that did not reach a final stop from its last record; an existing output that
starts from another (r_h, Q) is refused.

Usage: python scripts/ring_qscan.py src.json idx out.json [--M=] [--dirs=-1,+1] [--ds0=0.1]
       [--dQ0=] [--dsmax=0.5] [--nmax=40] [--tcap=30] [--stall=15] [--stop-We=5]
       [--qmin=] [--qmax=] [--w=We:1,Q:1,lam:1,shape:0.1] [--Mmax=128] [--Q0=]
  --M modes (default 64; for a record with Q, e.g. from a sheet, its own M); --nmax
  accepted steps per direction (in total, over restarts); --tcap and --stall in minutes:
  wall clock per direction and without an accepted step.  --Q0 re-solves the start at this
  Q (default: the record's Q); with --nmax=0 the output is just that ring.
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from numpy.polynomial import chebyshev as C

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.geometry import decay_rate  # noqa: E402
from bubbles.ring import (RingMap, RingPanels, RingState, _init, _ring_F, closure_ok, default_nproc,  # noqa: E402
                          residual_floor, ring_curve_points, ring_jacobian, ring_panels, ring_theta, ring_z_max)
from bubbles.ring_init import curve_theta, design_map, fit_ring  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

C_H, RES_TOL, RES_MAX, TAIL_MAX = 6.0, 1e-6, 1e-4, 1e-6
MAXIT, EASY_IT = 8, 6
FINAL = ('We', 'Q range', 'pinch', 'Gamma0', 'resolution', 'step')   # no restart after these
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def save_json(obj, path):
    """Write atomically, so readers never see a half-written file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f)
    os.replace(path + '.tmp', path)


# ---------------------------------------------------------------------------
# extended system

def F_ext(X, rm, P, rht, lin):
    """Ring rows at Q = X[-1] and the linear row lin = (w, c): w.X - c."""
    F, st = _ring_F(X[:-1], rm.b, P.breaks, rht, float(X[-1]))
    return np.append(F, lin[0] @ X - lin[1]), st


def gamma1(st):
    """Sheet strength of the pure flux flow V^{-1}[1] = d gamma/dQ."""
    return st.op.solve(np.ones(st.P.N))


def dF_dQ(st, We):
    """Exact Q column of the ring rows (closure and r_h rows do not depend on Q)."""
    col = np.zeros(st.M + 3)
    col[:st.M + 1] = st.proj @ (-We * st.gamma * gamma1(st))
    return col


def jac_rows(X, st, rm, P, rht, nproc=None):
    """(M+3) x (M+4) Jacobian of the ring rows; a fresh Pool carries the current Q."""
    M = len(X) - 4
    Y, Q = X[:-1], float(X[-1])
    with Pool(nproc or default_nproc(), initializer=_init,
              initargs=(dict(b=rm.b, breaks=P.breaks, rht=rht, Q=Q),)) as pool:
        F0 = np.append(st.F(Y[M + 2], Y[M]), st.g.r_h - rht)
        J = ring_jacobian(Y, st, rm, P, rht, Q, pool=pool, F0=F0)
    return np.column_stack([J, dF_dQ(st, Y[M + 2])])


def converged(F, M, dX, r_o, rht, lin):
    fb = np.abs(F[:M + 1]).max()
    rows = closure_ok(F[:M + 3], M, max(1.0, rht)) and abs(F[M + 3]) < 1e-11 * max(1.0, abs(lin[1]))
    return rows and (fb < max(1e-11, residual_floor(r_o)) or (fb < 1e-9 and np.abs(dX).max() < 1e-10))


def newton(X, rm, P, rht, lin, J, maxit=MAXIT, nproc=None):
    """Quasi-Newton with backtracking on the ring rows plus lin: from the ring-row Jacobian J
    (computed here if None), good-Broyden updated after every step (one or two iterations
    fewer than plain chord steps), recomputed (at most once) if the line search fails or
    the contraction is poor.  Returns (X, st, ok, its, nJ, J)."""
    M = len(X) - 4
    F, st = F_ext(X, rm, P, rht, lin)
    nJ = 0
    if J is None:
        J, nJ = jac_rows(X, st, rm, P, rht, nproc), 1
    if converged(F, M, np.full(1, np.inf), st.g.r_o, rht, lin):
        return X, st, True, 0, nJ, J
    for it in range(maxit):
        dX = np.linalg.solve(np.vstack([J, lin[0]]), -F)
        f0 = np.linalg.norm(F)
        step, ok = 1.0, False
        for _ in range(6):
            try:
                Ft, stt = F_ext(X + step * dX, rm, P, rht, lin)
                ok = np.all(np.isfinite(Ft)) and np.linalg.norm(Ft) < (1 - 0.1 * step) * f0 + 1e-13
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                break
            step *= 0.5
        if not ok:
            if nJ:
                return X, st, False, it + 1, nJ, J
            J, nJ = jac_rows(X, st, rm, P, rht, nproc), 1
            continue
        rate = np.linalg.norm(Ft) / max(f0, 1e-300)
        sX = step * dX
        J = J + np.outer(Ft[:M + 3] - F[:M + 3] - J @ sX, sX) / (sX @ sX)    # good Broyden
        X, F, st = X + sX, Ft, stt
        if converged(F, M, sX, st.g.r_o, rht, lin):
            return X, st, True, it + 1, nJ, J
        if rate > 0.3 and nJ == 0 and np.abs(F[:M + 1]).max() > 1e-9:
            J, nJ = jac_rows(X, st, rm, P, rht, nproc), 1
    return X, st, False, maxit, nJ, J


def solve_fixed_Q(X, rm, P, rht, nproc=None):
    """Re-solve at fixed (r_h, Q); the Jacobian is renewed if the solution moved."""
    M = len(X) - 4
    eQ = np.zeros(M + 4)
    eQ[-1] = 1.0
    Xs, st, ok, its, _, J = newton(X, rm, P, rht, (eQ, float(X[-1])), None, maxit=14, nproc=nproc)
    if ok and np.abs(Xs - X).max() > 1e-6:
        J = jac_rows(Xs, st, rm, st.P, rht, nproc)
    return Xs, st, ok, its, J


def reduce_ring(Y, rm, M):
    """Re-represent (a, lam, rho_h, We) with M modes on a curvature-adapted map."""
    pts, _ = ring_curve_points(Y, rm)
    s, sm, th = curve_theta(pts)
    rm2 = design_map(sm, th, s[-1], K=min(48, M // 2), ell=0.3, alpha=1.0)
    a, rho = fit_ring(pts, rm2, M)
    M0 = len(Y) - 3
    return np.concatenate([a, [Y[M0], rho, Y[M0 + 2]]]), rm2


def refine(X, rm, st, rht, Mmax, nproc=None):
    """Remap when res_hi > RES_TOL or the coefficient tail > TAIL_MAX (and M < Mmax): 1.35 M
    modes on a fresh map, re-solved at fixed (r_h, Q).  Returns (X, rm, st, J), None when
    no remap is needed, False when it failed."""
    M = len(X) - 4
    res_hi, tl = st.res_hi(X[M + 2], X[M]), float(np.abs(X[int(0.85 * M):M]).max())
    if (res_hi <= RES_TOL and tl <= TAIL_MAX) or M >= Mmax:
        return None
    M2 = min(Mmax, int(np.ceil(1.35 * M / 4) * 4))
    log(f"   refine: M {M} -> {M2} (res_hi {res_hi:.1e}, tail {tl:.0e})")
    try:
        Yr, rm2 = reduce_ring(X[:-1], rm, M2)
        Xr = np.append(Yr, X[-1])
        Pr = ring_panels(Xr[:M2], rm2, M2, c_h=C_H, tol=1e-11)
        Xr, st2, ok, _, J = solve_fixed_Q(Xr, rm2, Pr, rht, nproc)
    except Exception as ex:  # noqa: BLE001
        log(f"   refine failed: {ex}")
        return False
    return (Xr, rm2, st2, J) if ok else False


# ---------------------------------------------------------------------------
# tangent, bordered determinant, records

def weights(X, ref, w):
    M = len(X) - 4
    W = np.full(M + 4, w['shape'])
    W[M] = w['lam'] / ref['lam']
    W[M + 1] = w['shape'] / ref['rho_h']
    W[M + 2] = w['We'] / abs(X[M + 2])
    W[M + 3] = w['Q'] / ref['Q']
    return W


def tangent(J, W, ref):
    """Null vector of the ring rows, unit in |W x|, oriented by (W^2 t).ref > 0."""
    t = np.linalg.svd(J / W)[2][-1] / W
    return t if (W ** 2 * t) @ ref > 0 else -t


def phys(rec, M, d=1):
    """Orientation vector from a record's tangent (lam, We, Q components), along d."""
    v = np.zeros(M + 4)
    v[M], v[M + 2], v[M + 3] = rec['dlam_ds'], rec['dWe_ds'], rec['dQ_ds']
    return d * v


def bordered(J, W, tau_s):
    """Sign of det[J; (W^2 tau_s)^T] and its smallest singular value times that sign."""
    B = np.vstack([J, W ** 2 * tau_s])
    sgn = float(np.linalg.slogdet(B)[0])
    return sgn, sgn * float(np.linalg.svd(B, compute_uv=False)[-1])


def z_dip(Y, rm):
    """Minimum of z(t) between the first and last local maximum (z_max for a single hump)."""
    z = ring_curve_points(Y, rm, n=4001)[0][:, 1]
    mx = np.nonzero((z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:]))[0] + 1
    return float(z[mx[0]:mx[-1] + 1].min()) if len(mx) else float(z.min())


def make_record(X, st, rm, J, W, tau, d, s, **extra):
    """Record with the rings_up.json keys and the sheet fields; tau along the direction d
    of travel (the stored tangent and the det sign refer to increasing s_arc)."""
    M = len(X) - 4
    Y, Q = X[:-1], float(X[-1])
    g = st.g
    We, lam = float(Y[M + 2]), float(Y[M])
    _, tht0 = ring_theta(Y[:M], np.array([0.0]))
    k_waist = -tht0[0] / (g.L * rm.Sp(np.array([0.0]))[0])
    rate, needed = decay_rate(Y[:M])
    R, a_tube = 0.5 * (g.r_h + g.r_o), 0.5 * (g.r_o - g.r_h)
    tau_s = (d or 1) * tau
    sgn, sb = bordered(J, W, tau_s)
    rec = dict(r_h=float(g.r_h), We=We, lam=lam, r_o=float(g.r_o), z_max=ring_z_max(Y, rm),
               circulation=float(st.circulation), kappa1_waist=float(k_waist),
               H_waist_rh=float((k_waist - 1 / g.r_h) * g.r_h), u_max=float(np.sqrt(st.q.max())),
               kmax=float(np.abs(g.kappa1).max()), res_hi=st.res_hi(We, lam),
               tail=float(np.abs(Y[int(0.85 * M):M]).max()), F=float(np.abs(st.F(We, lam)[:M + 1]).max()),
               M=M, N=st.P.N, decay=float(rate), modes_needed=int(needed), a=Y[:M].tolist(),
               rho_h=float(Y[M + 1]), map_b=rm.b.tolist(), breaks=st.P.breaks.tolist(),
               Q=Q, Qhat=float(2 * Q / R ** 2), R=float(R),
               u_waist=float(C.chebval(-1.0, st.P.rule.V2C @ st.gamma[:st.P.p])),
               We2D=float(We * st.circulation ** 2 / (4 * np.pi ** 2 * a_tube)), z_dip=z_dip(Y, rm),
               det_sign=sgn, sigma_min=float(np.linalg.svd(J[:, :-1], compute_uv=False)[-1]), sigma_b=sb,
               dQ_ds=float(tau_s[-1]), dWe_ds=float(tau_s[M + 2]), dlam_ds=float(tau_s[M]),
               dir=d, s_arc=float(d * s))
    rec.update(extra)
    return rec


def state_from(rec):
    X = np.concatenate([rec['a'], [rec['lam'], rec['rho_h'], rec['We'], rec.get('Q', 0.0)]])
    rm, P = RingMap(rec['map_b']), RingPanels(np.array(rec['breaks']))
    st = RingState(X[:-4], X[-3], rm, P, Q=float(X[-1]))
    return X, rm, st


def locate_branch(A, B, d, rm, rht, nproc, iters=3):
    """Regula falsi on the signed smallest singular value f of the bordered Jacobian between
    accepted points A and B (dicts X, J, tau, W, s, f), along the arclength from A."""
    lo, hi, best = A, B, None
    M = len(A['X']) - 4
    for _ in range(iters):
        s = lo['s'] + (hi['s'] - lo['s']) * lo['f'] / (lo['f'] - hi['f'])
        Xp = A['X'] + (s - A['s']) * A['tau']
        row = A['W'] ** 2 * A['tau']
        try:
            P = ring_panels(Xp[:M], rm, M, c_h=C_H, tol=1e-11)
            X, st, ok, its, nJ, _ = newton(Xp, rm, P, rht, (row, float(row @ Xp)), A['J'], nproc=nproc)
            if not ok:
                break
            J = jac_rows(X, st, rm, st.P, rht, nproc)
        except Exception as ex:  # noqa: BLE001
            log(f"   locate BRANCH: {ex}")
            break
        W = weights(X, A['ref'], A['w'])
        tau = tangent(J, W, A['tau'])
        best = dict(X=X, st=st, J=J, tau=tau, W=W, s=s, f=bordered(J, W, d * tau)[1], its=its, nJ=nJ)
        log(f"   locate BRANCH: s={s:.5f} Q={X[-1]:+.6e} We={X[M + 2]:.9f} f={best['f']:+.3e}")
        if np.sign(best['f']) == np.sign(lo['f']):
            lo = best
        else:
            hi = best
    return best


# ---------------------------------------------------------------------------
# driver

def options(argv):
    o = {a[2:].split('=', 1)[0]: a.split('=', 1)[1] for a in argv if a.startswith('--') and '=' in a}
    pos = [a for a in argv if not a.startswith('--')]
    w = dict(We=1.0, Q=1.0, lam=1.0, shape=0.1)
    w.update({k: float(v) for k, v in (kv.split(':') for kv in o.get('w', '').split(',') if kv)})
    num = lambda k, dflt: float(o[k]) if o.get(k, '') != '' else dflt  # noqa: E731
    return dict(src=pos[0], idx=int(pos[1]), out=pos[2], M=num('M', None),
                dirs=[int(x) for x in o.get('dirs', '-1,+1').split(',')], ds0=num('ds0', 0.1),
                dQ0=num('dQ0', None), dsmax=num('dsmax', 0.5), nmax=int(num('nmax', 40)),
                tcap=num('tcap', 30), stall=num('stall', 15), stop_We=num('stop-We', 5.0),
                qmin=num('qmin', None), qmax=num('qmax', None), w=w, Mmax=int(num('Mmax', 128)),
                Q0=num('Q0', None))


def main(argv):
    o = options(argv)
    out, w, nproc = o['out'], o['w'], default_nproc()
    meta_path = out + '.meta.json'
    recs = json.load(open(out)) if os.path.exists(out) else []
    start = next((r for r in recs if r['dir'] == 0), None)
    p0 = json.load(open(o['src']))[o['idx']]
    Q0 = p0.get('Q', 0.0) if o['Q0'] is None else o['Q0']
    if start is None:              # a meta file left from a lost scan does not count
        meta = dict(runs=[], status={}, src=o['src'], idx=o['idx'], r_h=float(p0['r_h']))
    else:
        if (abs(start['r_h'] - p0['r_h']) > 1e-8 * max(1.0, p0['r_h'])
                or abs(start['Q'] - Q0) > 1e-9 * max(1.0, abs(Q0))):
            sys.exit(f"{out} continues a scan from r_h={start['r_h']:.10g} Q={start['Q']:+.10g}, not "
                     f"{o['src']}:{o['idx']} at r_h={p0['r_h']:.10g} Q={Q0:+.10g}; use another output")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else dict(runs=[], status={})
        for k, v in (('src', o['src']), ('idx', o['idx']), ('r_h', float(p0['r_h']))):
            meta.setdefault(k, v)
    meta['runs'].append(dict(argv=argv, started=time.strftime('%Y-%m-%d %H:%M:%S'), nproc=nproc))
    status, rht = meta['status'], float(meta['r_h'])

    def save():
        recs.sort(key=lambda r: r['s_arc'])
        save_json(recs, out)
        meta.update(ref=ref, options=o, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(meta, meta_path)

    cache = None
    if start is None:
        M = int(o['M']) if o['M'] is not None else (len(p0['a']) if 'Q' in p0 else 64)
        Y0 = np.concatenate([p0['a'], [p0['lam'], p0['rho_h'], p0['We']]])
        rm = RingMap(p0['map_b'])
        if len(p0['a']) != M:
            Y0, rm = reduce_ring(Y0, rm, M)
        X = np.append(Y0, Q0)
        P = ring_panels(X[:-4], rm, M, c_h=C_H, tol=1e-11)
        log(f"start {o['src']}:{o['idx']} r_h={rht:.6g} Q={X[-1]:+.6g} We={p0['We']:.10f} "
            f"M {len(p0['a'])}->{M} N={P.N} nproc={nproc}")
        X, st, ok, its, J = solve_fixed_Q(X, rm, P, rht, nproc)
        log(f"re-solve ok={ok} its={its} We={X[M + 2]:.10f} (stored {p0['We']:.10f}) lam={X[M]:+.10f} "
            f"circ={st.circulation:+.8f} (stored {p0['circulation']:+.8f})")
        if not ok:
            sys.exit(1)
        for _ in range(4):
            r = refine(X, rm, st, rht, o['Mmax'], nproc)
            if not r:
                break
            X, rm, st, J = r
        M = len(X) - 4
        R0 = 0.5 * (st.g.r_h + st.g.r_o)
        ref = dict(lam=max(1.0, abs(X[M])), rho_h=float(X[M + 1]), Q=0.5 * R0 ** 2)
        W = weights(X, ref, w)
        eQ = np.zeros(M + 4)
        eQ[-1] = 1.0
        tau = tangent(J, W, eQ)
        start = make_record(X, st, rm, J, W, tau, 0, 0.0, its=its, nJ=1, ds=None, flags=[])
        recs.append(start)
        cache = (X, rm, st, J)
        log(f"start: M={M} res_hi={start['res_hi']:.1e} dWe/dQ={tau[M + 2] / tau[-1]:+.6g} "
            f"dlam/dQ={tau[M] / tau[-1]:+.6g} sigma_min={start['sigma_min']:.3e} det={start['det_sign']:+.0f} "
            f"Q_ref={ref['Q']:.5g}")
    else:
        ref = dict(lam=max(1.0, abs(start['lam'])), rho_h=start['rho_h'], Q=0.5 * start['R'] ** 2)
    save()

    for d in o['dirs']:
        key = f"{d:+d}"
        if status.get(key, {}).get('reason') in FINAL:
            log(f"dir {key}: finished earlier ({status[key]['reason']})")
            continue
        mine = sorted((r for r in recs if r['dir'] == d and 'BRANCH' not in r['flags']),
                      key=lambda r: abs(r['s_arc']))
        if len(mine) >= o['nmax']:
            status[key] = dict(reason='nmax', n=len(mine))
            save()
            log(f"dir {key}: {len(mine)} points, nmax reached")
            continue
        base = mine[-1] if mine else start
        if base is start and cache is not None:
            X, rm, st, J = cache
        else:
            X, rm, st = state_from(base)
            J = jac_rows(X, st, rm, st.P, rht, nproc)
        M = len(X) - 4
        W = weights(X, ref, w)
        tau = tangent(J, W, phys(base, M, d))
        dQ0 = o['dQ0'] if o['dQ0'] is not None else (0.05 * rht ** 1.5 if rht <= 0.24 else None)
        ds_first = dQ0 / abs(tau[-1]) if dQ0 is not None else o['ds0']
        ds = base['ds'] if mine and base.get('ds') else ds_first
        ds_min = 1e-2 * min(ds_first, o['dsmax'])
        s = abs(base['s_arc'])
        prev = base
        cur = dict(X=X, J=J, tau=tau, W=W, s=s, f=base['sigma_b'], ref=ref, w=w)
        g0 = [i for i, r in enumerate(mine) if 'GAMMA0' in r['flags']]
        gamma_left = 3 - (len(mine) - 1 - g0[0]) if g0 else None
        n, easy, reason = len(mine), 0, None
        t_dir = t_acc = time.time()
        log(f"dir {key}: from s_arc={base['s_arc']:+.4f} Q={X[-1]:+.6e} We={X[M + 2]:.9f} ds={ds:.3g} n={n}")
        while reason is None:
            if n >= o['nmax']:
                reason = 'nmax'
                break
            if gamma_left is not None and gamma_left <= 0:
                reason = 'Gamma0'
                break
            if time.time() - t_dir > 60 * o['tcap']:
                reason = 'tcap'
                break
            if time.time() - t_acc > 60 * o['stall']:
                reason = 'stall'
                break
            R = 0.5 * (st.g.r_h + st.g.r_o)
            dst = min(ds, o['dsmax'], 0.1 * R ** 2 / max(abs(tau[-1]), 1e-300))
            Xp = X + dst * tau
            row = W ** 2 * tau
            try:
                P = ring_panels(Xp[:M], rm, M, c_h=C_H, tol=1e-11)
                Xn, stn, ok, its, nJ, _ = newton(Xp, rm, P, rht, (row, float(row @ Xp)), J, nproc=nproc)
                jump = float(np.linalg.norm(W * (Xn - Xp))) / dst
                why = (f"not converged (its={its} nJ={nJ})" if not ok else
                       f"corrector landed {jump:.2f} ds from the predictor" if jump > 0.5 else None)
                if why is None:
                    Jn = jac_rows(Xn, stn, rm, stn.P, rht, nproc)
                    Wn = weights(Xn, ref, w)
                    tn = tangent(Jn, Wn, tau)
                    turn = float((Wn * tau) @ (Wn * tn)) / np.linalg.norm(Wn * tau)
                    if turn < 0.5:
                        why = f"tangent turned by {np.degrees(np.arccos(turn)):.0f} deg"
            except Exception as ex:  # noqa: BLE001
                why = f"error {ex}"
            if why:
                ds, easy = 0.5 * dst, 0
                log(f"   dir {key}: {why}; ds -> {ds:.3g}")
                if ds < ds_min:
                    reason = 'step'
                continue
            t_acc = time.time()
            X, st, s, J, W, flags = Xn, stn, s + dst, Jn, Wn, []
            sgn, f = bordered(J, W, d * tn)
            if sgn != prev['det_sign']:
                # in the representation of the previous point, so before any remap
                flags.append('DET_SIGN')
                loc = locate_branch(cur, dict(X=X, J=J, tau=tn, W=W, s=s, f=f), d, rm, rht, nproc)
                if loc is not None:
                    recs.append(make_record(loc['X'], loc['st'], rm, loc['J'], loc['W'], loc['tau'], d,
                                            loc['s'], its=loc['its'], nJ=loc['nJ'], ds=None, flags=['BRANCH']))
                else:
                    flags.append('BRANCH')
            tau = tn
            r = refine(X, rm, st, rht, o['Mmax'], nproc)
            if r:
                ori = dict(dlam_ds=tau[M], dWe_ds=tau[M + 2], dQ_ds=tau[-1])
                X, rm, st, J = r
                M = len(X) - 4
                W = weights(X, ref, w)
                tau = tangent(J, W, phys(ori, M))
                flags.append('rebased')
            elif r is False:
                flags.append('remap failed')
                reason = 'resolution'
            rec = make_record(X, st, rm, J, W, tau, d, s, its=its, nJ=nJ, ds=dst, flags=flags)
            for k, flag in (('dQ_ds', 'FOLD_Q'), ('dWe_ds', 'EXTREMUM_We'), ('lam', 'LAM0'),
                            ('circulation', 'GAMMA0')):
                if np.sign(rec[k]) != np.sign(prev[k]):
                    flags.append(flag)
            if gamma_left is not None:
                gamma_left -= 1
            elif 'GAMMA0' in flags:
                gamma_left = 3
            if rec['z_dip'] < 0.1 * rec['z_max']:
                flags.append('PINCH')
                reason = 'pinch'
            if rec['res_hi'] > RES_MAX or rec['tail'] > TAIL_MAX:
                reason = 'resolution'
            if rec['We'] > o['stop_We']:
                reason = 'We'
            if (o['qmin'] is not None and rec['Q'] < o['qmin']) or (o['qmax'] is not None and rec['Q'] > o['qmax']):
                reason = 'Q range'
            recs.append(rec)
            n += 1
            prev, cur = rec, dict(X=X, J=J, tau=tau, W=W, s=s, f=rec['sigma_b'], ref=ref, w=w)
            status[key] = dict(reason=reason, n=n)
            save()
            log(f" dir {key} {n:3d} s={s:.4f} Q={rec['Q']:+.6e} Qhat={rec['Qhat']:+.4f} We={rec['We']:.9f} "
                f"lam={rec['lam']:+.7f} circ={rec['circulation']:+.6f} We2D={rec['We2D']:.4f} "
                f"uw={rec['u_waist']:+.4f} r_o={rec['r_o']:.5f} zmax={rec['z_max']:.5f} "
                f"zdip={rec['z_dip'] / rec['z_max']:.3f} kmax={rec['kmax']:.2f} res_hi={rec['res_hi']:.1e} "
                f"tail={rec['tail']:.0e} M={M} N={rec['N']} det={rec['det_sign']:+.0f} "
                f"smin={rec['sigma_min']:.3e} its={its} nJ={nJ} ds={dst:.3g} jump={jump:.2f} "
                f"turn={np.degrees(np.arccos(min(1.0, turn))):.0f}deg {' '.join(flags)}")
            easy = easy + 1 if its <= EASY_IT and nJ == 0 else 0
            ds = 1.5 * dst if easy >= 3 else dst
            easy = 0 if easy >= 3 else easy
        status[key] = dict(reason=reason, n=n)
        save()
        log(f"dir {key}: stop ({reason}) after {n} points")
    log("done")


if __name__ == '__main__':
    main(sys.argv[1:])
