"""Continue the branch of zero-flux ring bubbles in the hole radius r_h.

Starting from a converged ring (npz with Y, b, breaks), step r_h
geometrically towards r_min (factor < 1) or upwards (factor > 1).  The
parameter map and the number of modes are re-adapted when the Fourier tail
or the truncation indicator grows.

Usage: python scripts/run_rings.py start out.json factor r_stop [M_max] [--neck] [--tail=1e-13] [--thin]

start is a state file (npz with Y, b, breaks) or a point of an earlier run,
given as run.json:index.  With --neck the parameter map comes from
ring_init.neck_map (clustering adapted to the catenoid neck, redesigned
whenever r_h has shrunk by 1.3), and resolution is judged by the Fourier tail
of the tangent-angle coefficients (max |a_k|, k > 0.85 M, below TAIL_TOL)
instead of the residual indicator res_hi, which the 1/r_h curvatures of the
neck inflate.

With --thin (large rings, r_h >~ 100) the predictor rescales the last ring by the
thin-ring asymptotics instead of extrapolating: the cross-section is a slightly
deformed circle of radius a = (2/(3 pi R))^(1/2), so the shape coefficients are
kept, rho_h = r_h/L scales like r_h/a, and We and lam move with the leading-order
formulas.  Polynomial or tangent predictors drifted into spurious near-solutions
(|F| ~ 1e-6) beyond r_h ~ 450; the rescaled guess starts at |F| ~ 1e-7.
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.geometry import decay_rate  # noqa: E402
from bubbles.ring import (RingGeometry, RingMap, RingPanels, RingState, _init, default_nproc,  # noqa: E402
                          residual_floor, ring_curve_points, ring_jacobian, ring_newton, ring_panels, ring_theta,
                          ring_z_max)
from bubbles.ring_init import curve_theta, design_map, fit_ring, neck_map  # noqa: E402

NECK = '--neck' in sys.argv
THIN = '--thin' in sys.argv
# --tail=eps: judge resolution by the Fourier tail (max |a_k|, k > 0.85 M) < eps also with the
# curvature map (default there: res_hi < RES_TOL); with --neck the tail test is always used
TAIL = [float(x.split('=')[1]) for x in sys.argv if x.startswith('--tail=')]
argv = [x for x in sys.argv if not x.startswith('--')]
start, out, factor, r_stop = argv[1], argv[2], float(argv[3]), float(argv[4])
M_MAX = int(argv[5]) if len(argv) > 5 else 400
C_H, RES_TOL, TAIL_TOL, M_NECK0 = 6.0, 1e-6, 1e-12, 160
AHEAD, REMAP_AT = 3.0, 1.2   # neck map designed for r_h/AHEAD, redesigned once r_h < REMAP_AT * that
r_map = [None]                # r_design of the current neck map


def save_json(obj, path):
    """Write atomically, so readers never see a half-written file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f)
    os.replace(path + '.tmp', path)


def log(m):
    print(m, flush=True)


def remap(Y, rm, M_new, ell=0.3, alpha=1.0, K=96):
    M = len(Y) - 3
    if NECK:
        r_now = RingGeometry(Y[:M], Y[M + 1], rm, RingPanels(np.linspace(0, np.pi, 513))).r_h
        r_map[0] = r_now / AHEAD if factor < 1 else r_now
        rm2, _ = neck_map(Y, rm, f=0.3, ell=0.3, alpha=0.5, K=384, width=0.02, r_design=r_map[0])
        pts, g = ring_curve_points(Y, rm, n=400000)
        a, rho = fit_ring(pts, rm2, M_new, n=32768)
    else:
        pts, g = ring_curve_points(Y, rm)
        s, sm, th = curve_theta(pts)
        rm2 = design_map(sm, th, s[-1], K=K, ell=ell, alpha=alpha)
        a, rho = fit_ring(pts, rm2, M_new)
    return np.concatenate([a, [Y[M], rho, Y[M + 2]]]), rm2


def thin_ring(R):
    """Leading-order thin ring of centre radius R: tube radius a, We and lam."""
    a = np.sqrt(2 / (3 * np.pi * R))
    L = np.log(8 * R / a)
    We = 4 * a * (L - 2) ** 2 / (3 * R ** 2 * (L - 2.5))
    G = np.pi * R / (L - 2)
    return a, We, 1 / a - We * G ** 2 / (8 * np.pi ** 2 * a ** 2)


def thin_predictor(Y, st, target):
    """Rescale a large ring to hole radius target (see --thin in the docstring)."""
    M = len(Y) - 3
    a0, W0, l0 = thin_ring(0.5 * (st.g.r_h + st.g.r_o))
    R1 = target + a0
    for _ in range(5):
        R1 = target + thin_ring(R1)[0]
    a1, W1, l1 = thin_ring(R1)
    Yp = Y.copy()
    Yp[M + 1] = Y[M + 1] * (target / st.g.r_h) * (a0 / a1)
    Yp[M + 2] = Y[M + 2] * W1 / W0
    Yp[M] = Y[M] + (l1 - l0)
    return Yp


def tail(Y):
    M = len(Y) - 3
    return float(np.abs(Y[int(0.85 * M):M]).max())


def resolved(rec, Y):
    if TAIL:
        return tail(Y) < TAIL[0]
    return tail(Y) < TAIL_TOL if NECK else rec['res_hi'] < RES_TOL


def record(Y, st, rm, hist):
    M = len(Y) - 3
    g = st.g
    _, tht0 = ring_theta(Y[:M], np.array([0.0]))
    k_waist = -tht0[0] / (g.L * rm.Sp(np.array([0.0]))[0])
    rate, needed = decay_rate(Y[:M])
    return dict(r_h=float(g.r_h), We=float(Y[M + 2]), lam=float(Y[M]), r_o=float(g.r_o),
                z_max=ring_z_max(Y, rm), circulation=float(st.circulation),
                kappa1_waist=float(k_waist), H_waist_rh=float((k_waist - 1 / g.r_h) * g.r_h),
                u_max=float(np.sqrt(st.q.max())), kmax=float(np.abs(g.kappa1).max()),
                res_hi=st.res_hi(Y[M + 2], Y[M]), tail=tail(Y), F=hist[-1][0] if hist else None,
                M=M, N=st.P.N, decay=float(rate), modes_needed=int(needed),
                a=Y[:M].tolist(), rho_h=float(Y[M + 1]), map_b=rm.b.tolist(),
                breaks=st.P.breaks.tolist())


# the work runs only in the main process: on macOS (spawn) the pool's workers import this
# script, and must not repeat it
if __name__ == '__main__':
    if os.environ.get('BUBBLES_MP_START'):   # e.g. spawn, to reproduce macOS on Linux
        import multiprocessing
        multiprocessing.set_start_method(os.environ['BUBBLES_MP_START'])
    if '.json:' in start:
        fn, idx = start.rsplit(':', 1)
        p0 = json.load(open(fn))[int(idx)]
        Y = np.concatenate([p0['a'], [p0['lam'], p0['rho_h'], p0['We']]])
        rm, P = RingMap(p0['map_b']), RingPanels(np.array(p0['breaks']))
    else:
        d = np.load(start)  # (a refined start can be passed as *_last.npz)
        Y, rm = d['Y'], RingMap(d['b'])
        P = RingPanels(d['breaks'])
    M = len(Y) - 3
    st = RingState(Y[:M], Y[M + 1], rm, P)
    r_fix = st.g.r_h


J_res = [None]   # Jacobian of the last re-solve in resolve() (same map and M as its result)


def resolve(Y, rm, st, target, force=False):
    """Remap and increase M until resolved (res_hi, or the tail with --neck).

    force: remap once even if resolved (with --neck, after r_h has shrunk)."""
    M = len(Y) - 3
    for it in range(8):
        rec = record(Y, st, rm, [(0, 0)])
        if not (force and it == 0) and (resolved(rec, Y) or M >= M_MAX):
            return Y, rm, st, True
        if NECK and first_neck[0]:
            M_new = M_NECK0          # first switch to a neck map: its own starting size
            first_neck[0] = False
        else:
            # a forced redesign is first tried at the same M: the old map's tail says
            # nothing about the new one
            grow = 1.0 if (force and it == 0) else (1.25 if NECK else 1.35)
            M_new = min(M_MAX, int(np.ceil(grow * M / 4) * 4))
        log(f"   refine: M {M} -> {M_new} (res_hi {rec['res_hi']:.1e}, tail {rec['tail']:.1e})")
        Yr, rm2 = remap(Y, rm, M_new)
        P = ring_panels(Yr[:M_new], rm2, M_new, c_h=C_H, tol=1e-11)
        Yr, st2, hist, J_res[0] = ring_newton(Yr, rm2, P, target, log=lambda m: log('     ' + m))
        if not hist or not hist[-1][2] or (hist[-1][0] > max(1e-9, residual_floor(st2.g.r_o))
                                            and hist[-1][1] > 1e-10):
            log("   re-convergence failed")
            return Y, rm, st, False
        Y, rm, st, M = Yr, rm2, st2, M_new
        log(f"      We={Y[-1]:.12f} lam={Y[M]:+.10f} N={st.P.N}")
    return Y, rm, st, True


first_neck = [NECK]   # the start state is re-represented at M_NECK0 (then grown as needed)

if __name__ == '__main__':
    Y, rm, st, ok = resolve(Y, rm, st, r_fix, force=NECK)
    points = [record(Y, st, rm, [(0, 0)])]
    save_json(points, out)
    np.savez(out.replace('.json', '_last.npz'), Y=Y, b=rm.b, breaks=st.P.breaks)
    log(f"start r_h={points[0]['r_h']:.6f} We={points[0]['We']:.12f}")
    hist_p, hist_Y = [np.log(st.g.r_h)], [Y.copy()]
    J = None
    r_h = st.g.r_h
    t0 = time.time()
    step_fac = factor
    while (factor < 1 and r_h > r_stop) or (factor > 1 and r_h < r_stop):
        target = r_h * step_fac
        M = len(Y) - 3
        n = min(len(hist_p), 3)
        lp = np.log(target)
        if THIN:
            Yp = thin_predictor(Y, st, target)
        elif J is None or J.shape[1] != len(Y):
            # Jacobian at the current solution (needed for the tangent predictor)
            P0 = st.P
            pool = Pool(default_nproc(), initializer=_init,
                        initargs=(dict(b=rm.b, breaks=P0.breaks, rht=r_h, Q=0.0),))
            F0 = np.concatenate([st.F(Y[M + 2], Y[M]), [st.g.r_h - r_h]])
            J = ring_jacobian(Y, st, rm, P0, r_h, pool=pool, F0=F0)
            pool.close()
        if not THIN:
            # tangent: d/dlog(r_h) of the solution = r_h J^{-1} e_constraint
            e = np.zeros(J.shape[0])
            e[-1] = 1.0
            tang = r_h * np.linalg.lstsq(J, e, rcond=None)[0]
            if n <= 2:
                Yp = Y + (lp - np.log(r_h)) * tang
            else:
                Ps = np.array(hist_p[-n:])
                Ys = np.array(hist_Y[-n:])
                coef = np.linalg.solve(np.vander(Ps - Ps[-1], n), Ys)
                Yp = (np.vander(np.array([lp - Ps[-1]]), n) @ coef)[0]
        P = ring_panels(Yp[:M], rm, M, c_h=C_H, tol=1e-11)
        pool = Pool(default_nproc(), initializer=_init, initargs=(dict(b=rm.b, breaks=P.breaks, rht=target, Q=0.0),))
        try:
            # st stays the state of the accepted Y until the step is accepted, so a retry
            # rebuilds J and the tangent at a consistent (Y, rm, st)
            Yn, st_n, hist, Jn = ring_newton(Yp, rm, P, target,
                                             J=J if (J is not None and J.shape[1] == len(Yp)) else None,
                                             log=lambda m: log('     ' + m), pool=pool, maxit=25)
        except Exception as e:  # noqa: BLE001
            hist, Yn = [(np.inf, np.inf)], None
            log(f"   newton error {e}")
        pool.close()
        ok = (Yn is not None and hist and hist[-1][2]
              and (hist[-1][0] < max(1e-9, residual_floor(st_n.g.r_o)) or hist[-1][1] < 1e-10))
        if not ok:
            step_fac = np.sqrt(step_fac)
            J = None
            log(f"   failed at r_h={target:.5g} (|F|={hist[-1][0] if hist else None}); step factor -> {step_fac:.4f}")
            if abs(np.log(step_fac)) < 0.005:
                log("   step too small, stopping")
                break
            continue
        Y, J, st = Yn, Jn, st_n
        M = len(Y) - 3
        rec = record(Y, st, rm, hist)
        # resolution control (with --neck also redesign the map as the neck shrinks)
        shrunk = NECK and factor < 1 and rec['r_h'] < REMAP_AT * r_map[0]
        if not resolved(rec, Y) or shrunk:
            M_old, b_old = M, rm.b.copy()
            Y, rm, st, okr = resolve(Y, rm, st, target, force=shrunk)
            if not okr:
                log("   stopping (resolution)")
                break
            M = len(Y) - 3
            rec = record(Y, st, rm, hist)
            if M != M_old or not np.array_equal(b_old, rm.b):
                hist_p, hist_Y, J = [], [], J_res[0]
        points.append(rec)
        hist_p.append(np.log(rec['r_h']))
        hist_Y.append(Y.copy())
        r_h = rec['r_h']
        step_fac = factor
        log(f"{len(points) - 1:3d} r_h={r_h:.6e} We={rec['We']:.12f} lam={rec['lam']:+.10f} r_o={rec['r_o']:.6f} "
            f"zmax={rec['z_max']:.5f} circ={rec['circulation']:+.8f} H_w*r_h={rec['H_waist_rh']:+.4f} "
            f"M={rec['M']} N={rec['N']} res_hi={rec['res_hi']:.1e} tail={rec['tail']:.0e} t={time.time() - t0:.0f}s")
        save_json(points, out)
        np.savez(out.replace('.json', '_last.npz'), Y=Y, b=rm.b, breaks=st.P.breaks)
