"""Toroidal (ring) bubbles: geometry, flow and residual.

Cross-section in the meridian half plane, symmetric under z -> -z.  The upper
half is parametrised by t in [0, pi] from the inner rim (r_h, 0) (tangent
pointing up) over the top to the outer rim (r_o, 0) (tangent pointing down):

    x'(t) = L S'(t) (cos theta, sin theta),
    theta(t) = pi/2 - t + sum_{k=1}^M a_k sin(k t),

the lower half is the mirror image.  S'(t) = (1 + sum_k b_k cos kt)/pi is a
parameter map (int_0^pi S' = 1, so L is the arclength of the upper half).
Closure requires z(pi) = 0, imposed as an equation.  L is fixed by the volume
4 pi/3 and x(0) = (L rho_h, 0).

Outward normal n = (-sin theta, cos theta), H = -theta_t/(L S') - sin theta / r
(at the inner rim kappa_2 = -1/r_h < 0: the neck is saddle shaped).

Flow: vortex-sheet formulation (bubbles.sheet) with flux constant Q (Q = 0:
no net flux through the hole, the rings that can connect continuously to the
touching bubble).
"""
import numpy as np
from numpy.polynomial import chebyshev as C

from .quad import PanelRule, cheb_eval
from .sheet import SheetOperator

VOL = 4.0 * np.pi / 3.0


class RingMap:
    def __init__(self, b=()):
        self.b = np.asarray(b, float)

    def Sp(self, t):
        t = np.asarray(t, float)
        k = np.arange(1, len(self.b) + 1)
        if len(k) == 0:
            return np.full_like(t, 1 / np.pi)
        return (1 + np.cos(np.multiply.outer(t, k)) @ self.b) / np.pi

    def S(self, t):
        t = np.asarray(t, float)
        k = np.arange(1, len(self.b) + 1)
        if len(k) == 0:
            return t / np.pi
        return (t + np.sin(np.multiply.outer(t, k)) @ (self.b / k)) / np.pi

    def inverse(self, sigma):
        sigma = np.asarray(sigma, float)
        t = np.pi * sigma
        for _ in range(80):
            dt = (self.S(t) - sigma) / self.Sp(t)
            t = np.clip(t - dt, 0, np.pi)
            if np.max(np.abs(dt)) < 1e-15:
                break
        return t


def ring_theta(a, t):
    t = np.asarray(t, float)
    k = np.arange(1, len(a) + 1)
    arg = np.multiply.outer(t, k)
    return np.pi / 2 - t + np.sin(arg) @ a, -1.0 + np.cos(arg) @ (k * a)


class RingPanels:
    def __init__(self, breaks, p=16):
        self.breaks = np.asarray(breaks, float)
        self.rule = PanelRule(p)
        self.p = p
        self.np = len(self.breaks) - 1
        self.a, self.b = self.breaks[:-1], self.breaks[1:]
        self.h = self.b - self.a
        self.t = (self.a + self.b)[:, None] / 2 + self.h[:, None] / 2 * self.rule.x[None, :]
        self.w = self.h[:, None] / 2 * self.rule.w[None, :]
        self.tf, self.wf = self.t.ravel(), self.w.ravel()
        self.N = self.np * p

    def diff(self, f):
        F = f.reshape(f.shape[:-1] + (self.np, self.p))
        return (np.einsum('ij,...pj->...pi', self.rule.Dm, F) * (2 / self.h)[:, None]).reshape(f.shape)


def ring_panels(a, rmap, M, p=16, tol=1e-13, c_h=2.0, max_panels=3000):
    rule = PanelRule(p)
    n0 = max(4, int(np.ceil(np.pi / (c_h / max(M, 1)))))
    breaks = list(np.linspace(0, np.pi, n0 + 1))

    def ok(lo, hi):
        t = (lo + hi) / 2 + (hi - lo) / 2 * rule.x
        th, tht = ring_theta(a, t)
        sp = rmap.Sp(t)
        for f, sc in ((th, 1.0), (tht * (hi - lo), 1.0), (sp, np.abs(sp).max())):
            c = rule.V2C @ f
            if np.abs(c[-3:]).max() > tol * sc:
                return False
        return True
    changed = True
    while changed:
        changed = False
        new = [breaks[0]]
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            if not ok(lo, hi) and len(breaks) < max_panels:
                new.append((lo + hi) / 2)
                changed = True
            new.append(hi)
        breaks = new
    changed = True
    while changed:
        changed = False
        h = np.diff(breaks)
        new = [breaks[0]]
        for k in range(len(h)):
            nb = [h[q] for q in (k - 1, k + 1) if 0 <= q < len(h)]
            if nb and h[k] > 2.0 * min(nb):
                new.append((breaks[k] + breaks[k + 1]) / 2)
                changed = True
            new.append(breaks[k + 1])
        breaks = new
    return RingPanels(np.array(breaks), p)


class RingGeometry:
    def __init__(self, a, rho_h, rmap, panels):
        P = panels
        self.a, self.rho_h, self.rmap, self.P = np.asarray(a, float), rho_h, rmap, P
        t = P.tf
        th, tht = ring_theta(self.a, t)
        Sp = rmap.Sp(t)
        self.theta, self.theta_t, self.Sp = th, tht, Sp
        xp1 = np.stack([Sp * np.cos(th), Sp * np.sin(th)]).reshape(2, P.np, P.p)
        B1 = np.einsum('kj,cpj->cpk', P.rule.V2C, xp1)
        A1 = np.zeros((2, P.np, P.p + 1))
        start = np.array([rho_h, 0.0])
        for q in range(P.np):
            for c in range(2):
                A1[c, q] = C.chebint(B1[c, q], lbnd=-1, scl=P.h[q] / 2)
                A1[c, q, 0] += start[c]
            start = np.array([C.chebval(1.0, A1[0, q]), C.chebval(1.0, A1[1, q])])
        self.z_end1 = start[1]          # closure residual (unit length)
        self.r_end1 = start[0]
        x1 = cheb_eval(A1, P.rule.x[None, None, :])
        r1, z1 = x1[0].ravel(), x1[1].ravel()
        # rho_h^2 int dz1 = rho_h^2 z_end1 vanishes on a closed curve; subtracting it keeps an
        # open intermediate curve from counting a disk of radius ~rho_h (rho_h ~ 7e3 at
        # r_h = 454), which made the closure row nearly invisible to Newton
        V1 = -2 * np.pi * np.sum(P.wf * (r1 ** 2 - rho_h ** 2) * Sp * np.sin(th))
        self.V1 = V1
        L = (VOL / V1) ** (1 / 3) if V1 > 0 else np.nan
        self.L = L
        self.A, self.B = L * A1, L * B1
        self.r, self.z = L * r1, L * z1
        self.c = L * Sp
        self.r_h, self.r_o = L * rho_h, L * self.r_end1
        self.kappa1 = -tht / self.c
        self.kappa2 = -np.sin(th) / self.r
        self.H = self.kappa1 + self.kappa2
        self.ds = P.wf * self.c


class RingState:
    def __init__(self, a, rho_h, rmap, panels, Q=0.0, sheet_opts=None):
        self.g = g = RingGeometry(a, rho_h, rmap, panels)
        self.P = panels
        self.M = len(a)
        self.op = SheetOperator(g.A, g.B, panels.h, panels.rule, **(sheet_opts or {}))
        self.gamma = self.op.solve(Q - 0.5 * self.op.r ** 2)
        self.q = self.gamma ** 2
        k = np.arange(self.M + 1)
        self.proj = (2 / np.pi) * np.cos(np.outer(k, panels.tf)) * panels.wf[None, :]
        k_hi = np.arange(self.M + 1, 2 * self.M + 1)
        self.proj_hi = (2 / np.pi) * np.cos(np.outer(k_hi, panels.tf)) * panels.wf[None, :]
        self.circulation = 2 * np.sum(self.gamma * g.ds)

    def residual_nodal(self, We, lam):
        return self.g.H - 0.5 * We * self.q - lam

    def F(self, We, lam):
        return np.append(self.proj @ self.residual_nodal(We, lam), self.g.z_end1)

    def res_hi(self, We, lam):
        return float(np.abs(self.proj_hi @ self.residual_nodal(We, lam)).max())


# ---------------------------------------------------------------------------
# Newton solver (finite-difference Jacobian, parallel columns)

_CTX = {}


def default_nproc():
    """Worker processes for finite-difference Jacobians (env BUBBLES_NPROC)."""
    import os
    return int(os.environ.get('BUBBLES_NPROC', min(4, os.cpu_count() or 1)))


def _constraint(Y, st, target):
    """The extra equation: hole radius r_h = target (a number), or, for
    pseudo-arclength continuation, a linear condition w.Y = c (target = ('lin', w, c))."""
    if target is None:
        return []
    if isinstance(target, tuple) and target[0] == 'lin':
        return [float(np.dot(target[1], Y) - target[2])]
    return [st.g.r_h - target]


def _ring_F(Y, b, breaks, r_h_target, Q):
    M = len(Y) - 3
    a, lam, rho_h, We = Y[:M], Y[M], Y[M + 1], Y[M + 2]
    st = RingState(a, rho_h, RingMap(b), RingPanels(breaks), Q=Q)
    F = st.F(We, lam)
    return np.concatenate([F, _constraint(Y, st, r_h_target)]), st


def _col(args):
    Y, j, h, F0 = args
    c = _CTX
    # relative step for rho_h (= r_h/L, from ~3e-6 at the smallest necks to ~3e4 at the
    # largest rings): an absolute 1e-7 is a 3% step at r_h ~ 1e-5 and drowns in the
    # round-off of F at r_h ~ 250; for the shape coefficients it stays absolute below 1
    M = len(Y) - 3
    h = h * (abs(float(Y[j])) if j == M + 1 else max(1.0, abs(float(Y[j]))))
    Yp = Y.copy()
    Yp[j] += h
    Fp, _ = _ring_F(Yp, c['b'], c['breaks'], c['rht'], c['Q'])
    if F0 is not None:                       # forward difference
        return (Fp - F0) / h
    Ym = Y.copy()
    Ym[j] -= h
    Fm, _ = _ring_F(Ym, c['b'], c['breaks'], c['rht'], c['Q'])
    return (Fp - Fm) / (2 * h)


def _init(ctx):
    _CTX.update(ctx)


def ring_jacobian(Y, st, rmap, panels, r_h_target, Q=0.0, h=1e-7, nproc=None, pool=None,
                  F0=None):
    """Jacobian of (F, constraint) w.r.t. (a, lam, rho_h, We); a and rho_h by
    finite differences (parallel; forward if F0 is given), lam and We
    analytically."""
    M = len(Y) - 3
    cols = list(range(M)) + [M + 1]
    own = pool is None
    if own:
        from multiprocessing import Pool
        pool = Pool(nproc or default_nproc(), initializer=_init,
                    initargs=(dict(b=rmap.b, breaks=panels.breaks, rht=r_h_target, Q=Q),))
    res = pool.map(_col, [(Y, j, h, F0) for j in cols], chunksize=1)
    if own:
        pool.close()
    nrow = M + 2 + (1 if r_h_target is not None else 0)
    J = np.zeros((nrow, M + 3))
    for j, c in zip(cols, res):
        J[:, j] = c
    J[:M + 1, M] = st.proj @ (-np.ones(panels.N))
    J[:M + 1, M + 2] = st.proj @ (-0.5 * st.q)
    if isinstance(r_h_target, tuple) and r_h_target[0] == 'lin':
        J[-1, :] = r_h_target[1]            # exact row for a linear constraint
    return J


def residual_floor(r_o):
    """Round-off floor of max|F|: the sheet equation V[gamma] = Q - r^2/2 loses about
    log10(r_o^2) digits to cancellation (observed plateau 2.4e-11 at r_o = 166)."""
    return 3e-15 * max(1.0, float(r_o) ** 2)


def closure_ok(F, M, scale=1.0):
    """Closure row (z_end1, unit length) and the constraint row at round-off level; these
    rows have no r^2 cancellation, so residual_floor does not apply to them."""
    ok = abs(F[M + 1]) < 1e-13
    if len(F) > M + 2:
        ok = ok and abs(F[M + 2]) < 1e-11 * scale
    return bool(ok)


def ring_newton(Y, rmap, panels, r_h_target, Q=0.0, maxit=25, tol=1e-11, nproc=None,
                J=None, log=print, pool=None):
    """Damped (quasi-)Newton; the Jacobian is recomputed when the contraction
    is poor.  Returns (Y, state, history, J)."""
    from multiprocessing import Pool
    own = pool is None
    if own:
        pool = Pool(nproc or default_nproc(), initializer=_init,
                    initargs=(dict(b=rmap.b, breaks=panels.breaks, rht=r_h_target, Q=Q),))
    F, st = _ring_F(Y, rmap.b, panels.breaks, r_h_target, Q)
    hist = []
    fresh = False
    floor = residual_floor(st.g.r_o)
    M = len(Y) - 3
    if isinstance(r_h_target, tuple):
        scale = max(1.0, abs(float(r_h_target[2])))
    else:
        scale = max(1.0, abs(float(r_h_target or 1.0)))
    try:
        for it in range(maxit):
            if hist and hist[-1][0] < floor and hist[-1][2]:
                break
            if J is None or (len(hist) >= 2 and hist[-1][0] > 0.5 * hist[-2][0] and not fresh):
                J = ring_jacobian(Y, st, rmap, panels, r_h_target, Q, pool=pool, F0=F)
                fresh = True
            else:
                fresh = False
            dY = np.linalg.lstsq(J, -F, rcond=None)[0]
            f0 = np.linalg.norm(F)
            step = 1.0
            for _ in range(8):
                Yt = Y + step * dY
                try:
                    Ft, stt = _ring_F(Yt, rmap.b, panels.breaks, r_h_target, Q)
                    ok = np.all(np.isfinite(Ft)) and np.linalg.norm(Ft) < (1 - 0.1 * step) * f0 + 1e-13
                except Exception:
                    ok = False
                if ok:
                    break
                step *= 0.5
            else:
                if not fresh:
                    J = ring_jacobian(Y, st, rmap, panels, r_h_target, Q, pool=pool, F0=F)
                    fresh = True
                    continue
                log(f"  line search failed at it {it}")
                break
            Y, F, st = Yt, Ft, stt
            # history: (max |Bernoulli rows|, |step|, closure and constraint rows at round-off)
            hist.append((float(np.abs(F[:M + 1]).max()), float(step * np.abs(dY).max()), closure_ok(F, M, scale)))
            log(f"  it {it}: |F|={hist[-1][0]:.3e} |dY|={hist[-1][1]:.3e} step={step} "
                f"We={Y[M + 2]:.12f} lam={Y[M]:+.10f} r_h={st.g.r_h:.6f} circ={st.circulation:+.8f}")
            if hist[-1][2] and (hist[-1][1] < tol or hist[-1][0] < 1e-13):
                break
    finally:
        if own:
            pool.close()
    return Y, st, hist, J


def ring_z_max(Y, rmap, n=4001):
    """Half-thickness max z of the cross-section (not just over the nodes):
    dense sampling plus a parabolic refinement at the maximum."""
    pts, _ = ring_curve_points(Y, rmap, n=n)
    z = pts[:, 1]
    k = int(np.clip(np.argmax(z), 1, len(z) - 2))
    zm, z0, zp = z[k - 1], z[k], z[k + 1]
    den = zm - 2 * z0 + zp
    return float(z0 - 0.125 * (zp - zm) ** 2 / den) if den < 0 else float(z0)


def ring_curve_points(Y, rmap, n=20000):
    """Dense polyline of the upper half of the cross-section."""
    M = len(Y) - 3
    a, rho_h = Y[:M], Y[M + 1]
    P = RingPanels(np.linspace(0, np.pi, 257))
    g = RingGeometry(a, rho_h, rmap, P)
    t = np.linspace(0, np.pi, n)
    idx = np.clip(np.searchsorted(P.b, t), 0, P.np - 1)
    u = np.clip((2 * t - P.a[idx] - P.b[idx]) / P.h[idx], -1, 1)
    x = cheb_eval(g.A[:, idx, :], u[None, :, None])[..., 0]
    return np.stack([x[0], x[1]], axis=1), g


def branch_max(r_h, We, half=2):
    """Maximum of We along a sampled branch: quartic in log r_h through the 2*half + 1
    samples around the largest one (the samples are ~15% apart, so the largest
    sample can sit 3% off the maximum).  Returns (r_h_max, We_max)."""
    r_h, We = np.asarray(r_h, float), np.asarray(We, float)
    o = np.argsort(r_h)
    r_h, We = r_h[o], We[o]
    i = int(np.clip(np.argmax(We), half, len(We) - half - 1))
    x = np.log(r_h[i - half:i + half + 1])
    c = np.polyfit(x - x[half], We[i - half:i + half + 1], 2 * half)
    xs = np.linspace(x[1], x[-2], 20001) - x[half]
    k = int(np.argmax(np.polyval(c, xs)))
    return float(np.exp(xs[k] + x[half])), float(np.polyval(c, xs[k]))
