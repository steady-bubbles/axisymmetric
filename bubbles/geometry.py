"""Fore-aft symmetric axisymmetric bubble surfaces.

The generating curve in the meridian half plane (r >= 0, z) runs from the
front pole (0, z_N) through the equator (r_eq, 0) to the rear pole (0, -z_N).
It is parametrised by t in [0, pi] through its tangent angle

    x'(t) = L S'(t) (cos theta(t), sin theta(t)),
    theta(t) = -t + sum_{j=1}^M a_j sin(2 j t),

where S: [0, pi] -> [0, 1] is a fixed parameter map (normalised arclength,
S(t) = t/pi by default) and L the total meridional length.  The unit sphere
is a = 0, L = pi.  The parity of theta + t (odd about t = 0 and t = pi/2)
encodes axis regularity and fore-aft symmetry; the curve closes
automatically.  L is fixed by the volume constraint |B| = 4 pi / 3.

Outward normal n = (-sin theta, cos theta), tangent T = (cos theta, sin theta).
Principal curvatures (positive for convex bodies):
    kappa_1 = -theta_t / (L S'),    kappa_2 = -sin(theta) / r,
H = kappa_1 + kappa_2 (= 2 on the unit sphere).
"""
import numpy as np
from numpy.polynomial import chebyshev as C

from .quad import PanelRule, cheb_eval

VOL_BALL = 4.0 * np.pi / 3.0


class ParamMap:
    """S(t) = (t + sum_k b_k sin(2kt)) / pi; must satisfy S' > 0."""

    def __init__(self, b=()):
        self.b = np.asarray(b, dtype=float)

    def Sp(self, t):
        k = np.arange(1, len(self.b) + 1)
        if len(k) == 0:
            return np.full_like(np.asarray(t, float), 1.0 / np.pi)
        return (1.0 + np.cos(2 * np.multiply.outer(t, k)) @ (2 * k * self.b)) / np.pi

    def S(self, t):
        k = np.arange(1, len(self.b) + 1)
        if len(k) == 0:
            return np.asarray(t, float) / np.pi
        return (t + np.sin(2 * np.multiply.outer(t, k)) @ self.b) / np.pi


def theta_eval(a, t):
    """theta, theta_t at t for coefficients a (length M)."""
    t = np.asarray(t, float)
    j = np.arange(1, len(a) + 1)
    arg = 2 * np.multiply.outer(t, j)
    th = -t + np.sin(arg) @ a
    tht = -1.0 + np.cos(arg) @ (2 * j * a)
    return th, tht


class Panels:
    """Panels on [0, pi], mirror symmetric about pi/2.

    ``breaks_half`` are increasing breakpoints in [0, pi/2] including both ends.
    Nodes are stored panel by panel; node index k and N-1-k are mirror images
    (t -> pi - t).  The first N/2 nodes lie in [0, pi/2].
    """

    def __init__(self, breaks_half, p=16):
        bh = np.asarray(breaks_half, float)
        assert abs(bh[0]) < 1e-15 and abs(bh[-1] - np.pi / 2) < 1e-14
        bh[0], bh[-1] = 0.0, np.pi / 2
        self.breaks_half = bh
        self.breaks = np.concatenate([bh, np.pi - bh[-2::-1]])
        self.rule = PanelRule(p)
        self.p = p
        self.np = len(self.breaks) - 1
        self.nph = self.np // 2
        a, b = self.breaks[:-1], self.breaks[1:]
        self.a, self.b = a, b
        self.h = b - a
        self.mid = (a + b) / 2
        self.t = self.mid[:, None] + self.h[:, None] / 2 * self.rule.x[None, :]
        # enforce exact mirror symmetry of the nodes
        tf = self.t.ravel()
        N = tf.size
        tf[N // 2:] = np.pi - tf[:N // 2][::-1]
        self.t = tf.reshape(self.np, p)
        self.w = self.h[:, None] / 2 * self.rule.w[None, :]
        self.N = N
        self.Nh = N // 2
        self.tf = self.t.ravel()
        self.wf = self.w.ravel()

    @classmethod
    def uniform(cls, n_half, p=16, pole_refine=0):
        bh = np.linspace(0, np.pi / 2, n_half + 1)
        if pole_refine:
            extra = bh[1] * 0.5 ** np.arange(1, pole_refine + 1)
            bh = np.sort(np.concatenate([bh, extra]))
        return cls(bh, p)

    def diff(self, f):
        """d/dt of nodal data f (shape (..., N)), panelwise spectral."""
        F = f.reshape(f.shape[:-1] + (self.np, self.p))
        dF = np.einsum('ij,...pj->...pi', self.rule.Dm, F) * (2.0 / self.h)[:, None]
        return dF.reshape(f.shape)

    def cumint(self, f):
        """int_0^t f dt at the nodes (f nodal, shape (..., N))."""
        F = f.reshape(f.shape[:-1] + (self.np, self.p))
        hh = (self.h / 2)[:, None]
        inner = np.einsum('ij,...pj->...pi', self.rule.Cm, F) * hh
        tot = np.einsum('j,...pj->...p', self.rule.w, F) * self.h / 2
        start = np.concatenate([np.zeros(f.shape[:-1] + (1,)),
                                np.cumsum(tot, axis=-1)[..., :-1]], axis=-1)
        return (inner + start[..., None]).reshape(f.shape)

    def integrate(self, f):
        return f @ self.wf


class Geometry:
    """All nodal geometric quantities of the curve on given panels."""

    def __init__(self, a, panels, pmap=None):
        self.a = np.asarray(a, float)
        self.panels = P = panels
        self.pmap = pmap if pmap is not None else ParamMap()
        t = P.tf
        th, tht = theta_eval(self.a, t)
        Sp = self.pmap.Sp(t)
        self.theta, self.theta_t, self.Sp = th, tht, Sp
        ct, st = np.cos(th), np.sin(th)
        # unit-length curve (L = 1)
        r1 = P.cumint(Sp * ct)
        z1 = P.cumint(Sp * st)
        # fix z(pi/2) = 0; pi/2 is a breakpoint so the value is the half integral
        zhalf = P.integrate(np.where(np.arange(P.N) < P.Nh, Sp * st, 0.0))
        z1 = z1 - zhalf
        self.z1N = -zhalf
        V1 = -np.pi * P.integrate(r1 ** 2 * Sp * st)
        self.V1 = V1
        L = (VOL_BALL / V1) ** (1.0 / 3.0)
        self.L = L
        self.r, self.z = L * r1, L * z1
        self.r1, self.z1 = r1, z1
        self.c = L * Sp                       # |x'(t)|
        self.T = np.stack([ct, st])
        self.n = np.stack([-st, ct])
        self.xp = self.c * self.T             # x'(t)
        self.kappa1 = -tht / self.c
        self.kappa2 = -st / self.r
        self.H = self.kappa1 + self.kappa2
        self.ds = P.wf * self.c               # arclength weights
        self.zN = L * self.z1N
        # Chebyshev data per panel for off-node evaluation:
        #   B: coefficients of x'(t) (degree p-1) in local u
        #   A: coefficients of x(t) (degree p), A' = (h/2) B
        p = P.p
        xp_pan = self.xp.reshape(2, P.np, p)
        B = np.einsum('kj,cpj->cpk', P.rule.V2C, xp_pan)
        A = np.zeros((2, P.np, p + 1))
        start = np.stack([self.r.reshape(P.np, p), self.z.reshape(P.np, p)])
        for c_ in range(2):
            for q in range(P.np):
                A[c_, q] = C.chebint(B[c_, q], lbnd=-1, scl=P.h[q] / 2)
        # integration constants from the left panel ends
        x_start = np.zeros((2, P.np))
        x_start[0, 0] = 0.0
        x_start[1, 0] = self.zN
        for q in range(1, P.np):
            # chebint with lbnd=-1 vanishes at u=-1; the value at u=1 is the panel integral
            x_start[:, q] = x_start[:, q - 1] + np.array(
                [C.chebval(1.0, A[0, q - 1]), C.chebval(1.0, A[1, q - 1])])
        A[:, :, 0] += x_start
        self.A, self.B = A, B
        # replace nodal positions by the (identical up to rounding) polynomial values
        xn = cheb_eval(A, P.rule.x[None, None, :])
        self.r_cheb_err = np.max(np.abs(xn - start))
        self.r = xn[0].ravel()
        self.z = xn[1].ravel()
        self.kappa2 = -st / self.r
        self.H = self.kappa1 + self.kappa2

    # derived diagnostics -------------------------------------------------
    def volume(self):
        P = self.panels
        return -np.pi * P.integrate(self.r ** 2 * self.c * self.T[1])

    def area(self):
        return 2 * np.pi * self.panels.integrate(self.r * self.c)

    def eval_theta(self, t):
        return theta_eval(self.a, t)[0]


def fit_theta_coeffs(theta_of_t, M, n=None):
    """Sine coefficients a_j (j=1..M) of theta(t) + t in sin(2jt).

    theta_of_t is evaluated on midpoints of [0, pi/2]; the periodic
    trapezoidal rule is spectrally accurate for the odd periodic extension.
    """
    n = n or max(4 * M, 64)
    t = (np.arange(1, n + 1) - 0.5) * np.pi / (2 * n)
    f = theta_of_t(t) + t
    j = np.arange(1, M + 1)
    return (2.0 / n) * (np.sin(2 * np.outer(j, t)) @ f)


def curve_to_theta(rfun, zfun, drfun, dzfun, pmap=None, npts=4000):
    """Return a function t -> theta(t) for a parametric fore-aft symmetric
    curve psi in [0, pi] -> (r, z), with t defined through the parameter map
    S(t) = s/L (s arclength).  Uses Gauss-Legendre arclength and Newton
    inversion."""
    from numpy.polynomial.legendre import leggauss
    pmap = pmap or ParamMap()
    xg, wg = leggauss(40)

    def arclen_comp(psi, nseg=64):
        psi = np.atleast_1d(psi).astype(float)
        tot = np.zeros_like(psi)
        for k in range(nseg):
            lo, hi = psi * k / nseg, psi * (k + 1) / nseg
            s = lo[:, None] + (hi - lo)[:, None] / 2 * (xg[None, :] + 1)
            tot += (hi - lo) / 2 * (np.hypot(drfun(s), dzfun(s)) @ wg)
        return tot

    Ltot = arclen_comp(np.array([np.pi]))[0]

    def theta_of_t(t):
        t = np.asarray(t, float)
        target = pmap.S(t) * Ltot
        psi = np.pi * pmap.S(t)  # initial guess
        for _ in range(30):
            f = arclen_comp(psi) - target
            fp = np.hypot(drfun(psi), dzfun(psi))
            dpsi = f / fp
            psi = psi - dpsi
            if np.max(np.abs(dpsi)) < 1e-15:
                break
        return np.arctan2(dzfun(psi), drfun(psi))

    return theta_of_t


def spheroid_coeffs(aspect, M, pmap=None):
    """Tangent-angle coefficients of a spheroid with aspect ratio
    (equatorial radius)/(polar half axis) = aspect (oblate if > 1)."""
    a_, c_ = aspect, 1.0
    th = curve_to_theta(lambda p: a_ * np.sin(p), lambda p: c_ * np.cos(p),
                        lambda p: a_ * np.cos(p), lambda p: -c_ * np.sin(p), pmap)

    def th_unwrapped(t):
        v = th(t)
        # theta is continuous in (-pi, 0] on [0, pi/2]
        return np.where(v > 0.5, v - 2 * np.pi, v)

    return fit_theta_coeffs(th_unwrapped, M)


def adaptive_panels(a, pmap=None, p=16, tol=1e-13, h_max=None, n_init=4,
                    max_panels=4000):
    """Panels on [0, pi/2] (mirrored).

    Starts from uniform panels of length <= h_max (in t), then bisects panels
    on which the trailing Chebyshev coefficients of theta, h*theta_t or S'
    exceed ``tol``; finally neighbouring panel lengths are limited to a ratio
    of 2.  ``h_max ~ 2/M`` makes the panels resolve the Fourier modes
    sin(2jt), j <= M (needed for an accurate Jacobian).
    """
    from .quad import PanelRule
    pmap = pmap or ParamMap()
    rule = PanelRule(p)
    n0 = n_init
    if h_max is not None:
        n0 = max(n0, int(np.ceil((np.pi / 2) / h_max)))
    breaks = list(np.linspace(0, np.pi / 2, n0 + 1))

    def ok(lo, hi):
        t = (lo + hi) / 2 + (hi - lo) / 2 * rule.x
        th, tht = theta_eval(a, t)
        sp = pmap.Sp(t)
        for f, scale in ((th, 1.0), (tht * (hi - lo), 1.0), (sp, np.abs(sp).max())):
            c = rule.V2C @ f
            if np.abs(c[-3:]).max() > tol * scale:
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
    return Panels(np.array(breaks), p)


def invert_map(pmap, sigma, t0=None):
    """t with S(t) = sigma (Newton; S is increasing)."""
    sigma = np.asarray(sigma, float)
    t = np.pi * sigma if t0 is None else np.array(t0, float)
    for _ in range(60):
        dt = (pmap.S(t) - sigma) / pmap.Sp(t)
        t = t - dt
        if np.max(np.abs(dt)) < 1e-15:
            break
    return t


def reparametrize(a, pmap_old, pmap_new, M_new, n=None):
    """Coefficients a_new representing the same curve with pmap_new."""
    n = n or max(4 * M_new, 256)
    tt = (np.arange(1, n + 1) - 0.5) * np.pi / (2 * n)
    to = invert_map(pmap_old, pmap_new.S(tt))
    th = theta_eval(a, to)[0]
    f = th + tt
    j = np.arange(1, M_new + 1)
    return (2.0 / n) * (np.sin(2 * np.outer(j, tt)) @ f)


def decay_rate(c, lo=1e-11, hi=1e-3):
    """Exponential decay rate (per mode) of |c_j| fitted where lo<|c_j|<hi
    (relative to max|c|).  Returns (rate, modes needed for relative 1e-13)."""
    c = np.abs(np.asarray(c))
    cm = c.max()
    if cm == 0:
        return np.inf, 1
    rel = c / cm
    j = np.arange(1, len(c) + 1)
    sel = (rel > lo) & (rel < hi)
    # envelope: running max from the tail
    env = np.maximum.accumulate(rel[::-1])[::-1]
    nz = np.nonzero(env > 1e-13)[0]
    needed = int(nz.max() + 1) if len(nz) else 1
    if sel.sum() < 3:
        return np.inf, needed
    A = np.vstack([np.ones(sel.sum()), j[sel]]).T
    coef, *_ = np.linalg.lstsq(A, np.log(env[sel]), rcond=None)
    return -coef[1], needed


def map_family(beta):
    """S'(t) = (1 + beta_1 cos 2t + beta_2 cos 4t + ...)/pi."""
    beta = np.atleast_1d(np.asarray(beta, float))
    k = np.arange(1, len(beta) + 1)
    return ParamMap(beta / (2 * k))


def optimise_map(a, pmap_old, n_par=1, M_test=None, bounds=0.97):
    """Choose map_family parameters maximising the Fourier decay rate of the
    tangent angle of the curve given by (a, pmap_old)."""
    from scipy.optimize import minimize, minimize_scalar
    M_test = M_test or max(2 * len(a), 128)

    def objective(beta):
        pm = map_family(beta)
        tchk = np.linspace(0, np.pi / 2, 721)
        if pm.Sp(tchk).min() <= 0.02 / np.pi:
            return 1e3
        an = reparametrize(a, pmap_old, pm, M_test)
        rate, _ = decay_rate(an)
        return -rate if np.isfinite(rate) else -50.0

    if n_par == 1:
        res = minimize_scalar(objective, bounds=(-bounds, bounds), method='bounded',
                              options=dict(xatol=1e-3))
        best = np.array([res.x])
    else:
        x0 = np.zeros(n_par)
        x0[0] = minimize_scalar(lambda b: objective([b] + [0] * (n_par - 1)),
                                bounds=(-bounds, bounds), method='bounded').x
        res = minimize(objective, x0, method='Nelder-Mead',
                       options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
        best = res.x
    pm = map_family(best)
    an = reparametrize(a, pmap_old, pm, M_test)
    rate, needed = decay_rate(an)
    return pm, best, rate, needed
