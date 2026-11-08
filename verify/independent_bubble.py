#!/usr/bin/env python
"""Independent solver for steady axisymmetric vacuum bubbles in a uniform potential stream.

Written as an independent cross-check of the boundary-integral code in bubbles/ (nothing is imported
or copied from it).  Run from the repository root:

    python verify/independent_bubble.py            # full run, about 1 minute on 4 cores

Results go to results/checks/independent_bubble.json (all long-double values are stored as decimal
strings with ~19 significant digits).

Problem (unit equivalent radius, unit stream speed along +z, bubble frame)
--------------------------------------------------------------------------
    u = grad(phi), Laplace(phi) = 0 outside B, dphi/dn = 0 on S, u -> e_z at infinity,
    H - (We/2)|u|^2 = lambda on S,  |B| = 4 pi / 3,   H = k1 + k2 (= 2 on the unit sphere),
    shapes axisymmetric and symmetric under z -> -z.

Method (deliberately different from the tangent-angle / Nystrom-BIE / Galerkin code under test)
---------------------------------------------------------------------------------------------
* Shape: a stretched radial graph over the polar angle,
        r = F(t) sin t,  z = sigma F(t) cos t,  F(t) = exp( sum_{k<N} c_k cos 2kt ),  t in [0, pi/2],
  with a fixed stretch sigma (chosen close to 1/aspect ratio).  Rays from the centre are the same in the
  stretched and physical planes, so this is a star-shaped representation; with sigma ~ 1/chi the
  cosine coefficients of log F decay geometrically (to ~1e-19 by k ~ 40 at the fold).
  Curvature (k1 meridional, k2 = n_r / r) and the volume (an exact Fejer-type rule for
  int F^3 sin t dt) are evaluated analytically from the series.
* Flow: Stokes stream function Psi = r^2/2 + sum_i q_i [G(r,z;rho_i,zeta_i) + G(r,z;rho_i,-zeta_i)]
  (method of fundamental solutions, ring vortices in z-symmetric pairs).  G is the ring-vortex stream
  function in Lamb's Landen form G = (R1+R2)/(2 pi) [K(k) - E(k)], k = (R1-R2)/(R1+R2); K, E and
  K-E come from an arithmetic-geometric-mean iteration written here (K-E summed without
  cancellation), with an analytic gradient.  The ring positions are Z(t_i - i tau) where
  Z(t) = r(t) + i z(t) is the analytically continued meridian (for a spheroid this is a confocal
  ellipse; tau < atanh(sigma) keeps it inside), i.e. sources follow the local geometry.  The Dirichlet
  condition Psi = 0 is imposed by weighted least squares (weight 1/r: the vector potential Psi/r) at
  2*ns boundary points; the surface speed is u = (1/r) dPsi/dn.
* Precision: matrix, curvature, volume and residuals are evaluated in 80-bit long double; the least
  squares problem is solved with a double-precision pivoted QR plus long-double iterative refinement.
  Exact-solution tests (sphere, aspect-4 oblate spheroid) give surface speeds to ~1e-17.
* Equations: collocation of H - (We/2)u^2 - lambda at t_j = (j+1/2) pi/(2N), j < N, the volume, and one
  parameter equation (z_pole, We or lambda prescribed).  Unknowns c_0..c_{N-1}, lambda, We.  Newton with
  a forward-difference Jacobian in double precision, then chord iterations with long-double residuals.
* Branch: natural-parameter continuation in z_pole from the sphere, re-expressing the shape in a new
  sigma as the bubble flattens.  The fold is located by a secant iteration on dWe/dz_pole (implicit
  differentiation with the Jacobian), then We(z_pole) is sampled on 9 Chebyshev nodes around it and
  We_max is the maximum of the interpolant (insensitive to the poorly conditioned fold location).

Checks performed by the driver
------------------------------
* ring kernel vs mpmath (Legendre-form formula); MFS vs exact potential flow past the unit sphere
  (|u| = 1.5 sin theta) and past an aspect-4 oblate spheroid (oblate spheroidal coordinates), the
  latter both in the matched stretch and in a mismatched stretch (non-trivial F, 64 modes);
* small We: d(chi)/dWe -> 9/32 and d(lambda)/dWe -> -3/4 (polynomial extrapolation from We <= 8e-4);
* fold: 6 resolutions (N = 32..80, sigma = 0.92..1.12 x aspect, 59..119 ring pairs, three source
  offsets, 2x and 3x oversampling); an a-posteriori residual on 1000 off-collocation points with a
  different MFS; the same fold search in plain double precision;
* reference points: solved at the printed (rounded) z_pole, and -- sharper, needing no z_pole --
  at the printed We, comparing lambda there (the We offset implied by the lambda mismatch is reported);
* dimple onset (lambda = 0) at 3 resolutions, We at z_pole = 0.5 and 0.7.
"""
import os
import sys

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import time
import warnings
import numpy as np
import scipy.linalg as sla
import mpmath as mp

warnings.filterwarnings('ignore', category=RuntimeWarning)
LD = np.longdouble
PI_LD = LD('3.141592653589793238462643383279502884')
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(REPO, 'results', 'checks', 'independent_bubble.json')

# points: three bubbles of an earlier run of scripts/convergence.py (three discretisations of the main
# solver agree there to 1e-13), whose seed branch is not kept; each is defined by its z_pole, at which
# this solver solves, so they remain valid points of the branch.
REFERENCE = dict(
    fold=dict(We_max=1.6176521718921, chi=3.86456346, z_pole=0.39258860, lam=0.18858122, r_eq=1.51718355),
    points=[dict(z_pole='0.89072', We=0.531630365991074, lam=1.581887304553710),
            dict(z_pole='0.60347', We=1.428377354367535, lam=0.667804377059941),
            dict(z_pole='0.40338', We=1.617183045702324, lam=0.208609325157767)],
    dimple=dict(We=1.5620469061382, z_pole=0.2715119267342))


def _pi(dt):
    return PI_LD if dt == LD else np.pi


def numstr(v):
    """shortest round-trip decimal string of a float64 / long double value"""
    if isinstance(v, (float, int)):
        return repr(float(v))
    return np.format_float_scientific(v, unique=True)


def tomp(v):
    return mp.mpf(numstr(v))


# ----------------------------------------------------------------------------------------------
# complete elliptic integrals (AGM) and the ring-vortex stream function
# ----------------------------------------------------------------------------------------------
def agm_kme(k, kp):
    """K(k), E(k), K(k)-E(k) for modulus k (complementary modulus kp supplied, to avoid 1-k^2).
    AGM: a0=1, b0=kp, c0=k, c_{n+1} = c_n^2/(4 a_{n+1}); K = pi/(2 a_inf), K-E = K sum 2^(n-1) c_n^2."""
    dt = k.dtype.type
    eps = np.finfo(k.dtype).eps
    a = np.ones_like(k)
    b = kp.copy()
    c = k.copy()
    s = c * c / 2
    p = dt(0.5)
    for _ in range(64):
        an = (a + b) / 2
        b = np.sqrt(a * b)
        c = c * c / (4 * an)
        a = an
        p = p * 2
        t = p * c * c
        s = s + t
        if np.all(t <= 1e-3 * eps * s) and np.max(c) < 0.1 * np.sqrt(eps):
            break
    K = _pi(k.dtype) / (2 * a)
    KmE = K * s
    return K, K - KmE, KmE


def ring_psi(r, z, rho, zeta, grad=True):
    """Stokes stream function of a unit ring vortex of radius rho at height zeta, and its gradient.
    psi = (R1+R2)/(2 pi) [K(k)-E(k)], k = (R1-R2)/(R1+R2), R1,2 = max/min distance to the ring."""
    pi = _pi(r.dtype)
    dz = z - zeta
    R1 = np.sqrt((r + rho) ** 2 + dz * dz)
    R2 = np.sqrt((r - rho) ** 2 + dz * dz)
    S = R1 + R2
    D = 4 * r * rho / S                       # = R1 - R2 without cancellation
    k = D / S
    kp = 2 * np.sqrt(R1 * R2) / S
    K, E, KmE = agm_kme(k, kp)
    psi = S * KmE / (2 * pi)
    if not grad:
        return psi
    P = R1 * R2
    psi_z = dz / (2 * pi) * (KmE * S / P - 2 * r * rho * D * E / (P * P))
    psi_r = (KmE * (r * S - rho * D) / P + D * E * rho * (rho * rho - r * r + dz * dz) / (P * P)) / (2 * pi)
    return psi, psi_r, psi_z


def ring_pair(r, z, rho, zeta, grad=False):
    """matrix of the z-symmetric pairs of rings (rho_i, +-zeta_i) evaluated at points (r, z)"""
    R, Z, P, Q = r[:, None], z[:, None], rho[None, :], zeta[None, :]
    if not grad:
        return ring_psi(R, Z, P, Q, False) + ring_psi(R, Z, P, -Q, False)
    a = ring_psi(R, Z, P, Q)
    b = ring_psi(R, Z, P, -Q)
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


# ----------------------------------------------------------------------------------------------
# shape: stretched radial graph
# ----------------------------------------------------------------------------------------------
class Shape:
    def __init__(self, c, sigma, dt=LD):
        self.dt = dt
        self.c = np.asarray(c, dtype=dt)
        self.sigma = dt(sigma)
        self.k2 = 2 * np.arange(len(self.c)).astype(dt)

    def geom(self, t):
        t = np.asarray(t, dtype=self.dt)
        A = np.outer(t, self.k2)
        C, Sn = np.cos(A), np.sin(A)
        f = C @ self.c
        f1 = -(Sn @ (self.k2 * self.c))
        f2 = -(C @ (self.k2 ** 2 * self.c))
        F = np.exp(f)
        F1 = F * f1
        F2 = F * (f2 + f1 * f1)
        s, co, sg = np.sin(t), np.cos(t), self.sigma
        r = F * s
        r1 = F1 * s + F * co
        r2 = F2 * s + 2 * F1 * co - F * s
        z = sg * F * co
        z1 = sg * (F1 * co - F * s)
        z2 = sg * (F2 * co - 2 * F1 * s - F * co)
        sp = np.sqrt(r1 * r1 + z1 * z1)
        nr, nz = -z1 / sp, r1 / sp              # outward normal
        k1 = (z1 * r2 - r1 * z2) / sp ** 3      # meridional curvature (+1 on the unit sphere)
        k2 = nr / r                             # azimuthal curvature
        return dict(r=r, z=z, nr=nr, nz=nz, H=k1 + k2, k1=k1, k2=k2)


def source_points(c, sigma, ns, tau):
    """ring positions Z(t_i - i tau) on the continued meridian (double precision; fixed per solve).
    Modes with 2 k tau > 18 are dropped so that round-off in high modes is not amplified."""
    c = np.asarray(c, float)
    kmax = int(min(len(c), max(4, np.floor(9.0 / tau))))
    k2 = 2.0 * np.arange(kmax)
    ts = 0.5 * np.pi * (np.arange(ns) + 0.5) / ns
    tc = ts - 1j * tau
    F = np.exp(np.cos(np.outer(tc, k2)) @ c[:kmax])
    Z = F * np.sin(tc) + 1j * sigma * F * np.cos(tc)
    rho, zeta = Z.real.copy(), Z.imag.copy()
    # sanity: every ring strictly inside the body (ray test in the stretched plane)
    ang = np.arctan2(sigma * rho, zeta)
    Fb = np.exp(np.cos(np.outer(ang, 2.0 * np.arange(len(c)))) @ c)
    if not (np.all(rho > 0) and np.all(zeta > 0) and np.all(np.hypot(rho, zeta / sigma) < Fb)):
        raise RuntimeError('MFS source curve leaves the body; reduce tau')
    return rho, zeta


def fejer_sin(P, dt):
    """nodes t_j = (j+1/2) pi/(2P) and weights: sum w_j g(t_j) = int_0^{pi/2} g(t) sin t dt, exact for
    g = cos 2kt, k < P (g even, pi-periodic).  Uses int_0^{pi/2} cos 2kt sin t dt = 1/(1-4k^2)."""
    t = (np.arange(P) + dt(0.5)) / P * (_pi(dt) / 2)
    k = np.arange(1, P).astype(dt)
    w = (1 + 2 * (np.cos(2 * np.outer(t, k)) @ (1 / (1 - 4 * k * k)))) / P
    return t, w


# ----------------------------------------------------------------------------------------------
# MFS solve
# ----------------------------------------------------------------------------------------------
def mfs_solve(r, z, rho, zeta, nref):
    """weighted least squares for the ring strengths; returns q and max weighted residual |Psi/r|"""
    dt = r.dtype.type
    A = ring_pair(r, z, rho, zeta)
    w = 1 / r
    Aw = A * w[:, None]
    b = -(r * r / 2) * w
    sc = 1 / np.abs(Aw).max(axis=0)
    As = Aw * sc
    Q, R, piv = sla.qr(As.astype(float), mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > 1e-16 * d[0]))
    n = As.shape[1]

    def lsq(v):
        y = Q[:, :rank].T @ np.asarray(v, float)
        x = np.zeros(n)
        x[piv[:rank]] = sla.solve_triangular(R[:rank, :rank], y)
        return x

    q = lsq(b).astype(dt)
    for _ in range(nref):
        q = q + lsq(b - As @ q).astype(dt)
    res = float(np.abs(b - As @ q).max())
    return q * sc, res


def surface_speed(q, rho, zeta, g):
    """signed tangential surface speed (1/r) dPsi/dn"""
    _, pr, pz = ring_pair(g['r'], g['z'], rho, zeta, grad=True)
    Pr = g['r'] + pr @ q
    Pz = pz @ q
    return (Pr * g['nr'] + Pz * g['nz']) / g['r']


# ----------------------------------------------------------------------------------------------
# discretised bubble problem
# ----------------------------------------------------------------------------------------------
class Disc:
    """one discretisation: N shape modes / collocation points, stretch sigma, ns ring pairs at
    offset tau, 2*ns MFS boundary points"""

    def __init__(self, N, sigma, ns, tau, mcol=None):
        self.N, self.sigma, self.ns, self.tau = int(N), float(sigma), int(ns), float(tau)
        self.mcol = int(mcol or 2 * ns)
        self.grid = {}
        for dt in (np.float64, LD):
            pi = _pi(dt)
            tj = (np.arange(self.N) + dt(0.5)) / self.N * (pi / 2)
            tc = (np.arange(self.mcol) + dt(0.5)) / self.mcol * (pi / 2)
            tv, wv = fejer_sin(max(4 * self.N, 160), dt)
            Cv = np.cos(np.outer(tv, 2 * np.arange(self.N).astype(dt)))
            self.grid[dt] = (tj, tc, wv, Cv)

    def describe(self):
        return dict(N=self.N, sigma=self.sigma, ns=self.ns, tau=self.tau, mcol=self.mcol)

    def residual(self, x, mode, par, src, dt=LD, info=False):
        N = self.N
        x = np.asarray(x, dtype=dt)
        c, lam, We = x[:N], x[N], x[N + 1]
        tj, tc, wv, Cv = self.grid[dt]
        sh = Shape(c, self.sigma, dt)
        gc = sh.geom(tc)
        rho, zeta = src[0].astype(dt), src[1].astype(dt)
        q, lsq = mfs_solve(gc['r'], gc['z'], rho, zeta, nref=3 if dt == LD else 0)
        g = sh.geom(tj)
        u = surface_speed(q, rho, zeta, g)
        R = np.empty(N + 2, dtype=dt)
        R[:N] = g['H'] - We / 2 * u * u - lam
        R[N] = dt(self.sigma) * (wv @ np.exp(Cv @ c) ** 3) - 1
        if mode == 'zp':
            R[N + 1] = np.log(dt(self.sigma)) + c.sum() - np.log(dt(par))
        elif mode == 'We':
            R[N + 1] = We - dt(par)
        elif mode == 'lam':
            R[N + 1] = lam - dt(par)
        else:
            raise ValueError(mode)
        if info:
            return R, dict(lsq=lsq, u=u, g=g, q=q)
        return R

    def jacobian(self, x, mode, par, src, R0=None, h=1e-7):
        x = np.asarray(x, float)
        if R0 is None:
            R0 = self.residual(x, mode, par, src, np.float64)
        J = np.empty((len(x), len(x)))
        for j in range(len(x)):
            xp = x.copy()
            xp[j] += h
            J[:, j] = (self.residual(xp, mode, par, src, np.float64) - R0) / h
        return J


def parse_par(par):
    return LD(par) if isinstance(par, str) else LD(repr(float(par)))


def solve(D, x0, mode, par, src=None, J=None, verbose=False, ld=True):
    """double-precision Newton (forward-difference Jacobian, reused while it contracts well), then
    chord iterations with long-double residuals.  Sources are fixed from the starting shape."""
    t0 = time.time()
    N = D.N
    par_ld = parse_par(par)
    par_d = float(par_ld)
    x = np.array(x0, dtype=float)
    if src is None:
        src = source_points(x[:N], D.sigma, D.ns, D.tau)
    njac = 0
    prev = None
    fresh = False
    for it in range(40):
        R = D.residual(x, mode, par_d, src, np.float64)
        if J is None:
            J = D.jacobian(x, mode, par_d, src, R)
            njac += 1
            fresh = True
        dx = np.linalg.solve(J, -R)
        if prev is not None and not fresh and np.abs(dx).max() > 0.2 * prev and np.abs(dx).max() > 1e-10:
            J = D.jacobian(x, mode, par_d, src, R)
            njac += 1
            dx = np.linalg.solve(J, -R)
        fresh = False
        x = x + dx
        last, prev = prev, np.abs(dx).max()
        if verbose:
            print(f'    d-it {it}: |R|={np.abs(R).max():.2e} |dx|={prev:.2e}', flush=True)
        if prev < 1e-12 or (last is not None and prev < 1e-9 and prev > 0.5 * last):
            break        # converged, or stagnating at the double-precision noise floor
        if it == 39:
            raise RuntimeError('Newton (double) did not converge')
    if not ld:      # plain double precision (used only to show the double-precision floor)
        R, inf = D.residual(x, mode, par_d, src, np.float64, info=True)
        return x.astype(LD), dict(res_ld=float(np.abs(R).max()), lsq=inf['lsq'], njac=njac, ld_hist=[],
                                  time=time.time() - t0, src=src, J=J, umax=float(np.abs(inf['u']).max()))
    xl = x.astype(LD)
    hist = []
    for it in range(12):
        R = D.residual(xl, mode, par_ld, src, LD)
        dx = np.linalg.solve(J, -np.asarray(R, float))
        xl = xl + dx.astype(LD)
        hist.append((float(np.abs(R).max()), float(np.abs(dx).max())))
        if verbose:
            print(f'    ld-it {it}: |R|={hist[-1][0]:.2e} |dx|={hist[-1][1]:.2e}', flush=True)
        if hist[-1][1] < 2e-19 or (len(hist) >= 3 and hist[-1][1] > 0.3 * hist[-2][1]):
            break
    R, inf = D.residual(xl, mode, par_ld, src, LD, info=True)
    return xl, dict(res_ld=float(np.abs(R).max()), lsq=inf['lsq'], njac=njac, ld_hist=hist,
                    time=time.time() - t0, src=src, J=J, umax=float(np.abs(inf['u']).max()))


def props(x, D):
    N = D.N
    sh = Shape(np.asarray(x[:N], LD), D.sigma, LD)
    g = sh.geom(np.array([PI_LD / 2], dtype=LD))
    zp = LD(D.sigma) * np.exp(np.asarray(x[:N], LD).sum())
    req = g['r'][0]
    return dict(z_pole=zp, r_eq=req, chi=req / zp, lam=LD(x[N]), We=LD(x[N + 1]))


def reparam(c_old, sig_old, sig_new, N_new, P=None):
    """re-express a shape in a new stretch sigma (and number of modes): solve for the old parameter on
    the rays of the new grid, then take the cosine coefficients of log F_new (double precision; used
    only to build starting guesses)."""
    P = P or max(4 * N_new, 200)
    sh = Shape(np.asarray(c_old, float), sig_old, float)
    tp = 0.5 * np.pi * (np.arange(P) + 0.5) / P
    tg = np.linspace(1e-9, np.pi / 2, 40001)
    g = sh.geom(tg)
    t = np.interp(tp, np.arctan2(sig_new * g['r'], g['z']), tg)
    for _ in range(60):
        h = 1e-7
        g0, g1 = sh.geom(t), sh.geom(t + h)
        a0 = np.arctan2(sig_new * g0['r'], g0['z']) - tp
        a1 = np.arctan2(sig_new * g1['r'], g1['z']) - tp
        dt_ = -a0 * h / (a1 - a0)
        t = t + dt_
        if np.abs(dt_).max() < 1e-15:
            break
    g0 = sh.geom(t)
    lf = np.log(np.sqrt(g0['r'] ** 2 + (g0['z'] / sig_new) ** 2))
    k = np.arange(N_new)
    c = (2.0 / P) * (np.cos(np.outer(k, 2 * tp)) @ lf)
    c[0] *= 0.5
    return c


def convert(x, D_old, D_new):
    """starting guess for D_new from a solution of D_old"""
    c = reparam(np.asarray(x[:D_old.N], float), D_old.sigma, D_new.sigma, D_new.N)
    return np.concatenate([c, np.asarray(x[D_old.N:], float)])


def tau_ns(sigma, f, K, sigma_m=None, ns_min=16):
    """source offset tau and number of ring pairs.  For an ellipse of aspect sigma_m the continued
    meridian degenerates (focal segment) at a shift atanh(sigma_m) in the natural parameter; in a
    stretch sigma != sigma_m the shift in that parameter is between tau*sigma/sigma_m and tau*sigma_m/sigma,
    so tau = f * atanh(sigma_m) * min(sigma/sigma_m, sigma_m/sigma); ns = ceil(K / tau)."""
    sm = sigma if sigma_m is None else sigma_m
    tau = min(f * np.arctanh(min(sm, 0.999)) * min(sigma / sm, sm / sigma), 0.6)
    return tau, max(ns_min, int(np.ceil(K / tau)))


def aspect(x, D):
    p = props(x, D)
    return float(p['z_pole'] / p['r_eq'])


# ----------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------
def validate():
    out = {}
    t0 = time.time()
    # (a) ring kernel vs mpmath (direct Legendre-form formula, derivatives by mp.diff)
    mp.mp.dps = 30
    rng = np.random.default_rng(12345)

    def psi_mp(r, z, rho, zeta):
        R1s = (r + rho) ** 2 + (z - zeta) ** 2
        m = 4 * r * rho / R1s
        return mp.sqrt(R1s) / (4 * mp.pi) * ((2 - m) * mp.ellipk(m) - 2 * mp.ellipe(m))

    worst = {np.float64: [0.0, 0.0], LD: [0.0, 0.0]}
    for _ in range(60):
        r, z = 10 ** rng.uniform(-3, 0.3), rng.uniform(-1, 1)
        rho, zeta = rng.uniform(0.02, 1.6), rng.uniform(-1, 1)
        rs, zs, rhos, zetas = (mp.mpf(repr(v)) for v in (r, z, rho, zeta))
        pm = psi_mp(rs, zs, rhos, zetas)
        prm = mp.diff(lambda v: psi_mp(v, zs, rhos, zetas), rs)
        pzm = mp.diff(lambda v: psi_mp(rs, v, rhos, zetas), zs)
        for dt in (np.float64, LD):
            a = [np.array([dt(repr(v))]) for v in (r, z, rho, zeta)]
            p, pr, pz = ring_psi(*a)
            gn = abs(prm) + abs(pzm)
            worst[dt][0] = max(worst[dt][0], abs(float((tomp(p[0]) - pm) / pm)))
            worst[dt][1] = max(worst[dt][1], float((abs(tomp(pr[0]) - prm) + abs(tomp(pz[0]) - pzm)) / gn))
    out['ring_kernel_vs_mpmath'] = {'double': dict(psi_rel=worst[np.float64][0], grad_rel=worst[np.float64][1]),
                                    'longdouble': dict(psi_rel=worst[LD][0], grad_rel=worst[LD][1])}

    def mfs_speed_test(c, sigma, ns, tau, t_eval, nref, dt, src=None):
        sh = Shape(c, sigma, dt)
        if src is None:
            rho, zeta = source_points(c, sigma, ns, tau)
        else:
            rho, zeta = src
        rho, zeta = rho.astype(dt), zeta.astype(dt)
        tc = (np.arange(2 * ns) + dt(0.5)) / (2 * ns) * (_pi(dt) / 2)
        gc = sh.geom(tc)
        q, lsq = mfs_solve(gc['r'], gc['z'], rho, zeta, nref)
        g = sh.geom(t_eval)
        return surface_speed(q, rho, zeta, g), g, lsq

    # (b) sphere: |u| = 1.5 sin(theta)
    sph = {}
    for dt, nref in ((np.float64, 0), (LD, 3)):
        te = (np.arange(1, 1000) / LD(1000) * (PI_LD / 2)).astype(dt)
        u, g, lsq = mfs_speed_test([0.0], 1.0, 24, 0.6, te, nref, dt)
        sph['double' if dt == np.float64 else 'longdouble'] = dict(
            ns=24, tau=0.6, max_err_speed=float(np.abs(np.abs(u) - 1.5 * np.sin(te)).max()),
            max_err_H=float(np.abs(g['H'] - 2).max()), lsq_res=lsq)
    out['sphere'] = sph

    # (c) oblate spheroid, semi-axes a=1 (equatorial), b=1/4 (polar), stream along the short axis.
    #     exact: oblate spheroidal coordinates r = c cosh(mu) sin(nu), z = c sinh(mu) cos(nu),
    #     phi = c cos(nu) [zeta + B (zeta arccot zeta - 1)], zeta = sinh(mu); dphi/dmu = 0 at zeta0 = b/c:
    #     |u| = sin(nu) G / sqrt(zeta0^2 + cos^2 nu),  G = zeta0 + B (zeta0 arccot zeta0 - 1),
    #     B = -1/(arccot zeta0 - zeta0/(1+zeta0^2)).  Our parameter t equals nu for sigma = b/a.
    sphd = {}
    for dt, nref in ((np.float64, 0), (LD, 3)):
        a, b = dt(1), dt('0.25')
        cc = np.sqrt(a * a - b * b)
        z0 = b / cc
        act = np.arctan(1 / z0)
        B = -1 / (act - z0 / (1 + z0 * z0))
        G = z0 + B * (z0 * act - 1)
        te = (np.arange(1, 1000) / LD(1000) * (PI_LD / 2)).astype(dt)
        uex = np.sin(te) * G / np.sqrt(z0 ** 2 + np.cos(te) ** 2)
        den = np.sqrt(a * a * np.cos(te) ** 2 + b * b * np.sin(te) ** 2)
        Hex = a * b / den ** 3 + b / (a * den)
        rows = []
        for ns, f in ((60, 0.7), (80, 0.7), (100, 0.55)):
            tau = f * np.arctanh(0.25)
            u, g, lsq = mfs_speed_test([0.0], 0.25, ns, tau, te, nref, dt)
            rows.append(dict(ns=ns, tau=float(tau), max_err_speed=float(np.abs(np.abs(u) - uex).max()),
                             max_speed=float(uex.max()), lsq_res=lsq))
        # same spheroid in a mismatched stretch sigma=0.32 (non-trivial F, N=64 modes from reparam)
        sig2 = 0.32
        # exact log F on the new rays (r = a sin nu, z = b cos nu = sig2 F cos t), cosine coefficients
        # by a 600-point DCT, all in the working precision
        tp = (np.arange(600) + dt(0.5)) / 600 * (_pi(dt) / 2)
        nu = np.arctan2(np.sin(tp) * b, dt(sig2) * np.cos(tp))
        lf = np.log(np.sqrt(np.sin(nu) ** 2 + (b * np.cos(nu) / dt(sig2)) ** 2))
        c2 = (np.cos(np.outer(np.arange(64).astype(dt), 2 * tp)) @ lf) * 2 / 600
        c2[0] /= 2
        sh2 = Shape(c2, sig2, dt)
        te2 = (np.arange(1, 1000) / LD(1000) * (PI_LD / 2)).astype(dt)
        g2 = sh2.geom(te2)
        nu2 = np.arctan2(g2['r'], g2['z'] / b)               # spheroid parameter of these points
        uex2 = np.sin(nu2) * G / np.sqrt(z0 ** 2 + np.cos(nu2) ** 2)
        den2 = np.sqrt(a * a * np.cos(nu2) ** 2 + b * b * np.sin(nu2) ** 2)
        Hex2 = a * b / den2 ** 3 + b / (a * den2)
        tau2, ns2 = tau_ns(sig2, 0.6, 13.5, sigma_m=0.25)
        u2, _, lsq2 = mfs_speed_test(c2, sig2, ns2, tau2, te2, nref, dt, src=source_points(np.asarray(c2, float), sig2, ns2, tau2))
        D2 = Disc(64, sig2, 10, 0.1)
        vol = (dt(sig2) * (D2.grid[dt][2] @ np.exp(D2.grid[dt][3] @ c2) ** 3)) * 4 / 3 * _pi(dt)
        sphd['double' if dt == np.float64 else 'longdouble'] = dict(
            aspect=4.0, matched_sigma=rows, max_err_H_matched=float(np.abs(Shape([0.0], 0.25, dt).geom(te)['H'] - Hex).max()),
            mismatched_sigma=dict(sigma=sig2, N=64, ns=ns2, tau=float(tau2),
                                  max_err_speed=float(np.abs(np.abs(u2) - uex2).max()),
                                  max_err_H=float(np.abs(g2['H'] - Hex2).max()),
                                  volume_err=float(abs(vol - 4 * _pi(dt) * b / 3)), lsq_res=lsq2))
    out['oblate_spheroid_aspect4'] = sphd
    out['time_s'] = time.time() - t0
    return out


def small_We_check():
    """slopes of chi and lambda at We -> 0 (expect 9/32 and -3/4), polynomial extrapolation"""
    D = Disc(24, 1.0, 24, 0.6)
    x = np.zeros(26)
    x[24] = 2.0
    rows = []
    for We in (1e-4, 2e-4, 4e-4, 8e-4):
        x, inf = solve(D, x, 'We', We)
        p = props(x, D)
        rows.append(dict(We=We, chi=float(p['chi']), lam=float(p['lam']), z_pole=float(p['z_pole']),
                         res=inf['res_ld']))
    W = np.array([r['We'] for r in rows])
    s_chi = np.array([(r['chi'] - 1) / r['We'] for r in rows])
    s_lam = np.array([(r['lam'] - 2) / r['We'] for r in rows])
    # slope(We) = s0 + s1 We + s2 We^2 + s3 We^3 -> s0
    V = np.vander(W, 4, increasing=True)
    s0_chi = np.linalg.solve(V, s_chi)[0]
    s0_lam = np.linalg.solve(V, s_lam)[0]
    return dict(points=rows, dchi_dWe_at0=s0_chi, expected_dchi=9 / 32, err_dchi=s0_chi - 9 / 32,
                dlam_dWe_at0=s0_lam, expected_dlam=-0.75, err_dlam=s0_lam + 0.75)


# ----------------------------------------------------------------------------------------------
# branch scan (double precision, for starting guesses)
# ----------------------------------------------------------------------------------------------
def branch_scan(zps, N=48, f=0.7, K=13.5, verbose=True):
    D = Disc(N, 1.0, 24, 0.6)
    x = np.zeros(N + 2)
    x[N] = 2.0
    x, _ = solve(D, x, 'We', 0.05)
    sols = [(D, np.asarray(x, float))]
    for zp in zps:
        Dp, xp = sols[-1]
        pp = props(xp, Dp)
        sn = round(float(pp['z_pole'] / pp['r_eq']), 4)
        tau, ns = tau_ns(sn, f, K)
        Dn = Disc(N, sn, ns, tau)
        xg = convert(xp, Dp, Dn)
        if len(sols) >= 2:
            D0, x0 = sols[-2]
            z0, z1 = float(props(x0, D0)['z_pole']), float(pp['z_pole'])
            xg = xg + (zp - z1) / (z1 - z0) * (xg - convert(x0, D0, Dn))
        x, inf = solve(Dn, xg, 'zp', zp)
        x = np.asarray(x, float)
        sols.append((Dn, x))
        if verbose:
            print(f'  scan z_pole={zp:.4f} sigma={sn:.4f} ns={ns} We={x[N + 1]:.12f} lam={x[N]:.12f} '
                  f'|R|ld={inf["res_ld"]:.1e} lsq={inf["lsq"]:.1e} tail|c|={np.abs(x[N - 6:N]).max():.1e}', flush=True)
    return [(D.describe(), x.tolist()) for D, x in sols]


def nearest(scan, zp):
    best = None
    for Dd, x in scan:
        D = Disc(**Dd)
        z = float(props(x, D)['z_pole'])
        if best is None or abs(z - zp) < best[0]:
            best = (abs(z - zp), D, np.array(x))
    return best[1], best[2]


# ----------------------------------------------------------------------------------------------
# per-resolution tasks
# ----------------------------------------------------------------------------------------------
def ldstr(v):
    return numstr(LD(v))


def state(x, D, inf):
    p = props(x, D)
    return dict(We=ldstr(p['We']), lam=ldstr(p['lam']), z_pole=ldstr(p['z_pole']), chi=ldstr(p['chi']),
                r_eq=ldstr(p['r_eq']), res_ld=inf['res_ld'], lsq=inf['lsq'], umax=inf['umax'],
                tail_c=float(np.abs(np.asarray(x[D.N - 4:D.N], float)).max()))


def fold_task(args):
    res, scan = args
    t0 = time.time()
    N, f, K = res['N'], res['f'], res['K']
    Dm, xm = nearest(scan, 0.3926)
    sigma_m = round(aspect(xm, Dm), 4)
    sigma = round(res['sig_fac'] * sigma_m, 4)
    tau, ns = tau_ns(sigma, f, K, sigma_m)
    D = Disc(N, sigma, ns, tau, mcol=res.get('mfac', 2) * ns)
    out = dict(resolution=dict(D.describe(), f=f, K=K, sig_fac=res['sig_fac'], sigma_m=sigma_m))
    log = []
    use_ld = res.get('ld', True)
    out['resolution']['arithmetic'] = 'long double' if use_ld else 'double'
    # --- secant on dWe/dz_pole using implicit differentiation
    sols = {}

    def solve_at(zp, mode='zp', par=None):
        if sols:
            zs = np.array(list(sols.keys()))
            zn = zs[np.argmin(np.abs(zs - zp))]
            xn, Jn, dxdz = sols[zn]
            xg = np.asarray(xn, float) + dxdz * (zp - zn)
            x, inf = solve(D, xg, mode, par if par is not None else numstr(float(zp)), J=Jn, ld=use_ld)
        else:
            Ds, xs = nearest(scan, zp)
            x, inf = solve(D, convert(xs, Ds, D), mode, par if par is not None else numstr(float(zp)), ld=use_ld)
        J = D.jacobian(np.asarray(x, float), 'zp', float(props(x, D)['z_pole']), inf['src'])
        e = np.zeros(N + 2)
        zz = float(props(x, D)['z_pole'])
        e[N + 1] = 1.0 / zz
        dxdz = np.linalg.solve(J, e)
        if mode == 'zp':
            sols[zp] = (x, J, dxdz)
        return x, inf, dxdz[N + 1], J

    za, zb = 0.3955, 0.3895
    xa, ia, ga, _ = solve_at(za)
    xb, ib, gb, _ = solve_at(zb)
    for it in range(8):
        zc = zb - gb * (zb - za) / (gb - ga)
        xc, ic, gc, _ = solve_at(zc)
        log.append(dict(z=zc, dWe_dz=gc, We=ldstr(xc[N + 1])))
        za, ga, zb, gb = zb, gb, zc, gc
        if abs(zb - za) < 1e-9:
            break
    z0 = zb
    # --- Chebyshev interpolation of We(z_pole) on 9 nodes, half-width delta
    delta = 0.003
    nn = 9
    th = np.pi * (np.arange(nn) + 0.5) / nn
    nodes = z0 + delta * np.cos(th)
    Wn = []
    nres = []
    for zk in nodes:
        xk, ik, _, _ = solve_at(float(zk))
        Wn.append(LD(xk[N + 1]))
        nres.append(ik['res_ld'])
    mp.mp.dps = 30
    Wm = [tomp(w) for w in Wn]
    # Chebyshev coefficients on the nodes (exact interpolation)
    cheb = [mp.mpf(2) / nn * mp.fsum(Wm[j] * mp.cos(m * mp.pi * (j + mp.mpf(0.5)) / nn) for j in range(nn))
            for m in range(nn)]
    cheb[0] /= 2

    def p(s):
        return mp.fsum(cheb[m] * mp.chebyt(m, s) for m in range(nn))

    smax = mp.findroot(lambda s: mp.diff(p, s), 0)
    We_fit = p(smax)
    z_fit = z0 + delta * float(smax)
    # interpolation-error indicator: maximum of the degree-6 truncation of the same interpolant
    def p6(s):
        return mp.fsum(cheb[m] * mp.chebyt(m, s) for m in range(7))

    We_fit6 = p6(mp.findroot(lambda s: mp.diff(p6, s), smax))
    # --- final solve at the fitted fold location, and at the reference's fold z_pole
    xf, inf, gf, _ = solve_at(z_fit)
    xr, ir, gr, _ = solve_at(REFERENCE['fold']['z_pole'])
    dense = dense_check(D, xf) if use_ld else None
    out['fold'] = dict(We_max=mp.nstr(We_fit, 19), z_pole_fit=z_fit, state_at_fit=state(xf, D, inf),
                       dense_check=dense,
                       dWe_dz_at_fit=gf, We_at_fit_minus_We_max=float(tomp(LD(xf[N + 1])) - We_fit),
                       state_at_reference_z_pole=state(xr, D, ir),
                       secant_z0=z0, cheb_nodes_halfwidth=delta, cheb_coeffs=[float(v) for v in cheb],
                       We_max_deg6_truncation_minus_interp=float(We_fit6 - We_fit),
                       node_We=[numstr(w) for w in Wn], node_res_ld_max=max(nres), secant_log=log)
    out['time_s'] = time.time() - t0
    return out


def dense_check(D, x, K2=13.0, tau_factor=0.85, npts=1000):
    """a-posteriori check of a converged solution: recompute the flow with a different MFS (offset
    0.85 tau, ns = K2/tau rings, 3x oversampling) and evaluate H - (We/2)u^2 - lambda at npts points that
    are not collocation points; also recompute the volume with a 4x finer rule."""
    N = D.N
    x = np.asarray(x, LD)
    tau2 = tau_factor * D.tau
    ns2 = int(np.ceil(K2 / tau2))
    src2 = source_points(np.asarray(x[:N], float), D.sigma, ns2, tau2)
    rho, zeta = src2[0].astype(LD), src2[1].astype(LD)
    sh = Shape(x[:N], D.sigma, LD)
    tc = (np.arange(3 * ns2) + LD(0.5)) / (3 * ns2) * (PI_LD / 2)
    gc = sh.geom(tc)
    q, lsq = mfs_solve(gc['r'], gc['z'], rho, zeta, nref=3)
    t = (np.arange(npts) + LD('0.25')) / npts * (PI_LD / 2)
    g = sh.geom(t)
    u = surface_speed(q, rho, zeta, g)
    R = g['H'] - x[N + 1] / 2 * u * u - x[N]
    tv, wv = fejer_sin(16 * N, LD)
    F = np.exp(np.cos(np.outer(tv, 2 * np.arange(N).astype(LD))) @ x[:N])
    vol = LD(D.sigma) * (wv @ F ** 3) - 1
    return dict(ns=ns2, tau=float(tau2), mcol=3 * ns2, npts=npts, max_abs_residual=float(np.abs(R).max()),
                rms_residual=float(np.sqrt(np.mean(R * R))), mfs_lsq=lsq, volume_rel_err=float(vol))


def point_task(args):
    """one branch point at prescribed z_pole (string, used exactly), in a discretisation with stretch
    sig_fac * (local aspect); if a reference (We, lam) is given, also solve at fixed We = We_ref and
    compare lam and z_pole there (needs no z_pole rounding)."""
    res, scan, zr, ref = args
    t0 = time.time()
    Ds, xs = nearest(scan, float(zr))
    sigma_m = round(aspect(xs, Ds), 4)
    sigma = round(res['sig_fac'] * sigma_m, 4)
    tau, ns = tau_ns(sigma, res['f'], res['K'], sigma_m)
    D = Disc(res['N'], sigma, ns, tau, mcol=res.get('mfac', 2) * ns)
    N = D.N
    x, inf = solve(D, convert(xs, Ds, D), 'zp', zr)
    J = D.jacobian(np.asarray(x, float), 'zp', float(LD(zr)), inf['src'])
    e = np.zeros(N + 2)
    e[N + 1] = 1.0 / float(LD(zr))
    dxdz = np.linalg.solve(J, e)
    out = dict(z_pole=zr, resolution=dict(D.describe(), f=res['f'], K=res['K'], sig_fac=res['sig_fac'],
                                          sigma_m=sigma_m),
               at_zp=state(x, D, inf), dWe_dz=dxdz[N + 1], dlam_dz=dxdz[N])
    if ref is not None:
        st = out['at_zp']
        out['We_diff'] = float(LD(st['We']) - LD(numstr(ref['We'])))
        out['lam_diff'] = float(LD(st['lam']) - LD(numstr(ref['lam'])))
        out['rounding_tol_We'] = abs(dxdz[N + 1]) * 5e-6
        out['rounding_tol_lam'] = abs(dxdz[N]) * 5e-6
        x2, i2 = solve(D, np.asarray(x, float), 'We', numstr(ref['We']))
        st2 = state(x2, D, i2)
        out['at_reference_We'] = st2
        out['lam_diff_at_ref_We'] = float(LD(st2['lam']) - LD(numstr(ref['lam'])))
        out['zp_diff_at_ref_We'] = float(LD(st2['z_pole']) - LD(zr))
    out['time_s'] = time.time() - t0
    return out


def dimple_task(args):
    res, scan = args
    t0 = time.time()
    N, f, K = res['N'], res['f'], res['K']
    Ds, xs = nearest(scan, 0.2715)
    p = props(xs, Ds)
    sigma_m = round(float(p['z_pole'] / p['r_eq']), 4)
    sigma = round(res['sig_fac'] * sigma_m, 4)
    tau, ns = tau_ns(sigma, f, K, sigma_m)
    D = Disc(N, sigma, ns, tau, mcol=res.get('mfac', 2) * ns)
    x, inf = solve(D, convert(xs, Ds, D), 'lam', '0')
    # re-solve once with sources rebuilt from the converged shape
    x, inf = solve(D, np.asarray(x, float), 'lam', '0')
    return dict(resolution=dict(D.describe(), f=f, K=K, sig_fac=res['sig_fac'], sigma_m=sigma_m),
                state=state(x, D, inf), time_s=time.time() - t0)


# ----------------------------------------------------------------------------------------------
def main():
    T0 = time.time()
    results = dict(description=__doc__.split('Method')[0].strip(), reference=REFERENCE)
    print('validation ...', flush=True)
    results['validation'] = validate()
    v = results['validation']
    print(json.dumps(v, indent=1, default=float)[:3000], flush=True)
    print('small-We check ...', flush=True)
    results['small_We'] = small_We_check()
    print(f"  dchi/dWe(0) = {results['small_We']['dchi_dWe_at0']:.10f} (9/32 = 0.28125), "
          f"dlam/dWe(0) = {results['small_We']['dlam_dWe_at0']:.10f} (-3/4)", flush=True)
    print('branch scan (double precision starting guesses) ...', flush=True)
    t1 = time.time()
    zps = [0.97, 0.94, 0.91, 0.89, 0.86, 0.83, 0.80, 0.77, 0.74, 0.71, 0.70, 0.67, 0.64, 0.61, 0.60, 0.58,
           0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.41, 0.40, 0.395, 0.39, 0.385, 0.38, 0.37, 0.36, 0.35,
           0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.275, 0.27]
    scan = branch_scan(zps)
    results['scan_time_s'] = time.time() - t1
    # resolutions: shape modes N, stretch factor (sigma = sig_fac * local aspect), source offset factor f,
    # K = ns * tau (number of ring pairs ns = K / tau), MFS boundary points mcol = mfac * ns
    resolutions = [dict(N=32, sig_fac=1.00, f=0.70, K=11.0),
                   dict(N=40, sig_fac=1.04, f=0.80, K=12.5),
                   dict(N=48, sig_fac=1.00, f=0.70, K=13.5),
                   dict(N=56, sig_fac=0.96, f=0.62, K=14.5),
                   dict(N=64, sig_fac=1.12, f=0.55, K=15.5),
                   dict(N=80, sig_fac=0.92, f=0.60, K=16.0, mfac=3)]
    point_res = resolutions[2:5]
    dimple_res = [dict(N=56, sig_fac=1.00, f=0.70, K=13.5), dict(N=72, sig_fac=0.95, f=0.60, K=15.0),
                  dict(N=88, sig_fac=1.08, f=0.55, K=16.0)]
    double_res = [dict(resolutions[2], ld=False), dict(resolutions[4], ld=False)]
    tasks = [('fold', (r, scan)) for r in resolutions]
    tasks += [('fold', (r, scan)) for r in double_res]
    tasks += [('dimple', (r, scan)) for r in dimple_res]
    for r in point_res:
        tasks += [('point', (r, scan, ref['z_pole'], ref)) for ref in REFERENCE['points']]
        tasks += [('point', (r, scan, zr, None)) for zr in ('0.5', '0.7')]
    from multiprocessing import get_context
    print(f'{len(tasks)} tasks (fold searches, extra points, dimple onset) on 4 processes ...', flush=True)
    t2 = time.time()
    with get_context('fork').Pool(4) as pool:
        outs = pool.map(run_task, tasks, chunksize=1)
    results['parallel_time_s'] = time.time() - t2
    folds = [o for (k, _), o in zip(tasks, outs) if k == 'fold' and o['resolution']['arithmetic'] == 'long double']
    folds_double = [o for (k, _), o in zip(tasks, outs) if k == 'fold' and o['resolution']['arithmetic'] == 'double']
    results['fold_runs_double_precision'] = folds_double
    dimples = [o for (k, _), o in zip(tasks, outs) if k == 'dimple']
    points = [o for (k, _), o in zip(tasks, outs) if k == 'point']
    results['fold_runs'] = folds
    results['dimple_runs'] = dimples
    results['point_runs'] = points
    # ----- summary
    mp.mp.dps = 30
    ref = REFERENCE['fold']
    table = []
    for o in folds:
        r, fo = o['resolution'], o['fold']
        st, sr = fo['state_at_fit'], fo['state_at_reference_z_pole']
        table.append(dict(N=r['N'], sigma=r['sigma'], ns=r['ns'], tau=r['tau'], mcol=r['mcol'], We_max=fo['We_max'],
                          We_max_minus_reference=float(mp.mpf(fo['We_max']) - mp.mpf(numstr(ref['We_max']))),
                          z_pole_fold=fo['z_pole_fit'], chi=st['chi'], lam=st['lam'], r_eq=st['r_eq'],
                          at_ref_z_pole=dict(chi=sr['chi'], lam=sr['lam'], r_eq=sr['r_eq'], We=sr['We']),
                          residual_ld=st['res_ld'], mfs_lsq=st['lsq'], shape_tail=st['tail_c'],
                          cheb_tail=max(abs(v) for v in fo['cheb_coeffs'][-2:]), node_res_max=fo['node_res_ld_max'],
                          dense_max_residual=fo['dense_check']['max_abs_residual'], time_s=o['time_s']))
    Wm = [mp.mpf(t['We_max']) for t in table]
    hi = Wm[1:]
    summ = dict(fold_table=table,
                We_max_spread_all=float(max(Wm) - min(Wm)), We_max_spread_N_ge_40=float(max(hi) - min(hi)),
                We_max_mean_N_ge_40=mp.nstr(mp.fsum(hi) / len(hi), 18),
                We_max_reference=ref['We_max'],
                mean_minus_reference=float(mp.fsum(hi) / len(hi) - mp.mpf(numstr(ref['We_max']))))
    # fold state: the reference's printed (chi, lam, r_eq) correspond to which z_pole on our branch?
    # (linear interpolation between our solutions at our fold z* and at the reference z_pole)
    o = folds[-1]['fold']
    z1, z2 = mp.mpf(o['state_at_fit']['z_pole']), mp.mpf(o['state_at_reference_z_pole']['z_pole'])
    fs = {}
    for key in ('chi', 'lam', 'r_eq'):
        v1, v2 = mp.mpf(o['state_at_fit'][key]), mp.mpf(o['state_at_reference_z_pole'][key])
        slope = (v1 - v2) / (z1 - z2)
        fs[key] = dict(ours_at_our_fold=o['state_at_fit'][key], ours_at_ref_z_pole=o['state_at_reference_z_pole'][key],
                       reference=ref[key], d_dz=float(slope),
                       z_pole_where_ours_equals_reference=float(z2 + (mp.mpf(numstr(ref[key])) - v2) / slope))
    summ['fold_state_comparison'] = dict(our_fold_z_pole=o['state_at_fit']['z_pole'], reference_fold_z_pole=ref['z_pole'],
                                         quantities=fs)
    if folds_double:
        Wd = [mp.mpf(o['fold']['We_max']) for o in folds_double]
        summ['double_precision_We_max'] = [o['fold']['We_max'] for o in folds_double]
        summ['double_precision_minus_long_double_mean'] = [float(w - mp.fsum(hi) / len(hi)) for w in Wd]
    pts = {}
    for o in points:
        d = pts.setdefault(o['z_pole'], dict(We=[], lam=[], rows=[]))
        d['We'].append(o['at_zp']['We'])
        d['lam'].append(o['at_zp']['lam'])
        row = dict(N=o['resolution']['N'], We=o['at_zp']['We'], lam=o['at_zp']['lam'], dWe_dz=o['dWe_dz'],
                   residual_ld=o['at_zp']['res_ld'])
        for k in ('We_diff', 'lam_diff', 'rounding_tol_We', 'rounding_tol_lam', 'lam_diff_at_ref_We', 'zp_diff_at_ref_We'):
            if k in o:
                row[k] = o[k]
        if 'lam_diff_at_ref_We' in o:
            # same shape-branch, We offset implied by the lambda mismatch at fixed We: lam_ours(W) = lam_ref(W - dWe)
            dlam_dWe = o['dlam_dz'] / o['dWe_dz']
            row['implied_We_offset'] = -o['lam_diff_at_ref_We'] / dlam_dWe
            row['implied_rel_We_offset'] = row['implied_We_offset'] / float(LD(o['at_zp']['We']))
        d['rows'].append(row)
    for zr, d in pts.items():
        d['We_spread'] = float(max(mp.mpf(v) for v in d['We']) - min(mp.mpf(v) for v in d['We']))
        d['lam_spread'] = float(max(mp.mpf(v) for v in d['lam']) - min(mp.mpf(v) for v in d['lam']))
    summ['points'] = pts
    dref = REFERENCE['dimple']
    summ['dimple'] = dict(We=[o['state']['We'] for o in dimples], z_pole=[o['state']['z_pole'] for o in dimples],
                          We_minus_reference=[float(mp.mpf(o['state']['We']) - mp.mpf(numstr(dref['We']))) for o in dimples],
                          z_pole_minus_reference=[float(mp.mpf(o['state']['z_pole']) - mp.mpf(numstr(dref['z_pole'])))
                                                  for o in dimples],
                          chi=[o['state']['chi'] for o in dimples], r_eq=[o['state']['r_eq'] for o in dimples])
    results['summary'] = summ
    results['total_time_s'] = time.time() - T0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=1, default=float)
    # ----- printout
    print('\nresolution study at the fold (We_max = max of Chebyshev interpolant of We(z_pole)):')
    for t in table:
        print(f"  N={t['N']:3d} sigma={t['sigma']:.4f} ns={t['ns']:3d} mcol={t['mcol']:3d} tau={t['tau']:.4f}: "
              f"We_max={t['We_max']}  (-ref {t['We_max_minus_reference']:+.2e})  z*={t['z_pole_fold']:.10f} "
              f"chi={float(t['chi']):.8f} lam={float(t['lam']):.8f} r_eq={float(t['r_eq']):.8f} "
              f"|R|={t['residual_ld']:.1e} lsq={t['mfs_lsq']:.1e} ({t['time_s']:.0f}s)")
    print(f"  spread (all) {summ['We_max_spread_all']:.2e}, spread (N>=40) {summ['We_max_spread_N_ge_40']:.2e}, "
          f"mean(N>=40) {summ['We_max_mean_N_ge_40']}, mean - reference {summ['mean_minus_reference']:+.2e}")
    for zr, d in pts.items():
        print(f"  z_pole={zr}: We={d['We']} lam={d['lam']} spreads {d['We_spread']:.1e} {d['lam_spread']:.1e}")
        for row in d['rows']:
            if 'We_diff' in row:
                print(f"     N={row['N']}: We-ref {row['We_diff']:+.2e} (rounding tol {row['rounding_tol_We']:.1e}), "
                      f"lam-ref {row['lam_diff']:+.2e} (tol {row['rounding_tol_lam']:.1e}); at We=We_ref: "
                      f"lam-ref {row['lam_diff_at_ref_We']:+.2e}, z_pole-{zr} {row['zp_diff_at_ref_We']:+.2e}")
    print(f"  fold state vs reference: {json.dumps(summ['fold_state_comparison'], default=float)}")
    for o in folds_double:
        print(f"  DOUBLE precision run N={o['resolution']['N']}: We_max={o['fold']['We_max']}")
    for o in folds:
        print(f"  dense check N={o['resolution']['N']}: {json.dumps(o['fold']['dense_check'], default=float)}")
    for zr, d in pts.items():
        for row in d['rows']:
            if 'implied_We_offset' in row:
                print(f"  z_pole={zr} N={row['N']}: implied We offset {row['implied_We_offset']:+.2e} "
                      f"(relative {row['implied_rel_We_offset']:+.2e})")
    print(f"  dimple onset: We={summ['dimple']['We']} (-ref {summ['dimple']['We_minus_reference']}), "
          f"z_pole={summ['dimple']['z_pole']} (-ref {summ['dimple']['z_pole_minus_reference']})")
    print(f'total time {time.time() - T0:.0f}s; wrote {OUT}')


def run_task(task):
    kind, args = task
    return dict(fold=fold_task, point=point_task, dimple=dimple_task)[kind](args)


if __name__ == '__main__':
    if np.finfo(np.longdouble).nmant < 63:
        print('warning: numpy long double is not 80-bit on this platform (e.g. macOS arm64, Windows); '
              'the extended-precision parts of this check then run in double precision', flush=True)
    main()
