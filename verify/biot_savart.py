"""Independent Biot-Savart evaluation of the velocity of an axisymmetric
vortex sheet plus the uniform stream e_z.

    u(x) = e_z + (1/4pi) int_S (gamma e_phi) x (x - y) / |x - y|^3 dA,
    dA = rho ds dphi,  y = (rho cos phi, rho sin phi, zeta).

For a target x = (r0, 0, z0) (meridian plane phi = 0) this gives

    u_r = (1/4pi) int ds gamma rho int_0^{2pi} dphi (z0 - zeta) cos(phi) / D^3
    u_z = (1/4pi) int ds gamma rho int_0^{2pi} dphi (rho - r0 cos(phi)) / D^3
    D^2 = r0^2 + rho^2 - 2 r0 rho cos(phi) + (z0 - zeta)^2.

Both integrands are even in phi; the phi integral is done by brute-force
Gauss-Legendre quadrature on [0, pi] with geometric grading towards phi = 0
(the near-singular azimuth), the s integral by Gauss-Legendre on the
solver's panels, adaptively bisected near the target.  Source data (r, z,
ds/dt, gamma) are interpolated from the solver's nodal values with our own
barycentric Lagrange interpolation.  Nothing from bubbles.kernels /
bubbles.sheet is used here.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def bary_weights(x):
    d = x[:, None] - x[None, :]
    np.fill_diagonal(d, 1.0)
    return 1.0 / np.prod(d, axis=1)


def lagrange_matrix(xn, bw, u):
    """L[k, j] = l_j(u_k)."""
    u = np.asarray(u, float)
    d = u[:, None] - xn[None, :]
    exact = d == 0.0
    d = np.where(exact, 1.0, d)
    t = bw[None, :] / d
    L = t / t.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    if hit.any():
        L[hit] = exact[hit].astype(float)
    return L


class Curve:
    """Generating curve of the sheet given on panels in a parameter t.

    vals: dict with arrays (npan, p) at the Gauss nodes of each panel:
          'r', 'z', 'c' (= ds/dt), 'gamma'.
    ta, tb: panel ends in t.  mirror: add the mirror image z -> -z with the
    same gamma (the solver's even sheet)."""

    def __init__(self, ta, tb, vals, mirror=True):
        self.ta, self.tb = np.asarray(ta, float), np.asarray(tb, float)
        self.npan, self.p = vals['r'].shape
        self.xn, _ = leggauss(self.p)
        self.bw = bary_weights(self.xn)
        self.V = np.stack([vals[k] for k in ('r', 'z', 'c', 'gamma')], axis=-1)  # (npan, p, 4)
        self.mirror = mirror
        # dense samples for distance computations
        us = np.linspace(-1, 1, 33)
        Ls = lagrange_matrix(self.xn, self.bw, us)
        dense = np.einsum('kj,pjc->pkc', Ls, self.V)
        self.dense_r = dense[..., 0].ravel()
        self.dense_z = dense[..., 1].ravel()

    def interp(self, q, u):
        """values (n, 4) at panel q (n,) and local coordinate u (n,)."""
        d = u[:, None] - self.xn[None, :]
        exact = d == 0.0
        d = np.where(exact, 1.0, d)
        t = self.bw[None, :] / d
        L = t / t.sum(axis=1, keepdims=True)
        hit = exact.any(axis=1)
        if hit.any():
            L[hit] = exact[hit].astype(float)
        return np.einsum('nj,njc->nc', L, self.V[q])

    def dist(self, r0, z0, nfine=4097):
        """distance from (r0, z0) to the sheet: coarse samples, then a fine
        search on the nearest panel and its neighbours."""
        best = np.inf
        halves = [z0] + ([-z0] if self.mirror else [])
        ns = len(self.dense_r) // self.npan
        uf = np.linspace(-1, 1, nfine)
        for zt in halves:
            dd = np.hypot(self.dense_r - r0, self.dense_z - zt)
            q0 = int(np.argmin(dd)) // ns
            for q in range(max(q0 - 1, 0), min(q0 + 2, self.npan)):
                v = self.interp(np.full(nfine, q), uf)
                best = min(best, np.hypot(v[:, 0] - r0, v[:, 1] - zt).min())
        return best

    def sources(self, r0, z0, nq=16, fac=1.5, maxdepth=48, nsamp=9):
        """Quadrature points (rho, zeta, w = gamma ds) adapted to target (r0, z0)."""
        xq, wq = leggauss(nq)
        us = np.linspace(-1, 1, nsamp)
        out = []
        halves = [(1.0, z0)] + ([(-1.0, -z0)] if self.mirror else [])
        for sgn, zt in halves:
            q = np.arange(self.npan)
            lo = -np.ones(self.npan)
            hi = np.ones(self.npan)
            final_q, final_lo, final_hi = [], [], []
            for depth in range(maxdepth + 1):
                if len(q) == 0:
                    break
                uu = lo[:, None] + (hi - lo)[:, None] * (us[None, :] + 1) / 2
                v = self.interp(np.repeat(q, nsamp), uu.ravel()).reshape(len(q), nsamp, 4)
                dmin = np.hypot(v[..., 0] - r0, v[..., 1] - zt).min(axis=1)
                # arclength of the sub-panel: trapezoid of c dt over the samples
                dt = (self.tb[q] - self.ta[q]) / 2 * (hi - lo)
                arc = np.trapezoid(v[..., 2], dx=1.0 / (nsamp - 1), axis=1) * dt
                split = (dmin < fac * arc) & (depth < maxdepth)
                keep = ~split
                final_q.append(q[keep]); final_lo.append(lo[keep]); final_hi.append(hi[keep])
                mid = (lo[split] + hi[split]) / 2
                q = np.concatenate([q[split], q[split]])
                lo, hi = np.concatenate([lo[split], mid]), np.concatenate([mid, hi[split]])
            fq = np.concatenate(final_q)
            flo = np.concatenate(final_lo)
            fhi = np.concatenate(final_hi)
            u = (flo[:, None] + fhi[:, None]) / 2 + (fhi - flo)[:, None] / 2 * xq[None, :]
            w = (fhi - flo)[:, None] / 2 * wq[None, :] * ((self.tb[fq] - self.ta[fq]) / 2)[:, None]
            v = self.interp(np.repeat(fq, nq), u.ravel())
            ww = w.ravel() * v[:, 2] * v[:, 3]          # gamma ds
            out.append(np.stack([v[:, 0], sgn * v[:, 1], ww], axis=1))
        return np.concatenate(out)


def phi_rule(eps, nq=16, ratio=2.0):
    """Gauss rule on [0, pi] graded geometrically towards 0 (smallest panel eps)."""
    eps = float(np.clip(eps, 1e-15, np.pi / 4))
    br = [0.0, eps]
    while br[-1] * ratio < np.pi:
        br.append(br[-1] * ratio)
    br.append(np.pi)
    br = np.array(br)
    if br[-1] - br[-2] < 0.3 * (br[-2] - br[-3]):     # avoid a sliver panel
        br = np.delete(br, -2)
    x, w = leggauss(nq)
    a, b = br[:-1], br[1:]
    ph = ((a + b)[:, None] + (b - a)[:, None] * x[None, :]) / 2
    wp = (b - a)[:, None] / 2 * w[None, :]
    return ph.ravel(), wp.ravel()


def velocity(curve, r0, z0, nq=16, nphi=16, fac=1.5, eps_fac=0.2, chunk=200000, stream=True, d=None):
    """u = (u_r, u_z) at (r0, z0) [stream=True adds e_z]."""
    if d is None:
        d = curve.dist(r0, z0)
    src = curve.sources(r0, z0, nq=nq, fac=fac)
    rho, zeta, w = src[:, 0], src[:, 1], src[:, 2]
    eps = eps_fac * d / max(r0, rho.max() * 1e-3, 1e-300)
    ph, wph = phi_rule(eps, nq=nphi)
    cph = np.cos(ph)
    one_m_c = 2 * np.sin(ph / 2) ** 2            # 1 - cos(phi) without cancellation
    ur = uz = 0.0
    n = len(rho)
    step = max(1, chunk // len(ph))
    for k in range(0, n, step):
        rr, zz, ww = rho[k:k + step, None], zeta[k:k + step, None], w[k:k + step]
        dz = z0 - zz
        D2 = (r0 - rr) ** 2 + 2 * r0 * rr * one_m_c[None, :] + dz ** 2
        iD3 = D2 ** -1.5
        Ir = (iD3 * cph[None, :]) @ wph          # int_0^pi cos(phi)/D^3
        Iz = (iD3 * (rr - r0 * cph[None, :])) @ wph
        ur = ur + np.sum(ww * rr[:, 0] * dz[:, 0] * Ir)
        uz = uz + np.sum(ww * rr[:, 0] * Iz)
    fac2 = 2.0 / (4 * np.pi)                     # 2 for [0, pi] -> [0, 2pi]
    return np.array([fac2 * ur, fac2 * uz + (1.0 if stream else 0.0)]), d, len(rho), len(ph)


def ring_curve(st):
    g, P = st.g, st.P
    sh = (P.np, P.p)
    vals = dict(r=g.r.reshape(sh), z=g.z.reshape(sh), c=g.c.reshape(sh),
                gamma=np.asarray(st.gamma).reshape(sh))
    return Curve(P.a, P.b, vals, mirror=True)
