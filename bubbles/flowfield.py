"""Flow field and surface data off the computational nodes (for plotting).

With the fluid at rest inside the bubble (u = 0 in B), the velocity field is
the uniform stream plus the azimuthal vortex sheet on S whose strength is
the surface speed gamma = d phi/ds.  Hence the Stokes stream function

    Psi(x) = r^2/2 + int_S gamma(s') Psi_ring(x; y(s')) ds',

with Psi_ring the stream function of a vortex ring of unit circulation,

    Psi_ring = (r rho / (pi sqrt(a))) [ (2/3) R_D(0, m1, 1) - R_F(0, m1, 1) ],

a = (r + rho)^2 + (z - zeta)^2, m1 = ((r - rho)^2 + (z - zeta)^2)/a (Lamb's
(2/k - k)K - (2/k)E written without cancellation).  Psi = 0 on S and on the
axis; streamlines are its level sets, Psi = r_inf^2/2 for the streamline
starting at radius r_inf far upstream.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import elliprd, elliprf

from .geometry import theta_eval
from .quad import cheb_eval


def ring_stream_function(r, z, rho, zeta):
    a = (r + rho) ** 2 + (z - zeta) ** 2
    m1 = ((r - rho) ** 2 + (z - zeta) ** 2) / a
    return r * rho / (np.pi * np.sqrt(a)) * (2.0 / 3.0 * elliprd(0.0, m1, 1.0)
                                             - elliprf(0.0, m1, 1.0))


def _locate(P, t):
    idx = np.clip(np.searchsorted(P.b, t, side='right'), 0, P.np - 1)
    idx = np.where(t >= P.b[idx], np.minimum(idx + 1, P.np - 1), idx)
    idx = np.where(t < P.a[idx], np.maximum(idx - 1, 0), idx)
    u = (2 * t - P.a[idx] - P.b[idx]) / P.h[idx]
    return idx, np.clip(u, -1, 1)


def surface_samples(st, t):
    """Position, signed surface speed and H at parameters t in [0, pi]."""
    g, P = st.g, st.P
    t = np.asarray(t, float)
    idx, u = _locate(P, t)
    x = cheb_eval(g.A[:, idx, :], u[None, :, None])[..., 0]
    th, tht = theta_eval(st.a, t)
    c = g.L * st.pmap.Sp(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -tht / c - np.sin(th) / x[0]
    # at the poles kappa_2 = kappa_1
    pole = x[0] < 1e-12
    H = np.where(pole, -2 * tht / c, H)
    Lm = P.rule.lagrange(u)
    phs = st.phi_s_full.reshape(P.np, P.p)[idx]
    gamma = np.einsum('nj,nj->n', Lm, phs)
    return x[0], x[1], gamma, H


class StreamFunction:
    """Evaluate Psi from a solved BubbleState (sources upsampled on panels)."""

    def __init__(self, st, n_sub=32):
        g, P = st.g, st.P
        xs, ws = leggauss(n_sub)
        u = np.broadcast_to(xs, (P.np, n_sub))
        y = cheb_eval(g.A, u[None])
        yp = cheb_eval(g.B, u[None])
        Lm = P.rule.lagrange(u)
        gam = np.einsum('pqj,pj->pq', Lm, st.phi_s_full.reshape(P.np, P.p))
        w = ws[None, :] * (P.h / 2)[:, None] * np.hypot(yp[0], yp[1])
        self.rho = y[0].ravel()
        self.zeta = y[1].ravel()
        self.wg = (w * gam).ravel()

    def __call__(self, r, z, chunk=4000):
        r = np.asarray(r, float).ravel()
        z = np.asarray(z, float).ravel()
        out = np.empty_like(r)
        for k in range(0, len(r), chunk):
            rr, zz = r[k:k + chunk, None], z[k:k + chunk, None]
            out[k:k + chunk] = rr[:, 0] ** 2 / 2 + ring_stream_function(
                rr, zz, self.rho[None, :], self.zeta[None, :]) @ self.wg
        return out
