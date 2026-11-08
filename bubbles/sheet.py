"""Vortex-sheet (stream-function) formulation of the exterior flow.

With the fluid at rest inside the body, the flow is the uniform stream plus
an azimuthal vortex sheet on S whose strength gamma is the surface speed
(u = gamma T on the fluid side).  The Stokes stream function is

    Psi(x) = r^2/2 + int_S gamma(s') Psi_ring(x; y(s')) ds',

and the body is a stream surface: Psi = Q on S.  For a body touching the axis
(a bubble) Q = 0; for a torus Q is the flux constant (flux through the hole
is 2 pi Q).  This gives the first-kind equation

    V[gamma] = Q - r^2/2   on S,

with a logarithmically singular kernel.  It is used for toroidal bubbles,
where the potential formulation would need an extra circulation mode.

Geometry interface: the upper half of a z -> -z symmetric generating curve,
given by panels with Chebyshev coefficients A (positions, (2, n, deg+1)) and
B (x'(t), (2, n, deg)), panel lengths h in t, a PanelRule, and nodal
positions.  The lower half is the mirror image (r, -z); gamma is even.
"""
import numpy as np
import scipy.linalg as sla
from scipy.special import elliprd, elliprf

from .quad import cheb_divided_differences, cheb_eval, graded_rule


def psi_ring_kernel(r, rho, dz, b=None):
    a = (r + rho) ** 2 + dz ** 2
    if b is None:
        b = (r - rho) ** 2 + dz ** 2
    m1 = b / a
    return r * rho / (np.pi * np.sqrt(a)) * (2.0 / 3.0 * elliprd(0.0, m1, 1.0)
                                             - elliprf(0.0, m1, 1.0))


def _closest(A, X, Z, nsamp=65, newton=5):
    from numpy.polynomial import chebyshev as C
    us = np.linspace(-1, 1, nsamp)
    y = cheb_eval(A, us[None, None, :])
    d2 = (y[0] - X[:, None]) ** 2 + (y[1] - Z[:, None]) ** 2
    u = us[np.argmin(d2, axis=1)]
    A1 = np.stack([np.stack([C.chebder(A[c, q]) for q in range(A.shape[1])]) for c in range(2)])
    A2 = np.stack([np.stack([C.chebder(A1[c, q]) for q in range(A.shape[1])]) for c in range(2)])
    for _ in range(newton):
        uu = u[:, None]
        y = cheb_eval(A, uu[None])[..., 0]
        y1 = cheb_eval(A1, uu[None])[..., 0]
        y2 = cheb_eval(A2, uu[None])[..., 0]
        f = (y[0] - X) * y1[0] + (y[1] - Z) * y1[1]
        fp = y1[0] ** 2 + y1[1] ** 2 + (y[0] - X) * y2[0] + (y[1] - Z) * y2[1]
        step = np.where(fp > 0, f / np.where(fp > 0, fp, 1.0), 0.0)
        u = np.clip(u - step, -1.0, 1.0)
    return u


class SheetOperator:
    """Folded matrix of V on the upper-half nodes (gamma even under z -> -z)."""

    def __init__(self, A, B, h, rule, eta=1.0, sigma=0.5, ng=10, depth=40, chunk=256):
        n, p = A.shape[1], rule.p
        self.n, self.p = n, p
        x = cheb_eval(A, np.broadcast_to(rule.x, (n, p))[None])      # nodes (2, n, p)
        xp = cheb_eval(B, np.broadcast_to(rule.x, (n, p))[None])
        cn = np.hypot(xp[0], xp[1])
        w = (h / 2)[:, None] * rule.w[None, :] * cn                   # ds weights
        r, z = x[0].ravel(), x[1].ravel()
        self.r, self.z, self.ds = r, z, w.ravel()
        N = n * p
        with np.errstate(divide='ignore', invalid='ignore'):
            Vd = psi_ring_kernel(r[:, None], r[None, :], z[:, None] - z[None, :])
            Vm = psi_ring_kernel(r[:, None], r[None, :], z[:, None] + z[None, :])
        Vd *= self.ds[None, :]
        Vm *= self.ds[None, :]
        # special pairs, for the target itself (direct) and its mirror image
        ends = cheb_eval(A, np.tile([-1.0, 1.0], (1, n, 1)))
        samp = np.concatenate([x, ends], axis=2)                     # (2, n, p+2)
        ell = w.sum(axis=1)
        own = np.repeat(np.arange(n), p)
        uloc = np.tile(rule.x, n)
        xi_all = {}
        for mirror, M_ in ((False, Vd), (True, Vm)):
            zt = -z if mirror else z
            d = np.sqrt((r[:, None, None] - samp[0][None]) ** 2
                        + (zt[:, None, None] - samp[1][None]) ** 2).min(axis=2)
            special = d < eta * ell[None, :]
            if not mirror:
                special[np.arange(N), own] = True
            ii, pp = np.nonzero(special)
            is_self = (pp == own[ii]) & (not mirror)
            us = np.empty(len(ii))
            us[is_self] = uloc[ii[is_self]]
            ns = np.nonzero(~is_self)[0]
            if len(ns):
                us[ns] = _closest(A[:, pp[ns], :], r[ii[ns]], zt[ii[ns]])
            dloc = np.where(is_self, 0.0, 2.0 * d[ii, pp] / ell[pp])
            with np.errstate(divide='ignore'):
                dep = np.where(is_self, depth, np.clip(
                    np.ceil(np.log2(2.0 / np.maximum(dloc, 1e-300))) + 4, 4, depth)).astype(int)
            for D in np.unique(dep):
                if D not in xi_all:
                    xi_all[D] = graded_rule(sigma, ng, int(D))
                xi, om = xi_all[D]
                grp = np.nonzero(dep == D)[0]
                for c0 in range(0, len(grp), chunk):
                    sel = grp[c0:c0 + chunk]
                    i_, p_, u_, sf = ii[sel], pp[sel], us[sel], is_self[sel]
                    delta = np.concatenate([(u_[:, None] + 1) * xi[None, :],
                                            -(1 - u_[:, None]) * xi[None, :]], axis=1)
                    uq = u_[:, None] - delta
                    wq = np.concatenate([(u_[:, None] + 1) * om[None, :],
                                         (1 - u_[:, None]) * om[None, :]], axis=1)
                    Ap = A[:, p_, :]
                    y = cheb_eval(Ap, uq[None])
                    yp = cheb_eval(B[:, p_, :], uq[None])
                    cq = np.hypot(yp[0], yp[1])
                    rt = r[i_][:, None]
                    dz = zt[i_][:, None] - y[1]
                    b = (rt - y[0]) ** 2 + dz ** 2
                    if sf.any():
                        k = np.nonzero(sf)[0]
                        D1, _ = cheb_divided_differences(Ap[:, k, :], uq[k][None], u_[k][:, None][None])
                        dl = delta[k]
                        dz[k] = dl * D1[1]
                        b[k] = dl ** 2 * (D1[0] ** 2 + D1[1] ** 2)
                    K = psi_ring_kernel(rt, y[0], dz, b)
                    W = wq * (h[p_] / 2)[:, None] * cq
                    blk = np.einsum('nq,nqj->nj', W * K, rule.lagrange(uq))
                    cols = p_[:, None] * p + np.arange(p)[None, :]
                    M_[i_[:, None], cols] = blk
        self.V = Vd + Vm
        self.lu = sla.lu_factor(self.V)

    def solve(self, rhs):
        return sla.lu_solve(self.lu, rhs)


def bubble_sheet_speed(st, **opts):
    """Surface speed of a solved BubbleState via the sheet formulation (check)."""
    g, P = st.g, st.P
    n = P.nph
    op = SheetOperator(g.A[:, :n, :], g.B[:, :n, :], P.h[:n], P.rule, **opts)
    gam = op.solve(-0.5 * op.r ** 2)
    return gam, op
