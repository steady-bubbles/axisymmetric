"""Nystrom discretisation of the exterior Neumann problem on an axisymmetric,
fore-aft symmetric surface.

For a harmonic function psi in the exterior, decaying at infinity, Green's
representation formula gives on the surface

    (1/2) psi - K psi = - S [d_n psi],

with K the double layer and S the single layer operator (normal n pointing
into the fluid).  The disturbance potential phi_d = phi - z of the uniform
flow phi -> z satisfies d_n phi_d = -n_z.

Discretisation: Gauss-Legendre panels in the parameter t.  Smooth
interactions use the panel rule; for (target, panel) pairs closer than
``eta`` panel lengths (including the self panel and the mirror image of the
target across the symmetry axis, where the ring kernels have a second
logarithmic singularity) the panel integral is recomputed with a
geometrically graded Gauss-Legendre rule clustered at the nearest point, with
the density interpolated by the panel Lagrange basis and the geometry by the
panel Chebyshev polynomials.  On the self panel the geometric quantities
entering the kernel are computed by Chebyshev divided differences, avoiding
cancellation as source and target coalesce.

All densities in this problem are odd under t -> pi - t, so only the nodes
in [0, pi/2] are unknowns ("folded" matrices).
"""
import numpy as np
import scipy.linalg as sla

from .kernels import ring_kernels_dz
from .quad import cheb_divided_differences, cheb_eval, graded_rule


class ExteriorNeumann:

    def __init__(self, geom, eta=1.0, sigma=0.5, ng=10, depth=40, chunk=256):
        self.geom = g = geom
        self.P = P = geom.panels
        self.eta = eta
        Nh, p = P.Nh, P.p
        ti = slice(0, Nh)
        r_t, z_t = g.r[ti], g.z[ti]
        rho, zeta = g.r, g.z
        nr, nz = g.n
        with np.errstate(divide='ignore', invalid='ignore'):
            Gf, Df = ring_kernels_dz(r_t[:, None], rho[None, :],
                                     z_t[:, None] - zeta[None, :],
                                     nr[None, :], nz[None, :])
        Gf *= g.ds[None, :]
        Df *= g.ds[None, :]

        # ---- detect special (target, panel) pairs ---------------------------
        ends = cheb_eval(g.A, np.tile([-1.0, 1.0], (1, P.np, 1)))   # (2, np, 2)
        # sample points: nodes + endpoints, shape (2, np, p+2)
        samp = np.concatenate([np.stack([g.r, g.z]).reshape(2, P.np, p), ends], axis=2)
        ell = g.ds.reshape(P.np, p).sum(axis=1)
        self.ell = ell
        dx = r_t[:, None, None] - samp[0][None]
        dxm = -r_t[:, None, None] - samp[0][None]
        dzz = z_t[:, None, None] - samp[1][None]
        d_dir = np.sqrt(dx ** 2 + dzz ** 2).min(axis=2)   # (Nh, np)
        d_mir = np.sqrt(dxm ** 2 + dzz ** 2).min(axis=2)
        dmin = np.minimum(d_dir, d_mir)
        special = dmin < eta * ell[None, :]
        own = np.repeat(np.arange(P.nph), p)              # panel of each target
        special[np.arange(Nh), own] = True
        ii, pp = np.nonzero(special)
        self.n_special = len(ii)
        is_self = pp == own[ii]
        # local coordinate of the singular point for every pair
        ustar = np.empty(len(ii))
        uloc = np.tile(P.rule.x, P.nph)
        ustar[is_self] = uloc[ii[is_self]]
        ns = np.nonzero(~is_self)[0]
        if len(ns):
            use_mir = d_mir[ii[ns], pp[ns]] < d_dir[ii[ns], pp[ns]]
            X = np.where(use_mir, -r_t[ii[ns]], r_t[ii[ns]])
            Z = z_t[ii[ns]]
            ustar[ns] = self._closest(g.A[:, pp[ns], :], X, Z)

        # ---- graded quadrature on special pairs -------------------------------
        # grading depth: full for self pairs, ~log2(panel/distance) otherwise
        dloc = np.where(is_self, 0.0, 2.0 * dmin[ii, pp] / ell[pp])
        with np.errstate(divide='ignore'):
            depth_pair = np.where(is_self, depth,
                                  np.clip(np.ceil(np.log2(2.0 / np.maximum(dloc, 1e-300))) + 4,
                                          4, depth)).astype(int)
        # adjacency: source panel shares an endpoint with the target panel
        tp = own[ii]
        right_nb = pp == tp + 1
        left_nb = pp == tp - 1
        self._assemble_special(Gf, Df, ii, pp, ustar, is_self, left_nb, right_nb,
                               depth_pair, sigma, ng, chunk, r_t, z_t, uloc)

        self.Gfull, self.Dfull = Gf, Df
        # fold for odd densities: psi(pi - t) = -psi(t)
        self.S = Gf[:, :Nh] - Gf[:, ::-1][:, :Nh]
        self.K = Df[:, :Nh] - Df[:, ::-1][:, :Nh]
        self.lu = sla.lu_factor(0.5 * np.eye(Nh) - self.K)

    def _assemble_special(self, Gf, Df, ii, pp, ustar, is_self, left_nb, right_nb,
                          depth_pair, sigma, ng, chunk, r_t, z_t, uloc):
        g, P = self.geom, self.P
        p = P.p
        own = np.repeat(np.arange(P.nph), p)
        for D in np.unique(depth_pair):
            xi, om = graded_rule(sigma, ng, int(D))
            grp = np.nonzero(depth_pair == D)[0]
            for c0 in range(0, len(grp), chunk):
                sel = grp[c0:c0 + chunk]
                i_, p_, us = ii[sel], pp[sel], ustar[sel]
                sf, lnb, rnb = is_self[sel], left_nb[sel], right_nb[sel]
                # offsets delta = u* - u kept explicitly (u may round to u*)
                delta = np.concatenate([(us[:, None] + 1) * xi[None, :],
                                        -(1 - us[:, None]) * xi[None, :]], axis=1)
                u = us[:, None] - delta
                wu = np.concatenate([(us[:, None] + 1) * om[None, :],
                                     (1 - us[:, None]) * om[None, :]], axis=1)
                Ap = g.A[:, p_, :]
                Bp = g.B[:, p_, :]
                y = cheb_eval(Ap, u[None])
                yp = cheb_eval(Bp, u[None])
                cn = np.hypot(yp[0], yp[1])
                nrq, nzq = -yp[1] / cn, yp[0] / cn
                rt = r_t[i_][:, None]
                ztq = z_t[i_][:, None]
                dz = ztq - y[1]
                b = (rt - y[0]) ** 2 + dz ** 2
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndb = (nrq * (rt - y[0]) + nzq * dz) / b
                if sf.any():
                    k = np.nonzero(sf)[0]
                    v = us[k][:, None]
                    D1, D2 = cheb_divided_differences(Ap[:, k, :], u[k][None], v[None])
                    dl = delta[k]
                    dz[k] = dl * D1[1]
                    nd1 = D1[0] ** 2 + D1[1] ** 2
                    b[k] = dl ** 2 * nd1
                    ndb[k] = (nrq[k] * D2[0] + nzq[k] * D2[1]) / nd1
                nb = lnb | rnb
                if nb.any():
                    # x_t - y = [x_P(v) - x_P(e_P)] + [x_Q(e_Q) - x_Q(u)]
                    k = np.nonzero(nb)[0]
                    eQ = np.where(rnb[k], -1.0, 1.0)[:, None]   # shared end in Q
                    eP = -eQ                                   # shared end in P
                    v = uloc[i_[k]][:, None]
                    AP = g.A[:, own[i_[k]], :]
                    D1P, _ = cheb_divided_differences(AP, eP[None], v[None])
                    D1Q, D2Q = cheb_divided_differences(Ap[:, k, :], u[k][None],
                                                        np.broadcast_to(eQ, u[k].shape)[None])
                    dP = v - eP
                    dQ = eQ - u[k]
                    dxr = dP * D1P[0] + dQ * D1Q[0]
                    dxz = dP * D1P[1] + dQ * D1Q[1]
                    dz[k] = dxz
                    b[k] = dxr ** 2 + dxz ** 2
                    ndx = dP * (nrq[k] * D1P[0] + nzq[k] * D1P[1]) \
                        + dQ ** 2 * (nrq[k] * D2Q[0] + nzq[k] * D2Q[1])
                    ndb[k] = ndx / b[k]
                G, Dk = ring_kernels_dz(rt, y[0], dz, nrq, nzq, b, ndb)
                W = wu * (P.h[p_] / 2)[:, None] * cn
                Lm = P.rule.lagrange(u)
                bG = np.einsum('nq,nqj->nj', W * G, Lm)
                bD = np.einsum('nq,nqj->nj', W * Dk, Lm)
                cols = p_[:, None] * p + np.arange(p)[None, :]
                Gf[i_[:, None], cols] = bG
                Df[i_[:, None], cols] = bD

    @staticmethod
    def _closest(A, X, Z, nsamp=33, newton=4):
        """Local parameter u in [-1,1] of the point of the panel polynomial
        curves A (2, n, deg+1) closest to (X, Z)."""
        us = np.linspace(-1, 1, nsamp)
        y = cheb_eval(A, us[None, None, :])
        d2 = (y[0] - X[:, None]) ** 2 + (y[1] - Z[:, None]) ** 2
        u = us[np.argmin(d2, axis=1)]
        from numpy.polynomial import chebyshev as C
        A1 = np.stack([np.stack([C.chebder(A[c, q]) for q in range(A.shape[1])])
                       for c in range(2)])
        A2 = np.stack([np.stack([C.chebder(A1[c, q]) for q in range(A.shape[1])])
                       for c in range(2)])
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

    # ------------------------------------------------------------------------
    def solve(self, g_neu):
        """Solve (1/2 - K) psi = -S g for odd Neumann data g (half nodes)."""
        return sla.lu_solve(self.lu, -(self.S @ g_neu))

    def disturbance_potential(self):
        nz = self.geom.n[1][:self.P.Nh]
        return self.solve(-nz)

    def full_odd(self, f_half):
        return np.concatenate([f_half, -f_half[::-1]])

    def full_even(self, f_half):
        return np.concatenate([f_half, f_half[::-1]])
