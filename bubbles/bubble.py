"""Residual and Jacobian of the steady bubble problem.

Unknowns X = (a_1, ..., a_M, lambda) (tangent-angle coefficients and the
Bernoulli constant), parameter We.  On the surface S of the unit-volume
bubble

    R(t) = H(t) - (We/2) q(t) - lambda,      q = |grad phi|^2 = (d phi/ds)^2,

and the discrete equations are the Galerkin coefficients

    F_k = (4/pi) int_0^{pi/2} R(t) cos(2kt) dt,   k = 0..M.

The Jacobian uses the shape derivative of the exterior Neumann problem: for
a boundary displacement dx = v n + w T, the derivative phi' of the potential
is harmonic, decays, and satisfies d_n phi' = div_S (v grad_S phi); the
derivative of q at the displaced material point is

    dq = 2 phi_s d_s phi' - 2 kappa_1 v phi_s^2 + w d_s (phi_s^2).
"""
import numpy as np

from .bie import ExteriorNeumann
from .geometry import Geometry, ParamMap


class BubbleState:
    """Everything computed for one shape (geometry, potential, residual)."""

    def __init__(self, a, panels, pmap=None, bie_opts=None):
        self.a = np.asarray(a, float)
        self.M = len(self.a)
        self.P = P = panels
        self.pmap = pmap or ParamMap()
        self.g = g = Geometry(self.a, P, self.pmap)
        self.bie = ExteriorNeumann(g, **(bie_opts or {}))
        Nh = P.Nh
        self.phid = self.bie.disturbance_potential()
        phi = g.z[:Nh] + self.phid
        self.phi = phi
        phi_t = P.diff(self.bie.full_odd(phi))
        self.phi_s_full = phi_t / g.c
        self.phi_s = self.phi_s_full[:Nh]
        self.q = self.phi_s ** 2
        self.H = g.H[:Nh]
        th = P.tf[:Nh]
        k = np.arange(self.M + 1)
        # Galerkin projector onto cos(2kt), k = 0..M
        self.proj = (4.0 / np.pi) * np.cos(2 * np.outer(k, th)) * P.wf[:Nh][None, :]
        # projector onto the unresolved modes M < k <= 2M (truncation indicator)
        k_hi = np.arange(self.M + 1, 2 * self.M + 1)
        self.proj_hi = (4.0 / np.pi) * np.cos(2 * np.outer(k_hi, th)) * P.wf[:Nh][None, :]

    def residual_nodal(self, We, lam):
        return self.H - 0.5 * We * self.q - lam

    def residual(self, We, lam):
        return self.proj @ self.residual_nodal(We, lam)

    def residual_hi(self, We, lam):
        """max |Galerkin coefficient| of R for modes M < k <= 2M."""
        return float(np.abs(self.proj_hi @ self.residual_nodal(We, lam)).max())

    # ------------------------------------------------------------------
    def shape_variations(self):
        """Nodal (full) displacement fields dx_j and curvature variations dH_j
        for the unit perturbations a_j -> a_j + 1 (with volume rescaling)."""
        g, P = self.g, self.P
        t = P.tf
        M = self.M
        j = np.arange(1, M + 1)
        dth = np.sin(2 * np.outer(j, t))                  # (M, N)
        dtht = (2 * j)[:, None] * np.cos(2 * np.outer(j, t))
        st, ct = np.sin(g.theta), np.cos(g.theta)
        Sp = g.Sp
        dr1 = P.cumint(-Sp * st * dth)
        dz1 = P.cumint(Sp * ct * dth)
        half = np.arange(P.N) < P.Nh
        dz1 = dz1 - (P.wf * half * Sp * ct * dth).sum(axis=1)[:, None]
        r1 = g.r1
        dV1 = -np.pi * P.integrate(2 * r1 * dr1 * Sp * st + r1 ** 2 * Sp * ct * dth)
        dLL = -dV1 / (3.0 * g.V1)                           # dL / L
        L = g.L
        dx = np.stack([L * dr1 + dLL[:, None] * g.r,
                       L * dz1 + dLL[:, None] * g.z])       # (2, M, N)
        dH1 = -dtht / Sp - ct * dth / r1 + st * dr1 / r1 ** 2
        dH = dH1 / L - g.H * dLL[:, None]
        return dx, dH

    def jacobian(self, We):
        """d F / d (a, lambda) and d F / d We."""
        g, P, bie = self.g, self.P, self.bie
        Nh = P.Nh
        dx, dH = self.shape_variations()
        n, T = g.n, g.T
        v = dx[0] * n[0] + dx[1] * n[1]                    # (M, N)
        w = dx[0] * T[0] + dx[1] * T[1]
        phs = self.phi_s_full
        # Neumann data of phi': (1/(r c)) d/dt (r v phi_s)
        gneu = P.diff(g.r * v * phs) / (g.r * g.c)
        dphi = bie.solve(gneu[:, :Nh].T).T                 # (M, Nh)
        dphi_s = (P.diff(np.concatenate([dphi, -dphi[:, ::-1]], axis=1)) / g.c)[:, :Nh]
        qs = P.diff(phs ** 2) / g.c                        # d_s q (full)
        dq = (2 * phs[:Nh] * dphi_s - 2 * g.kappa1[:Nh] * v[:, :Nh] * phs[:Nh] ** 2
              + w[:, :Nh] * qs[:Nh])
        dR = dH[:, :Nh] - 0.5 * We * dq                     # (M, Nh)
        J = np.empty((self.M + 1, self.M + 1))
        J[:, :self.M] = self.proj @ dR.T
        J[:, self.M] = self.proj @ (-np.ones(Nh))
        dWe = self.proj @ (-0.5 * self.q)
        return J, dWe

    # ------------------------------------------------------------------
    def diagnostics(self):
        g = self.g
        Nh = self.P.Nh
        # equatorial radius at t = pi/2 from the polynomial of the last half panel
        from .quad import cheb_eval
        Aeq = g.A[:, self.P.nph - 1, :]
        req = float(cheb_eval(Aeq[0][None], np.array([[1.0]]))[0, 0])
        zN = g.zN
        zmax = g.z[:Nh].max()
        # added-mass coefficient C_A = (1/|B|) int_S phi_d n_z dS (= 1/2 for a ball);
        # the disturbance kinetic energy is (rho/2) C_A |B| V^2
        Nh = self.P.Nh
        CA = 2 * (2 * np.pi * np.sum(self.phid * g.n[1][:Nh] * g.r[:Nh] * g.ds[:Nh])) / (4 * np.pi / 3)
        return dict(r_eq=req, z_pole=zN, z_max=zmax, aspect=req / zN,
                    star_margin=float(star_margin(g)),
                    aspect_ext=req / zmax, L=g.L, area=g.area(),
                    H_pole=float(self._H_pole()), kmax=float(np.abs(g.kappa1).max()),
                    added_mass=float(CA), u_max=float(np.sqrt(self.q.max())))

    def _H_pole(self):
        """H at t = 0: kappa_1 = kappa_2 = -theta_t(0)/(L S'(0))."""
        from .geometry import theta_eval
        _, tht0 = theta_eval(self.a, np.array([0.0]))
        return 2 * (-tht0[0]) / (self.g.L * self.pmap.Sp(np.array([0.0]))[0])


def star_margin(g):
    """min over the surface of sin(angle between position vector and tangent).

    Positive iff the bubble is strictly star-shaped with respect to its centre,
    i.e. a radial graph over the sphere as in the Meyer-Niebel-Seis setting.
    The minimum is located on the panel polynomials to ~1e-14.
    """
    from scipy.optimize import minimize_scalar
    from .quad import cheb_eval
    P = g.panels
    Nh = P.Nh

    def f_nodes():
        r, z = g.r[:Nh], g.z[:Nh]
        rp, zp = g.xp[0][:Nh], g.xp[1][:Nh]
        return (z * rp - r * zp) / (np.hypot(r, z) * np.hypot(rp, zp))

    fn = f_nodes()
    best = fn.min()
    for k in np.argsort(fn)[:3]:
        q = k // P.p

        def f(u, q=q):
            uu = np.array([[[u]]])
            x = cheb_eval(g.A[:, q:q + 1, :], uu)[:, 0, 0]
            xp = cheb_eval(g.B[:, q:q + 1, :], uu)[:, 0, 0]
            return (x[1] * xp[0] - x[0] * xp[1]) / (np.hypot(*x) * np.hypot(*xp))
        res = minimize_scalar(f, bounds=(-1, 1), method='bounded',
                              options=dict(xatol=1e-10))
        best = min(best, res.fun)
    return best
