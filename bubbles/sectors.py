"""Second variation of steady bubbles and rings by symmetry sector.

Steady bubbles are the critical points of E = |S| - (We/2) m(B) at fixed
volume, where m = int phi_d n_z dS is the axial added mass (its shape
derivative is int (|grad phi|^2 - 1) v dS).  The second variation for a
normal displacement v(s) cos(m phi), with parity even or odd under z -> -z, is

    E''(v) = int (|grad_S v|^2 + m^2 v^2/r^2 - |A|^2 v^2) dS
             + We int kappa_1 phi_s^2 v^2 dS + We int g psi dS,

g = (1/r) d_s(r v phi_s) the Neumann data of the flow perturbation psi cos(m phi)
(int g psi = -int_fluid |grad psi|^2 <= 0).  It is the self-adjoint form of the
Jacobian of R = H - (We/2)|grad phi|^2 - lam, so a zero eigenvalue in a sector
m >= 1 (or in the odd sectors) is a symmetry-breaking bifurcation, and the
m = 0 even sector holds the fold.  Translations are exact zero modes: n_z in
m = 0 odd and n_r in m = 1 even; they are projected out.

Mode-m ring kernels are the existing m = 0 kernels plus the azimuthal integral
of (cos m phi - 1) times the point kernel, which is regular at phi = 0
(Gauss-Legendre in a sinh-clustered variable).

Rings: the upper half of the cross-section is mirrored to a closed curve with
the bubble code's node layout (node k <-> node N-1-k is z -> -z), so the same
ExteriorNeumann solver applies; the base flow is the vortex-sheet surface
speed (any flux Q).  For m >= 1 and in m = 0 odd the flow perturbation is single
valued and neither the hole flux nor the circulation changes.  In m = 0 even the
single-valued solve holds the circulation fixed; at fixed flux the circulation
moves by -Gamma_1 dQ|_Gamma[v], with gamma_1 = V^{-1}[1] the pure flux flow,
Gamma_1 = 2 sum gamma_1 ds and dQ|_Gamma[v] = -int v gamma gamma_1 dS/(2 pi Gamma_1),
which adds a rank-one term:

    Hess_Q = Hess_Gamma - We/(2 pi Gamma_1) b b^T,   b = int v gamma gamma_1 dS.

Its zero along the zero-flux branch falls at the We maximum (r_h = 0.11622).

Monotonicity in m (We >= 0, fixed parity): E''_{m+1}(v) - E''_m(v) >=
(2m + 1) int v^2/r^2 dS, since g does not depend on m and the flow term
-D_m(psi_m) grows with m.  So the m = 1 even sector bounds all m >= 2
(RingSector.mc_bounds).
"""
import numpy as np
import scipy.linalg as sla

from . import bie
from .kernels import ring_kernels_dz as K0
from .ring import ring_theta

NU = 48
_xg, _wg = np.polynomial.legendre.leggauss(NU)


def make_kernel(m):
    """m-mode ring kernels (G, D) with the same call signature as ring_kernels_dz."""
    def kern(r, rho, dz, nrho, nzeta, b=None, ndx_over_b=None):
        G0, D0 = K0(r, rho, dz, nrho, nzeta, b=b, ndx_over_b=ndx_over_b)
        if m == 0:
            return G0, D0
        if b is None:
            b = (r - rho) ** 2 + dz ** 2
        ndx = nrho * (r - rho) + nzeta * dz if ndx_over_b is None else ndx_over_b * b
        shp = np.broadcast(r, rho, dz, nrho, nzeta, b).shape
        r_, rho_, b_ = [np.broadcast_to(x, shp) for x in (r, rho, b)]
        ndx_ = np.broadcast_to(ndx, shp)
        nr_ = np.broadcast_to(nrho, shp)
        rr = r_ * rho_
        eps = np.maximum(np.sqrt(np.maximum(b_, 1e-300) / rr), 1e-15)
        U = np.arcsinh(np.pi / eps)
        dG = np.zeros(shp)
        dD = np.zeros(shp)
        for x, w in zip(_xg, _wg):
            u = 0.5 * U * (x + 1)
            ph = eps * np.sinh(u)
            jac = 0.5 * U * w * eps * np.cosh(u)
            s2 = np.sin(0.5 * ph) ** 2
            d2 = b_ + 4 * rr * s2
            d = np.sqrt(d2)
            cm1 = -2 * np.sin(0.5 * m * ph) ** 2
            dG += jac * cm1 / d
            dD += jac * cm1 * (ndx_ - 2 * nr_ * r_ * s2) / (d2 * d)
        return G0 + dG * rho_ / (2 * np.pi), D0 + dD * rho_ / (2 * np.pi)
    return kern


def neumann_ops(geom, m):
    """Folded single-layer and LU of (1/2 - K) for the m-mode exterior Neumann
    problem, per parity of the density."""
    bie.ring_kernels_dz = make_kernel(m)
    try:
        op = bie.ExteriorNeumann(geom)
    finally:
        bie.ring_kernels_dz = K0
    Nh = geom.panels.Nh
    Gf, Df = op.Gfull, op.Dfull
    ops = {}
    for par, sg in (('odd', -1.0), ('even', 1.0)):
        S = Gf[:, :Nh] + sg * Gf[:, ::-1][:, :Nh]
        K = Df[:, :Nh] + sg * Df[:, ::-1][:, :Nh]
        ops[par] = (S, sla.lu_factor(0.5 * np.eye(Nh) - K))
    return ops


def _hessian(G, P, m, phs, We, V, ops):
    """(Hess, Gram) on the rows of V (functions on the full node set)."""
    r, c = G.r, G.c
    dS = 2 * np.pi * r * P.wf * c
    Vs = P.diff(V) / c
    A2 = G.kappa1 ** 2 + G.kappa2 ** 2
    Hs = (Vs * dS) @ Vs.T + (V * dS * (m ** 2 / r ** 2 - A2)) @ V.T
    Hk = (V * dS * G.kappa1 * phs ** 2) @ V.T
    Gn = P.diff(r * V * phs) / (r * c)
    Nh = P.Nh
    # the Neumann data has the parity opposite to v (phi_s is odd for bubbles' phi)
    return Hs, Hk, Gn, dS, Nh


def reduce_eigs(H, Gm, C=None, tol=1e-11, vectors=False):
    """Eigenvalues (ascending) of H x = mu Gm x on {C x = 0}, robust to a near-singular
    Gram; with vectors=True also the coefficient vectors x (columns, Gm-orthonormal)."""
    lam, U = np.linalg.eigh(Gm)
    keep = lam > tol * lam.max()
    B = U[:, keep] / np.sqrt(lam[keep])
    if C is not None:
        B = B @ sla.null_space(C @ B)
    Hq = B.T @ H @ B
    w, Y = np.linalg.eigh(0.5 * (Hq + Hq.T))
    return (w, B @ Y) if vectors else w


class Sector:
    """Sector m of the second variation at a bubble (bubbles.bubble.BubbleState)."""

    def __init__(self, st, m):
        self.st, self.m = st, m
        self.ops = neumann_ops(st.g, m) if m else None
        if m == 0:
            Nh = st.P.Nh
            Gf, Df = st.bie.Gfull, st.bie.Dfull
            self.ops = {}
            for par, sg in (('odd', -1.0), ('even', 1.0)):
                S = Gf[:, :Nh] + sg * Gf[:, ::-1][:, :Nh]
                K = Df[:, :Nh] + sg * Df[:, ::-1][:, :Nh]
                self.ops[par] = (S, sla.lu_factor(0.5 * np.eye(Nh) - K))

    def neumann(self, gh, par):
        S, lu = self.ops[par]
        return sla.lu_solve(lu, -(S @ gh))

    def hessian(self, We, parity, K, basis=None):
        """Basis sin^m t cos(k t) (k even <-> v even under z -> -z), k < K."""
        st, m = self.st, self.m
        P = st.P
        t = P.tf
        ks = np.arange(0 if parity == 'even' else 1, K, 2)
        V = np.sin(t)[None, :] ** m * np.cos(np.outer(ks, t)) if basis is None else basis
        return self._form(We, parity, V)

    def _form(self, We, parity, V):
        st, m = self.st, self.m
        g, P = st.g, st.P
        Hs, Hk, Gn, dS, Nh = _hessian(g, P, m, st.phi_s_full, We, V, self.ops)
        gpar = 'odd' if parity == 'even' else 'even'
        Psi = self.neumann(Gn[:, :Nh].T, gpar).T
        sg = -1.0 if gpar == 'odd' else 1.0
        Psif = np.concatenate([Psi, sg * Psi[:, ::-1]], axis=1)
        Hf = (Gn * dS) @ Psif.T
        Hess = Hs + We * (Hk + 0.5 * (Hf + Hf.T))
        return Hess, (V * dS) @ V.T, dict(V=V, dS=dS)

    def eigs(self, We, parity, K, nev=3):
        """Lowest eigenvalues, translations and (m = 0 even) volume changes removed.
        Returns (eigenvalues, Rayleigh quotient of the exact translation or None)."""
        m, g = self.m, self.st.g
        T = {(0, 'odd'): g.n[1], (1, 'even'): g.n[0]}.get((m, parity))
        H, Gm, d = self.hessian(We, parity, K)
        if T is not None:
            H, Gm, d = self.hessian(We, parity, K, basis=np.vstack([T, d['V']]))
            return reduce_eigs(H, Gm, C=Gm[0:1, :])[:nev], H[0, 0] / Gm[0, 0]
        C = (d['V'] @ d['dS'])[None, :] if (m == 0 and parity == 'even') else None
        return reduce_eigs(H, Gm, C)[:nev], None


class _FullPanels:
    def __init__(self, Pu):
        self.p, self.rule = Pu.p, Pu.rule
        self.nph, self.np = Pu.np, 2 * Pu.np
        self.Nh, self.N = Pu.N, 2 * Pu.N
        self.h = np.concatenate([Pu.h, Pu.h[::-1]])
        self.tf = np.concatenate([Pu.tf, 2 * np.pi - Pu.tf[::-1]])
        self.wf = np.concatenate([Pu.wf, Pu.wf[::-1]])

    def diff(self, f):
        F = f.reshape(f.shape[:-1] + (self.np, self.p))
        dF = np.einsum('ij,...pj->...pi', self.rule.Dm, F) * (2.0 / self.h)[:, None]
        return dF.reshape(f.shape)


class _FullGeom:
    """Ring cross-section (upper half from bubbles.ring.RingGeometry) mirrored
    to a closed curve, with the attributes ExteriorNeumann needs."""

    def __init__(self, gu, Pu):
        self.panels = _FullPanels(Pu)
        ev = lambda x: np.concatenate([x, x[::-1]])  # noqa: E731
        self.r, self.z = ev(gu.r), np.concatenate([gu.z, -gu.z[::-1]])
        nr, nz = -np.sin(gu.theta), np.cos(gu.theta)
        self.n = np.stack([ev(nr), np.concatenate([nz, -nz[::-1]])])
        self.c, self.ds = ev(gu.c), ev(gu.ds)
        self.kappa1, self.kappa2 = ev(gu.kappa1), ev(gu.kappa2)
        s = np.array([1.0, -1.0])[:, None, None]
        sgnA = (-1.0) ** np.arange(gu.A.shape[2])
        sgnB = (-1.0) ** np.arange(gu.B.shape[2])
        self.A = np.concatenate([gu.A, s * gu.A[:, ::-1, :] * sgnA[None, None, :]], axis=1)
        self.B = np.concatenate([gu.B, -s * gu.B[:, ::-1, :] * sgnB[None, None, :]], axis=1)


class RingSector:
    """Sector m of the second variation at a ring (bubbles.ring.RingState, any flux Q)."""

    def __init__(self, st, m):
        self.m = m
        self.G = _FullGeom(st.g, st.P)
        self.phs = np.concatenate([st.gamma, st.gamma[::-1]])
        # pure flux flow and its circulation, for the fixed-flux term in m = 0 even
        self.g1 = st.op.solve(np.ones_like(st.op.r))
        self.G1 = 2 * np.sum(self.g1 * st.g.ds)
        self.a = st.g.a
        self.ops = neumann_ops(self.G, m)

    @staticmethod
    def basis(parity, K, tau):
        """cos(k tau), k < K (even) or sin(k tau), 0 < k < K (odd), as rows."""
        return np.cos(np.outer(np.arange(0, K), tau)) if parity == 'even' else np.sin(np.outer(np.arange(1, K), tau))

    def translation(self, parity, tau=None):
        """The exact zero mode of the sector, n_z in 0 odd and n_r in 1 even (None in the
        others), at the nodes or at the parameters tau of the closed curve."""
        k = {(0, 'odd'): 1, (1, 'even'): 0}.get((self.m, parity))
        if k is None:
            return None
        if tau is None:
            return self.G.n[k]
        th = ring_theta(self.a, tau)[0]
        return -np.sin(th) if k == 0 else np.cos(th)

    def hessian(self, We, parity, K, extra=None, parts=False):
        """(Hess, Gram) on the rows `extra` followed by cos(k tau), k < K (even) or
        sin(k tau), 0 < k < K (odd), tau the parameter of the closed curve.  parts=True
        also returns dict(V, dS, Hs, Hk, Hf) with Hess = Hs + We (Hk + Hf)."""
        G, P, m = self.G, self.G.panels, self.m
        V = self.basis(parity, K, P.tf)
        if extra is not None:
            V = np.vstack([extra, V])
        Hs, Hk, Gn, dS, Nh = _hessian(G, P, m, self.phs, We, V, self.ops)
        gpar = 'odd' if parity == 'even' else 'even'
        S, lu = self.ops[gpar]
        Psi = sla.lu_solve(lu, -(S @ Gn[:, :Nh].T)).T
        sg = -1.0 if gpar == 'odd' else 1.0
        Psif = np.concatenate([Psi, sg * Psi[:, ::-1]], axis=1)
        Hf = (Gn * dS) @ Psif.T
        Hf = 0.5 * (Hf + Hf.T)
        H, Gm = Hs + We * (Hk + Hf), (V * dS) @ V.T
        return (H, Gm, dict(V=V, dS=dS, Hs=Hs, Hk=Hk, Hf=Hf)) if parts else (H, Gm)

    def eigs(self, We, parity, K=64, nev=3, constraint='Q', vectors=False, at=None):
        """Lowest eigenvalues with the translation (0 odd, 1 even) removed; in 0 even the
        volume and the hole flux (constraint='Q') or the circulation ('Gamma') are fixed.
        Returns (eigenvalues, Rayleigh quotient of the exact translation or None); with
        vectors=True also the eigenvectors as normal displacements (rows) at the nodes, or
        evaluated from their series at the curve parameters `at`."""
        if constraint not in ('Q', 'Gamma'):
            raise ValueError(f"constraint must be 'Q' or 'Gamma', not {constraint!r}")
        T = self.translation(parity)
        H, Gm, d = self.hessian(We, parity, K, extra=T, parts=True)
        C, tr = None, None
        if T is not None:
            C, tr = Gm[0:1, :], H[0, 0] / Gm[0, 0]
        elif (self.m, parity) == (0, 'even'):
            C = (d['V'] @ d['dS'])[None, :]
            if constraint == 'Q':
                b = (d['V'] * d['dS']) @ (self.phs * np.concatenate([self.g1, self.g1[::-1]]))
                H = H - We / (2 * np.pi * self.G1) * np.outer(b, b)
        w, X = reduce_eigs(H, Gm, C, vectors=True)
        if not vectors:
            return w[:nev], tr
        V = d['V']
        if at is not None:
            V = self.basis(parity, K, at)
            if T is not None:
                V = np.vstack([self.translation(parity, at), V])
        return w[:nev], tr, X[:, :nev].T @ V

    def mc_bounds(self, We, K=64):
        """Bracket on m_c, the largest unstable even sector (sectors 1..m_c are unstable by
        monotonicity in m), from the lowest m = 1 even eigenpair (mu_1, v_1) alone:

            mu(m) >= mu_1 + (m^2 - 1)/r_o^2,
            mu(m) <= mu_1 + (m^2 - 1) <r^-2>_v1 + We D_1,   D_1 = int |grad psi_1|^2/|v_1|^2.

        Returns dict(mu1, mc_lo, mc_hi, D1, r2): all m <= mc_lo unstable, all m > mc_hi
        stable (r2 = <r^-2>_v1)."""
        if self.m != 1:
            raise ValueError('mc_bounds needs the m = 1 sector')
        H, Gm, d = self.hessian(We, 'even', K, extra=self.translation('even'), parts=True)
        w, X = reduce_eigs(H, Gm, Gm[0:1, :], vectors=True)
        mu, c = float(w[0]), X[:, 0]
        v = c @ d['V']
        nrm = np.sum(v ** 2 * d['dS'])
        r2 = float(np.sum(v ** 2 * d['dS'] / self.G.r ** 2) / nrm)
        D1 = float(-(c @ d['Hf'] @ c) / nrm)
        out = dict(mu1=mu, mc_lo=0, mc_hi=0, D1=D1, r2=r2)
        if mu < 0:
            up = np.sqrt(1 + abs(mu) * self.G.r.max() ** 2)
            lo = np.sqrt(max(1 + (abs(mu) - We * D1) / r2, 0.0))
            out.update(mc_lo=max(1, int(np.ceil(lo)) - 1), mc_hi=int(np.ceil(up)) - 1)
        return out
