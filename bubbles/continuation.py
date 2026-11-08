"""Pseudo-arclength continuation of the bubble branch in (shape, lambda, We).

Unknown vector Y = (a_1..a_M, lambda, We) for a given parameter map S(t).
Each step:
  * panels: uniform with h <= c_h / M (resolve the Fourier modes) refined
    adaptively for the geometry; fixed during the corrector,
  * tangent = null vector of the extended Jacobian [dF/da, dF/dlam, dF/dWe],
  * Newton corrector on (F(Y) = 0, tau . (Y - Y_pred) = 0),
  * a posteriori checks: Fourier decay of a and the nodal residual max|R(t)|
    (spectral truncation); M is increased when needed,
  * periodically the parameter map is re-optimised for fast Fourier decay
    and the solution re-represented.
"""
import json
import time

import numpy as np

from .bubble import BubbleState
from .geometry import (ParamMap, adaptive_panels, decay_rate, map_family,
                       optimise_map, reparametrize)


class Branch:

    def __init__(self, M=16, pmap=None, panel_tol=1e-13, c_h=2.0, p=16,
                 newton_tol=1e-12, res_tol=1e-12, M_max=1024, M_margin=8,
                 map_every=4, bie_opts=None, log=print):
        self.M = M
        self.pmap = pmap or ParamMap()
        self.panel_tol = panel_tol
        self.c_h = c_h
        self.p = p
        self.newton_tol = newton_tol
        self.res_tol = res_tol
        self.M_max = M_max
        self.M_margin = M_margin
        self.map_every = map_every
        self.bie_opts = bie_opts or {}
        self.log = log
        self.points = []
        self.map_beta = None

    # ------------------------------------------------------------------
    def panels_for(self, a):
        M = len(a)
        return adaptive_panels(a, self.pmap, p=self.p, tol=self.panel_tol,
                               h_max=self.c_h / M)

    def state(self, a, panels):
        return BubbleState(a, panels, self.pmap, self.bie_opts)

    def eval(self, Y, panels, jac=True):
        M = len(Y) - 2
        st = self.state(Y[:M], panels)
        F = st.residual(Y[M + 1], Y[M])
        if not jac:
            return st, F, None
        J, dWe = st.jacobian(Y[M + 1])
        JY = np.hstack([J, dWe[:, None]])
        return st, F, JY

    @staticmethod
    def tangent(JY, ref=None, ref_tail_only=False):
        _, _, Vt = np.linalg.svd(JY)
        tau = Vt[-1]
        if ref is not None:
            s = np.dot(tau[-2:], ref[-2:]) if ref_tail_only else np.dot(tau, ref)
            if s < 0:
                tau = -tau
        return tau / np.linalg.norm(tau)

    def valid(self, Y):
        """Geometric admissibility: embedded curve in the half plane, i.e.
        r > 0 and z > 0 on the open upper half (0 < t < pi/2)."""
        from .geometry import Geometry, Panels
        M = len(Y) - 2
        g = Geometry(Y[:M], Panels.uniform(16), self.pmap)
        Nh = g.panels.Nh
        return bool(np.all(np.isfinite(g.r)) and g.zN > 0 and g.r[:Nh].min() > 0
                    and g.z[:Nh - 1].min() > 0)

    def correct(self, Y0, tau, ds, panels, maxit=12):
        """Newton for F(Y) = 0, tau.(Y - Y0) = ds starting at Y0 + ds tau."""
        Y = Y0 + ds * tau
        hist = []
        for it in range(maxit):
            if not self.valid(Y):
                return None, hist + [('invalid geometry', it)], None
            try:
                st, F, JY = self.eval(Y, panels)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
                return None, hist + [(str(e)[:40], it)], None
            G = np.append(F, np.dot(tau, Y - Y0) - ds)
            A = np.vstack([JY, tau])
            dY = np.linalg.solve(A, -G)
            Y = Y + dY
            hist.append((float(np.abs(F).max()), float(np.abs(dY).max())))
            if not np.all(np.isfinite(Y)):
                return None, hist, None
            if hist[-1][1] < self.newton_tol:
                break
            if it >= 3 and hist[-1][1] > 0.3 * hist[-2][1] and hist[-1][1] > 1e3 * self.newton_tol:
                return None, hist, None
        else:
            if hist[-1][1] > 1e3 * self.newton_tol:
                return None, hist, None
        st, F, JY = self.eval(Y, panels)
        return Y, hist, (st, F, JY)

    @staticmethod
    def resize(Y, M_new):
        M = len(Y) - 2
        a = np.zeros(M_new)
        a[:min(M, M_new)] = Y[:min(M, M_new)]
        return np.concatenate([a, Y[M:]])

    def record(self, Y, st, hist, ds):
        M = len(Y) - 2
        Rn = st.residual_nodal(Y[M + 1], Y[M])
        d = st.diagnostics()
        rate, needed = decay_rate(Y[:M])
        J, _ = st.jacobian(Y[M + 1])
        sgn, logdet = np.linalg.slogdet(J)
        sv = np.linalg.svd(J, compute_uv=False)
        d.update(We=float(Y[M + 1]), lam=float(Y[M]), M=M, N=st.P.N,
                 n_panels=st.P.np, res_nodal=float(np.abs(Rn).max()),
                 res_galerkin=float(np.abs(st.residual(Y[M + 1], Y[M])).max()),
                 res_hi=st.residual_hi(Y[M + 1], Y[M]),
                 det_sign=float(sgn), sigma_min=float(sv[-1]),
                 decay=float(rate), modes_needed=needed, ds=ds,
                 newton_its=len(hist), a=Y[:M].tolist(),
                 map_b=self.pmap.b.tolist(),
                 breaks_half=st.P.breaks_half.tolist())
        self.points.append(d)
        return d

    # ------------------------------------------------------------------
    def start_at_sphere(self, We0=1e-3):
        from .newton import newton_fixed_we
        a0 = np.zeros(self.M)
        panels = self.panels_for(a0)
        a, lam, st, hist = newton_fixed_we(a0, 2.0, We0, panels, self.pmap,
                                           tol=self.newton_tol, bie_opts=self.bie_opts)
        return np.concatenate([a, [lam, We0]])

    def _reconverge(self, Y, tau_ref):
        """Re-solve after a change of representation (ds = 0 corrector)."""
        M = len(Y) - 2
        panels = self.panels_for(Y[:M])
        st, F, JY = self.eval(Y, panels)
        tau = self.tangent(JY, tau_ref, ref_tail_only=True)
        Yc, hist, res = self.correct(Y, tau, 0.0, panels)
        if Yc is None:
            raise RuntimeError(f"re-convergence failed {hist}")
        tau = self.tangent(res[2], tau)
        return Yc, tau, res[0], hist

    def maybe_remap(self, Y, tau, force=False, resolve=None):
        """Re-optimise the parameter map; ``resolve(Y)`` re-converges a
        re-represented point and returns (Y, tau, state)."""
        resolve = resolve or (lambda Yn: self._reconverge(Yn, tau)[:3])
        M = len(Y) - 2
        a = Y[:M]
        rate_old, needed_old = decay_rate(a)
        pm, beta, rate, needed = optimise_map(a, self.pmap, n_par=1, M_test=max(2 * M, 96))
        if not force and not (rate > 1.1 * rate_old):
            return Y, tau, False
        M_new = int(min(max(needed + self.M_margin, 16), self.M_max))
        M_new = int(np.ceil(M_new / 4) * 4)
        a_new = reparametrize(a, self.pmap, pm, M_new)
        old_map = self.pmap
        self.pmap = pm
        Yn = np.concatenate([a_new, Y[M:]])
        try:
            Yc, taun, st = resolve(Yn)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            self.pmap = old_map
            self.log(f"   remap failed ({e}); keeping old map")
            return Y, tau, False
        self.log(f"   remap: beta={beta[0]:+.4f} decay {rate_old:.3f}->{rate:.3f} "
                 f"M {M}->{M_new}")
        self.map_beta = beta
        return Yc, taun, True

    def maybe_increase_M(self, Y, tau, st, resolve=None):
        resolve = resolve or (lambda Yn: self._reconverge(Yn, tau)[:3])
        M = len(Y) - 2
        Rn = st.residual_hi(Y[M + 1], Y[M])
        rate, needed = decay_rate(Y[:M])
        if (needed > M - self.M_margin // 2 or Rn > self.res_tol) and M < self.M_max:
            M_new = int(min(max(needed + self.M_margin, int(1.2 * M)), self.M_max))
            M_new = int(np.ceil(M_new / 4) * 4)
            self.log(f"   increasing M {M} -> {M_new} (needed {needed}, res_hi {Rn:.1e})")
            Yn = self.resize(Y, M_new)
            Yc, taun, st = resolve(Yn)
            return Yc, taun, True
        return Y, tau, False

    def run_natural(self, Y, name, values, save=None, stop=None, max_halvings=8):
        """Natural-parameter continuation in a geometric functional
        (e.g. 'log_z_pole'), We being an unknown.  Polynomial extrapolation
        predictor in the parameter."""
        phi = geometric_functional(name)
        hist_p, hist_Y = [], []
        M = len(Y) - 2
        hist_p.append(phi(Y[:M], self.pmap))
        hist_Y.append(Y.copy())
        values = list(values)
        k = 0
        t0 = time.time()
        halvings = 0
        while k < len(values):
            val = values[k]
            M = len(Y) - 2
            # predictor
            n = min(len(hist_p), 3)
            Ps = np.array(hist_p[-n:])
            Ys = np.array([self.resize(y, M) for y in hist_Y[-n:]])
            if n == 1:
                Yp = Ys[0].copy()
            else:
                Vm = np.vander(Ps - Ps[-1], n)
                coef = np.linalg.solve(Vm, Ys)
                Yp = np.vander(np.array([val - Ps[-1]]), n) @ coef
                Yp = Yp[0]
            ok = self.valid(Yp)
            if ok:
                try:
                    Yc, st, JY, hist = solve_constrained(self, Yp, name, val)
                    ok = self.valid(Yc) and hist[-1][1] < 1e3 * self.newton_tol
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    ok = False
            if not ok:
                halvings += 1
                if halvings > max_halvings:
                    self.log("   too many halvings, stopping")
                    break
                values.insert(k, 0.5 * (val + hist_p[-1]))
                self.log(f"   step failed; inserting {values[k]:.6g}")
                continue
            halvings = 0
            Y = Yc

            def resolve(Yn, val=val):
                Yr, str_, JYr, hr = solve_constrained(self, Yn, name, val)
                return Yr, None, str_
            Y, _, changed = self.maybe_increase_M(Y, None, st, resolve=resolve)
            if changed:
                st = self.state(Y[:-2], self.panels_for(Y[:-2]))
            if self.map_every and (len(self.points) % self.map_every == 0):
                Y, _, changed = self.maybe_remap(Y, None, resolve=resolve)
                if changed:
                    st = self.state(Y[:-2], self.panels_for(Y[:-2]))
                    hist_p, hist_Y = [], []   # representation changed
            hist_p.append(phi(Y[:-2], self.pmap))
            hist_Y.append(Y.copy())
            d = self.record(Y, st, hist, 0.0)
            self.log(f"{len(self.points) - 1:4d} We={d['We']:.12f} lam={d['lam']:+.10f} "
                     f"zp={d['z_pole']:.4e} chi_ext={d['aspect_ext']:.6f} M={d['M']} "
                     f"N={d['N']} res_hi={d['res_hi']:.1e} its={len(hist)} "
                     f"t={time.time() - t0:.0f}s")
            if save:
                self.save(save)
            if stop and stop(d):
                break
            k += 1
        return Y

    def run(self, Y, ds=0.05, ds_min=1e-7, ds_max=0.2, n_steps=200,
            stop=None, save=None, tau=None):
        M = len(Y) - 2
        panels = self.panels_for(Y[:M])
        st, F, JY = self.eval(Y, panels)
        ref = np.zeros(len(Y))
        ref[-1] = 1.0
        tau = self.tangent(JY, tau if tau is not None else ref)
        if not self.points:
            self.record(Y, st, [], 0.0)
        step = 0
        t0 = time.time()
        while step < n_steps:
            M = len(Y) - 2
            panels = self.panels_for(Y[:M])
            Ynew, hist, res = self.correct(Y, tau, ds, panels)
            if Ynew is None:
                ds *= 0.5
                self.log(f"   step rejected, ds -> {ds:.3e} (last {hist[-1] if hist else ''})")
                if ds < ds_min:
                    self.log("   ds below minimum, stopping")
                    break
                continue
            st, F, JY = res
            tau = self.tangent(JY, tau)
            Y = Ynew
            Y, tau, changed = self.maybe_increase_M(Y, tau, st)
            if changed:
                M = len(Y) - 2
                st = self.state(Y[:M], self.panels_for(Y[:M]))
            step += 1
            if self.map_every and step % self.map_every == 0:
                Y, tau, changed = self.maybe_remap(Y, tau)
                if changed:
                    M = len(Y) - 2
                    st = self.state(Y[:M], self.panels_for(Y[:M]))
            d = self.record(Y, st, hist, ds)
            self.log(f"{len(self.points) - 1:4d} We={d['We']:.12f} lam={d['lam']:+.10f} "
                     f"chi={d['aspect']:.8f} zp={d['z_pole']:.4e} Hp={d['H_pole']:+.3e} "
                     f"M={d['M']} N={d['N']} res_hi={d['res_hi']:.1e} "
                     f"its={len(hist)} ds={ds:.2e} t={time.time() - t0:.0f}s")
            if save:
                self.save(save)
            if stop and stop(d):
                break
            nits = len(hist)
            if nits <= 3:
                ds = min(ds * 1.5, ds_max)
            elif nits >= 6:
                ds *= 0.7
        return Y, tau

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.points, f)


def geometric_functional(name):
    """Scalar functionals of the shape (computed without the BIE)."""
    from .geometry import Geometry, Panels

    def f(a, pmap):
        g = Geometry(a, Panels.uniform(16), pmap)
        if name == 'z_pole':
            return g.zN
        if name == 'log_z_pole':
            return np.log(g.zN)
        raise KeyError(name)
    return f


def solve_constrained(br, Y, name, value, panels=None, maxit=15, tol=None, h=1e-7):
    """Newton for F(Y) = 0 together with Phi(a) = value (Phi geometric).

    We is an unknown; useful near folds in We and for approaching the
    touching limit with the pole height as parameter."""
    tol = tol or br.newton_tol
    phi = geometric_functional(name)
    M = len(Y) - 2
    panels = panels or br.panels_for(Y[:M])
    hist = []
    for it in range(maxit):
        st, F, JY = br.eval(Y, panels)
        a = Y[:M]
        p0 = phi(a, br.pmap)
        grad = np.zeros(M + 2)
        for j in range(M):
            e = np.zeros(M)
            e[j] = h
            grad[j] = (phi(a + e, br.pmap) - phi(a - e, br.pmap)) / (2 * h)
        A = np.vstack([JY, grad])
        G = np.append(F, p0 - value)
        dY = np.linalg.solve(A, -G)
        Y = Y + dY
        hist.append((float(np.abs(F).max()), float(np.abs(dY).max())))
        if hist[-1][1] < tol:
            break
    st, F, JY = br.eval(Y, panels)
    return Y, st, JY, hist
