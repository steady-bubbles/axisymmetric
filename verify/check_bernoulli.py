"""Independent Bernoulli / geometry check of computed zero-flux ring bubbles.

Usage: OMP_NUM_THREADS=1 python check_bernoulli.py up:17 up:23 thin:4 ...

Independent pieces (not using the solver's geometry code):
  * curve: x'(t) = L S'(t) (cos th, sin th) integrated spectrally on the full
    closed curve t in [0, 2 pi) (lower half = t in (pi, 2 pi), mirror symmetry
    of the parametrisation), th and S' synthesised by FFT from a, b.
  * curvature F: Fourier differentiation of the sampled POINTS (r(t), z(t))
    (parametrisation-invariant formula), no use of theta_t.
  * curvature S: periodic quintic spline through the solver's own dense
    polyline (ring_curve_points), mirrored and parametrised by chord length.
  * Bernoulli residual with gamma interpolated (panel barycentric and a global
    cosine fit) on dense points between the nodes.
  * volume by trapezoid (spectral) and by Richardson-extrapolated polygon.
  * force identity int_upper H r dr = r_o + r_h.
  * interior stream function (Lamb's formula with ellipk/ellipe) = Q = 0.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import ellipe, ellipk

warnings.filterwarnings('ignore')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)
from bubbles.ring import RingMap, RingPanels, RingState, ring_curve_points, ring_theta  # noqa: E402

VOL = 4 * np.pi / 3
FILES = dict(up=os.path.join(REPO, 'results/rings/rings_up.json'),
             thin=os.path.join(REPO, 'results/rings/rings_thin.json'),
             thin2=os.path.join(REPO, 'results/rings/rings_thin2.json'),
             thin7=os.path.join(REPO, 'results/rings/rings_thin7.json'),
             hp=os.path.join(REPO, 'results/rings/rings_up_hp.json'))


# --------------------------------------------------------------------------- curve
def synth(coef_sin=None, coef_cos=None, c0=0.0, Nt=2 ** 16):
    """Values on t_j = 2 pi j/Nt of c0 + sum a_k sin kt + sum b_k cos kt (via FFT)."""
    c = np.zeros(Nt, complex)
    c[0] = c0
    if coef_sin is not None:
        k = np.arange(1, len(coef_sin) + 1)
        c[k] += coef_sin / (2j)
        c[-k] -= coef_sin / (2j)
    if coef_cos is not None:
        k = np.arange(1, len(coef_cos) + 1)
        c[k] += coef_cos / 2
        c[-k] += coef_cos / 2
    return np.real(np.fft.ifft(c) * Nt)


class FourierCurve:
    def __init__(self, a, b, rho_h, L=None, Nt=2 ** 16):
        self.Nt = Nt
        t = 2 * np.pi * np.arange(Nt) / Nt
        th = np.pi / 2 - t + synth(coef_sin=a, Nt=Nt)
        Sp = synth(coef_cos=np.asarray(b), c0=1.0, Nt=Nt) / np.pi
        rp1, zp1 = Sp * np.cos(th), Sp * np.sin(th)
        n = np.fft.fftfreq(Nt, 1.0 / Nt)
        self.n = n
        R1, Z1 = np.fft.fft(rp1) / Nt, np.fft.fft(zp1) / Nt
        self.mean_rp, self.mean_zp = R1[0].real, Z1[0].real   # closure: mean z' = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ker = np.where(n != 0, 1.0 / (1j * n), 0.0)
        ker[Nt // 2] = 0.0
        dr1 = np.real(np.fft.ifft(R1 * ker) * Nt - np.sum(R1 * ker))   # r - r_h (unit L)
        r1 = rho_h + dr1
        z1 = np.real(np.fft.ifft(Z1 * ker) * Nt - np.sum(Z1 * ker))
        # unit-L volume: V = -pi oint r^2 z' dt (clockwise curve)
        V1 = -np.pi * np.sum(r1 ** 2 * zp1) * (2 * np.pi / Nt)
        self.V1 = V1
        self.L_own = (VOL / V1) ** (1 / 3)
        L = self.L_own if L is None else L
        self.L = L
        self.t, self.r, self.z = t, L * r1, L * z1
        self.r_h0, self.dr = L * rho_h, L * dr1     # relative coordinate (less roundoff)
        self.tail_r = np.abs(R1[np.abs(n) > Nt / 4]).max() / np.abs(R1).max()
        # spectra of the POINTS (for the fit-based curvature)
        self.Rh, self.Zh = np.fft.fft(self.dr) / Nt, np.fft.fft(self.z) / Nt
        self.Rh[Nt // 2] = 0
        self.Zh[Nt // 2] = 0

    def deriv_grid(self):
        n = self.n
        f = lambda X, p: np.real(np.fft.ifft(X * (1j * n) ** p) * self.Nt)
        return f(self.Rh, 1), f(self.Zh, 1), f(self.Rh, 2), f(self.Zh, 2)

    def eval(self, t, kmax=None):
        """Trig interpolant of the points and its derivatives at arbitrary t."""
        Nt = self.Nt
        kmax = kmax or 260
        sel = np.abs(self.n) <= kmax
        n, R, Z = self.n[sel], self.Rh[sel], self.Zh[sel]
        out = []
        for ch in np.array_split(np.asarray(t), max(1, len(t) // 512)):
            E = np.exp(1j * np.outer(ch, n))
            row = []
            for p in range(3):
                fac = (1j * n) ** p
                row += [np.real(E @ (R * fac)), np.real(E @ (Z * fac))]
            row[0] = row[0] + self.r_h0
            out.append(np.array(row))
        return np.concatenate(out, axis=1)   # r, z, r_t, z_t, r_tt, z_tt


def curvatures(rt, zt, rtt, ztt, r):
    sp = np.hypot(rt, zt)
    k1 = -(rt * ztt - zt * rtt) / sp ** 3
    k2 = -(zt / sp) / r
    return k1, k2


# --------------------------------------------------------------------------- gamma interpolation
def gamma_panel(st, t):
    P = st.P
    idx = np.clip(np.searchsorted(P.b, t, side='left'), 0, P.np - 1)
    u = np.clip((2 * t - P.a[idx] - P.b[idx]) / P.h[idx], -1, 1)
    Lm = P.rule.lagrange(u)
    return np.einsum('nj,nj->n', Lm, st.gamma.reshape(P.np, P.p)[idx])


def gamma_cosfit(st, K):
    P = st.P
    w = np.sqrt(P.wf)
    A = np.cos(np.outer(P.tf, np.arange(K + 1)))
    c = np.linalg.lstsq(A * w[:, None], st.gamma * w, rcond=None)[0]
    fitres = np.abs(A @ c - st.gamma).max()
    return c, fitres


def cos_eval(c, t):
    return np.cos(np.outer(t, np.arange(len(c)))) @ c


# --------------------------------------------------------------------------- stream function
def psi_ring_lamb(r, z, rho, zeta):
    """Stokes stream function of a unit-circulation ring (Lamb), own implementation."""
    r1sq = (r - rho) ** 2 + (z - zeta) ** 2
    r2sq = (r + rho) ** 2 + (z - zeta) ** 2
    m = 4 * r * rho / r2sq                         # k^2
    return np.sqrt(r2sq) / (4 * np.pi) * ((2 - m) * ellipk(m) - 2 * ellipe(m))


def check_ring(tag, idx):
    p = json.load(open(FILES[tag]))[idx]
    t0 = time.time()
    a = np.array(p['a'])
    rm = RingMap(np.array(p['map_b']))
    P = RingPanels(np.array(p['breaks']))
    st = RingState(a, p['rho_h'], rm, P)
    g = st.g
    We, lam = p['We'], p['lam']
    out = dict(ring=f"{tag}[{idx}]", r_h_json=p['r_h'], We=We, lam=lam, N=P.N,
               res_hi_json=p['res_hi'], solver_F=float(np.abs(st.F(We, lam)).max()))

    # ---- (1) independent curve
    fc = FourierCurve(a, rm.b, p['rho_h'], L=g.L)
    fown = FourierCurve(a, rm.b, p['rho_h'], L=None)
    Nt = fc.Nt
    out['L_solver'] = g.L
    out['L_own_rel_diff'] = fown.L_own / g.L - 1
    out['closure_mean_zp'] = fc.mean_zp
    out['closure_solver_z_end'] = g.z_end1
    out['spectral_tail_rp'] = fc.tail_r
    r_h, r_o = fc.r[0], fc.r[Nt // 2]
    out['r_h_own'], out['r_o_own'] = r_h, r_o
    out['r_h_diff'] = r_h - p['r_h']
    out['r_o_diff'] = r_o - p['r_o']
    out['z_at_pi_own'] = fc.z[Nt // 2]
    # z_max: grid max then Newton on z_t = 0
    j = np.argmax(fc.z)
    tm = fc.t[j]
    for _ in range(8):
        v = fc.eval(np.array([tm]))
        tm -= v[3, 0] / v[5, 0]
    zmax = fc.eval(np.array([tm]))[1, 0]
    rzmax = fc.eval(np.array([tm]))[0, 0]
    out['z_max_own'] = zmax
    out['z_max_json'] = p['z_max']
    out['z_max_diff'] = zmax - p['z_max']
    out['z_max_nodes_diff'] = g.z.max() - p['z_max']
    out['r_at_zmax'] = rzmax

    # node positions: my curve vs solver nodes
    vn = fc.eval(P.tf)
    out['pos_nodes_maxdiff'] = float(np.max(np.hypot(vn[0] - g.r, vn[1] - g.z)))
    # solver's dense polyline vs my curve
    nS = 20001
    pts, _ = ring_curve_points(np.concatenate([a, [lam, p['rho_h'], We]]), rm, n=nS)
    tS = np.linspace(0, np.pi, nS)
    vS = fc.eval(tS)
    out['pos_polyline_maxdiff'] = float(np.max(np.hypot(vS[0] - pts[:, 0], vS[1] - pts[:, 1])))

    # curvature, method F (Fourier differentiation of the points) at the nodes
    k1F_n, k2F_n = curvatures(vn[2], vn[3], vn[4], vn[5], vn[0])
    sc1 = np.abs(g.kappa1).max()
    sc2 = np.abs(g.kappa2).max()
    out['kappa1_F_vs_nodes'] = float(np.abs(k1F_n - g.kappa1).max() / sc1)
    out['kappa2_F_vs_nodes'] = float(np.abs(k2F_n - g.kappa2).max() / sc2)
    # curvature, method S (quintic periodic spline on solver polyline, chord length)
    for nsub in (2, 8, 32, 64, 128):
        q = pts[::nsub]
        if (nS - 1) % nsub:
            q = np.vstack([q, pts[-1]])
        tq = tS[::nsub] if (nS - 1) % nsub == 0 else np.append(tS[::nsub], tS[-1])
        closed = np.vstack([q, q[-2:0:-1] * [1, -1], q[:1]])    # mirrored, closed
        d = np.hypot(*np.diff(closed, axis=0).T)
        s = np.concatenate([[0], np.cumsum(d)])
        spl = make_interp_spline(s, closed, k=5, bc_type='periodic')
        m = len(q)
        sq = s[:m]
        d1, d2 = spl(sq, 1), spl(sq, 2)
        k1S, k2S = curvatures(d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1], q[:, 0])
        th, tht = ring_theta(a, tq)
        cS = g.L * rm.Sp(tq)
        k1sol, k2sol = -tht / cS, -np.sin(th) / q[:, 0]
        out[f'kappa1_S{nsub}_vs_formula'] = float(np.abs(k1S - k1sol).max() / sc1)
        out[f'kappa2_S{nsub}_vs_formula'] = float(np.abs(k2S - k2sol).max() / sc2)
        if nsub == 2:
            k1S2, k2S2, tq2 = k1S, k2S, tq
        out[f'perimeter_S{nsub}'] = float(s[-1])
    # perimeter (spectral)
    rt, zt, rtt, ztt = fc.deriv_grid()
    out['perimeter_own'] = float(np.sum(np.hypot(rt, zt)) * 2 * np.pi / Nt)
    out['perimeter_solver_2L'] = 2 * g.L

    # ---- (2) Bernoulli residual pointwise (independent H, interpolated gamma)
    tD = np.linspace(0, np.pi, 6001)
    vD = fc.eval(tD)
    k1D, k2D = curvatures(vD[2], vD[3], vD[4], vD[5], vD[0])
    HD = k1D + k2D
    gD = gamma_panel(st, tD)
    Hmax = np.abs(HD).max()
    resD = HD - 0.5 * We * gD ** 2 - lam
    out['H_max'] = float(Hmax)
    out['bern_dense_panelgamma'] = float(np.abs(resD).max() / Hmax)
    Kfit = 256
    cg, fitres = gamma_cosfit(st, Kfit)
    out['gamma_cosfit_K'] = Kfit
    out['gamma_cosfit_noderes'] = float(fitres / np.abs(st.gamma).max())
    gD2 = cos_eval(cg, tD)
    out['gamma_interp_diff'] = float(np.abs(gD2 - gD).max() / np.abs(st.gamma).max())
    resD2 = HD - 0.5 * We * gD2 ** 2 - lam
    out['bern_dense_cosgamma'] = float(np.abs(resD2).max() / Hmax)
    # at the nodes: independent H with the solver's gamma
    resN = (k1F_n + k2F_n) - 0.5 * We * st.gamma ** 2 - lam
    out['bern_nodes_indepH'] = float(np.abs(resN).max() / Hmax)
    out['bern_nodes_solverH'] = float(np.abs(st.residual_nodal(We, lam)).max() / Hmax)
    # on the spline points (method S curvature) with panel gamma
    gS = gamma_panel(st, tq2)
    resS = k1S2 + k2S2 - 0.5 * We * gS ** 2 - lam
    out['bern_splineH'] = float(np.abs(resS).max() / Hmax)
    out['bern_splineH_interior'] = float(np.abs(resS[20:-20]).max() / Hmax)
    out['u_max'] = float(np.abs(st.gamma).max())
    out['dyn_over_H'] = float((0.5 * We * gD ** 2).max() / Hmax)

    # ---- (3) volume
    # spectral trapezoid, two formulas: -pi oint r^2 dz  and  2 pi oint r z dr
    V_a = -np.pi * np.sum(fc.r ** 2 * zt) * 2 * np.pi / Nt
    V_b = 2 * np.pi * np.sum(fc.r * fc.z * rt) * 2 * np.pi / Nt
    out['vol_rel_err_r2dz'] = float(V_a / VOL - 1)
    out['vol_rel_err_rzdr'] = float(V_b / VOL - 1)

    def poly_vol(q):
        closed = np.vstack([q, q[-2:0:-1] * [1, -1], q[:1]])
        x, y = closed[:, 0], closed[:, 1]
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]
        return 2 * np.pi * abs(np.sum((x[:-1] + x[1:]) * cr) / 6)
    V1p, V2p = poly_vol(pts[::2]), poly_vol(pts)     # h and h/2 (nS-1 even)
    out['vol_rel_err_polygon_richardson'] = float(((4 * V2p - V1p) / 3) / VOL - 1)
    out['vol_rel_err_polygon_raw'] = float(V2p / VOL - 1)

    # ---- (4) force identity  lam a + We/(4R) int_upper gamma^2 r dr = 1
    aa, RR = (r_o - r_h) / 2, (r_o + r_h) / 2
    out['a_tube'] = aa
    out['R_centre'] = RR
    rt_n = vn[2]
    Ig = np.sum(P.wf * st.gamma ** 2 * vn[0] * rt_n)
    out['force_identity_nodes_minus1'] = float(lam * aa + We / (4 * RR) * Ig - 1)
    # independent Gauss-Legendre rule, interpolated gamma (cos fit)
    xg, wg = np.polynomial.legendre.leggauss(3000)
    tg = (xg + 1) * np.pi / 2
    wg = wg * np.pi / 2
    vg = fc.eval(tg)
    k1g, k2g = curvatures(vg[2], vg[3], vg[4], vg[5], vg[0])
    gg = cos_eval(cg, tg)
    Ig2 = np.sum(wg * gg ** 2 * vg[0] * vg[2])
    out['force_identity_GL_minus1'] = float(lam * aa + We / (4 * RR) * Ig2 - 1)
    out['geom_identity_intHrdr_over_2R_minus1'] = float(np.sum(wg * (k1g + k2g) * vg[0] * vg[2]) / (2 * RR) - 1)
    out['force_dyn_part'] = float(We / (4 * RR) * Ig2)
    out['force_capillary_part_lam_a'] = float(lam * aa)

    # ---- shape analysis: deviation from a circle
    zz = fc.z
    xc = fc.dr - aa                     # r - R, from the relative coordinate
    phi = np.arctan2(zz, xc)
    rad = np.hypot(xc, zz)
    o = np.argsort(phi)
    phiu = 2 * np.pi * np.arange(4096) / 4096
    radu = np.interp(phiu, phi[o], rad[o], period=2 * np.pi)
    Rad = np.fft.rfft(radu) / len(phiu)
    rad0 = Rad[0].real
    eps_n = [float(2 * Rad[k].real / rad0) for k in range(1, 7)]   # cos(n phi), phi=0 outer rim
    out['rho0'] = float(rad0)
    out['eps_n_rel'] = eps_n
    out['max_dev_from_circle_aR'] = float(np.abs(rad - aa).max() / aa)
    out['one_minus_zmax_over_a'] = float(1 - zmax / aa)
    out['eps'] = float(aa / RR)
    # m = 2 balance estimate: eps_2 ~ a^2/(6R^2) + a D_2/3, D_2 = cos(2phi) coeff of (We/2) gamma^2
    tq_all = fc.t[:Nt // 2 + 1]
    gq = cos_eval(cg, tq_all[::8])
    phq = phi[:Nt // 2 + 1][::8]
    phfull = np.concatenate([phq, -phq[::-1]])
    gfull = np.concatenate([gq, gq[::-1]])
    o2 = np.argsort(phfull)
    g2u = np.interp(phiu, phfull[o2], gfull[o2] ** 2, period=2 * np.pi)
    G2 = np.fft.rfft(g2u) / len(phiu)
    D2 = 0.5 * We * 2 * G2[2].real
    out['D2_dyn_m2'] = float(D2)
    out['eps2_geom_pred'] = float(aa ** 2 / (6 * RR ** 2))
    out['eps2_dyn_pred'] = float(aa * D2 / 3)
    out['eps2_pred_total'] = float(aa ** 2 / (6 * RR ** 2) + aa * D2 / 3)
    gm = np.fft.rfft(np.interp(phiu, phfull[o2], gfull[o2], period=2 * np.pi)) / len(phiu)
    out['gamma_phi_modes'] = [float(gm[0].real)] + [float(2 * gm[k].real) for k in (1, 2, 3)]
    # thin-ring prediction, no solver input except a, R, We:  W = 1 - (Lam-1/2)/(4(Lam-2))
    Lam = np.log(8 * RR / aa)
    W = 1 - (Lam - 0.5) / (4 * (Lam - 2))
    out['thin_Lambda'] = float(Lam)
    out['thin_2W'] = float(2 * W)
    out['eps2_thin_pred'] = float(aa ** 2 / (6 * RR ** 2) + aa * We * W ** 2 / 3)

    # ---- bonus: interior stream function must equal Q = 0 (own Lamb kernel, smooth quadrature)
    xs, ws = np.polynomial.legendre.leggauss(32)
    u = np.broadcast_to(xs, (P.np, 32))
    tsub = ((P.a + P.b)[:, None] / 2 + P.h[:, None] / 2 * u).ravel()
    wsub = (P.h[:, None] / 2 * ws[None, :]).ravel()
    vs = fc.eval(tsub)
    gs = gamma_panel(st, tsub)
    dsw = wsub * np.hypot(vs[2], vs[3]) * gs
    # targets on rays from the tube centre at fractions f of the distance to the surface
    for fs, key in (((0.0, 0.3, 0.6), 'deep'), ((0.8, 0.9), 'near')):
        phs = np.linspace(0, np.pi, 9)
        trr, tzz = [], []
        for ph in phs:
            k = np.argmin(np.abs(phi - ph))
            for f in fs:
                trr.append(RR + f * rad[k] * np.cos(ph))
                tzz.append(f * rad[k] * np.sin(ph))
        tr, tz = np.array(trr), np.array(tzz)
        psi = tr ** 2 / 2 + (psi_ring_lamb(tr[:, None], tz[:, None], vs[0][None], vs[1][None])
                             + psi_ring_lamb(tr[:, None], tz[:, None], vs[0][None], -vs[1][None])) @ dsw
        out[f'psi_interior_{key}_max_abs'] = float(np.abs(psi).max())
        out[f'psi_interior_{key}_rel_to_R2/2'] = float(np.abs(psi).max() / (RR ** 2 / 2))
    out['time_s'] = time.time() - t0
    return out


if __name__ == '__main__':
    res = []
    for arg in sys.argv[1:]:
        tag, i = arg.split(':')
        o = check_ring(tag, int(i))
        res.append(o)
        print(json.dumps(o, indent=None, default=float), flush=True)
    os.makedirs(os.path.join(REPO, 'results', 'verify'), exist_ok=True)
    outfn = os.path.join(REPO, 'results', 'verify', 'bernoulli_' + '_'.join(a.replace(':', '') for a in sys.argv[1:]) + '.json')
    json.dump(res, open(outfn, 'w'), indent=1, default=float)
