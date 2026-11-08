"""Referee check: fully independent flow solve + geometry for stored rings.

Only the JSON coefficients (a, map_b, rho_h, We, lam) are used.  The curve is
integrated by composite Gauss-Legendre on a uniform t-grid of the FULL closed
curve t in [0, 2 pi); the sheet equation  int gamma Psi_ring ds = -r^2/2  is
solved by a Nystrom method with Kress log-splitting (spectral for smooth
periodic curves), own Lamb kernel via scipy ellipk/ellipkm1/ellipe.
The solver (bubbles.ring) is used ONLY afterwards for comparison.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import ellipe, ellipk, ellipkm1

VOL = 4 * np.pi / 3
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FILES = dict(up=os.path.join(REPO, 'results/rings/rings_up.json'),
             thin=os.path.join(REPO, 'results/rings/rings_thin.json'),
             thin2=os.path.join(REPO, 'results/rings/rings_thin2.json'),
             thin7=os.path.join(REPO, 'results/rings/rings_thin7.json'),
             hp=os.path.join(REPO, 'results/rings/rings_up_hp.json'))


def th_sp(a, b, t):
    k = np.arange(1, len(a) + 1)
    arg = np.multiply.outer(t, k)
    th = np.pi / 2 - t + np.sin(arg) @ a
    tht = -1.0 + np.cos(arg) @ (k * a)
    kb = np.arange(1, len(b) + 1)
    argb = np.multiply.outer(t, kb)
    sp = (1 + np.cos(argb) @ b) / np.pi
    spt = -(np.sin(argb) @ (kb * b)) / np.pi
    return th, tht, sp, spt


def curve(a, b, rho_h, N, q=10):
    """Unit-L curve on t_j = 2 pi j / N (full period) by composite GL."""
    t = 2 * np.pi * np.arange(N) / N
    h = 2 * np.pi / N
    xg, wg = np.polynomial.legendre.leggauss(q)
    tq = (t[:, None] + h / 2 * (xg[None, :] + 1)).ravel()
    th, _, sp, _ = th_sp(a, b, tq)
    dr = (sp * np.cos(th)).reshape(N, q) @ wg * h / 2
    dz = (sp * np.sin(th)).reshape(N, q) @ wg * h / 2
    r1 = rho_h + np.concatenate([[0], np.cumsum(dr)])
    z1 = np.concatenate([[0], np.cumsum(dz)])
    closure = (r1[-1] - r1[0], z1[-1] - z1[0])
    r1, z1 = r1[:-1], z1[:-1]
    th, tht, sp, spt = th_sp(a, b, t)
    rt1, zt1 = sp * np.cos(th), sp * np.sin(th)
    # volume of revolution, clockwise curve: V = -pi oint r^2 z' dt (trapezoid = spectral)
    V1 = -np.pi * np.sum(r1 ** 2 * zt1) * h
    L = (VOL / V1) ** (1 / 3)
    return dict(t=t, r=L * r1, z=L * z1, rt=L * rt1, zt=L * zt1, th=th, tht=tht, sp=sp,
                L=L, closure=(L * closure[0], L * closure[1]), N=N)


def fourier_deriv(f, p=1):
    N = len(f)
    n = np.fft.fftfreq(N, 1.0 / N)
    F = np.fft.fft(f)
    if p % 2 == 1:
        F[N // 2] = 0
    return np.real(np.fft.ifft(F * (1j * n) ** p))


def trig_eval(f, t):
    """Trigonometric interpolant of grid values f (uniform full period) at t."""
    N = len(f)
    F = np.fft.fft(f) / N
    n = np.fft.fftfreq(N, 1.0 / N)
    F = F.copy()
    F[N // 2] *= 0.5
    n = np.concatenate([n, [N // 2]])
    F = np.concatenate([F, [F[N // 2]]])
    n[N // 2] = -N // 2
    return np.real(np.exp(1j * np.outer(t, n)) @ F)


def psi_ring(r, z, rho, zeta):
    A = (r + rho) ** 2 + (z - zeta) ** 2
    B = (r - rho) ** 2 + (z - zeta) ** 2
    m1 = B / A
    k = np.sqrt(1 - m1)
    Km = ellipkm1(m1)              # K(m), m = 1 - m1, accurate near m = 1
    Em = ellipe(1 - m1)
    pre = np.sqrt(r * rho) / (2 * np.pi)
    psi = pre * ((2 / k - k) * Km - (2 / k) * Em)
    alog = -pre / np.pi * ((2 / k) * ellipe(m1) - k * ellipk(m1))   # coefficient of ln m1
    return psi, alog, A, B


def sheet_solve(c):
    N = c['N']
    n = N // 2
    t, r, z = c['t'], c['r'], c['z']
    sp = np.hypot(c['rt'], c['zt'])          # |x'(t)|
    with np.errstate(divide='ignore', invalid='ignore'):
        psi, alog, A, B = psi_ring(r[:, None], z[:, None], r[None, :], z[None, :])
    d = t[:, None] - t[None, :]
    with np.errstate(divide='ignore'):
        lg = np.log(4 * np.sin(d / 2) ** 2)
    K1 = alog * sp[None, :]
    K2 = psi * sp[None, :] - K1 * lg
    ii = np.arange(N)
    K1[ii, ii] = -(r / (4 * np.pi)) * sp
    K2[ii, ii] = sp * r / (2 * np.pi) * (np.log(8 * r / sp) - 2)
    # Kress weights R(t_i - t_j)
    m = np.arange(1, n)
    dl = 2 * np.pi * np.arange(N) / N
    Rv = -(2 * np.pi / n) * (np.cos(np.outer(dl, m)) @ (1.0 / m)) - (np.pi / n ** 2) * np.cos(n * dl)
    Rm = Rv[(ii[:, None] - ii[None, :]) % N]
    Mat = Rm * K1 + (np.pi / n) * K2
    lu = lu_factor(Mat)
    gam = lu_solve(lu, -0.5 * r ** 2)
    return gam, sp, Mat


def main(tag, idx, Ns=(1024, 2048)):
    p = json.load(open(FILES[tag]))[idx]
    a, b = np.array(p['a']), np.array(p['map_b'])
    We, lam = p['We'], p['lam']
    out = dict(ring=f'{tag}[{idx}]', r_h=p['r_h'], We=We, lam=lam)
    sols = {}
    for N in Ns:
        t0 = time.time()
        c = curve(a, b, p['rho_h'], N)
        gam, sp, Mat = sheet_solve(c)
        # curvature: (i) analytic parametrisation formula, (ii) Fourier diff of the points
        H_form = -c['tht'] / (c['L'] * c['sp']) - np.sin(c['th']) / c['r']
        rt, zt = fourier_deriv(c['r']), fourier_deriv(c['z'])
        rtt, ztt = fourier_deriv(c['r'], 2), fourier_deriv(c['z'], 2)
        s3 = np.hypot(rt, zt) ** 3
        k1 = -(rt * ztt - zt * rtt) / s3
        k2 = -(zt / np.hypot(rt, zt)) / c['r']
        H_four = k1 + k2
        res = H_four - 0.5 * We * gam ** 2 - lam
        res_f = H_form - 0.5 * We * gam ** 2 - lam
        Hmax = np.abs(H_form).max()
        circ = np.sum(gam * sp) * 2 * np.pi / N
        sols[N] = dict(c=c, gam=gam, sp=sp)
        out[N] = dict(L=c['L'], closure=c['closure'], cond=None,
                      H_four_vs_form=float(np.abs(H_four - H_form).max() / Hmax),
                      bern_res_fourierH=float(np.abs(res).max() / Hmax),
                      bern_res_formulaH=float(np.abs(res_f).max() / Hmax),
                      circulation=float(circ),
                      gamma_evenness=float(np.abs(gam - np.roll(gam[::-1], 1)).max() / np.abs(gam).max()),
                      gamma_spec_tail=float(np.abs(np.fft.rfft(gam))[N // 4:].max() / np.abs(np.fft.rfft(gam)).max()),
                      time=time.time() - t0)
        print(N, out[N], flush=True)
    N1, N2 = Ns[0], Ns[-1]
    g1 = sols[N1]['gam']
    g2 = sols[N2]['gam'][::N2 // N1]
    out['gamma_N_conv'] = float(np.abs(g1 - g2).max() / np.abs(g2).max())
    c = sols[N2]['c']
    gam = sols[N2]['gam']
    N = N2
    # Bernoulli between grid points (trig interpolation of my gamma, analytic H) on [0, pi]
    tD = np.linspace(0, np.pi, 4001)
    gD = trig_eval(gam, tD)
    th, tht, spD, _ = th_sp(a, b, tD)
    rD = trig_eval(c['r'] - c['r'].mean(), tD) + c['r'].mean()
    HD = -tht / (c['L'] * spD) - np.sin(th) / rD
    out['bern_res_dense'] = float(np.abs(HD - 0.5 * We * gD ** 2 - lam).max() / np.abs(HD).max())
    out['dyn_over_H_max'] = float((0.5 * We * gD ** 2).max() / np.abs(HD).max())
    # ----- comparison with the solver (only now)
    sys.path.insert(0, REPO)
    from bubbles.ring import RingMap, RingPanels, RingState
    st = RingState(a, p['rho_h'], RingMap(b), RingPanels(np.array(p['breaks'])))
    gS = trig_eval(gam, st.P.tf)
    out['gamma_vs_solver'] = float(np.abs(np.abs(gS) - np.abs(st.gamma)).max() / np.abs(st.gamma).max())
    out['gamma_sign_same'] = bool(np.all(np.sign(gS) == np.sign(st.gamma)) or np.all(np.sign(gS) == -np.sign(st.gamma)))
    out['circ_solver'] = float(st.circulation)
    out['circ_json'] = p.get('circulation')
    out['L_vs_solver'] = float(c['L'] / st.g.L - 1)
    # bernoulli at solver nodes with MY gamma and solver H
    out['bern_solverH_mygamma'] = float(np.abs(st.g.H - 0.5 * We * gS ** 2 - lam).max() / np.abs(st.g.H).max())
    # ----- geometry / shape from my curve
    r, z = c['r'], c['z']
    rh, ro = r[0], r[N // 2]
    out['r_h_mine_minus_json'] = float(rh - p['r_h'])
    out['r_o_mine_minus_json'] = float(ro - p['r_o'])
    out['z_pi'] = float(z[N // 2])
    tz = np.linspace(0, np.pi, 200001)
    zz = trig_eval(z, tz[::50])
    j = np.argmax(zz)
    tt = np.linspace(tz[::50][max(j - 1, 0)], tz[::50][min(j + 1, len(zz) - 1)], 2001)
    zmax = trig_eval(z, tt).max()
    aa, RR = (ro - rh) / 2, (ro + rh) / 2
    out['a'] = float(aa)
    out['R'] = float(RR)
    out['z_max_mine'] = float(zmax)
    out['z_max_json'] = p['z_max']
    out['one_minus_zmax_over_a'] = float(1 - zmax / aa)
    # polar description about (R, 0); cos(n phi) content of the radius
    phi = np.unwrap(np.arctan2(z, r - RR))
    rad = np.hypot(r - RR, z)
    out['rad_min_over_a'] = float(rad.min() / aa)
    out['rad_max_over_a'] = float(rad.max() / aa)
    # force identity with MY gamma: lam a + We/(4R) int_upper gamma^2 r dr = 1 (trapezoid, spectral on full period)
    # int_upper f r r_t dt = (1/2) oint f r r_t dt with sign: upper half t in [0,pi]; integrand odd-symmetric -> use half
    xg, wg = np.polynomial.legendre.leggauss(1500)
    tg, wg = (xg + 1) * np.pi / 2, wg * np.pi / 2
    thg, _, spg, _ = th_sp(a, b, tg)
    rg = trig_eval(c['r'] - c['r'].mean(), tg) + c['r'].mean()
    rtg = c['L'] * spg * np.cos(thg)
    Ig = np.sum(wg * trig_eval(gam, tg) ** 2 * rg * rtg)
    thg2, thtg2, _, _ = th_sp(a, b, tg)
    Hg = -thtg2 / (c['L'] * spg) - np.sin(thg2) / rg
    out['geom_identity_intHrdr_over_2R_minus1'] = float(np.sum(wg * Hg * rg * rtg) / (2 * RR) - 1)
    out['force_identity_minus1'] = float(lam * aa + We / (4 * RR) * Ig - 1)
    out['force_dyn_part'] = float(We / (4 * RR) * Ig)
    # thin-ring numbers
    Lam = np.log(8 * RR / aa)
    W = 1 - (Lam - 0.5) / (4 * (Lam - 2))
    out['Lambda'] = float(Lam)
    out['2W'] = float(2 * W)
    out['eps2_geom'] = float(aa ** 2 / (6 * RR ** 2))
    out['eps2_thin_formula'] = float(aa ** 2 / (6 * RR ** 2) + aa * We * W ** 2 / 3)
    out['onemz_thin_formula'] = 2 * out['eps2_thin_formula']
    print(json.dumps(out, default=float), flush=True)
    return out


if __name__ == '__main__':
    res = []
    Ns = tuple(int(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (1024, 2048)
    for arg in sys.argv[1].split(','):
        tag, i = arg.split(':')
        res.append(main(tag, int(i), Ns))
    os.makedirs(os.path.join(REPO, 'results', 'verify'), exist_ok=True)
    fn = os.path.join(REPO, 'results', 'verify', 'kress_' + sys.argv[1].replace(':', '').replace(',', '_') + '.json')
    json.dump(res, open(fn, 'w'), default=lambda o: o if not isinstance(o, tuple) else list(o), indent=1)
