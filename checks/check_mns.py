"""Small-We check of the bubble branch against Meyer, Niebel & Seis.

MNS (arXiv:2503.05503, Thm 1.1 with gamma = 0, R = 1, eps*delta_out = We)
predict for the unique symmetric near-spherical bubble

    d(theta) = -(3/32) We (3 cos^2 theta - 1) + o(We)      (radial distance
                                                            sphere -> surface)
    chi      = 1 + (9/32) We + O(We^2)                     (Remark 1.2 (3))

and hence lambda = H_pole = 2 - (3/4) We + O(We^2) (the pole curvature of
the shape above).  This script solves the bubble equations at fixed We for
We = 1e-4 ... 0.4 at two resolutions with the boundary-integral solver of
`bubbles/` and measures

  * E_shape(We) = max_theta |d(theta) + (3/32) We (3cos^2 theta - 1)| / We,
    which must tend to 0 (it is O(We));
  * the Taylor coefficients of chi and lambda in We, by least-squares fits
    of (chi - 1)/We and (lambda - 2)/We on We <= 0.1.

Output: results/checks/mns.json
"""
import json
import os
import sys
import time

import numpy as np

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bubbles.geometry import ParamMap, adaptive_panels  # noqa: E402
from bubbles.newton import newton_fixed_we  # noqa: E402


def radial_profile(st, n=2001):
    """rho(theta) - 1 on a uniform grid of polar angles theta in [0, pi/2]."""
    g, P = st.g, st.P
    Nh = P.Nh
    r, z = g.r[:Nh], g.z[:Nh]
    th = np.arctan2(r, z)          # polar angle from the +z axis
    rho = np.hypot(r, z)
    o = np.argsort(th)
    th, rho = th[o], rho[o]
    # the curve is star-shaped here: rho(theta) is single valued and smooth;
    # high-order interpolation through the panel nodes (cubic spline is
    # accurate to ~1e-12 at these node densities, far below the We^2 signal)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(np.r_[-th[::-1], th], np.r_[rho[::-1], rho])
    tq = np.linspace(0.0, np.pi / 2, n)
    return tq, cs(tq) - 1.0


def sweep(M, c_h, We_list, tol=1e-13):
    pmap = ParamMap()
    a = np.zeros(M)
    lam = 2.0
    out = []
    for We in We_list:
        panels = adaptive_panels(a, pmap, p=16, tol=1e-14, h_max=c_h / M)
        a, lam, st, hist = newton_fixed_we(a, lam, We, panels, pmap, tol=tol)
        panels = adaptive_panels(a, pmap, p=16, tol=1e-14, h_max=c_h / M)
        a, lam, st, hist = newton_fixed_we(a, lam, We, panels, pmap, tol=tol)
        d = st.diagnostics()
        tq, dev = radial_profile(st)
        eta_mns = -(3.0 / 32.0) * We * (3 * np.cos(tq) ** 2 - 1)
        # Legendre P2 coefficient of the deviation (exact projection on the sphere)
        w = np.sin(tq)
        P2 = 0.5 * (3 * np.cos(tq) ** 2 - 1)
        from scipy.integrate import simpson
        c2 = 5.0 * simpson(dev * P2 * w, x=tq)      # (2l+1)/2 * int_0^pi = (2l+1) int_0^{pi/2}
        out.append(dict(
            We=We, lam=float(lam), chi=float(d['r_eq'] / d['z_pole']),
            r_eq=d['r_eq'], z_pole=d['z_pole'], M=M, N=int(st.P.N),
            res=float(np.abs(st.residual(We, lam)).max()),
            res_hi=float(st.residual_hi(We, lam)),
            a_tail=float(np.abs(a[-3:]).max()),
            shape_err_over_We=float(np.abs(dev - eta_mns).max() / We),
            P2_coeff_over_We=float(c2 / We),
            newton_its=len(hist)))
        print(f"M={M} We={We:.3e} chi={out[-1]['chi']:.15f} lam={lam:.15f} "
              f"E_shape={out[-1]['shape_err_over_We']:.3e} c2/We={c2 / We:+.10f} "
              f"res={out[-1]['res']:.1e} its={len(hist)}", flush=True)
    return out


def fit_coeffs(We, y, deg):
    """Least-squares polynomial fit of y(We); returns coefficients c0..c_deg."""
    V = np.vander(We, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(V, y, rcond=None)
    return c, float(np.abs(V @ c - y).max())


def main():
    t0 = time.time()
    We_list = list(np.geomspace(1e-4, 0.1, 19)) + [0.15, 0.2, 0.3, 0.4]
    runs = {"M16": sweep(16, 2.0, We_list), "M28": sweep(28, 1.0, We_list)}
    res = {"runs": runs, "fits": {}}
    for key, pts in runs.items():
        We = np.array([p['We'] for p in pts])
        chi = np.array([p['chi'] for p in pts])
        lam = np.array([p['lam'] for p in pts])
        m = We <= 0.1 + 1e-12
        fits = {}
        for deg in (3, 4, 5):
            cc, rc = fit_coeffs(We[m], (chi[m] - 1) / We[m], deg)
            cl, rl = fit_coeffs(We[m], (lam[m] - 2) / We[m], deg)
            fits[f"deg{deg}"] = dict(chi=cc[:3].tolist(), chi_resid=rc,
                                     lam=cl[:3].tolist(), lam_resid=rl)
        res["fits"][key] = fits
    # agreement of the two resolutions
    a, b = runs["M16"], runs["M28"]
    res["resolution_spread"] = dict(
        chi=max(abs(p['chi'] - q['chi']) for p, q in zip(a, b)),
        lam=max(abs(p['lam'] - q['lam']) for p, q in zip(a, b)))
    res["prediction"] = dict(chi_slope=9 / 32, lam_slope=-3 / 4,
                             shape="-(3/32) We (3 cos^2 theta - 1)",
                             P2_coeff_over_We=-3 / 16)
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.join(ROOT, "results", "checks"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "checks", "mns.json"), "w") as f:
        json.dump(res, f, indent=1)
    for key in res["fits"]:
        for deg, v in res["fits"][key].items():
            print(key, deg, "chi:", v["chi"], "lam:", v["lam"])
    print("spread", res["resolution_spread"], f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
