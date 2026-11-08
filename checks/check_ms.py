"""Thin-ring check of the ring branch against Meyer & Seis, Theorem 1.3.

Meyer & Seis, "Steady ring-shaped vortex sheets" (arXiv:2409.08220), construct
thin hollow vortex rings with surface tension by the implicit function
theorem.  Their precise Theorem 1.3 (numbers as in arXiv:2409.08220v2) is stated in
non-dimensional variables.  For a vacuum interior (rho = 0) it reads, as the
aspect ratio eps -> 0,

    W     = (1/4pi) (log(8/eps) - 1/2) + (pi/2) beta + o(1)        (speed)
    gamma = (3/8pi) log(8/eps) - 15/(16 pi) - (pi/4) beta + o(1)   (flux constant)
    nu    = -1/(4 pi^2) + beta + o(1)                                (Bernoulli)

where (their substitutions just before (1.33), with b = R b_ext)

    eps   = eps_bar / R      eps_bar: cross-section radius (|Omega| = pi eps_bar^2)
                             R: radial coordinate of the cross-section's centroid
    beta  = 2 R^3 eps sigma / (rho b^2) = 2 eps_bar sigma / (rho b_ext^2)
    W     = R^2 W_bar / b = R W_bar / b_ext
    gamma = gamma_bar / b

In the units of this project (equivalent radius 1, stream speed V = W_bar = 1,
We = rho V^2 / sigma, circulation Gamma):

    b_ext = |Gamma|,  beta = 2 eps_bar / (We Gamma^2),  W = R / |Gamma|,
    gamma = -Q / (R |Gamma|)   (Q: our flux constant, flux 2 pi Q along the
                                stream; MS's relative stream function is the
                                mirror image of ours, hence the sign),
    nu    = beta * eps_bar * lambda   (multiply H - (We/2)|u|^2 = lambda by
                                       beta * eps_bar and compare with (1.41)).

The zero-flux branch has Q = 0, i.e. gamma = 0 exactly: in the MS family it
is the curve beta(eps) on which their gamma vanishes, to leading order
beta = (3/(2 pi^2)) (log(8/eps) - 5/2).  This is admissible in their
theorem: 1/beta -> 0, eps*beta -> 0 and eps |beta'| <~ beta.

The script measures area and centroid of every ring cross-section (of the
high-resolution branch results/hires/rings_merged.json when present,
else of the stored rings in results/rings/), evaluates the three differences (computed - MS)
along the zero-flux branch and along the thin flux scans, and writes
results/checks/ms.json. The differences fall like eps^2 log^2(1/eps); they are
fitted by A eps^2 log^2(1/eps) + B eps^2 log(1/eps) + C eps^2 with relative
weights, on the rings with eps < 0.05 and r_h <= 320 (beyond, the remainders of W
and gamma reach the 1e-8 accuracy of the computed We).

Remark: the informal Theorem 1.1, formula (1.19), writes the surface-tension
term of the speed as (pi R / (b_ext rho)) eps_bar sigma.  That is not
dimensionally a speed (apparently a misprint); the precise Theorem 1.3 gives
(pi / (R b_ext rho)) eps_bar sigma, which is what is tested here.
"""
import json
import os
import sys

import numpy as np

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "page"))

from bubbles.ring import RingGeometry, RingMap, RingPanels  # noqa: E402
from data import load_json, path  # noqa: E402

ZERO_FLUX_FILES = ["rings_neck3", "rings_neck2", "rings_neck", "rings_down2", "rings_down",
                   "rings_up_hp", "rings_thin", "rings_thin2", "rings_thin3", "rings_thin7"]
# the merged high-resolution branch (scripts/merge_hires_rings.py), when present: it replaces the
# stored files, from which it already takes the ranges no new chain reached
HIRES = "results/hires/rings_merged.json"
# the remainder fits use the thin rings up to this r_h (beyond, the remainders of W and gamma
# reach the 1e-8 accuracy of the computed We)
FIT_RH_MAX = 320.0


def zero_flux_files():
    if os.path.exists(path(HIRES)):
        return [HIRES]
    return [f"results/rings/{f}.json" for f in ZERO_FLUX_FILES]


SHEET_FILES = ["sheet_rh0.069", "sheet_rh0.120", "sheet_rh0.241", "sheet_rh0.422",
               "sheet_rh0.977", "sheet_rh1.964", "sheet_rh5.225", "sheet_rh14.9"]


def cross_section(p):
    """Area and centroid of the meridian cross-section (both halves)."""
    g = RingGeometry(np.array(p['a']), p['rho_h'], RingMap(np.array(p['map_b'])),
                     RingPanels(np.array(p['breaks'])))
    w = g.P.wf
    rp = g.c * np.cos(g.theta)
    Au = np.sum(w * g.z * rp)              # area of the upper half, int z dr
    rc = np.sum(w * g.r * g.z * rp) / Au
    A = 2 * Au
    return dict(area=float(A), R=float(rc), eps_bar=float(np.sqrt(A / np.pi)),
                pappus=float(2 * np.pi * rc * A - 4 * np.pi / 3), z_end=float(g.z_end1))


def ms_quantities(p, Q=0.0):
    cs = cross_section(p)
    G = abs(p['circulation'])
    We, lam = p['We'], p['lam']
    eb, R = cs['eps_bar'], cs['R']
    eps = eb / R
    beta = 2 * eb / (We * G ** 2)
    W = R / G
    gam = -Q / (R * G)
    nu = beta * eb * lam
    L8 = np.log(8 / eps)
    W_ms = (L8 - 0.5) / (4 * np.pi) + 0.5 * np.pi * beta
    g_ms = 3 * L8 / (8 * np.pi) - 15 / (16 * np.pi) - 0.25 * np.pi * beta
    n_ms = -1 / (4 * np.pi ** 2) + beta
    return dict(r_h=p['r_h'], We=We, lam=lam, Gamma=p['circulation'], Q=Q,
                R=R, eps_bar=eb, eps=eps, beta=beta, We_MS=2 / beta,
                W=W, W_MS=W_ms, dW=W - W_ms,
                gamma=gam, gamma_MS=g_ms, dgamma=gam - g_ms,
                nu=nu, nu_MS=n_ms, dnu=nu - n_ms,
                pappus_err=cs['pappus'], z_end=cs['z_end'],
                beta_zero_flux_leading=3 / (2 * np.pi ** 2) * (L8 - 2.5))


def zero_flux_branch():
    seen, out = set(), []
    for f in zero_flux_files():
        for p in load_json(f):
            key = round(np.log(p['r_h']), 9)
            if key in seen:
                continue
            seen.add(key)
            out.append(ms_quantities(p))
    out.sort(key=lambda d: d['r_h'])
    return out


def sheet_scans():
    scans = {}
    for f in SHEET_FILES:
        pts = load_json(f"results/rings/{f}.json")
        scans[f] = sorted((ms_quantities(p, Q=p['Q']) for p in pts if 'a' in p),
                          key=lambda d: d['Q'])
    return scans


def main():
    zf = zero_flux_branch()
    print(f"{'r_h':>10} {'eps':>10} {'beta':>9} {'W':>10} {'W-W_MS':>10} {'g_MS':>10} {'nu-nu_MS':>10} pappus")
    for d in zf[::4] + [zf[-1]]:
        print(f"{d['r_h']:10.4g} {d['eps']:10.4g} {d['beta']:9.4f} {d['W']:10.6f} {d['dW']:+10.3e} "
              f"{d['gamma_MS']:+10.3e} {d['dnu']:+10.3e} {d['pappus_err']:.1e}")
    sc = sheet_scans()
    for f, pts in sc.items():
        m = [abs(p['dW']) for p in pts]
        print(f, len(pts), f"eps={pts[0]['eps']:.3g}", f"max|dW|={max(m):.3e}",
              f"max|dgamma|={max(abs(p['dgamma']) for p in pts):.3e}",
              f"max|dnu|={max(abs(p['dnu']) for p in pts):.3e}")
    # fits of the remainders on the thin rings: d ~ eps^2 (A log^2(1/eps) + B log(1/eps) + C), relative
    # weights, up to r_h = FIT_RH_MAX
    thin = [d for d in zf if d['eps'] < 0.05 and d['r_h'] <= FIT_RH_MAX]
    fits = {}
    for k in ("dW", "dgamma", "dnu"):
        e = np.array([d['eps'] for d in thin])
        y = np.array([d[k] for d in thin])
        L = np.log(1 / e)
        V = np.stack([e ** 2 * L ** 2, e ** 2 * L, e ** 2], axis=1)
        c, *_ = np.linalg.lstsq(V / np.abs(y)[:, None], np.sign(y), rcond=None)
        fits[k] = dict(eps2log2=float(c[0]), eps2log=float(c[1]), eps2=float(c[2]),
                       max_rel_resid=float((np.abs(V @ c - y) / np.abs(y)).max()), n=len(thin),
                       r_h_max=FIT_RH_MAX,
                       ratio_eps2log2=[float((y / (e ** 2 * L ** 2)).min()), float((y / (e ** 2 * L ** 2)).max())])
    print(f"remainder fits on eps < 0.05, r_h <= {FIT_RH_MAX:g}:", fits)
    out = dict(zero_flux=zf, sheet=sc, remainder_fits=fits,
               formulas=dict(W="(log(8/eps)-1/2)/(4pi) + pi beta/2",
                             gamma="3 log(8/eps)/(8pi) - 15/(16pi) - pi beta/4",
                             nu="-1/(4pi^2) + beta"))
    with open(os.path.join(ROOT, "results", "checks", "ms.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
