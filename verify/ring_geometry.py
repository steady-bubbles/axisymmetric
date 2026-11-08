"""Circularity measurement of the ring cross-sections (geometry only).
Usage: python verify/ring_geometry.py  -> prints table, writes results/verify/geom.json"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)
from bubbles.ring import RingMap, RingPanels, RingGeometry, ring_theta
from bubbles.quad import cheb_eval

ROOT = os.path.join(REPO, 'results', 'rings') + os.sep


def load_geom(p, split=1):
    a = np.array(p['a'])
    br = np.array(p['breaks'])
    if split > 1:   # refine panels (accuracy check)
        br = np.unique(np.concatenate([np.linspace(br[i], br[i + 1], split + 1) for i in range(len(br) - 1)]))
    g = RingGeometry(a, p['rho_h'], RingMap(np.array(p['map_b'])), RingPanels(br))
    return g


def eval_x(g, t):
    P = g.P
    t = np.asarray(t, float)
    idx = np.clip(np.searchsorted(P.b, t, side='right'), 0, P.np - 1)
    u = np.clip((2 * t - P.a[idx] - P.b[idx]) / P.h[idx], -1, 1)
    x = cheb_eval(g.A[:, idx, :], u[None, :, None])[..., 0]
    return x[0], x[1]


def measure(p, split=1, nmax=8):
    g = load_geom(p, split)
    w = g.P.wf
    th, c, r, z = g.theta, g.c, g.r, g.z
    rp, zp = c * np.cos(th), c * np.sin(th)          # d/dt
    Au = np.sum(w * z * rp)                           # area of upper half
    rc = np.sum(w * r * z * rp) / Au                  # area centroid (z_c = 0)
    X = r - rc
    rho = np.hypot(X, z)
    phi = np.arctan2(z, X)
    dphi = (X * zp - z * rp) / rho ** 2               # dphi/dt (should be < 0: star-shaped)
    star = bool(np.all(dphi < 0))
    a0 = -np.sum(w * rho * dphi) / np.pi
    cn = np.array([-2 / np.pi * np.sum(w * rho * np.cos(n * phi) * dphi) for n in range(1, nmax + 1)]) / a0
    # dense sampling for max |rho - a0|
    td = np.linspace(0, np.pi, 40001)
    rd, zd = eval_x(g, td)
    rhod = np.hypot(rd - rc, zd)
    maxdev = float(np.abs(rhod - a0).max() / a0)
    # z_max: Newton on theta(t) = 0
    a = g.a
    tq = td[np.argmax(zd)]
    for _ in range(50):
        f, fp = ring_theta(a, np.array([tq]))
        dt = f[0] / fp[0]
        tq -= dt
        if abs(dt) < 1e-15:
            break
    _, zmax = eval_x(g, np.array([tq]))
    zmax = float(zmax[0])
    rho_ex = None
    return dict(r_h=float(g.r_h), r_o=float(g.r_o), r_c=float(rc), area=float(2 * Au), a0=float(a0),
                c=[float(x) for x in cn], maxdev=maxdev, zmax=zmax,
                aspect=float((g.r_o - g.r_h) / (2 * zmax)), star=star, z_end=float(g.z_end1),
                a_eq=float(np.sqrt(2 * Au / np.pi)), R_mid=float((g.r_o + g.r_h) / 2),
                rho_on_axis=[float(np.max(rhod)), float(np.min(rhod))])


if __name__ == '__main__':
    out = []
    for fn, tag in (('rings_up.json', 'up'), ('rings_thin.json', 'thin')):
        for i, p in enumerate(json.load(open(ROOT + fn))):
            m = measure(p)
            m2 = measure(p, split=2)
            m.update(tag=f'{tag}:{i}', We=p['We'], lam=p['lam'], circ=p['circulation'],
                     err_c2=abs(m['c'][1] - m2['c'][1]), err_aspect=abs(m['aspect'] - m2['aspect']),
                     err_a0=abs(m['a0'] - m2['a0']))
            out.append(m)
            print(f"{m['tag']:8s} r_h={m['r_h']:9.5g} We={m['We']:.5g} a0={m['a0']:.6g} r_c={m['r_c']:.6g} "
                  f"c1={m['c'][0]:+.3e} c2={m['c'][1]:+.3e} c3={m['c'][2]:+.3e} c4={m['c'][3]:+.2e} "
                  f"maxdev={m['maxdev']:.3e} asp-1={m['aspect']-1:.4e} star={m['star']} "
                  f"err(c2)={m['err_c2']:.1e} err(asp)={m['err_aspect']:.1e}", flush=True)
    os.makedirs(os.path.dirname(os.path.join(REPO, 'results', 'verify', 'geom.json')), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, 'results', 'verify', 'geom.json'), 'w'), indent=1)
