"""The first zero-flux ring, glued from the near-touching bubble.

A catenoidal neck of radius r_h is glued into the last bubble of a branch file
(bubbles/ring_init.glue_neck). The cross-section is fitted with a curvature-adapted
parameter map (design_map with K = 64, fit_ring with M modes), and Newton's method
(bubbles.ring.ring_newton) solves at fixed r_h, starting from the bubble's We and
lambda. The result is written as an npz (Y, b, breaks), which scripts/run_rings.py
takes as its start; run_rings.py refines it (modes, map) before its first step.

The rings of the first computation all descend from the ring at r_h = 0.039236
(results/rings/rings_up.json:0, M = 128). It was made this way; the original driver
was not kept, and this script does the same:

    python scripts/first_ring.py results/branch.json 0.03923612667872822 48 first_ring.npz
    python scripts/run_rings.py first_ring.npz first_ring.json 1.07 0.0392 128

(about 4 and 7 min on 4 cores). The second command only refines the start ring,
since r_stop lies below it; the refined ring, first_ring.json[0], agrees with
rings_up.json:0 to 1e-12 in We and lambda.

Usage: python scripts/first_ring.py branch.json r_h M out.npz
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.ring import ring_newton, ring_panels  # noqa: E402
from bubbles.ring_init import bubble_sheet, curve_theta, design_map, fit_ring, glue_neck  # noqa: E402


def main(branch, r_h, M, out):
    p = json.load(open(branch))[-1]
    r_b, z_b, _ = bubble_sheet(p)
    print(f"bubble: z_pole={p['z_pole']:.4e} We={p['We']:.12f} lam={p['lam']:+.10f}", flush=True)
    pts, info = glue_neck(r_b, z_b, r_h)
    print(f"neck of radius r_h={r_h} glued in: {info}", flush=True)
    s, sm, th = curve_theta(pts)
    rm = design_map(sm, th, s[-1], K=64)
    a, rho = fit_ring(pts, rm, M)
    Y = np.concatenate([a, [p['lam'], rho, p['We']]])
    P = ring_panels(a, rm, M, c_h=6.0, tol=1e-11)
    t0 = time.time()
    Y, st, hist, _ = ring_newton(Y, rm, P, r_h, maxit=30)
    print(f"Newton: {len(hist)} iterations, |F| {hist[0][0]:.1e} -> {hist[-1][0]:.1e}, "
          f"{time.time() - t0:.0f} s", flush=True)
    print(f"ring: r_h={st.g.r_h:.10g} We={Y[M + 2]:.12f} lam={Y[M]:+.10f}", flush=True)
    np.savez(out, Y=Y, b=rm.b, breaks=st.P.breaks)
    print('wrote', out)


if __name__ == '__main__':
    if {'-h', '--help'} & set(sys.argv[1:]):
        print(__doc__)
        sys.exit()
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
