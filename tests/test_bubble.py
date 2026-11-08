import warnings

import numpy as np

from bubbles.bubble import BubbleState
from bubbles.geometry import Panels, map_family
from bubbles.newton import newton_fixed_we

warnings.filterwarnings('ignore')


def test_sphere_speed():
    P = Panels.uniform(8)
    st = BubbleState(np.zeros(8), P)
    t = P.tf[:P.Nh]
    assert np.max(np.abs(st.q - 9 / 4 * np.sin(t) ** 2)) < 2e-11
    assert np.max(np.abs(st.H - 2)) < 1e-12
    assert abs(st.diagnostics()['added_mass'] - 0.5) < 1e-13


def test_jacobian_against_finite_differences():
    a = np.array([0.15, 0.03, -0.01, 0.004, 0.001, -5e-4])
    pm = map_family([0.3])
    We, lam = 1.3, 1.7
    P = Panels.uniform(12)
    st = BubbleState(a, P, pm)
    J, dWe = st.jacobian(We)
    h = 1e-6
    Jfd = np.empty_like(J)
    for j in range(len(a)):
        e = np.zeros(len(a))
        e[j] = h
        Jfd[:, j] = (BubbleState(a + e, P, pm).residual(We, lam)
                     - BubbleState(a - e, P, pm).residual(We, lam)) / (2 * h)
    Jfd[:, -1] = (st.residual(We, lam + h) - st.residual(We, lam - h)) / (2 * h)
    assert np.max(np.abs(J - Jfd)) < 1e-7 * np.max(np.abs(J))
    dWe_fd = (st.residual(We + h, lam) - st.residual(We - h, lam)) / (2 * h)
    assert np.max(np.abs(dWe - dWe_fd)) < 1e-8


def test_dilation_invariance_of_speed():
    """The shape derivative of q vanishes for a dilation (q is scale free)."""
    a = np.array([0.1, 0.02, -0.005])
    P = Panels.uniform(8)
    st = BubbleState(a, P)
    # dilation dx = x: normal part v = x.n, tangential part w = x.T
    g = st.g
    v = g.r * g.n[0] + g.z * g.n[1]
    w = g.r * g.T[0] + g.z * g.T[1]
    phs = st.phi_s_full
    gneu = P.diff(g.r * v * phs) / (g.r * g.c)
    dphi = st.bie.solve(gneu[:P.Nh])
    dphi_s = (P.diff(st.bie.full_odd(dphi)) / g.c)[:P.Nh]
    qs = P.diff(phs ** 2) / g.c
    Nh = P.Nh
    dq = 2 * phs[:Nh] * dphi_s - 2 * g.kappa1[:Nh] * v[:Nh] * phs[:Nh] ** 2 + w[:Nh] * qs[:Nh]
    # q is invariant under dilations, so the shape-derivative formula must give 0
    assert np.max(np.abs(dq)) < 1e-8


def test_small_weber_asymptotics():
    """Meyer-Niebel-Seis (1.24) with gamma = 0: d = -(3/32) We (3cos^2 - 1),
    and aspect ratio chi = 1 + 9/32 We + O(We^2)."""
    P = Panels.uniform(8)
    We = 1e-4
    a, lam, st, hist = newton_fixed_we(np.zeros(24), 2.0, We, P)
    g = st.g
    rad = np.hypot(g.r, g.z)
    pol = np.arctan2(g.r, g.z)
    eta = (rad - 1) / We
    assert np.max(np.abs(eta + 3 / 32 * (3 * np.cos(pol) ** 2 - 1))) < 1e-4
    chi = st.diagnostics()['aspect']
    assert abs((chi - 1) / We - 9 / 32) < 2e-4
    assert abs((lam - 2) / We + 0.75) < 1e-3
