import warnings

import mpmath as mp
import numpy as np
import pytest

from bubbles.bie import ExteriorNeumann
from bubbles.geometry import (Geometry, Panels, ParamMap, adaptive_panels,
                              map_family, spheroid_coeffs)
from bubbles.kernels import ring_kernels

warnings.filterwarnings('ignore')


def brute_ring(r, z, rho, zeta, nr, nz, n=20000):
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dx = r - rho * np.cos(ph)
    dy = -rho * np.sin(ph)
    dz = z - zeta
    D2 = dx ** 2 + dy ** 2 + dz ** 2
    G = np.mean(rho / (4 * np.pi * np.sqrt(D2))) * 2 * np.pi
    nd = dx * nr * np.cos(ph) + dy * nr * np.sin(ph) + dz * nz
    D = np.mean(rho * nd / (4 * np.pi * D2 ** 1.5)) * 2 * np.pi
    return G, D


def test_ring_kernels_against_azimuthal_quadrature():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r, rho = rng.uniform(0.01, 2, 2)
        z, zeta = rng.uniform(-1, 1, 2)
        al = rng.uniform(0, 2 * np.pi)
        G, D = ring_kernels(r, z, rho, zeta, np.cos(al), np.sin(al))
        Gb, Db = brute_ring(r, z, rho, zeta, np.cos(al), np.sin(al))
        assert abs(G - Gb) < 1e-13 * abs(Gb)
        assert abs(D - Db) < 1e-12 * max(abs(Db), 1e-3)


def spheroid_A(aspect):
    """phi_d = A z on the surface of a spheroid in axial uniform flow."""
    a, c = mp.mpf(aspect), mp.mpf(1)
    al = a * a * c * mp.quad(lambda l: 1 / ((c * c + l) ** 1.5 * (a * a + l)), [0, 1, 10, mp.inf])
    return float(al / (2 - al))


def test_sphere():
    P = Panels.uniform(4)
    g = Geometry(np.zeros(4), P)
    bie = ExteriorNeumann(g)
    t = P.tf[:P.Nh]
    assert np.max(np.abs(bie.disturbance_potential() - np.cos(t) / 2)) < 1e-14


@pytest.mark.parametrize("aspect,beta", [(0.5, 0.0), (2.0, 0.0), (4.0, 0.75), (8.0, 0.89)])
def test_spheroids_exact(aspect, beta):
    pm = map_family([beta]) if beta else ParamMap()
    a = spheroid_coeffs(aspect, 160, pm)
    P = adaptive_panels(a, pm, tol=1e-13, h_max=0.06)
    g = Geometry(a, P, pm)
    bie = ExteriorNeumann(g)
    err = np.max(np.abs(bie.disturbance_potential() - spheroid_A(aspect) * g.z[:P.Nh]))
    assert err < 2e-13
