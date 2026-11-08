"""Azimuthally integrated Laplace kernels for axisymmetric surfaces.

For a target point x = (r, z) and a source ring y = (rho, zeta) in the meridian
half plane, with outward unit normal n_y = (n_rho, n_zeta) of the source, we
return (per unit meridional arclength of the source)

    G(x, y) = int_0^{2pi} rho / (4 pi |x - y|) dphi
    D(x, y) = int_0^{2pi} rho (x - y) . n_y / (4 pi |x - y|^3) dphi

i.e. the single and double layer kernels of the 3D Laplace equation,
G_3d = 1/(4 pi |x - y|).  Complete elliptic integrals are expressed through
Carlson's symmetric integrals, which are accurate in the logarithmic regime
k -> 1 and avoid the cancellation K - E for k -> 0:

    K(m) = R_F(0, 1-m, 1),   K(m) - E(m) = (m/3) R_D(0, 1-m, 1).

With a = (r + rho)^2 + dz^2, b = (r - rho)^2 + dz^2, m = 4 r rho / a and
m1 = 1 - m = b / a one finds

    G = rho K / (pi sqrt(a)),
    D = rho / (4 pi) * [ 4 (n_y . (x - y)) E / (b sqrt(a))
                         - 8 n_rho r R_D / (3 a^{3/2}) ].

The quantity (n_y . (x - y)) / b is bounded as y -> x but suffers from
cancellation when computed from coordinates.  Callers that know it accurately
(e.g. via divided differences) pass it in as ``ndx_over_b`` together with an
accurate ``b`` and ``dz``.
"""
import numpy as np
from scipy.special import elliprd, elliprf

FOUR_PI = 4.0 * np.pi


def ring_kernels_dz(r, rho, dz, nrho, nzeta, b=None, ndx_over_b=None):
    """Kernels (G, D) in terms of dz = z - zeta (all arrays broadcastable)."""
    if b is None:
        b = (r - rho) ** 2 + dz ** 2
    a = (r + rho) ** 2 + dz ** 2
    m1 = b / a
    m = 4.0 * r * rho / a
    rf = elliprf(0.0, m1, 1.0)
    rd = elliprd(0.0, m1, 1.0)
    sa = np.sqrt(a)
    G = rho * rf / (np.pi * sa)
    E = rf - (m / 3.0) * rd
    if ndx_over_b is None:
        ndx_over_b = (nrho * (r - rho) + nzeta * dz) / b
    D = rho / FOUR_PI * (4.0 * ndx_over_b * E / sa
                         - 8.0 * nrho * r * rd / (3.0 * a * sa))
    return G, D


def ring_kernels(r, z, rho, zeta, nrho, nzeta):
    """Kernels (G, D) for target (r, z), source (rho, zeta), normal n_y."""
    return ring_kernels_dz(r, rho, z - zeta, nrho, nzeta)
