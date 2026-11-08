"""Method of fundamental solutions for the zero-flux exterior flow of a torus-like body.
Independent of bubbles/sheet.py: vortex rings (exact Green's function of E^2) placed on a
curve INSIDE the tube, collocation Psi = r^2/2 + sum q_k Psi_ring = 0 on the surface
(z -> -z symmetric pairs), least squares; gradients by complex-step differentiation."""
import numpy as np
from scipy.special import elliprd, elliprf


def psi_ring(r, z, rho, zeta):
    a = (r + rho) ** 2 + (z - zeta) ** 2
    b = (r - rho) ** 2 + (z - zeta) ** 2
    m1 = b / a
    return r * rho / (np.pi * np.sqrt(a)) * (2.0 / 3.0 * elliprd(0.0, m1, 1.0) - elliprf(0.0, m1, 1.0))


def basis(r, z, rs, zs):
    """Psi of symmetric pair of unit rings at (rs, +-zs), evaluated at (r, z) (any dtype)."""
    return psi_ring(r[:, None], z[:, None], rs[None, :], zs[None, :]) + \
        psi_ring(r[:, None], z[:, None], rs[None, :], -zs[None, :])


def solve(rc, zc, rs, zs, Q=0.0):
    """Collocation points (rc, zc) on the upper half surface, sources (rs, zs) (upper half)."""
    A = basis(rc.astype(float), zc.astype(float), rs, zs)
    rhs = Q - 0.5 * rc ** 2
    sc = np.abs(A).max(axis=0)
    q, *_ = np.linalg.lstsq(A / sc, rhs, rcond=1e-15)
    q = q / sc
    return q


def grad_psi(r, z, rs, zs, q, h=1e-25):
    """(Psi_r, Psi_z) of Psi = r^2/2 + sum q basis, by complex step."""
    s = np.maximum(np.abs(r), 1.0) * h
    pr = (basis(r + 1j * s, z + 0j, rs, zs) @ q).imag / s + r
    pz = (basis(r + 0j, z + 1j * s, rs, zs) @ q).imag / s
    return pr, pz


def psi_val(r, z, rs, zs, q):
    return 0.5 * r ** 2 + basis(r, z, rs, zs) @ q


def speed(r, z, nr, nz, rs, zs, q):
    """Signed tangential speed u.T = (1/r) dPsi/dn (n outward from the body, into the fluid)."""
    pr, pz = grad_psi(r, z, rs, zs, q)
    return (pr * nr + pz * nz) / r


def torus(R, a, nsrc=64, ncol=400, fsrc=0.5):
    """Exact circular torus: returns function giving |u| Fourier coefficients and circulation."""
    phc = np.pi * (np.arange(ncol) + 0.5) / ncol                    # upper half, phi from outer rim
    rc, zc = R + a * np.cos(phc), a * np.sin(phc)
    ps = np.pi * (np.arange(nsrc) + 0.5) / nsrc
    rs, zs = R + fsrc * a * np.cos(ps), fsrc * a * np.sin(ps)
    q = solve(rc, zc, rs, zs)
    G = 2 * q.sum()                                                  # total enclosed vortex strength
    return dict(q=q, rs=rs, zs=zs, G=G)


def torus_speed_fourier(R, a, nsrc=64, fsrc=0.5, nq=256, nmax=6):
    T = torus(R, a, nsrc=nsrc, ncol=max(4 * nsrc, 300), fsrc=fsrc)
    ph = np.pi * (np.arange(nq) + 0.5) / nq
    r, z = R + a * np.cos(ph), a * np.sin(ph)
    u = speed(r, z, np.cos(ph), np.sin(ph), T['rs'], T['zs'], T['q'])
    # boundary check: Psi on a finer set
    ph2 = np.linspace(0, np.pi, 1001)
    psi_err = np.abs(psi_val(R + a * np.cos(ph2), a * np.sin(ph2), T['rs'], T['zs'], T['q'])).max()
    # Fourier cosine coefficients on [0, pi] (midpoint rule = exact for trig polys of degree < 2nq)
    cu = np.array([(1 if n == 0 else 2) * np.mean(u * np.cos(n * ph)) for n in range(nmax + 1)])
    cu2 = np.array([(1 if n == 0 else 2) * np.mean(u ** 2 * np.cos(n * ph)) for n in range(nmax + 1)])
    # circulation as line integral of u around the full section: 2 * int_0^pi u a dphi
    circ_line = 2 * a * np.pi * np.mean(u)
    return dict(G=T['G'], circ_line=circ_line, cu=cu, cu2=cu2, psi_err=psi_err, u=u, ph=ph)
