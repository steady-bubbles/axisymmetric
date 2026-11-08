"""Polynomial tools on [-1, 1]: Gauss-Legendre panels, interpolation,
Chebyshev coefficients, divided differences and graded singular rules."""
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial.legendre import leggauss


class PanelRule:
    """p-point Gauss-Legendre rule on [-1, 1] with associated matrices."""

    def __init__(self, p=16):
        self.p = p
        x, w = leggauss(p)
        self.x, self.w = x, w
        # barycentric weights
        diff = x[:, None] - x[None, :]
        np.fill_diagonal(diff, 1.0)
        self.bw = 1.0 / np.prod(diff, axis=1)
        # Chebyshev Vandermonde at the nodes and its inverse (values -> coeffs)
        V = C.chebvander(x, p - 1)
        self.V2C = np.linalg.inv(V)
        # differentiation matrix d/du at the nodes
        Dm = np.zeros((p, p))
        for j in range(p):
            e = np.zeros(p)
            e[j] = 1.0
            Dm[:, j] = C.chebval(x, C.chebder(self.V2C @ e))
        self.Dm = Dm
        # cumulative integration matrix: (int_{-1}^{x_i} l_j)
        Cm = np.zeros((p, p))
        for j in range(p):
            e = np.zeros(p)
            e[j] = 1.0
            Cm[:, j] = C.chebval(x, C.chebint(self.V2C @ e, lbnd=-1))
        self.Cm = Cm

    def lagrange(self, u):
        """Lagrange basis l_j(u) at points u (any shape) -> shape u.shape+(p,)."""
        u = np.asarray(u, dtype=float)
        d = u[..., None] - self.x
        exact = d == 0.0
        d = np.where(exact, 1.0, d)
        t = self.bw / d
        L = t / t.sum(axis=-1, keepdims=True)
        if exact.any():
            hit = exact.any(axis=-1)
            L[hit] = exact[hit].astype(float)
        return L


def cheb_eval(coef, u):
    """Evaluate Chebyshev series; coef has shape (..., deg+1) broadcasting
    against u of shape (..., Q).  Clenshaw recurrence."""
    n = coef.shape[-1]
    b1 = np.zeros(np.broadcast_shapes(coef.shape[:-1] + (1,), u.shape))
    b2 = np.zeros_like(b1)
    u2 = 2.0 * u
    for k in range(n - 1, 0, -1):
        b1, b2 = coef[..., k:k + 1] + u2 * b1 - b2, b1
    return coef[..., 0:1] + u * b1 - b2


def cheb_divided_differences(coef, u, v):
    """First and second divided differences of a Chebyshev series f.

    Returns (D1, D2) with
        D1 = (f(v) - f(u)) / (v - u),
        D2 = (f(v) - f(u) - f'(u)(v - u)) / (v - u)^2,
    computed by stable recurrences (no cancellation as u -> v).
    coef: (..., n); u: (..., Q); v: (..., 1).
    """
    n = coef.shape[-1]
    shape = np.broadcast_shapes(u.shape, v.shape, coef.shape[:-1] + (1,))
    Tv_prev = np.ones(shape)          # T_0(v)
    Tv = np.broadcast_to(v, shape).copy()  # T_1(v)
    Dk_prev = np.zeros(shape)         # D_0
    Dk = np.ones(shape)               # D_1
    Ek_prev = np.zeros(shape)         # E_0
    Ek = np.zeros(shape)              # E_1
    D1 = coef[..., 1:2] * Dk
    D2 = np.zeros(shape)
    u2 = 2.0 * u
    for k in range(1, n - 1):
        Dn = u2 * Dk + 2.0 * Tv - Dk_prev
        En = u2 * Ek + 2.0 * Dk - Ek_prev
        Tn = 2.0 * v * Tv - Tv_prev
        Dk_prev, Dk = Dk, Dn
        Ek_prev, Ek = Ek, En
        Tv_prev, Tv = Tv, Tn
        D1 = D1 + coef[..., k + 1:k + 2] * Dk
        D2 = D2 + coef[..., k + 1:k + 2] * Ek
    return D1, D2


def graded_rule(sigma=0.5, n=10, depth=48):
    """Composite Gauss-Legendre rule on [0, 1] geometrically graded towards 0.

    Subintervals [sigma^{k+1}, sigma^k], k = 0..depth-1, plus [0, sigma^depth],
    each with n Gauss-Legendre points.  Integrates f(s) + g(s) log(s) with
    analytic f, g to near machine precision.
    """
    x, w = leggauss(n)
    xs, ws = [], []
    edges = sigma ** np.arange(depth + 1)
    edges = np.append(edges, 0.0)
    for k in range(depth + 1):
        hi, lo = edges[k], edges[k + 1]
        xs.append(lo + (hi - lo) * (x + 1) / 2)
        ws.append((hi - lo) / 2 * w)
    return np.concatenate(xs), np.concatenate(ws)
