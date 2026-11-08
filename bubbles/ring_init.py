"""Initial guesses for ring bubbles near the touching configuration.

A catenoidal neck r = r_h cosh(z / r_h) (zero mean curvature) is glued into
the near-touching bubble: the catenoid is followed up to the radius R_m where
its slope matches the bubble sheet, the sheet is lifted by the height
mismatch, and the lift is tapered to zero towards the outer rim.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .bubble import BubbleState
from .flowfield import surface_samples
from .geometry import Panels, ParamMap
from .ring import RingMap


def bubble_sheet(p, n=6000):
    st = BubbleState(np.array(p['a']), Panels(np.array(p['breaks_half'])), ParamMap(p['map_b']))
    t = np.linspace(0, np.pi / 2, n)
    r, z, gam, H = surface_samples(st, t)
    r[0] = 0.0
    z[-1] = 0.0
    return r, z, st


def glue_neck(r_b, z_b, r_h, taper=0.7):
    """Upper half of a ring cross-section (inner rim -> outer rim) as points."""
    r_eq = r_b.max()
    # sheet as a graph on [0, r_eq - taper*0.5] for slope matching
    k = np.argmax(r_b)
    g = CubicSpline(r_b[:k + 1], z_b[:k + 1])
    rr = np.linspace(r_h * 1.0001, 0.8 * r_eq, 20000)
    slope_c = r_h / np.sqrt(rr ** 2 - r_h ** 2)
    diff = slope_c - g(rr, 1)
    j = np.nonzero(np.diff(np.sign(diff)))[0][0]
    R_m = rr[j]
    zc_m = r_h * np.arccosh(R_m / r_h)
    delta = zc_m - g(R_m)
    # catenoid part, parametrised by z
    zc = np.linspace(0, zc_m, 400)
    neck = np.stack([r_h * np.cosh(zc / r_h), zc], axis=1)
    # lifted sheet with taper to the rim
    sel = r_b > R_m
    rs, zs = r_b[sel], z_b[sel]
    r1 = r_eq - taper
    w = np.where(rs < r1, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip((rs - r1) / taper, 0, 1))))
    sheet = np.stack([rs, zs + delta * w], axis=1)
    pts = np.vstack([neck, sheet])
    return pts, dict(R_m=R_m, delta=delta)


def curve_theta(pts):
    """Arclength and unwrapped tangent angle of a polyline (dense)."""
    d = np.diff(pts, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0], np.cumsum(ds)])
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    sm = 0.5 * (s[1:] + s[:-1])
    return s, sm, th


def design_map(sm, th, s_tot, K=48, ell=0.15, alpha=1.0 / 3.0, n=4000, smooth=0.0):
    """Parameter map with dt/dsigma ~ (1 + (ell kappa)^2)^(alpha/2).

    For small catenoidal necks use neck_map instead."""
    kap = np.gradient(th, sm)
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        kap = gaussian_filter1d(kap, smooth)
    sig = sm / s_tot
    dens = (1 + (ell * kap) ** 2) ** (alpha / 2)
    cum = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(sig))])
    tt_of_sig = np.pi * cum / cum[-1]
    tt = (np.arange(1, n + 1) - 0.5) * np.pi / n
    sig_of_t = np.interp(tt, tt_of_sig, sig)
    f = np.pi * sig_of_t - tt
    k = np.arange(1, K + 1)
    # S(t) = (t + sum b_k sin(kt)/k)/pi  ->  f = sum (b_k/k) sin(kt)
    ck = (2.0 / n) * (np.sin(np.outer(k, tt)) @ f)
    return RingMap(ck * k)


def fit_ring(pts, rmap, M, n=4096):
    s, sm, th = curve_theta(pts)
    s_tot = s[-1]
    tt = (np.arange(1, n + 1) - 0.5) * np.pi / n
    sig = rmap.S(tt)
    thf = np.interp(sig * s_tot, sm, th)
    f = thf - (np.pi / 2 - tt)
    k = np.arange(1, M + 1)
    a = (2.0 / n) * (np.sin(np.outer(k, tt)) @ f)
    rho_h = pts[0, 0] / s_tot
    return a, rho_h


def neck_map(Y, rmap, f=0.3, ell=0.3, alpha=0.5, K=384, width=0.02, n=16384, r_design=None):
    """Smooth parameter map for a ring with a small catenoidal neck.

    dt/dsigma ~ (1 + (ell kappa_s)^2)^(alpha/2) + A g(sigma),
    g(sigma) = (pi/2L) / sqrt(sin^2(pi sigma/2L) + sinh^2(pi r_h/2L)),
    sigma = arclength from the waist, L = total arclength.  g is the
    2L-periodic version of 1/sqrt(sigma^2 + r_h^2): its singularities
    sigma = +-i r_h coincide with those of the catenoid, and it is even about
    both ends, so sigma(t) - L t/pi is odd about t = 0 and t = pi and the sine
    coefficients of the map decay exponentially (a plain asinh term has a
    kink at the outer rim and gives only O(k^-2)).  Its integral is an
    incomplete elliptic integral.  kappa_s is the meridional curvature of the
    current solution (spectral, not finite differences), with the neck
    (sigma < 40 r_h) masked out, Gaussian-smoothed over `width` * L with even
    reflection at both ends.  A gives the neck term the fraction f of the
    parameter.  Returns the map and |b_k| (its own Fourier tail).

    r_design (default: the current r_h) replaces r_h in g.  A map designed for
    r_design keeps the solution's singularities sigma = +-i r_h at the full
    distance A pi/2 (in units of the t-scale) for every r_h >= r_design, but
    for r_h = 0.8 r_design that distance already drops to A asin(0.8), about
    60%.  So when the continuation goes to smaller holes, design ahead.
    """
    from scipy.ndimage import gaussian_filter1d
    from .ring import RingGeometry, RingPanels
    M = len(Y) - 3
    a, rho_h = Y[:M], Y[M + 1]
    P = RingPanels(np.linspace(0, np.pi, 513))
    g = RingGeometry(a, rho_h, rmap, P)
    L = g.L
    r_h = r_design if r_design is not None else g.r_h
    tq = np.linspace(0, np.pi, 200001)
    sq = L * rmap.S(tq)
    from .ring import ring_theta
    _, tht = ring_theta(a, tq)
    kap = -tht / (L * rmap.Sp(tq))
    # uniform sigma grid for the smooth part
    ns = 40001
    sg = np.linspace(0, L, ns)
    kg = CubicSpline(sq, kap)(sg)
    w = 0.5 * (1 + np.tanh((sg - 40 * r_h) / (10 * r_h)))
    base = (1 + (ell * kg * w) ** 2) ** (alpha / 2)
    # even reflection at both ends keeps the smoothed density symmetric there
    ext = np.concatenate([base[:0:-1], base, base[-2::-1]])
    ext = gaussian_filter1d(ext, width * (ns - 1), truncate=10.0)
    base = ext[ns - 1:2 * ns - 1]
    Cb = CubicSpline(sg, np.concatenate([[0], np.cumsum(0.5 * (base[1:] + base[:-1]) * np.diff(sg))]))
    Tb = float(Cb(L))
    from scipy.special import ellipkinc
    c2 = np.sinh(np.pi * r_h / (2 * L)) ** 2
    m = 1 / (1 + c2)
    F0 = ellipkinc(np.pi / 2, m)

    def G(s):
        return (F0 - ellipkinc(np.pi / 2 - np.pi * s / (2 * L), m)) / np.sqrt(1 + c2)

    def g(s):
        return (np.pi / (2 * L)) / np.sqrt(np.sin(np.pi * s / (2 * L)) ** 2 + c2)
    A = f / (1 - f) * Tb / G(L)

    def T(s):
        return Cb(s) + A * G(s)

    def dT(s):
        return Cb(s, 1) + A * g(s)
    Ttot = float(T(L))
    tt = (np.arange(1, n + 1) - 0.5) * np.pi / n
    target = tt / np.pi * Ttot
    s = np.interp(target, T(sg), sg)
    for _ in range(50):
        ds = (T(s) - target) / dT(s)
        s = np.clip(s - ds, 0, L)
        if np.abs(ds).max() < 1e-15 * L:
            break
    fv = np.pi * s / L - tt
    # DST-II on the midpoints tt: c_k = (2/n) sum_j fv_j sin(k tt_j).  The FFT avoids
    # the rounding of sin at arguments k*tt ~ 10^3, which leaves a 1e-13 floor in b_k
    from scipy.fft import dst
    ck = dst(fv, type=2)[:K] / n
    k = np.arange(1, K + 1)
    return RingMap(ck * k), np.abs(ck * k)


def tip_map(sm, th, s_tot, rho_tip, end, f=0.3, ell=0.3, alpha=0.5, K=128, width=0.02, n=16384):
    """Parameter map for a cross-section with a blunt tip of radius rho_tip at one end
    (end = 0: the inner rim, sigma = 0; end = 1: the outer rim, sigma = L), as at the
    facing ends of two nested rings across a narrow gap.

    dt/dsigma ~ (1 + (ell kappa)^2)^(alpha/2) + A g(d),  d = distance from the tip end,
    g(d) = (pi/2L) / sqrt(sin^2(pi d/2L) + sinh^2(pi rho_tip/2L)),
    the neck_map term with r_h -> rho_tip: even about both ends, so the sine coefficients
    of the map decay exponentially.  kappa from the polyline (sm, th) with the tip
    (d < 3 rho_tip) masked out, Gaussian-smoothed over width * L with even reflection.  A
    gives the tip term the fraction f of the parameter."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.special import ellipkinc
    L = float(s_tot)
    ns = 40001
    sg = np.linspace(0, L, ns)
    kap = np.interp(sg, sm, np.gradient(th, sm))
    d = sg if end == 0 else L - sg
    w = 0.5 * (1 + np.tanh((d - 3 * rho_tip) / rho_tip))
    base = (1 + (ell * kap * w) ** 2) ** (alpha / 2)
    ext = np.concatenate([base[:0:-1], base, base[-2::-1]])
    ext = gaussian_filter1d(ext, width * (ns - 1), truncate=10.0)
    base = ext[ns - 1:2 * ns - 1]
    Cb = CubicSpline(sg, np.concatenate([[0], np.cumsum(0.5 * (base[1:] + base[:-1]) * np.diff(sg))]))
    Tb = float(Cb(L))
    c2 = np.sinh(np.pi * rho_tip / (2 * L)) ** 2
    m = 1 / (1 + c2)
    F0 = ellipkinc(np.pi / 2, m)

    def G0(s):          # integral of g from the inner rim, for a tip there
        return (F0 - ellipkinc(np.pi / 2 - np.pi * s / (2 * L), m)) / np.sqrt(1 + c2)

    def g0(s):
        return (np.pi / (2 * L)) / np.sqrt(np.sin(np.pi * s / (2 * L)) ** 2 + c2)
    GL = float(G0(L))
    if end == 0:
        G, g = G0, g0
    else:
        def G(s):
            return GL - G0(L - s)

        def g(s):
            return g0(L - s)
    A = f / (1 - f) * Tb / GL

    def T(s):
        return Cb(s) + A * G(s)

    def dT(s):
        return Cb(s, 1) + A * g(s)
    Ttot = float(T(L))
    tt = (np.arange(1, n + 1) - 0.5) * np.pi / n
    target = tt / np.pi * Ttot
    s = np.interp(target, T(sg), sg)
    for _ in range(50):
        ds = (T(s) - target) / dT(s)
        s = np.clip(s - ds, 0, L)
        if np.abs(ds).max() < 1e-15 * L:
            break
    fv = np.pi * s / L - tt
    from scipy.fft import dst
    ck = dst(fv, type=2)[:K] / n
    k = np.arange(1, K + 1)
    return RingMap(ck * k)
