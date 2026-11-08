"""Export ring frames (from scripts/run_rings.py and scripts/ring_qscan.py output) for the explorer.

Each frame: the upper half of the cross-section from the inner rim (r_h, 0) to the outer
rim (r_o, 0), surface speed |u| = |gamma|, H and the pointwise residual of
H = lam + (We/2)|u|^2 along it (log10 in tenths, between the nodes), and streamlines, level sets of
Psi = r^2/2 + V[gamma] (Psi = 0 on the axis, Psi = Q on the ring, 2 pi Q the flux through
the hole):
  lines   in the explorer's fixed window r <= 3.2, z <= 2.2 at Psi = r_inf^2/2 (as for the
          bubbles; only rings that fit in that window), and for Q < 0 at Psi = Q/2 (the
          fluid circling the tube backwards through the hole),
  flines  in the fit-frame window fwin = [r_c, w, h] centred on the tube (r in
          [r_c - w, r_c + w], z in [0, h]) at evenly spaced Psi between its extremes,
  slines  for rings that do not fit the fixed window: in the window swin = [r_c, 3.2, 2.2]
          of the fixed window's size and scale, moved to the tube centre r_c, at evenly
          spaced Psi between its extremes (the explorer's default view of these rings, so
          the tube is drawn at one scale along the whole branch).
For Q > 0 the level Psi = Q off the ring has index -1: the fluid between it and the axis
passes through the hole (flux 2 pi Q).  For Q < 0 Psi > Q off the ring (the flow along the
ring does not stagnate), so there is none.
Coordinates are integers: r = r0 + r_int xy, z = z_int xy (xy per frame, from the neck
radius and the tube size); line points in units of 1e-3 (lines) or h/1000 about (r_c, 0)
(flines) or 1e-3 about (r_c, 0) (slines).  The deviations the explorer magnifies are below that rounding for thin rings and
the smallest necks, so they are exported at full precision (3 significant digits):
  tdev    (r_h > TUBE_RH, the explorer's tube panel) the cross-section closed by its mirror
          image about its centroid (r_c, 0): a0, its mean radius over the polar angle, r_c,
          max |rho/a0 - 1| over the samples, and rho/a0 - 1 = d unit at every step-th sample,
  ndev    (r_h <= TUBE_RH, the neck panel) r/r_h - cosh(z/r_h) = d unit at the samples from
          the inner rim up to the first with r >= 6 r_h.

Sheet records (--sheets, the Q-scans of scripts/ring_qscan.py at fixed r_h) with
res_hi < 1e-6 become a second family, sheet_frames: per column (r_h) at most --sheet-max
records spread evenly in Qhat = 2Q/R^2, coarser (--sheet-nt samples).  A Q = 0 record is
kept only when no zero-flux frame has its r_h.  Missing sheet files are skipped.

Usage: python scripts/export_rings_viz.py rings.json [more.json ...] out.json [--max-ro=inf]
       [--sheets='results/rings/sheet_rh*.json'] [--sheet-max=7] [--sheet-nt=181]
       [--reuse=earlier_out.json] [--nproc=$BUBBLES_NPROC or 2]
--reuse takes the streamlines (lines, fwin, flines, swin, slines: the expensive part, about
30 min on 2 processes for the zero-flux frames) from the frames of an earlier export with the
same record (r_h, Q, We) and sampling, computes only the streamline sets that export lacks, and
recomputes the rest (a few seconds per frame), so adding the
sheet columns later, or a change of the other fields, costs only the new streamlines.
"""
import glob
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from numpy.polynomial.legendre import leggauss

warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from bubbles.flowfield import ring_stream_function  # noqa: E402
from bubbles.quad import cheb_eval  # noqa: E402
from bubbles.ring import RingMap, RingPanels, RingState, ring_theta  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from export_viz import NR, NZ, R_INF, R_MAX, Z_MAX, rdp  # noqa: E402

NT = 361
FIT = 0.42            # the tube fills this fraction of the fit window (width and height)
NFL = 14              # about this many Psi levels in the fit window
TUBE_RH = 0.15        # as in the explorer: tube panel above, neck panel below
TSTEP = 3             # tdev at every TSTEP-th sample
NSL = 16              # about this many Psi levels in the shifted fixed-size window
STREAMS = ('lines', 'fwin', 'flines', 'swin', 'slines')


def resolved(p):
    # res_hi for the curvature map, or the Fourier tail for the neck map (whose res_hi the
    # 1/r_h neck curvatures inflate)
    return 'We' in p and (p['res_hi'] < 1e-6 or p.get('tail', 1.0) < 1e-11)


def fit_window(p):
    """[r_c, w, h]: the tube centre and the half-width and height of the fit-frame window,
    in the fixed window's aspect 3.2 : 2.2."""
    a = 0.5 * (p['r_o'] - p['r_h'])
    k = max(a / (FIT * R_MAX), p['z_max'] / (FIT * Z_MAX))
    return [0.5 * (p['r_o'] + p['r_h']), R_MAX * k, Z_MAX * k]


class Psi:
    """Stream function of the ring's vortex sheet (upper sources + mirror) plus r^2/2."""

    def __init__(self, st, x):
        P, g = st.P, st.g
        xs, ws = leggauss(24)
        uu = np.broadcast_to(xs, (P.np, 24))
        y = cheb_eval(g.A, uu[None])
        yp = cheb_eval(g.B, uu[None])
        gm = np.einsum('pqj,pj->pq', P.rule.lagrange(uu), st.gamma.reshape(P.np, P.p))
        self.w = (ws[None, :] * (P.h / 2)[:, None] * np.hypot(yp[0], yp[1]) * gm).ravel()
        self.rho, self.zeta = y[0].ravel(), y[1].ravel()
        from matplotlib.path import Path
        self.poly = Path(np.concatenate([np.stack([x[0], x[1]], 1), np.stack([x[0][::-1], -x[1][::-1]], 1)]))

    def grid(self, rg, zg):
        """Psi on the grid, masked inside the cross-section."""
        RR, ZZ = np.meshgrid(rg, zg)
        inside = self.poly.contains_points(np.stack([RR.ravel(), ZZ.ravel()], 1)).reshape(RR.shape)
        pts = np.stack([RR[~inside], ZZ[~inside]], 1)
        val = np.empty(len(pts))
        for k in range(0, len(pts), 3000):
            rr, zz = pts[k:k + 3000, :1], pts[k:k + 3000, 1:]
            val[k:k + 3000] = rr[:, 0] ** 2 / 2 + (ring_stream_function(rr, zz, self.rho[None], self.zeta[None])
                                                   + ring_stream_function(rr, zz, self.rho[None], -self.zeta[None])) @ self.w
        out = np.ma.masked_all(RR.shape)
        out[~inside] = val
        return out


def contours(rg, zg, F, levels, r0, unit, eps):
    """[[index, [r0, z0, r1, z1, ...]], ...] for the (index, level) pairs, simplified by
    RDP to eps and stored as integers in `unit` about (r0, 0)."""
    import contourpy
    gen = contourpy.contour_generator(rg, zg, F)
    out = []
    for k, lev in levels:
        for seg in gen.lines(lev):
            if len(seg) < 4:
                continue
            seg = rdp(np.asarray(seg), eps) - [r0, 0.0]
            out.append([k, np.round(seg / unit).astype(int).ravel().tolist()])
    return out


def digits3(d):
    """{'unit', 'd'}: the values d as integers in a unit of 3 significant digits of max |d|."""
    m = float(np.abs(d).max())
    unit = 10.0 ** (np.floor(np.log10(m)) - 2) if m > 0 else 1.0
    return dict(unit=float(f'{unit:.0e}'), d=np.round(d / unit).astype(int).tolist())


def tube_dev(g, x):
    """The tube panel's deviation from the mean circle at the samples x, with Gauss
    quadrature on the panels for the centroid r_c (of the cross-section closed by its mirror
    image) and the mean radius a0 = (1/pi) int rho dphi about (r_c, 0)."""
    P = g.P
    xs, ws = leggauss(24)
    uu = np.broadcast_to(xs, (P.np, 24))
    (r, z), (rp, zp) = cheb_eval(g.A, uu[None]), cheb_eval(g.B, uu[None])
    w = ws[None, :] * (P.h / 2)[:, None]
    rc = np.sum(w * r * z * rp) / np.sum(w * z * rp)
    a0 = abs(np.sum(w * ((r - rc) * zp - z * rp) / np.hypot(r - rc, z))) / np.pi
    d = np.hypot(x[0] - rc, x[1]) / a0 - 1
    return dict(a0=float(f'{a0:.7g}'), rc=float(rc), max=float(f'{np.abs(d).max():.3g}'), step=TSTEP,
                **digits3(d[::TSTEP]))


def neck_dev(x, r_h):
    """The neck panel's r/r_h - cosh(z/r_h) from the inner rim to the first sample at r >= 6 r_h."""
    n = int(np.argmax(x[0] >= 6 * r_h)) or x.shape[1]
    return digits3(x[0, :n] / r_h - np.cosh(x[1, :n] / r_h))


def fits(p):
    """The ring fits the fixed window (as the explorer's fits())."""
    return p['r_o'] <= R_MAX - 0.05


def even_levels(F, Q, n):
    """(index, Psi) pairs evenly spaced (about n of them) between the extremes of F, with
    the levels of the ring itself (Psi = Q) left out, plus Psi = Q (index -1) for Q > 0."""
    lo, hi = float(F.min()), float(F.max())
    step = (hi - lo) / n
    return [(k, Q + j * step) for k, j in enumerate(range(int(np.floor((lo - Q) / step)) + 1, int(np.ceil((hi - Q) / step))))
            if j != 0] + ([(-1, Q)] if Q > 0 else [])


def streamlines(p, st, x, Q, have=None):
    """lines (fixed window, rings that fit in it), fwin and flines (fit window), swin and
    slines (the shifted fixed-size window, rings that do not fit).  Parts already in `have`
    (an earlier export) are kept and not recomputed."""
    out = dict(have or {})
    need_l = fits(p) and 'lines' not in out
    need_f = 'flines' not in out
    need_s = not fits(p) and 'slines' not in out
    if not (need_l or need_f or need_s):
        return out
    psi = Psi(st, x)
    # fixed window: the explorer's r_inf levels, Psi = Q (Q > 0; bold) or Q/2 (Q < 0)
    if need_l:
        rg, zg = np.linspace(0, R_MAX, NR), np.linspace(0, Z_MAX, NZ)
        lev = list(enumerate(rinf ** 2 / 2 for rinf in R_INF))
        lev += [(-1, Q)] if Q > 0 else [(len(R_INF), Q / 2)] if Q < 0 else []
        out['lines'] = contours(rg, zg, psi.grid(rg, zg), lev, 0.0, 1e-3, 0.0025)
    # fit window: Psi levels evenly spaced between the extremes found in it
    if need_f:
        rc, w, h = fit_window(p)
        dx = 2 * w / (NR - 1)
        rg = np.arange(max(0.0, rc - w), rc + w + dx / 2, dx)
        zg = np.linspace(0, h, int(np.ceil(h / dx)) + 1)
        F = psi.grid(rg, zg)
        out['fwin'] = [float(f'{v:.7g}') for v in (rc, w, h)]
        out['flines'] = contours(rg, zg, F, even_levels(F, Q, NFL), rc, h * 1e-3, 0.0025 * h / Z_MAX)
    # shifted window: the fixed window's size and grid spacing, centred on the tube
    if need_s:
        rc = 0.5 * (p['r_o'] + p['r_h'])
        dx = 2 * R_MAX / (NR - 1)
        rg = np.arange(max(0.0, rc - R_MAX), rc + R_MAX + dx / 2, dx)
        zg = np.linspace(0, Z_MAX, NZ)
        F = psi.grid(rg, zg)
        out['swin'] = [float(f'{v:.7g}') for v in (rc, R_MAX, Z_MAX)]
        out['slines'] = contours(rg, zg, F, even_levels(F, Q, NSL), rc, 1e-3, 0.0025)
    return out


def frame(job):
    p, nt, keep = job
    rm = RingMap(p['map_b'])
    P = RingPanels(np.array(p['breaks']))
    a = np.array(p['a'])
    Q = p.get('Q', 0.0)
    st = RingState(a, p['rho_h'], rm, P, Q=Q)
    g = st.g
    # samples uniform in t (the map clusters them at the neck and the rims)
    t = np.linspace(0, np.pi, nt)
    idx = np.clip(np.searchsorted(P.b, t), 0, P.np - 1)
    u = np.clip((2 * t - P.a[idx] - P.b[idx]) / P.h[idx], -1, 1)
    x = cheb_eval(g.A[:, idx, :], u[None, :, None])[..., 0]
    th, tht = ring_theta(a, t)
    H = -tht / (g.L * rm.Sp(t)) - np.sin(th) / x[0]
    gam = np.einsum('nj,nj->n', P.rule.lagrange(u), st.gamma.reshape(P.np, P.p)[idx])
    res = np.abs(H - p['lam'] - 0.5 * p['We'] * gam ** 2)
    x[1, 0] = x[1, -1] = 0.0
    out = {k: p[k] for k in ('r_h', 'We', 'lam', 'r_o', 'z_max', 'circulation', 'kmax',
                             'u_max', 'M', 'N') if k in p}
    if 'H_waist_rh' in p:
        out['H_waist_rh'] = p['H_waist_rh']
    if Q != 0.0:
        R = 0.5 * (p['r_h'] + p['r_o'])
        out.update(Q=Q, Qhat=2 * Q / R ** 2)
    out.update(streamlines(p, st, x, Q, keep))
    if p['r_h'] > TUBE_RH:
        out['tdev'] = tube_dev(g, x)
    else:
        out['ndev'] = neck_dev(x, p['r_h'])
    # positions: integers in xy (from the neck radius and the tube), offset r0 for large rings
    tube = min(p['r_h'], 0.5 * (p['r_o'] - p['r_h']))
    xy = float(min(1e-5, max(1e-8, 10.0 ** (np.floor(np.log10(tube)) - 4))))
    r0 = float(p['r_h']) if p['r_h'] > 1 else 0.0
    out.update(kind='ring', xy=xy, r0=r0,
               r=np.round((x[0] - r0) / xy).astype(int).tolist(),
               z=np.round(x[1] / xy).astype(int).tolist(),
               u=np.round(np.abs(gam) * 1e4).astype(int).tolist(),
               H=np.round(H * 1e3).astype(int).tolist(),
               res=np.round(10 * np.log10(np.maximum(res, 1e-30))).astype(int).tolist())
    return out


def key(p, nt):
    return round(p['r_h'], 9), round(p.get('Q', 0.0), 12), round(p['We'], 12), nt


def pick_sheet(recs, nmax, have):
    """Per column (r_h) at most nmax records spread evenly in Qhat; Q = 0 only when the
    zero-flux frames lack that r_h."""
    cols = {}
    for p in recs:
        cols.setdefault(round(p['r_h'], 6), []).append(p)
    out = []
    for rh, ps in sorted(cols.items()):
        ps = sorted({round(p['Q'], 12): p for p in ps}.values(), key=lambda p: p['Q'])
        ps = [p for p in ps if p['Q'] != 0.0 or not any(abs(h / p['r_h'] - 1) < 1e-9 for h in have)]
        qh = np.array([2 * p['Q'] / (0.5 * (p['r_h'] + p['r_o'])) ** 2 for p in ps])
        if len(ps) > nmax:
            keep = sorted({int(np.argmin(np.abs(qh - q))) for q in np.linspace(qh[0], qh[-1], nmax)})
            ps = [ps[k] for k in keep]
        out += ps
    return out


if __name__ == '__main__':
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    srcs, out = args[:-1], args[-1]
    max_ro = float(opts.get('max-ro', 'inf'))
    nproc = int(opts.get('nproc', os.environ.get('BUBBLES_NPROC', 2)))
    pts = []
    for s_ in srcs:
        pts += json.load(open(s_))
    ok = [p for p in pts if resolved(p) and p.get('Q', 0.0) == 0.0 and p['r_o'] <= max_ro]
    print(f'{len(ok)} resolved zero-flux rings of {len(pts)} records')
    # the same ring in several runs (e.g. a high-precision rerun listed last): keep the last
    pts = sorted({round(p['r_h'], 9): p for p in ok}.values(), key=lambda p: p['r_h'])
    sheet = []
    for fn in sorted(glob.glob(opts.get('sheets', os.path.join(ROOT, 'results', 'rings', 'sheet_rh*.json')))):
        try:
            recs = json.load(open(fn))
        except (ValueError, OSError) as e:     # e.g. a run rewriting its file right now
            print(f'skipping {fn}: {e}')
            continue
        if isinstance(recs, list):            # not the runs' <out>.meta.json
            sheet += [p for p in recs if 'a' in p and p['res_hi'] < 1e-6]
    sheet = pick_sheet(sheet, int(opts.get('sheet-max', 7)), [p['r_h'] for p in pts])
    nts = int(opts.get('sheet-nt', 181))
    jobs = [(p, NT) for p in pts] + [(p, nts) for p in sheet]
    # streamlines of an earlier export of the same records (e.g. to add the sheet frames later)
    old = {}
    if 'reuse' in opts:
        if not os.path.exists(opts['reuse']):
            raise SystemExit(f"--reuse: no file {opts['reuse']}")
        doc = json.load(open(opts['reuse']))
        old = {key(f, len(f['r'])): {k: f[k] for k in STREAMS if k in f}
               for f in doc.get('frames', []) + doc.get('sheet_frames', []) if 'flines' in f}
    jobs = [(p, nt, old.get(key(p, nt))) for p, nt in jobs]
    n_new = sum(j[2] is None or (not fits(j[0]) and 'slines' not in j[2]) for j in jobs)
    print(f'{sum(j[2] is not None for j in jobs)} frames with streamlines reused, {n_new} need new streamlines; '
          f'{len(jobs)} frames on {nproc} processes')
    with Pool(nproc) as pool:
        frames = pool.map(frame, jobs, chunksize=1)
    doc = dict(scale=dict(u=1e-4, H=1e-3, lines=1e-3, res=0.1), frames=frames[:len(pts)],
               sheet_frames=frames[len(pts):])
    with open(out + '.tmp', 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    os.replace(out + '.tmp', out)
    print('wrote', out, len(pts), 'frames', len(sheet), 'sheet frames', os.path.getsize(out) / 1e6, 'MB')
