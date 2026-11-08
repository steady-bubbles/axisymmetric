"""Export a bubble branch for the page: per frame the meridian curve, surface
speed and curvature along the arc, and streamlines.

z_max, aspect_ext = r_eq/z_max and u_max are taken from dense sampling of the
surface (NDENSE points, refined by a parabola through the largest sample), not
from the quadrature nodes as in the branch records, which lie below the true
maxima by up to 5e-7 (the sphere: z_max = 0.99999980 and u_max = 1.4999997
instead of 1 and 1.5).

Usage: python scripts/export_viz.py [results/hires/branch.json] [results/hires/viz/bubble_viz.json]
           [nproc] [--step=2] [--reuse=earlier_viz.json]
(paths relative to the current directory; nproc defaults to BUBBLES_NPROC, else 3)
--step=k exports every k-th state (and the last); the committed export used k = 2 (the
default). The touching fit uses all states.
--reuse takes the streamlines of frames with the same We and z_pole from an
earlier export (the expensive part) and recomputes the rest.
"""
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.bubble import BubbleState  # noqa: E402
from bubbles.flowfield import StreamFunction, surface_samples  # noqa: E402
from bubbles.geometry import Panels, ParamMap  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

R_MAX, Z_MAX = 3.2, 2.2
NR, NZ = 129, 89
R_INF = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.45, 1.7, 2.0, 2.3, 2.65, 3.0]
NT = 161


def rdp(pts, eps):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    nrm = np.hypot(*ab)
    if nrm == 0:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / nrm
    k = int(np.argmax(d))
    if d[k] > eps:
        left = rdp(pts[:k + 1], eps)
        right = rdp(pts[k:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


NDENSE = 8001


def dense_max(t, y):
    """Largest value of samples y(t), refined by the parabola through the largest sample and
    its neighbours (or the end sample itself)."""
    k = int(np.argmax(y))
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(y1 - (y0 - y2) ** 2 / (8 * den))
    return float(y[k])


def extrema(st):
    """True z_max and u_max of a state from dense samples of the half meridian."""
    t = np.linspace(0, np.pi / 2, NDENSE)
    _, z, gam, _ = surface_samples(st, t)
    return dense_max(t, z), dense_max(t, np.abs(gam))


def frame(job):
    p, old = job if isinstance(job, tuple) else (job, None)
    st = BubbleState(np.array(p['a']), Panels(np.array(p['breaks_half'])), ParamMap(p['map_b']))
    t = np.linspace(0, np.pi / 2, NT)
    r, z, gam, H = surface_samples(st, t)
    r[0], z[-1] = 0.0, 0.0
    s = st.g.L * st.pmap.S(t)
    z_max, u_max = extrema(st)
    if old is not None:
        lines = old
    else:
        lines = streamlines(st, r, z)
    keys = ['We', 'lam', 'z_pole', 'r_eq', 'z_max', 'aspect', 'aspect_ext', 'star_margin',
            'kmax', 'u_max', 'added_mass', 'area', 'M', 'N']
    out = {k: p[k] for k in keys}
    out.update(z_max=max(z_max, p['z_pole']), u_max=u_max)
    out['aspect_ext'] = p['r_eq'] / out['z_max']
    out.update(r=np.round(r * 1e4).astype(int).tolist(), z=np.round(z * 1e4).astype(int).tolist(),
               s=np.round(s * 1e4).astype(int).tolist(),
               u=np.round(np.abs(gam) * 1e4).astype(int).tolist(),
               H=np.round(H * 1e3).astype(int).tolist(), lines=lines)
    return out


def streamlines(st, r, z):
    import contourpy
    # stream function on the upper half plane; mask the bubble interior
    sf = StreamFunction(st, n_sub=24)
    rg = np.linspace(0, R_MAX, NR)
    zg = np.linspace(0, Z_MAX, NZ)
    RR, ZZ = np.meshgrid(rg, zg)
    ztop = np.interp(RR, r, z, right=-1.0)
    inside = (RR <= r.max()) & (ZZ <= ztop)
    Psi = np.full(RR.shape, -1.0)
    Psi[~inside] = sf(RR[~inside], ZZ[~inside])
    gen = contourpy.contour_generator(rg, zg, Psi)
    lines = []
    for k, rinf in enumerate(R_INF):
        for seg in gen.lines(rinf ** 2 / 2):
            if len(seg) < 4:
                continue
            seg = rdp(np.asarray(seg), 0.0025)
            lines.append([k, (np.round(seg * 1e3).astype(int)).ravel().tolist()])
    return lines


if __name__ == '__main__':
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    SRC = argv[0] if len(argv) > 0 else 'results/hires/branch.json'
    OUT = argv[1] if len(argv) > 1 else 'results/hires/viz/bubble_viz.json'
    NPROC = int(argv[2]) if len(argv) > 2 else int(os.environ.get('BUBBLES_NPROC', 3))
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    STEP = int(opts.get('step', 2))
    pts = json.load(open(SRC))
    sel = [pts[i] for i in sorted(set(range(0, len(pts), STEP)) | {len(pts) - 1})]
    key = lambda f: (round(f['We'], 12), round(f['z_pole'], 12))
    old = {}
    if 'reuse' in opts:
        old = {key(f): f['lines'] for f in json.load(open(opts['reuse']))['frames']}
    jobs = [(p, old.get(key(p))) for p in sel]
    print(f'{len(jobs)} frames, streamlines reused for {sum(j[1] is not None for j in jobs)}', flush=True)
    with Pool(NPROC) as pool:
        frames = pool.map(frame, jobs, chunksize=4)
    # fold, dimple onset and loss of star-shapedness (scripts/special_points.py): from the branch's
    # folder if it has them, else from results/special_points.json
    spf = os.path.join(os.path.dirname(SRC), 'special_points.json')
    if not os.path.exists(spf):
        spf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'special_points.json')
    sp = json.load(open(spf))
    fin = sp[list(sp)[-1]]
    zp = np.array([p['z_pole'] for p in pts])
    We = np.array([p['We'] for p in pts])
    s = zp < 1e-3
    We_star = float(np.linalg.lstsq(np.vander(zp[s], 4), We[s], rcond=None)[0][-1])
    lam_star = float(np.linalg.lstsq(np.vander(zp[s], 4), np.array([p['lam'] for p in pts])[s],
                                     rcond=None)[0][-1])
    special = {k: {kk: fin[k][kk] for kk in ('We', 'z_pole', 'aspect_ext', 'lam')}
               for k in ('fold', 'dimple_onset', 'star_loss')}
    special['touching'] = dict(We=We_star, z_pole=0.0, lam=lam_star,
                               aspect_ext=pts[-1]['aspect_ext'])
    data = dict(scale=dict(xy=1e-4, u=1e-4, H=1e-3, lines=1e-3), r_inf=R_INF,
                window=dict(r=R_MAX, z=Z_MAX), special=special, frames=frames,
                source=dict(file=SRC, states=len(pts), step=STEP, fit_states=int(s.sum())))
    with open(OUT, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print('wrote', OUT, os.path.getsize(OUT) / 1e6, 'MB')
