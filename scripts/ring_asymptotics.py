"""Small-hole asymptotics of the zero-flux ring branch.

Matched asymptotics (catenoid neck r = r_h cosh(z/r_h) glued to the touching
bubble) says that, seen from outside the neck, a ring with hole radius r_h
looks like a bubble whose pole sits at the catenoid height
z_eff = r_h log(1/r_h).  So for every outer quantity X

    X_ring(r_h) = X_bubble(z_pole = r_h log(1/r_h)) + O(r_h),

with X = We, lambda and |Gamma| <-> 2 phi_pole (the potential jump between
the poles).  In particular We = We* + We'(0) r_h log(1/r_h) + O(r_h) with
We'(0) = dWe/dz_pole at touching = 2.0364.

This script tests that law on the computed rings: it evaluates the bubble
branch at z_eff, fits D_X = (X_ring - X_bubble(z_eff))/r_h by
dA*log(1/r_h) + B + C*r_h*log(1/r_h) [+ E*r_h] and reports the leading
coefficient X'(0) + dA next to the bubble slope X'(0).

Usage: python scripts/ring_asymptotics.py [rings.json ...] [--branch=results/hires/branch.json]
           [--outdir=results/hires/asymptotics] [--rh-max=0.0393]
  (default rings: results/hires/rings_merged.json; --rh-max=inf takes all rings)
Writes ring_branch.json, asymptotics.md, ring_asymptotics.png and
bubble_pole_potential.json (a cache) to outdir. The defaults are the high-resolution
data, written to results/hires/asymptotics/. For the first computation:
  python scripts/ring_asymptotics.py results/rings/rings_down_part1.json \
      results/rings/rings_down2.json results/rings/rings_neck.json results/rings/rings_neck2.json \
      results/rings/rings_neck3.json --branch=results/branch.json --rh-max=inf --outdir=out
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

warnings.filterwarnings('ignore')

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from bubbles.bubble import BubbleState  # noqa: E402
from bubbles.geometry import Panels, ParamMap  # noqa: E402

RES = os.path.join(ROOT, 'results')
OPTS = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
SRC = [a for a in sys.argv[1:] if not a.startswith('--')] or [os.path.join(RES, 'hires', 'rings_merged.json')]
Z_CUT = 0.25          # bubble frames used for X_bubble(z_eff)
BRANCH = OPTS.get('branch', os.path.join(RES, 'hires', 'branch.json'))
RH_MAX = float(OPTS.get('rh-max', '0.0393'))
OUT = OPTS.get('outdir', os.path.join(RES, 'hires', 'asymptotics'))
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'bubble_pole_potential.json')
KEEP = ['r_h', 'We', 'lam', 'circulation', 'r_o', 'z_max', 'kappa1_waist', 'H_waist_rh',
        'u_max', 'kmax', 'res_hi', 'tail', 'M', 'N']
RES_TOL, TAIL_TOL = 1e-6, 1e-11   # resolved: res_hi (curvature map) or Fourier tail (neck map)
WINDOWS = (1e-3, 3e-4, 1e-4)      # small-hole fit windows r_h < cut

BLUE, ORANGE = '#2a78d6', '#eb6834'
TEXT2, GRID = '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': TEXT2, 'axes.labelcolor': '#0b0b0b',
    'xtick.color': TEXT2, 'ytick.color': TEXT2, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
    'figure.dpi': 150, 'savefig.bbox': 'tight'})


def pole_potential(st, k=10):
    """phi at the pole: even polynomial fit in t through the first nodes."""
    t = st.P.tf[:st.P.Nh]
    return float(np.polyval(np.polyfit(t[:k] ** 2, st.phi[:k], 5), 0.0))


def bubble_side():
    """Last monotone stretch of the bubble branch (z_pole -> 0) with phi_pole."""
    B = json.load(open(BRANCH))
    zp = np.array([p['z_pole'] for p in B])
    i0 = len(B) - 1
    while i0 > 0 and zp[i0 - 1] > zp[i0]:
        i0 -= 1
    seg = [p for p in B[i0:] if p['z_pole'] < Z_CUT]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows = []
    for p in seg:
        key = repr(p['z_pole'])
        if key not in cache:
            st = BubbleState(np.array(p['a']), Panels(np.array(p['breaks_half'])),
                             ParamMap(p['map_b']))
            cache[key] = pole_potential(st)
        rows.append((p['z_pole'], p['We'], p['lam'], cache[key]))
    with open(CACHE, 'w') as f:
        json.dump(cache, f, indent=0)
    return np.array(sorted(rows))


def load_rings():
    """Resolved ring points of all runs (later files win at equal r_h).

    Points of the curvature-map runs below r_h ~ 5e-3 hit the mode cap and
    are left out (res_hi > RES_TOL); neck-map points are judged by their tail."""
    pts, dropped = {}, 0
    for fn in SRC:
        if not os.path.exists(fn):
            continue
        for p in json.load(open(fn)):
            if p['r_h'] > RH_MAX:
                continue
            ok = p['res_hi'] < RES_TOL or p.get('tail', 1.0) < TAIL_TOL
            if ok:
                pts[round(p['r_h'], 10)] = {k: p[k] for k in KEEP if k in p}
            else:
                dropped += 1
    print(f'{len(pts)} resolved ring points, {dropped} under-resolved left out')
    return sorted(pts.values(), key=lambda p: p['r_h'])


def lsq(cols, y):
    X = np.array(cols).T
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c, float(np.abs(X @ c - y).max())


def main():
    b = bubble_side()
    zb = b[:, 0]
    small = zb < 2e-3
    star, slope, curv = {}, {}, {}
    spl = {}
    for j, name in [(1, 'We'), (2, 'lam'), (3, 'Gam')]:
        y = b[:, j] * (2 if name == 'Gam' else 1)         # |Gamma| <-> 2 phi_pole
        c = np.polyfit(zb[small], y[small], 3)
        star[name], slope[name], curv[name] = c[-1], c[-2], c[-3]
        spl[name] = CubicSpline(zb, y)

    rings = load_rings()
    with open(os.path.join(OUT, 'ring_branch.json'), 'w') as f:
        json.dump(rings, f, indent=1)
    r = np.array([p['r_h'] for p in rings])
    ell = np.log(1 / r)
    ze = r * ell
    X = {'We': np.array([p['We'] for p in rings]), 'lam': np.array([p['lam'] for p in rings]),
         'Gam': np.abs([p['circulation'] for p in rings])}

    fits = {}
    for name in X:
        D = (X[name] - spl[name](ze)) / r
        one = np.ones_like(r)
        f3, e3 = lsq([ell, one, r * ell], D)
        f4, e4 = lsq([ell, one, r * ell, r], D)
        f2, e2 = lsq([one, r * ell], D)
        fits[name] = dict(D=D, f2=f2, e2=e2, f3=f3, e3=e3, f4=f4, e4=e4, win={})
        for cut in WINDOWS:
            m = r < cut
            if m.sum() < 8:
                continue
            cols = [ell, one, r * ell, r, r * ell ** 2]
            fits[name]['win'][cut] = {k: lsq([c[m] for c in cols[:k]], D[m]) + (int(m.sum()),)
                                      for k in (4, 5)}
    # direct fit, We* fixed from the bubble side
    cW, eW = lsq([ze, r, r * ze], X['We'] - star['We'])

    sym = {'We': 'We', 'lam': 'λ', 'Gam': '|Γ|'}
    L = ['# Small-hole asymptotics of the zero-flux ring branch', '',
         'Generated by `scripts/ring_asymptotics.py` from ' +
         ', '.join('`' + os.path.relpath(s, ROOT) + '`' for s in SRC) +
         (f' (rings with r_h <= {RH_MAX:g})' if RH_MAX < float('inf') else ' (rings)') +
         ' and `' + os.path.relpath(BRANCH, ROOT) + '` (bubbles).', '',
         'Law tested: X_ring(r_h) = X_bubble(z_pole = r_h log(1/r_h)) + O(r_h) for '
         'X = We, λ, |Γ| (|Γ| is compared with 2φ_pole of the bubble).', '',
         '## Bubble side at touching (cubic fit, z_pole < 2e-3)', '',
         '| X | X* | X\'(0) = dX/dz_pole | X\'\'(0)/2 |', '|---|---|---|---|']
    for n in X:
        L.append(f'| {sym[n]} | {star[n]:.12f} | {slope[n]:.8f} | {curv[n]:.5f} |')
    L += ['', f'(|Γ|* = 2φ* = {star["Gam"]:.11f}.)', '',
          '## Leading coefficient from the rings', '',
          f'{len(r)} rings, r_h = {r.min():.4g} … {r.max():.4g} '
          f'(z_eff = {ze.min():.4f} … {ze.max():.4f}).', '',
          'D_X = (X_ring − X_bubble(z_eff))/r_h is fitted by dA·log(1/r_h) + B + C·r_h log(1/r_h) '
          '[+ E·r_h]; the ring value of the leading coefficient is X\'(0) + dA.', '',
          '| X | X\'(0) (bubble) | 3-term: X\'(0)+dA | B | max resid | 4-term: X\'(0)+dA | B | '
          'max resid | dA = 0: B | C | max resid |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for n in X:
        F = fits[n]
        L.append(f'| {sym[n]} | {slope[n]:.5f} | {slope[n] + F["f3"][0]:.4f} | {F["f3"][1]:.3f} | '
                 f'{F["e3"]:.1e} | {slope[n] + F["f4"][0]:.4f} | {F["f4"][1]:.3f} | {F["e4"]:.1e} | '
                 f'{F["f2"][0]:.3f} | {F["f2"][1]:.2f} | {F["e2"]:.1e} |')
    L += ['', '## Small holes: leading coefficient and O(r_h) constant', '',
          'Fits of D_X on windows r_h < cut with dA·log(1/r_h) + B + C r_h log(1/r_h) + E r_h '
          '[+ F r_h log²(1/r_h)]. B is the O(r_h) constant: X_ring = X_bubble(r_h log(1/r_h)) + B r_h + ….', '',
          '| X | window | n | terms | X\'(0) + dA | relative to X\'(0) | B | max resid |',
          '|---|---|---|---|---|---|---|---|']
    for n in X:
        for cut, W in fits[n]['win'].items():
            for k, (c, e, cnt) in W.items():
                L.append(f'| {sym[n]} | r_h < {cut:.0e} | {cnt} | {k} | {slope[n] + c[0]:.7f} | '
                         f'{c[0] / slope[n]:+.1e} | {c[1]:.5f} | {e:.1e} |')
    L += ['', f'Direct fit We − We* = A r_h log(1/r_h) + B r_h + C r_h² log(1/r_h) with We* = '
          f'{star["We"]:.10f} fixed: A = {cW[0]:.4f}, B = {cW[1]:.3f}, C = {cW[2]:.2f} '
          f'(max residual {eW:.1e}); prediction A = {slope["We"]:.4f}. This absolute least-squares fit is '
          'dominated by the largest holes, where We − We* is largest; the window fits above are the sharp test.', '',
          '## Ring points', '',
          'map: "curv." = curvature density (resolved when res_hi < 1e-6), "neck" = '
          '`ring_init.neck_map` (resolved when the Fourier tail max|a_k|, k > 0.85M, is < 1e-11).', '',
          '| r_h | We | We_d = 2We | λ | Γ/(VR) | r_o | z_max | M | map | res_hi or tail |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for p in rings:
        res = f'tail {p["tail"]:.0e}' if 'tail' in p else f'res_hi {p["res_hi"]:.1e}'
        L.append(f'| {p["r_h"]:.6e} | {p["We"]:.10f} | {2 * p["We"]:.10f} | {p["lam"]:.10f} | '
                 f'{p["circulation"]:.8f} | {p["r_o"]:.6f} | {p["z_max"]:.6f} | {p["M"]} | '
                 f'{"neck" if "tail" in p else "curv."} | {res} |')
    with open(os.path.join(OUT, 'asymptotics.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L[:40]))

    # figure: values (top) and the O(r_h) remainder (bottom)
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex='col')
    rr = np.geomspace(r.min() * 0.6, r.max() * 1.1, 200)
    zz = rr * np.log(1 / rr)
    lab = {'We': 'We', 'lam': r'$\lambda$', 'Gam': r'$|\Gamma|/(VR)$'}
    for k, n in enumerate(X):
        a = ax[0, k]
        a.semilogx(rr, spl[n](zz), color=BLUE, lw=1.6,
                   label=r'bubble at $z_{pole}=r_h\log(1/r_h)$')
        a.semilogx(rr, star[n] + slope[n] * zz, color=TEXT2, lw=1, ls='--',
                   label=r"$X^*+X'(0)\,r_h\log(1/r_h)$")
        a.semilogx(r, X[n], 'o', ms=5, color=ORANGE, mec='white', mew=1, label='rings', zorder=5)
        a.axhline(star[n], color=TEXT2, lw=0.8, ls=':')
        a.set_ylabel(lab[n])
        if k == 0:
            a.legend(frameon=False, fontsize=8, loc='upper left')
        a = ax[1, k]
        F = fits[n]
        c = F['f2']
        a.semilogx(rr, c[0] + c[1] * zz, color=TEXT2, lw=1, ls='--',
                   label=r'fit $B + C\,r_h\log(1/r_h)$ (all rings)')
        if 1e-4 in F['win']:
            Bw = F['win'][1e-4][5][0][1]
            a.axhline(Bw, color=BLUE, lw=1, ls=':', label=r'$B$ from $r_h<10^{-4}$')
        a.semilogx(r, F['D'], 'o', ms=5, color=ORANGE, mec='white', mew=1, zorder=5,
                   label=r'rings: $(X_{ring}-X_{bubble}(z_{eff}))/r_h$')
        a.set_ylabel(r'$(X_{ring}-X_{bubble}(z_{eff}))/r_h$', fontsize=9)
        a.set_xlabel(r'hole radius $r_h$')
        if k == 0:
            a.legend(frameon=False, fontsize=8, loc='lower left')
    for a in ax.ravel():
        a.set_xticks([1e-5, 1e-4, 1e-3, 1e-2])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0, 0].set_title('Weber number', fontsize=10, loc='left')
    ax[0, 1].set_title('Bernoulli constant', fontsize=10, loc='left')
    ax[0, 2].set_title(r'circulation $|\Gamma|$ vs. bubble $2\varphi_{pole}$', fontsize=10, loc='left')
    fig.tight_layout()
    out = os.path.join(OUT, 'ring_asymptotics.png')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
