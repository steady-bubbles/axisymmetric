"""The bubble branch from the sphere to touching, densely sampled at high resolution.

Same two stages as scripts/run_branch.py (pseudo-arclength through the fold until
z_pole < z_switch, then natural continuation in log z_pole), with a smaller
largest arclength step, a finer log z_pole spacing, and a finer discretisation:
more Fourier modes beyond those the decay test asks for (M_margin) and shorter
panels (h <= c_h / M). The truncation tolerance stays at 1e-12: the indicator
res_hi has a round-off floor near 1e-13, and with res_tol = 1e-13 the mode
count ran away past the fold (M 40 -> 88 and rising). The exact sphere
(We = 0) is put first (scripts/prepend_sphere.py).

Usage: python scripts/run_branch_hires.py out.json [--z_min=1e-5] [--ds_max=0.0125]
           [--dlog=0.03125] [--res_tol=1e-12] [--M_margin=16] [--c_h=1.4] [--panel_tol=1e-14]
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.continuation import Branch  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

opts = dict(z_min=1e-5, ds_max=0.0125, dlog=0.03125, res_tol=1e-12, M_margin=16, c_h=1.4,
            panel_tol=1e-14, z_switch=0.05)
args = [a for a in sys.argv[1:] if not a.startswith('--')]
for a in sys.argv[1:]:
    if a.startswith('--'):
        k, v = a[2:].split('=')
        if k not in opts:
            sys.exit(f'unknown option --{k}')
        opts[k] = type(opts[k])(float(v)) if not isinstance(opts[k], int) else int(v)
if not args:
    sys.exit(__doc__)
out = args[0]
print('options', opts, flush=True)

t0 = time.time()
br = Branch(M=16, panel_tol=opts['panel_tol'], res_tol=opts['res_tol'], c_h=opts['c_h'],
            M_margin=opts['M_margin'])
Y = br.start_at_sphere(1e-3)
Y, tau = br.run(Y, ds=min(0.03, opts['ds_max']), ds_max=opts['ds_max'], n_steps=4000, save=out,
                stop=lambda d: d['z_pole'] < opts['z_switch'])
print(f'stage 1 done: {len(br.points)} points, {time.time() - t0:.0f} s', flush=True)
lz = np.log(br.points[-1]['z_pole'])
values = np.arange(lz - opts['dlog'], np.log(opts['z_min']) - 1e-9, -opts['dlog'])
Y = br.run_natural(Y, 'log_z_pole', values, save=out)
br.save(out)
np.savez(out.replace('.json', '_final.npz'), Y=Y, map_b=br.pmap.b)
# the branch starts at the sphere itself (We = 0), not at the first continuation step
import prepend_sphere  # noqa: E402
prepend_sphere.main(out)
print(f'done: {len(br.points)} points, {time.time() - t0:.0f} s', flush=True)
