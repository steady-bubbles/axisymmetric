"""Follow the branch of vacuum bubbles from the sphere (We = 0) through the
fold in We towards the touching limit (front and rear poles meet).

Stage 1: pseudo-arclength continuation from We ~ 0 until z_pole < z_switch.
Stage 2: natural continuation in log(z_pole), We unknown.

Usage: python scripts/run_branch.py [out.json] [z_min] [ds_max] [dlog]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.continuation import Branch  # noqa: E402

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

out = sys.argv[1] if len(sys.argv) > 1 else 'results/branch.json'
z_min = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
z_switch = 0.05
ds_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
dlog = float(sys.argv[4]) if len(sys.argv) > 4 else 0.125

br = Branch(M=16, panel_tol=1e-13, res_tol=1e-12, c_h=2.0)
Y = br.start_at_sphere(1e-3)
Y, tau = br.run(Y, ds=0.03, ds_max=ds_max, n_steps=600, save=out,
                stop=lambda d: d['z_pole'] < z_switch)
lz = np.log(br.points[-1]['z_pole'])
values = np.arange(lz - dlog, np.log(z_min) - 1e-9, -dlog)
Y = br.run_natural(Y, 'log_z_pole', values, save=out)
br.save(out)
np.savez(out.replace('.json', '_final.npz'), Y=Y, map_b=br.pmap.b)
