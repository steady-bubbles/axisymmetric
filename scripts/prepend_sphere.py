"""Put the exact sphere (We = 0) at the start of a bubble branch file.

The continuation starts at We = 1e-3 (Branch.start_at_sphere); this adds the
state at We = 0 itself, the unit sphere with lambda = 2, solved and recorded
by the same code, so that the branch begins at the sphere. Nothing is changed
if the first record already has We = 0.

Usage: python scripts/prepend_sphere.py results/hires/branch.json
"""
import json
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


def sphere_record(M=16, **opts):
    br = Branch(M=M, log=lambda *a: None, **opts)
    Y = br.start_at_sphere(0.0)
    st = br.state(Y[:M], br.panels_for(Y[:M]))
    return br.record(Y, st, [(0.0, 0.0)], 0.0)


def main(path, **opts):
    pts = json.load(open(path))
    if pts and pts[0]['We'] == 0.0:
        print(f'{path}: already starts at We = 0')
        return
    d = sphere_record(**opts)
    assert abs(d['lam'] - 2) < 1e-12 and np.abs(d['a']).max() < 1e-12, (d['lam'], max(map(abs, d['a'])))
    with open(path + '.tmp', 'w') as f:
        json.dump([d] + pts, f)
    os.replace(path + '.tmp', path)
    print(f"{path}: sphere added (lam = {d['lam']!r}, z_pole = {d['z_pole']!r}), {len(pts) + 1} states")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/hires/branch.json')
