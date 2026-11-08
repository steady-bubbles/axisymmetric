"""Lowest eigenvalues of the second variation by symmetry sector along the bubble
branch (azimuthal mode m, parity under z -> -z), with the translations and the
volume change removed, and the number of negative eigenvalues per sector.  A change
of that number in any sector other than m = 0 even (the fold) would be a
symmetry-breaking bifurcation; by the monotonicity in m (bubbles/sectors.py) the
sectors m = 0, 1 decide all m.

Usage: python scripts/bubble_sectors.py out.json [--every=1] [--m=0,1] [--K=2] [--nev=4]
       [--nproc=$BUBBLES_NPROC or 2]
(--K: basis size as a multiple of the number of Fourier modes M.)  Points are spread
over a Pool; the output (a list of rows i, z_pole, We, lam, aspect, K, '<m><e|o>' (lowest
eigenvalues), '..._trans' (Rayleigh quotient of the exact translation), nneg) is
rewritten after each point.  A restart reuses the rows computed with the same K and
computes only the sectors (with nev eigenvalues) they lack; a point that fails is logged
and retried on the next run.
"""
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from bubbles.bubble import BubbleState  # noqa: E402
from bubbles.geometry import Panels, ParamMap  # noqa: E402
from bubbles.sectors import Sector  # noqa: E402


def _sector_row(i, p, ms, kf, nev, prev):
    t0 = time.time()
    st = BubbleState(np.array(p['a']), Panels(np.array(p['breaks_half'])), ParamMap(p['map_b']))
    K = int(kf * len(p['a']))
    row = dict(prev or {}, i=i, z_pole=p['z_pole'], We=p['We'], lam=p['lam'], aspect=p['aspect_ext'], K=kf)
    row['nneg'] = dict(row.get('nneg', {}))
    for m in ms:
        sec = Sector(st, m)
        for par in ('even', 'odd'):
            ev, tr = sec.eigs(p['We'], par, K, nev=nev)
            k = f'{m}{par[0]}'
            row[k] = [float(x) for x in ev]
            row['nneg'][k] = int(np.sum(ev < 0))
            if tr is not None:
                row[f'{k}_trans'] = float(tr)
    row['time'] = round(time.time() - t0, 1)
    return row


def sector_row(task):
    """Sectors m in ms (both parities) at branch point i, added to the earlier row prev (or
    None); task = (i, point, ms, K factor, nev, prev).  A failure returns {i, error}."""
    try:
        return _sector_row(*task)
    except Exception:  # noqa: BLE001
        return dict(i=task[0], error=traceback.format_exc())


def save_json(obj, path):
    """Write atomically, so readers never see a half-written file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def main():
    out = [a for a in sys.argv[1:] if not a.startswith('--')][0]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    every = int(opts.get('every', 1))
    ms = [int(x) for x in opts.get('m', '0,1').split(',')]
    kf, nev = float(opts.get('K', 2)), int(opts.get('nev', 4))
    nproc = int(opts.get('nproc', os.environ.get('BUBBLES_NPROC', 2)))
    B = json.load(open(os.path.join(ROOT, 'results', 'branch.json')))
    idx = sorted(set(list(range(0, len(B), every)) + [len(B) - 1]))
    keys = [f'{m}{q}' for m in ms for q in 'eo']
    old = {d['i']: d for d in (json.load(open(out)) if os.path.exists(out) else [])
           if 'nneg' in d and d.get('K') == kf}
    rows, tasks, failed = [d for i, d in old.items() if i not in idx], [], []
    for i in idx:
        prev = old.get(i)
        todo = [m for m in ms if prev is None or any(len(prev.get(f'{m}{q}', [])) < nev for q in 'eo')]
        if todo:
            tasks.append((i, B[i], todo, kf, nev, prev))
        else:
            rows.append(prev)
    print(f'{len(idx) - len(tasks)} points done before, {len(tasks)} to do on {nproc} processes', flush=True)
    with Pool(nproc) as pool:
        for row in pool.imap_unordered(sector_row, tasks):
            if 'error' in row:
                failed.append(row['i'])
                print(f"FAILED point {row['i']}:\n{row['error']}", flush=True)
                continue
            rows.append(row)
            rows.sort(key=lambda d: d['i'])
            save_json(rows, out)
            print(f"{row['i']:3d} z_pole={row['z_pole']:.3e} We={row['We']:.5f} "
                  + ' '.join(f"{k}={row[k][0]:+.3f}({row['nneg'][k]})" for k in keys)
                  + f" {row['time']:.0f}s", flush=True)
    rows = [d for d in sorted(rows, key=lambda d: d['i']) if all(k in d['nneg'] for k in keys)]
    for k in keys:
        for d0, d1 in zip(rows[:-1], rows[1:]):
            if d0['nneg'][k] != d1['nneg'][k]:
                print(f"{k}: {d0['nneg'][k]} -> {d1['nneg'][k]} negative between points {d0['i']} "
                      f"(We = {d0['We']:.5f}) and {d1['i']} (We = {d1['We']:.5f})")
    if failed:
        print(f'{len(failed)} points failed: {sorted(failed)}')


if __name__ == '__main__':
    main()
