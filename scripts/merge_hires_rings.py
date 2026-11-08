"""Merge the ring chains of scripts/hires_rings.py into one zero-flux branch.

The chains overlap where the branch was cut for separate machines and where a
chain was restarted elsewhere. They are taken in priority order: a ring is kept
unless it lies within the r_h interval already covered by a chain taken before
(widened by 3%, less than half the step factor 1.07). Rings that are not resolved
(res_hi < 1e-6 with the curvature map, or Fourier tail < 1e-11; the test of
scripts/export_rings_viz.py) are left out first. Ranges that no new chain reaches
are then filled with the stored rings of results/rings/ (M = 128-320), so
the result always spans the whole branch; `source` in each record names its file.

  touch   touch0_p*        (to r_h = 1.0e-5)
  neck    neck0_p*
  downN, downN2 (neck map), down (curvature map, M = 240: under-resolved below r_h ~ 0.009)
  upA, upB, thin, thinD, thinC

Usage: python scripts/merge_hires_rings.py [out.json] [--report] [--dir=results/hires/rings]

--dir names the folder of the chain files (default: the committed chains); out.json
defaults to results/hires/rings_merged.json.
"""
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = os.path.join(ROOT, 'results', 'hires', 'rings')
PRIORITY = ['touch', 'neck', 'downN', 'downN2', 'down', 'upA', 'upB', 'thin', 'thinD', 'thinC']
STORED = ['neck3', 'neck2', 'neck', 'down2', 'down', 'up_hp', 'thin', 'thin2', 'thin3', 'thin7']
MARGIN = 1.03


def chain_files(name):
    """Part files of a chain (all segments, all parts), in order."""
    def key(f):
        return [int(x) for x in os.path.basename(f)[len(name):-5].replace('_p', ' ').split()]
    return sorted(glob.glob(os.path.join(HIRES, f'{name}[0-9]_p[0-9]*.json')), key=key)


def load(f):
    pts = json.load(open(f))
    rel = os.path.relpath(f, ROOT)
    for p in pts:
        p['source'] = rel
    return pts


def rkey(p):
    # rings are 7-10% apart in r_h; a resumed part re-solves its start ring, whose r_h may differ
    # from the previous part's last one in the 10th digit
    return round(math.log(p['r_h']), 6)


def resolved(p):
    return p['res_hi'] < 1e-6 or p.get('tail', 1.0) < 1e-11


def covered(intervals, r):
    return any(lo / MARGIN <= r <= hi * MARGIN for lo, hi in intervals)


def merge(report=False):
    taken, intervals, rows = [], [], []
    groups = [(n, [p for f in chain_files(n) for p in load(f)]) for n in PRIORITY]
    groups += [(f'stored:{s}', load(os.path.join(ROOT, 'results', 'rings', f'rings_{s}.json'))) for s in STORED]
    for name, pts in groups:
        keep = [p for p in pts if resolved(p) and not covered(intervals, p['r_h'])]
        # one record per hole radius within a chain: a resumed part re-solves its start ring (the later
        # solve is kept); the key is relative (log r_h), since the hole radii go down to 1e-5
        keep = list({rkey(p): p for p in keep}.values())
        if keep:
            r = [p['r_h'] for p in keep]
            if not name.startswith('stored:'):
                intervals.append((min(r), max(r)))
            taken += keep
        rows.append((name, len(pts), keep))
    # stored files share their end rings (rings_down ends where rings_down2 starts): keep the first taken
    seen, uniq = set(), []
    for p in taken:
        if rkey(p) not in seen:
            seen.add(rkey(p))
            uniq.append(p)
    taken = sorted(uniq, key=lambda p: p['r_h'])
    if report:
        final = {id(p) for p in taken}
        for name, n, keep in rows:
            k = [p['r_h'] for p in keep if id(p) in final]
            print(f'{name:14s} {n:4d} records, {len(k):4d} kept' +
                  (f'  r_h {min(k):.4g} .. {max(k):.4g}' if k else ''))
        gaps = [(a['r_h'], b['r_h']) for a, b in zip(taken, taken[1:]) if b['r_h'] / a['r_h'] > 1.25]
        print(f'{len(taken)} rings, r_h {taken[0]["r_h"]:.4g} .. {taken[-1]["r_h"]:.4g}; '
              f'steps over 25%: {[(round(a, 6), round(b, 6)) for a, b in gaps]}')
    return taken


if __name__ == '__main__':
    if {'-h', '--help'} & set(sys.argv[1:]):
        print(__doc__)
        sys.exit()
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    for a in sys.argv[1:]:
        if a.startswith('--dir='):
            HIRES = os.path.abspath(a.split('=', 1)[1])
    pts = merge(report='--report' in sys.argv)
    out = args[0] if args else os.path.join(ROOT, 'results', 'hires', 'rings_merged.json')
    with open(out + '.tmp', 'w') as f:
        json.dump(pts, f)
    os.replace(out + '.tmp', out)
    print('wrote', os.path.relpath(out, ROOT) if out.startswith(ROOT) else out)
