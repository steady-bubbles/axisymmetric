"""Lowest eigenvalues of the second variation by symmetry sector along ring runs, the
number of negative eigenvalues per sector, and the crossings (changes of that number)
between neighbouring records.

Usage: python scripts/ring_sectors.py out.json run.json [run2.json ...] [--sectors=0o,0eQ,0eG,1e,1o]
       [--m=2,3,4] [--m-every=1] [--m-rhmax=inf] [--bounds] [--nev=4] [--K=64] [--c_h=20]
       [--nproc=$BUBBLES_NPROC or 2] [--from=rows.json,...] [--idx=lo:hi]

Runs are plain lists of ring records (rings_*.json, sheet_rh*.json, fixedQ_*.json; Q = 0 if
a record has none).  Sectors: '0o', '1e', '1o' (translations removed), '0eQ' and '0eG'
(m = 0 even, volume fixed and hole flux or circulation fixed).  --m adds both parities of
the direct sectors m on every --m-every-th record of a run with r_h <= --m-rhmax
(and, with --idx=lo:hi, only on the records lo <= idx < hi of each run: e.g. a scan end).
--bounds brackets m_c, the largest unstable even sector, from the 1e eigenpair alone
(RingSector.mc_bounds).  Records are spread over a Pool and the output is rewritten after
each one.  A restart (or --from, earlier outputs) reuses the rows computed with the same K
and c_h and computes only the sectors (with nev eigenvalues) and bounds they lack; a
record that fails is logged, listed under 'failed' and retried on the next run.

Rows: src, idx, r_h, Q, We, lam, N, K, c_h, res (max |F| of the record at these panels),
circulation, We2D = We Gamma^2/(4 pi^2 a) and mc_rp = (R/a) sqrt(1 - We2D) (the local
Rayleigh-Plateau count with swirl; a, R the tube and ring radii), mu {sector: nev lowest
eigenvalues}, nneg {sector: number of negative ones}, trans {'0o', '1e': Rayleigh quotient
of the exact translation, ~ the accuracy of mu}, mc_lo, mc_hi (--bounds); the earlier flat
keys ('1e', '1e_trans', '2e', ...) are kept for the old readers.
Crossings: changes of nneg between neighbours on one family (runs at one Q merged and
ordered by r_h; a Q-scan by its signed arclength; see families), located on the polynomial
through the eigenvalue that changes sign at the two bracketing records and up to one
neighbour on each side (a cubic in log r_h, or in the arclength of a scan or a line), with
r_h, Q, We and lam from the same interpolant: sector, m, parity, constraint, kind
('bifurcation', or 'fold' in 0e: a fold in We at fixed Q or Gamma, or a branch point in the
symmetric class), r_h, Q, We, lam, direction (along the family), bracket, Q_bracket, nneg,
srcs, interp (number of records in the interpolant), resolved (|mu| on both sides above 10x
the translation gauge, None without 0o or 1e; at the thin end 1o and the 0e neck mode sink
below it).
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubbles.ring import RingMap, RingState, ring_panels  # noqa: E402
from bubbles.sectors import RingSector  # noqa: E402

CONSTRAINT = {'Q': 'Q', 'G': 'Gamma', '': 'Q'}       # '' (m >= 1, 0o): neither enters


def parse_sector(s):
    """'1e' -> (1, 'even', 'Q'), '0eG' -> (0, 'even', 'Gamma'); 0e needs Q or G."""
    k = len(s) - len(s.lstrip('0123456789'))
    m, par, con = int(s[:k]), {'e': 'even', 'o': 'odd'}[s[k]], s[k + 1:]
    if (m, par) == (0, 'even') and con not in ('Q', 'G'):
        raise ValueError(f'sector {s}: m = 0 even needs a constraint, 0eQ or 0eG')
    if con and (m, par) != (0, 'even'):
        raise ValueError(f'sector {s}: a constraint only enters m = 0 even')
    return m, par, CONSTRAINT[con]


def save_json(obj, path):
    """Write atomically, so readers never see a half-written file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def _sector_row(src, idx, p, sectors, cfg, prev):
    t0 = time.time()
    a = np.array(p['a'])
    rm = RingMap(np.array(p['map_b']))
    P = ring_panels(a, rm, len(a), c_h=cfg['c_h'], tol=1e-10)
    Q, We = p.get('Q', 0.0), p['We']
    st = RingState(a, p['rho_h'], rm, P, Q=Q)
    g = st.g
    R, at = 0.5 * (g.r_h + g.r_o), 0.5 * (g.r_o - g.r_h)
    We2D = We * st.circulation ** 2 / (4 * np.pi ** 2 * at)
    row = dict(prev or {}, src=src, idx=idx, r_h=p['r_h'], Q=Q, We=We, lam=p['lam'], N=P.N, K=cfg['K'],
               c_h=cfg['c_h'], res=float(np.abs(st.F(We, p['lam'])[:-1]).max()), circulation=float(st.circulation),
               We2D=float(We2D), mc_rp=float(R / at * np.sqrt(max(1 - We2D, 0.0))))
    for k in ('mu', 'nneg', 'trans'):
        row[k] = dict(row.get(k, {}))
    row.update({k: p[k] for k in ('Qhat', 'dir', 's_arc', 's') if k in p})
    ms = sorted({parse_sector(s)[0] for s in sectors} | ({1} if cfg['bounds'] else set()))
    for m in ms:
        sec = RingSector(st, m)
        for s in sectors:
            mm, par, con = parse_sector(s)
            if mm != m:
                continue
            ev, tr = sec.eigs(We, par, K=cfg['K'], nev=cfg['nev'], constraint=con)
            row['mu'][s] = [float(x) for x in ev]
            row['nneg'][s] = int(np.sum(ev < 0))
            if tr is not None:
                row['trans'][s] = float(tr)
            if s[-1] in 'eo':                     # flat keys of the earlier output
                row[s] = row['mu'][s]
                if tr is not None:
                    row[f'{s}_trans'] = float(tr)
        if m == 1 and cfg['bounds']:
            b = sec.mc_bounds(We, K=cfg['K'])
            row.update(mc_lo=b['mc_lo'], mc_hi=b['mc_hi'])
        del sec
    row['time'] = round(time.time() - t0, 1)
    return row


def sector_row(task):
    """Sectors `sectors` (and the m_c bounds if cfg['bounds']) at one record, added to the
    earlier row prev (or None); task = (src, idx, record, sectors, cfg, prev).  A failure
    returns {src, idx, r_h, Q, error}."""
    src, idx, p = task[:3]
    try:
        return _sector_row(*task)
    except Exception:  # noqa: BLE001
        return dict(src=src, idx=idx, r_h=p['r_h'], Q=p.get('Q', 0.0), error=traceback.format_exc())


def families(rows):
    """Rows on one-parameter families as (rows, x), in order along the parameter x: runs at
    one Q that are monotone in r_h (the zero-flux branch is spread over several files)
    merged and ordered by r_h, x = log r_h; any other run (a Q-scan at fixed r_h, a line that
    turns in r_h or moves in Q) by its arclength s_arc or s, else a Q-scan by Q and a line in
    file order, x the chord length in (log r_h, We)."""
    runs, fam = {}, {}
    for d in rows:
        runs.setdefault(d['src'], []).append(d)
    for src, rs in runs.items():
        rs.sort(key=lambda d: d['idx'])
        dr = np.diff([d['r_h'] for d in rs])
        mono = len({d['Q'] for d in rs}) == 1 and (np.all(dr > 0) or np.all(dr < 0))
        fam.setdefault(('Q', rs[0]['Q']) if mono else ('run', src), []).extend(rs)
    out = []
    for (kind, _), rs in fam.items():
        k = next((k for k in ('s_arc', 's') if all(k in d for d in rs)), None)
        if kind == 'Q':
            rs.sort(key=lambda d: d['r_h'])
            x = np.log([d['r_h'] for d in rs])
        elif k is not None or len({d['r_h'] for d in rs}) == 1:
            k = k or 'Q'
            rs.sort(key=lambda d: d[k])
            x = np.array([d[k] for d in rs], float)
        else:
            x = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(np.log([d['r_h'] for d in rs])),
                                                          np.diff([d['We'] for d in rs])))])
        out.append((rs, x))
    return out


def interp_zero(t, y, guess):
    """Real zero in [0, 1] of the polynomial through (t, y) nearest guess, else guess."""
    r = [z.real for z in np.roots(np.polyfit(t, y, len(t) - 1)) if abs(z.imag) < 1e-9 and 0 <= z.real <= 1]
    return min(r, key=lambda z: abs(z - guess)) if r else guess


def poly_at(t, y, tz):
    """Value at tz of the polynomial through (t, y)."""
    return float(np.polyval(np.polyfit(t, y, len(t) - 1), tz))


def crossings(rows):
    out = []
    for fam, x in families(rows):
        for k in sorted({k for d in fam for k in d['nneg']}):
            m, par, con = parse_sector(k)
            sel = [i for i, d in enumerate(fam) if k in d['nneg']]
            pts, xs = [fam[i] for i in sel], x[sel]
            for i in range(len(pts) - 1):
                d0, d1 = pts[i], pts[i + 1]
                n0, n1 = d0['nneg'][k], d1['nneg'][k]
                if n0 == n1:
                    continue
                # the bracket and one neighbour on each side, scaled to t = 0, 1 at the bracket;
                # a neighbour closer than 0.1 bracket widths would only amplify noise
                t = (xs - xs[i]) / (xs[i + 1] - xs[i])
                win = [j for j in (i - 1, i, i + 1, i + 2) if 0 <= j < len(pts)
                       and (j in (i, i + 1) or t[j] < -0.1 or t[j] > 1.1)]
                tw = t[win]
                tr = [abs(v) for d in (d0, d1) for v in d['trans'].values()]
                for jj in range(min(n0, n1), max(n0, n1)):
                    mu0, mu1 = d0['mu'][k][jj], d1['mu'][k][jj]
                    lin = float(np.clip(mu0 / (mu0 - mu1), 0, 1)) if mu0 != mu1 else 0.5
                    tz = interp_zero(tw, [pts[j]['mu'][k][jj] for j in win], lin)
                    val = {q: poly_at(tw, [pts[j][q] for j in win], tz) for q in ('Q', 'We', 'lam')}
                    out.append(dict(
                        sector=k, m=m, parity=par, constraint=con if (m, par) == (0, 'even') else None,
                        kind='fold' if (m, par) == (0, 'even') else 'bifurcation',
                        r_h=float(np.exp(poly_at(tw, [np.log(pts[j]['r_h']) for j in win], tz))), **val,
                        direction='to negative' if n1 > n0 else 'to positive',
                        bracket=[d0['r_h'], d1['r_h']], Q_bracket=[d0['Q'], d1['Q']], nneg=[n0, n1],
                        srcs=[[d0['src'], d0['idx']], [d1['src'], d1['idx']]], interp=len(win),
                        resolved=bool(min(abs(mu0), abs(mu1)) > 10 * max(tr)) if tr else None))
    return sorted(out, key=lambda c: (c['r_h'], c['Q']))


def key_of(d):
    return round(d['r_h'], 9), round(d['Q'], 9)


def load_rows(fn):
    """Rows of an earlier output (json with 'rows') or one JSON row per line."""
    try:
        d = json.load(open(fn))
        return d['rows'] if isinstance(d, dict) else d
    except ValueError:
        return [json.loads(line) for line in open(fn) if line.strip()]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    out, runs = args[0], args[1:]
    sectors = opts.get('sectors', '0o,0eQ,0eG,1e,1o').split(',')
    for s in sectors:
        parse_sector(s)
    ms = [int(x) for x in opts['m'].split(',')] if opts.get('m') else []
    if any(m < 1 for m in ms):
        raise ValueError('--m takes m >= 1; m = 0 goes in --sectors (0o, 0eQ, 0eG)')
    m_every, m_rhmax = int(opts.get('m-every', 1)), float(opts.get('m-rhmax', np.inf))
    lo, hi = (int(x) if x else None for x in opts['idx'].split(':')) if opts.get('idx') else (None, None)
    cfg = dict(K=int(opts.get('K', 64)), c_h=float(opts.get('c_h', 20)), nev=int(opts.get('nev', 4)),
               bounds='bounds' in opts)
    nproc = int(opts.get('nproc', os.environ.get('BUBBLES_NPROC', 2)))
    settings = dict(cfg, sectors=sectors, ms=ms, m_every=m_every, m_rhmax=m_rhmax if np.isfinite(m_rhmax) else None,
                    runs=runs, **({'idx': opts['idx']} if opts.get('idx') else {}))

    def wanted(idx, p):
        inside = (lo is None or idx >= lo) and (hi is None or idx < hi)
        direct = [f'{m}{q}' for m in ms for q in 'eo'] if idx % m_every == 0 and p['r_h'] <= m_rhmax and inside else []
        return sectors + [s for s in direct if s not in sectors]

    old = {}
    for fn in ([out] if os.path.exists(out) else []) + (opts['from'].split(',') if opts.get('from') else []):
        old.update((key_of(d), d) for d in load_rows(fn)
                   if 'mu' in d and d.get('K') == cfg['K'] and d.get('c_h') == cfg['c_h'])
    rows, failed, done, tasks = [], [], set(), []
    for fn in runs:
        for idx, p in enumerate(json.load(open(fn))):
            key = key_of(dict(p, Q=p.get('Q', 0.0)))
            if key in done:                       # runs share their end points
                continue
            done.add(key)
            prev = old.get(key)
            todo = [s for s in wanted(idx, p) if prev is None or len(prev['mu'].get(s, [])) < cfg['nev']]
            bounds = cfg['bounds'] and (prev is None or 'mc_lo' not in prev)
            if todo or bounds:
                tasks.append((fn, idx, p, todo, dict(cfg, bounds=bounds), prev))
            else:
                rows.append(dict(prev, src=fn, idx=idx))
    print(f'{len(rows)} records done before, {len(tasks)} to do on {nproc} processes', flush=True)

    def result():
        rs = sorted(rows, key=lambda d: (d['Q'], d['r_h']))
        return dict(settings=settings, rows=rs, crossings=crossings(rs), failed=failed)
    with Pool(nproc) as pool:
        for row in pool.imap_unordered(sector_row, tasks):
            if 'error' in row:
                failed.append(row)
                print(f"FAILED {row['src']}:{row['idx']} r_h={row['r_h']:.5g} Q={row['Q']:+.4g}\n{row['error']}",
                      flush=True)
            else:
                rows.append(row)
                print(f"{row['src']}:{row['idx']} r_h={row['r_h']:.5g} Q={row['Q']:+.4g} We={row['We']:.5f} "
                      + ' '.join(f"{k}={v[0]:+.4f}({row['nneg'][k]})" for k, v in row['mu'].items())
                      + (f" m_c in [{row['mc_lo']}, {row['mc_hi']}]" if 'mc_lo' in row else '')
                      + f" mc_rp={row['mc_rp']:.2f} {row['time']:.0f}s", flush=True)
            save_json(result(), out)
    res = result()
    save_json(res, out)
    for c in res['crossings']:
        print(f"{c['kind']} {c['sector']}: r_h = {c['r_h']:.6g}, Q = {c['Q']:+.4g} (We = {c['We']:.6f}, "
              f"lam = {c['lam']:+.4f}) nneg {c['nneg'][0]} -> {c['nneg'][1]}"
              + {True: '', False: ' (below the accuracy gauge)', None: ' (no accuracy gauge: 0o, 1e not computed)'}[
                  c['resolved']])
    if failed:
        print(f'{len(failed)} records failed: ' + ', '.join(f"{d['src']}:{d['idx']}" for d in failed))


if __name__ == '__main__':
    main()
