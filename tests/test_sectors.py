"""Sector second variation: exact checks on the sphere, translation zero modes, the
monotonicity in m, the fixed-flux m = 0 even sector of rings, and the crossings, restarts
and failures of scripts/ring_sectors.py."""
import json
import os
import sys

import numpy as np
import pytest
from scipy.special import lpmv

from bubbles.bubble import BubbleState
from bubbles.geometry import Panels, ParamMap
from bubbles.ring import RingMap, RingState, branch_max, ring_panels
from bubbles.sectors import RingSector, Sector

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import ring_sectors as rs  # noqa: E402


def _sphere():
    return BubbleState(np.zeros(16), Panels.uniform(8), ParamMap())


def test_mode_m_neumann_on_sphere():
    # exterior Neumann data Y_l^m on the unit sphere -> psi = -Y_l^m / (l + 1)
    st = _sphere()
    t, Nh = st.P.tf, st.P.Nh
    for m in (0, 1, 2, 3):
        sec = Sector(st, m)
        for l in range(max(m, 1), m + 3):
            Y = lpmv(m, l, np.cos(t))
            par = 'even' if (l - m) % 2 == 0 else 'odd'
            psi = sec.neumann(Y[:Nh], par)
            assert np.abs(psi + Y[:Nh] / (l + 1)).max() < 1e-10 * max(1, np.abs(Y).max())


def test_sphere_spectrum_at_zero_weber():
    # E'' = int |grad v|^2 - 2 v^2 on the unit sphere: (l - 1)(l + 2) on Y_l^m,
    # with the volume change (l = 0) and the translations (l = 1) removed
    st = _sphere()
    expect = {(0, 'even'): 4, (0, 'odd'): 10, (1, 'even'): 10, (1, 'odd'): 4, (2, 'even'): 4, (2, 'odd'): 10}
    for (m, par), mu in expect.items():
        ev, tr = Sector(st, m).eigs(0.0, par, 20)
        assert abs(ev[0] - mu) < 1e-8, (m, par, ev[:3])
        if tr is not None:
            assert abs(tr) < 1e-10


def test_ring_translation_is_a_zero_mode():
    # converged zero-flux ring at r_h = 0.039 (the Rayleigh quotient of a translation
    # is of the size of the base-state residual)
    p = json.load(open(os.path.join(ROOT, 'results', 'rings', 'rings_down_part1.json')))[0]
    Y = np.concatenate([p['a'], [p['lam'], p['rho_h'], p['We']]])
    rm = RingMap(np.array(p['map_b']))
    M = len(Y) - 3
    P = ring_panels(Y[:M], rm, M, c_h=20.0, tol=1e-10)
    st = RingState(Y[:M], Y[M + 1], rm, P)
    for m, par in ((1, 'even'), (0, 'odd')):
        ev, tr = RingSector(st, m).eigs(Y[M + 2], par, K=48)
        assert abs(tr) < 1e-7, (m, par, tr)


def _ring(p):
    a = np.array(p['a'])
    rm = RingMap(np.array(p['map_b']))
    return RingState(a, p['rho_h'], rm, ring_panels(a, rm, len(a), c_h=20.0, tol=1e-10), Q=p.get('Q', 0.0))


def _rings_up():
    return json.load(open(os.path.join(ROOT, 'results', 'rings', 'rings_up.json')))


def test_ring_sectors_grow_with_m():
    # r_h = 0.977: mu_1(m+1) >= mu_1(m) + (2m+1)/r_o^2 (monotonicity lemma), the m_c bracket
    # from the 1e eigenpair holds the direct count (2e < 0 < 3e), and the 1e eigenvectors
    # have their eigenvalues as Rayleigh quotients and are orthogonal to the translation
    p = _rings_up()[23]
    st = _ring(p)
    mu, sec1 = [], None
    for m in range(1, 6):
        sec = RingSector(st, m)
        mu.append(sec.eigs(p['We'], 'even')[0][0])
        if m == 1:
            sec1 = sec
    r_o = st.g.r_o
    for m in range(1, 5):
        assert mu[m] - mu[m - 1] >= (2 * m + 1) / r_o ** 2 - 1e-8, (m, mu)
    b = sec1.mc_bounds(p['We'])
    assert abs(b['mu1'] - mu[0]) < 1e-10 and (b['mc_lo'], b['mc_hi']) == (2, 3)
    assert b['mc_lo'] <= sum(x < 0 for x in mu) <= b['mc_hi']
    ev, tr, V = sec1.eigs(p['We'], 'even', nev=2, vectors=True)
    G = sec1.G
    dS = 2 * np.pi * G.r * G.panels.wf * G.c
    H, Gm = sec1.hessian(p['We'], 'even', 0, extra=V)
    assert np.allclose(np.diag(H) / np.diag(Gm), ev, rtol=0, atol=1e-9)
    assert abs(Gm[0, 1]) < 1e-10 and abs(np.sum(V[0] * G.n[0] * dS)) < 1e-10
    # the series evaluated at the nodes of the upper half (translation row from the normals)
    Vt = sec1.eigs(p['We'], 'even', nev=2, vectors=True, at=st.P.tf)[2]
    assert np.abs(Vt - V[:, :st.P.N]).max() < 1e-10 * np.abs(V).max()


def test_ring_m0_even_fixed_flux_crosses_at_the_We_maximum():
    # the fixed-flux, volume-constrained 0e sector: a neck mode below zero throughout and a
    # second eigenvalue crossing zero at the We maximum of the branch (r_h = 0.116211,
    # We = 1.5012047), located by the driver's cubic through rings_up[6..9]; without the
    # rank-one flux term (fixed circulation) the zero sits at 0.11559
    U = _rings_up()
    r_max, W_max = branch_max([p['r_h'] for p in U], [p['We'] for p in U])
    rows = []
    for i in range(6, 10):
        p = U[i]
        sec = RingSector(_ring(p), 0)
        eQ, tr = sec.eigs(p['We'], 'even', nev=3)
        eG, _ = sec.eigs(p['We'], 'even', nev=3, constraint='Gamma')
        assert tr is None and eQ[0] < -20 and np.all(eQ <= eG + 1e-10)
        rows.append(dict(src='rings_up', idx=i, r_h=p['r_h'], Q=0.0, We=p['We'], lam=p['lam'], trans={},
                         mu={'0eQ': list(eQ), '0eG': list(eG)},
                         nneg={'0eQ': int(sum(eQ < 0)), '0eG': int(sum(eG < 0))}))
    assert [d['nneg']['0eQ'] for d in rows] == [2, 2, 1, 1]      # rings_up[7] (r_h = 0.104), [8] (0.120)
    c = {d['sector']: d for d in rs.crossings(rows)}
    assert c['0eQ']['kind'] == 'fold' and c['0eQ']['interp'] == 4 and c['0eQ']['resolved'] is None
    assert abs(c['0eQ']['r_h'] - r_max) < 2e-5 and abs(c['0eQ']['We'] - W_max) < 1e-6, (c['0eQ'], r_max, W_max)
    assert abs(c['0eG']['r_h'] - r_max) > 4e-4
    with pytest.raises(ValueError):
        sec.eigs(U[8]['We'], 'even', constraint='flux')


def test_fixed_flux_identity():
    # at fixed circulation the hole flux moves by dQ = -int v gamma gamma_1 dS/(2 pi Gamma_1)
    # (the rank-one term of 0e); finite differences of Gamma = Gamma_0 + Q Gamma_1 at Q = 0.5
    # on a closed cross-section (even k only), along a_2, a_4 and rho_h
    a0 = np.zeros(8)
    a0[1], a0[3] = 0.15, 0.05
    rm = RingMap()
    P = ring_panels(a0, rm, 8, c_h=2.0, tol=1e-12)
    rho0, Q, h = 0.4, 0.5, 1e-5
    st = RingState(a0, rho0, rm, P, Q=Q)
    sec = RingSector(st, 0)
    G = sec.G
    w = 2 * np.pi * G.r * G.panels.wf * G.c * sec.phs * np.concatenate([sec.g1, sec.g1[::-1]])
    n = np.stack([-np.sin(st.g.theta), np.cos(st.g.theta)])
    for k in (1, 3, None):
        sts = []
        for sg in (1, -1):
            a = a0.copy()
            if k is not None:
                a[k] += sg * h
            sts.append(RingState(a, rho0 + (sg * h if k is None else 0), rm, P, Q=Q))
        dQ = -(sts[0].circulation - sts[1].circulation) / (2 * h) / sec.G1
        v = np.sum(n * (np.stack([sts[0].g.r, sts[0].g.z]) - np.stack([sts[1].g.r, sts[1].g.z])), axis=0) / (2 * h)
        b = np.sum(np.concatenate([v, v[::-1]]) * w)
        assert abs(dQ + b / (2 * np.pi * sec.G1)) < 1e-7 * abs(dQ), (k, dQ, b)


def test_crossings_interpolation():
    # polynomial eigenvalues are located exactly, in log r_h along a constant-Q family (two
    # files merged; the neighbour 0.1495 is too near the bracket and dropped) and in the
    # arclength of a Q-scan (Q, We from the same cubic); resolved compares with the
    # translation gauge at the bracket (None without one)
    lr = np.log([0.1, 0.12, 0.1495, 0.15, 0.16, 0.2])
    mu = lambda x: (x - np.log(0.155)) * (x + 5)                 # noqa: E731
    rows = [dict(src=f'f{i % 2}', idx=i, r_h=float(np.exp(x)), Q=0.0, We=1 + x ** 2, lam=x,
                 mu={'1e': [float(mu(x))]}, nneg={'1e': int(mu(x) < 0)}, trans={'1e': 1e-3}) for i, x in enumerate(lr)]
    (c,) = rs.crossings(rows)
    assert c['interp'] == 3 and abs(c['r_h'] - 0.155) < 1e-12 and abs(c['We'] - 1 - np.log(0.155) ** 2) < 1e-12
    assert c['direction'] == 'to positive' and np.allclose(c['bracket'], [0.15, 0.16]) and c['resolved'] is True
    s = np.array([-0.6, -0.3, 0.0, 0.2, 0.5])
    mu = lambda s: (s - 0.1) * (2 + s)                           # noqa: E731
    rows = [dict(src='scan', idx=i, r_h=2.0, Q=float(3 * x + x ** 3), s_arc=float(x), We=float(x ** 2), lam=0.0,
                 mu={'2e': [float(mu(x))]}, nneg={'2e': int(mu(x) < 0)}, trans={}) for i, x in enumerate(s)]
    (c,) = rs.crossings(rows[::-1])
    assert c['interp'] == 4 and abs(c['Q'] - 0.301) < 1e-12 and abs(c['We'] - 0.01) < 1e-12 and c['resolved'] is None
    rows[1]['trans'] = {'1e': 0.5}
    assert rs.crossings(rows)[0]['resolved'] is None
    rows[2]['trans'] = {'1e': 0.1}
    assert rs.crossings(rows)[0]['resolved'] is False


def test_ring_sectors_restart_and_failures(tmp_path, monkeypatch):
    # a record that fails is listed under 'failed'; a second run with --m only adds the
    # direct sectors and keeps what the first wrote (marked here); another K recomputes
    a = np.zeros(8)
    a[1], a[3] = 0.15, 0.05
    rm = RingMap()
    st = RingState(a, 0.4, rm, ring_panels(a, rm, 8, c_h=2.0, tol=1e-10), Q=0.5)
    rec = dict(a=a.tolist(), map_b=[], rho_h=0.4, We=1.0, lam=0.0, r_h=float(st.g.r_h), Q=0.5)
    run, out = str(tmp_path / 'run.json'), str(tmp_path / 'out.json')
    json.dump([rec, dict(rec, rho_h=float('nan'), r_h=0.7)], open(run, 'w'))
    argv = ['ring_sectors.py', out, run, '--sectors=0o,1e', '--bounds', '--K=16', '--c_h=2', '--nproc=1']

    def go(*extra):
        monkeypatch.setattr(sys, 'argv', argv + list(extra))
        rs.main()
        return json.load(open(out))
    d = go()
    assert len(d['rows']) == 1 and [f['r_h'] for f in d['failed']] == [0.7] and 'NaN' in d['failed'][0]['error']
    assert set(d['rows'][0]['mu']) == {'0o', '1e'} and 'mc_lo' in d['rows'][0] and d['rows'][0]['K'] == 16
    d['rows'][0]['mu']['0o'][0] = 123.0
    json.dump(d, open(out, 'w'))
    row = go('--m=2')['rows'][0]
    assert set(row['mu']) == {'0o', '1e', '2e', '2o'} and row['mu']['0o'][0] == 123.0
    row = go('--K=12')['rows'][0]
    assert set(row['mu']) == {'0o', '1e'} and row['K'] == 12 and row['mu']['0o'][0] != 123.0
