"""Continuation in the hole flux Q (scripts/ring_qscan.py) on rings_up[17] (r_h = 0.422):
exact Q column, Gamma affine in Q, Q = 0 re-solve and record contract at M = 64; the
Pool context carrying Q and the driver (restart, stale meta file, foreign output) at
M = 20, where a Jacobian costs a few seconds."""
import json
import os
import sys

import numpy as np
import pytest

from bubbles.ring import RingMap, RingState, ring_panels

ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC = os.path.join(ROOT, 'results', 'rings', 'rings_up.json')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import ring_qscan as qs  # noqa: E402

M, M_SMALL = 64, 20


def reduced(m):
    p = json.load(open(SRC))[17]
    Y, rm = qs.reduce_ring(np.concatenate([p['a'], [p['lam'], p['rho_h'], p['We']]]), RingMap(p['map_b']), m)
    P = ring_panels(Y[:m], rm, m, c_h=qs.C_H, tol=1e-11)
    return p, np.append(Y, 0.0), rm, P


@pytest.fixture(scope='module')
def ring():
    return reduced(M)


@pytest.fixture(scope='module')
def small():
    return reduced(M_SMALL)


def test_exact_Q_column(ring):
    p, X0, rm, P = ring
    lin = (np.zeros(M + 4), 0.0)
    h = 1e-6
    for Q in (0.0, 0.05):
        X = X0.copy()
        X[-1] = Q
        _, st = qs.F_ext(X, rm, P, p['r_h'], lin)
        col = qs.dF_dQ(st, X[M + 2])
        Fp, _ = qs.F_ext(X + h * np.eye(M + 4)[-1], rm, P, p['r_h'], lin)
        Fm, _ = qs.F_ext(X - h * np.eye(M + 4)[-1], rm, P, p['r_h'], lin)
        fd = (Fp - Fm)[:M + 3] / (2 * h)
        assert np.abs(col - fd).max() < 1e-6 * np.abs(col).max(), Q


def test_circulation_affine_in_Q(ring):
    _, X, rm, P = ring
    Q = 0.05
    st0 = RingState(X[:M], X[M + 1], rm, P, Q=0.0)
    stQ = RingState(X[:M], X[M + 1], rm, P, Q=Q)
    G1 = 2 * np.sum(qs.gamma1(st0) * st0.g.ds)
    assert abs(stQ.circulation - st0.circulation - Q * G1) < 1e-12


def test_Q0_resolve_and_record(ring):
    p, X, rm, P = ring
    X, st, ok, its, J = qs.solve_fixed_Q(X, rm, P, p['r_h'], nproc=2)
    assert ok
    assert abs(X[M + 2] - p['We']) < 1e-9
    assert abs(st.circulation - p['circulation']) < 1e-8
    ref = dict(lam=max(1.0, abs(X[M])), rho_h=X[M + 1], Q=0.5 * (0.5 * (st.g.r_h + st.g.r_o)) ** 2)
    W = qs.weights(X, ref, dict(We=1.0, Q=1.0, lam=1.0, shape=0.1))
    tau = qs.tangent(J, W, np.eye(M + 4)[-1])
    assert np.abs(J @ tau).max() < 1e-10 * np.abs(J).max()
    assert abs(tau[M + 2] / tau[-1] - 0.240) < 2e-3          # dWe/dQ at Q = 0 (flux report)
    rec = qs.make_record(X, st, rm, J, W, tau, 0, 0.0, its=its, nJ=1, ds=None, flags=[])
    sheet = {'Q', 'Qhat', 'R', 'u_waist', 'We2D', 'det_sign', 'sigma_min', 'dir', 's_arc', 'its', 'nJ', 'flags'}
    assert set(p) | sheet <= set(rec)
    assert abs(rec['r_h'] - p['r_h']) < 1e-10 and rec['res_hi'] < 1e-6
    json.dumps(rec)


def test_jacobian_carries_Q(small):
    """Finite-difference columns of jac_rows at Q = 0.05 along a random (a, rho_h) direction
    against a central difference; a worker context left at Q = 0 would miss the Q
    dependence of those columns (2e3 times the forward-difference error of 2e-5 here)."""
    p, X, rm, P = small
    m, lin = M_SMALL, (np.zeros(M_SMALL + 4), 0.0)
    rng = np.random.default_rng(1)
    v = np.zeros(m + 4)
    v[:m] = rng.standard_normal(m) / (1.0 + np.arange(m))
    v[m + 1] = 0.5 * X[m + 1]

    def fd(Q, h=1e-6):
        Xq = X.copy()
        Xq[-1] = Q
        Fp, Fm = (qs.F_ext(Xq + sg * h * v, rm, P, p['r_h'], lin)[0] for sg in (1, -1))
        return (Fp - Fm)[:m + 3] / (2 * h)

    Xq = X.copy()
    Xq[-1] = 0.05
    J = qs.jac_rows(Xq, qs.F_ext(Xq, rm, P, p['r_h'], lin)[1], rm, P, p['r_h'], nproc=2)
    Jv, ref = J @ v, fd(0.05)
    err = np.abs(Jv - ref).max()
    assert err < 1e-4 * np.abs(ref).max()
    assert np.abs(fd(0.0) - ref).max() > 100 * err


def test_scan_restart(small, tmp_path, monkeypatch):
    """A stale meta file without output is ignored, a re-run continues from the last record,
    and an output that starts from another ring is refused."""
    p = small[0]
    monkeypatch.setenv('BUBBLES_NPROC', '2')
    out = str(tmp_path / 'sheet.json')
    with open(out + '.meta.json', 'w') as f:
        json.dump(dict(runs=[], status={'+1': dict(reason='step', n=5)}), f)
    args = [SRC, '17', out, f'--M={M_SMALL}', '--dirs=+1', '--ds0=0.05']
    qs.main(args + ['--nmax=1'])
    qs.main(args + ['--nmax=2'])
    recs = json.load(open(out))
    meta = json.load(open(out + '.meta.json'))
    assert [r['dir'] for r in recs] == [0, 1, 1]
    assert meta['status']['+1'] == dict(reason='nmax', n=2) and len(meta['runs']) == 2
    assert np.allclose([r['s_arc'] for r in recs], [0.0, 0.05, 0.1]) and np.all(np.diff([r['Q'] for r in recs]) > 0)
    assert abs(recs[0]['We'] - p['We']) < 1e-9 and abs(recs[0]['dWe_ds'] / recs[0]['dQ_ds'] - 0.240) < 2e-3
    for r in recs[1:]:
        assert abs(r['r_h'] - p['r_h']) < 1e-10 and r['res_hi'] < 1e-6 and r['M'] == M_SMALL
        assert r['det_sign'] == recs[0]['det_sign'] and not r['flags']
        Q = r['Q']                                     # fit of the flux report's scan (M = 64)
        assert abs(r['We'] - (1.11546 + 0.246 * Q + 15.70 * Q ** 2 + 6.6 * Q ** 3)) < 1e-4
    with pytest.raises(SystemExit):
        qs.main([SRC, '18', out, '--nmax=0'])
